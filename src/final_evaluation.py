"""Score the locked model selections once on the held-out test split.

Models are the ones validated: fitted on training rows only. Selections come from the
checked-in validation report, so nothing here can be tuned against test results.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from src.features import DEMAND_FEATURES, EXPIRY_FEATURES, RISK_FRACTION
from src.metrics import alert_intervals, alert_metrics, demand_metrics
from src.splits import HORIZON_DAYS
from src.train import BOOSTING, SEED, _demand_boosting, _rows, fit_demand_models, fit_expiry_models

SELECTED_DEMAND = "gradient_boosting_tuned"
ERROR_DETAILS = ["as_of_date", "batch_id", "product_id", "category", "days_to_expiry",
                 "quantity_received", "quantity_remaining", "units_ahead_fefo", "mean_28d",
                 "projected_leftover_fraction", "outcome_status"]


def demand_test_predictions(demand, params):
    """30-day predictions for test rows from every model, plus the fitted tuned model."""
    train, X_train = _rows(demand, "train", DEMAND_FEATURES)
    test, X = _rows(demand, "test", DEMAND_FEATURES)
    tuned = _demand_boosting({**BOOSTING, **params}).fit(X_train, train.target_30d)
    predictions = pd.DataFrame({
        "moving_average_28d": test.mean_28d * HORIZON_DAYS,
        "seasonal_naive": test.sales_same_window_last_year,
        "ridge": fit_demand_models(demand)["ridge"].predict(X).clip(min=0),
        SELECTED_DEMAND: tuned.predict(X).clip(min=0),
    }, index=test.index)
    return predictions, tuned


def expiry_test_alerts(expiry, selection):
    """Rule and candidate alerts for test batches, using cutoffs locked on validation."""
    test, X = _rows(expiry, "test", EXPIRY_FEATURES)
    alerts = pd.DataFrame({"rule": test.projected_leftover_fraction > RISK_FRACTION}, index=test.index)
    for name, model in fit_expiry_models(expiry).items():
        alerts[name] = model.predict_proba(X)[:, 1] >= selection["candidates"][name]["cutoff"]
    return alerts


def typical_values(X_train):
    """Median of each numeric input, and the most common category, from training rows."""
    return {feature: X_train[feature].mode()[0] if feature == "category" else X_train[feature].median()
            for feature in DEMAND_FEATURES}


def _feature_sensitivity(model, row, typical_by_feature):
    """How the prediction moves if one input is set to its typical training value."""
    base = float(model.predict(row)[0])
    effects = []
    for feature in DEMAND_FEATURES:
        typical = typical_by_feature[feature]
        changed = float(model.predict(row.assign(**{feature: typical}))[0])
        effects.append({"feature": feature, "value": row[feature].iloc[0], "typical_train_value": typical,
                        "prediction_change_from_actual_value": base - changed})
    effects.sort(key=lambda e: -abs(e["prediction_change_from_actual_value"]))
    return effects[:5]


def _group_metrics(test, predictions, column):
    selected = predictions[SELECTED_DEMAND]
    return [{column: key, "rows": len(index), **demand_metrics(test.target_30d[index], selected[index]),
             "seasonal_naive_wape": demand_metrics(test.target_30d[index],
                                                   predictions.seasonal_naive[index])["wape"]}
            for key, index in test.groupby(column).groups.items()]


def _demand_error_analysis(demand, predictions, tuned):
    test, X = _rows(demand, "test", DEMAND_FEATURES)
    _, X_train = _rows(demand, "train", DEMAND_FEATURES)
    selected = predictions[SELECTED_DEMAND]
    errors = test.target_30d - selected
    importance = permutation_importance(tuned, X, test.target_30d, n_repeats=5, random_state=SEED,
                                        scoring="neg_mean_absolute_error")
    worst = errors.abs().idxmax()
    return {
        "mean_actual_minus_predicted": float(errors.mean()),
        "by_category": _group_metrics(test, predictions, "category"),
        "by_origin": _group_metrics(test, predictions, "origin_date"),
        "permutation_importance": sorted(
            ({"feature": f, "mae_increase": float(v)} for f, v in zip(DEMAND_FEATURES, importance.importances_mean)),
            key=lambda row: -row["mae_increase"]),
        "largest_error": {
            "origin_date": test.origin_date[worst], "product_id": test.product_id[worst],
            "category": test.category[worst], "actual": float(test.target_30d[worst]),
            "predicted": float(selected[worst]),
            "seasonal_naive": float(predictions.seasonal_naive[worst]),
            "feature_sensitivity": _feature_sensitivity(tuned, X.loc[[worst]], typical_values(X_train))},
    }


def _rule_errors(test, rule, batches):
    """Explain each rule mistake. Final balances are used here only, after prediction."""
    actual = test.expiry_risk.astype(int)
    final = batches.set_index("batch_id").quantity_remaining
    rows = []
    for index in test.index[rule != actual.astype(bool)]:
        row = test.loc[index]
        details = {column: row[column] for column in ERROR_DETAILS}
        sold = row.quantity_remaining - final[row.batch_id]
        unsold = final[row.batch_id] / row.quantity_received
        rows.append({"kind": "false_alarm" if rule[index] else "missed", **details,
                     "projected_sellable": max(0.0, row.mean_28d * row.days_to_expiry - row.units_ahead_fefo),
                     "actual_sold_after_snapshot": sold, "actual_unsold_fraction": unsold})
    return sorted(rows, key=lambda r: -abs(r["projected_leftover_fraction"] - r["actual_unsold_fraction"]))


def evaluate_on_test(demand, expiry, batches, selection):
    params = selection["demand_boosting"]["selected"]
    predictions, tuned = demand_test_predictions(demand, params)
    demand_test, _ = _rows(demand, "test", DEMAND_FEATURES)
    expiry_test, _ = _rows(expiry, "test", EXPIRY_FEATURES)
    alerts = expiry_test_alerts(expiry, selection["expiry_alert"])
    actual = expiry_test.expiry_risk.astype(int)
    return {
        "evaluated_split": "test",
        "demand": {
            "test_rows": len(demand_test), "test_origins": int(demand_test.origin_date.nunique()),
            "selected": SELECTED_DEMAND, "selected_params": params,
            "models": {name: demand_metrics(demand_test.target_30d, predictions[name]) for name in predictions},
            "error_analysis": _demand_error_analysis(demand, predictions, tuned)},
        "expiry": {
            "test_batches": len(actual), "positive": int(actual.sum()), "negative": int((actual == 0).sum()),
            "selected": selection["expiry_alert"]["selected"],
            "models": {name: {**alert_metrics(actual, alerts[name]),
                              "intervals_95": alert_intervals(actual, alerts[name])} for name in alerts},
            "error_analysis": {"rule_errors": _rule_errors(expiry_test, alerts.rule, batches)}},
        "notes": ["Selections were locked on validation before this evaluation.",
                  "Synthetic stress-test data; not evidence of real-pharmacy accuracy."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--selection", type=Path, default=Path("reports/day3-validation.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        _run(args)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


def _run(args):
    selection = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    report = evaluate_on_test(pd.read_csv(args.data_dir / "demand_features.csv"),
                              pd.read_csv(args.data_dir / "expiry_features.csv"),
                              pd.read_csv(args.data_dir / "batches.csv"), selection)
    output = json.dumps(json.loads(json.dumps(report, default=lambda v: v.item()),
                                   parse_float=lambda v: round(float(v), 4)), indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
