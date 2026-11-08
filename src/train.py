"""Fit baselines and candidate models on the training split; score validation only.

The test split is deliberately never read here. It is opened once, after model and
alert-threshold selection (see docs/PRD.md).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import DEMAND_FEATURES, EXPIRY_FEATURES, RISK_FRACTION
from src.metrics import alert_intervals, alert_metrics, demand_metrics, paired_f1_difference
from src.splits import HORIZON_DAYS

SEED = 42
ALERT_CUTOFF = 0.5  # untuned comparison point; select_expiry_alert chooses a cutoff on validation
CUTOFFS = tuple(round(0.1 + 0.05 * step, 2) for step in range(17))
# Fixed, modest settings. Early stopping stays off: it would draw a random, non-chronological
# holdout from the training rows.
BOOSTING = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "early_stopping": False, "random_state": SEED}
# Deliberately small: 9 validation origins cannot support a broad search.
DEMAND_BOOSTING_GRID = [{"max_iter": iters, "max_leaf_nodes": leaves}
                        for iters in (100, 200, 400) for leaves in (7, 15, 31)]


def _encode(categorical, *numeric_steps):
    numeric = make_pipeline(*numeric_steps) if numeric_steps else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder=numeric)


def _linear_steps():
    return SimpleImputer(strategy="median", add_indicator=True), StandardScaler()


def _rows(frame, split, features):
    rows = frame[frame.split == split]
    if rows.empty:
        raise ValueError(f"no {split} rows to work with")
    return rows, rows[features]


def _demand_boosting(params):
    return make_pipeline(_encode(["category"]), HistGradientBoostingRegressor(**params))


def fit_demand_models(demand):
    train, X = _rows(demand, "train", DEMAND_FEATURES)
    models = {
        # Month is one-hot here so a linear model can represent non-monotonic seasonality.
        "ridge": make_pipeline(_encode(["category", "month"], *_linear_steps()), Ridge(alpha=1.0)),
        "gradient_boosting": _demand_boosting(BOOSTING),
    }
    return {name: model.fit(X, train.target_30d) for name, model in models.items()}


def fit_expiry_models(expiry):
    train, X = _rows(expiry, "train", EXPIRY_FEATURES)
    y = train.expiry_risk.astype(int)
    models = {
        "logistic_regression": make_pipeline(
            _encode(["category"], *_linear_steps()),
            LogisticRegression(max_iter=2000, random_state=SEED)),
        "gradient_boosting": make_pipeline(_encode(["category"]),
                                           HistGradientBoostingClassifier(**BOOSTING)),
    }
    return {name: model.fit(X, y) for name, model in models.items()}


def demand_predictions(demand):
    """Predicted 30-day units for every validation row, one column per model."""
    validation, X = _rows(demand, "validation", DEMAND_FEATURES)
    predictions = pd.DataFrame({
        "moving_average_28d": validation.mean_28d * HORIZON_DAYS,
        "seasonal_naive": validation.sales_same_window_last_year,
    }, index=validation.index)
    for name, model in fit_demand_models(demand).items():
        predictions[name] = model.predict(X).clip(min=0)
    return predictions


def expiry_alerts(expiry):
    """Alert decision for every validation batch, one boolean column per model."""
    validation, _ = _rows(expiry, "validation", EXPIRY_FEATURES)
    alerts = pd.DataFrame({"rule": validation.projected_leftover_fraction > RISK_FRACTION},
                          index=validation.index)
    for name, probability in _expiry_probabilities(expiry).items():
        alerts[name] = probability >= ALERT_CUTOFF
    return alerts


def _expiry_probabilities(expiry):
    _, X = _rows(expiry, "validation", EXPIRY_FEATURES)
    return pd.DataFrame({name: model.predict_proba(X)[:, 1]
                         for name, model in fit_expiry_models(expiry).items()}, index=X.index)


def best_cutoff(y_true, probability):
    """Probability cutoff with the highest F1; ties go to the cutoff closest to 0.5."""
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    return max(CUTOFFS, key=lambda c: (alert_metrics(y_true, probability >= c)["f1"] or 0,
                                       -abs(c - 0.5)))


def tune_demand_boosting(demand):
    """Small grid fitted on training rows and compared on validation MAE."""
    train, X = _rows(demand, "train", DEMAND_FEATURES)
    validation, X_validation = _rows(demand, "validation", DEMAND_FEATURES)
    trials = []
    for params in DEMAND_BOOSTING_GRID:
        model = _demand_boosting({**BOOSTING, **params}).fit(X, train.target_30d)
        predicted = model.predict(X_validation).clip(min=0)
        trials.append({"params": params, **demand_metrics(validation.target_30d, predicted)})
    return {"selected": min(trials, key=lambda t: t["mae"])["params"], "trials": trials}


def select_expiry_alert(expiry):
    """Keep the rule unless a candidate's paired F1 advantage is entirely above zero."""
    validation, _ = _rows(expiry, "validation", EXPIRY_FEATURES)
    actual = validation.expiry_risk.astype(int)
    rule = validation.projected_leftover_fraction > RISK_FRACTION
    candidates = {}
    for name, probability in _expiry_probabilities(expiry).items():
        cutoff = best_cutoff(actual, probability)
        alert = probability >= cutoff
        interval = paired_f1_difference(actual, alert, rule)
        candidates[name] = {"cutoff": cutoff, **alert_metrics(actual, alert),
                            "f1_minus_rule_interval_95": interval,
                            "clearly_beats_rule": interval[0] > 0}
    winners = [name for name, result in candidates.items() if result["clearly_beats_rule"]]
    return {"rule": alert_metrics(actual, rule), "candidates": candidates,
            "selected": max(winners, key=lambda n: candidates[n]["f1"]) if winners else "rule"}


def evaluate_demand(demand):
    predictions = demand_predictions(demand)
    actual = demand.loc[predictions.index, "target_30d"]
    return {"evaluated_split": "validation", "train_rows": int((demand.split == "train").sum()),
            "validation_rows": len(actual),
            "validation_origins": int(demand.loc[predictions.index, "origin_date"].nunique()),
            "models": {name: demand_metrics(actual, predictions[name]) for name in predictions}}


def evaluate_expiry(expiry):
    alerts = expiry_alerts(expiry)
    actual = expiry.loc[alerts.index, "expiry_risk"].astype(int)
    return {"evaluated_split": "validation", "train_batches": int((expiry.split == "train").sum()),
            "validation_batches": len(actual), "positive": int(actual.sum()),
            "negative": int((actual == 0).sum()), "probability_cutoff": ALERT_CUTOFF,
            "models": {name: {**alert_metrics(actual, alerts[name]),
                              "intervals_95": alert_intervals(actual, alerts[name])}
                       for name in alerts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    demand = pd.read_csv(args.data_dir / "demand_features.csv")
    expiry = pd.read_csv(args.data_dir / "expiry_features.csv")
    report = {
        "demand": evaluate_demand(demand),
        "expiry": evaluate_expiry(expiry),
        "selection": {"demand_boosting": tune_demand_boosting(demand),
                      "expiry_alert": select_expiry_alert(expiry)},
        "notes": ["Validation only; the test split has not been evaluated.",
                  "Cutoffs and hyperparameters are chosen on the same validation rows they "
                  "are scored on, so selected scores are optimistic.",
                  "Synthetic stress-test data; not evidence of real-pharmacy accuracy."],
    }
    output = json.dumps(json.loads(json.dumps(report), parse_float=lambda v: round(float(v), 4)),
                        indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
