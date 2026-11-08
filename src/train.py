"""Fit baselines and candidate models on the training split; score validation only.

The test split is deliberately never read here. It is opened once, after model and
alert-threshold selection (see docs/PRD.md).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import DEMAND_FEATURES, EXPIRY_FEATURES, RISK_FRACTION
from src.metrics import alert_intervals, alert_metrics, demand_metrics
from src.splits import HORIZON_DAYS

SEED = 42
ALERT_CUTOFF = 0.5  # shortcut: untuned default, chosen on validation in the tuning step
# Fixed, modest settings. Early stopping stays off: it would draw a random, non-chronological
# holdout from the training rows.
BOOSTING = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "early_stopping": False, "random_state": SEED}


def _encode(categorical, *numeric_steps):
    def columns(frame):
        return [c for c in frame.columns if c not in categorical]
    numeric = make_pipeline(*numeric_steps) if numeric_steps else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", numeric, columns),
    ])


def _linear_steps():
    return SimpleImputer(strategy="median", add_indicator=True), StandardScaler()


def _rows(frame, split, features):
    rows = frame[frame.split == split]
    if rows.empty:
        raise ValueError(f"no {split} rows to work with")
    return rows, rows[features]


def fit_demand_models(demand):
    train, X = _rows(demand, "train", DEMAND_FEATURES)
    models = {
        # Month is one-hot here so a linear model can represent non-monotonic seasonality.
        "ridge": make_pipeline(_encode(["category", "month"], *_linear_steps()), Ridge(alpha=1.0)),
        "gradient_boosting": make_pipeline(_encode(["category"]),
                                           HistGradientBoostingRegressor(**BOOSTING)),
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
    validation, X = _rows(expiry, "validation", EXPIRY_FEATURES)
    alerts = pd.DataFrame({"rule": validation.projected_leftover_fraction > RISK_FRACTION},
                          index=validation.index)
    for name, model in fit_expiry_models(expiry).items():
        alerts[name] = model.predict_proba(X)[:, 1] >= ALERT_CUTOFF
    return alerts


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
    report = {
        "demand": evaluate_demand(pd.read_csv(args.data_dir / "demand_features.csv")),
        "expiry": evaluate_expiry(pd.read_csv(args.data_dir / "expiry_features.csv")),
        "notes": ["Validation only; the test split has not been evaluated.",
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
