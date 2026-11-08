"""Build and load the saved demand model, and produce advisory stock guidance.

The saved model is the one evaluated on the held-out test set: fitted on training rows
only, with the settings locked on validation. Expiry alerts use the transparent rule, so
they need no fitted model.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data_quality import validate_dataset
from src.features import (DEMAND_FEATURES, RISK_FRACTION, WARMUP_DAYS, build_demand_features,
                          expiry_features_at)
from src.final_evaluation import SELECTED_DEMAND, _feature_sensitivity, typical_values
from src.train import BOOSTING, _demand_boosting, _rows

MODEL_PATH = Path("models/demand_gradient_boosting.joblib")
TABLES = ("products", "batches", "sales", "daily_sales", "inventory_snapshots")
REQUIRED_KEYS = {"model", "features", "params", "typical_values", "trained_through", "test_wape",
                 "test_expiry_rule", "scikit_learn"}


def load_tables(data_dir):
    tables = {name: pd.read_csv(Path(data_dir) / f"{name}.csv") for name in TABLES}
    validate_dataset(tables)  # refuse corrupt data instead of forecasting from it
    tables["batches"]["expiry_risk"] = tables["batches"].expiry_risk.astype("Int64")
    return tables


def build_artifact(demand, selection, test_report):
    train, X = _rows(demand, "train", DEMAND_FEATURES)
    params = selection["demand_boosting"]["selected"]
    model = _demand_boosting({**BOOSTING, **params}).fit(X, train.target_30d)
    return {"model": model, "features": list(DEMAND_FEATURES), "params": params,
            "typical_values": typical_values(X), "trained_through": train.origin_date.max(),
            "test_wape": test_report["demand"]["models"][SELECTED_DEMAND]["wape"],
            "test_expiry_rule": test_report["expiry"]["models"]["rule"],
            "scikit_learn": sklearn.__version__}


def save_artifact(artifact, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path=MODEL_PATH):
    # joblib can run code on load, so only artifacts built locally by this project are loaded.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # unpickling a damaged file can raise almost anything
        raise ValueError(f"{path} is not a valid saved model") from exc
    if (not isinstance(artifact, dict) or not REQUIRED_KEYS <= set(artifact)
            or artifact["features"] != list(DEMAND_FEATURES)):
        raise ValueError(f"{path} is not a valid saved model with the expected features")
    if artifact["scikit_learn"] != sklearn.__version__:
        raise ValueError(f"{path} was built with scikit-learn {artifact['scikit_learn']}, "
                         f"but {sklearn.__version__} is installed; rebuild it")
    return artifact


def available_dates(tables):
    """Sundays with enough sales history to compute every demand feature."""
    days = pd.to_datetime(tables["daily_sales"].date.drop_duplicates()).sort_values()
    days = days[days >= days.min() + pd.Timedelta(days=WARMUP_DAYS - 1)]
    return days[days.dt.dayofweek == 6].dt.strftime("%Y-%m-%d").tolist()


def _demand_rows(tables, as_of):
    rows = build_demand_features(tables, labeled_only=False)
    rows = rows[rows.origin_date == as_of]
    if rows.empty:
        raise ValueError(f"no forecast is available for review date {as_of}")
    return rows


def forecast_demand(artifact, tables, as_of):
    """30-day forecast per product at a review date, with stock guidance."""
    rows = _demand_rows(tables, as_of).reset_index(drop=True)
    forecast = artifact["model"].predict(rows[artifact["features"]]).clip(min=0)
    result = rows[["product_id", "category", "on_hand_units", "target_30d"]].merge(
        tables["products"][["product_id", "product_name"]], on="product_id")
    result = result.rename(columns={"target_30d": "recorded_30d"})
    result["forecast_30d"] = forecast
    result["low_30d"] = forecast * (1 - artifact["test_wape"])
    result["high_30d"] = forecast * (1 + artifact["test_wape"])
    result["shortfall_units"] = (result.forecast_30d - result.on_hand_units).clip(lower=0)
    result["action"] = np.where(result.shortfall_units > 0, "Review reorder", "Stock covers forecast")
    return result


def explain_forecast(artifact, tables, as_of, product_id):
    """Largest effects on one forecast, from resetting each input to a typical value."""
    rows = _demand_rows(tables, as_of)
    row = rows.loc[rows.product_id == product_id, artifact["features"]]
    return _feature_sensitivity(artifact["model"], row, artifact["typical_values"])


def expiry_guidance(tables, as_of):
    """Apply the expiry rule to every batch within 60 days of expiry at a review date."""
    known_dates = set(tables["daily_sales"].date)
    if as_of not in known_dates or pd.Timestamp(as_of).dayofweek != 6:
        raise ValueError(f"no snapshot is available for review date {as_of}")
    rows = expiry_features_at(tables, as_of).merge(
        tables["batches"][["batch_id", "expiry_date"]], on="batch_id")
    rows["alert"] = rows.projected_leftover_fraction > RISK_FRACTION
    # A missing sales rate means unknown, which must not read as "no risk".
    rows["action"] = np.select([rows.projected_leftover_fraction.isna(), rows.alert],
                               ["Not enough sales history", "Prioritise rotation review"], "On track")
    columns = ["batch_id", "product_id", "category", "expiry_date", "days_to_expiry",
               "quantity_remaining", "units_ahead_fefo", "mean_28d",
               "projected_leftover_fraction", "alert", "action"]
    return rows[columns].sort_values(["alert", "projected_leftover_fraction", "batch_id"],
                                     ascending=[False, False, True]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--selection", type=Path, default=Path("reports/day3-validation.json"))
    parser.add_argument("--test-report", type=Path, default=Path("reports/test-evaluation.json"))
    args = parser.parse_args()
    try:
        _run(args)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


def _run(args):
    selection = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    test_report = json.loads(args.test_report.read_text(encoding="utf-8"))
    demand = pd.read_csv(args.data_dir / "demand_features.csv")
    path = save_artifact(build_artifact(demand, selection, test_report), args.model)
    tables = load_tables(args.data_dir)
    latest = available_dates(tables)[-1]
    sample = forecast_demand(load_artifact(path), tables, latest).iloc[0]
    print(f"Saved {path}")
    print(f"Sample from the reloaded model, review date {latest}: {sample.product_id} "
          f"forecast {sample.forecast_30d:,.0f} units over the next 30 days ({sample.action})")


if __name__ == "__main__":
    main()
