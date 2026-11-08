"""Point-in-time feature tables for demand forecasting and batch expiry risk.

Every feature uses only data recorded through the end of its as-of date: daily sales,
Sunday inventory snapshots and batch facts known at receipt. Simulator-only fields,
final batch balances and observed outcomes are never features; outcomes are labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import HORIZON_DAYS, assign_demand_split, assign_expiry_split

WARMUP_DAYS = 56
NEAR_EXPIRY_DAYS = 60
RISK_FRACTION = 0.2
WINDOWS = (7, 14, 28, 56)

PRODUCT_FEATURES = ["category", "unit_price", "typical_shelf_life_days"]
DEMAND_FEATURES = [
    *(f"sales_{w}d" for w in WINDOWS), "mean_28d", "std_28d", "zero_days_28d",
    "sales_same_window_last_year", "on_hand_units", "month", "week_of_year",
    *PRODUCT_FEATURES,
]
EXPIRY_FEATURES = [
    "days_to_expiry", "days_since_received", "quantity_received", "quantity_remaining",
    "fraction_remaining", "units_ahead_fefo", "product_on_hand", "sales_7d", "mean_28d",
    "projected_leftover_fraction", *PRODUCT_FEATURES,
]
EXPIRY_LABELS = ["expiry_risk", "outcome_status", "outcome_date"]


def _sales_history(daily_sales):
    """Rolling sales statistics per (date, product), each including that date only."""
    wide = daily_sales.pivot(index="date", columns="product_id", values="quantity_sold")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    stats = {f"sales_{w}d": wide.rolling(w, min_periods=w).sum() for w in WINDOWS}
    stats["mean_28d"] = wide.rolling(28, min_periods=28).mean()
    stats["std_28d"] = wide.rolling(28, min_periods=28).std()
    stats["zero_days_28d"] = (wide == 0).astype(int).rolling(28, min_periods=28).sum()
    # Same 30 days one year (52 weeks) earlier; weekday-aligned and fully in the past.
    stats["sales_same_window_last_year"] = (
        wide.rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).sum().shift(364 - HORIZON_DAYS))
    # Target: units recorded over t+1..t+30, defined only when the whole window exists.
    stats["target_30d"] = wide.rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).sum().shift(-HORIZON_DAYS)
    frame = pd.concat({name: table.stack(future_stack=True) for name, table in stats.items()},
                      axis=1)
    frame.index.names = ["date", "product_id"]
    return frame.reset_index()


def _snapshots(tables):
    snaps = tables["inventory_snapshots"].merge(
        tables["batches"][["batch_id", "received_date", "expiry_date", "quantity_received"]],
        on="batch_id", how="left", validate="many_to_one")
    snaps["product_on_hand"] = snaps.groupby(["as_of_date", "product_id"]).quantity_remaining.transform("sum")
    # FEFO queue: stock of the same product that sells first (earlier expiry, then batch order).
    snaps = snaps.sort_values(["as_of_date", "product_id", "expiry_date", "batch_id"])
    snaps["units_ahead_fefo"] = (snaps.groupby(["as_of_date", "product_id"]).quantity_remaining.cumsum()
                                 - snaps.quantity_remaining)
    return snaps


def build_demand_features(tables, labeled_only=True):
    """One row per (Sunday origin, product) after warm-up.

    With labeled_only (training/evaluation) keep origins whose complete 30-day target
    is observed; otherwise also keep recent origins for inference (target is missing).
    """
    history = _sales_history(tables["daily_sales"])
    start = history.date.min()
    rows = history[(history.date.dt.dayofweek == 6) &
                   (history.date >= start + pd.Timedelta(days=WARMUP_DAYS - 1))].copy()
    if labeled_only:
        rows = rows[rows.target_30d.notna()]
    rows["origin_date"] = rows.pop("date").dt.strftime("%Y-%m-%d")
    on_hand = (tables["inventory_snapshots"].groupby(["as_of_date", "product_id"])
               .quantity_remaining.sum().rename("on_hand_units"))
    rows = rows.merge(on_hand, left_on=["origin_date", "product_id"], right_index=True, how="left")
    rows["on_hand_units"] = rows.on_hand_units.fillna(0).astype(int)
    origin = pd.to_datetime(rows.origin_date)
    rows["month"] = origin.dt.month
    rows["week_of_year"] = origin.dt.isocalendar().week.astype(int)
    rows = rows.merge(tables["products"][["product_id", *PRODUCT_FEATURES]], on="product_id")
    rows["split"] = assign_demand_split(rows.origin_date, tables["daily_sales"].date.max()).to_numpy()
    columns = ["origin_date", "product_id", *DEMAND_FEATURES, "target_30d", "split"]
    return rows[columns].sort_values(["origin_date", "product_id"]).reset_index(drop=True)


def _near_expiry_snapshots(tables):
    snaps = _snapshots(tables)
    as_of = pd.to_datetime(snaps.as_of_date)
    snaps["days_to_expiry"] = (pd.to_datetime(snaps.expiry_date) - as_of).dt.days
    return snaps[snaps.days_to_expiry <= NEAR_EXPIRY_DAYS]


def build_expiry_features(tables):
    """One row per batch: its first Sunday snapshot within 60 days of expiry."""
    snaps = _near_expiry_snapshots(tables)
    return _expiry_rows(tables, snaps.sort_values(["batch_id", "as_of_date"]).drop_duplicates("batch_id"))


def expiry_features_at(tables, as_of_date):
    """Every batch within 60 days of expiry at one snapshot date, for inference; no labels."""
    snaps = _near_expiry_snapshots(tables)
    rows = _expiry_rows(tables, snaps[snaps.as_of_date == as_of_date])
    return rows.drop(columns=[*EXPIRY_LABELS, "split"])


def _expiry_rows(tables, snaps):
    snaps = snaps.copy()
    as_of = pd.to_datetime(snaps.as_of_date)
    snaps["days_since_received"] = (as_of - pd.to_datetime(snaps.received_date)).dt.days
    snaps["fraction_remaining"] = snaps.quantity_remaining / snaps.quantity_received

    history = _sales_history(tables["daily_sales"])[["date", "product_id", "sales_7d", "mean_28d"]]
    history["date"] = history.date.dt.strftime("%Y-%m-%d")
    rows = snaps.merge(history, left_on=["as_of_date", "product_id"],
                       right_on=["date", "product_id"], how="left").drop(columns="date")
    # Transparent heuristic: units the recent sales rate can clear before expiry after
    # the FEFO queue ahead of this batch. Missing rate (warm-up) stays missing.
    sellable = (rows.mean_28d * rows.days_to_expiry - rows.units_ahead_fefo).clip(lower=0)
    rows["projected_leftover_fraction"] = ((rows.quantity_remaining - sellable).clip(lower=0)
                                           / rows.quantity_received)
    rows = rows.merge(tables["products"][["product_id", *PRODUCT_FEATURES]], on="product_id")
    rows = rows.merge(tables["batches"][["batch_id", *EXPIRY_LABELS]], on="batch_id")
    rows["split"] = assign_expiry_split(rows.as_of_date, rows.outcome_date,
                                        tables["daily_sales"].date.max()).to_numpy()
    rows.loc[rows.mean_28d.isna(), "split"] = "excluded"  # still in feature warm-up
    columns = ["as_of_date", "batch_id", "product_id", *EXPIRY_FEATURES, *EXPIRY_LABELS, "split"]
    return rows[columns].sort_values(["as_of_date", "batch_id"]).reset_index(drop=True)


def split_summary(tables, demand=None, expiry=None):
    demand = build_demand_features(tables) if demand is None else demand
    expiry = build_expiry_features(tables) if expiry is None else expiry
    summary = {"demand_rows": len(demand), "expiry_rows": len(expiry), "demand": {}, "expiry": {}}
    for split, rows in demand.groupby("split"):
        summary["demand"][split] = {
            "rows": len(rows), "origins": int(rows.origin_date.nunique()),
            "first_origin": rows.origin_date.min(), "last_origin": rows.origin_date.max(),
            "mean_target_30d": round(float(rows.target_30d.mean()), 2)}
    for split, rows in expiry.groupby("split"):
        summary["expiry"][split] = {
            "rows": len(rows), "independent_batches": int(rows.batch_id.nunique()),
            "positive": int((rows.expiry_risk == 1).sum()),
            "negative": int((rows.expiry_risk == 0).sum()),
            "unknown_outcome": int(rows.expiry_risk.isna().sum())}
    summary["notes"] = [
        "Counts describe data support, not model performance.",
        "Expiry rows are one near-expiry snapshot per batch; unknown outcomes are excluded.",
        "Expiry rows without 28 days of sales history (warm-up) are excluded.",
        "Excluded demand origins have 30-day targets that cross a split boundary.",
    ]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        _run(args)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


def _run(args):
    names = ["products", "batches", "daily_sales", "inventory_snapshots"]
    tables = {name: pd.read_csv(args.data_dir / f"{name}.csv") for name in names}
    tables["batches"]["expiry_risk"] = tables["batches"].expiry_risk.astype("Int64")
    demand, expiry = build_demand_features(tables), build_expiry_features(tables)
    for name, frame in [("demand_features", demand), ("expiry_features", expiry)]:
        frame.to_csv(args.data_dir / f"{name}.csv", index=False, lineterminator="\n")
        print(f"{name}: {len(frame):,} rows -> {args.data_dir / (name + '.csv')}")
    output = json.dumps(split_summary(tables, demand, expiry), indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
