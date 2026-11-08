"""Validate generated CSVs and report observed outcomes without fitting a model."""

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = {
    "products": ["product_id"], "batches": ["batch_id"], "sales": ["sale_id"],
    "daily_sales": ["date", "product_id"],
    "inventory_snapshots": ["as_of_date", "batch_id"],
}


COLUMNS = {
    "products": ["product_id", "product_name", "category", "unit_price", "typical_shelf_life_days",
                 "base_daily_demand"],
    "batches": ["batch_id", "product_id", "manufacture_date", "received_date", "expiry_date",
                "quantity_received", "quantity_remaining", "outcome_status", "outcome_date", "expiry_risk"],
    "sales": ["sale_id", "sale_date", "product_id", "batch_id", "quantity_sold", "unit_price"],
    "daily_sales": ["date", "product_id", "quantity_sold", "simulated_demand", "unmet_demand"],
    "inventory_snapshots": ["as_of_date", "batch_id", "product_id", "quantity_remaining"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_dataset(tables):
    for name, columns in COLUMNS.items():
        missing = set(columns) - set(tables[name].columns)
        require(not missing, f"{name}: missing columns {sorted(missing)}")
    for name, keys in KEYS.items():
        frame = tables[name]
        require(not frame.duplicated(keys).any(), f"{name}: duplicate key")
        required = frame.drop(columns=["outcome_date", "expiry_risk"], errors="ignore")
        require(not required.isna().any().any(), f"{name}: missing required value")
    products, batches, sales, daily, snapshots = (tables[n] for n in KEYS)
    require(not products.empty and not batches.empty and not daily.empty, "empty dataset")
    for name in ("batches", "sales", "daily_sales", "inventory_snapshots"):
        require(tables[name].product_id.isin(products.product_id).all(), f"{name}: unknown product")
    for name, columns in {
        "products": ["base_daily_demand", "typical_shelf_life_days"],
        "batches": ["quantity_received", "quantity_remaining"],
        "sales": ["quantity_sold"], "daily_sales": ["quantity_sold", "simulated_demand", "unmet_demand"],
        "inventory_snapshots": ["quantity_remaining"],
    }.items():
        values = tables[name][columns]
        require(((values >= 0) & (values % 1 == 0)).all().all(), f"{name}: invalid quantities")
    require((batches.quantity_received > 0).all(), "empty received batch")
    require((sales.quantity_sold > 0).all(), "empty sale")
    require((snapshots.quantity_remaining > 0).all(), "empty snapshot")
    for name in ("products", "sales"):
        prices = tables[name].unit_price
        require(((prices > 0) & (prices < float("inf"))).all(), f"{name}: invalid price")

    # Parse before comparisons so malformed dates cannot pass lexicographic checks.
    for frame, columns in [(batches, ["manufacture_date", "received_date", "expiry_date", "outcome_date"]),
                           (sales, ["sale_date"]), (daily, ["date"]), (snapshots, ["as_of_date"])]:
        for column in columns:
            pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
    start, end = daily.date.min(), daily.date.max()
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    require(len(daily) == len(products) * days, "missing product-day records")
    require(((batches.manufacture_date <= batches.received_date) &
             (batches.received_date < batches.expiry_date) &
             batches.received_date.between(start, end)).all(), "invalid batch dates")

    joined = sales.merge(batches, on="batch_id", how="left", suffixes=("", "_batch"),
                         validate="many_to_one")
    require((joined.product_id == joined.product_id_batch).all(), "sale batch/product mismatch")
    require(((joined.sale_date >= joined.received_date) & (joined.sale_date < joined.expiry_date) &
             joined.sale_date.between(start, end)).all(), "invalid sale date")
    sold = sales.groupby("batch_id").quantity_sold.sum().reindex(batches.batch_id, fill_value=0).to_numpy()
    require((batches.quantity_received == sold + batches.quantity_remaining).all(), "batch balance mismatch")
    actual = sales.groupby(["sale_date", "product_id"]).quantity_sold.sum()
    expected = daily.set_index(["date", "product_id"]).quantity_sold
    require((actual.reindex(expected.index, fill_value=0) == expected).all(), "daily sales mismatch")
    require((daily.simulated_demand == daily.quantity_sold + daily.unmet_demand).all(), "demand balance mismatch")

    censored = (batches.expiry_date > end) & (batches.quantity_remaining > 0)
    sold_out = batches.quantity_remaining == 0
    expired = ~censored & ~sold_out
    require((batches.loc[censored, "outcome_status"] == "censored").all() and
            batches.loc[censored, ["outcome_date", "expiry_risk"]].isna().all().all(),
            "censored outcomes must remain unknown")
    require((batches.loc[sold_out, "outcome_status"] == "sold_out").all() and
            (batches.loc[sold_out, "expiry_risk"] == 0).all(), "invalid sold-out outcome")
    last_sale = sales.groupby("batch_id").sale_date.max()
    require((batches.loc[sold_out, "outcome_date"].to_numpy() ==
             last_sale.reindex(batches.loc[sold_out, "batch_id"]).to_numpy()).all(), "invalid sold-out date")
    require((batches.loc[expired, "outcome_status"] == "expired").all() and
            (batches.loc[expired, "outcome_date"] == batches.loc[expired, "expiry_date"]).all(),
            "invalid expired outcome")
    resolved = ~censored
    require(batches.loc[resolved, "expiry_risk"].notna().all(), "missing resolved label")
    expected_risk = (batches.quantity_remaining > 0.2 * batches.quantity_received).astype(int)
    require((batches.loc[resolved, "expiry_risk"] == expected_risk[resolved]).all(), "incorrect expiry label")

    expected_snapshots = 0
    for sunday in pd.date_range(start, end, freq="W-SUN").strftime("%Y-%m-%d"):
        active = ((batches.received_date <= sunday) & (batches.expiry_date > sunday) &
                  (~sold_out | (batches.outcome_date.fillna("") > sunday)))
        expected_snapshots += int(active.sum())
    require(len(snapshots) == expected_snapshots, "missing inventory snapshots")
    if not snapshots.empty:
        history = snapshots.merge(batches, on="batch_id", how="left", suffixes=("", "_final"),
                                  validate="many_to_one")
        require((history.product_id == history.product_id_final).all(), "snapshot batch/product mismatch")
        require(((history.as_of_date >= history.received_date) &
                 (history.as_of_date < history.expiry_date) & history.as_of_date.between(start, end)).all(),
                "invalid snapshot date")
        require((pd.to_datetime(history.as_of_date).dt.dayofweek == 6).all(), "snapshot must be Sunday")
        cumulative = sales.groupby(["batch_id", "sale_date"], as_index=False).quantity_sold.sum()
        cumulative["sold_so_far"] = cumulative.groupby("batch_id").quantity_sold.cumsum()
        cumulative["date"] = pd.to_datetime(cumulative.sale_date)
        history["date"] = pd.to_datetime(history.as_of_date)
        history = pd.merge_asof(history.sort_values("date"), cumulative.sort_values("date"),
                                on="date", by="batch_id", direction="backward")
        require((history.quantity_remaining == history.quantity_received -
                 history.sold_so_far.fillna(0)).all(), "snapshot uses wrong historical balance")

    periods = {}
    # Outcome counts are readiness diagnostics, not train/validation/test model scores.
    for name, lo, hi in [("early", "2022-01-01", "2023-05-31"),
                          ("middle", "2023-06-01", "2023-08-31"),
                          ("late", "2023-09-01", "2023-12-31")]:
        rows = batches.loc[batches.outcome_date.fillna("").between(lo, hi)]
        periods[name] = {"start_date": lo, "end_date": hi,
                         "positive": int((rows.expiry_risk == 1).sum()),
                         "negative": int((rows.expiry_risk == 0).sum())}
    return {
        "checks_passed": True, "start_date": start, "end_date": end,
        "rows": {name: len(frame) for name, frame in tables.items()},
        "outcomes": {"positive": int((batches.expiry_risk == 1).sum()),
                     "negative": int((batches.expiry_risk == 0).sum()), "censored": int(censored.sum())},
        "outcome_periods": periods,
        "stockout_product_days": int((daily.unmet_demand > 0).sum()),
        "zero_sales_product_days": int((daily.quantity_sold == 0).sum()),
        "limitations": ["Synthetic stress scenarios; prevalence is not calibrated to real pharmacies.",
                        "Outcome periods are not leakage-safe model splits by themselves.",
                        "Repeated snapshots are correlated; count independent batches when evaluating."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        tables = {name: pd.read_csv(args.data_dir / f"{name}.csv") for name in KEYS}
        report = validate_dataset(tables)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
