import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import (DEMAND_FEATURES, EXPIRY_FEATURES, build_demand_features,
                          build_expiry_features, split_summary)

FORBIDDEN = {"base_daily_demand", "simulated_demand", "unmet_demand", "outcome_status",
             "outcome_date", "expiry_risk", "target_30d"}


@pytest.fixture(scope="module")
def long_run():
    return generate_dataset(seed=42, num_products=6, days=240)


@pytest.fixture(scope="module")
def short_run():
    # Ends on Sunday 2022-04-24, so the last origin and snapshot have no later data at all.
    return generate_dataset(seed=42, num_products=6, days=114)


def test_feature_lists_exclude_simulator_and_label_fields():
    assert not FORBIDDEN & set(DEMAND_FEATURES)
    assert not FORBIDDEN & set(EXPIRY_FEATURES)
    # Final balances live in batches.quantity_remaining; only snapshot stock may be used.
    assert "final_quantity_remaining" not in EXPIRY_FEATURES


def test_demand_target_is_next_thirty_days_of_recorded_sales(long_run):
    demand = build_demand_features(long_run)
    daily = long_run["daily_sales"]
    row = demand.iloc[len(demand) // 2]
    origin = pd.Timestamp(row.origin_date)
    window = daily[(daily.product_id == row.product_id) &
                   (pd.to_datetime(daily.date) > origin) &
                   (pd.to_datetime(daily.date) <= origin + pd.Timedelta(days=30))]
    assert len(window) == 30
    assert row.target_30d == window.quantity_sold.sum()
    past = daily[(daily.product_id == row.product_id) &
                 (pd.to_datetime(daily.date) <= origin) &
                 (pd.to_datetime(daily.date) > origin - pd.Timedelta(days=7))]
    assert row.sales_7d == past.quantity_sold.sum()


def test_demand_origins_are_sundays_after_warmup_with_complete_targets(long_run):
    demand = build_demand_features(long_run)
    origins = pd.to_datetime(demand.origin_date)
    assert (origins.dt.dayofweek == 6).all()
    assert origins.min() >= pd.Timestamp("2022-01-01") + pd.Timedelta(days=55)
    assert origins.max() + pd.Timedelta(days=30) <= pd.Timestamp(long_run["daily_sales"].date.max())
    assert not demand.duplicated(["origin_date", "product_id"]).any()


def _features_through(frame, date_column, last_date, columns):
    rows = frame.loc[frame[date_column] <= last_date, ["product_id", date_column, *columns]]
    return rows.sort_values(["product_id", date_column]).reset_index(drop=True)


def test_demand_features_ignore_everything_after_the_origin(long_run, short_run):
    # The generator is prefix-stable, so the short run is the long run with the future removed.
    last = short_run["daily_sales"].date.max()
    long_features = _features_through(build_demand_features(long_run, labeled_only=False),
                                      "origin_date", last, DEMAND_FEATURES)
    short_features = _features_through(build_demand_features(short_run, labeled_only=False),
                                       "origin_date", last, DEMAND_FEATURES)
    assert last in set(short_features.origin_date)
    pd.testing.assert_frame_equal(long_features, short_features)


def test_inference_rows_have_features_but_no_target(short_run):
    demand = build_demand_features(short_run, labeled_only=False)
    latest = demand[demand.origin_date == short_run["daily_sales"].date.max()]
    assert len(latest) == 6
    assert latest.target_30d.isna().all() and (latest.split == "excluded").all()
    assert latest[["sales_28d", "on_hand_units"]].notna().all().all()


def test_expiry_features_ignore_final_balances_and_future_outcomes(long_run, short_run):
    last = short_run["daily_sales"].date.max()
    long_rows = build_expiry_features(long_run)
    short_rows = build_expiry_features(short_run)
    long_features = _features_through(long_rows, "as_of_date", last, ["batch_id", *EXPIRY_FEATURES])
    short_features = _features_through(short_rows, "as_of_date", last, ["batch_id", *EXPIRY_FEATURES])
    assert len(long_features) > 0
    pd.testing.assert_frame_equal(long_features, short_features)


def test_expiry_uses_first_snapshot_within_sixty_days_once_per_batch(long_run):
    expiry = build_expiry_features(long_run)
    assert not expiry.batch_id.duplicated().any()
    assert expiry.days_to_expiry.between(1, 60).all()
    snaps = long_run["inventory_snapshots"].merge(
        long_run["batches"][["batch_id", "expiry_date"]], on="batch_id")
    snaps["days"] = (pd.to_datetime(snaps.expiry_date) - pd.to_datetime(snaps.as_of_date)).dt.days
    first = snaps[snaps.days <= 60].groupby("batch_id").as_of_date.min()
    assert expiry.set_index("batch_id").as_of_date.sort_index().equals(first.sort_index())


def test_fefo_queue_counts_only_earlier_expiring_stock_of_same_product(long_run):
    expiry = build_expiry_features(long_run)
    row = expiry.sort_values("units_ahead_fefo").iloc[-1]
    snaps = long_run["inventory_snapshots"].merge(
        long_run["batches"][["batch_id", "expiry_date"]], on="batch_id")
    me = snaps[(snaps.batch_id == row.batch_id) & (snaps.as_of_date == row.as_of_date)].iloc[0]
    ahead = snaps[(snaps.product_id == row.product_id) & (snaps.as_of_date == row.as_of_date) &
                  ((snaps.expiry_date < me.expiry_date) |
                   ((snaps.expiry_date == me.expiry_date) & (snaps.batch_id < me.batch_id)))]
    assert row.units_ahead_fefo == ahead.quantity_remaining.sum() > 0


def test_default_split_support_matches_day2_feasibility_counts():
    summary = split_summary(generate_dataset())
    expiry = summary["expiry"]
    assert [(expiry[s]["positive"], expiry[s]["negative"])
            for s in ("train", "validation", "test")] == [(262, 179), (25, 29), (71, 32)]
    for name in ("demand", "expiry"):
        assert sum(v["rows"] for v in summary[name].values()) == summary[f"{name}_rows"]
