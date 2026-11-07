import pandas as pd
import pytest

from src.data_generator import generate_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(seed=42, num_products=8, days=180)


def test_deterministic_and_prefix_stable(dataset):
    short = generate_dataset(seed=42, num_products=8, days=90)
    repeat = generate_dataset(seed=42, num_products=8, days=180)
    for name in dataset:
        pd.testing.assert_frame_equal(dataset[name], repeat[name])
    for name, date in [("sales", "sale_date"), ("daily_sales", "date"),
                       ("inventory_snapshots", "as_of_date")]:
        pd.testing.assert_frame_equal(
            dataset[name].loc[dataset[name][date] <= "2022-03-31"].reset_index(drop=True),
            short[name],
        )


def test_snapshots_match_receipts_minus_sales_at_that_date(dataset):
    batches = dataset["batches"].set_index("batch_id")
    sales = dataset["sales"]
    for row in dataset["inventory_snapshots"].itertuples():
        batch = batches.loc[row.batch_id]
        sold = sales.loc[(sales.batch_id == row.batch_id) &
                         (sales.sale_date <= row.as_of_date), "quantity_sold"].sum()
        assert row.quantity_remaining == batch.quantity_received - sold
        assert batch.received_date <= row.as_of_date < batch.expiry_date
        assert row.quantity_remaining > 0


def test_unknown_outcomes_are_not_negative_labels(dataset):
    batches = dataset["batches"]
    unresolved = batches.outcome_status == "censored"
    assert unresolved.any()
    assert batches.loc[unresolved, "expiry_risk"].isna().all()
    assert batches.loc[unresolved, "outcome_date"].isna().all()
    expired = batches.outcome_status == "expired"
    assert expired.any()
    expected = (batches.loc[expired, "quantity_remaining"] >
                0.2 * batches.loc[expired, "quantity_received"]).astype("Int64")
    pd.testing.assert_series_equal(batches.loc[expired, "expiry_risk"], expected,
                                  check_names=False)
    sold_out = batches.outcome_status == "sold_out"
    assert (batches.loc[sold_out, "expiry_risk"] == 0).all()
    assert (batches.loc[sold_out, "quantity_remaining"] == 0).all()


def test_daily_sales_include_every_product_day_and_balance(dataset):
    daily = dataset["daily_sales"]
    assert len(daily) == 8 * 180
    assert not daily.duplicated(["date", "product_id"]).any()
    assert (daily.simulated_demand == daily.quantity_sold + daily.unmet_demand).all()
    actual = dataset["sales"].groupby(["sale_date", "product_id"]).quantity_sold.sum()
    expected = daily.set_index(["date", "product_id"]).quantity_sold
    pd.testing.assert_series_equal(actual.reindex(expected.index, fill_value=0), expected,
                                  check_names=False)


def test_tiny_run_has_csv_headers_even_without_snapshots():
    tiny = generate_dataset(num_products=1, days=1)
    assert "as_of_date" in tiny["inventory_snapshots"].columns
