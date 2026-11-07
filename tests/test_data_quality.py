import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.data_quality import validate_dataset


@pytest.fixture
def tables():
    return generate_dataset(seed=42, num_products=4, days=90)


def test_valid_dataset_reports_counts_and_unknown_outcomes(tables):
    report = validate_dataset(tables)
    assert report["checks_passed"]
    assert report["rows"]["daily_sales"] == 360
    assert report["outcomes"]["censored"] > 0


@pytest.mark.parametrize("table,column", [("batches", "quantity_received"),
                                          ("inventory_snapshots", "quantity_remaining"),
                                          ("daily_sales", "quantity_sold")])
def test_corrupt_balances_fail(tables, table, column):
    tables[table].loc[0, column] += 1
    with pytest.raises(ValueError):
        validate_dataset(tables)


def test_missing_day_fails(tables):
    tables["daily_sales"] = tables["daily_sales"].iloc[1:]
    with pytest.raises(ValueError, match="product-day"):
        validate_dataset(tables)


def test_unknown_outcome_cannot_be_marked_safe(tables):
    censored = tables["batches"].outcome_status == "censored"
    tables["batches"].loc[censored, "expiry_risk"] = 0
    with pytest.raises(ValueError, match="censored"):
        validate_dataset(tables)


def test_sale_at_expiry_is_rejected(tables):
    sale = tables["sales"].iloc[0]
    expiry = tables["batches"].set_index("batch_id").loc[sale.batch_id, "expiry_date"]
    tables["sales"].loc[0, "sale_date"] = expiry
    with pytest.raises(ValueError, match="sale date"):
        validate_dataset(tables)


def test_snapshot_cannot_reference_another_product(tables):
    tables["inventory_snapshots"].loc[0, "product_id"] = "unknown"
    with pytest.raises(ValueError, match="product"):
        validate_dataset(tables)


def test_missing_snapshot_fails(tables):
    tables["inventory_snapshots"] = tables["inventory_snapshots"].iloc[1:]
    with pytest.raises(ValueError, match="snapshot"):
        validate_dataset(tables)


def test_small_dataset_with_no_snapshots_is_valid():
    report = validate_dataset(generate_dataset(num_products=1, days=1))
    assert report["rows"]["inventory_snapshots"] == 0
