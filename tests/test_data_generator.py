import pandas as pd
import pytest

from src.data_generator import generate_batches_and_sales, generate_products, set_seed


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_nonpositive_product_count(count):
    with pytest.raises(ValueError, match="positive"):
        generate_products(count)


@pytest.mark.parametrize("days", [0, -1])
def test_rejects_nonpositive_history(days):
    with pytest.raises(ValueError, match="positive"):
        generate_batches_and_sales(generate_products(1), days=days)


def test_default_has_observed_expiry_examples_in_each_period():
    set_seed(42)
    batches, _ = generate_batches_and_sales(generate_products())
    expiry = pd.to_datetime(batches.expiry_date)
    positive = batches.quantity_remaining > 0.2 * batches.quantity_received
    # These are independent batch outcomes, not duplicated weekly snapshots.
    for start, end in [("2022-01-01", "2023-05-31"),
                       ("2023-06-01", "2023-08-31"),
                       ("2023-09-01", "2023-12-31")]:
        assert (positive & expiry.between(start, end)).sum() >= 20
