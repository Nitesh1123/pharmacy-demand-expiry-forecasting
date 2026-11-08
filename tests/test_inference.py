import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import EXPIRY_LABELS, build_demand_features, build_expiry_features, expiry_features_at
from src.inference import (available_dates, build_artifact, explain_forecast, expiry_guidance,
                           forecast_demand, load_artifact, save_artifact)

SELECTION = {"demand_boosting": {"selected": {"max_iter": 50, "max_leaf_nodes": 7}}}
TEST_REPORT = {"demand": {"models": {"gradient_boosting_tuned": {"wape": 0.05}}},
               "expiry": {"models": {"rule": {"precision": 0.9, "recall": 0.8}}}}
SUNDAY = "2023-06-04"


class Constant:
    def predict(self, X):
        return np.full(len(X), 100.0)


@pytest.fixture(scope="module")
def tables():
    return generate_dataset(seed=7, num_products=16, days=730)


@pytest.fixture(scope="module")
def artifact(tables):
    return build_artifact(build_demand_features(tables), SELECTION, TEST_REPORT)


def test_saved_artifact_reloads_and_gives_identical_forecasts(tables, artifact, tmp_path):
    path = save_artifact(artifact, tmp_path / "model.joblib")
    pd.testing.assert_frame_equal(forecast_demand(load_artifact(path), tables, SUNDAY),
                                  forecast_demand(artifact, tables, SUNDAY))


def test_missing_or_mismatched_artifacts_are_rejected(artifact, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "absent.joblib")
    path = tmp_path / "wrong.joblib"
    joblib.dump({**artifact, "features": ["sales_7d"]}, path)
    with pytest.raises(ValueError, match="features"):
        load_artifact(path)


def test_forecast_comes_from_the_loaded_model_with_test_error_band(tables, artifact):
    result = forecast_demand({**artifact, "model": Constant()}, tables, SUNDAY)
    assert len(result) == 16
    assert (result.forecast_30d == 100).all()
    assert result.low_30d.iloc[0] == pytest.approx(95)
    assert result.high_30d.iloc[0] == pytest.approx(105)


def test_guidance_compares_forecast_with_stock_on_hand(tables, artifact):
    result = forecast_demand({**artifact, "model": Constant()}, tables, SUNDAY)
    short = result.on_hand_units < 100
    assert (result.loc[short, "shortfall_units"] == 100 - result.loc[short, "on_hand_units"]).all()
    assert (result.loc[~short, "shortfall_units"] == 0).all()
    assert (result.action[short] == "Review reorder").all()


def test_forecast_needs_no_data_after_the_review_date(artifact):
    full = generate_dataset(seed=7, num_products=16, days=240)
    cut = generate_dataset(seed=7, num_products=16, days=114)  # ends Sunday 2022-04-24
    columns = ["product_id", "forecast_30d", "on_hand_units", "action"]
    pd.testing.assert_frame_equal(forecast_demand(artifact, full, "2022-04-24")[columns],
                                  forecast_demand(artifact, cut, "2022-04-24")[columns])


def test_unknown_review_date_is_rejected(tables, artifact):
    with pytest.raises(ValueError, match="review date"):
        forecast_demand(artifact, tables, "2023-06-05")
    with pytest.raises(ValueError, match="review date"):
        expiry_guidance(tables, "2023-06-05")


def test_review_dates_are_sundays_with_enough_history(tables):
    dates = pd.to_datetime(available_dates(tables))
    assert (dates.dayofweek == 6).all()
    assert dates.min() >= pd.Timestamp("2022-02-25")
    assert dates.is_monotonic_increasing


def test_explanation_lists_the_largest_drivers(tables, artifact):
    drivers = explain_forecast(artifact, tables, SUNDAY, tables["products"].product_id[0])
    assert len(drivers) == 5
    effects = [abs(d["prediction_change_from_actual_value"]) for d in drivers]
    assert effects == sorted(effects, reverse=True)


def test_expiry_rows_cover_every_near_expiry_batch_without_labels(tables):
    rows = expiry_features_at(tables, SUNDAY)
    assert not set(EXPIRY_LABELS + ["split"]) & set(rows.columns)
    snaps = tables["inventory_snapshots"].merge(tables["batches"][["batch_id", "expiry_date"]], on="batch_id")
    snaps = snaps[snaps.as_of_date == SUNDAY]
    days = (pd.to_datetime(snaps.expiry_date) - pd.Timestamp(SUNDAY)).dt.days
    assert set(rows.batch_id) == set(snaps[days <= 60].batch_id)


def test_expiry_rows_match_the_evaluated_feature_table(tables):
    at_date = expiry_features_at(tables, SUNDAY).set_index("batch_id")
    selected = build_expiry_features(tables)
    selected = selected[selected.as_of_date == SUNDAY].set_index("batch_id")
    assert len(selected) > 0
    columns = [c for c in selected.columns if c in at_date.columns]
    pd.testing.assert_frame_equal(at_date.loc[selected.index, columns], selected[columns])


def test_expiry_guidance_needs_no_data_after_the_review_date():
    full = generate_dataset(seed=7, num_products=16, days=240)
    cut = generate_dataset(seed=7, num_products=16, days=114)
    pd.testing.assert_frame_equal(expiry_guidance(full, "2022-04-24"), expiry_guidance(cut, "2022-04-24"))


def test_expiry_guidance_applies_the_rule_and_flags_short_history(tables):
    result = expiry_guidance(tables, SUNDAY)
    assert (result.alert == (result.projected_leftover_fraction > 0.2)).all()
    assert (result.action[result.alert] == "Prioritise rotation review").all()
    early = generate_dataset(seed=7, num_products=16, days=30)
    first_sunday = expiry_guidance(early, "2022-01-02")
    assert (first_sunday.action == "Not enough sales history").all()
    assert not first_sunday.alert.any()
