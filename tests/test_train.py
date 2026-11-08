import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import (DEMAND_FEATURES, EXPIRY_FEATURES, build_demand_features,
                          build_expiry_features)
from src.train import (demand_predictions, evaluate_demand, evaluate_expiry, expiry_alerts,
                       fit_demand_models, fit_expiry_models)


@pytest.fixture(scope="module")
def tables():
    return generate_dataset(seed=42, num_products=16, days=730)


@pytest.fixture(scope="module")
def demand(tables):
    return build_demand_features(tables)


@pytest.fixture(scope="module")
def expiry(tables):
    return build_expiry_features(tables)


def test_models_receive_only_declared_features(demand, expiry):
    for model in fit_demand_models(demand).values():
        assert list(model.feature_names_in_) == DEMAND_FEATURES
    for model in fit_expiry_models(expiry).values():
        assert list(model.feature_names_in_) == EXPIRY_FEATURES


def test_demand_baselines_follow_their_definitions(demand):
    predictions = demand_predictions(demand)
    validation = demand[demand.split == "validation"]
    pd.testing.assert_series_equal(predictions.moving_average_28d, validation.mean_28d * 30,
                                   check_names=False)
    pd.testing.assert_series_equal(predictions.seasonal_naive,
                                   validation.sales_same_window_last_year, check_names=False)
    assert (predictions >= 0).all().all()


def test_expiry_rule_alerts_on_projected_leftover_above_twenty_percent(expiry):
    alerts = expiry_alerts(expiry)
    validation = expiry[expiry.split == "validation"]
    expected = validation.projected_leftover_fraction > 0.2
    pd.testing.assert_series_equal(alerts.rule, expected, check_names=False)


def test_validation_labels_do_not_influence_fitted_models(demand, expiry):
    scrambled = demand.copy()
    scrambled.loc[scrambled.split == "validation", "target_30d"] = 0
    pd.testing.assert_frame_equal(demand_predictions(demand), demand_predictions(scrambled))
    flipped = expiry.copy()
    held_out = flipped.split == "validation"
    flipped.loc[held_out, "expiry_risk"] = 1 - flipped.loc[held_out, "expiry_risk"]
    pd.testing.assert_frame_equal(expiry_alerts(expiry), expiry_alerts(flipped))


def test_reports_are_unchanged_when_test_rows_are_removed(demand, expiry):
    assert evaluate_demand(demand) == evaluate_demand(demand[demand.split != "test"])
    assert evaluate_expiry(expiry) == evaluate_expiry(expiry[expiry.split != "test"])


def test_reports_are_repeatable_and_cover_baselines_and_candidates(demand, expiry):
    first = evaluate_demand(demand)
    assert first == evaluate_demand(demand)
    assert set(first["models"]) == {"moving_average_28d", "seasonal_naive", "ridge", "gradient_boosting"}
    assert first["evaluated_split"] == "validation"
    result = evaluate_expiry(expiry)
    assert set(result["models"]) == {"rule", "logistic_regression", "gradient_boosting"}
    for model in result["models"].values():
        assert model["tp"] + model["fp"] + model["fn"] + model["tn"] == result["validation_batches"]
        assert set(model["intervals_95"]) == {"precision", "recall", "f1"}


def test_missing_validation_rows_are_rejected(demand):
    with pytest.raises(ValueError, match="validation"):
        evaluate_demand(demand[demand.split == "train"])
