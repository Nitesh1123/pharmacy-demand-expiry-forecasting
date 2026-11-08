import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import (DEMAND_FEATURES, EXPIRY_FEATURES, build_demand_features,
                          build_expiry_features)
from src.train import (DEMAND_BOOSTING_GRID, best_cutoff, demand_predictions, evaluate_demand,
                       evaluate_expiry, expiry_alerts, select_expiry_alert, tune_demand_boosting,
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


def test_best_cutoff_maximises_f1_and_breaks_ties_towards_one_half():
    assert best_cutoff([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.2]) == 0.5  # every cutoff in (0.3, 0.8] ties
    assert best_cutoff([1, 1, 0, 0], [0.3, 0.25, 0.2, 0.1]) == 0.25


def test_demand_tuning_picks_lowest_validation_mae_and_ignores_test_rows(demand):
    tuned = tune_demand_boosting(demand)
    assert len(tuned["trials"]) == len(DEMAND_BOOSTING_GRID)
    best = min(tuned["trials"], key=lambda t: t["mae"])
    assert tuned["selected"] == best["params"]
    assert tuned == tune_demand_boosting(demand[demand.split != "test"])


def test_expiry_selection_keeps_rule_unless_a_candidate_clears_the_paired_bar(expiry):
    result = select_expiry_alert(expiry)
    winners = [n for n, c in result["candidates"].items() if c["clearly_beats_rule"]]
    if winners:
        assert result["selected"] in winners
    else:
        assert result["selected"] == "rule"
    for candidate in result["candidates"].values():
        low, _ = candidate["f1_minus_rule_interval_95"]
        assert candidate["clearly_beats_rule"] == (low > 0)
    assert result == select_expiry_alert(expiry[expiry.split != "test"])


def test_expiry_selection_switches_only_when_a_candidate_clears_the_bar(expiry, monkeypatch):
    monkeypatch.setattr("src.train.paired_f1_difference", lambda *_: [0.01, 0.2])
    result = select_expiry_alert(expiry)
    best = max(result["candidates"], key=lambda n: result["candidates"][n]["f1"])
    assert result["selected"] == best
    monkeypatch.setattr("src.train.paired_f1_difference", lambda *_: [0.0, 0.2])
    assert select_expiry_alert(expiry)["selected"] == "rule"
