import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import DEMAND_FEATURES, build_demand_features, build_expiry_features
from src.final_evaluation import (demand_test_predictions, evaluate_on_test, expiry_test_alerts)

SELECTION = {"demand_boosting": {"selected": {"max_iter": 50, "max_leaf_nodes": 7}},
             "expiry_alert": {"selected": "rule", "candidates": {
                 "logistic_regression": {"cutoff": 0.55}, "gradient_boosting": {"cutoff": 0.6}}}}


@pytest.fixture(scope="module")
def tables():
    # Seed 7 and 16 products: a separate dataset, so these tests never see the default test split.
    return generate_dataset(seed=7, num_products=16, days=730)


@pytest.fixture(scope="module")
def demand(tables):
    return build_demand_features(tables)


@pytest.fixture(scope="module")
def expiry(tables):
    return build_expiry_features(tables)


@pytest.fixture(scope="module")
def report(tables, demand, expiry):
    return evaluate_on_test(demand, expiry, tables["batches"], SELECTION)


def test_test_predictions_cover_test_rows_and_baselines_follow_definitions(demand):
    predictions, _ = demand_test_predictions(demand, {"max_iter": 50, "max_leaf_nodes": 7})
    test = demand[demand.split == "test"]
    assert list(predictions.index) == list(test.index)
    pd.testing.assert_series_equal(predictions.moving_average_28d, test.mean_28d * 30, check_names=False)
    assert set(predictions) == {"moving_average_28d", "seasonal_naive", "ridge", "gradient_boosting_tuned"}


def test_models_are_fitted_without_validation_or_test_labels(demand, expiry):
    params = {"max_iter": 50, "max_leaf_nodes": 7}
    changed = demand.copy()
    changed.loc[changed.split.isin(["validation", "test"]), "target_30d"] = 0
    pd.testing.assert_frame_equal(demand_test_predictions(demand, params)[0],
                                  demand_test_predictions(changed, params)[0])
    flipped = expiry.copy()
    held_out = flipped.split.isin(["validation", "test"]) & flipped.expiry_risk.notna()
    flipped.loc[held_out, "expiry_risk"] = 1 - flipped.loc[held_out, "expiry_risk"]
    pd.testing.assert_frame_equal(expiry_test_alerts(expiry, SELECTION["expiry_alert"]),
                                  expiry_test_alerts(flipped, SELECTION["expiry_alert"]))


def test_expiry_alerts_use_locked_cutoffs(expiry):
    alerts = expiry_test_alerts(expiry, SELECTION["expiry_alert"])
    test = expiry[expiry.split == "test"]
    pd.testing.assert_series_equal(alerts.rule, test.projected_leftover_fraction > 0.2, check_names=False)
    assert set(alerts) == {"rule", "logistic_regression", "gradient_boosting"}


def test_report_marks_selected_models_and_counts_every_test_batch(report, expiry):
    assert report["evaluated_split"] == "test"
    assert report["demand"]["selected"] == "gradient_boosting_tuned"
    assert report["expiry"]["selected"] == "rule"
    batches = int((expiry.split == "test").sum())
    assert report["expiry"]["test_batches"] == batches
    for model in report["expiry"]["models"].values():
        assert model["tp"] + model["fp"] + model["fn"] + model["tn"] == batches


def test_demand_error_analysis_is_consistent(report, demand):
    analysis = report["demand"]["error_analysis"]
    assert {row["feature"] for row in analysis["permutation_importance"]} == set(DEMAND_FEATURES)
    worst = analysis["largest_error"]
    row = demand[(demand.origin_date == worst["origin_date"]) & (demand.product_id == worst["product_id"])].iloc[0]
    assert worst["actual"] == row.target_30d
    assert len(worst["feature_sensitivity"]) == 5
    assert len(analysis["by_origin"]) == demand[demand.split == "test"].origin_date.nunique()


def test_expiry_errors_are_real_rule_mistakes(report, expiry):
    test = expiry[expiry.split == "test"].set_index("batch_id")
    errors = report["expiry"]["error_analysis"]["rule_errors"]
    assert len(errors) == report["expiry"]["models"]["rule"]["fp"] + report["expiry"]["models"]["rule"]["fn"]
    for error in errors:
        alert = test.loc[error["batch_id"], "projected_leftover_fraction"] > 0.2
        assert int(test.loc[error["batch_id"], "expiry_risk"]) != int(alert)
        assert error["kind"] == ("false_alarm" if alert else "missed")


def test_report_is_repeatable(report, tables, demand, expiry):
    assert report == evaluate_on_test(demand, expiry, tables["batches"], SELECTION)
