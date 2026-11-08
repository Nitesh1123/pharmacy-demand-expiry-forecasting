import pytest

from src.metrics import alert_intervals, alert_metrics, demand_metrics, paired_f1_difference


def test_demand_metrics_known_values():
    result = demand_metrics([10, 0, 30], [8, 4, 30])
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx((20 / 3) ** 0.5)
    assert result["wape"] == pytest.approx(6 / 40)


def test_wape_is_undefined_when_no_units_were_sold():
    assert demand_metrics([0, 0], [1, 2])["wape"] is None


def test_alert_metrics_known_values():
    result = alert_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_no_alerts_leaves_precision_undefined_and_recall_zero():
    result = alert_metrics([1, 0], [0, 0])
    assert result["precision"] is None
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_no_positives_or_alerts_leaves_every_score_undefined():
    result = alert_metrics([0, 0], [0, 0])
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None


def test_intervals_are_repeatable_and_contain_the_estimate():
    y_true, alert = [1] * 30 + [0] * 30, [1] * 24 + [0] * 6 + [1] * 9 + [0] * 21
    first, second = alert_intervals(y_true, alert), alert_intervals(y_true, alert)
    assert first == second
    estimate = alert_metrics(y_true, alert)
    for name in ("precision", "recall", "f1"):
        low, high = first[name]
        assert 0 <= low <= estimate[name] <= high <= 1
        assert low < high


def test_paired_difference_of_identical_alerts_is_zero():
    y_true, alert = [1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0]
    assert paired_f1_difference(y_true, alert, alert) == [0.0, 0.0]


def test_paired_difference_is_positive_when_one_alert_is_clearly_better():
    y_true = [1] * 40 + [0] * 40
    better, worse = y_true, [0] * 20 + [1] * 20 + [1] * 20 + [0] * 20
    low, high = paired_f1_difference(y_true, better, worse)
    assert 0 < low <= high
    assert paired_f1_difference(y_true, better, worse) == [low, high]
