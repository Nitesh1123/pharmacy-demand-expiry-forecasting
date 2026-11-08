"""Demand error and expiry-alert metrics with explicit zero-denominator behaviour."""

import numpy as np

SCORES = ("precision", "recall", "f1")


def _ratio(numerator, denominator):
    # Undefined scores are None, not 0: "no alerts raised" is not "every alert was wrong".
    return float(numerator / denominator) if denominator else None


def demand_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)
    return {"mae": float(errors.mean()), "rmse": float(np.sqrt((errors ** 2).mean())),
            "wape": _ratio(errors.sum(), y_true.sum())}


def alert_metrics(y_true, alert):
    y_true, alert = np.asarray(y_true, dtype=bool), np.asarray(alert, dtype=bool)
    tp, fp = int((alert & y_true).sum()), int((alert & ~y_true).sum())
    fn, tn = int((~alert & y_true).sum()), int((~alert & ~y_true).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def paired_f1_difference(y_true, alert_a, alert_b, resamples=1000, seed=42):
    """95% bootstrap interval for F1(a) - F1(b), resampling the same batches for both."""
    y_true, alert_a, alert_b = (np.asarray(v, dtype=bool) for v in (y_true, alert_a, alert_b))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(y_true), len(y_true))
        f1_a = alert_metrics(y_true[chosen], alert_a[chosen])["f1"]
        f1_b = alert_metrics(y_true[chosen], alert_b[chosen])["f1"]
        if f1_a is not None and f1_b is not None:
            differences.append(f1_a - f1_b)
    return [float(q) for q in np.percentile(differences, [2.5, 97.5])]


def alert_intervals(y_true, alert, resamples=1000, seed=42):
    """95% percentile bootstrap intervals, resampling independent batches."""
    y_true, alert = np.asarray(y_true, dtype=bool), np.asarray(alert, dtype=bool)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in SCORES}
    for _ in range(resamples):
        chosen = rng.integers(0, len(y_true), len(y_true))
        result = alert_metrics(y_true[chosen], alert[chosen])
        for name in SCORES:
            if result[name] is not None:
                samples[name].append(result[name])
    return {name: ([float(q) for q in np.percentile(values, [2.5, 97.5])] if values else None)
            for name, values in samples.items()}
