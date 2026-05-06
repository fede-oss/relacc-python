import pytest

from relacc import metrics as Metrics


def test_get_metric_names_can_exclude_dtw_metrics():
    assert Metrics.get_metric_names(include_dtw=False) == Metrics.BASE_METRIC_NAMES
    assert Metrics.get_metric_names() == Metrics.METRIC_NAMES


def test_compute_metrics_rejects_unknown_metric_before_evaluation():
    with pytest.raises(ValueError, match="Unknown metric: nope"):
        Metrics.compute_metrics(None, None, metric_names=["nope"])
