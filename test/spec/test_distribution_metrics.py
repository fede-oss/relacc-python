from relacc import distribution_metrics as DistributionMetrics


def test_compute_distribution_metrics_placeholder_default():
    metrics = DistributionMetrics.compute_distribution_metrics([1.0, 2.0, 3.0], [5.0])
    assert metrics["placeholderCountGap"] == 2.0


def test_compute_distribution_metrics_reuses_materialized_inputs(monkeypatch):
    def first_metric(reference_values, candidate_values):
        return float(sum(reference_values) + sum(candidate_values))

    def second_metric(reference_values, candidate_values):
        return float(len(list(reference_values)) + len(list(candidate_values)))

    monkeypatch.setattr(
        DistributionMetrics,
        "_DISTRIBUTION_METRIC_DEFINITIONS",
        (
            ("firstMetric", first_metric),
            ("secondMetric", second_metric),
        ),
    )

    metrics = DistributionMetrics.compute_distribution_metrics(
        (value for value in [1.0, 2.0]),
        (value for value in [3.0]),
    )

    assert metrics["firstMetric"] == 6.0
    assert metrics["secondMetric"] == 3.0
