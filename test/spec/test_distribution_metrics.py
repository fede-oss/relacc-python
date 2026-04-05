import pytest
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

from relacc import distribution_metrics as DistributionMetrics


def test_compute_distribution_metrics_default_values():
    reference = [0.0, 1.0]
    candidate = [2.0, 3.0]

    metrics = DistributionMetrics.compute_distribution_metrics(reference, candidate)

    assert metrics["wassersteinDistance"] == pytest.approx(
        wasserstein_distance(reference, candidate)
    )
    assert metrics["energyDistance"] == pytest.approx(
        energy_distance(reference, candidate)
    )
    ks_result = ks_2samp(reference, candidate)
    assert metrics["ksStatistic"] == pytest.approx(ks_result.statistic)
    assert metrics["ksPValue"] == pytest.approx(ks_result.pvalue)


def test_compute_distribution_metrics_reuses_materialized_inputs_and_rounds(monkeypatch):
    def first_metric(reference_values, candidate_values):
        return float(sum(reference_values) + sum(candidate_values))

    def second_metric(reference_values, candidate_values):
        return float(len(list(reference_values)) + len(list(candidate_values)) + 0.1234)

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
        round_precision=2,
    )

    assert metrics["firstMetric"] == 6.0
    assert metrics["secondMetric"] == 3.12
