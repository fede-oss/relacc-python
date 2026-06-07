import math

import pytest
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

from relacc import distribution_metrics as DistributionMetrics
from relacc.distribution_metrics import DistributionMetricDefinition


def test_compute_distribution_metrics_default_values():
    reference = [0.0, 1.0, 2.0, 3.0]
    candidate = [1.0, 2.0, 3.0, 4.0]

    metrics = DistributionMetrics.compute_distribution_metrics(reference, candidate)

    assert metrics["wassersteinDistance"] == pytest.approx(
        wasserstein_distance(reference, candidate)
    )
    assert metrics["earthMoverDistance"] == pytest.approx(
        wasserstein_distance(reference, candidate)
    )
    assert metrics["wassersteinDistanceP2"] == pytest.approx(1.0)
    assert metrics["energyDistance"] == pytest.approx(
        energy_distance(reference, candidate)
    )
    ks_result = ks_2samp(reference, candidate)
    assert metrics["ksStatistic"] == pytest.approx(ks_result.statistic)
    assert metrics["ksPValue"] == pytest.approx(ks_result.pvalue)
    assert metrics["klDivergenceReferenceToCandidate"] == pytest.approx(math.log(2) / 4)
    assert metrics["klDivergenceCandidateToReference"] == pytest.approx(math.log(2) / 4)
    assert metrics["jeffreysDivergence"] == pytest.approx(math.log(2) / 2)
    assert 0.0 < metrics["jensenShannonDivergence"] < metrics["jeffreysDivergence"]
    assert metrics["totalVariationDistance"] == pytest.approx(0.25)


def test_kl_divergence_reports_direction_and_infinite_unsupported_bins():
    reference = [0.0, 0.0, 1.0, 1.0]
    candidate = [0.0, 0.0, 0.0, 0.0]

    metrics = DistributionMetrics.compute_distribution_metrics(reference, candidate)

    assert metrics["klDivergenceReferenceToCandidate"] == float("inf")
    assert metrics["klDivergenceCandidateToReference"] == pytest.approx(math.log(2))
    assert metrics["jeffreysDivergence"] == float("inf")
    assert metrics["jensenShannonDivergence"] == pytest.approx(
        0.5
        * (
            0.5 * math.log(0.5 / 0.75)
            + 0.5 * math.log(0.5 / 0.25)
            + math.log(1.0 / 0.75)
        )
    )
    assert metrics["totalVariationDistance"] == pytest.approx(0.5)


def test_distribution_metrics_are_nan_for_empty_inputs():
    metrics = DistributionMetrics.compute_distribution_metrics([], [1.0])

    assert set(metrics.keys()) == set(DistributionMetrics.DISTRIBUTION_METRIC_NAMES)
    assert all(math.isnan(value) for value in metrics.values())


def test_distribution_metric_semantics_identify_symmetric_and_asymmetric_metrics():
    semantics = {
        entry["name"]: entry
        for entry in DistributionMetrics.DISTRIBUTION_METRIC_SEMANTICS
    }

    assert semantics["earthMoverDistance"]["symmetry"] == DistributionMetrics.SYMMETRIC
    assert semantics["wassersteinDistanceP2"]["symmetry"] == DistributionMetrics.SYMMETRIC
    assert semantics["totalVariationDistance"]["symmetry"] == DistributionMetrics.SYMMETRIC
    assert (
        semantics["klDivergenceReferenceToCandidate"]["symmetry"]
        == DistributionMetrics.ASYMMETRIC
    )
    assert DistributionMetrics.DISTRIBUTION_METRIC_SYMMETRY[
        "klDivergenceCandidateToReference"
    ] == DistributionMetrics.ASYMMETRIC


def test_compute_distribution_metrics_reuses_materialized_inputs_and_rounds(monkeypatch):
    def first_metric(reference_values, candidate_values):
        return float(sum(reference_values) + sum(candidate_values))

    def second_metric(reference_values, candidate_values):
        return float(len(list(reference_values)) + len(list(candidate_values)) + 0.1234)

    monkeypatch.setattr(
        DistributionMetrics,
        "_DISTRIBUTION_METRIC_DEFINITIONS",
        (
            DistributionMetricDefinition(
                "firstMetric",
                first_metric,
                DistributionMetrics.SYMMETRIC,
                "test metric",
            ),
            DistributionMetricDefinition(
                "secondMetric",
                second_metric,
                DistributionMetrics.SYMMETRIC,
                "test metric",
            ),
        ),
    )

    metrics = DistributionMetrics.compute_distribution_metrics(
        (value for value in [1.0, 2.0]),
        (value for value in [3.0]),
        round_precision=2,
    )

    assert metrics["firstMetric"] == 6.0
    assert metrics["secondMetric"] == 3.12


def test_compute_distribution_metrics_does_not_round_non_finite_values(monkeypatch):
    monkeypatch.setattr(
        DistributionMetrics,
        "_DISTRIBUTION_METRIC_DEFINITIONS",
        (
            DistributionMetricDefinition(
                "infiniteMetric",
                lambda _reference_values, _candidate_values: float("inf"),
                DistributionMetrics.SYMMETRIC,
                "test metric",
            ),
        ),
    )

    metrics = DistributionMetrics.compute_distribution_metrics(
        [1.0],
        [2.0],
        round_precision=2,
    )

    assert metrics["infiniteMetric"] == float("inf")
