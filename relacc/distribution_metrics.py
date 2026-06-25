"""Distribution-vs-distribution metrics for scalar sample collections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence, Tuple

from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

from relacc.utils.math import MathUtil


DistributionMetricFn = Callable[[Iterable[float], Iterable[float]], float]

SYMMETRIC = "symmetric"
ASYMMETRIC = "asymmetric"


@dataclass(frozen=True)
class DistributionMetricDefinition:
    name: str
    fn: DistributionMetricFn
    symmetry: str
    description: str


def _finite_values(values: Iterable[float]) -> list[float]:
    finite_values = []
    for value in values:
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            finite_values.append(numeric_value)
    return finite_values


def _has_values(
    reference_values: Sequence[float],
    candidate_values: Sequence[float],
) -> bool:
    return len(reference_values) > 0 and len(candidate_values) > 0


def _wasserstein_distance(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    reference = _finite_values(reference_values)
    candidate = _finite_values(candidate_values)
    if not _has_values(reference, candidate):
        return float("nan")
    return float(wasserstein_distance(reference, candidate))


def _wasserstein_distance_p(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
    p: int,
) -> float:
    reference = sorted(_finite_values(reference_values))
    candidate = sorted(_finite_values(candidate_values))
    if not _has_values(reference, candidate):
        return float("nan")

    breakpoints = sorted(
        {0.0, 1.0}
        | {index / len(reference) for index in range(1, len(reference))}
        | {index / len(candidate) for index in range(1, len(candidate))}
    )
    total = 0.0
    for lower, upper in zip(breakpoints, breakpoints[1:]):
        midpoint = (lower + upper) / 2.0
        reference_index = min(math.floor(midpoint * len(reference)), len(reference) - 1)
        candidate_index = min(math.floor(midpoint * len(candidate)), len(candidate) - 1)
        total += (upper - lower) * abs(
            reference[reference_index] - candidate[candidate_index]
        ) ** p
    return total ** (1.0 / p)


def _wasserstein_distance_p2(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    return _wasserstein_distance_p(reference_values, candidate_values, p=2)


def _energy_distance(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    reference = _finite_values(reference_values)
    candidate = _finite_values(candidate_values)
    if not _has_values(reference, candidate):
        return float("nan")
    return float(energy_distance(reference, candidate))


def _ks_statistic(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    reference = _finite_values(reference_values)
    candidate = _finite_values(candidate_values)
    if not _has_values(reference, candidate):
        return float("nan")
    return float(ks_2samp(reference, candidate).statistic)


def _histogram_bin_count(reference_size: int, candidate_size: int) -> int:
    return max(1, math.ceil(math.sqrt(reference_size + candidate_size)))


def _shared_histogram_probabilities(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> tuple[list[float], list[float]] | None:
    reference = _finite_values(reference_values)
    candidate = _finite_values(candidate_values)
    if not _has_values(reference, candidate):
        return None

    combined = reference + candidate
    lower = min(combined)
    upper = max(combined)
    if math.isclose(lower, upper, abs_tol=1e-12):
        return ([1.0], [1.0])

    bin_count = _histogram_bin_count(len(reference), len(candidate))
    bin_width = (upper - lower) / bin_count

    def probabilities(values: Sequence[float]) -> list[float]:
        counts = [0] * bin_count
        for value in values:
            index = min(int((value - lower) / bin_width), bin_count - 1)
            counts[index] += 1
        return [count / len(values) for count in counts]

    return probabilities(reference), probabilities(candidate)


def _kl_divergence(
    source_probabilities: Sequence[float],
    target_probabilities: Sequence[float],
) -> float:
    total = 0.0
    for source_probability, target_probability in zip(
        source_probabilities,
        target_probabilities,
    ):
        if source_probability == 0.0:
            continue
        if target_probability == 0.0:
            return float("inf")
        total += source_probability * math.log(source_probability / target_probability)
    return total


def _kl_divergence_reference_to_candidate(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    probabilities = _shared_histogram_probabilities(reference_values, candidate_values)
    if probabilities is None:
        return float("nan")
    reference_probabilities, candidate_probabilities = probabilities
    return _kl_divergence(reference_probabilities, candidate_probabilities)


def _kl_divergence_candidate_to_reference(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    probabilities = _shared_histogram_probabilities(reference_values, candidate_values)
    if probabilities is None:
        return float("nan")
    reference_probabilities, candidate_probabilities = probabilities
    return _kl_divergence(candidate_probabilities, reference_probabilities)


def _jeffreys_divergence(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    reference_to_candidate = _kl_divergence_reference_to_candidate(
        reference_values,
        candidate_values,
    )
    candidate_to_reference = _kl_divergence_candidate_to_reference(
        reference_values,
        candidate_values,
    )
    return reference_to_candidate + candidate_to_reference


def _jensen_shannon_divergence(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    probabilities = _shared_histogram_probabilities(reference_values, candidate_values)
    if probabilities is None:
        return float("nan")
    reference_probabilities, candidate_probabilities = probabilities
    midpoint_probabilities = [
        (reference_probability + candidate_probability) / 2.0
        for reference_probability, candidate_probability in zip(
            reference_probabilities,
            candidate_probabilities,
        )
    ]
    return (
        _kl_divergence(reference_probabilities, midpoint_probabilities)
        + _kl_divergence(candidate_probabilities, midpoint_probabilities)
    ) / 2.0


def _total_variation_distance(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    probabilities = _shared_histogram_probabilities(reference_values, candidate_values)
    if probabilities is None:
        return float("nan")
    reference_probabilities, candidate_probabilities = probabilities
    return sum(
        abs(reference_probability - candidate_probability)
        for reference_probability, candidate_probability in zip(
            reference_probabilities,
            candidate_probabilities,
        )
    ) / 2.0


_DISTRIBUTION_METRIC_DEFINITIONS: Tuple[DistributionMetricDefinition, ...] = (
    DistributionMetricDefinition(
        "wassersteinDistance",
        _wasserstein_distance,
        SYMMETRIC,
        "First Wasserstein distance between empirical scalar distributions.",
    ),
    DistributionMetricDefinition(
        "earthMoverDistance",
        _wasserstein_distance,
        SYMMETRIC,
        "Earth mover distance; for one-dimensional samples this is Wasserstein-1.",
    ),
    DistributionMetricDefinition(
        "wassersteinDistanceP2",
        _wasserstein_distance_p2,
        SYMMETRIC,
        "Second Wasserstein distance between empirical scalar distributions.",
    ),
    DistributionMetricDefinition(
        "energyDistance",
        _energy_distance,
        SYMMETRIC,
        "Energy distance between empirical scalar distributions.",
    ),
    DistributionMetricDefinition(
        "ksStatistic",
        _ks_statistic,
        SYMMETRIC,
        "Kolmogorov-Smirnov two-sample statistic.",
    ),
    DistributionMetricDefinition(
        "klDivergenceReferenceToCandidate",
        _kl_divergence_reference_to_candidate,
        ASYMMETRIC,
        "Binned KL divergence from the reference distribution to the candidate distribution.",
    ),
    DistributionMetricDefinition(
        "klDivergenceCandidateToReference",
        _kl_divergence_candidate_to_reference,
        ASYMMETRIC,
        "Binned KL divergence from the candidate distribution to the reference distribution.",
    ),
    DistributionMetricDefinition(
        "jeffreysDivergence",
        _jeffreys_divergence,
        SYMMETRIC,
        "Symmetric KL variant formed by summing both directional KL divergences.",
    ),
    DistributionMetricDefinition(
        "jensenShannonDivergence",
        _jensen_shannon_divergence,
        SYMMETRIC,
        "Symmetric and finite KL-family divergence against the average distribution.",
    ),
    DistributionMetricDefinition(
        "totalVariationDistance",
        _total_variation_distance,
        SYMMETRIC,
        "Half the L1 distance between the two binned probability distributions.",
    ),
)

DISTRIBUTION_METRIC_NAMES: Tuple[str, ...] = tuple(
    definition.name for definition in _DISTRIBUTION_METRIC_DEFINITIONS
)
DISTRIBUTION_METRIC_SYMMETRY: Dict[str, str] = {
    definition.name: definition.symmetry for definition in _DISTRIBUTION_METRIC_DEFINITIONS
}
DISTRIBUTION_METRIC_SEMANTICS: Tuple[Dict[str, str], ...] = tuple(
    {
        "name": definition.name,
        "symmetry": definition.symmetry,
        "description": definition.description,
    }
    for definition in _DISTRIBUTION_METRIC_DEFINITIONS
)


def compute_distribution_metrics(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
    round_precision: int | None = None,
) -> Dict[str, float]:
    """Compute all registered distribution metrics for two value collections."""
    ref_values = list(reference_values)
    cand_values = list(candidate_values)

    results: Dict[str, float] = {}
    for definition in _DISTRIBUTION_METRIC_DEFINITIONS:
        value = definition.fn(ref_values, cand_values)
        if round_precision is not None and math.isfinite(value):
            value = MathUtil.roundTo(value, round_precision)
        results[definition.name] = value
    return results
