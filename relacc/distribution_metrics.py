"""Distribution-vs-distribution metrics for scalar sample collections."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

from relacc.utils.math import MathUtil


DistributionMetricFn = Callable[[Iterable[float], Iterable[float]], float]


def _wasserstein_distance(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    return float(wasserstein_distance(reference_values, candidate_values))


def _energy_distance(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    return float(energy_distance(reference_values, candidate_values))


def _ks_statistic(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    return float(ks_2samp(reference_values, candidate_values).statistic)


def _ks_pvalue(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> float:
    return float(ks_2samp(reference_values, candidate_values).pvalue)


_DISTRIBUTION_METRIC_DEFINITIONS: Tuple[Tuple[str, DistributionMetricFn], ...] = (
    ("wassersteinDistance", _wasserstein_distance),
    ("energyDistance", _energy_distance),
    ("ksStatistic", _ks_statistic),
    ("ksPValue", _ks_pvalue),
)

DISTRIBUTION_METRIC_NAMES: Tuple[str, ...] = tuple(
    name for name, _ in _DISTRIBUTION_METRIC_DEFINITIONS
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
    for name, metric_fn in _DISTRIBUTION_METRIC_DEFINITIONS:
        value = metric_fn(ref_values, cand_values)
        if round_precision is not None:
            value = MathUtil.roundTo(value, round_precision)
        results[name] = value
    return results
