"""Distribution-vs-distribution metric placeholders.

- ``relacc.metrics`` is for per-gesture metrics that can be used in pairwise and
  one-vs-many flows.
- this file includes metrics for comparing distributions which compare two *sets* of
  samples/statistics and should not run in pairwise mode by default.

"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple


# A distribution metric receives two datasets/feature collections and returns a
# scalar distance/divergence score (lower is usually better).
DistributionMetricFn = Callable[[Iterable[float], Iterable[float]], float]


def _placeholder_metric(reference_values: Iterable[float], candidate_values: Iterable[float]) -> float:
    """Temporary placeholder metric.

    Current behavior:
    - materializes both iterables
    - returns absolute difference in sample counts
    
    Replace with real implementations in the distribution pipeline.
    """
    ref = list(reference_values)
    cand = list(candidate_values)
    return float(abs(len(ref) - len(cand)))


_DISTRIBUTION_METRIC_DEFINITIONS: Tuple[Tuple[str, DistributionMetricFn], ...] = (
    ("placeholderCountGap", _placeholder_metric),
)

DISTRIBUTION_METRIC_NAMES: Tuple[str, ...] = tuple(
    name for name, _ in _DISTRIBUTION_METRIC_DEFINITIONS
)


def compute_distribution_metrics(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
) -> Dict[str, float]:
    """Compute all registered distribution metrics for two value collections."""
    results: Dict[str, float] = {}
    for name, metric_fn in _DISTRIBUTION_METRIC_DEFINITIONS:
        results[name] = metric_fn(reference_values, candidate_values)
    return results
