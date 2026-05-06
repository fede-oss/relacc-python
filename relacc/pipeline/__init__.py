"""Comparison helpers for relacc."""

from .distribution import (
    DISTRIBUTION_MODE,
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    GROUP_BY_PARENT_DIR,
    format_distribution_rows_csv,
    run_distribution_comparison,
)
from .pairwise import (
    METRIC_NAMES,
    PairSpec,
    discover_pairs,
    format_pair_rows_csv,
    run_pairwise_comparison,
)

__all__ = [
    "DISTRIBUTION_MODE",
    "GROUP_BY_FILENAME_LABEL",
    "GROUP_BY_MODES",
    "GROUP_BY_PARENT_DIR",
    "METRIC_NAMES",
    "PairSpec",
    "discover_pairs",
    "format_distribution_rows_csv",
    "format_pair_rows_csv",
    "run_distribution_comparison",
    "run_pairwise_comparison",
]
