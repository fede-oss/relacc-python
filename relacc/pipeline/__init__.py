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
from .one_vs_many import (
    ONE_VS_MANY_MODE,
    format_one_vs_many_result,
    run_one_vs_many_comparison,
)

__all__ = [
    "DISTRIBUTION_MODE",
    "GROUP_BY_FILENAME_LABEL",
    "GROUP_BY_MODES",
    "GROUP_BY_PARENT_DIR",
    "METRIC_NAMES",
    "ONE_VS_MANY_MODE",
    "PairSpec",
    "discover_pairs",
    "format_distribution_rows_csv",
    "format_one_vs_many_result",
    "format_pair_rows_csv",
    "run_distribution_comparison",
    "run_one_vs_many_comparison",
    "run_pairwise_comparison",
]
