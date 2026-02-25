"""Pairwise comparison helpers for relacc."""

from .pairwise import (
    METRIC_NAMES,
    PairSpec,
    discover_pairs,
    format_pair_rows_csv,
    run_pairwise_comparison,
)

__all__ = [
    "METRIC_NAMES",
    "PairSpec",
    "discover_pairs",
    "format_pair_rows_csv",
    "run_pairwise_comparison",
]
