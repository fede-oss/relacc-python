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
from .reporting import (
    CLASS_SCHEME_AUTO,
    CLASS_SCHEME_FILENAME_LABEL,
    CLASS_SCHEME_PARENT_DIR,
    CLASS_SCHEMES,
    DEFAULT_SAMPLE_LIMIT,
    REPORTING_MODE,
    ReportingEntry,
    ReportingSampleGroup,
    SOURCE_FOLDER_NAMES,
    build_sample_manifest,
    discover_reporting_sample_groups,
    load_reporting_entries,
    select_reporting_samples,
)

__all__ = [
    "CLASS_SCHEME_AUTO",
    "CLASS_SCHEME_FILENAME_LABEL",
    "CLASS_SCHEME_PARENT_DIR",
    "CLASS_SCHEMES",
    "DEFAULT_SAMPLE_LIMIT",
    "DISTRIBUTION_MODE",
    "GROUP_BY_FILENAME_LABEL",
    "GROUP_BY_MODES",
    "GROUP_BY_PARENT_DIR",
    "METRIC_NAMES",
    "PairSpec",
    "REPORTING_MODE",
    "ReportingEntry",
    "ReportingSampleGroup",
    "SOURCE_FOLDER_NAMES",
    "build_sample_manifest",
    "discover_pairs",
    "discover_reporting_sample_groups",
    "format_distribution_rows_csv",
    "format_pair_rows_csv",
    "load_reporting_entries",
    "run_distribution_comparison",
    "run_pairwise_comparison",
    "select_reporting_samples",
]
