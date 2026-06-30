#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from scipy import stats as scipy_stats

from relacc.distribution_metrics import (
    DISTRIBUTION_METRIC_NAMES,
    compute_distribution_metrics,
)

from relacc.dtw import DEFAULT_EXACT_RATE_THRESHOLD, recommended_window
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import METRIC_NAMES, compute_metrics
from relacc.pipeline._common import (
    compute_pair_metrics_from_points,
    format_csv_rows,
    list_csv_files,
    normalize_summary_shape,
    read_points,
    sampling_rate_for_sets,
    summary_sampling_rate,
    write_jsonl_rows,
)
from relacc.pipeline.dataset_discovery import (
    CLASS_SCHEME_AUTO,
    CLASS_SCHEMES,
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    dataset_and_class_for_relative_path,
    input_dataset_hint,
    normalize_class_scheme,
    normalize_group_by,
)
from relacc.pipeline.reporting import (
    ReportingEntry,
)
from relacc.pipeline.report_compare import (
    _baseline_stats,
    _compare_class,
    _compare_direct_distribution_pairs_class,
    _compare_human_baseline_class,
    _effective_dtw_window,
    _raw_metric_outputs_from_values,
    _rounded_metric_values,
)
from relacc.pipeline.report_exports import (
    _json_safe,
    _write_csv,
    _write_json,
    _write_readme,
    write_combined_report_exports,
)
from relacc.pipeline.report_runner import (
    ProgressReporter,
    _candidate_groups,
    _count_run_candidates,
    _entries_by_class,
    _format_duration,
    _int_cast,
    _load_reporting_entries,
    _optional_int_cast,
    _output_run_dir,
    _run_id,
    _run_reports,
    _safe_path_part,
    _seed_for_entries,
    _select_entries,
)
from relacc.pipeline.report_schema import (
    AGGREGATE_SUMMARY_COLUMNS,
    BASELINE_COLUMNS,
    COMBINED_AGGREGATE_SUMMARIES_FILENAME,
    COMBINED_BASELINE_FILENAME,
    COMBINED_BASELINE_STATS_FILENAME,
    COMBINED_BETWEEN_GROUPS_FILENAME,
    COMBINED_BETWEEN_GROUPS_STATS_FILENAME,
    COMBINED_DISTRIBUTION_FILENAME,
    COMBINED_OUTPUT_DIRNAME,
    COMBINED_PAIRWISE_FILENAME,
    COMBINED_RAW_DISTRIBUTIONS_FILENAME,
    COMBINED_RAW_METRICS_FILENAME,
    COMBINED_REPORT_FILENAME,
    COMBINED_STATS_FILENAME,
    COMBINED_SUMMARY_DISTRIBUTION_FILENAME,
    COMBINED_WITHIN_COMPARISON_FILENAME,
    COMBINED_WITHIN_COMPARISON_STATS_FILENAME,
    COMBINED_WITHIN_REFERENCE_FILENAME,
    COMBINED_WITHIN_REFERENCE_STATS_FILENAME,
    DISTRIBUTION_COLUMNS,
    DISTRIBUTION_DERIVED_VALUE_COLUMNS,
    DISTRIBUTION_OUTPUT_VALUE_COLUMNS,
    INDEPENDENT_UNIT,
    PAIRWISE_COLUMNS,
    PAIR_VALUES_INDEPENDENT,
    REMOVED_INFERENTIAL_FIELDS,
    STATISTICS_SCHEMA_VERSION,
    STATISTICAL_MODE,
    STATS_COLUMNS,
    statistical_contract_fields,
)
from relacc.pipeline.report_stats import (
    _aggregate_rows_for_record_set,
    _aggregate_summary_stats,
    _delta,
    _distribution_summary,
    _finite_metric_values,
    _lightweight_distribution_rows,
    _numeric_value,
    _quantile,
    _ratio,
    _raw_distribution_outputs_from_rows,
    _shape_statistic,
    _summary_stats,
    _validate_bounded_stats,
    build_combined_aggregate_summaries,
)
from relacc.utils.math import MathUtil
from relacc.utils.runlog import (
    add_run_logging_arguments,
    append_run_log,
    build_run_metadata,
    record_effective_config,
    run_logged_experiment,
    sidecar_paths,
    verbosity_from_opt,
    write_run_metadata,
)


DEFAULT_DATASETS_ROOT = Path("/Users/fede/S6-Project/datasets")
DEFAULT_RATE = 24
DEFAULT_VARIANT_LABEL = "root"


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run human-summary comparisons, human baselines, direct pairwise "
            "distribution evidence, and combined plotting exports for every "
            "generated source/dataset/class."
        )
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory containing humans and generated dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where pairwise report artifacts will be written.",
    )
    parser.add_argument(
        "--rate",
        default=str(DEFAULT_RATE),
        help="Fixed resampling rate for all comparisons. Default: 24.",
    )
    parser.add_argument("--round", default="3", help="Decimal precision for metrics.")
    parser.add_argument(
        "--alignment",
        type=PtAlignType.normalize,
        default=str(PtAlignType.CHRONOLOGICAL),
        help="Alignment mode value accepted by the existing pipelines.",
    )
    parser.add_argument(
        "--summary",
        default="medoid",
        help="Summary-shape mode. Default: medoid.",
    )
    parser.add_argument("--popular", action="store_true", help="Use popular stroke count.")
    parser.add_argument("--exact-dtw", action="store_true", help="Disable DTW windowing.")
    parser.add_argument("--dtw-window", default=None, help="Optional explicit DTW window.")
    parser.add_argument(
        "--group-by",
        default=GROUP_BY_FILENAME_LABEL,
        choices=GROUP_BY_MODES,
        help="Grouping mode for deriving dataset/class keys.",
    )
    parser.add_argument(
        "--class-scheme",
        default=CLASS_SCHEME_AUTO,
        choices=CLASS_SCHEMES,
        help="Class-label extraction scheme. Default: auto.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Optional comma-separated generated source names to run.",
    )
    parser.add_argument(
        "--sample-limit-per-class",
        default=None,
        help=(
            "Optional limit applied to both human/reference and generated samples "
            "for all outputs. Use for smoke tests; omit for full summary outputs."
        ),
    )
    parser.add_argument(
        "--distribution-sample-limit-per-class",
        default="16",
        help=(
            "Limit applied only to direct distribution cross-products. Default: 16. "
            "Use 'none' or 'full' to materialize all direct pairs."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        default=None,
        help="Optional deterministic random seed for class-level sampling.",
    )
    parser.add_argument(
        "--class-limit-per-run",
        default=None,
        help="Optional number of classes per run to process; intended for smoke tests.",
    )
    parser.add_argument(
        "--skip-direct-distribution-pairs",
        action="store_true",
        help="Skip within-reference and between-groups direct-pair exports.",
    )
    parser.add_argument(
        "--skip-raw-jsonl",
        action="store_true",
        help=(
            "Do not materialize/write combined/raw_metrics.jsonl or "
            "combined/raw_distributions.jsonl. Wide CSV outputs are still written."
        ),
    )
    add_run_logging_arguments(parser)
    return parser


def main(argv=None):
    parser = _build_parser()
    opt = parser.parse_args(argv)

    output_root = Path(opt.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sidecar_paths(
        output_root,
        opt.log_dir,
        stem="run-all-pairwise-reports",
        output_is_dir=True,
    )
    metadata = build_run_metadata(parser, opt, argv, "run-all-pairwise-reports")
    write_run_metadata(paths, metadata)
    return run_logged_experiment(
        paths,
        lambda: _run_reports(opt, output_root, paths, metadata),
    )


if __name__ == "__main__":
    raise SystemExit(main())
