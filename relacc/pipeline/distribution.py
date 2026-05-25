from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from relacc.distribution_metrics import DISTRIBUTION_METRIC_NAMES, compute_distribution_metrics
from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import METRIC_NAMES
from relacc.utils.math import MathUtil

from ._common import (
    compute_pair_metrics_from_points,
    csv_escape,
    effective_dtw_window,
    infer_label_from_filename,
    load_csv_entries,
    normalize_summary_shape,
    sampling_rate_for_sets,
)


DISTRIBUTION_MODE = "distribution"
GROUP_BY_FILENAME_LABEL = "filename-label"
GROUP_BY_PARENT_DIR = "parent-dir"
GROUP_BY_MODES: Tuple[str, str] = (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_PARENT_DIR,
)


@dataclass(frozen=True)
class GestureEntry:
    key: str
    path: str
    class_key: str
    points: list


@dataclass(frozen=True)
class ClassComparisonSpec:
    class_key: str
    reference_entries: Tuple[GestureEntry, ...]
    candidate_entries: Tuple[GestureEntry, ...]


def _normalize_group_by(group_by: str | None):
    mode = (group_by or GROUP_BY_FILENAME_LABEL).strip().lower()
    if mode not in GROUP_BY_MODES:
        raise ValueError(
            "Invalid group-by mode (%s). Supported values: filename-label, parent-dir."
            % group_by
        )
    return mode


def _filename_label_class_key(relative_csv_path: str) -> str:
    return infer_label_from_filename(relative_csv_path, "class")


def _parent_dir_class_key(relative_csv_path: str) -> str:
    parts = relative_csv_path.split("/")
    if len(parts) == 1:
        return "."
    return "/".join(parts[:-1])


def _class_key_for_relative_path(relative_csv_path: str, group_by: str) -> str:
    if group_by == GROUP_BY_FILENAME_LABEL:
        return _filename_label_class_key(relative_csv_path)
    return _parent_dir_class_key(relative_csv_path)


def _group_entries_by_class(entries, group_by: str):
    grouped: Dict[str, List[GestureEntry]] = {}
    for key, path, points in entries:
        class_key = _class_key_for_relative_path(key, group_by)
        grouped.setdefault(class_key, []).append(
            GestureEntry(key=key, path=path, class_key=class_key, points=points)
        )
    return grouped


def _invalid_class_entry(class_key: str, reason: str, reference_count: int, candidate_count: int):
    return {
        "classKey": class_key,
        "reason": reason,
        "referenceCount": reference_count,
        "candidateCount": candidate_count,
    }


def discover_class_comparisons(reference_input: str, candidate_input: str, group_by: str):
    reference_entries = load_csv_entries(reference_input)
    if len(reference_entries) == 0:
        raise ValueError("No reference CSV files found.")

    candidate_entries = load_csv_entries(candidate_input)
    if len(candidate_entries) == 0:
        raise ValueError("No candidate CSV files found.")

    reference_groups = _group_entries_by_class(reference_entries, group_by)
    candidate_groups = _group_entries_by_class(candidate_entries, group_by)

    valid_classes: List[ClassComparisonSpec] = []
    skipped_classes = []
    invalid_classes = []

    for class_key in sorted(set(reference_groups.keys()) | set(candidate_groups.keys())):
        references = tuple(reference_groups.get(class_key, []))
        candidates = tuple(candidate_groups.get(class_key, []))

        if len(references) == 0:
            skipped_classes.append(
                _invalid_class_entry(class_key, "missingReference", 0, len(candidates))
            )
            continue
        if len(candidates) == 0:
            skipped_classes.append(
                _invalid_class_entry(class_key, "missingCandidate", len(references), 0)
            )
            continue
        if len(references) < 2:
            invalid_classes.append(
                _invalid_class_entry(
                    class_key,
                    "needAtLeastTwoReferenceSamples",
                    len(references),
                    len(candidates),
                )
            )
            continue

        valid_classes.append(
            ClassComparisonSpec(
                class_key=class_key,
                reference_entries=references,
                candidate_entries=candidates,
            )
        )

    return valid_classes, skipped_classes, invalid_classes


def _unordered_reference_pairs(reference_entries: Sequence[GestureEntry]):
    return list(combinations(reference_entries, 2))


def _candidate_reference_pairs(
    reference_entries: Sequence[GestureEntry],
    candidate_entries: Sequence[GestureEntry],
):
    return [
        (reference_entry, candidate_entry)
        for reference_entry in reference_entries
        for candidate_entry in candidate_entries
    ]


def _summary_stats(values: Sequence[float], round_precision: int | None):
    finite_values = [value for value in values if math.isfinite(value)]
    n = len(finite_values)
    mean = statistics.fmean(finite_values)
    mdn = statistics.median(finite_values)
    sd = statistics.stdev(finite_values) if n > 1 else 0.0
    minimum = min(finite_values)
    maximum = max(finite_values)

    return {
        "mean": MathUtil.roundTo(mean, round_precision),
        "mdn": MathUtil.roundTo(mdn, round_precision),
        "sd": MathUtil.roundTo(sd, round_precision),
        "min": MathUtil.roundTo(minimum, round_precision),
        "max": MathUtil.roundTo(maximum, round_precision),
        "n": n,
    }


def _build_result_entry(
    scope: str,
    class_key: str | None,
    gesture_metric: str,
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    round_precision: int,
    reference_count: int,
    candidate_count: int,
):
    return {
        "scope": scope,
        "classKey": class_key,
        "gestureMetric": gesture_metric,
        "referenceCount": reference_count,
        "candidateCount": candidate_count,
        "baselineSampleCount": len(baseline_values),
        "candidateSampleCount": len(candidate_values),
        "baselineStats": _summary_stats(baseline_values, round_precision),
        "candidateStats": _summary_stats(candidate_values, round_precision),
        "distributionMetrics": compute_distribution_metrics(
            baseline_values,
            candidate_values,
            round_precision=round_precision,
        ),
    }


def _metric_samples_for_class(
    spec: ClassComparisonSpec,
    rate: int | None,
    alignment_type: int,
    summary_shape: str | None,
    popular_shape: bool,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    reference_points = [entry.points for entry in spec.reference_entries]
    effective_rate = sampling_rate_for_sets(reference_points, rate)
    selected_dtw_window = effective_dtw_window(effective_rate, dtw_window, exact_dtw)

    baseline_samples = {name: [] for name in METRIC_NAMES}
    candidate_samples = {name: [] for name in METRIC_NAMES}

    for left_entry, right_entry in _unordered_reference_pairs(spec.reference_entries):
        forward_values = compute_pair_metrics_from_points(
            left_entry.points,
            right_entry.points,
            spec.class_key,
            effective_rate,
            alignment_type=alignment_type,
            summary_shape=summary_shape,
            popular_shape=popular_shape,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        backward_values = compute_pair_metrics_from_points(
            right_entry.points,
            left_entry.points,
            spec.class_key,
            effective_rate,
            alignment_type=alignment_type,
            summary_shape=summary_shape,
            popular_shape=popular_shape,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        for metric_name in METRIC_NAMES:
            baseline_samples[metric_name].append(
                (forward_values[metric_name] + backward_values[metric_name]) / 2.0
            )

    for reference_entry, candidate_entry in _candidate_reference_pairs(
        spec.reference_entries,
        spec.candidate_entries,
    ):
        values = compute_pair_metrics_from_points(
            reference_entry.points,
            candidate_entry.points,
            spec.class_key,
            effective_rate,
            alignment_type=alignment_type,
            summary_shape=summary_shape,
            popular_shape=popular_shape,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        for metric_name in METRIC_NAMES:
            candidate_samples[metric_name].append(values[metric_name])

    return baseline_samples, candidate_samples, effective_rate, selected_dtw_window


def run_distribution_comparison(
    reference_input: str,
    candidate_input: str,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int = 3,
    group_by: str = GROUP_BY_FILENAME_LABEL,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    summary_shape = normalize_summary_shape(summary_shape)
    group_by = _normalize_group_by(group_by)
    valid_classes, skipped_classes, invalid_classes = discover_class_comparisons(
        reference_input,
        candidate_input,
        group_by,
    )

    if len(valid_classes) == 0:
        raise ValueError("No valid classes found for distribution comparison.")

    per_class_results = []
    overall_samples = {
        metric_name: {"baseline": [], "candidate": []}
        for metric_name in METRIC_NAMES
    }

    total_reference_count = 0
    total_candidate_count = 0
    effective_dtw_windows = set()

    for spec in valid_classes:
        baseline_samples, candidate_samples, _, selected_dtw_window = _metric_samples_for_class(
            spec,
            rate=rate,
            alignment_type=alignment_type,
            summary_shape=summary_shape,
            popular_shape=popular_shape,
            dtw_window=dtw_window,
            exact_dtw=exact_dtw,
        )
        effective_dtw_windows.add(selected_dtw_window)
        total_reference_count += len(spec.reference_entries)
        total_candidate_count += len(spec.candidate_entries)

        for metric_name in METRIC_NAMES:
            baseline_values = baseline_samples[metric_name]
            candidate_values = candidate_samples[metric_name]
            overall_samples[metric_name]["baseline"].extend(baseline_values)
            overall_samples[metric_name]["candidate"].extend(candidate_values)
            per_class_results.append(
                _build_result_entry(
                    "class",
                    spec.class_key,
                    metric_name,
                    baseline_values,
                    candidate_values,
                    round_precision,
                    len(spec.reference_entries),
                    len(spec.candidate_entries),
                )
            )

    overall_results = [
        _build_result_entry(
            "overall",
            None,
            metric_name,
            overall_samples[metric_name]["baseline"],
            overall_samples[metric_name]["candidate"],
            round_precision,
            total_reference_count,
            total_candidate_count,
        )
        for metric_name in METRIC_NAMES
    ]

    return {
        "metadata": {
            "comparisonMode": DISTRIBUTION_MODE,
            "groupBy": group_by,
            "validClassCount": len(valid_classes),
            "skippedClasses": skipped_classes,
            "invalidClasses": invalid_classes,
            "rate": rate,
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "roundPrecision": round_precision,
            "dtwWindow": (
                next(iter(effective_dtw_windows))
                if len(effective_dtw_windows) == 1
                else None
            ),
            "exactDtw": bool(exact_dtw),
        },
        "results": {
            "perClass": per_class_results,
            "overall": overall_results,
        },
    }


def format_distribution_rows_csv(results: Dict[str, Sequence[Dict[str, object]]]) -> str:
    columns: List[str] = [
        "scope",
        "classKey",
        "gestureMetric",
        "referenceCount",
        "candidateCount",
        "baselineSampleCount",
        "candidateSampleCount",
        "baselineMean",
        "baselineMdn",
        "baselineSd",
        "baselineMin",
        "baselineMax",
        "candidateMean",
        "candidateMdn",
        "candidateSd",
        "candidateMin",
        "candidateMax",
        *DISTRIBUTION_METRIC_NAMES,
    ]
    lines = [",".join(columns)]

    for row in list(results.get("perClass", [])) + list(results.get("overall", [])):
        baseline_stats = row.get("baselineStats", {})
        candidate_stats = row.get("candidateStats", {})
        distribution_metrics = row.get("distributionMetrics", {})
        flat_row = {
            "scope": row.get("scope"),
            "classKey": row.get("classKey"),
            "gestureMetric": row.get("gestureMetric"),
            "referenceCount": row.get("referenceCount"),
            "candidateCount": row.get("candidateCount"),
            "baselineSampleCount": row.get("baselineSampleCount"),
            "candidateSampleCount": row.get("candidateSampleCount"),
            "baselineMean": baseline_stats.get("mean"),
            "baselineMdn": baseline_stats.get("mdn"),
            "baselineSd": baseline_stats.get("sd"),
            "baselineMin": baseline_stats.get("min"),
            "baselineMax": baseline_stats.get("max"),
            "candidateMean": candidate_stats.get("mean"),
            "candidateMdn": candidate_stats.get("mdn"),
            "candidateSd": candidate_stats.get("sd"),
            "candidateMin": candidate_stats.get("min"),
            "candidateMax": candidate_stats.get("max"),
        }
        flat_row.update(distribution_metrics)

        lines.append(",".join(csv_escape(flat_row.get(column, "")) for column in columns))

    return "\n".join(lines)
