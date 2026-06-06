from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from scipy import stats

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
SUMMARY_STAT_NAMES: Tuple[str, ...] = (
    "mean",
    "mdn",
    "sd",
    "variance",
    "meanCi95Low",
    "meanCi95High",
    "min",
    "max",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "skewness",
    "kurtosis",
    "normalityPValue",
    "n",
)
WITHIN_REFERENCE_GROUP = "withinReference"
WITHIN_COMPARISON_GROUP = "withinComparison"
BETWEEN_GROUPS = "betweenGroups"
COMPARISON_GROUP_TYPES: Tuple[str, str, str] = (
    WITHIN_REFERENCE_GROUP,
    WITHIN_COMPARISON_GROUP,
    BETWEEN_GROUPS,
)
RATIO_STAT_NAMES: Tuple[str, str, str] = ("mean", "mdn", "sd")


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
        "referenceGroupCount": reference_count,
        "comparisonGroupCount": candidate_count,
    }


def discover_class_comparisons(reference_input: str, candidate_input: str, group_by: str):
    reference_entries = load_csv_entries(reference_input)
    if len(reference_entries) == 0:
        raise ValueError("No reference CSV files found.")

    candidate_entries = load_csv_entries(candidate_input)
    if len(candidate_entries) == 0:
        raise ValueError("No comparison CSV files found.")

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
                _invalid_class_entry(class_key, "missingComparison", len(references), 0)
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


def _unordered_pairs(entries: Sequence[GestureEntry]):
    return list(combinations(entries, 2))


def _between_group_pairs(
    reference_entries: Sequence[GestureEntry],
    candidate_entries: Sequence[GestureEntry],
):
    return [
        (reference_entry, candidate_entry)
        for reference_entry in reference_entries
        for candidate_entry in candidate_entries
    ]


def _rounded_stat(value: float, round_precision: int | None):
    return MathUtil.roundTo(value, round_precision) if math.isfinite(value) else value


def _empty_summary_stats(round_precision: int | None):
    return {
        stat_name: (
            0
            if stat_name == "n"
            else _rounded_stat(float("nan"), round_precision)
        )
        for stat_name in SUMMARY_STAT_NAMES
    }


def _validate_summary_stats(stats: Dict[str, float]) -> Dict[str, float]:
    minimum = stats["min"]
    maximum = stats["max"]
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        return stats

    for stat_name in ("mean", "mdn", "q05", "q25", "q50", "q75", "q95"):
        value = stats[stat_name]
        if math.isfinite(value) and not minimum <= value <= maximum:
            raise ValueError(
                "Invalid summary statistics: %s=%s is outside min=%s and max=%s."
                % (stat_name, value, minimum, maximum)
            )
    return stats


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[int(position)]

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + ((upper - lower) * (position - lower_index))


def _shape_statistic(
    finite_values: Sequence[float],
    minimum_n: int,
    statistic_fn,
    round_precision: int | None,
) -> float:
    if len(finite_values) < minimum_n:
        return _rounded_stat(float("nan"), round_precision)
    if len(set(finite_values)) == 1:
        return _rounded_stat(0.0, round_precision)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(statistic_fn(finite_values))
    return _rounded_stat(value, round_precision)


def _normality_p_value(
    finite_values: Sequence[float],
    round_precision: int | None,
) -> float:
    if len(finite_values) < 8 or len(set(finite_values)) == 1:
        return _rounded_stat(float("nan"), round_precision)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value = stats.normaltest(finite_values)
    return _rounded_stat(float(p_value), round_precision)


def _mean_ci_95(
    mean: float,
    sd: float,
    n: int,
    round_precision: int | None,
) -> tuple[float, float]:
    if n == 0:
        empty = _rounded_stat(float("nan"), round_precision)
        return empty, empty
    if n == 1 or sd == 0:
        rounded_mean = _rounded_stat(mean, round_precision)
        return rounded_mean, rounded_mean

    margin = float(stats.t.ppf(0.975, n - 1)) * (sd / math.sqrt(n))
    return (
        _rounded_stat(mean - margin, round_precision),
        _rounded_stat(mean + margin, round_precision),
    )


def _summary_stats(values: Sequence[float], round_precision: int | None):
    finite_values = [value for value in values if math.isfinite(value)]
    n = len(finite_values)
    if n == 0:
        return _empty_summary_stats(round_precision)

    sorted_values = sorted(finite_values)
    mean = statistics.fmean(finite_values)
    mdn = statistics.median(finite_values)
    sd = statistics.stdev(finite_values) if n > 1 else 0.0
    variance = statistics.variance(finite_values) if n > 1 else 0.0
    minimum = min(finite_values)
    maximum = max(finite_values)
    mean_ci_95_low, mean_ci_95_high = _mean_ci_95(mean, sd, n, round_precision)

    return _validate_summary_stats({
        "mean": _rounded_stat(mean, round_precision),
        "mdn": _rounded_stat(mdn, round_precision),
        "sd": _rounded_stat(sd, round_precision),
        "variance": _rounded_stat(variance, round_precision),
        "meanCi95Low": mean_ci_95_low,
        "meanCi95High": mean_ci_95_high,
        "min": _rounded_stat(minimum, round_precision),
        "max": _rounded_stat(maximum, round_precision),
        "q05": _rounded_stat(_quantile(sorted_values, 0.05), round_precision),
        "q25": _rounded_stat(_quantile(sorted_values, 0.25), round_precision),
        "q50": _rounded_stat(_quantile(sorted_values, 0.50), round_precision),
        "q75": _rounded_stat(_quantile(sorted_values, 0.75), round_precision),
        "q95": _rounded_stat(_quantile(sorted_values, 0.95), round_precision),
        "skewness": _shape_statistic(
            finite_values,
            3,
            lambda series: stats.skew(series, bias=False),
            round_precision,
        ),
        "kurtosis": _shape_statistic(
            finite_values,
            4,
            lambda series: stats.kurtosis(series, fisher=True, bias=False),
            round_precision,
        ),
        "normalityPValue": _normality_p_value(finite_values, round_precision),
        "n": n,
    })


def _safe_ratio(numerator, denominator, round_precision: int | None):
    if numerator is None or denominator is None:
        return _rounded_stat(float("nan"), round_precision)
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return _rounded_stat(float("nan"), round_precision)
    if not math.isfinite(numerator_value) or not math.isfinite(denominator_value):
        return _rounded_stat(float("nan"), round_precision)
    if denominator_value == 0:
        return _rounded_stat(float("nan"), round_precision)
    return _rounded_stat(numerator_value / denominator_value, round_precision)


def _within_comparison_to_reference_ratios(
    within_reference_stats: Dict[str, object],
    within_comparison_stats: Dict[str, object],
    round_precision: int | None,
):
    return {
        stat_name: _safe_ratio(
            within_comparison_stats.get(stat_name),
            within_reference_stats.get(stat_name),
            round_precision,
        )
        for stat_name in RATIO_STAT_NAMES
    }


def _add_legacy_result_fields(result: Dict[str, object]):
    result["referenceCount"] = result["referenceGroupCount"]
    result["candidateCount"] = result["comparisonGroupCount"]
    result["baselineSampleCount"] = result["withinReferenceSampleCount"]
    result["candidateSampleCount"] = result["betweenGroupsSampleCount"]
    result["baselineStats"] = result["withinReferenceStats"]
    result["candidateStats"] = result["betweenGroupsStats"]
    return result


def _build_result_entry(
    scope: str,
    class_key: str | None,
    gesture_metric: str,
    within_reference_values: Sequence[float],
    within_comparison_values: Sequence[float],
    between_group_values: Sequence[float],
    round_precision: int,
    reference_count: int,
    candidate_count: int,
    include_legacy_fields: bool = False,
):
    within_reference_stats = _summary_stats(within_reference_values, round_precision)
    within_comparison_stats = _summary_stats(within_comparison_values, round_precision)
    between_group_stats = _summary_stats(between_group_values, round_precision)
    result = {
        "scope": scope,
        "classKey": class_key,
        "gestureMetric": gesture_metric,
        "referenceGroupCount": reference_count,
        "comparisonGroupCount": candidate_count,
        "withinReferenceSampleCount": len(within_reference_values),
        "withinComparisonSampleCount": len(within_comparison_values),
        "betweenGroupsSampleCount": len(between_group_values),
        "withinReferenceStats": within_reference_stats,
        "withinComparisonStats": within_comparison_stats,
        "betweenGroupsStats": between_group_stats,
        "withinComparisonToReferenceRatios": _within_comparison_to_reference_ratios(
            within_reference_stats,
            within_comparison_stats,
            round_precision,
        ),
        "distributionMetrics": compute_distribution_metrics(
            within_reference_values,
            between_group_values,
            round_precision=round_precision,
        ),
    }
    if include_legacy_fields:
        _add_legacy_result_fields(result)
    return result


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

    samples = {
        group_type: {metric_name: [] for metric_name in METRIC_NAMES}
        for group_type in COMPARISON_GROUP_TYPES
    }

    for left_entry, right_entry in _unordered_pairs(spec.reference_entries):
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
            samples[WITHIN_REFERENCE_GROUP][metric_name].append(
                (forward_values[metric_name] + backward_values[metric_name]) / 2.0
            )

    for left_entry, right_entry in _unordered_pairs(spec.candidate_entries):
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
            samples[WITHIN_COMPARISON_GROUP][metric_name].append(
                (forward_values[metric_name] + backward_values[metric_name]) / 2.0
            )

    for reference_entry, candidate_entry in _between_group_pairs(
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
            samples[BETWEEN_GROUPS][metric_name].append(values[metric_name])

    return samples, effective_rate, selected_dtw_window


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
    reference_group_name: str = "reference",
    comparison_group_name: str = "comparison",
    include_legacy_fields: bool = False,
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
        group_type: {metric_name: [] for metric_name in METRIC_NAMES}
        for group_type in COMPARISON_GROUP_TYPES
    }

    total_reference_count = 0
    total_candidate_count = 0
    effective_dtw_windows = set()

    for spec in valid_classes:
        class_samples, _, selected_dtw_window = _metric_samples_for_class(
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
            within_reference_values = class_samples[WITHIN_REFERENCE_GROUP][metric_name]
            within_comparison_values = class_samples[WITHIN_COMPARISON_GROUP][metric_name]
            between_group_values = class_samples[BETWEEN_GROUPS][metric_name]
            overall_samples[WITHIN_REFERENCE_GROUP][metric_name].extend(
                within_reference_values
            )
            overall_samples[WITHIN_COMPARISON_GROUP][metric_name].extend(
                within_comparison_values
            )
            overall_samples[BETWEEN_GROUPS][metric_name].extend(between_group_values)
            per_class_results.append(
                _build_result_entry(
                    "class",
                    spec.class_key,
                    metric_name,
                    within_reference_values,
                    within_comparison_values,
                    between_group_values,
                    round_precision,
                    len(spec.reference_entries),
                    len(spec.candidate_entries),
                    include_legacy_fields=include_legacy_fields,
                )
            )

    overall_results = [
        _build_result_entry(
            "overall",
            None,
            metric_name,
            overall_samples[WITHIN_REFERENCE_GROUP][metric_name],
            overall_samples[WITHIN_COMPARISON_GROUP][metric_name],
            overall_samples[BETWEEN_GROUPS][metric_name],
            round_precision,
            total_reference_count,
            total_candidate_count,
            include_legacy_fields=include_legacy_fields,
        )
        for metric_name in METRIC_NAMES
    ]

    return {
        "metadata": {
            "comparisonMode": DISTRIBUTION_MODE,
            "groupBy": group_by,
            "referenceGroupName": reference_group_name,
            "comparisonGroupName": comparison_group_name,
            "comparisonGroups": {
                WITHIN_REFERENCE_GROUP: {
                    "description": "Pairwise distances among samples in the reference group.",
                    "groupName": reference_group_name,
                },
                WITHIN_COMPARISON_GROUP: {
                    "description": "Pairwise distances among samples in the comparison group.",
                    "groupName": comparison_group_name,
                },
                BETWEEN_GROUPS: {
                    "description": "Distances from reference-group samples to comparison-group samples.",
                    "groupNames": [reference_group_name, comparison_group_name],
                },
            },
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


def _append_group_stat_columns(columns: List[str], prefix: str):
    for stat_name in SUMMARY_STAT_NAMES:
        if stat_name == "n":
            continue
        columns.append(f"{prefix}{stat_name[:1].upper()}{stat_name[1:]}")


def _add_group_stats_to_row(
    flat_row: Dict[str, object],
    prefix: str,
    stats_values: Dict[str, object],
):
    for stat_name in SUMMARY_STAT_NAMES:
        if stat_name == "n":
            continue
        flat_row[f"{prefix}{stat_name[:1].upper()}{stat_name[1:]}"] = stats_values.get(
            stat_name
        )


def _generic_distribution_columns() -> List[str]:
    columns: List[str] = [
        "scope",
        "classKey",
        "gestureMetric",
        "referenceGroupCount",
        "comparisonGroupCount",
        "withinReferenceSampleCount",
        "withinReferenceFiniteSampleCount",
        "withinComparisonSampleCount",
        "withinComparisonFiniteSampleCount",
        "betweenGroupsSampleCount",
        "betweenGroupsFiniteSampleCount",
    ]
    _append_group_stat_columns(columns, WITHIN_REFERENCE_GROUP)
    _append_group_stat_columns(columns, WITHIN_COMPARISON_GROUP)
    _append_group_stat_columns(columns, BETWEEN_GROUPS)
    columns.extend(
        [
            "withinComparisonToReferenceMeanRatio",
            "withinComparisonToReferenceMdnRatio",
            "withinComparisonToReferenceSdRatio",
            *DISTRIBUTION_METRIC_NAMES,
        ]
    )
    return columns


def _legacy_distribution_columns() -> List[str]:
    columns: List[str] = [
        "scope",
        "classKey",
        "gestureMetric",
        "referenceCount",
        "candidateCount",
        "baselineSampleCount",
        "baselineFiniteSampleCount",
        "candidateSampleCount",
        "candidateFiniteSampleCount",
    ]
    _append_group_stat_columns(columns, "baseline")
    _append_group_stat_columns(columns, "candidate")
    columns.extend(DISTRIBUTION_METRIC_NAMES)
    return columns


def format_distribution_rows_csv(
    results: Dict[str, Sequence[Dict[str, object]]],
    legacy_column_names: bool = False,
) -> str:
    columns = (
        _legacy_distribution_columns()
        if legacy_column_names
        else _generic_distribution_columns()
    )
    lines = [",".join(columns)]

    for row in list(results.get("perClass", [])) + list(results.get("overall", [])):
        within_reference_stats = row.get("withinReferenceStats", {})
        within_comparison_stats = row.get("withinComparisonStats", {})
        between_group_stats = row.get("betweenGroupsStats", {})
        ratios = row.get("withinComparisonToReferenceRatios", {})
        distribution_metrics = row.get("distributionMetrics", {})
        if legacy_column_names:
            flat_row = {
                "scope": row.get("scope"),
                "classKey": row.get("classKey"),
                "gestureMetric": row.get("gestureMetric"),
                "referenceCount": row.get("referenceGroupCount"),
                "candidateCount": row.get("comparisonGroupCount"),
                "baselineSampleCount": row.get("withinReferenceSampleCount"),
                "baselineFiniteSampleCount": within_reference_stats.get("n"),
                "candidateSampleCount": row.get("betweenGroupsSampleCount"),
                "candidateFiniteSampleCount": between_group_stats.get("n"),
            }
            _add_group_stats_to_row(flat_row, "baseline", within_reference_stats)
            _add_group_stats_to_row(flat_row, "candidate", between_group_stats)
        else:
            flat_row = {
                "scope": row.get("scope"),
                "classKey": row.get("classKey"),
                "gestureMetric": row.get("gestureMetric"),
                "referenceGroupCount": row.get("referenceGroupCount"),
                "comparisonGroupCount": row.get("comparisonGroupCount"),
                "withinReferenceSampleCount": row.get("withinReferenceSampleCount"),
                "withinReferenceFiniteSampleCount": within_reference_stats.get("n"),
                "withinComparisonSampleCount": row.get("withinComparisonSampleCount"),
                "withinComparisonFiniteSampleCount": within_comparison_stats.get("n"),
                "betweenGroupsSampleCount": row.get("betweenGroupsSampleCount"),
                "betweenGroupsFiniteSampleCount": between_group_stats.get("n"),
                "withinComparisonToReferenceMeanRatio": ratios.get("mean"),
                "withinComparisonToReferenceMdnRatio": ratios.get("mdn"),
                "withinComparisonToReferenceSdRatio": ratios.get("sd"),
            }
            _add_group_stats_to_row(
                flat_row,
                WITHIN_REFERENCE_GROUP,
                within_reference_stats,
            )
            _add_group_stats_to_row(
                flat_row,
                WITHIN_COMPARISON_GROUP,
                within_comparison_stats,
            )
            _add_group_stats_to_row(flat_row, BETWEEN_GROUPS, between_group_stats)
        flat_row.update(distribution_metrics)

        lines.append(",".join(csv_escape(flat_row.get(column, "")) for column in columns))

    return "\n".join(lines)
