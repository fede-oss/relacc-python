from __future__ import annotations

import math
import json
import statistics
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from scipy import stats

from relacc.distribution_metrics import (
    DISTRIBUTION_METRIC_NAMES,
    DISTRIBUTION_METRIC_SEMANTICS,
    compute_distribution_metrics,
)
from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import METRIC_NAMES
from relacc.utils.math import MathUtil

from ._common import (
    csv_escape,
    effective_dtw_window,
    infer_label_from_filename,
    load_csv_entries,
    normalize_summary_shape,
    sampling_rate_for_sets,
)
from . import pair_evidence as PairEvidence
from .dataset_discovery import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    GROUP_BY_PARENT_DIR,
    class_key_for_relative_path,
    filename_label_class_key,
    normalize_group_by,
    parent_dir_class_key,
)


DISTRIBUTION_MODE = "distribution"
SUMMARY_STAT_NAMES: Tuple[str, ...] = (
    "mean",
    "mdn",
    "sd",
    "variance",
    "min",
    "max",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "skewness",
    "kurtosis",
    "n",
)
STATISTICAL_MODE = "descriptive-pair-distances"
INDEPENDENT_UNIT = "gesture-file"
PAIR_VALUES_INDEPENDENT = False
STATISTICS_SCHEMA_VERSION = 2
REMOVED_INFERENTIAL_FIELDS: Tuple[str, ...] = (
    "meanCi95Low",
    "meanCi95High",
    "normalityPValue",
    "ksPValue",
)
WITHIN_REFERENCE_GROUP = PairEvidence.WITHIN_REFERENCE_SAMPLE_KIND
WITHIN_COMPARISON_GROUP = PairEvidence.WITHIN_COMPARISON_SAMPLE_KIND
BETWEEN_GROUPS = PairEvidence.BETWEEN_GROUPS_SAMPLE_KIND
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
    return normalize_group_by(group_by)


def _filename_label_class_key(relative_csv_path: str) -> str:
    return filename_label_class_key(relative_csv_path)


def _parent_dir_class_key(relative_csv_path: str) -> str:
    return parent_dir_class_key(relative_csv_path)


def _class_key_for_relative_path(relative_csv_path: str, group_by: str) -> str:
    return class_key_for_relative_path(relative_csv_path, group_by)


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

    return _validate_summary_stats({
        "mean": _rounded_stat(mean, round_precision),
        "mdn": _rounded_stat(mdn, round_precision),
        "sd": _rounded_stat(sd, round_precision),
        "variance": _rounded_stat(variance, round_precision),
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


def _statistical_contract_fields() -> Dict[str, object]:
    return {
        "statisticalMode": STATISTICAL_MODE,
        "independentUnit": INDEPENDENT_UNIT,
        "pairValuesIndependent": PAIR_VALUES_INDEPENDENT,
        "statisticsSchemaVersion": STATISTICS_SCHEMA_VERSION,
        "removedInferentialFields": list(REMOVED_INFERENTIAL_FIELDS),
    }


def _statistical_contract_csv_fields() -> Dict[str, object]:
    fields = _statistical_contract_fields()
    fields["removedInferentialFields"] = json.dumps(
        fields["removedInferentialFields"],
        separators=(",", ":"),
    )
    return fields


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
        **_statistical_contract_fields(),
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
    return_raw: bool = False,
):
    alignment_type = PtAlignType.normalize(alignment_type)
    reference_points = [entry.points for entry in spec.reference_entries]
    candidate_points = [entry.points for entry in spec.candidate_entries]
    effective_rate = sampling_rate_for_sets(reference_points + candidate_points, rate)
    selected_dtw_window = effective_dtw_window(effective_rate, dtw_window, exact_dtw)

    samples = {
        group_type: {metric_name: [] for metric_name in METRIC_NAMES}
        for group_type in COMPARISON_GROUP_TYPES
    }
    raw_metric_outputs = []
    pair_options = PairEvidence.PairMetricOptions(
        label=spec.class_key,
        effective_rate=effective_rate,
        alignment_type=alignment_type,
        summary_shape=summary_shape,
        popular_shape=popular_shape,
        metric_names=METRIC_NAMES,
        dtw_window=selected_dtw_window,
        exact_dtw=exact_dtw,
    )
    raw_base_fields = {
        "schemaVersion": 1,
        "recordType": "rawMetricOutput",
        "comparisonMode": DISTRIBUTION_MODE,
        "classKey": spec.class_key,
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment_type,
        "alignmentName": PtAlignType.name(alignment_type),
        "summary": summary_shape,
        "popular": bool(popular_shape),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }

    for left_entry, right_entry in _unordered_pairs(spec.reference_entries):
        evidence = PairEvidence.compute_bidirectional_pair_evidence(
            PairEvidence.endpoint_for(left_entry),
            PairEvidence.endpoint_for(right_entry),
            pair_options,
        )
        for metric_name in METRIC_NAMES:
            value = evidence.values[metric_name]
            samples[WITHIN_REFERENCE_GROUP][metric_name].append(value)
        raw_metric_outputs.extend(
            PairEvidence.distribution_within_reference_rows(evidence, raw_base_fields)
        )

    for left_entry, right_entry in _unordered_pairs(spec.candidate_entries):
        evidence = PairEvidence.compute_bidirectional_pair_evidence(
            PairEvidence.endpoint_for(left_entry),
            PairEvidence.endpoint_for(right_entry),
            pair_options,
        )
        for metric_name in METRIC_NAMES:
            value = evidence.values[metric_name]
            samples[WITHIN_COMPARISON_GROUP][metric_name].append(value)
        raw_metric_outputs.extend(
            PairEvidence.distribution_within_comparison_rows(evidence, raw_base_fields)
        )

    for reference_entry, candidate_entry in _between_group_pairs(
        spec.reference_entries,
        spec.candidate_entries,
    ):
        evidence = PairEvidence.compute_directional_pair_evidence(
            PairEvidence.endpoint_for(reference_entry),
            PairEvidence.endpoint_for(candidate_entry),
            pair_options,
        )
        for metric_name in METRIC_NAMES:
            samples[BETWEEN_GROUPS][metric_name].append(evidence.values[metric_name])
        raw_metric_outputs.extend(
            PairEvidence.distribution_between_groups_rows(evidence, raw_base_fields)
        )

    if return_raw:
        return samples, raw_metric_outputs, effective_rate, selected_dtw_window
    return samples, effective_rate, selected_dtw_window


def _raw_distribution_outputs(results: Sequence[Dict[str, object]]):
    rows = []
    for result in results:
        base = {
            "schemaVersion": 1,
            "recordType": "rawDistributionOutput",
            "comparisonMode": DISTRIBUTION_MODE,
            **_statistical_contract_fields(),
            "scope": result.get("scope"),
            "classKey": result.get("classKey"),
            "gestureMetric": result.get("gestureMetric"),
            "referenceGroupCount": result.get("referenceGroupCount"),
            "comparisonGroupCount": result.get("comparisonGroupCount"),
            "withinReferenceSampleCount": result.get("withinReferenceSampleCount"),
            "withinComparisonSampleCount": result.get("withinComparisonSampleCount"),
            "betweenGroupsSampleCount": result.get("betweenGroupsSampleCount"),
        }
        for distribution_metric, value in result.get("distributionMetrics", {}).items():
            rows.append(
                {
                    **base,
                    "distributionMetric": distribution_metric,
                    "value": value,
                }
            )
        for ratio_name, value in result.get("withinComparisonToReferenceRatios", {}).items():
            formatted_ratio_name = ratio_name[:1].upper() + ratio_name[1:]
            rows.append(
                {
                    **base,
                    "distributionMetric": (
                        "withinComparisonToReference%sRatio" % formatted_ratio_name
                    ),
                    "value": value,
                }
            )
    return rows


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
    alignment_type = PtAlignType.normalize(alignment_type)
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
    raw_metric_outputs = []

    for spec in valid_classes:
        (
            class_samples,
            class_raw_metric_outputs,
            _,
            selected_dtw_window,
        ) = _metric_samples_for_class(
            spec,
            rate=rate,
            alignment_type=alignment_type,
            summary_shape=summary_shape,
            popular_shape=popular_shape,
            dtw_window=dtw_window,
            exact_dtw=exact_dtw,
            return_raw=True,
        )
        raw_metric_outputs.extend(class_raw_metric_outputs)
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
    raw_distribution_outputs = [
        *_raw_distribution_outputs(per_class_results),
        *_raw_distribution_outputs(overall_results),
    ]

    return {
        "metadata": {
            "comparisonMode": DISTRIBUTION_MODE,
            **_statistical_contract_fields(),
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
            "alignmentName": PtAlignType.name(alignment_type),
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "roundPrecision": round_precision,
            "dtwWindow": (
                next(iter(effective_dtw_windows))
                if len(effective_dtw_windows) == 1
                else None
            ),
            "exactDtw": bool(exact_dtw),
            "distributionMetricSemantics": DISTRIBUTION_METRIC_SEMANTICS,
        },
        "results": {
            "perClass": per_class_results,
            "overall": overall_results,
        },
        "rawMetricOutputs": raw_metric_outputs,
        "rawDistributionOutputs": raw_distribution_outputs,
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
        "statisticalMode",
        "independentUnit",
        "pairValuesIndependent",
        "statisticsSchemaVersion",
        "removedInferentialFields",
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
        "statisticalMode",
        "independentUnit",
        "pairValuesIndependent",
        "statisticsSchemaVersion",
        "removedInferentialFields",
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
                **_statistical_contract_csv_fields(),
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
                **_statistical_contract_csv_fields(),
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
