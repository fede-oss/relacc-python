from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

from relacc.dtw import DEFAULT_EXACT_RATE_THRESHOLD, recommended_window
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import compute_metrics
from relacc.pipeline._common import (
    compute_pair_metrics_from_points,
    sampling_rate_for_sets,
    summary_sampling_rate,
)
from relacc.pipeline.reporting import ReportingEntry
from relacc.pipeline.report_schema import (
    statistical_contract_fields as _statistical_contract_fields,
)
from relacc.pipeline.report_stats import _summary_stats, _variant_label
from relacc.utils.math import MathUtil


def _rounded_metric_values(values: Dict[str, float], round_precision: int | None):
    if round_precision is None:
        return values
    return {
        metric_name: MathUtil.roundTo(value, round_precision)
        for metric_name, value in values.items()
    }


def _raw_metric_outputs_from_values(
    row: Dict[str, object],
    raw_values: Dict[str, float],
    record_set: str,
) -> List[Dict[str, object]]:
    base = {
        "schemaVersion": 1,
        "recordType": "rawMetricOutput",
        "recordSet": record_set,
        "comparisonMode": row.get("mode"),
        "runId": row.get("runId"),
        "source": row.get("source"),
        "dataset": row.get("dataset"),
        "variant": row.get("variant"),
        "classKey": row.get("classKey"),
        "referenceInput": row.get("referenceInput"),
        "referenceCount": row.get("referenceCount"),
        "candidateCount": row.get("candidateCount"),
        "rate": row.get("rate"),
        "requestedRate": row.get("requestedRate"),
        "alignment": row.get("alignment"),
        "alignmentName": row.get("alignmentName"),
        "summary": row.get("summary"),
        "popular": row.get("popular"),
        "dtwWindow": row.get("dtwWindow"),
        "exactDtw": row.get("exactDtw"),
    }
    for optional_key in [
        "pairKey",
        "candidateFile",
        "sampleKey",
        "sampleFile",
    ]:
        if optional_key in row:
            base[optional_key] = row.get(optional_key)

    return [
        {
            **base,
            "metric": metric_name,
            "value": value,
        }
        for metric_name, value in raw_values.items()
    ]


def _effective_dtw_window(rate: int, requested_window: int | None, exact_dtw: bool):
    if exact_dtw:
        return None
    if requested_window is not None:
        return requested_window
    if rate <= DEFAULT_EXACT_RATE_THRESHOLD:
        return None
    return recommended_window(rate)


def _compare_class(
    reference_entries: Sequence[ReportingEntry],
    candidate_entries: Sequence[ReportingEntry],
    within_comparison_entries: Sequence[ReportingEntry],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    reference_input: Path,
    rate: int | None,
    alignment: int,
    summary_shape: str | None,
    popular: bool,
    round_precision: int | None,
    metric_names: Sequence[str],
    dtw_window: int | None,
    exact_dtw: bool,
    collect_raw_outputs: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    reference_points = [entry.points for entry in reference_entries]
    candidate_points = [entry.points for entry in candidate_entries]
    effective_rate = summary_sampling_rate(reference_points, candidate_points, rate)
    selected_dtw_window = _effective_dtw_window(effective_rate, dtw_window, exact_dtw)
    summary_label = f"{dataset_name}/{class_key}/human-summary"
    reference_gestures = [
        Gesture(entry.points, summary_label, effective_rate)
        for entry in reference_entries
    ]
    reference_summary = SummaryGesture(
        reference_gestures,
        alignment,
        summary_shape,
        popular,
    )

    rows = []
    raw_metric_outputs = []
    for candidate_entry in candidate_entries:
        candidate = Gesture(candidate_entry.points, summary_label, effective_rate)
        raw_metric_values = compute_metrics(
            candidate,
            reference_summary,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
        )
        metric_values = _rounded_metric_values(raw_metric_values, round_precision)
        row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "pairKey": Path(candidate_entry.key).with_suffix("").as_posix(),
            "candidateFile": candidate_entry.path,
            "referenceInput": str(reference_input),
            "mode": "reference-summary",
            "referenceCount": len(reference_entries),
            "candidateCount": len(candidate_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(metric_values)
        rows.append(row)
        if collect_raw_outputs:
            raw_metric_outputs.extend(
                _raw_metric_outputs_from_values(
                    row,
                    raw_metric_values,
                    "comparison-to-reference-summary",
                )
            )

    within_comparison_rows = []
    for left_entry, right_entry in combinations(within_comparison_entries, 2):
        forward_values = compute_pair_metrics_from_points(
            left_entry.points,
            right_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        backward_values = compute_pair_metrics_from_points(
            right_entry.points,
            left_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        raw_metric_values = {
            metric_name: (forward_values[metric_name] + backward_values[metric_name]) / 2.0
            for metric_name in metric_names
        }
        row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "pairKey": "%s::%s"
            % (
                Path(left_entry.key).with_suffix("").as_posix(),
                Path(right_entry.key).with_suffix("").as_posix(),
            ),
            "candidateFile": "%s::%s" % (left_entry.path, right_entry.path),
            "referenceInput": str(reference_input),
            "mode": "within-comparison",
            "referenceCount": len(reference_entries),
            "candidateCount": len(within_comparison_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(_rounded_metric_values(raw_metric_values, round_precision))
        within_comparison_rows.append(row)
        if collect_raw_outputs:
            raw_metric_outputs.extend(
                _raw_metric_outputs_from_values(
                    row,
                    raw_metric_values,
                    "within-comparison",
                )
            )

    stats_rows = _summary_stats(
        rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )
    within_comparison_stats_rows = _summary_stats(
        within_comparison_rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )
    metadata = {
        **_statistical_contract_fields(),
        "runId": run_id,
        "source": source_name,
        "dataset": dataset_name,
        "variant": _variant_label(variant),
        "classKey": class_key,
        "mode": "reference-summary",
        "referenceCount": len(reference_entries),
        "candidateCount": len(candidate_entries),
        "withinComparisonPairs": len(within_comparison_rows),
        "withinComparisonCandidateCount": len(within_comparison_entries),
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment,
        "alignmentName": PtAlignType.name(alignment),
        "summary": summary_shape,
        "popular": bool(popular),
        "roundPrecision": round_precision,
        "metricNames": list(metric_names),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }
    return (
        rows,
        stats_rows,
        within_comparison_rows,
        within_comparison_stats_rows,
        raw_metric_outputs,
        metadata,
    )


def _baseline_stats(
    rows: Sequence[Dict[str, object]],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    summary_shape: str | None,
    round_precision: int | None,
):
    return _summary_stats(
        rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )


def _compare_direct_distribution_pairs_class(
    reference_entries: Sequence[ReportingEntry],
    candidate_entries: Sequence[ReportingEntry],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    reference_input: Path,
    rate: int | None,
    alignment: int,
    summary_shape: str | None,
    popular: bool,
    round_precision: int | None,
    metric_names: Sequence[str],
    dtw_window: int | None,
    exact_dtw: bool,
    collect_raw_outputs: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    reference_points = [entry.points for entry in reference_entries]
    candidate_points = [entry.points for entry in candidate_entries]
    effective_rate = sampling_rate_for_sets(reference_points + candidate_points, rate)
    selected_dtw_window = _effective_dtw_window(effective_rate, dtw_window, exact_dtw)

    within_reference_rows = []
    between_group_rows = []
    raw_metric_outputs = []

    for left_entry, right_entry in combinations(reference_entries, 2):
        forward_values = compute_pair_metrics_from_points(
            left_entry.points,
            right_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        backward_values = compute_pair_metrics_from_points(
            right_entry.points,
            left_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        raw_metric_values = {
            metric_name: (forward_values[metric_name] + backward_values[metric_name]) / 2.0
            for metric_name in metric_names
        }
        row = {
            "runId": run_id,
            "source": "human",
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "pairKey": "%s::%s"
            % (
                Path(left_entry.key).with_suffix("").as_posix(),
                Path(right_entry.key).with_suffix("").as_posix(),
            ),
            "candidateFile": "%s::%s" % (left_entry.path, right_entry.path),
            "referenceInput": str(reference_input),
            "mode": "within-reference",
            "referenceCount": len(reference_entries),
            "candidateCount": len(candidate_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(_rounded_metric_values(raw_metric_values, round_precision))
        within_reference_rows.append(row)
        if collect_raw_outputs:
            raw_metric_outputs.extend(
                _raw_metric_outputs_from_values(
                    row,
                    raw_metric_values,
                    "within-reference",
                )
            )

    for reference_entry in reference_entries:
        for candidate_entry in candidate_entries:
            raw_metric_values = compute_pair_metrics_from_points(
                reference_entry.points,
                candidate_entry.points,
                class_key,
                effective_rate,
                alignment_type=alignment,
                summary_shape=summary_shape,
                popular_shape=popular,
                round_precision=None,
                metric_names=metric_names,
                dtw_window=selected_dtw_window,
                exact_dtw=exact_dtw,
            )
            row = {
                "runId": run_id,
                "source": source_name,
                "dataset": dataset_name,
                "variant": _variant_label(variant),
                "classKey": class_key,
                "pairKey": "%s::%s"
                % (
                    Path(reference_entry.key).with_suffix("").as_posix(),
                    Path(candidate_entry.key).with_suffix("").as_posix(),
                ),
                "candidateFile": candidate_entry.path,
                "referenceInput": reference_entry.path,
                "mode": "between-groups",
                "referenceCount": len(reference_entries),
                "candidateCount": len(candidate_entries),
                "rate": effective_rate,
                "requestedRate": rate,
                "alignment": alignment,
                "alignmentName": PtAlignType.name(alignment),
                "summary": summary_shape,
                "popular": bool(popular),
                "dtwWindow": selected_dtw_window,
                "exactDtw": bool(exact_dtw),
            }
            row.update(_rounded_metric_values(raw_metric_values, round_precision))
            between_group_rows.append(row)
            if collect_raw_outputs:
                raw_metric_outputs.extend(
                    _raw_metric_outputs_from_values(
                        row,
                        raw_metric_values,
                        "between-groups",
                    )
                )

    within_reference_stats_rows = _summary_stats(
        within_reference_rows,
        run_id,
        "human",
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )
    between_group_stats_rows = _summary_stats(
        between_group_rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )
    metadata = {
        "runId": run_id,
        "source": source_name,
        "dataset": dataset_name,
        "variant": _variant_label(variant),
        "classKey": class_key,
        "mode": "direct-distribution-pairs",
        "referenceCount": len(reference_entries),
        "candidateCount": len(candidate_entries),
        "withinReferencePairs": len(within_reference_rows),
        "betweenGroupPairs": len(between_group_rows),
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment,
        "alignmentName": PtAlignType.name(alignment),
        "summary": summary_shape,
        "popular": bool(popular),
        "roundPrecision": round_precision,
        "metricNames": list(metric_names),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }
    return (
        within_reference_rows,
        within_reference_stats_rows,
        between_group_rows,
        between_group_stats_rows,
        raw_metric_outputs,
        metadata,
    )


def _compare_human_baseline_class(
    reference_entries: Sequence[ReportingEntry],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    reference_input: Path,
    rate: int | None,
    alignment: int,
    summary_shape: str | None,
    popular: bool,
    round_precision: int | None,
    metric_names: Sequence[str],
    dtw_window: int | None,
    exact_dtw: bool,
    collect_raw_outputs: bool = True,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    baseline_source_name = "human"
    reference_points = [entry.points for entry in reference_entries]
    effective_rate = sampling_rate_for_sets(reference_points, rate)
    selected_dtw_window = _effective_dtw_window(effective_rate, dtw_window, exact_dtw)
    summary_label = f"{dataset_name}/{class_key}/human-summary"
    reference_gestures = [
        Gesture(entry.points, summary_label, effective_rate)
        for entry in reference_entries
    ]
    reference_summary = SummaryGesture(
        reference_gestures,
        alignment,
        summary_shape,
        popular,
    )

    rows = []
    raw_metric_outputs = []
    for reference_entry, reference_gesture in zip(reference_entries, reference_gestures):
        raw_metric_values = compute_metrics(
            reference_gesture,
            reference_summary,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
        )
        metric_values = _rounded_metric_values(raw_metric_values, round_precision)
        row = {
            "runId": run_id,
            "source": baseline_source_name,
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "sampleKey": Path(reference_entry.key).with_suffix("").as_posix(),
            "sampleFile": reference_entry.path,
            "referenceInput": str(reference_input),
            "mode": "human-summary-baseline",
            "referenceCount": len(reference_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(metric_values)
        rows.append(row)
        if collect_raw_outputs:
            raw_metric_outputs.extend(
                _raw_metric_outputs_from_values(
                    row,
                    raw_metric_values,
                    "human-baseline",
                )
            )

    stats_rows = _baseline_stats(
        rows,
        run_id,
        baseline_source_name,
        dataset_name,
        variant,
        class_key,
        summary_shape,
        round_precision,
    )
    metadata = {
        "runId": run_id,
        "source": source_name,
        "baselineSource": baseline_source_name,
        "dataset": dataset_name,
        "variant": _variant_label(variant),
        "classKey": class_key,
        "mode": "human-summary-baseline",
        "referenceCount": len(reference_entries),
        "baselineRows": len(rows),
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment,
        "alignmentName": PtAlignType.name(alignment),
        "summary": summary_shape,
        "popular": bool(popular),
        "roundPrecision": round_precision,
        "metricNames": list(metric_names),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }
    return rows, stats_rows, raw_metric_outputs, metadata
