from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

from relacc.dtw import DEFAULT_EXACT_RATE_THRESHOLD, recommended_window
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import compute_metrics
from relacc.pipeline import pair_evidence as PairEvidence
from relacc.pipeline._common import sampling_rate_for_sets, summary_sampling_rate
from relacc.pipeline.reporting import ReportingEntry
from relacc.pipeline.report_schema import (
    statistical_contract_fields as _statistical_contract_fields,
)
from relacc.pipeline.report_stats import _summary_stats, _variant_label


def _rounded_metric_values(values: Dict[str, float], round_precision: int | None):
    return PairEvidence.rounded_metric_values(values, round_precision)


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

    return PairEvidence.metric_value_rows(base, raw_values)


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
    pair_options = PairEvidence.PairMetricOptions(
        label=class_key,
        effective_rate=effective_rate,
        alignment_type=alignment,
        summary_shape=summary_shape,
        popular_shape=popular,
        metric_names=metric_names,
        dtw_window=selected_dtw_window,
        exact_dtw=exact_dtw,
    )
    for left_entry, right_entry in combinations(within_comparison_entries, 2):
        left_endpoint = PairEvidence.endpoint_for(left_entry)
        right_endpoint = PairEvidence.endpoint_for(right_entry)
        evidence = PairEvidence.compute_bidirectional_pair_evidence(
            left_endpoint,
            right_endpoint,
            pair_options,
        )
        raw_metric_values = evidence.values
        row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "pairKey": PairEvidence.joined_pair_key(left_endpoint, right_endpoint),
            "candidateFile": PairEvidence.joined_pair_path(
                left_endpoint,
                right_endpoint,
            ),
            "referenceInput": str(reference_input),
            "mode": PairEvidence.WITHIN_COMPARISON_RECORD_SET,
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
                    PairEvidence.WITHIN_COMPARISON_RECORD_SET,
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
    pair_options = PairEvidence.PairMetricOptions(
        label=class_key,
        effective_rate=effective_rate,
        alignment_type=alignment,
        summary_shape=summary_shape,
        popular_shape=popular,
        metric_names=metric_names,
        dtw_window=selected_dtw_window,
        exact_dtw=exact_dtw,
    )

    for left_entry, right_entry in combinations(reference_entries, 2):
        left_endpoint = PairEvidence.endpoint_for(left_entry)
        right_endpoint = PairEvidence.endpoint_for(right_entry)
        evidence = PairEvidence.compute_bidirectional_pair_evidence(
            left_endpoint,
            right_endpoint,
            pair_options,
        )
        raw_metric_values = evidence.values
        row = {
            "runId": run_id,
            "source": "human",
            "dataset": dataset_name,
            "variant": _variant_label(variant),
            "classKey": class_key,
            "pairKey": PairEvidence.joined_pair_key(left_endpoint, right_endpoint),
            "candidateFile": PairEvidence.joined_pair_path(
                left_endpoint,
                right_endpoint,
            ),
            "referenceInput": str(reference_input),
            "mode": PairEvidence.WITHIN_REFERENCE_RECORD_SET,
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
                    PairEvidence.WITHIN_REFERENCE_RECORD_SET,
                )
            )

    for reference_entry in reference_entries:
        for candidate_entry in candidate_entries:
            reference_endpoint = PairEvidence.endpoint_for(reference_entry)
            candidate_endpoint = PairEvidence.endpoint_for(candidate_entry)
            evidence = PairEvidence.compute_directional_pair_evidence(
                reference_endpoint,
                candidate_endpoint,
                pair_options,
            )
            raw_metric_values = evidence.values
            row = {
                "runId": run_id,
                "source": source_name,
                "dataset": dataset_name,
                "variant": _variant_label(variant),
                "classKey": class_key,
                "pairKey": PairEvidence.joined_pair_key(
                    reference_endpoint,
                    candidate_endpoint,
                ),
                "candidateFile": candidate_endpoint.path,
                "referenceInput": reference_endpoint.path,
                "mode": PairEvidence.BETWEEN_GROUPS_RECORD_SET,
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
                        PairEvidence.BETWEEN_GROUPS_RECORD_SET,
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
