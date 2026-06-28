from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import METRIC_NAMES

from . import pair_evidence as PairEvidence
from ._common import (
    csv_escape,
    effective_dtw_window,
    normalize_summary_shape,
    sampling_rate_for_sets,
)
from .distribution import GROUP_BY_FILENAME_LABEL, _normalize_group_by
from .reporting import (
    CANDIDATE_SOURCE,
    CLASS_SCHEME_AUTO,
    DEFAULT_CANDIDATE_SOURCE_NAME,
    DEFAULT_REFERENCE_SOURCE_NAME,
    DEFAULT_SAMPLE_LIMIT,
    REFERENCE_SOURCE,
    REPORTING_MODE,
    ReportingEntry,
    ReportingSampleGroup,
    _normalize_class_scheme,
    _validate_sample_limit,
    build_sample_manifest,
    discover_reporting_sample_groups,
)


BASELINE_PAIR_TYPE = "baseline"
CANDIDATE_PAIR_TYPE = "candidate"
RAW_BASELINE_PAIRS_FILENAME = "raw_baseline_pairs.csv"
RAW_CANDIDATE_PAIRS_FILENAME = "raw_candidate_pairs.csv"
RAW_COMPARISON_COLUMNS = (
    "datasetKey",
    "classKey",
    "pairType",
    "direction",
    "pairKey",
    "metric",
    "referenceSource",
    "referenceSourceRole",
    "referenceKey",
    "referenceFile",
    "candidateSource",
    "candidateSourceRole",
    "candidateKey",
    "candidateFile",
    "value",
    "rate",
    "alignment",
    "alignmentName",
    "summary",
    "popular",
    "dtwWindow",
    "exactDtw",
)


def _raw_pair_key(
    dataset_key: str,
    class_key: str,
    pair_type: str,
    reference_key: str,
    candidate_key: str,
    direction: str,
) -> str:
    return "%s:%s:%s:%s:%s:%s" % (
        dataset_key,
        class_key,
        pair_type,
        reference_key,
        candidate_key,
        direction,
    )


def _rounded_metric_values(values: Dict[str, float], round_precision: int | None):
    return PairEvidence.rounded_metric_values(values, round_precision)


def _raw_metric_rows(
    group: ReportingSampleGroup,
    pair_type: str,
    direction: str,
    reference_entry: ReportingEntry,
    candidate_entry: ReportingEntry,
    values: Dict[str, float],
    effective_rate: int,
    alignment_type: int,
    summary_shape: str | None,
    popular_shape: bool,
    selected_dtw_window: int | None,
    exact_dtw: bool,
    reference_source_name: str,
    reference_source_role: str,
    candidate_source_name: str,
    candidate_source_role: str,
) -> List[Dict[str, object]]:
    pair_key = _raw_pair_key(
        group.dataset_key,
        group.class_key,
        pair_type,
        reference_entry.key,
        candidate_entry.key,
        direction,
    )
    return [
        {
            "datasetKey": group.dataset_key,
            "classKey": group.class_key,
            "pairType": pair_type,
            "direction": direction,
            "pairKey": pair_key,
            "metric": metric_name,
            "referenceSource": reference_source_name,
            "referenceSourceRole": reference_source_role,
            "referenceKey": reference_entry.key,
            "referenceFile": reference_entry.path,
            "candidateSource": candidate_source_name,
            "candidateSourceRole": candidate_source_role,
            "candidateKey": candidate_entry.key,
            "candidateFile": candidate_entry.path,
            "value": value,
            "rate": effective_rate,
            "alignment": alignment_type,
            "alignmentName": PtAlignType.name(alignment_type),
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        for metric_name, value in values.items()
    ]


def build_raw_comparison_tables(
    sample_groups: Sequence[ReportingSampleGroup],
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int | None = None,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
    reference_source_name: str = DEFAULT_REFERENCE_SOURCE_NAME,
    candidate_source_name: str = DEFAULT_CANDIDATE_SOURCE_NAME,
) -> Dict[str, object]:
    """Build raw metric rows for report distribution evidence.

    This is computationally expensive: per dataset/class it performs
    ``N_reference choose 2 * 2`` baseline comparisons plus
    ``N_reference * N_candidate`` candidate comparisons, then emits one row per
    metric for every comparison. Keep ``sample_limit`` bounded for ordinary
    reports, and use full mode only when a large raw dump is intentional.
    """

    alignment_type = PtAlignType.normalize(alignment_type)
    normalized_summary = normalize_summary_shape(summary_shape)
    selected_metric_names = tuple(metric_names or METRIC_NAMES)
    baseline_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    effective_dtw_windows = set()

    for group in sample_groups:
        reference_points = [entry.points for entry in group.reference_entries]
        if len(reference_points) == 0:
            continue

        candidate_points = [entry.points for entry in group.candidate_entries]
        effective_rate = sampling_rate_for_sets(reference_points + candidate_points, rate)
        selected_dtw_window = effective_dtw_window(
            effective_rate,
            dtw_window,
            exact_dtw,
        )
        effective_dtw_windows.add(selected_dtw_window)
        pair_options = PairEvidence.PairMetricOptions(
            label=group.class_key,
            effective_rate=effective_rate,
            alignment_type=alignment_type,
            summary_shape=normalized_summary,
            popular_shape=popular_shape,
            metric_names=selected_metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )

        for left_entry, right_entry in combinations(group.reference_entries, 2):
            for evidence in PairEvidence.directed_pair_evidences(
                PairEvidence.endpoint_for(left_entry),
                PairEvidence.endpoint_for(right_entry),
                pair_options,
            ):
                baseline_rows.extend(
                    _raw_metric_rows(
                        group,
                        BASELINE_PAIR_TYPE,
                        evidence.direction,
                        evidence.left,
                        evidence.right,
                        _rounded_metric_values(evidence.values, round_precision),
                        effective_rate,
                        alignment_type,
                        normalized_summary,
                        popular_shape,
                        selected_dtw_window,
                        exact_dtw,
                        reference_source_name,
                        REFERENCE_SOURCE,
                        reference_source_name,
                        REFERENCE_SOURCE,
                    )
                )

        for reference_entry in group.reference_entries:
            for candidate_entry in group.candidate_entries:
                evidence = PairEvidence.compute_directional_pair_evidence(
                    PairEvidence.endpoint_for(reference_entry),
                    PairEvidence.endpoint_for(candidate_entry),
                    pair_options,
                )
                candidate_rows.extend(
                    _raw_metric_rows(
                        group,
                        CANDIDATE_PAIR_TYPE,
                        evidence.direction,
                        evidence.left,
                        evidence.right,
                        _rounded_metric_values(evidence.values, round_precision),
                        effective_rate,
                        alignment_type,
                        normalized_summary,
                        popular_shape,
                        selected_dtw_window,
                        exact_dtw,
                        reference_source_name,
                        REFERENCE_SOURCE,
                        candidate_source_name,
                        CANDIDATE_SOURCE,
                    )
                )

    return {
        "metadata": {
            "reportingMode": REPORTING_MODE,
            "rawComparisonMode": "metric-pairs",
            "groupCount": len(sample_groups),
            "baselineRowCount": len(baseline_rows),
            "candidateRowCount": len(candidate_rows),
            "rate": rate,
            "alignment": alignment_type,
            "alignmentName": PtAlignType.name(alignment_type),
            "summary": normalized_summary,
            "popular": bool(popular_shape),
            "roundPrecision": round_precision,
            "metricNames": list(selected_metric_names),
            "dtwWindow": (
                next(iter(effective_dtw_windows))
                if len(effective_dtw_windows) == 1
                else None
            ),
            "exactDtw": bool(exact_dtw),
            "sourceNames": {
                REFERENCE_SOURCE: reference_source_name,
                CANDIDATE_SOURCE: candidate_source_name,
            },
        },
        "rawBaselinePairs": baseline_rows,
        "rawCandidatePairs": candidate_rows,
    }


def format_raw_comparison_rows_csv(rows: Sequence[Dict[str, object]]) -> str:
    lines = [",".join(RAW_COMPARISON_COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(csv_escape(row.get(column, "")) for column in RAW_COMPARISON_COLUMNS)
        )
    return "\n".join(lines)


def write_raw_comparison_exports(
    output_dir: str | Path,
    raw_tables: Dict[str, object],
) -> Dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = target_dir / RAW_BASELINE_PAIRS_FILENAME
    candidate_path = target_dir / RAW_CANDIDATE_PAIRS_FILENAME
    baseline_path.write_text(
        format_raw_comparison_rows_csv(raw_tables["rawBaselinePairs"]),
        encoding="utf-8",
    )
    candidate_path.write_text(
        format_raw_comparison_rows_csv(raw_tables["rawCandidatePairs"]),
        encoding="utf-8",
    )
    return {
        "rawBaselinePairs": str(baseline_path),
        "rawCandidatePairs": str(candidate_path),
    }


def export_raw_comparison_tables(
    reference_input: str,
    candidate_input: str,
    output_dir: str | Path | None = None,
    group_by: str = GROUP_BY_FILENAME_LABEL,
    class_scheme: str = CLASS_SCHEME_AUTO,
    sample_limit: int | None = DEFAULT_SAMPLE_LIMIT,
    random_seed: int | str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int | None = None,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
    reference_source_name: str = DEFAULT_REFERENCE_SOURCE_NAME,
    candidate_source_name: str = DEFAULT_CANDIDATE_SOURCE_NAME,
) -> Dict[str, object]:
    """Export raw comparison values for the selected report samples.

    Full mode (``sample_limit=None``) can be very expensive on real datasets,
    because all pairwise distribution evidence is materialized before writing
    CSV files. Prefer bounded sample limits unless you explicitly need a full
    raw dump for offline analysis.
    """

    if dtw_window is not None and exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")

    selected_limit = None if sample_limit is None else _validate_sample_limit(sample_limit)
    sample_groups = discover_reporting_sample_groups(
        reference_input,
        candidate_input,
        group_by=group_by,
        class_scheme=class_scheme,
        sample_limit=selected_limit,
        random_seed=random_seed,
    )
    raw_tables = build_raw_comparison_tables(
        sample_groups,
        rate=rate,
        alignment_type=alignment_type,
        summary_shape=summary_shape,
        popular_shape=popular_shape,
        round_precision=round_precision,
        metric_names=metric_names,
        dtw_window=dtw_window,
        exact_dtw=exact_dtw,
        reference_source_name=reference_source_name,
        candidate_source_name=candidate_source_name,
    )
    raw_tables["metadata"] = {
        **raw_tables["metadata"],
        "groupBy": _normalize_group_by(group_by),
        "classScheme": _normalize_class_scheme(class_scheme),
        "sampleLimit": sample_limit,
        "effectiveSampleLimit": selected_limit,
        "samplingMode": "seeded-random" if random_seed is not None else "stable",
        "randomSeed": random_seed,
        "sampleManifest": build_sample_manifest(
            sample_groups,
            reference_source_name=reference_source_name,
            candidate_source_name=candidate_source_name,
        ),
    }

    if output_dir is not None:
        raw_tables["files"] = write_raw_comparison_exports(output_dir, raw_tables)

    return raw_tables


__all__ = [
    "BASELINE_PAIR_TYPE",
    "CANDIDATE_PAIR_TYPE",
    "RAW_BASELINE_PAIRS_FILENAME",
    "RAW_CANDIDATE_PAIRS_FILENAME",
    "RAW_COMPARISON_COLUMNS",
    "build_raw_comparison_tables",
    "export_raw_comparison_tables",
    "format_raw_comparison_rows_csv",
    "write_raw_comparison_exports",
]
