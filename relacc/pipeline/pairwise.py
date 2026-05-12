from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import METRIC_NAMES, compute_metrics
from ._common import (
    SUMMARY_SHAPES,
    compute_pair_metrics_from_points,
    effective_dtw_window,
    list_csv_files,
    load_csv_entries,
    normalize_summary_shape,
    pair_key,
    read_points,
    sampling_rate,
    sampling_rate_for_sets,
)


@dataclass(frozen=True)
class PairSpec:
    key: str
    reference_file: str
    candidate_file: str


DIRECT_MODE = "direct"
SUMMARY_MODE = "summary"
COMPARISON_MODES: Tuple[str, str] = (DIRECT_MODE, SUMMARY_MODE)


def _list_csv_files(path: Path) -> Dict[str, Path]:
    return list_csv_files(path)


def _pair_key(relative_csv_path: str) -> str:
    return pair_key(relative_csv_path)


def _normalize_summary_shape(summary_shape: str | None):
    return normalize_summary_shape(summary_shape)


def _normalize_mode(comparison_mode: str | None):
    mode = (comparison_mode or DIRECT_MODE).strip().lower()
    if mode not in COMPARISON_MODES:
        raise ValueError(
            "Invalid comparison mode (%s). Supported values: direct, summary." % mode
        )
    return mode


def _top_level_filename_key(relative_csv_path: str) -> str:
    parts = relative_csv_path.split("/")
    if len(parts) < 2:
        return parts[0]
    return "/".join([parts[0], parts[-1]])


def _filename_key(relative_csv_path: str) -> str:
    return relative_csv_path.split("/")[-1]


def _unique_index(
    keys: Sequence[str],
    key_func: Callable[[str], str],
) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {}
    for key in keys:
        grouped.setdefault(key_func(key), []).append(key)
    return {
        match_key: values[0]
        for match_key, values in grouped.items()
        if len(values) == 1
    }


def _collect_directory_pairs(
    ref_files: Dict[str, Path],
    cand_files: Dict[str, Path],
) -> List[Tuple[str, str]]:
    ref_remaining = set(ref_files.keys())
    cand_remaining = set(cand_files.keys())
    pairs: List[Tuple[str, str]] = []

    exact_keys = sorted(ref_remaining & cand_remaining)
    for key in exact_keys:
        pairs.append((key, key))
    ref_remaining -= set(exact_keys)
    cand_remaining -= set(exact_keys)

    for key_func in [_top_level_filename_key, _filename_key]:
        ref_index = _unique_index(sorted(ref_remaining), key_func)
        cand_index = _unique_index(sorted(cand_remaining), key_func)
        matched_lookup_keys = sorted(set(ref_index.keys()) & set(cand_index.keys()))
        if not matched_lookup_keys:
            continue

        matched_refs = set()
        matched_cands = set()
        for match_key in matched_lookup_keys:
            ref_key = ref_index[match_key]
            cand_key = cand_index[match_key]
            pairs.append((ref_key, cand_key))
            matched_refs.add(ref_key)
            matched_cands.add(cand_key)
        ref_remaining -= matched_refs
        cand_remaining -= matched_cands

    return pairs


def discover_pairs(reference_input: str, candidate_input: str, strict: bool = True):
    ref_path = Path(reference_input)
    cand_path = Path(candidate_input)

    if ref_path.is_file() and cand_path.is_file():
        key = _pair_key(ref_path.name)
        pairs = [PairSpec(key=key, reference_file=str(ref_path), candidate_file=str(cand_path))]
        return pairs, [], []

    if ref_path.is_file() != cand_path.is_file():
        raise ValueError(
            "Both inputs must be files or both must be directories."
        )

    ref_files = _list_csv_files(ref_path)
    cand_files = _list_csv_files(cand_path)

    directory_pairs = _collect_directory_pairs(ref_files, cand_files)
    if not directory_pairs:
        raise ValueError("No matching CSV files found between input directories.")

    matched_ref_keys = {ref_key for ref_key, _ in directory_pairs}
    matched_cand_keys = {cand_key for _, cand_key in directory_pairs}
    missing_in_candidate = sorted(set(ref_files.keys()) - matched_ref_keys)
    missing_in_reference = sorted(set(cand_files.keys()) - matched_cand_keys)

    if strict and (missing_in_candidate or missing_in_reference):
        raise ValueError(
            "Unmatched files found (reference-only: %d, candidate-only: %d). "
            "Use strict=False (or --no-strict in the CLI) to ignore unmatched files."
            % (len(missing_in_candidate), len(missing_in_reference))
        )

    pairs = [
        PairSpec(
            key=_pair_key(ref_key),
            reference_file=str(ref_files[ref_key]),
            candidate_file=str(cand_files[cand_key]),
        )
        for ref_key, cand_key in sorted(directory_pairs)
    ]
    return pairs, missing_in_candidate, missing_in_reference


def _read_points(csv_file: str):
    return read_points(csv_file)


def _sampling_rate_for_sets(point_sets, rate):
    return sampling_rate_for_sets(point_sets, rate)


def _sampling_rate(reference_points, candidate_points, rate):
    return sampling_rate(reference_points, candidate_points, rate)


def compare_pair(
    pair: PairSpec,
    label: str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int = 3,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    reference_points = _read_points(pair.reference_file)
    candidate_points = _read_points(pair.candidate_file)

    pair_label = label or pair.key
    effective_rate = _sampling_rate(reference_points, candidate_points, rate)
    selected_dtw_window = effective_dtw_window(
        effective_rate,
        dtw_window,
        exact_dtw,
    )

    selected_metric_names = tuple(metric_names or METRIC_NAMES)
    metrics = compute_pair_metrics_from_points(
        reference_points,
        candidate_points,
        pair_label,
        effective_rate,
        alignment_type=alignment_type,
        summary_shape=summary_shape,
        popular_shape=popular_shape,
        round_precision=round_precision,
        metric_names=selected_metric_names,
        dtw_window=selected_dtw_window,
        exact_dtw=exact_dtw,
    )

    row = {
        "pairKey": pair.key,
        "label": pair_label,
        "referenceFile": pair.reference_file,
        "candidateFile": pair.candidate_file,
        "mode": DIRECT_MODE,
        "referenceCount": 1,
        "rate": effective_rate,
        "alignment": alignment_type,
        "summary": summary_shape,
        "popular": bool(popular_shape),
        "dtwWindow": selected_dtw_window,
    }
    row.update(metrics)
    return row


def compare_against_reference_summary(
    reference_input: str,
    candidate_input: str,
    label: str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int = 3,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    reference_root = Path(reference_input)
    candidate_root = Path(candidate_input)

    reference_files = _list_csv_files(reference_root)
    if len(reference_files) == 0:
        raise ValueError("No reference CSV files found.")

    candidate_files = _list_csv_files(candidate_root)
    if len(candidate_files) == 0:
        raise ValueError("No candidate CSV files found.")

    reference_entries = load_csv_entries(reference_root)
    candidate_entries = load_csv_entries(candidate_root)

    # In summary mode the reference summary should be independent of candidate data.
    reference_points = [entry[2] for entry in reference_entries]
    effective_rate = _sampling_rate_for_sets(reference_points, rate)
    selected_dtw_window = effective_dtw_window(
        effective_rate,
        dtw_window,
        exact_dtw,
    )

    summary_label = label or "reference-summary"
    reference_gestures = [
        Gesture(points, summary_label, effective_rate)
        for _, _, points in reference_entries
    ]
    reference_summary = SummaryGesture(
        reference_gestures,
        alignment_type,
        summary_shape,
        popular_shape,
    )
    selected_metric_names = tuple(metric_names or METRIC_NAMES)

    results = []
    for candidate_key, candidate_path, candidate_points in candidate_entries:
        pair_key = _pair_key(candidate_key)
        pair_label = label or pair_key
        candidate = Gesture(candidate_points, pair_label, effective_rate)
        metrics = compute_metrics(
            candidate,
            reference_summary,
            round_precision=round_precision,
            metric_names=selected_metric_names,
            dtw_window=selected_dtw_window,
        )

        row = {
            "pairKey": pair_key,
            "label": pair_label,
            "referenceFile": str(reference_root),
            "candidateFile": candidate_path,
            "mode": SUMMARY_MODE,
            "referenceCount": len(reference_entries),
            "rate": effective_rate,
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "dtwWindow": selected_dtw_window,
        }
        row.update(metrics)
        results.append(row)

    return results, [], [], len(reference_entries)


def run_pairwise_comparison(
    reference_input: str,
    candidate_input: str,
    label: str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    strict: bool = True,
    round_precision: int = 3,
    comparison_mode: str = DIRECT_MODE,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    summary_shape = _normalize_summary_shape(summary_shape)
    mode = _normalize_mode(comparison_mode)
    selected_metric_names = tuple(metric_names or METRIC_NAMES)

    if mode == DIRECT_MODE:
        pairs, missing_in_candidate, missing_in_reference = discover_pairs(
            reference_input,
            candidate_input,
            strict=strict,
        )
        results = [
            compare_pair(
                pair,
                label=label,
                rate=rate,
                alignment_type=alignment_type,
                summary_shape=summary_shape,
                popular_shape=popular_shape,
                round_precision=round_precision,
                metric_names=selected_metric_names,
                dtw_window=dtw_window,
                exact_dtw=exact_dtw,
            )
            for pair in pairs
        ]
        reference_count = len(results)
    else:
        results, missing_in_candidate, missing_in_reference, reference_count = (
            compare_against_reference_summary(
                reference_input,
                candidate_input,
                label=label,
                rate=rate,
                alignment_type=alignment_type,
                summary_shape=summary_shape,
                popular_shape=popular_shape,
                round_precision=round_precision,
                metric_names=selected_metric_names,
                dtw_window=dtw_window,
                exact_dtw=exact_dtw,
            )
        )

    effective_windows = {row.get("dtwWindow") for row in results}
    metadata_dtw_window = effective_windows.pop() if len(effective_windows) == 1 else None

    return {
        "metadata": {
            "comparisonMode": mode,
            "pairCount": len(results),
            "referenceCount": reference_count,
            "missingInCandidate": missing_in_candidate,
            "missingInReference": missing_in_reference,
            "strict": bool(strict),
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "rate": rate,
            "label": label,
            "roundPrecision": round_precision,
            "metricNames": list(selected_metric_names),
            "dtwWindow": metadata_dtw_window,
            "exactDtw": bool(exact_dtw),
        },
        "pairs": results,
    }


def format_pair_rows_csv(
    rows: Sequence[Dict[str, object]],
    metric_names: Sequence[str] | None = None,
) -> str:
    selected_metric_names = tuple(metric_names or METRIC_NAMES)
    columns: List[str] = [
        "pairKey",
        "label",
        "referenceFile",
        "candidateFile",
        "mode",
        "referenceCount",
        "rate",
        "alignment",
        "summary",
        "popular",
        "dtwWindow",
        *selected_metric_names,
    ]
    lines = [",".join(columns)]

    for row in rows:
        fields: List[str] = []
        for column in columns:
            value = row.get(column, "")
            if value is None:
                value = ""
            text = str(value)
            text = text.replace('"', '""')
            if "," in text or '"' in text:
                text = '"%s"' % text
            fields.append(text)
        lines.append(",".join(fields))

    return "\n".join(lines)
