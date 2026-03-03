from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from relacc.geom.pointset import PointSet
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import METRIC_NAMES, compute_metrics
from relacc.utils.csv import CSVUtil
from relacc.utils.math import MathUtil


@dataclass(frozen=True)
class PairSpec:
    key: str
    reference_file: str
    candidate_file: str


def _list_csv_files(path: Path) -> Dict[str, Path]:
    if not path.exists():
        raise FileNotFoundError("Path does not exist: %s" % path)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("Expected a .csv file: %s" % path)
        return {path.name: path}

    files: Dict[str, Path] = {}
    for csv_path in sorted(path.rglob("*.csv")):
        rel = csv_path.relative_to(path).as_posix()
        files[rel] = csv_path
    return files


def _pair_key(relative_csv_path: str) -> str:
    rel_path = Path(relative_csv_path)
    return rel_path.with_suffix("").as_posix()


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

    common_keys = sorted(set(ref_files.keys()) & set(cand_files.keys()))
    if not common_keys:
        raise ValueError("No matching CSV files found between input directories.")

    missing_in_candidate = sorted(set(ref_files.keys()) - set(cand_files.keys()))
    missing_in_reference = sorted(set(cand_files.keys()) - set(ref_files.keys()))

    if strict and (missing_in_candidate or missing_in_reference):
        raise ValueError(
            "Unmatched files found (reference-only: %d, candidate-only: %d). "
            "Use strict=False to ignore unmatched files."
            % (len(missing_in_candidate), len(missing_in_reference))
        )

    pairs = [
        PairSpec(
            key=_pair_key(key),
            reference_file=str(ref_files[key]),
            candidate_file=str(cand_files[key]),
        )
        for key in common_keys
    ]
    return pairs, missing_in_candidate, missing_in_reference


def _read_points(csv_file: str):
    state = {}

    def _done(points):
        state["points"] = points

    CSVUtil.readGesture(csv_file, _done)
    points = state.get("points")
    if not points:
        raise ValueError("No points parsed from CSV file: %s" % csv_file)
    return points


def _sampling_rate(reference_points, candidate_points, rate):
    if rate is not None:
        parsed_rate = int(rate)
        if parsed_rate < 1:
            raise ValueError("Sampling rate must be >= 1.")
        return parsed_rate

    max_strokes = max(
        PointSet.countStrokes(reference_points),
        PointSet.countStrokes(candidate_points),
    )
    return max(24, MathUtil.factorial(max_strokes))


def _compute_metrics(candidate: Gesture, summary: SummaryGesture, round_precision: int):
    return compute_metrics(candidate, summary, round_precision=round_precision)


def compare_pair(
    pair: PairSpec,
    label: str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int = 3,
):
    reference_points = _read_points(pair.reference_file)
    candidate_points = _read_points(pair.candidate_file)

    pair_label = label or pair.key
    effective_rate = _sampling_rate(reference_points, candidate_points, rate)

    reference = Gesture(reference_points, pair_label, effective_rate)
    candidate = Gesture(candidate_points, pair_label, effective_rate)
    summary = SummaryGesture([reference], alignment_type, summary_shape, popular_shape)

    metrics = _compute_metrics(candidate, summary, round_precision)

    row = {
        "pairKey": pair.key,
        "label": pair_label,
        "referenceFile": pair.reference_file,
        "candidateFile": pair.candidate_file,
        "rate": effective_rate,
        "alignment": alignment_type,
        "summary": summary_shape,
        "popular": bool(popular_shape),
    }
    row.update(metrics)
    return row


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
):
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
        )
        for pair in pairs
    ]

    return {
        "metadata": {
            "pairCount": len(results),
            "missingInCandidate": missing_in_candidate,
            "missingInReference": missing_in_reference,
            "strict": bool(strict),
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "rate": rate,
            "label": label,
            "roundPrecision": round_precision,
        },
        "pairs": results,
    }


def format_pair_rows_csv(rows: Sequence[Dict[str, object]]) -> str:
    columns: List[str] = [
        "pairKey",
        "label",
        "referenceFile",
        "candidateFile",
        "rate",
        "alignment",
        "summary",
        "popular",
        *METRIC_NAMES,
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
