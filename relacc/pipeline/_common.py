from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from relacc.dtw import (
    DEFAULT_EXACT_RATE_THRESHOLD,
    recommended_window as recommended_dtw_window,
)
from relacc.geom.pointset import PointSet
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.gesture_input import read_csv_points, read_gesture_points
from relacc.metrics import compute_metrics


SUMMARY_SHAPES = {"centroid", "medoid", "kcentroid", "kmedoid"}
MIN_AUTO_SAMPLING_RATE = 24
TARGET_POINTS_PER_STROKE = 8
MAX_AUTO_SAMPLING_RATE = 256


def list_csv_files(path: Path) -> Dict[str, Path]:
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


def pair_key(relative_csv_path: str) -> str:
    rel_path = Path(relative_csv_path)
    return rel_path.with_suffix("").as_posix()


def infer_label_from_filename(relative_path: str, label_kind: str = "gesture") -> str:
    filename = relative_path.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 3 or not parts[1]:
        raise ValueError(
            "Cannot derive %s label from filename (%s). "
            "Expected at least 3 '-' separated parts."
            % (label_kind, relative_path)
        )
    return parts[1]


def normalize_summary_shape(summary_shape: str | None):
    if summary_shape is None:
        return None

    normalized_summary = summary_shape.strip().lower()
    if normalized_summary not in SUMMARY_SHAPES:
        raise ValueError(
            "Invalid summary shape (%s). Supported values: centroid, medoid, kcentroid, kmedoid."
            % summary_shape
        )
    return normalized_summary


def read_points(csv_file: str):
    return read_csv_points(csv_file)


def load_csv_entries(input_path: str | Path) -> List[Tuple[str, str, list]]:
    root = Path(input_path)
    return [
        (key, str(path), read_points(str(path)))
        for key, path in sorted(list_csv_files(root).items())
    ]


def _max_stroke_count(point_sets):
    max_strokes = 1
    for points in point_sets:
        stroke_count = PointSet.countStrokes(points)
        if stroke_count > max_strokes:
            max_strokes = stroke_count
    return max_strokes


def sampling_rate_for_sets(point_sets, rate):
    max_strokes = _max_stroke_count(point_sets)

    if rate is not None:
        parsed_rate = int(rate)
        if parsed_rate < 1:
            raise ValueError("Sampling rate must be >= 1.")
        if parsed_rate < max_strokes:
            raise ValueError(
                "Sampling rate must be at least the maximum stroke count (%s)."
                % max_strokes
            )
        return parsed_rate

    if max_strokes > MAX_AUTO_SAMPLING_RATE:
        raise ValueError(
            "Automatic sampling does not support more than 256 stroke runs; "
            "provide an explicit sampling rate."
        )
    return min(
        MAX_AUTO_SAMPLING_RATE,
        max(MIN_AUTO_SAMPLING_RATE, TARGET_POINTS_PER_STROKE * max_strokes),
    )


def summary_sampling_rate(reference_points, candidate_points, rate):
    reference_points = list(reference_points)
    candidate_points = list(candidate_points)

    if rate is not None:
        return sampling_rate_for_sets(reference_points + candidate_points, rate)

    reference_rate = sampling_rate_for_sets(reference_points, None)
    candidate_max_strokes = _max_stroke_count(candidate_points)
    if candidate_max_strokes > MAX_AUTO_SAMPLING_RATE:
        raise ValueError(
            "Automatic sampling does not support more than 256 stroke runs; "
            "provide an explicit sampling rate."
        )
    return max(reference_rate, candidate_max_strokes)


def sampling_rate(reference_points, candidate_points, rate):
    return sampling_rate_for_sets([reference_points, candidate_points], rate)


def effective_dtw_window(
    effective_rate: int,
    requested_window: int | None = None,
    exact_dtw: bool = False,
) -> int | None:
    if exact_dtw:
        return None
    if requested_window is not None:
        return requested_window
    if effective_rate <= DEFAULT_EXACT_RATE_THRESHOLD:
        return None
    return recommended_dtw_window(effective_rate)


def output_format(output: str | None, requested_format: str | None, default: str = "json"):
    if output:
        ext = Path(output).suffix[1:].lower()
        if ext:
            return ext
    return (requested_format or default).lower()


def csv_escape(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace('"', '""')
    if "," in text or '"' in text or "\n" in text:
        return '"%s"' % text
    return text


def format_csv_rows(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(csv_escape(row.get(column, "")) for column in columns))
    return "\n".join(lines)


def jsonl_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonl_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonl_safe(item) for item in value]
    return value


def format_jsonl_rows(rows: Sequence[Dict[str, object]]) -> str:
    return "\n".join(
        json.dumps(jsonl_safe(row), sort_keys=True, allow_nan=False)
        for row in rows
    )


def default_raw_output_path(output: str | None, suffix: str = "raw-metrics.jsonl") -> str | None:
    if output is None:
        return None

    raw_suffix = "." + suffix.lstrip(".")
    path = Path(output)
    if path.suffix:
        return str(path.with_suffix(path.suffix + raw_suffix))
    return str(path.with_suffix(raw_suffix))


def write_jsonl_rows(path: str | Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = format_jsonl_rows(rows)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")


def compute_pair_metrics_from_points(
    reference_points,
    candidate_points,
    label: str,
    effective_rate: int,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    round_precision: int | None = None,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    selected_dtw_window = effective_dtw_window(effective_rate, dtw_window, exact_dtw)
    reference = Gesture(reference_points, label, effective_rate)
    candidate = Gesture(candidate_points, label, effective_rate)
    summary = SummaryGesture([reference], alignment_type, summary_shape, popular_shape)
    return compute_metrics(
        candidate,
        summary,
        round_precision=round_precision,
        metric_names=metric_names,
        dtw_window=selected_dtw_window,
    )
