from __future__ import annotations

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
from relacc.metrics import compute_metrics
from relacc.utils.csv import CSVUtil
from relacc.utils.math import MathUtil


SUMMARY_SHAPES = {"centroid", "medoid", "kcentroid", "kmedoid"}


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
    state = {}

    def _done(points):
        state["points"] = points

    CSVUtil.readGesture(csv_file, _done)
    points = state.get("points")
    if not points:
        raise ValueError("No points parsed from CSV file: %s" % csv_file)
    return points


def load_csv_entries(input_path: str | Path) -> List[Tuple[str, str, list]]:
    root = Path(input_path)
    return [
        (key, str(path), read_points(str(path)))
        for key, path in sorted(list_csv_files(root).items())
    ]


def sampling_rate_for_sets(point_sets, rate):
    if rate is not None:
        parsed_rate = int(rate)
        if parsed_rate < 1:
            raise ValueError("Sampling rate must be >= 1.")
        return parsed_rate

    max_strokes = 1
    for points in point_sets:
        stroke_count = PointSet.countStrokes(points)
        if stroke_count > max_strokes:
            max_strokes = stroke_count
    return max(24, MathUtil.factorial(max_strokes))


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
