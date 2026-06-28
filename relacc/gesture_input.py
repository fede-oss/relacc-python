from __future__ import annotations

from pathlib import Path
from typing import Callable

from relacc.geom.point import Point
from relacc.utils.csv import CSVUtil
from relacc.utils.json import JSONUtil


GestureCallback = Callable[[list[Point]], None]


def read_gesture(file_path: str, callback: GestureCallback) -> None:
    """Dispatch a gesture file to the legacy callback reader for its format."""
    _reader_for(file_path)(file_path, callback)


def read_gesture_points(file_path: str, *, require_points: bool = True) -> list[Point]:
    points = _read_points(file_path, read_gesture)
    if require_points and not points:
        raise ValueError("No points parsed from gesture file: %s" % file_path)
    return points


def read_csv_points(csv_file: str, *, require_points: bool = True) -> list[Point]:
    points = _read_points(csv_file, CSVUtil.readGesture)
    if require_points and not points:
        raise ValueError("No points parsed from CSV file: %s" % csv_file)
    return points


def _read_points(file_path: str, reader: Callable[[str, GestureCallback], None]) -> list[Point]:
    state: dict[str, list[Point]] = {}

    def _done(points: list[Point]) -> None:
        state["points"] = points

    reader(file_path, _done)
    return state.get("points") or []


def _reader_for(file_path: str) -> Callable[[str, GestureCallback], None]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return CSVUtil.readGesture
    if suffix == ".json":
        return JSONUtil.readGesture
    raise ValueError(
        "Invalid input file format (%s). Supported formats: json, csv." % suffix
    )
