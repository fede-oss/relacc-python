import json
from pathlib import Path

import pytest

from relacc.gesture_input import read_csv_points, read_gesture, read_gesture_points


def _write_csv(path: Path, rows):
    path.write_text("\n".join(rows), encoding="utf-8")


def _sample_rows():
    return [
        "stroke_id x y time is_writing",
        "0 10 20 0 1",
        "0 12 22 10 1",
    ]


def test_read_gesture_points_dispatches_csv_and_json(tmp_path):
    csv_file = tmp_path / "gesture.csv"
    json_file = tmp_path / "gesture.json"
    _write_csv(csv_file, _sample_rows())
    json_file.write_text(
        json.dumps({"strokes": [[[30, 40, 0], [31, 41, 5]]]}),
        encoding="utf-8",
    )

    csv_points = read_gesture_points(str(csv_file))
    json_points = read_gesture_points(str(json_file))

    assert [(point.X, point.Y, point.T, point.StrokeID) for point in csv_points] == [
        (10.0, 20.0, 0.0, 0),
        (12.0, 22.0, 10.0, 0),
    ]
    assert [(point.X, point.Y, point.T, point.StrokeID) for point in json_points] == [
        (30.0, 40.0, 0.0, 0),
        (31.0, 41.0, 5.0, 0),
    ]


def test_read_gesture_callback_adapter_preserves_empty_points(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")
    out = {}

    read_gesture(str(csv_file), lambda points: out.setdefault("points", points))

    assert out["points"] == []


def test_read_csv_points_empty_policy_can_raise_or_return_empty(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No points parsed from CSV file"):
        read_csv_points(str(csv_file))

    assert read_csv_points(str(csv_file), require_points=False) == []


def test_read_gesture_points_rejects_unknown_format(tmp_path):
    unknown = tmp_path / "gesture.txt"
    unknown.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid input file format"):
        read_gesture_points(str(unknown))
