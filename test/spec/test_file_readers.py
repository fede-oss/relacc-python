import json
import math

import pytest

from relacc.utils.args import Args
from relacc.utils.csv import CSVUtil
from relacc.utils.json import JSONUtil
from relacc.utils.math import MathUtil


def _read_csv(path):
    parsed = {}
    CSVUtil.readGesture(path, lambda points: parsed.setdefault("points", points))
    return parsed["points"]


def test_csv_reader_parses_a_generated_fixture(write_gesture_csv):
    path = write_gesture_csv(
        "samples/s01-zigzag-t01.csv",
        [
            (0, 0, 0, 0),
            (0, 10, 0, 10),
            (1, 10, 5, 20),
        ],
    )

    points = _read_csv(path)

    assert [(pt.X, pt.Y, pt.T, pt.StrokeID) for pt in points] == [
        (0.0, 0.0, 0.0, 0),
        (10.0, 0.0, 10.0, 0),
        (10.0, 5.0, 20.0, 1),
    ]


def test_csv_reader_skips_duplicate_timestamps_and_keeps_negative_time_as_nan(write_gesture_csv):
    path = write_gesture_csv(
        "samples/s02-zigzag-t02.csv",
        [
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 2, 2, 5),
            (0, 3, 3, -1),
        ],
    )

    points = _read_csv(path)

    assert [(pt.X, pt.Y, pt.T) for pt in points[:2]] == [(0.0, 0.0, 0.0), (2.0, 2.0, 5.0)]
    assert math.isnan(points[2].T)


def test_json_reader_uses_stroke_index_as_stroke_id(tmp_path):
    path = tmp_path / "gesture.json"
    path.write_text(
        json.dumps({"strokes": [[[0, 0, 0], [1, 0, 10]], [[1, 1, 20]]]}),
        encoding="utf-8",
    )

    parsed = {}
    JSONUtil.readGesture(path, lambda points: parsed.setdefault("points", points))

    assert [(pt.X, pt.Y, pt.T, pt.StrokeID) for pt in parsed["points"]] == [
        (0, 0, 0, 0),
        (1, 0, 10, 0),
        (1, 1, 20, 1),
    ]


def test_small_utils_have_concrete_outputs():
    args = Args({"rate": "32", "label": None})

    assert args.get("rate", castFn=int) == 32
    assert args.get("label", "fallback") == "fallback"
    assert MathUtil.roundTo(2.345, 2) == 2.35
    assert MathUtil.factorial(5) == 120

    with pytest.raises(ValueError, match="Invalid CSV header"):
        CSVUtil.readGesture(__file__, lambda points: None)
