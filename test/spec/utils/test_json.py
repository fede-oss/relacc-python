import json
import math
from pathlib import Path

from relacc.utils.json import JSONUtil


def test_json_parse_retains_equal_timestamps_in_order_across_strokes(tmp_path):
    tmp_file = Path(tmp_path) / "gesture.json"
    payload = {
        "strokes": [
            [[1, 2, 0], [3, 4, 0], [5, 6, 5], [7, 8, 5]],
            [[9, 10, 5], [11, 12, 10]],
        ]
    }
    tmp_file.write_text(json.dumps(payload), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    JSONUtil.readGesture(str(tmp_file), cb)
    points = out["points"]

    assert [(point.X, point.Y, point.T, point.StrokeID) for point in points] == [
        (1.0, 2.0, 0.0, 0),
        (3.0, 4.0, 0.0, 0),
        (5.0, 6.0, 5.0, 0),
        (7.0, 8.0, 5.0, 0),
        (9.0, 10.0, 5.0, 1),
        (11.0, 12.0, 10.0, 1),
    ]


def test_json_parse_ignores_invalid_and_strictly_decreasing_rows(tmp_path):
    tmp_file = Path(tmp_path) / "invalid-gesture.json"
    payload = {
        "strokes": [
            [
                [1, 2, 0],
                [2, 3, -1],
                [3, 4, math.nan],
                [math.inf, 5, 1],
                [5, -math.inf, 1],
                ["bad-x", 6, 1],
                [6, "bad-y", 1],
                [7, 8, "bad-time"],
                [8, 9],
                None,
                [9, 10, 10],
                [10, 11, 5],
                [11, 12, 10],
                [12, 13, 20],
            ]
        ]
    }
    tmp_file.write_text(json.dumps(payload), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    JSONUtil.readGesture(str(tmp_file), cb)

    assert [(point.X, point.Y, point.T) for point in out["points"]] == [
        (1.0, 2.0, 0.0),
        (9.0, 10.0, 10.0),
        (11.0, 12.0, 10.0),
        (12.0, 13.0, 20.0),
    ]
    assert all(math.isfinite(point.T) for point in out["points"])
