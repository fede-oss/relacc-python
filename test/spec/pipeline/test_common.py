import json

import pytest

from relacc.pipeline import _common as Common


def test_read_gesture_points_accepts_json_and_rejects_empty_or_unknown(tmp_path):
    json_file = tmp_path / "gesture.json"
    json_file.write_text(
        json.dumps({"strokes": [[[1, 2, 0], [3, 4, 5]]]}),
        encoding="utf-8",
    )

    points = Common.read_gesture_points(str(json_file))
    assert len(points) == 2
    assert points[0].X == 1
    assert points[1].T == 5

    empty_json = tmp_path / "empty.json"
    empty_json.write_text(json.dumps({"strokes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="No points parsed"):
        Common.read_gesture_points(str(empty_json))

    unknown = tmp_path / "gesture.txt"
    unknown.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid input file format"):
        Common.read_gesture_points(str(unknown))


def test_output_format_uses_output_extension_then_requested_or_default():
    assert Common.output_format("/tmp/report.csv", None) == "csv"
    assert Common.output_format(None, "xml") == "xml"
    assert Common.output_format(None, None, default="text") == "text"
