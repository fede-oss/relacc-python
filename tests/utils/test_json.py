import json
from pathlib import Path

from relacc.utils.json import JSONUtil


def test_json_parse_dedupe_and_nan(tmp_path):
    tmp_file = Path(tmp_path) / "gesture.json"
    payload = {
        "strokes": [
            [[1, 2, 0], [3, 4, 0], [5, 6, -1]],
            [[7, 8, 5]],
        ]
    }
    tmp_file.write_text(json.dumps(payload), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    JSONUtil.readGesture(str(tmp_file), cb)
    points = out["points"]

    assert len(points) == 3
    assert points[0].X == 1
    assert points[0].T == 0
    assert points[1].X == 5
    assert points[1].T != points[1].T
    assert points[2].StrokeID == 1
