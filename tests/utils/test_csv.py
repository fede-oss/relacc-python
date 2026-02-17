from pathlib import Path

from relacc.utils.csv import CSVUtil


def test_csv_parse_dedupe_and_nan(tmp_path):
    tmp_file = Path(tmp_path) / "gesture.csv"
    rows = [
        "stroke_id x y time is_writing",
        "0 10 20 0 1",
        "0 11 21 0 1",
        "1 12 22 -1 1",
    ]
    tmp_file.write_text("\n".join(rows), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(tmp_file), cb)
    points = out["points"]

    assert len(points) == 2
    assert points[0].X == 10
    assert points[0].T == 0
    assert points[1].X == 12
    assert points[1].T != points[1].T
    assert points[1].StrokeID == 1
