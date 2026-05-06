import gzip
import io
import tarfile
from contextlib import ExitStack
from pathlib import Path

import pytest

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


def test_csv_parse_tab_delimited_float_values(tmp_path):
    tmp_file = Path(tmp_path) / "gesture-tab.csv"
    rows = [
        "stroke_id\t x\t y\t time\t is_writing",
        "1.000000 \t 281.000000 \t -179.000000 \t 0.000000 \t 1.000000",
        "1.000000 \t 280.999996 \t -178.999992 \t 0.000000 \t 1.000000",
        "2.000000 \t 280.500000 \t -178.500000 \t 14.000000 \t 1.000000",
    ]
    tmp_file.write_text("\n".join(rows), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(tmp_file), cb)
    points = out["points"]

    assert len(points) == 2
    assert points[0].X == 281.0
    assert points[0].Y == -179.0
    assert points[0].T == 0.0
    assert points[0].StrokeID == 1
    assert points[1].X == 280.5
    assert points[1].T == 14.0
    assert points[1].StrokeID == 2


def test_csv_parse_gzip_text(tmp_path):
    tmp_file = Path(tmp_path) / "gesture.csv"
    gz_file = Path(tmp_path) / "gesture.csv.gz"
    rows = [
        "stroke_id x y time is_writing",
        "0 10 20 0 1",
        "0 11 21 10 1",
    ]
    tmp_file.write_text("\n".join(rows), encoding="utf-8")
    gz_file.write_bytes(gzip.compress(tmp_file.read_bytes()))

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(gz_file), cb)
    points = out["points"]

    assert len(points) == 2
    assert points[0].X == 10.0
    assert points[1].T == 10.0

def test_csv_parse_gzip_tar_payload_with_csv_extension(tmp_path):
    payload_file = Path(tmp_path) / "gesture.csv"
    rows = [
        "stroke_id\t x\t y\t time\t is_writing",
        "1.000000 \t 50.0 \t 88.0 \t 0.0 \t 1.0",
        "1.000000 \t 50.5 \t 88.5 \t 10.0 \t 1.0",
        "2.000000 \t 51.0 \t 89.0 \t 20.0 \t 1.0",
    ]
    csv_bytes = "\n".join(rows).encode("utf-8")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        info = tarfile.TarInfo(name="inner/gesture.csv")
        info.size = len(csv_bytes)
        archive.addfile(info, io.BytesIO(csv_bytes))

    payload_file.write_bytes(gzip.compress(tar_buffer.getvalue()))

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(payload_file), cb)
    points = out["points"]

    assert len(points) == 3
    assert points[0].StrokeID == 1
    assert points[1].T == 10.0
    assert points[2].StrokeID == 2


def test_csv_parse_gzip_tar_skips_non_csv_members(tmp_path):
    payload_file = Path(tmp_path) / "gesture.csv"
    csv_rows = [
        "stroke_id x y time is_writing",
        "1 50 88 0 1",
        "1 51 89 10 1",
    ]
    csv_bytes = "\n".join(csv_rows).encode("utf-8")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        readme_bytes = b"not,a,gesture\nhello\n"
        readme = tarfile.TarInfo(name="README.txt")
        readme.size = len(readme_bytes)
        archive.addfile(readme, io.BytesIO(readme_bytes))

        gesture = tarfile.TarInfo(name="inner/gesture.csv")
        gesture.size = len(csv_bytes)
        archive.addfile(gesture, io.BytesIO(csv_bytes))

    payload_file.write_bytes(gzip.compress(tar_buffer.getvalue()))

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(payload_file), cb)
    points = out["points"]

    assert len(points) == 2
    assert points[0].StrokeID == 1
    assert points[1].T == 10.0


def test_csv_parse_plain_tar_member_without_csv_extension_when_header_matches(tmp_path):
    payload_file = Path(tmp_path) / "gesture.csv"
    csv_rows = [
        "stroke_id x y time is_writing",
        "1 50 88 0 1",
        "1 51 89 10 1",
    ]
    csv_bytes = "\n".join(csv_rows).encode("utf-8")

    with tarfile.open(payload_file, mode="w") as archive:
        member = tarfile.TarInfo(name="payload.txt")
        member.size = len(csv_bytes)
        archive.addfile(member, io.BytesIO(csv_bytes))

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(payload_file), cb)
    points = out["points"]

    assert len(points) == 2
    assert points[0].X == 50.0
    assert points[1].T == 10.0


def test_csv_archive_without_csv_payload_raises(tmp_path):
    archive_path = Path(tmp_path) / "empty-archive.csv"

    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(name="nested/")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)

    with pytest.raises(ValueError, match="No CSV file found in archive"):
        with CSVUtil._open_text_auto(str(archive_path)):
            pass


def test_csv_archive_skips_members_without_payload(monkeypatch):
    class FakeMember:
        name = "gesture.csv"

        def isfile(self):
            return True

    class FakeArchive:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([FakeMember()])

        def extractfile(self, _member):
            return None

    monkeypatch.setattr(tarfile, "open", lambda *_args, **_kwargs: FakeArchive())

    with ExitStack() as stack:
        with pytest.raises(ValueError, match="No CSV file found in archive"):
            CSVUtil._open_archive_payload(stack, "dummy.csv")


def test_csv_header_index_from_text_handles_bom_and_blank_payload():
    assert CSVUtil._header_index_from_text("\n\n") is None

    index = CSVUtil._header_index_from_text(
        "\ufeffstroke_id,x,y,time,is_writing\n0,1,2,3,1\n"
    )

    assert index == {"stroke_id": 0, "x": 1, "y": 2, "time": 3, "is_writing": 4}


def test_csv_empty_file_returns_empty_points(tmp_path):
    tmp_file = Path(tmp_path) / "empty.csv"
    tmp_file.write_text("", encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(tmp_file), cb)
    assert out["points"] == []


def test_csv_invalid_header_raises(tmp_path):
    tmp_file = Path(tmp_path) / "bad-header.csv"
    tmp_file.write_text("a b c d\n1 2 3 4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid CSV header"):
        CSVUtil.readGesture(str(tmp_file), lambda _points: None)


def test_csv_comma_delimited_and_sparse_rows(tmp_path):
    tmp_file = Path(tmp_path) / "comma.csv"
    rows = [
        "stroke_id,x,y,time,is_writing",
        "",
        "0,10,20,0,1",
        "1,12,22",
        "1,13,23,10,1",
    ]
    tmp_file.write_text("\n".join(rows), encoding="utf-8")

    out = {}

    def cb(points):
        out["points"] = points

    CSVUtil.readGesture(str(tmp_file), cb)
    points = out["points"]
    assert len(points) == 2
    assert points[0].X == 10.0
    assert points[1].X == 13.0


def test_csv_empty_numeric_field_raises(tmp_path):
    tmp_file = Path(tmp_path) / "empty-numeric.csv"
    tmp_file.write_text("stroke_id,x,y,time,is_writing\n0,10,20,,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Empty numeric field"):
        CSVUtil.readGesture(str(tmp_file), lambda _points: None)
