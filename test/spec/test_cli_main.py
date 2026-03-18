import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import relacc_cli


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path, rows):
    path.write_text("\n".join(rows), encoding="utf-8")


def _sample_rows(offset=0):
    return [
        "stroke_id x y time is_writing",
        f"0 {10+offset} 20 0 1",
        f"0 {12+offset} 22 10 1",
        f"1 {14+offset} 24 20 1",
    ]


def test_main_json_stats_output(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    f2 = tmp_path / "s1-arrow-t2.csv"
    _write_csv(f1, _sample_rows(0))
    _write_csv(f2, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "-s",
            "-f",
            "json",
            "-r",
            "3",
            str(f1),
            str(f2),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert "metadata" in payload
    assert "results" in payload
    assert "shapeError" in payload["results"]
    assert "dtwDistance" in payload["results"]


def test_main_csv_and_xml_file_output(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    f2 = tmp_path / "s1-arrow-t2.csv"
    _write_csv(f1, _sample_rows(0))
    _write_csv(f2, _sample_rows(1))

    out_csv = tmp_path / "out.csv"
    out_xml = tmp_path / "out.xml"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "-s",
            "-o",
            str(out_csv),
            "-r",
            "3",
            str(f1),
            str(f2),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "-s",
            "-o",
            str(out_xml),
            "-r",
            "3",
            str(f1),
            str(f2),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_csv.exists()
    assert out_xml.exists()
    assert out_csv.read_text(encoding="utf-8").startswith("measure n mean")
    assert out_xml.read_text(encoding="utf-8").startswith("<?xml")


def test_get_stats_ignores_non_finite_values_when_possible():
    stats = relacc_cli.getStats([1.0, float("nan"), 3.0, float("inf")])
    assert stats == {
        "mean": 2.0,
        "mdn": 2.0,
        "sd": pytest.approx(1.414, abs=1e-3),
        "min": 1.0,
        "max": 3.0,
        "n": 2,
    }


def test_get_stats_returns_nan_when_all_values_are_non_finite():
    stats = relacc_cli.getStats([float("nan"), float("inf")])
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["mdn"])
    assert math.isnan(stats["sd"])
    assert math.isnan(stats["min"])
    assert math.isnan(stats["max"])
    assert stats["n"] == 0


def test_to_json_encodes_all_non_finite_stats_as_null():
    stats = {"shapeError": relacc_cli.getStats([float("nan"), float("inf")])}

    payload = json.loads(relacc_cli.toJSON(stats, {"format": "json"}))

    assert payload["results"]["shapeError"] == {
        "mean": None,
        "mdn": None,
        "sd": None,
        "min": None,
        "max": None,
        "n": 0,
    }
