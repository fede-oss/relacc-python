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
    assert payload["metadata"]["comparisonMode"] == "one-vs-many"
    assert "shapeError" in payload["results"]
    assert payload["metadata"]["args"]["exact_dtw"] is False
    assert payload["metadata"]["args"]["dtw_window"] is None
    assert "dtwDistance" in payload["results"]


def test_main_json_stats_output_with_exact_dtw(tmp_path):
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
            "--exact-dtw",
            str(f1),
            str(f2),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert payload["metadata"]["args"]["exact_dtw"] is True
    assert payload["metadata"]["args"]["dtw_window"] is None
    assert "dtwDistance" in payload["results"]


def test_main_json_stats_output_auto_switches_to_window_for_large_rate(tmp_path):
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
            "720",
            str(f1),
            str(f2),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert payload["metadata"]["args"]["dtw_window"] == 64


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
    assert out_csv.read_text(encoding="utf-8").startswith("measure,n,mean")
    assert out_xml.read_text(encoding="utf-8").startswith("<?xml")


def test_main_per_file_json_and_csv_output(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    f2 = tmp_path / "s1-arrow-t2.csv"
    _write_csv(f1, _sample_rows(0))
    _write_csv(f2, _sample_rows(1))

    json_res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
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

    payload = json.loads(json_res.stdout)
    assert payload["metadata"]["stats"] is False
    assert len(payload["results"]) == 2
    assert payload["results"][0]["file"] == "s1-arrow-t1"

    out_csv = tmp_path / "samples.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
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

    assert out_csv.read_text(encoding="utf-8").splitlines()[0].startswith(
        "file,inputFile,label,rate"
    )


def test_main_malformed_filename_label_inference_fails_cleanly(tmp_path):
    f1 = tmp_path / "arrow.csv"
    _write_csv(f1, _sample_rows(0))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            str(f1),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Cannot derive gesture label" in res.stderr


def test_get_stats_returns_nan_when_any_values_are_non_finite():
    stats = relacc_cli.getStats([1.0, float("nan"), 3.0, float("inf")])
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["mdn"])
    assert math.isnan(stats["sd"])
    assert math.isnan(stats["min"])
    assert math.isnan(stats["max"])
    assert stats["n"] == 4


def test_get_stats_returns_nan_when_all_values_are_non_finite():
    stats = relacc_cli.getStats([float("nan"), float("inf")])
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["mdn"])
    assert math.isnan(stats["sd"])
    assert math.isnan(stats["min"])
    assert math.isnan(stats["max"])
    assert stats["n"] == 2


def test_to_json_encodes_all_non_finite_stats_as_null():
    stats = {"shapeError": relacc_cli.getStats([float("nan"), float("inf")])}

    payload = json.loads(relacc_cli.toJSON(stats, {"format": "json"}))

    assert payload["results"]["shapeError"] == {
        "mean": None,
        "mdn": None,
        "sd": None,
        "min": None,
        "max": None,
        "n": 2,
    }


def test_main_json_stats_surface_non_finite_metrics(tmp_path):
    valid = tmp_path / "s1-arrow-t1.csv"
    invalid = tmp_path / "s1-arrow-t2.csv"
    _write_csv(valid, _sample_rows(0))
    invalid.write_text(
        "\n".join(
            [
                "stroke_id x y time is_writing",
                "0 10 20 -1 1",
                "0 12 22 -2 1",
                "1 14 24 -3 1",
            ]
        ),
        encoding="utf-8",
    )

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "-s",
            "-f",
            "json",
            str(valid),
            str(invalid),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)

    assert payload["results"]["shapeError"]["n"] == 2
    assert payload["results"]["shapeError"]["mean"] == 0.0
    assert payload["results"]["timeError"] == {
        "mean": None,
        "mdn": None,
        "sd": None,
        "min": None,
        "max": None,
        "n": 2,
    }


def test_main_rejects_dtw_window_with_exact_dtw(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--exact-dtw",
            "--dtw-window",
            "3",
            str(f1),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "--dtw-window cannot be combined with --exact-dtw" in res.stderr
