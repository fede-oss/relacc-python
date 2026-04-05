import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _sample_rows(offset=0):
    return [
        "stroke_id x y time is_writing",
        f"0 {10+offset} 20 0 1",
        f"0 {12+offset} 22 10 1",
        f"1 {14+offset} 24 20 1",
        f"1 {16+offset} 25 30 1",
    ]


def test_main_distribution_help():
    res = subprocess.run(
        [sys.executable, str(ROOT / "main-distribution.py"), "-h"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "reference" in res.stdout
    assert "candidate" in res.stdout


def test_main_distribution_json_stdout(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(reference_dir / "s02-arrow-02.csv", _sample_rows(1))
    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(2))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-distribution.py"),
            "-f",
            "json",
            str(reference_dir),
            str(candidate_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert payload["metadata"]["comparisonMode"] == "distribution"
    assert payload["metadata"]["validClassCount"] == 1
    assert len(payload["results"]["overall"]) > 0


def test_main_distribution_csv_file_output_parent_dir_grouping(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "arrow" / "ref-a.csv", _sample_rows(0))
    _write_csv(reference_dir / "arrow" / "ref-b.csv", _sample_rows(1))
    _write_csv(candidate_dir / "arrow" / "cand-a.csv", _sample_rows(2))

    out_csv = tmp_path / "distribution.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-distribution.py"),
            "--group-by",
            "parent-dir",
            "-o",
            str(out_csv),
            str(reference_dir),
            str(candidate_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert content.splitlines()[0].startswith("scope,classKey,gestureMetric")


def test_main_distribution_invalid_format_fails(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(reference_dir / "s02-arrow-02.csv", _sample_rows(1))
    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(2))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-distribution.py"),
            "-f",
            "xml",
            str(reference_dir),
            str(candidate_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Invalid output format" in res.stderr


def test_main_distribution_invalid_rate_fails(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(reference_dir / "s02-arrow-02.csv", _sample_rows(1))
    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(2))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-distribution.py"),
            "-r",
            "0",
            str(reference_dir),
            str(candidate_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Sampling rate must be >= 1" in res.stderr


def test_main_distribution_invalid_grouping_fails(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(reference_dir / "s02-arrow-02.csv", _sample_rows(1))
    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(2))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-distribution.py"),
            "--group-by",
            "broken",
            str(reference_dir),
            str(candidate_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "invalid choice" in res.stderr
