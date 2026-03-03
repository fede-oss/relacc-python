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


def test_main_pairwise_help():
    res = subprocess.run(
        [sys.executable, str(ROOT / "main-pairwise.py"), "-h"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "reference" in res.stdout
    assert "candidate" in res.stdout


def test_main_pairwise_json_stdout_single_pair(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows(0))
    _write_csv(cand, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "-f",
            "json",
            "-r",
            "5",
            str(ref),
            str(cand),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert payload["metadata"]["comparisonMode"] == "direct"
    assert payload["metadata"]["pairCount"] == 1
    assert payload["pairs"][0]["rate"] == 5
    assert payload["pairs"][0]["mode"] == "direct"
    assert "shapeError" in payload["pairs"][0]


def test_main_pairwise_alignment_zero_is_accepted(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows(0))
    _write_csv(cand, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "-f",
            "json",
            "-a",
            "0",
            str(ref),
            str(cand),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert payload["metadata"]["alignment"] == 0


def test_main_pairwise_csv_file_output_and_no_strict(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "a.csv", _sample_rows(0))
    _write_csv(reference_dir / "only_ref.csv", _sample_rows(1))
    _write_csv(candidate_dir / "a.csv", _sample_rows(2))
    _write_csv(candidate_dir / "only_cand.csv", _sample_rows(3))

    out_csv = tmp_path / "pairs.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "--no-strict",
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
    assert content.splitlines()[0].startswith("pairKey,label,referenceFile")


def test_main_pairwise_summary_mode_json(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "r1.csv", _sample_rows(0))
    _write_csv(reference_dir / "r2.csv", _sample_rows(1))
    _write_csv(candidate_dir / "c1.csv", _sample_rows(2))
    _write_csv(candidate_dir / "c2.csv", _sample_rows(3))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "--mode",
            "summary",
            "-m",
            "centroid",
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
    assert payload["metadata"]["comparisonMode"] == "summary"
    assert payload["metadata"]["referenceCount"] == 2
    assert payload["metadata"]["pairCount"] == 2
    assert all(row["mode"] == "summary" for row in payload["pairs"])


def test_main_pairwise_invalid_format_fails(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows(0))
    _write_csv(cand, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "-f",
            "xml",
            str(ref),
            str(cand),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Invalid output format" in res.stderr


def test_main_pairwise_invalid_rate_fails(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows(0))
    _write_csv(cand, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "-r",
            "0",
            str(ref),
            str(cand),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Sampling rate must be >= 1" in res.stderr


def test_main_pairwise_invalid_summary_fails(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows(0))
    _write_csv(cand, _sample_rows(1))

    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main-pairwise.py"),
            "--summary",
            "unknown",
            str(ref),
            str(cand),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Invalid summary shape" in res.stderr
