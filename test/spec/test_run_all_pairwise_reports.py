import csv
import json
from pathlib import Path

import pytest

import run_all_pairwise_reports as PairwiseReports


def _write_csv(path: Path, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "stroke_id x y time is_writing",
                f"0 {10 + offset} 20 0 1",
                f"0 {12 + offset} 22 10 1",
                f"1 {14 + offset} 24 20 1",
                f"1 {16 + offset} 25 30 1",
            ]
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path):
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_run_all_pairwise_reports_writes_lightweight_human_baseline(tmp_path):
    datasets_root = tmp_path / "datasets"
    output_dir = tmp_path / "report"

    _write_csv(
        datasets_root / "humans" / "1dollar" / "realTO" / "s01-arrow-fast-t01.csv",
        0,
    )
    _write_csv(
        datasets_root / "humans" / "1dollar" / "realTO" / "s02-arrow-fast-t02.csv",
        1,
    )
    _write_csv(
        datasets_root / "generated" / "1dollar" / "syntTO" / "g01-arrow-fast-t01.csv",
        2,
    )

    assert (
        PairwiseReports.main(
            [
                "--datasets-root",
                str(datasets_root),
                "--output-dir",
                str(output_dir),
                "--datasets",
                "1dollar",
                "--sources",
                "generated",
                "--rate",
                "8",
                "--verbosity",
                "2",
            ]
        )
        == 0
    )

    run_dir = output_dir / "generated" / "1dollar" / "syntTO"
    class_dir = run_dir / "classes" / "arrow"
    pairwise_rows = _read_csv(class_dir / "pairwise.csv")
    baseline_rows = _read_csv(class_dir / "baseline.csv")
    baseline_stats_rows = _read_csv(class_dir / "baseline_stats.csv")
    distribution_rows = _read_csv(class_dir / "distribution.csv")
    run_distribution_rows = _read_csv(run_dir / "distribution.csv")
    top_distribution_rows = _read_csv(output_dir / "distribution.csv")
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))

    assert len(pairwise_rows) == 1
    assert pairwise_rows[0]["mode"] == "reference-summary"
    assert len(baseline_rows) == 2
    assert {row["mode"] for row in baseline_rows} == {"human-summary-baseline"}
    assert {row["source"] for row in baseline_rows} == {"human"}
    assert [row["sampleKey"] for row in baseline_rows] == [
        "s01-arrow-fast-t01",
        "s02-arrow-fast-t02",
    ]
    assert len(baseline_stats_rows) > 0
    assert len(distribution_rows) == len(baseline_stats_rows)
    assert distribution_rows[0]["wassersteinDistance"] != ""
    assert distribution_rows[0]["baselineMean"] == baseline_stats_rows[0]["mean"]
    assert len(run_distribution_rows) == len(distribution_rows)
    assert top_distribution_rows == run_distribution_rows
    assert run_manifest["pairwiseRows"] == 1
    assert run_manifest["baselineRows"] == 2
    assert run_manifest["distributionRows"] == len(distribution_rows)
    assert run_manifest["classes"][0]["baselineMode"] == "human-summary-baseline"
    assert run_manifest["classes"][0]["distributionRows"] == len(distribution_rows)
    assert run_metadata["experiment"] == "run-all-pairwise-reports"
    assert run_metadata["runtimeArgs"]["verbose"] == 2
    assert run_metadata["effectiveConfig"]["outputDir"] == str(output_dir)
    assert "Parsed arguments (opt):" in (output_dir / "run.log").read_text(
        encoding="utf-8"
    )
    assert "planned 1 runs" in (output_dir / "stdout.log").read_text(encoding="utf-8")


def test_run_all_summary_validation_rejects_impossible_bounds():
    with pytest.raises(ValueError, match="outside min=0.0 and max=1.0"):
        PairwiseReports._validate_bounded_stats(
            {"mean": 17.0, "mdn": 0.0, "min": 0.0, "max": 1.0},
            ("mean", "mdn"),
            "strokeError",
        )
