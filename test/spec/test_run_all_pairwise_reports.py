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


def test_run_all_pairwise_reports_writes_generic_distribution_summary(tmp_path):
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
    _write_csv(
        datasets_root / "generated" / "1dollar" / "syntTO" / "g02-arrow-fast-t02.csv",
        3,
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
    within_reference_rows = _read_csv(class_dir / "within_reference.csv")
    between_group_rows = _read_csv(class_dir / "between_groups.csv")
    distribution_rows = _read_csv(class_dir / "distribution.csv")
    summary_distribution_rows = _read_csv(class_dir / "summary_distribution.csv")
    run_distribution_rows = _read_csv(run_dir / "distribution.csv")
    top_distribution_rows = _read_csv(output_dir / "distribution.csv")
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))

    assert len(pairwise_rows) == 2
    assert pairwise_rows[0]["mode"] == "reference-summary"
    assert pairwise_rows[0]["alignmentName"] == "chronological"
    assert len(baseline_rows) == 2
    assert {row["mode"] for row in baseline_rows} == {"human-summary-baseline"}
    assert {row["source"] for row in baseline_rows} == {"human"}
    assert [row["sampleKey"] for row in baseline_rows] == [
        "s01-arrow-fast-t01",
        "s02-arrow-fast-t02",
    ]
    assert len(baseline_stats_rows) > 0
    assert len(within_reference_rows) == 1
    assert len(between_group_rows) == 4
    assert {row["mode"] for row in within_reference_rows} == {"within-reference"}
    assert {row["mode"] for row in between_group_rows} == {"between-groups"}
    assert len(distribution_rows) == len(baseline_stats_rows)
    assert distribution_rows[0]["wassersteinDistance"] != ""
    assert distribution_rows[0]["jensenShannonDivergence"] != ""
    assert "baselineMean" not in distribution_rows[0]
    assert distribution_rows[0]["withinComparisonN"] == "1"
    assert distribution_rows[0]["betweenGroupsMean"] != ""
    assert len(summary_distribution_rows) == len(distribution_rows)
    assert (
        summary_distribution_rows[0]["withinReferenceMean"]
        == baseline_stats_rows[0]["mean"]
    )
    assert "withinComparisonToReferenceMeanRatio" in summary_distribution_rows[0]
    assert len(run_distribution_rows) == len(distribution_rows)
    assert top_distribution_rows == run_distribution_rows
    assert run_manifest["pairwiseRows"] == 2
    assert manifest["alignmentName"] == "chronological"
    assert run_manifest["baselineRows"] == 2
    assert run_manifest["withinReferenceRows"] == 1
    assert run_manifest["withinComparisonRows"] == 1
    assert run_manifest["betweenGroupsRows"] == 4
    assert run_manifest["distributionRows"] == len(distribution_rows)
    assert run_manifest["summaryDistributionRows"] == len(summary_distribution_rows)
    assert run_manifest["classes"][0]["baselineMode"] == "human-summary-baseline"
    assert run_manifest["classes"][0]["directDistributionMode"] == (
        "direct-distribution-pairs"
    )
    assert run_manifest["classes"][0]["withinComparisonRows"] == 1
    assert run_manifest["classes"][0]["withinReferenceRows"] == 1
    assert run_manifest["classes"][0]["betweenGroupsRows"] == 4
    assert run_manifest["classes"][0]["distributionRows"] == len(distribution_rows)
    assert run_metadata["experiment"] == "run-all-pairwise-reports"
    assert run_metadata["runtimeArgs"]["verbose"] == 2
    assert run_metadata["effectiveConfig"]["outputDir"] == str(output_dir)
    assert "Parsed arguments (opt):" in (output_dir / "run.log").read_text(
        encoding="utf-8"
    )
    assert "planned 1 runs" in (output_dir / "stdout.log").read_text(encoding="utf-8")

    combined_dir = output_dir / "combined"
    combined_pairwise_rows = _read_csv(combined_dir / "pairwise.csv")
    combined_stats_rows = _read_csv(combined_dir / "stats.csv")
    combined_baseline_rows = _read_csv(combined_dir / "baseline.csv")
    combined_within_reference_rows = _read_csv(combined_dir / "within_reference.csv")
    combined_within_comparison_rows = _read_csv(combined_dir / "within_comparison.csv")
    combined_between_group_rows = _read_csv(combined_dir / "between_groups.csv")
    combined_distribution_rows = _read_csv(combined_dir / "distribution.csv")
    combined_summary_distribution_rows = _read_csv(
        combined_dir / "summary_distribution.csv"
    )
    aggregate_rows = _read_csv(combined_dir / "aggregate_summaries.csv")
    report = json.loads((combined_dir / "report.json").read_text(encoding="utf-8"))
    raw_metric_rows = [
        json.loads(line)
        for line in (combined_dir / "raw_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    raw_distribution_rows = [
        json.loads(line)
        for line in (combined_dir / "raw_distributions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert combined_pairwise_rows == _read_csv(run_dir / "pairwise.csv")
    assert combined_stats_rows == _read_csv(run_dir / "stats.csv")
    assert combined_baseline_rows == baseline_rows
    assert combined_within_reference_rows == within_reference_rows
    assert combined_within_comparison_rows == _read_csv(
        run_dir / "within_comparison.csv"
    )
    assert combined_between_group_rows == between_group_rows
    assert combined_distribution_rows == top_distribution_rows
    assert combined_summary_distribution_rows == summary_distribution_rows
    assert {
        "comparison-to-reference-summary",
        "human-baseline",
        "within-reference",
        "within-comparison",
        "between-groups",
    } <= {row["recordSet"] for row in aggregate_rows}
    assert {"overall", "source", "dataset", "source-dataset", "run"} <= {
        row["scope"] for row in aggregate_rows
    }
    assert len(raw_metric_rows) == 10 * len(PairwiseReports.METRIC_NAMES)
    assert {row["recordType"] for row in raw_metric_rows} == {"rawMetricOutput"}
    assert {row["recordSet"] for row in raw_metric_rows} == {
        "comparison-to-reference-summary",
        "human-baseline",
        "within-reference",
        "within-comparison",
        "between-groups",
    }
    assert len(raw_distribution_rows) == (
        len(distribution_rows) * len(PairwiseReports.DISTRIBUTION_OUTPUT_VALUE_COLUMNS)
    )
    assert report["metadata"]["pairwiseRows"] == 2
    assert report["metadata"]["withinReferenceRows"] == 1
    assert report["metadata"]["withinComparisonRows"] == 1
    assert report["metadata"]["betweenGroupsRows"] == 4
    assert report["metadata"]["summary"] == "medoid"
    assert report["metadata"]["distributionSampleLimitPerClass"] == 16
    assert report["files"]["withinComparison"] == str(
        combined_dir / "within_comparison.csv"
    )
    assert report["files"]["summaryDistribution"] == str(
        combined_dir / "summary_distribution.csv"
    )
    assert manifest["combinedOutputs"]["directory"] == str(combined_dir)


def test_run_all_summary_validation_rejects_impossible_bounds():
    with pytest.raises(ValueError, match="outside min=0.0 and max=1.0"):
        PairwiseReports._validate_bounded_stats(
            {"mean": 17.0, "mdn": 0.0, "min": 0.0, "max": 1.0},
            ("mean", "mdn"),
            "strokeError",
        )
