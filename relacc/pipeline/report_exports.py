from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

from relacc.pipeline._common import format_csv_rows, write_jsonl_rows
from relacc.pipeline.report_schema import (
    AGGREGATE_SUMMARY_COLUMNS,
    BASELINE_COLUMNS,
    COMBINED_AGGREGATE_SUMMARIES_FILENAME,
    COMBINED_BASELINE_FILENAME,
    COMBINED_BASELINE_STATS_FILENAME,
    COMBINED_BETWEEN_GROUPS_FILENAME,
    COMBINED_BETWEEN_GROUPS_STATS_FILENAME,
    COMBINED_DISTRIBUTION_FILENAME,
    COMBINED_OUTPUT_DIRNAME,
    COMBINED_PAIRWISE_FILENAME,
    COMBINED_RAW_DISTRIBUTIONS_FILENAME,
    COMBINED_RAW_METRICS_FILENAME,
    COMBINED_REPORT_FILENAME,
    COMBINED_STATS_FILENAME,
    COMBINED_SUMMARY_DISTRIBUTION_FILENAME,
    COMBINED_WITHIN_COMPARISON_FILENAME,
    COMBINED_WITHIN_COMPARISON_STATS_FILENAME,
    COMBINED_WITHIN_REFERENCE_FILENAME,
    COMBINED_WITHIN_REFERENCE_STATS_FILENAME,
    DISTRIBUTION_COLUMNS,
    PAIRWISE_COLUMNS,
    REMOVED_INFERENTIAL_FIELDS,
    STATISTICS_SCHEMA_VERSION,
    STATS_COLUMNS,
    statistical_contract_fields as _statistical_contract_fields,
)


DEFAULT_VARIANT_LABEL = "root"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_readme(path: Path, title: str, extra_lines: Sequence[str] = ()) -> None:
    lines = [
        title,
        "=" * len(title),
        "",
        "Files in this folder:",
        "- pairwise.csv / pairwise.json: generated candidate gestures compared with the human summary gesture.",
        "- stats.csv: per-metric aggregate statistics for pairwise.csv.",
        "- baseline.csv / baseline.json: human/reference gestures compared with the human summary gesture.",
        "- baseline_stats.csv: per-metric aggregate statistics for baseline.csv.",
        "- within_reference.csv: direct human-human reference pairs used for direct distribution evidence.",
        "- within_reference_stats.csv: per-metric aggregate statistics for within_reference.csv.",
        "- within_comparison.csv: direct generated-generated pairs within the same generator/class.",
        "- within_comparison_stats.csv: per-metric aggregate statistics for within_comparison.csv.",
        "- between_groups.csv: direct human-generated pairs for the same class.",
        "- between_groups_stats.csv: per-metric aggregate statistics for between_groups.csv.",
        "- distribution.csv: distribution summaries and distribution metrics using within_reference, within_comparison, and between_groups.",
        "- summary_distribution.csv: distribution summaries using summary-relative baseline.csv and pairwise.csv plus within_comparison.csv.",
        "- manifest.json: machine-readable run metadata, counts, warnings, and child folders.",
        "- run.json / run.log / stdout.log / stderr.log: top-level reproducibility and captured console logs when present.",
        "",
        "Column notes:",
        f"- variant is '{DEFAULT_VARIANT_LABEL}' when files are directly under a generator/dataset folder, otherwise it is the source subfolder such as syntTO or recoTO.",
        "- summary records the summary strategy used for metric computation, for example medoid or kmedoid.",
        "- Empty skewness/kurtosis cells mean there were not enough finite samples: skewness needs at least 3 values, and both statistics require non-constant values.",
        "- Empty normalizedWassersteinDistance and SD ratio cells usually mean the reference standard deviation was 0 or unavailable.",
        "- inf in KL-family metrics means one empirical distribution assigned probability to a bin where the other distribution had zero probability; Jensen-Shannon divergence should remain finite.",
    ]
    if extra_lines:
        lines.extend(["", *extra_lines])
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_csv_rows(rows, columns), encoding="utf-8")


def write_combined_report_exports(
    output_root: Path,
    pairwise_rows: Sequence[Dict[str, object]],
    stats_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    baseline_stats_rows: Sequence[Dict[str, object]],
    within_reference_rows: Sequence[Dict[str, object]],
    within_reference_stats_rows: Sequence[Dict[str, object]],
    within_comparison_rows: Sequence[Dict[str, object]],
    within_comparison_stats_rows: Sequence[Dict[str, object]],
    between_group_rows: Sequence[Dict[str, object]],
    between_group_stats_rows: Sequence[Dict[str, object]],
    distribution_rows: Sequence[Dict[str, object]],
    summary_distribution_rows: Sequence[Dict[str, object]],
    aggregate_rows: Sequence[Dict[str, object]],
    raw_metric_outputs: Sequence[Dict[str, object]],
    raw_distribution_outputs: Sequence[Dict[str, object]],
    manifest: Dict[str, object],
    write_raw_jsonl: bool = True,
) -> Dict[str, str]:
    combined_dir = output_root / COMBINED_OUTPUT_DIRNAME
    combined_dir.mkdir(parents=True, exist_ok=True)

    pairwise_path = combined_dir / COMBINED_PAIRWISE_FILENAME
    stats_path = combined_dir / COMBINED_STATS_FILENAME
    baseline_path = combined_dir / COMBINED_BASELINE_FILENAME
    baseline_stats_path = combined_dir / COMBINED_BASELINE_STATS_FILENAME
    within_reference_path = combined_dir / COMBINED_WITHIN_REFERENCE_FILENAME
    within_reference_stats_path = combined_dir / COMBINED_WITHIN_REFERENCE_STATS_FILENAME
    within_comparison_path = combined_dir / COMBINED_WITHIN_COMPARISON_FILENAME
    within_comparison_stats_path = (
        combined_dir / COMBINED_WITHIN_COMPARISON_STATS_FILENAME
    )
    between_groups_path = combined_dir / COMBINED_BETWEEN_GROUPS_FILENAME
    between_groups_stats_path = combined_dir / COMBINED_BETWEEN_GROUPS_STATS_FILENAME
    distribution_path = combined_dir / COMBINED_DISTRIBUTION_FILENAME
    summary_distribution_path = combined_dir / COMBINED_SUMMARY_DISTRIBUTION_FILENAME
    aggregate_path = combined_dir / COMBINED_AGGREGATE_SUMMARIES_FILENAME
    raw_metrics_path = combined_dir / COMBINED_RAW_METRICS_FILENAME
    raw_distributions_path = combined_dir / COMBINED_RAW_DISTRIBUTIONS_FILENAME
    report_path = combined_dir / COMBINED_REPORT_FILENAME

    _write_csv(pairwise_path, pairwise_rows, PAIRWISE_COLUMNS)
    _write_csv(stats_path, stats_rows, STATS_COLUMNS)
    _write_csv(baseline_path, baseline_rows, BASELINE_COLUMNS)
    _write_csv(baseline_stats_path, baseline_stats_rows, STATS_COLUMNS)
    _write_csv(within_reference_path, within_reference_rows, PAIRWISE_COLUMNS)
    _write_csv(
        within_reference_stats_path,
        within_reference_stats_rows,
        STATS_COLUMNS,
    )
    _write_csv(within_comparison_path, within_comparison_rows, PAIRWISE_COLUMNS)
    _write_csv(
        within_comparison_stats_path,
        within_comparison_stats_rows,
        STATS_COLUMNS,
    )
    _write_csv(between_groups_path, between_group_rows, PAIRWISE_COLUMNS)
    _write_csv(between_groups_stats_path, between_group_stats_rows, STATS_COLUMNS)
    _write_csv(distribution_path, distribution_rows, DISTRIBUTION_COLUMNS)
    _write_csv(summary_distribution_path, summary_distribution_rows, DISTRIBUTION_COLUMNS)
    _write_csv(aggregate_path, aggregate_rows, AGGREGATE_SUMMARY_COLUMNS)
    if write_raw_jsonl:
        write_jsonl_rows(raw_metrics_path, raw_metric_outputs)
        write_jsonl_rows(raw_distributions_path, raw_distribution_outputs)

    _write_json(
        report_path,
        {
            "metadata": {
                "mode": manifest["mode"],
                "baselineMode": manifest["baselineMode"],
                **_statistical_contract_fields(),
                "datasetsRoot": manifest["datasetsRoot"],
                "outputDir": manifest["outputDir"],
                "groupBy": manifest["groupBy"],
                "classScheme": manifest["classScheme"],
                "rate": manifest["rate"],
                "roundPrecision": manifest["roundPrecision"],
                "alignment": manifest["alignment"],
                "alignmentName": manifest["alignmentName"],
                "summary": manifest["summary"],
                "popular": manifest["popular"],
                "exactDtw": manifest["exactDtw"],
                "dtwWindow": manifest["dtwWindow"],
                "sampleLimitPerClass": manifest.get("sampleLimitPerClass"),
                "distributionSampleLimitPerClass": manifest.get(
                    "distributionSampleLimitPerClass"
                ),
                "sampleSeed": manifest.get("sampleSeed"),
                "samplingMode": manifest.get("samplingMode"),
                "classLimitPerRun": manifest.get("classLimitPerRun"),
                "directDistributionPairs": manifest.get("directDistributionPairs"),
                "runCount": len(manifest["runs"]),
                "pairwiseRows": len(pairwise_rows),
                "statsRows": len(stats_rows),
                "baselineRows": len(baseline_rows),
                "baselineStatsRows": len(baseline_stats_rows),
                "withinReferenceRows": len(within_reference_rows),
                "withinReferenceStatsRows": len(within_reference_stats_rows),
                "withinComparisonRows": len(within_comparison_rows),
                "withinComparisonStatsRows": len(within_comparison_stats_rows),
                "betweenGroupsRows": len(between_group_rows),
                "betweenGroupsStatsRows": len(between_group_stats_rows),
                "distributionRows": len(distribution_rows),
                "summaryDistributionRows": len(summary_distribution_rows),
                "aggregateRows": len(aggregate_rows),
                "rawMetricRows": len(raw_metric_outputs),
                "rawDistributionRows": len(raw_distribution_outputs),
                "rawJsonlWritten": bool(write_raw_jsonl),
            },
            "files": {
                "pairwise": str(pairwise_path),
                "stats": str(stats_path),
                "baseline": str(baseline_path),
                "baselineStats": str(baseline_stats_path),
                "withinReference": str(within_reference_path),
                "withinReferenceStats": str(within_reference_stats_path),
                "withinComparison": str(within_comparison_path),
                "withinComparisonStats": str(within_comparison_stats_path),
                "betweenGroups": str(between_groups_path),
                "betweenGroupsStats": str(between_groups_stats_path),
                "distribution": str(distribution_path),
                "summaryDistribution": str(summary_distribution_path),
                "aggregateSummaries": str(aggregate_path),
                "rawMetrics": str(raw_metrics_path) if write_raw_jsonl else None,
                "rawDistributions": (
                    str(raw_distributions_path) if write_raw_jsonl else None
                ),
            },
            "columns": {
                "pairwise": list(PAIRWISE_COLUMNS),
                "stats": list(STATS_COLUMNS),
                "baseline": list(BASELINE_COLUMNS),
                "baselineStats": list(STATS_COLUMNS),
                "withinReference": list(PAIRWISE_COLUMNS),
                "withinReferenceStats": list(STATS_COLUMNS),
                "withinComparison": list(PAIRWISE_COLUMNS),
                "withinComparisonStats": list(STATS_COLUMNS),
                "betweenGroups": list(PAIRWISE_COLUMNS),
                "betweenGroupsStats": list(STATS_COLUMNS),
                "distribution": list(DISTRIBUTION_COLUMNS),
                "summaryDistribution": list(DISTRIBUTION_COLUMNS),
                "aggregateSummaries": list(AGGREGATE_SUMMARY_COLUMNS),
            },
            "runs": manifest["runs"],
            "warnings": manifest["warnings"],
            "planningWarnings": manifest.get("planningWarnings", []),
        },
    )
    _write_readme(
        combined_dir,
        "Combined Evaluation Outputs",
        [
            "These files concatenate rows across all completed source/dataset/variant runs.",
            (
                "Distribution outputs use descriptive-pair-distances statistical mode; "
                "this is the fixed statistical contract, not a selectable alternative "
                "to the distribution comparison workflow: "
                "independentUnit=gesture-file, pairValuesIndependent=false, "
                f"statisticsSchemaVersion={STATISTICS_SCHEMA_VERSION}, "
                "removedInferentialFields="
                f"{json.dumps(list(REMOVED_INFERENTIAL_FIELDS), separators=(',', ':'))}."
            ),
            "Use raw_metrics.jsonl for per-comparison histograms when you want pre-melted long-form metric samples.",
            "Use distribution.csv or summary_distribution.csv for aggregate plots that do not need every raw pair.",
        ],
    )
    return {
        "directory": str(combined_dir),
        "pairwise": str(pairwise_path),
        "stats": str(stats_path),
        "baseline": str(baseline_path),
        "baselineStats": str(baseline_stats_path),
        "withinReference": str(within_reference_path),
        "withinReferenceStats": str(within_reference_stats_path),
        "withinComparison": str(within_comparison_path),
        "withinComparisonStats": str(within_comparison_stats_path),
        "betweenGroups": str(between_groups_path),
        "betweenGroupsStats": str(between_groups_stats_path),
        "distribution": str(distribution_path),
        "summaryDistribution": str(summary_distribution_path),
        "aggregateSummaries": str(aggregate_path),
        "rawMetrics": str(raw_metrics_path) if write_raw_jsonl else None,
        "rawDistributions": str(raw_distributions_path) if write_raw_jsonl else None,
        "report": str(report_path),
    }
