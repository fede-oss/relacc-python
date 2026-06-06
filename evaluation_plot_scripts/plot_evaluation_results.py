#!/usr/bin/env python3
"""Create analysis plots from run_all_pairwise_reports.py outputs.

The report data uses generic comparison groups:

* baseline.csv: reference samples -> reference summary
* pairwise.csv: comparison samples -> reference summary
* distribution.csv: per-class/per-metric variability summaries

The preferred ratio is within-comparison variability divided by
within-reference variability. Older distribution.csv files are still accepted
through legacy-column fallbacks.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1e-12

CORE_METRICS = (
    "shapeError",
    "lengthError",
    "bendingError",
    "timeError",
    "velocityError",
    "dtwDistance",
)

MOVEMENT_METRICS = (
    "cornerSlowdown",
    "twoThirdsPowerLawR2",
    "highFrequencyRatio",
    "curvature",
    "strokeLengthStd",
    "meanStrokeDuration",
)

DTW_METRICS = (
    "dtwDistance",
    "ldtwDistance",
    "ddtwDistance",
    "wdtwDistance",
    "wddtwDistance",
)

SHAPE_METRICS = (
    "shapeError",
    "shapeVariability",
    "lengthError",
    "sizeError",
    "bendingError",
    "bendingVariability",
)

TIMING_METRICS = (
    "timeError",
    "timeVariability",
    "velocityError",
    "velocityVariability",
)

STROKE_METRICS = (
    "strokeError",
    "strokeOrderError",
    "strokeLengthStd",
    "meanStrokeDuration",
)

METRIC_FAMILIES = {
    "core": CORE_METRICS,
    "shape": SHAPE_METRICS,
    "timing": TIMING_METRICS,
    "movement": MOVEMENT_METRICS,
    "stroke": STROKE_METRICS,
    "dtw": DTW_METRICS,
}

REPRESENTATIVE_METRICS = (
    "shapeError",
    "dtwDistance",
    "velocityError",
    "curvature",
    "strokeLengthStd",
    "meanStrokeDuration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from report-output-full-metrics."
    )
    parser.add_argument(
        "--input-dir",
        default="report-output-full-metrics",
        help="Directory produced by run_all_pairwise_reports.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation-plots",
        help="Directory where plot folders and summary CSVs will be written.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=9,
        help="Number of representative source/dataset cases for raw histograms/ECDFs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: object) -> str:
    text = str(value if value not in (None, "") else "none")
    chars = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text]
    return "".join(chars) or "_"


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def first_float(row: dict, *field_names: str) -> float | None:
    for field_name in field_names:
        value = to_float(row.get(field_name))
        if value is not None:
            return value
    return None


def distribution_ratio(row: dict) -> float | None:
    return first_float(
        row,
        "withinComparisonToReferenceMeanRatio",
        "withinCandidateToBaselineMeanRatio",
        "meanRatio",
    )


def distribution_value(row: dict, generic_field: str, legacy_field: str) -> float | None:
    return first_float(row, generic_field, legacy_field)


def finite_values(rows: Iterable[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = to_float(row.get(metric))
        if value is None:
            continue
        # All current RELACC metrics are non-negative distances/differences.
        # Negative values indicate malformed input or a future incompatible metric.
        if value < -EPSILON:
            continue
        values.append(max(0.0, value))
    return values


def run_label(row: dict) -> str:
    variant = row.get("variant") or ""
    if variant:
        return f"{row.get('source')}/{variant}"
    return str(row.get("source"))


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("source")),
        str(row.get("dataset")),
        str(row.get("variant") or ""),
        str(row.get("classKey")),
    )


def find_report_files(input_dir: Path, filename: str) -> list[Path]:
    return sorted(input_dir.glob(f"*/*/{filename}")) + sorted(
        input_dir.glob(f"*/*/*/{filename}")
    )


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_distribution_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in find_report_files(input_dir, "distribution.csv"):
        for row in read_csv_rows(path):
            row["_file"] = str(path)
            row["_runLabel"] = run_label(row)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def valid_log_ratio(row: dict) -> float | None:
    reference_mean = distribution_value(row, "withinReferenceMean", "baselineMean")
    comparison_mean = distribution_value(row, "withinComparisonMean", "candidateMean")
    ratio = distribution_ratio(row)
    if (
        reference_mean is None
        or comparison_mean is None
        or ratio is None
        or reference_mean <= EPSILON
        or comparison_mean <= EPSILON
        or ratio <= EPSILON
    ):
        return None
    return abs(math.log(ratio))


def valid_normalized_w(row: dict) -> float | None:
    value = to_float(row.get("normalizedWassersteinDistance"))
    if value is not None and value >= 0:
        return value
    wasserstein = to_float(row.get("wassersteinDistance"))
    reference_sd = distribution_value(row, "withinReferenceSd", "baselineSd")
    if wasserstein is None or reference_sd is None or reference_sd <= EPSILON:
        return None
    return wasserstein / reference_sd


def valid_ks(row: dict) -> float | None:
    value = to_float(row.get("ksStatistic"))
    if value is None or value < 0 or value > 1:
        return None
    return value


def mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return statistics.fmean(finite)


def build_overview_tables(distribution_rows: list[dict], output_dir: Path) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in distribution_rows:
        grouped[(row["_runLabel"], row["dataset"], row.get("variant") or "")].append(row)

    overview_rows = []
    for (label, dataset, variant), rows in sorted(grouped.items()):
        by_family = {}
        for family, metrics in METRIC_FAMILIES.items():
            metric_set = set(metrics)
            by_family[f"{family}LogRatioScore"] = mean_or_none(
                valid_log_ratio(row) for row in rows if row["metric"] in metric_set
            )
            by_family[f"{family}NormalizedWasserstein"] = mean_or_none(
                valid_normalized_w(row) for row in rows if row["metric"] in metric_set
            )
            by_family[f"{family}KsStatistic"] = mean_or_none(
                valid_ks(row) for row in rows if row["metric"] in metric_set
            )
        overview_rows.append(
            {
                "sourceLabel": label,
                "dataset": dataset,
                "variant": variant,
                **by_family,
            }
        )

    metric_rows = []
    grouped_metric: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in distribution_rows:
        grouped_metric[(row["_runLabel"], row["dataset"], row.get("variant") or "", row["metric"])].append(row)
    for (label, dataset, variant, metric), rows in sorted(grouped_metric.items()):
        metric_rows.append(
            {
                "sourceLabel": label,
                "dataset": dataset,
                "variant": variant,
                "metric": metric,
                "logRatioScore": mean_or_none(valid_log_ratio(row) for row in rows),
                "normalizedWasserstein": mean_or_none(valid_normalized_w(row) for row in rows),
                "ksStatistic": mean_or_none(valid_ks(row) for row in rows),
                "withinComparisonToReferenceMeanRatio": mean_or_none(
                    distribution_ratio(row) for row in rows
                ),
                "withinComparisonToReferenceSdRatio": mean_or_none(
                    first_float(
                        row,
                        "withinComparisonToReferenceSdRatio",
                        "withinCandidateToBaselineSdRatio",
                        "sdRatio",
                    )
                    for row in rows
                ),
            }
        )

    write_csv(
        output_dir / "tables" / "source_dataset_scores.csv",
        overview_rows,
        [
            "sourceLabel",
            "dataset",
            "variant",
            "coreLogRatioScore",
            "coreNormalizedWasserstein",
            "coreKsStatistic",
            "shapeLogRatioScore",
            "timingLogRatioScore",
            "movementLogRatioScore",
            "strokeLogRatioScore",
            "dtwLogRatioScore",
        ],
    )
    write_csv(
        output_dir / "tables" / "source_dataset_metric_scores.csv",
        metric_rows,
        [
            "sourceLabel",
            "dataset",
            "variant",
            "metric",
            "logRatioScore",
            "normalizedWasserstein",
            "ksStatistic",
            "withinComparisonToReferenceMeanRatio",
            "withinComparisonToReferenceSdRatio",
        ],
    )
    return overview_rows, metric_rows


def matrix_from_rows(
    rows: list[dict],
    row_field: str,
    col_field: str,
    value_field: str,
) -> tuple[list[str], list[str], np.ndarray]:
    row_labels = sorted({str(row[row_field]) for row in rows if row.get(value_field) not in (None, "")})
    col_labels = sorted({str(row[col_field]) for row in rows if row.get(value_field) not in (None, "")})
    data = np.full((len(row_labels), len(col_labels)), np.nan)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            buckets[(str(row[row_field]), str(row[col_field]))].append(value_float)
    row_index = {label: idx for idx, label in enumerate(row_labels)}
    col_index = {label: idx for idx, label in enumerate(col_labels)}
    for key, values in buckets.items():
        data[row_index[key[0]], col_index[key[1]]] = statistics.fmean(values)
    return row_labels, col_labels, data


def plot_heatmap(
    rows: list[dict],
    row_field: str,
    col_field: str,
    value_field: str,
    title: str,
    path: Path,
    cmap: str = "viridis_r",
    value_label: str = "lower is better",
) -> None:
    row_labels, col_labels, data = matrix_from_rows(rows, row_field, col_field, value_field)
    if data.size == 0:
        return
    ensure_dir(path.parent)
    fig_w = max(8, 0.65 * len(col_labels) + 2)
    fig_h = max(4.5, 0.48 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_label)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if math.isfinite(data[i, j]) and data.shape[0] * data.shape[1] <= 140:
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overview(overview_rows: list[dict], output_dir: Path) -> None:
    plot_heatmap(
        overview_rows,
        "sourceLabel",
        "dataset",
        "coreLogRatioScore",
        "Core metric variability ratio by source and dataset",
        output_dir / "heatmaps" / "source_dataset_core_log_ratio.png",
        value_label="mean abs log(within-comparison / within-reference)",
    )
    plot_heatmap(
        overview_rows,
        "sourceLabel",
        "dataset",
        "coreNormalizedWasserstein",
        "Core metric reference-vs-between-group Wasserstein by source and dataset",
        output_dir / "heatmaps" / "source_dataset_core_normalized_wasserstein.png",
        value_label="Wasserstein / within-reference SD",
    )
    plot_heatmap(
        overview_rows,
        "sourceLabel",
        "dataset",
        "coreKsStatistic",
        "Core metric KS statistic by source and dataset",
        output_dir / "heatmaps" / "source_dataset_core_ks.png",
        value_label="KS statistic",
    )

    labels = sorted({row["sourceLabel"] for row in overview_rows})
    source_scores = []
    for label in labels:
        score = mean_or_none(
            row.get("coreLogRatioScore") for row in overview_rows if row["sourceLabel"] == label
        )
        if score is not None:
            source_scores.append((label, score))
    source_scores.sort(key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(source_scores) + 1)))
    ax.barh([item[0] for item in source_scores], [item[1] for item in source_scores], color="#4477aa")
    ax.invert_yaxis()
    ax.set_xlabel("Mean core abs log variability ratio, lower is better")
    ax.set_title("Overall source ranking across datasets")
    fig.tight_layout()
    ensure_dir(output_dir / "overview")
    fig.savefig(output_dir / "overview" / "overall_source_ranking_core.png", dpi=180)
    plt.close(fig)

    plot_grouped_bar(
        overview_rows,
        "coreLogRatioScore",
        "Core abs log variability ratio, lower is better",
        "Per-dataset comparison-group ranking: core metrics",
        output_dir / "grouped_bars" / "families" / "per_dataset_core_score_by_source.png",
    )


def plot_grouped_bar(
    rows: list[dict],
    value_field: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    labels = sorted({row["sourceLabel"] for row in rows})
    if not datasets or not labels:
        return
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(max(11, len(datasets) * 0.9), 6))
    x = np.arange(len(datasets))
    for idx, label in enumerate(labels):
        values = []
        for dataset in datasets:
            values.append(
                mean_or_none(
                    row.get(value_field)
                    for row in rows
                    if row["dataset"] == dataset and row["sourceLabel"] == label
                )
            )
        y = [np.nan if value is None else value for value in values]
        ax.bar(x + (idx - (len(labels) - 1) / 2) * width, y, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_bars_for_all_scores(
    overview_rows: list[dict],
    metric_rows: list[dict],
    output_dir: Path,
) -> None:
    for family in METRIC_FAMILIES:
        plot_grouped_bar(
            overview_rows,
            f"{family}LogRatioScore",
            f"{family} abs log variability ratio, lower is better",
            f"Per-dataset comparison-group ranking: {family} metrics",
            output_dir
            / "grouped_bars"
            / "families"
            / f"per_dataset_{safe_name(family)}_score_by_source.png",
        )

    metrics = sorted({row["metric"] for row in metric_rows})
    for metric in metrics:
        rows = [
            {
                "sourceLabel": row["sourceLabel"],
                "dataset": row["dataset"],
                "logRatioScore": row.get("logRatioScore"),
            }
            for row in metric_rows
            if row["metric"] == metric
        ]
        plot_grouped_bar(
            rows,
            "logRatioScore",
            f"{metric} abs log variability ratio, lower is better",
            f"Per-dataset comparison-group ranking: {metric}",
            output_dir
            / "grouped_bars"
            / "metrics"
            / f"per_dataset_{safe_name(metric)}_score_by_source.png",
        )


def plot_metric_family_breakdown(overview_rows: list[dict], output_dir: Path) -> None:
    family_rows = []
    for label in sorted({row["sourceLabel"] for row in overview_rows}):
        family_row = {"sourceLabel": label}
        for family in METRIC_FAMILIES:
            family_row[family] = mean_or_none(
                row.get(f"{family}LogRatioScore")
                for row in overview_rows
                if row["sourceLabel"] == label
            )
        family_rows.append(family_row)
    long_rows = [
        {"sourceLabel": row["sourceLabel"], "family": family, "score": row.get(family)}
        for row in family_rows
        for family in METRIC_FAMILIES
    ]
    plot_heatmap(
        long_rows,
        "sourceLabel",
        "family",
        "score",
        "Metric-family variability ratio by source",
        output_dir / "metric_families" / "source_metric_family_log_ratio.png",
        value_label="mean abs log variability ratio",
    )


def plot_metric_scatters(distribution_rows: list[dict], output_dir: Path) -> None:
    for metric in CORE_METRICS + MOVEMENT_METRICS:
        xs = []
        ys = []
        colors = []
        labels = sorted({row["_runLabel"] for row in distribution_rows})
        label_index = {label: idx for idx, label in enumerate(labels)}
        for row in distribution_rows:
            if row["metric"] != metric:
                continue
            x = distribution_value(row, "withinReferenceMean", "baselineMean")
            y = distribution_value(row, "betweenGroupsMean", "candidateMean")
            if x is None or y is None or x < 0 or y < 0:
                continue
            xs.append(x)
            ys.append(y)
            colors.append(label_index[row["_runLabel"]])
        if not xs:
            continue
        max_value = max(max(xs), max(ys))
        fig, ax = plt.subplots(figsize=(6.5, 6))
        scatter = ax.scatter(xs, ys, c=colors, cmap="tab10", alpha=0.42, s=12)
        ax.plot([0, max_value], [0, max_value], color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Within-reference mean")
        ax.set_ylabel("Between-group mean")
        ax.set_title(f"Within-reference vs between-group mean: {metric}")
        handles, _ = scatter.legend_elements(num=len(labels))
        if len(handles) == len(labels):
            ax.legend(handles, labels, title="source", fontsize=7, loc="best")
        fig.tight_layout()
        ensure_dir(output_dir / "scatter")
        fig.savefig(
            output_dir / "scatter" / f"within_reference_vs_between_groups_{safe_name(metric)}.png",
            dpi=180,
        )
        plt.close(fig)


def plot_ratio_boxplots(distribution_rows: list[dict], output_dir: Path) -> None:
    rows_by_label: dict[str, list[float]] = defaultdict(list)
    for row in distribution_rows:
        if row["metric"] not in CORE_METRICS:
            continue
        value = distribution_ratio(row)
        reference = distribution_value(row, "withinReferenceMean", "baselineMean")
        comparison = distribution_value(row, "withinComparisonMean", "candidateMean")
        if value is None or reference is None or comparison is None:
            continue
        if reference <= EPSILON or comparison <= EPSILON or value <= EPSILON:
            continue
        rows_by_label[row["_runLabel"]].append(value)
    labels = sorted(rows_by_label)
    data = [rows_by_label[label] for label in labels]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("withinComparisonMean / withinReferenceMean")
    ax.set_title("Core metric within-variability ratios by source")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    ensure_dir(output_dir / "boxplots")
    fig.savefig(
        output_dir / "boxplots" / "core_within_variability_ratio_by_source.png",
        dpi=180,
    )
    plt.close(fig)


def load_raw_run_rows(input_dir: Path, filename: str) -> dict[tuple[str, str, str], list[dict]]:
    by_run: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in find_report_files(input_dir, filename):
        rows = read_csv_rows(path)
        if not rows:
            continue
        first = rows[0]
        run_parts = str(first.get("runId", "")).split("/")
        if len(run_parts) >= 3:
            source, dataset, variant = run_parts[0], run_parts[1], "/".join(run_parts[2:])
        elif len(run_parts) == 2:
            source, dataset, variant = run_parts[0], run_parts[1], ""
        else:
            source, dataset, variant = (
                first.get("source", ""),
                first.get("dataset", ""),
                first.get("variant", "") or "",
            )
        key = (source, dataset, variant)
        by_run[key].extend(rows)
    return by_run


def choose_representative_cases(overview_rows: list[dict], count: int) -> list[dict]:
    valid = [
        row
        for row in overview_rows
        if row.get("coreLogRatioScore") is not None
        and math.isfinite(float(row["coreLogRatioScore"]))
    ]
    valid.sort(key=lambda row: float(row["coreLogRatioScore"]))
    if not valid:
        return []
    indexes = np.linspace(0, len(valid) - 1, num=min(count, len(valid)))
    selected = []
    seen = set()
    for index in indexes:
        row = valid[int(round(index))]
        key = (row["sourceLabel"], row["dataset"], row.get("variant") or "")
        if key not in seen:
            seen.add(key)
            selected.append(row)
    return selected


def class_scores_for_case(distribution_rows: list[dict], case: dict) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in distribution_rows:
        if row["_runLabel"] != case["sourceLabel"] or row["dataset"] != case["dataset"]:
            continue
        if (row.get("variant") or "") != (case.get("variant") or ""):
            continue
        if row["metric"] not in CORE_METRICS:
            continue
        score = valid_log_ratio(row)
        if score is not None:
            grouped[row["classKey"]].append(score)
    result = [
        (class_key, statistics.fmean(values))
        for class_key, values in grouped.items()
        if values
    ]
    result.sort(key=lambda item: item[1])
    return result


def plot_histogram_and_ecdf(
    baseline_values: list[float],
    candidate_values: list[float],
    title: str,
    histogram_path: Path,
    ecdf_path: Path,
) -> None:
    if len(baseline_values) < 2 or len(candidate_values) < 2:
        return
    combined = np.asarray(baseline_values + candidate_values, dtype=float)
    lo, hi = np.nanpercentile(combined, [1, 99])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo, hi = min(combined), max(combined)
    if lo == hi:
        return

    baseline_clip = [v for v in baseline_values if lo <= v <= hi]
    candidate_clip = [v for v in candidate_values if lo <= v <= hi]
    bins = np.linspace(lo, hi, 36)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(
        baseline_clip,
        bins=bins,
        alpha=0.55,
        density=True,
        label="reference summary baseline",
        color="#4477aa",
    )
    ax.hist(
        candidate_clip,
        bins=bins,
        alpha=0.55,
        density=True,
        label="comparison to reference summary",
        color="#cc6677",
    )
    ax.axvline(statistics.median(baseline_values), color="#225588", linestyle="--", linewidth=1)
    ax.axvline(statistics.median(candidate_values), color="#aa3355", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    ensure_dir(histogram_path.parent)
    fig.savefig(histogram_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for values, label, color in [
        (baseline_values, "reference summary baseline", "#4477aa"),
        (candidate_values, "comparison to reference summary", "#cc6677"),
    ]:
        sorted_values = np.sort(np.asarray(values, dtype=float))
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.step(sorted_values, y, where="post", label=label, color=color)
    ax.set_xlim(lo, hi)
    ax.set_title(title.replace("Histogram", "ECDF"))
    ax.set_ylabel("cumulative share")
    ax.legend()
    fig.tight_layout()
    ensure_dir(ecdf_path.parent)
    fig.savefig(ecdf_path, dpi=180)
    plt.close(fig)


def plot_representative_raw_distributions(
    input_dir: Path,
    output_dir: Path,
    distribution_rows: list[dict],
    overview_rows: list[dict],
    representative_count: int,
) -> list[dict]:
    cases = choose_representative_cases(overview_rows, representative_count)
    baseline_by_run = load_raw_run_rows(input_dir, "baseline.csv")
    candidate_by_run = load_raw_run_rows(input_dir, "pairwise.csv")
    case_rows = []
    for case in cases:
        source = case["sourceLabel"].split("/", 1)[0]
        variant = case.get("variant") or ""
        run_key = (source, case["dataset"], variant)
        baseline_rows = baseline_by_run.get(run_key, [])
        candidate_rows = candidate_by_run.get(run_key, [])
        if not baseline_rows or not candidate_rows:
            continue
        class_scores = class_scores_for_case(distribution_rows, case)
        if not class_scores:
            continue
        class_key = class_scores[len(class_scores) // 2][0]
        baseline_class_rows = [row for row in baseline_rows if row.get("classKey") == class_key]
        candidate_class_rows = [row for row in candidate_rows if row.get("classKey") == class_key]
        case_rows.append(
            {
                "sourceLabel": case["sourceLabel"],
                "dataset": case["dataset"],
                "variant": variant,
                "classKey": class_key,
                "coreLogRatioScore": case["coreLogRatioScore"],
            }
        )
        for metric in REPRESENTATIVE_METRICS:
            baseline_values = finite_values(baseline_class_rows, metric)
            candidate_values = finite_values(candidate_class_rows, metric)
            base = (
                f"{safe_name(case['sourceLabel'])}__{safe_name(case['dataset'])}"
                f"__{safe_name(class_key)}__{safe_name(metric)}"
            )
            title = (
                f"Histogram: {case['sourceLabel']} / {case['dataset']} / "
                f"{class_key} / {metric}"
            )
            plot_histogram_and_ecdf(
                baseline_values,
                candidate_values,
                title,
                output_dir / "histograms" / f"{base}.png",
                output_dir / "ecdfs" / f"{base}.png",
            )

    write_csv(
        output_dir / "tables" / "representative_raw_plot_cases.csv",
        case_rows,
        ["sourceLabel", "dataset", "variant", "classKey", "coreLogRatioScore"],
    )
    return case_rows


def compute_envelope_pass_rates(input_dir: Path, output_dir: Path) -> list[dict]:
    baseline_by_run = load_raw_run_rows(input_dir, "baseline.csv")
    candidate_by_run = load_raw_run_rows(input_dir, "pairwise.csv")
    rows = []
    for run_key, baseline_rows in sorted(baseline_by_run.items()):
        candidate_rows = candidate_by_run.get(run_key, [])
        if not candidate_rows:
            continue
        source, dataset, variant = run_key
        label = f"{source}/{variant}" if variant else source
        classes = sorted({row.get("classKey") for row in baseline_rows} | {row.get("classKey") for row in candidate_rows})
        for class_key in classes:
            baseline_class = [row for row in baseline_rows if row.get("classKey") == class_key]
            candidate_class = [row for row in candidate_rows if row.get("classKey") == class_key]
            for metric in CORE_METRICS + MOVEMENT_METRICS + STROKE_METRICS:
                baseline_values = finite_values(baseline_class, metric)
                candidate_values = finite_values(candidate_class, metric)
                if len(baseline_values) < 5 or len(candidate_values) == 0:
                    continue
                q05, q95 = np.quantile(np.asarray(baseline_values), [0.05, 0.95])
                passed = sum(1 for value in candidate_values if q05 <= value <= q95)
                rows.append(
                    {
                        "sourceLabel": label,
                        "source": source,
                        "dataset": dataset,
                        "variant": variant,
                        "classKey": class_key,
                        "metric": metric,
                        "referenceQ05": q05,
                        "referenceQ95": q95,
                        "comparisonN": len(candidate_values),
                        "passRate": passed / len(candidate_values),
                    }
                )

    write_csv(
        output_dir / "tables" / "reference_envelope_pass_rates.csv",
        rows,
        [
            "sourceLabel",
            "source",
            "dataset",
            "variant",
            "classKey",
            "metric",
            "referenceQ05",
            "referenceQ95",
            "comparisonN",
            "passRate",
        ],
    )

    summary_rows = []
    for label in sorted({row["sourceLabel"] for row in rows}):
        for dataset in sorted({row["dataset"] for row in rows if row["sourceLabel"] == label}):
            for family, metrics in METRIC_FAMILIES.items():
                values = [
                    row["passRate"]
                    for row in rows
                    if row["sourceLabel"] == label
                    and row["dataset"] == dataset
                    and row["metric"] in metrics
                ]
                value = mean_or_none(values)
                if value is not None:
                    summary_rows.append(
                        {
                            "sourceLabel": label,
                            "dataset": dataset,
                            "family": family,
                            "passRate": value,
                        }
                    )

    write_csv(
        output_dir / "tables" / "reference_envelope_pass_rate_summary.csv",
        summary_rows,
        ["sourceLabel", "dataset", "family", "passRate"],
    )

    core_rows = [row for row in summary_rows if row["family"] == "core"]
    plot_heatmap(
        core_rows,
        "sourceLabel",
        "dataset",
        "passRate",
        "Reference envelope pass rate, core metrics",
        output_dir / "envelope" / "source_dataset_core_pass_rate.png",
        cmap="viridis",
        value_label="share inside reference p05-p95; higher is better",
    )
    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    distribution_rows = load_distribution_rows(input_dir)
    if not distribution_rows:
        raise SystemExit(f"No distribution.csv files found under {input_dir}")

    overview_rows, metric_rows = build_overview_tables(distribution_rows, output_dir)
    plot_overview(overview_rows, output_dir)
    plot_grouped_bars_for_all_scores(overview_rows, metric_rows, output_dir)
    plot_metric_family_breakdown(overview_rows, output_dir)
    plot_metric_scatters(distribution_rows, output_dir)
    plot_ratio_boxplots(distribution_rows, output_dir)
    representative_cases = plot_representative_raw_distributions(
        input_dir,
        output_dir,
        distribution_rows,
        overview_rows,
        args.representative_count,
    )
    compute_envelope_pass_rates(input_dir, output_dir)

    summary = [
        {"item": "distribution_rows", "value": len(distribution_rows)},
        {"item": "source_dataset_rows", "value": len(overview_rows)},
        {"item": "representative_cases", "value": len(representative_cases)},
    ]
    write_csv(output_dir / "tables" / "generation_summary.csv", summary, ["item", "value"])
    print(f"Wrote evaluation plots and tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
