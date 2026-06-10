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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1e-12
HEATMAP_LOG_RANGE_THRESHOLD = 100.0
HEATMAP_SCALE_ROWS: list[dict] = []

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

PAIRWISE_HEATMAP_METRICS = (
    "shapeError",
    "dtwDistance",
    "velocityError",
)


@dataclass(frozen=True)
class AggregateStats:
    n: int
    mean: float | None
    sd: float | None
    ci_low: float | None
    ci_high: float | None


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


def infer_report_context(input_dir: Path, path: Path) -> dict[str, str]:
    parts = path.relative_to(input_dir).parts
    context = {
        "source": parts[0] if len(parts) >= 3 else "",
        "dataset": parts[1] if len(parts) >= 3 else "",
        "variant": "",
    }
    if len(parts) >= 4 and parts[2] not in {"classes", "combined"}:
        context["variant"] = parts[2]
    return context


def load_distribution_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in find_report_files(input_dir, "distribution.csv"):
        context = infer_report_context(input_dir, path)
        for row in read_csv_rows(path):
            for key, value in context.items():
                row.setdefault(key, value)
            if not row.get("metric") and row.get("gestureMetric"):
                row["metric"] = row["gestureMetric"]
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


def aggregate_stats(values: Iterable[float | None]) -> AggregateStats:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    n = len(finite)
    if n == 0:
        return AggregateStats(n=0, mean=None, sd=None, ci_low=None, ci_high=None)
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite) if n > 1 else 0.0
    if n <= 1 or sd == 0:
        return AggregateStats(n=n, mean=mean, sd=sd, ci_low=mean, ci_high=mean)
    margin = 1.96 * sd / math.sqrt(n)
    return AggregateStats(n=n, mean=mean, sd=sd, ci_low=mean - margin, ci_high=mean + margin)


def ci_half_width(stats: AggregateStats) -> float | None:
    if stats.mean is None or stats.ci_low is None or stats.ci_high is None:
        return None
    return max(stats.mean - stats.ci_low, stats.ci_high - stats.mean, 0.0)


def compact_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def mean_std_label(stats: AggregateStats) -> str:
    if stats.mean is None:
        return "n/a"
    return f"{compact_number(stats.mean)} +/- {compact_number(stats.sd or 0.0)}"


def ci_label(stats: AggregateStats) -> str:
    if stats.ci_low is None or stats.ci_high is None:
        return "95% CI n/a"
    return f"95% CI {compact_number(stats.ci_low)}-{compact_number(stats.ci_high)}"


def stats_fields(prefix: str, stats: AggregateStats) -> dict[str, float | int | None]:
    return {
        prefix: stats.mean,
        f"{prefix}N": stats.n,
        f"{prefix}Sd": stats.sd,
        f"{prefix}Ci95Low": stats.ci_low,
        f"{prefix}Ci95High": stats.ci_high,
    }


def quantile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "n": 0,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "iqr": None,
        }
    array = np.asarray(finite, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "n": len(finite),
        "min": float(np.min(array)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(array)),
        "iqr": float(q3 - q1),
    }


def iqr_label(summary: dict[str, float | int | None]) -> str:
    if summary["median"] is None or summary["q1"] is None or summary["q3"] is None:
        return "median n/a"
    return "%s [%s-%s]" % (
        compact_number(float(summary["median"])),
        compact_number(float(summary["q1"])),
        compact_number(float(summary["q3"])),
    )


def tukey_upper_whisker(values: Iterable[float]) -> float | None:
    summary = quantile_summary(values)
    q1 = summary["q1"]
    q3 = summary["q3"]
    iqr = summary["iqr"]
    if q1 is None or q3 is None or iqr is None:
        return None
    return float(q3) + 1.5 * float(iqr)


def build_overview_tables(distribution_rows: list[dict], output_dir: Path) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in distribution_rows:
        grouped[(row["_runLabel"], row["dataset"], row.get("variant") or "")].append(row)

    overview_rows = []
    for (label, dataset, variant), rows in sorted(grouped.items()):
        by_family = {}
        for family, metrics in METRIC_FAMILIES.items():
            metric_set = set(metrics)
            family_rows = [row for row in rows if row["metric"] in metric_set]
            by_family.update(
                stats_fields(
                    f"{family}LogRatioScore",
                    aggregate_stats(valid_log_ratio(row) for row in family_rows),
                )
            )
            by_family.update(
                stats_fields(
                    f"{family}NormalizedWasserstein",
                    aggregate_stats(valid_normalized_w(row) for row in family_rows),
                )
            )
            by_family.update(
                stats_fields(
                    f"{family}KsStatistic",
                    aggregate_stats(valid_ks(row) for row in family_rows),
                )
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
                **stats_fields(
                    "logRatioScore",
                    aggregate_stats(valid_log_ratio(row) for row in rows),
                ),
                **stats_fields(
                    "normalizedWasserstein",
                    aggregate_stats(valid_normalized_w(row) for row in rows),
                ),
                **stats_fields("ksStatistic", aggregate_stats(valid_ks(row) for row in rows)),
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

    overview_columns = ["sourceLabel", "dataset", "variant"]
    for family in METRIC_FAMILIES:
        for score_name in ("LogRatioScore", "NormalizedWasserstein", "KsStatistic"):
            prefix = f"{family}{score_name}"
            overview_columns.extend(
                [
                    prefix,
                    f"{prefix}N",
                    f"{prefix}Sd",
                    f"{prefix}Ci95Low",
                    f"{prefix}Ci95High",
                ]
            )

    write_csv(
        output_dir / "tables" / "source_dataset_scores.csv",
        overview_rows,
        overview_columns,
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
            "logRatioScoreN",
            "logRatioScoreSd",
            "logRatioScoreCi95Low",
            "logRatioScoreCi95High",
            "normalizedWasserstein",
            "normalizedWassersteinN",
            "normalizedWassersteinSd",
            "normalizedWassersteinCi95Low",
            "normalizedWassersteinCi95High",
            "ksStatistic",
            "ksStatisticN",
            "ksStatisticSd",
            "ksStatisticCi95Low",
            "ksStatisticCi95High",
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
) -> tuple[list[str], list[str], np.ndarray, dict[tuple[str, str], AggregateStats]]:
    row_labels = sorted({str(row[row_field]) for row in rows if row.get(value_field) not in (None, "")})
    col_labels = sorted({str(row[col_field]) for row in rows if row.get(value_field) not in (None, "")})
    data = np.full((len(row_labels), len(col_labels)), np.nan)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    explicit_stats: dict[tuple[str, str], AggregateStats] = {}
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            key = (str(row[row_field]), str(row[col_field]))
            buckets[key].append(value_float)
            sd = to_float(row.get(f"{value_field}Sd"))
            ci_low = to_float(row.get(f"{value_field}Ci95Low"))
            ci_high = to_float(row.get(f"{value_field}Ci95High"))
            n = to_float(row.get(f"{value_field}N"))
            if sd is not None or ci_low is not None or ci_high is not None:
                explicit_stats[key] = AggregateStats(
                    n=int(n) if n is not None else 1,
                    mean=value_float,
                    sd=sd,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
    row_index = {label: idx for idx, label in enumerate(row_labels)}
    col_index = {label: idx for idx, label in enumerate(col_labels)}
    stats_by_key = {}
    for key, values in buckets.items():
        if len(values) == 1 and key in explicit_stats:
            stats_by_key[key] = explicit_stats[key]
        else:
            stats_by_key[key] = aggregate_stats(values)
    for key, values in buckets.items():
        data[row_index[key[0]], col_index[key[1]]] = statistics.fmean(values)
    return row_labels, col_labels, data, stats_by_key


def finite_matrix_values(data: np.ndarray) -> np.ndarray:
    return data[np.isfinite(data)]


def heatmap_color_scale(data: np.ndarray) -> tuple[mcolors.Normalize | None, str, str]:
    values = finite_matrix_values(data)
    if values.size == 0:
        return None, "linear", "linear color scale"

    max_value = float(np.max(values))
    positive_values = values[values > EPSILON]
    if positive_values.size == 0 or max_value <= 0:
        return None, "linear", "linear color scale"

    min_value = float(np.min(values))
    min_positive = float(np.min(positive_values))
    dynamic_range = max_value / min_positive
    if dynamic_range < HEATMAP_LOG_RANGE_THRESHOLD:
        return None, "linear", "linear color scale"

    if min_value > 0:
        return (
            mcolors.LogNorm(vmin=min_positive, vmax=max_value),
            "log",
            "log10 color scale",
        )

    return (
        mcolors.SymLogNorm(
            linthresh=min_positive,
            linscale=1.0,
            vmin=min_value,
            vmax=max_value,
            base=10,
        ),
        "symlog",
        "symmetric log10 color scale; linear near zero",
    )


def record_heatmap_scale(
    path: Path,
    value_field: str,
    scale_key: str,
    scale_label: str,
    data: np.ndarray,
) -> None:
    values = finite_matrix_values(data)
    if values.size == 0:
        return
    positive_values = values[values > EPSILON]
    min_positive = (
        float(np.min(positive_values))
        if positive_values.size > 0
        else ""
    )
    dynamic_range = (
        float(np.max(values) / min_positive)
        if isinstance(min_positive, float) and min_positive > 0
        else ""
    )
    HEATMAP_SCALE_ROWS.append(
        {
            "plot": str(path),
            "valueField": value_field,
            "colorScale": scale_key,
            "colorScaleDescription": scale_label,
            "finiteCellCount": int(values.size),
            "min": float(np.min(values)),
            "minPositive": min_positive,
            "max": float(np.max(values)),
            "dynamicRange": dynamic_range,
        }
    )


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
    row_labels, col_labels, data, stats_by_key = matrix_from_rows(
        rows, row_field, col_field, value_field
    )
    if data.size == 0:
        return
    ensure_dir(path.parent)
    fig_w = max(8, 0.65 * len(col_labels) + 2)
    fig_h = max(4.5, 0.48 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(data)
    norm, scale_key, scale_label = heatmap_color_scale(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"{title}\nColor scale: {scale_label}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{value_label} ({scale_label})")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if math.isfinite(data[i, j]) and data.shape[0] * data.shape[1] <= 140:
                stats = stats_by_key.get((row_labels[i], col_labels[j]))
                if stats is not None and stats.n > 1 and stats.sd is not None:
                    ax.text(j, i - 0.09, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
                    ax.text(
                        j,
                        i + 0.12,
                        f"+/-{compact_number(stats.sd or 0.0)}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                    )
                else:
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    record_heatmap_scale(path, value_field, scale_key, scale_label, data)


def pair_keys_from_combined_key(pair_key: object) -> tuple[str, str] | None:
    if pair_key in (None, ""):
        return None
    text = str(pair_key)
    if "::" not in text:
        return None
    left, right = text.split("::", 1)
    if not left or not right:
        return None
    return left, right


def pairwise_matrix_from_rows(
    rows: list[dict],
    metric: str,
    left_field: str,
    right_field: str,
    symmetric: bool,
    neutral_diagonal: bool = True,
) -> tuple[list[str], list[str], np.ndarray]:
    metric_rows = [row for row in rows if row.get("metric") in (None, "", metric)]
    row_labels = sorted(
        {
            str(row[left_field])
            for row in metric_rows
            if row.get(left_field) not in (None, "")
        }
    )
    col_labels = sorted(
        {
            str(row[right_field])
            for row in metric_rows
            if row.get(right_field) not in (None, "")
        }
    )
    if symmetric:
        labels = sorted(set(row_labels) | set(col_labels))
        row_labels = labels
        col_labels = labels

    data = np.full((len(row_labels), len(col_labels)), np.nan)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        left = row.get(left_field)
        right = row.get(right_field)
        value = row.get("value") if row.get("metric") == metric else row.get(metric)
        value_float = to_float(value)
        if left in (None, "") or right in (None, "") or value_float is None:
            continue
        if value_float < -EPSILON:
            continue
        left_key = str(left)
        right_key = str(right)
        buckets[(left_key, right_key)].append(max(0.0, value_float))
        if symmetric:
            buckets[(right_key, left_key)].append(max(0.0, value_float))

    row_index = {label: idx for idx, label in enumerate(row_labels)}
    col_index = {label: idx for idx, label in enumerate(col_labels)}
    for (left_key, right_key), values in buckets.items():
        if left_key in row_index and right_key in col_index:
            data[row_index[left_key], col_index[right_key]] = statistics.fmean(values)

    if neutral_diagonal and len(row_labels) == len(col_labels) and row_labels == col_labels:
        np.fill_diagonal(data, np.nan)
    return row_labels, col_labels, data


def pairwise_matrix_from_combined_pair_keys(
    rows: list[dict],
    metric: str,
) -> tuple[list[str], list[str], np.ndarray]:
    expanded_rows = []
    for row in rows:
        keys = pair_keys_from_combined_key(row.get("pairKey"))
        if keys is None:
            continue
        left, right = keys
        expanded_rows.append({**row, "_leftKey": left, "_rightKey": right})
    return pairwise_matrix_from_rows(
        expanded_rows,
        metric,
        "_leftKey",
        "_rightKey",
        symmetric=True,
    )


def plot_pairwise_matrix_heatmap(
    row_labels: list[str],
    col_labels: list[str],
    data: np.ndarray,
    title: str,
    path: Path,
    value_label: str,
    scale_value_field: str,
) -> bool:
    if data.size == 0 or not np.isfinite(data).any():
        return False

    ensure_dir(path.parent)
    fig_w = max(7.5, 0.5 * len(col_labels) + 2.4)
    fig_h = max(5.2, 0.45 * len(row_labels) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.colormaps["viridis_r"].copy()
    cmap.set_bad(color="#f2f2f2")
    norm, scale_key, scale_label = heatmap_color_scale(data)
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("comparison gesture")
    ax.set_ylabel("reference gesture")
    ax.set_title(f"{title}\nColor scale: {scale_label}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{value_label} ({scale_label})")

    if data.shape[0] * data.shape[1] <= 160:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if math.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    record_heatmap_scale(path, scale_value_field, scale_key, scale_label, data)
    return True


def raw_pairwise_files(input_dir: Path, filename: str) -> list[Path]:
    return sorted(input_dir.rglob(filename))


def case_key_from_row_and_path(input_dir: Path, path: Path, row: dict) -> tuple[str, str, str, str]:
    try:
        parts = path.parent.relative_to(input_dir).parts
    except ValueError:
        parts = ()
    source = str(row.get("source") or row.get("candidateSource") or (parts[0] if len(parts) > 0 else "report"))
    dataset = str(row.get("dataset") or row.get("datasetKey") or (parts[1] if len(parts) > 1 else "dataset"))
    variant = str(row.get("variant") or "")
    class_key = str(row.get("classKey") or (parts[-1] if len(parts) >= 4 and parts[-2] == "classes" else "."))
    return source, dataset, variant, class_key


def group_rows_by_case(input_dir: Path, filename: str) -> dict[tuple[str, str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for path in raw_pairwise_files(input_dir, filename):
        for row in read_csv_rows(path):
            grouped[case_key_from_row_and_path(input_dir, path, row)].append(row)
    return grouped


def plot_pairwise_distance_heatmaps(
    input_dir: Path,
    output_dir: Path,
    metrics: Iterable[str] = PAIRWISE_HEATMAP_METRICS,
) -> list[dict]:
    generated_rows = []
    within_by_case = group_rows_by_case(input_dir, "within_comparison.csv")
    raw_baseline_by_case = group_rows_by_case(input_dir, "raw_baseline_pairs.csv")
    raw_candidate_by_case = group_rows_by_case(input_dir, "raw_candidate_pairs.csv")

    for case_key, rows in sorted(within_by_case.items()):
        source, dataset, variant, class_key = case_key
        for metric in metrics:
            labels, _, data = pairwise_matrix_from_combined_pair_keys(rows, metric)
            path = (
                output_dir
                / "pairwise_heatmaps"
                / safe_name(source)
                / safe_name(dataset)
                / safe_name(variant or "default")
                / safe_name(class_key)
                / f"within_comparison_{safe_name(metric)}.png"
            )
            if plot_pairwise_matrix_heatmap(
                labels,
                labels,
                data,
                f"Within-comparison gesture distances: {source} / {dataset} / {class_key} / {metric}",
                path,
                f"{metric}; lower is closer",
                f"withinComparison:{metric}",
            ):
                generated_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "variant": variant,
                        "classKey": class_key,
                        "metric": metric,
                        "pairType": "within-comparison",
                        "path": str(path),
                    }
                )

    for case_key, rows in sorted(raw_baseline_by_case.items()):
        source, dataset, variant, class_key = case_key
        for metric in metrics:
            labels, _, data = pairwise_matrix_from_rows(
                rows,
                metric,
                "referenceKey",
                "candidateKey",
                symmetric=False,
            )
            path = (
                output_dir
                / "pairwise_heatmaps"
                / safe_name(source)
                / safe_name(dataset)
                / safe_name(variant or "default")
                / safe_name(class_key)
                / f"within_reference_{safe_name(metric)}.png"
            )
            if plot_pairwise_matrix_heatmap(
                labels,
                labels,
                data,
                f"Within-reference gesture distances: {source} / {dataset} / {class_key} / {metric}",
                path,
                f"{metric}; lower is closer",
                f"withinReferenceRaw:{metric}",
            ):
                generated_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "variant": variant,
                        "classKey": class_key,
                        "metric": metric,
                        "pairType": "within-reference-raw",
                        "path": str(path),
                    }
                )

    for case_key, rows in sorted(raw_candidate_by_case.items()):
        source, dataset, variant, class_key = case_key
        for metric in metrics:
            row_labels, col_labels, data = pairwise_matrix_from_rows(
                rows,
                metric,
                "referenceKey",
                "candidateKey",
                symmetric=False,
                neutral_diagonal=False,
            )
            path = (
                output_dir
                / "pairwise_heatmaps"
                / safe_name(source)
                / safe_name(dataset)
                / safe_name(variant or "default")
                / safe_name(class_key)
                / f"between_groups_{safe_name(metric)}.png"
            )
            if plot_pairwise_matrix_heatmap(
                row_labels,
                col_labels,
                data,
                f"Between-group gesture distances: {source} / {dataset} / {class_key} / {metric}",
                path,
                f"{metric}; lower is closer",
                f"betweenGroupsRaw:{metric}",
            ):
                generated_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "variant": variant,
                        "classKey": class_key,
                        "metric": metric,
                        "pairType": "between-groups-raw",
                        "path": str(path),
                    }
                )

    if generated_rows:
        write_csv(
            output_dir / "tables" / "pairwise_heatmaps.csv",
            generated_rows,
            ["source", "dataset", "variant", "classKey", "metric", "pairType", "path"],
        )
    return generated_rows


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
        stats = aggregate_stats(
            row.get("coreLogRatioScore") for row in overview_rows if row["sourceLabel"] == label
        )
        if stats.mean is not None:
            source_scores.append((label, stats))
    source_scores.sort(key=lambda item: item[1].mean or math.inf)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(source_scores) + 1)))
    bar_values = [item[1].mean for item in source_scores]
    bar_errors = [ci_half_width(item[1]) or 0.0 for item in source_scores]
    bars = ax.barh(
        [item[0] for item in source_scores],
        bar_values,
        xerr=bar_errors,
        color="#4477aa",
        error_kw={"elinewidth": 0.9, "capsize": 2.5, "alpha": 0.75},
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean core abs log variability ratio, lower is better")
    ax.set_title("Overall source ranking across datasets")
    label_x_values = []
    for bar, (_, stats) in zip(bars, source_scores):
        error_width = ci_half_width(stats) or 0.0
        label_x = (stats.mean or 0.0) + error_width
        label_x_values.append(label_x)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"  {mean_std_label(stats)}",
            va="center",
            fontsize=7,
        )
    max_label_x = max(label_x_values or [0.0])
    ax.set_xlim(right=max(ax.get_xlim()[1], max_label_x * 1.18 if max_label_x else 1.0))
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
    total_bars = len(datasets) * len(labels)
    for idx, label in enumerate(labels):
        stats_values = []
        for dataset in datasets:
            stats_values.append(
                aggregate_stats(
                    row.get(value_field)
                    for row in rows
                    if row["dataset"] == dataset and row["sourceLabel"] == label
                )
            )
        y = [np.nan if stats.mean is None else stats.mean for stats in stats_values]
        yerr = [0.0 if ci_half_width(stats) is None else ci_half_width(stats) for stats in stats_values]
        bars = ax.bar(
            x + (idx - (len(labels) - 1) / 2) * width,
            y,
            width,
            yerr=yerr,
            label=label,
            error_kw={"elinewidth": 0.8, "capsize": 2, "alpha": 0.75},
        )
        if total_bars <= 36:
            for bar, stats in zip(bars, stats_values):
                if stats.mean is None:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    mean_std_label(stats),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=0 if total_bars <= 12 else 90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2)
    ax.margins(y=0.18)
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
        xerrs = []
        yerrs = []
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
            xerrs.append(
                max(
                    0.0,
                    distribution_value(row, "withinReferenceSd", "baselineSd") or 0.0,
                )
            )
            yerrs.append(
                max(
                    0.0,
                    distribution_value(row, "betweenGroupsSd", "candidateSd") or 0.0,
                )
            )
            colors.append(label_index[row["_runLabel"]])
        if not xs:
            continue
        max_value = max(max(xs), max(ys))
        fig, ax = plt.subplots(figsize=(6.5, 6))
        scatter = ax.scatter(xs, ys, c=colors, cmap="tab10", alpha=0.42, s=12)
        if len(xs) <= 250 and (any(xerrs) or any(yerrs)):
            ax.errorbar(
                xs,
                ys,
                xerr=xerrs,
                yerr=yerrs,
                fmt="none",
                ecolor="#555555",
                elinewidth=0.45,
                alpha=0.18,
                capsize=0,
                zorder=0,
            )
        ax.plot([0, max_value], [0, max_value], color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Within-reference mean")
        ax.set_ylabel("Between-group mean")
        ratio_stats = aggregate_stats(
            distribution_ratio(row)
            for row in distribution_rows
            if row["metric"] == metric and distribution_ratio(row) is not None
        )
        subtitle = f"variability ratio {mean_std_label(ratio_stats)}; {ci_label(ratio_stats)}"
        ax.set_title(f"Within-reference vs between-group mean: {metric}\n{subtitle}", fontsize=10)
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
    iqr_rows = []
    whisker_caps = []
    outlier_rows = []
    for label in labels:
        summary = quantile_summary(rows_by_label[label])
        whisker_cap = tukey_upper_whisker(rows_by_label[label])
        if whisker_cap is not None:
            whisker_caps.append(whisker_cap)
            outlier_count = sum(value > whisker_cap for value in rows_by_label[label])
            outlier_rows.append(
                {
                    "sourceLabel": label,
                    "upperWhisker": whisker_cap,
                    "outlierCount": outlier_count,
                    "maxOutlier": max((value for value in rows_by_label[label] if value > whisker_cap), default=None),
                }
            )
        iqr_rows.append(
            {
                "sourceLabel": label,
                "n": summary["n"],
                "min": summary["min"],
                "q1": summary["q1"],
                "median": summary["median"],
                "q3": summary["q3"],
                "max": summary["max"],
                "iqr": summary["iqr"],
            }
        )
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False, whis=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    if whisker_caps:
        main_ylim = max(max(whisker_caps) * 1.08, 1.5)
        ax.set_ylim(0.0, main_ylim)
    ax.set_ylabel("withinComparisonMean / withinReferenceMean")
    ax.set_title("Core metric within-variability ratios by source\nMain range only; outliers moved to separate view")
    ax.tick_params(axis="x", rotation=35)
    summary_lines = [
        f"{label}: {iqr_label(quantile_summary(rows_by_label[label]))}"
        for label in labels[:6]
    ]
    if len(labels) > 6:
        summary_lines.append(f"+ {len(labels) - 6} more")
    outlier_total = sum(int(row["outlierCount"]) for row in outlier_rows)
    if outlier_total:
        summary_lines.append(f"{outlier_total} outliers shown separately")
    if summary_lines:
        ax.text(
            0.99,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.82, "pad": 3},
        )
    fig.tight_layout()
    ensure_dir(output_dir / "boxplots")
    fig.savefig(
        output_dir / "boxplots" / "core_within_variability_ratio_by_source.png",
        dpi=180,
    )
    plt.close(fig)

    if whisker_caps:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5.5))
        positions = np.arange(1, len(labels) + 1)
        has_outliers = False
        outlier_label_added = False
        for position, label, whisker_cap in zip(positions, labels, whisker_caps):
            outliers = sorted(value for value in rows_by_label[label] if value > whisker_cap)
            if not outliers:
                continue
            has_outliers = True
            xs = np.full(len(outliers), position, dtype=float)
            ax.scatter(
                xs,
                outliers,
                color="#222222",
                alpha=0.72,
                s=18,
                label="Outlier ratios" if not outlier_label_added else None,
            )
            outlier_label_added = True
        if has_outliers:
            ax.set_yscale("log")
            ax.set_ylabel("withinComparisonMean / withinReferenceMean (outliers only)")
            ax.set_title("Core metric within-variability outliers by source\nValues above each source's 1.5x IQR whisker")
            ax.set_xticks(positions, labels, rotation=35)
            if outlier_label_added:
                ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(
                output_dir / "boxplots" / "core_within_variability_ratio_outliers_by_source.png",
                dpi=180,
            )
        plt.close(fig)

    write_csv(
        output_dir / "tables" / "core_within_variability_ratio_iqr.csv",
        iqr_rows,
        ["sourceLabel", "n", "min", "q1", "median", "q3", "max", "iqr"],
    )


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
    baseline_stats = aggregate_stats(baseline_values)
    candidate_stats = aggregate_stats(candidate_values)
    stat_text = "\n".join(
        item
        for item in [
            f"human {mean_std_label(baseline_stats)}; {ci_label(baseline_stats)}",
            f"generated {mean_std_label(candidate_stats)}; {ci_label(candidate_stats)}",
        ]
        if item
    )

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
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("density")
    ax.text(
        0.99,
        0.98,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.82, "pad": 3},
    )
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
    ax.set_title(title.replace("Histogram", "ECDF"), fontsize=10)
    ax.set_ylabel("cumulative share")
    ax.text(
        0.99,
        0.02,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.82, "pad": 3},
    )
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
    HEATMAP_SCALE_ROWS.clear()

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
    pairwise_heatmaps = plot_pairwise_distance_heatmaps(input_dir, output_dir)
    write_csv(
        output_dir / "tables" / "heatmap_color_scales.csv",
        HEATMAP_SCALE_ROWS,
        [
            "plot",
            "valueField",
            "colorScale",
            "colorScaleDescription",
            "finiteCellCount",
            "min",
            "minPositive",
            "max",
            "dynamicRange",
        ],
    )

    summary = [
        {"item": "distribution_rows", "value": len(distribution_rows)},
        {"item": "source_dataset_rows", "value": len(overview_rows)},
        {"item": "representative_cases", "value": len(representative_cases)},
        {"item": "pairwise_heatmaps", "value": len(pairwise_heatmaps)},
        {"item": "heatmap_color_scale_rows", "value": len(HEATMAP_SCALE_ROWS)},
    ]
    write_csv(output_dir / "tables" / "generation_summary.csv", summary, ["item", "value"])
    print(f"Wrote evaluation plots and tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
