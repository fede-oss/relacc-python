#!/usr/bin/env python3
"""Generate plots and derived tables for current RELACC evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1e-12
HEATMAP_LOG_RANGE_THRESHOLD = 100.0
NORMALITY_ALPHA = 0.05

CORE_METRICS = (
    "shapeError",
    "lengthError",
    "bendingError",
    "timeError",
    "velocityError",
    "dtwDistance",
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
MOVEMENT_METRICS = (
    "cornerSlowdown",
    "twoThirdsPowerLawR2",
    "highFrequencyRatio",
    "curvature",
)
STROKE_METRICS = (
    "strokeError",
    "strokeOrderError",
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
METRIC_FAMILIES = {
    "core": CORE_METRICS,
    "shape": SHAPE_METRICS,
    "timing": TIMING_METRICS,
    "movement": MOVEMENT_METRICS,
    "stroke": STROKE_METRICS,
    "dtw": DTW_METRICS,
}
ALL_METRICS = tuple(dict.fromkeys(sum((tuple(v) for v in METRIC_FAMILIES.values()), ())))
LOWER_IS_BETTER_METRICS = tuple(metric for metric in ALL_METRICS if metric != "twoThirdsPowerLawR2")
REPRESENTATIVE_METRICS = (
    "shapeError",
    "lengthError",
    "velocityError",
    "dtwDistance",
    "strokeLengthStd",
    "meanStrokeDuration",
)
HISTOGRAM_METRICS = (
    "shapeError",
    "dtwDistance",
    "velocityError",
    "strokeLengthStd",
    "meanStrokeDuration",
)
DISTANCE_METRICS = (
    "jensenShannonDivergence",
    "wassersteinDistance",
    "energyDistance",
    "ksStatistic",
    "totalVariationDistance",
)
DEFAULT_FORMATS = ("png",)
COMPONENT_FIELDS = (
    "summaryErrorScore",
    "variabilityScore",
    "distributionScore",
    "summaryDistributionScore",
)
COMPONENT_LABELS = {
    "summaryErrorScore": "summary error",
    "variabilityScore": "variability match",
    "distributionScore": "raw distribution",
    "summaryDistributionScore": "summary distribution",
}


@dataclass
class PlotRecord:
    path: Path
    title: str
    category: str
    description: str


PLOT_MANIFEST: list[PlotRecord] = []
HEATMAP_SCALE_ROWS: list[dict[str, object]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v2 plots from combined RELACC evaluation outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="report-output-eval-detailed-s24-20260608",
        help="Evaluation output directory containing combined/*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation-plots-detailed-s24",
        help="Directory for derived tables and plots.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Comma-separated output formats, for example png,pdf.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Raster plot DPI.")
    parser.add_argument("--top-classes", type=int, default=25)
    parser.add_argument("--max-histogram-classes", type=int, default=40)
    parser.add_argument(
        "--histogram-selection",
        choices=("curated", "top", "all"),
        default="curated",
        help="Use balanced curated examples, top divergent examples, or all available class/metric cases.",
    )
    parser.add_argument(
        "--histogram-page-size",
        type=int,
        default=24,
        help="Number of histogram cases per appendix overview page.",
    )
    parser.add_argument(
        "--histogram-cases-per-row",
        type=int,
        default=3,
        help="Number of class/metric cases per appendix page row. Each case uses two panels.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute derived ranking tables even when cached tables already exist.",
    )
    parser.add_argument(
        "--metric-family",
        choices=("core", "shape", "timing", "movement", "stroke", "dtw", "all"),
        default="all",
    )
    appendix = parser.add_mutually_exclusive_group()
    appendix.add_argument("--include-appendix", action="store_true", default=True)
    appendix.add_argument("--skip-appendix", action="store_false", dest="include_appendix")
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


def to_float_allow_inf(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2g}"


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "q1": None, "median": None, "q3": None, "iqr": None, "mean": None}
    arr = np.asarray(finite, dtype=float)
    q1, mdn, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "n": int(arr.size),
        "q1": float(q1),
        "median": float(mdn),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mean": float(np.mean(arr)),
    }


def median_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(statistics.median(finite))


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return next(reader, [])


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if columns is None:
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def label_from_parts(source: object, variant: object) -> str:
    source_text = str(source or "unknown")
    variant_text = str(variant or "").strip()
    if not variant_text or variant_text == "root":
        return source_text
    return f"{source_text}/{variant_text}"


def row_label(row: dict[str, object]) -> str:
    return label_from_parts(row.get("source"), row.get("variant"))


def metric_family(metric: str) -> str:
    for family, metrics in METRIC_FAMILIES.items():
        if metric in metrics:
            return family
    return "other"


def selected_metrics(args: argparse.Namespace) -> tuple[str, ...]:
    if args.metric_family == "all":
        return LOWER_IS_BETTER_METRICS
    return tuple(metric for metric in METRIC_FAMILIES[args.metric_family] if metric != "twoThirdsPowerLawR2")


def combined_dir_for(input_dir: Path) -> Path:
    combined = input_dir / "combined"
    if combined.exists():
        return combined
    return input_dir


def get_csv_path(input_dir: Path, filename: str) -> Path:
    combined = combined_dir_for(input_dir)
    path = combined / filename
    if path.exists():
        return path
    fallback = input_dir / filename
    if fallback.exists():
        return fallback
    return path


def rank_values(values: dict[str, float | None]) -> dict[str, float | None]:
    finite = [(key, float(value)) for key, value in values.items() if value is not None and math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    ranks: dict[str, float | None] = {key: None for key in values}
    index = 0
    while index < len(finite):
        end = index + 1
        while end < len(finite) and math.isclose(finite[end][1], finite[index][1], rel_tol=1e-12, abs_tol=1e-12):
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for key, _ in finite[index:end]:
            ranks[key] = avg_rank
        index = end
    return ranks


def normalized_ranks(values: dict[str, float | None]) -> dict[str, float | None]:
    ranks = rank_values(values)
    finite_count = sum(1 for value in ranks.values() if value is not None)
    if finite_count <= 1:
        return {key: (0.0 if value is not None else None) for key, value in ranks.items()}
    return {
        key: ((float(value) - 1.0) / (finite_count - 1) if value is not None else None)
        for key, value in ranks.items()
    }


def aggregate_component_rows(
    values: dict[tuple[str, str, str, str], list[float]],
    value_name: str,
) -> list[dict[str, object]]:
    rows = []
    for (generator, dataset, metric, family), vals in sorted(values.items()):
        q = quantiles(vals)
        rows.append(
            {
                "generator": generator,
                "dataset": dataset,
                "metric": metric,
                "family": family,
                f"{value_name}N": q["n"],
                f"{value_name}Median": q["median"],
                f"{value_name}Q1": q["q1"],
                f"{value_name}Q3": q["q3"],
                f"{value_name}Iqr": q["iqr"],
                f"{value_name}Mean": q["mean"],
            }
        )
    return rows


def load_stats_medians(path: Path, metrics: set[str]) -> dict[tuple[str, str, str, str, str], float]:
    data: dict[tuple[str, str, str, str, str], float] = {}
    if not path.exists():
        return data
    for row in read_csv_rows(path):
        metric = str(row.get("metric") or "")
        if metric not in metrics:
            continue
        value = to_float(row.get("mdn"))
        if value is None or value < -EPSILON:
            continue
        source = str(row.get("source") or "unknown")
        dataset = str(row.get("dataset") or "unknown")
        variant = str(row.get("variant") or "root")
        class_key = str(row.get("classKey") or "unknown")
        data[(source, dataset, variant, class_key, metric)] = max(0.0, value)
    return data


def add_ratio_component(
    numerator: dict[tuple[str, str, str, str, str], float],
    denominator: dict[tuple[str, str, str, str, str], float],
) -> dict[tuple[str, str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    denominator_by_class_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_source, dataset, _variant, class_key, metric), value in denominator.items():
        denominator_by_class_metric[(dataset, class_key, metric)].append(value)
    for key, num in numerator.items():
        source, dataset, variant, _class_key, metric = key
        den = denominator.get(key)
        if den is None:
            den = median_or_none(denominator_by_class_metric.get((dataset, _class_key, metric), []))
        if den is None or den <= EPSILON or num < -EPSILON:
            continue
        generator = label_from_parts(source, variant)
        grouped[(generator, dataset, metric, metric_family(metric))].append(max(0.0, num) / den)
    return grouped


def add_abs_log_ratio_component(
    numerator: dict[tuple[str, str, str, str, str], float],
    denominator: dict[tuple[str, str, str, str, str], float],
) -> dict[tuple[str, str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    denominator_by_class_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_source, dataset, _variant, class_key, metric), value in denominator.items():
        denominator_by_class_metric[(dataset, class_key, metric)].append(value)
    for key, num in numerator.items():
        source, dataset, variant, _class_key, metric = key
        den = denominator.get(key)
        if den is None:
            den = median_or_none(denominator_by_class_metric.get((dataset, _class_key, metric), []))
        if den is None or den <= EPSILON or num <= EPSILON:
            continue
        generator = label_from_parts(source, variant)
        grouped[(generator, dataset, metric, metric_family(metric))].append(abs(math.log(num / den)))
    return grouped


def normalize_metric_rows(rows: Sequence[dict[str, object]], value_field: str, score_field: str) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        dataset = str(row.get("dataset") or "")
        metric = str(row.get("metric") or "")
        generator = str(row.get("generator") or "")
        buckets[(dataset, metric)][generator] = to_float(row.get(value_field))
    scores: dict[tuple[str, str, str], float | None] = {}
    for (dataset, metric), values in buckets.items():
        norm = normalized_ranks(values)
        for generator, value in norm.items():
            scores[(dataset, metric, generator)] = value
    return [
        {**row, score_field: scores.get((str(row.get("dataset") or ""), str(row.get("metric") or ""), str(row.get("generator") or "")))}
        for row in rows
    ]


def distribution_component_rows(
    path: Path,
    metrics: set[str],
    value_prefix: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str, str], list[float]], list[dict[str, object]]]:
    if not path.exists():
        return [], defaultdict(list), []
    per_group_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    instability_rows = []
    for row in read_csv_rows(path):
        metric = str(row.get("metric") or "")
        if metric not in metrics:
            continue
        generator = row_label(row)
        dataset = str(row.get("dataset") or "unknown")
        family = metric_family(metric)
        finite_distances = []
        infinite_count = 0
        for distance_metric in DISTANCE_METRICS:
            raw = to_float_allow_inf(row.get(distance_metric))
            if raw is None:
                continue
            if math.isfinite(raw) and raw >= 0:
                finite_distances.append(float(raw))
            elif math.isinf(raw):
                infinite_count += 1
        if finite_distances:
            per_group_values[(generator, dataset, metric, family)].append(float(statistics.median(finite_distances)))
        if infinite_count:
            instability_rows.append(
                {
                    "source": row.get("source", ""),
                    "dataset": dataset,
                    "variant": row.get("variant", ""),
                    "classKey": row.get("classKey", ""),
                    "metric": metric,
                    "infiniteDistanceCount": infinite_count,
                    "file": path.name,
                }
            )
    component_rows = aggregate_component_rows(per_group_values, value_prefix)
    component_rows = normalize_metric_rows(component_rows, f"{value_prefix}Median", f"{value_prefix}NormalizedRank")
    return component_rows, per_group_values, instability_rows


def build_ranking_tables(
    input_dir: Path,
    output_dir: Path,
    metrics: tuple[str, ...],
) -> dict[str, list[dict[str, object]]]:
    metric_set = set(metrics)
    combined = combined_dir_for(input_dir)
    baseline = load_stats_medians(combined / "baseline_stats.csv", metric_set)
    pairwise = load_stats_medians(combined / "stats.csv", metric_set)
    within_reference = load_stats_medians(combined / "within_reference_stats.csv", metric_set)
    within_comparison = load_stats_medians(combined / "within_comparison_stats.csv", metric_set)

    summary_values = add_ratio_component(pairwise, baseline)
    variability_values = add_abs_log_ratio_component(within_comparison, within_reference)
    summary_rows = normalize_metric_rows(
        aggregate_component_rows(summary_values, "summaryErrorRatio"),
        "summaryErrorRatioMedian",
        "summaryErrorNormalizedRank",
    )
    variability_rows = normalize_metric_rows(
        aggregate_component_rows(variability_values, "variabilityLogRatio"),
        "variabilityLogRatioMedian",
        "variabilityNormalizedRank",
    )
    distribution_rows, _distribution_values, dist_instability = distribution_component_rows(
        combined / "distribution.csv", metric_set, "distributionDistance"
    )
    summary_distribution_rows, _summary_distribution_values, summary_dist_instability = distribution_component_rows(
        combined / "summary_distribution.csv", metric_set, "summaryDistributionDistance"
    )

    metric_score_rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for rows, field in (
        (summary_rows, "summaryError"),
        (variability_rows, "variability"),
        (distribution_rows, "distribution"),
        (summary_distribution_rows, "summaryDistribution"),
    ):
        for row in rows:
            key = (str(row["generator"]), str(row["dataset"]), str(row["metric"]))
            target = metric_score_rows.setdefault(
                key,
                {
                    "generator": key[0],
                    "dataset": key[1],
                    "metric": key[2],
                    "family": row.get("family") or metric_family(key[2]),
                },
            )
            for col, value in row.items():
                if col not in ("generator", "dataset", "metric", "family"):
                    target[col] = value
            rank_col = {
                "summaryError": "summaryErrorNormalizedRank",
                "variability": "variabilityNormalizedRank",
                "distribution": "distributionDistanceNormalizedRank",
                "summaryDistribution": "summaryDistributionDistanceNormalizedRank",
            }[field]
            target[f"{field}Score"] = row.get(rank_col)

    metric_rows = []
    for row in metric_score_rows.values():
        components = [to_float(row.get(field)) for field in COMPONENT_FIELDS]
        finite = [value for value in components if value is not None]
        row["aggregateScore"] = float(statistics.fmean(finite)) if finite else None
        metric_rows.append(row)
    metric_rows.sort(key=lambda row: (str(row["dataset"]), str(row["metric"]), str(row["generator"])))

    dataset_buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    family_buckets: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    generator_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        dataset_buckets[(str(row["generator"]), str(row["dataset"]))].append(row)
        family_buckets[(str(row["generator"]), str(row["dataset"]), str(row["family"]))].append(row)
        generator_buckets[str(row["generator"])].append(row)

    def summarize_score_rows(rows: list[dict[str, object]], extra: dict[str, object]) -> dict[str, object]:
        out = dict(extra)
        for field in COMPONENT_FIELDS:
            values = [to_float(row.get(field)) for row in rows]
            out[field] = median_or_none(values)
        agg_values = [to_float(row.get("aggregateScore")) for row in rows]
        out["aggregateScore"] = median_or_none(agg_values)
        out["metricCount"] = len({row.get("metric") for row in rows})
        out["componentCellCount"] = sum(1 for row in rows for field in COMPONENT_FIELDS if to_float(row.get(field)) is not None)
        return out

    dataset_rows = [
        summarize_score_rows(rows, {"generator": generator, "dataset": dataset})
        for (generator, dataset), rows in sorted(dataset_buckets.items())
    ]
    family_rows = [
        summarize_score_rows(rows, {"generator": generator, "dataset": dataset, "family": family})
        for (generator, dataset, family), rows in sorted(family_buckets.items())
    ]
    overall_rows = [
        summarize_score_rows(rows, {"generator": generator})
        for generator, rows in sorted(generator_buckets.items())
    ]
    for rows, scope_fields in ((overall_rows, ()), (dataset_rows, ("dataset",)), (family_rows, ("dataset", "family"))):
        grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(field) for field in scope_fields)].append(row)
        for group_rows in grouped.values():
            ranks = rank_values({str(row["generator"]): to_float(row.get("aggregateScore")) for row in group_rows})
            for row in group_rows:
                row["rank"] = ranks.get(str(row["generator"]))
    overall_rows.sort(key=lambda row: (to_float(row.get("rank")) or math.inf, str(row["generator"])))
    dataset_rows.sort(key=lambda row: (str(row["dataset"]), to_float(row.get("rank")) or math.inf, str(row["generator"])))
    family_rows.sort(key=lambda row: (str(row["family"]), str(row["dataset"]), to_float(row.get("rank")) or math.inf, str(row["generator"])))

    tables_dir = ensure_dir(output_dir / "tables")
    write_csv(tables_dir / "generator_dataset_metric_scores.csv", metric_rows)
    write_csv(tables_dir / "generator_dataset_scores.csv", dataset_rows)
    write_csv(tables_dir / "metric_family_scores.csv", family_rows)
    write_csv(tables_dir / "generator_overall_scores.csv", overall_rows)
    write_csv(tables_dir / "distribution_infinite_values.csv", dist_instability + summary_dist_instability)
    return {
        "metric": metric_rows,
        "dataset": dataset_rows,
        "family": family_rows,
        "overall": overall_rows,
        "summary_component": summary_rows,
        "variability_component": variability_rows,
        "distribution_component": distribution_rows,
        "summary_distribution_component": summary_distribution_rows,
    }


def load_cached_ranking_tables(output_dir: Path) -> dict[str, list[dict[str, object]]] | None:
    table_files = {
        "metric": output_dir / "tables" / "generator_dataset_metric_scores.csv",
        "dataset": output_dir / "tables" / "generator_dataset_scores.csv",
        "family": output_dir / "tables" / "metric_family_scores.csv",
        "overall": output_dir / "tables" / "generator_overall_scores.csv",
    }
    if not all(path.exists() for path in table_files.values()):
        return None
    return {key: list(read_csv_rows(path)) for key, path in table_files.items()}


def generator_order(overall_rows: Sequence[dict[str, object]]) -> list[str]:
    return [str(row["generator"]) for row in sorted(overall_rows, key=lambda r: (to_float(r.get("rank")) or math.inf, str(r["generator"])))]


def dataset_order(dataset_rows: Sequence[dict[str, object]]) -> list[str]:
    return sorted({str(row["dataset"]) for row in dataset_rows})


def save_figure(fig: plt.Figure, path_base: Path, formats: Sequence[str], dpi: int, title: str, category: str, description: str) -> None:
    ensure_dir(path_base.parent)
    first_path = path_base.with_suffix(f".{formats[0]}")
    for fmt in formats:
        fig.savefig(path_base.with_suffix(f".{fmt}"), dpi=dpi if fmt.lower() != "pdf" else None)
    PLOT_MANIFEST.append(PlotRecord(first_path, title, category, description))
    plt.close(fig)


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
        return mcolors.LogNorm(vmin=min_positive, vmax=max_value), "log", "log10 color scale"
    return (
        mcolors.SymLogNorm(linthresh=min_positive, linscale=1.0, vmin=min_value, vmax=max_value, base=10),
        "symlog",
        "symmetric log10 color scale; linear near zero",
    )


def centered_ratio_norm(data: np.ndarray) -> mcolors.Normalize | None:
    values = finite_matrix_values(data)
    positive = values[values > EPSILON]
    if positive.size == 0:
        return None
    logs = np.log2(positive)
    extent = max(abs(float(np.min(logs))), abs(float(np.max(logs))), 0.25)
    return mcolors.TwoSlopeNorm(vmin=2 ** (-extent), vcenter=1.0, vmax=2 ** extent)


def centered_ratio_norm_clipped(data: np.ndarray, percentile: float = 95.0) -> mcolors.Normalize | None:
    values = finite_matrix_values(data)
    positive = values[values > EPSILON]
    if positive.size == 0:
        return None
    logs = np.abs(np.log2(positive))
    extent = float(np.percentile(logs, percentile))
    extent = max(extent, 0.25)
    return mcolors.TwoSlopeNorm(vmin=2 ** (-extent), vcenter=1.0, vmax=2 ** extent)


def matrix_from_rows(
    rows: Sequence[dict[str, object]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    row_field: str,
    col_field: str,
    value_field: str,
) -> np.ndarray:
    data = np.full((len(row_labels), len(col_labels)), np.nan)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(value_field))
        if value is None:
            continue
        row_value = str(row.get(row_field) or "")
        col_value = str(row.get(col_field) or "")
        buckets[(row_value, col_value)].append(value)
    row_index = {label: index for index, label in enumerate(row_labels)}
    col_index = {label: index for index, label in enumerate(col_labels)}
    for key, values in buckets.items():
        if key[0] in row_index and key[1] in col_index:
            data[row_index[key[0]], col_index[key[1]]] = float(statistics.median(values))
    return data


def plot_heatmap_matrix(
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    path_base: Path,
    formats: Sequence[str],
    dpi: int,
    cbar_label: str,
    category: str,
    description: str,
    cmap: str = "viridis_r",
    norm: mcolors.Normalize | None = None,
    annotate: bool = True,
    annotation_kind: str = "value",
    max_annotated_cells: int = 900,
) -> None:
    if data.size == 0 or not np.isfinite(data).any():
        return
    fig_w = max(8.0, 0.65 * len(col_labels) + 2.5)
    fig_h = max(4.8, 0.38 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap_obj = plt.colormaps[cmap].copy()
    cmap_obj.set_bad(color="#f2f2f2")
    if norm is None:
        norm, scale_key, scale_label = heatmap_color_scale(data)
    else:
        scale_key, scale_label = "custom", "custom color scale"
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap_obj, norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    cell_count = data.shape[0] * data.shape[1]
    if annotate and cell_count <= max_annotated_cells:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if math.isfinite(data[i, j]):
                    text = compact_number(float(data[i, j]))
                    if annotation_kind == "rank":
                        text = str(int(round(float(data[i, j]))))
                    fontsize = 6.4 if cell_count <= 260 else 5.2
                    ax.text(j, i, text, ha="center", va="center", fontsize=fontsize)
    fig.tight_layout()
    save_figure(fig, path_base, formats, dpi, title, category, description)
    values = finite_matrix_values(data)
    if values.size:
        HEATMAP_SCALE_ROWS.append(
            {
                "plot": str(path_base.with_suffix(f".{formats[0]}")),
                "title": title,
                "colorScale": scale_key,
                "colorScaleDescription": scale_label,
                "finiteCellCount": int(values.size),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )


def plot_overall_ranking(rows: Sequence[dict[str, object]], formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    ordered = list(sorted(rows, key=lambda row: (to_float(row.get("rank")) or math.inf, str(row["generator"]))))
    labels = [str(row["generator"]) for row in ordered]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.44 * len(labels) + 1.5)))
    left = np.zeros(len(labels))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for color, field in zip(colors, COMPONENT_FIELDS):
        values = np.asarray([to_float(row.get(field)) if to_float(row.get(field)) is not None else 0.0 for row in ordered], dtype=float)
        ax.barh(y, values, left=left, color=color, label=COMPONENT_LABELS[field])
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("sum of normalized component ranks, lower is better")
    ax.set_title("Overall Generator Ranking")
    ax.legend(fontsize=8, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for idx, row in enumerate(ordered):
        score = to_float(row.get("aggregateScore"))
        if score is not None:
            rank_value = to_float(row.get("rank")) or float(idx + 1)
            ax.text(left[idx] + 0.02, idx, f"rank {int(rank_value)}  score {score:.2f}", va="center", fontsize=7)
    ax.margins(x=0.15)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(
        fig,
        output_dir / "ranking" / "overall_generator_ranking",
        formats,
        dpi,
        "Overall Generator Ranking",
        "ranking",
        "Stacked normalized component rank scores by generator.",
    )


def plot_ranking_heatmaps(tables: dict[str, list[dict[str, object]]], formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    gen_order = generator_order(tables["overall"])
    datasets = dataset_order(tables["dataset"])
    component_matrix = matrix_from_rows(
        tables["overall"],
        gen_order,
        list(COMPONENT_FIELDS),
        "generator",
        "_component",
        "value",
    )
    component_rows = []
    for row in tables["overall"]:
        for field in COMPONENT_FIELDS:
            component_rows.append({"generator": row["generator"], "_component": COMPONENT_LABELS[field], "value": row.get(field)})
    component_matrix = matrix_from_rows(
        component_rows,
        gen_order,
        [COMPONENT_LABELS[field] for field in COMPONENT_FIELDS],
        "generator",
        "_component",
        "value",
    )
    plot_heatmap_matrix(
        component_matrix,
        gen_order,
        [COMPONENT_LABELS[field] for field in COMPONENT_FIELDS],
        "Generator Score Components",
        output_dir / "ranking" / "generator_score_components",
        formats,
        dpi,
        "normalized rank, lower is better",
        "ranking",
        "Component scores used by the overall generator ranking.",
        annotate=True,
    )
    dataset_matrix = matrix_from_rows(
        tables["dataset"], datasets, gen_order, "dataset", "generator", "aggregateScore"
    )
    rank_rows = []
    for row in tables["dataset"]:
        rank_rows.append({**row, "rankValue": row.get("rank")})
    rank_matrix = matrix_from_rows(rank_rows, datasets, gen_order, "dataset", "generator", "rankValue")
    plot_heatmap_matrix(
        dataset_matrix,
        datasets,
        gen_order,
        "Best Generator by Dataset",
        output_dir / "ranking" / "dataset_generator_heatmap",
        formats,
        dpi,
        "dataset aggregate score, lower is better",
        "ranking",
        "Dataset-level generator aggregate scores.",
        annotate=True,
    )
    plot_heatmap_matrix(
        rank_matrix,
        datasets,
        gen_order,
        "Best Generator by Dataset (Ranks)",
        output_dir / "ranking" / "dataset_generator_rank_heatmap",
        formats,
        dpi,
        "rank",
        "ranking",
        "Dataset-level generator ranks; 1 is best.",
        cmap="magma_r",
        annotate=True,
        annotation_kind="rank",
    )
    for family in ("shape", "timing", "movement", "stroke", "dtw"):
        family_rows = [row for row in tables["family"] if row.get("family") == family]
        data = matrix_from_rows(family_rows, datasets, gen_order, "dataset", "generator", "aggregateScore")
        plot_heatmap_matrix(
            data,
            datasets,
            gen_order,
            f"{family.title()} Metrics by Generator and Dataset",
            output_dir / "ranking" / f"metric_family_{family}",
            formats,
            dpi,
            "family aggregate score, lower is better",
            "ranking",
            f"{family.title()} metric family aggregate scores.",
            annotate=True,
        )

    family_overall_rows = []
    for family in ("shape", "timing", "movement", "stroke", "dtw"):
        for gen in gen_order:
            values = [
                to_float(row.get("aggregateScore"))
                for row in tables["family"]
                if row.get("family") == family and row.get("generator") == gen
            ]
            family_overall_rows.append(
                {"family": family, "generator": gen, "score": median_or_none(values)}
            )
    family_data = matrix_from_rows(
        family_overall_rows,
        ["shape", "timing", "movement", "stroke", "dtw"],
        gen_order,
        "family",
        "generator",
        "score",
    )
    plot_heatmap_matrix(
        family_data,
        ["shape", "timing", "movement", "stroke", "dtw"],
        gen_order,
        "Metric Family Scores by Generator",
        output_dir / "ranking" / "metric_family_generator_scores",
        formats,
        dpi,
        "median family score, lower is better",
        "ranking",
        "Overall metric-family aggregate scores across datasets.",
        annotate=True,
    )


def stream_metric_values(path: Path, metrics: Sequence[str], predicate=None) -> dict[str, list[float]]:
    values = {metric: [] for metric in metrics}
    if not path.exists():
        return values
    header = set(csv_header(path))
    available = [metric for metric in metrics if metric in header]
    if not available:
        return values
    for row in read_csv_rows(path):
        if predicate is not None and not predicate(row):
            continue
        for metric in available:
            value = to_float(row.get(metric))
            if value is not None and value >= -EPSILON:
                values[metric].append(max(0.0, value))
    return values


def downsample(values: Sequence[float], max_n: int = 5000) -> list[float]:
    if len(values) <= max_n:
        return list(values)
    idx = np.linspace(0, len(values) - 1, max_n).astype(int)
    return [float(values[i]) for i in idx]


def plot_boxplot_panels(
    grouped: dict[str, dict[str, list[float]]],
    metrics: Sequence[str],
    title: str,
    path_base: Path,
    formats: Sequence[str],
    dpi: int,
    category: str,
    description: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    labels = list(grouped.keys())
    if not labels:
        return
    cols = 2
    rows = int(math.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(max(10, 0.65 * len(labels) + 4), 3.8 * rows), squeeze=False)
    any_plot = False
    for ax, metric in zip(axes.ravel(), metrics):
        data = [downsample(grouped[label].get(metric, [])) for label in labels]
        non_empty = [vals for vals in data if vals]
        if not non_empty:
            ax.axis("off")
            continue
        ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        any_plot = True
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    if not any_plot:
        plt.close(fig)
        return
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path_base, formats, dpi, title, category, description)


def class_metric_ratios_from_stats(
    combined: Path,
    metrics: Sequence[str],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    metric_set = set(metrics)
    pairwise_stats = load_stats_medians(combined / "stats.csv", metric_set)
    baseline_stats = load_stats_medians(combined / "baseline_stats.csv", metric_set)
    baseline_by_class_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_source, dataset, _variant, class_key, metric), value in baseline_stats.items():
        baseline_by_class_metric[(dataset, class_key, metric)].append(value)
    ratio_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {metric: [] for metric in metrics})
    baseline_grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {metric: [] for metric in metrics})
    for (_source, dataset, _variant, _class_key, metric), value in baseline_stats.items():
        baseline_grouped[dataset][metric].append(value)
    for key, value in pairwise_stats.items():
        den = baseline_stats.get(key)
        if den is None:
            _source, dataset, _variant, class_key, metric = key
            den = median_or_none(baseline_by_class_metric.get((dataset, class_key, metric), []))
        if den is None or den <= EPSILON:
            continue
        source, _dataset, variant, _class_key, metric = key
        gen = label_from_parts(source, variant)
        ratio_grouped[gen][metric].append(value / den)
    return dict(ratio_grouped), dict(baseline_grouped)


def plot_summary_error_outputs(
    input_dir: Path,
    tables: dict[str, list[dict[str, object]]],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    gen_order = generator_order(tables["overall"])
    ratio_rows = []
    for row in tables["metric"]:
        metric = str(row.get("metric") or "")
        if metric in REPRESENTATIVE_METRICS or metric in CORE_METRICS:
            ratio_rows.append(
                {
                    "metric": metric,
                    "generator": row.get("generator"),
                    "ratio": row.get("summaryErrorRatioMedian"),
                }
            )
    metrics = [metric for metric in LOWER_IS_BETTER_METRICS if any(row["metric"] == metric for row in ratio_rows)]
    data = matrix_from_rows(ratio_rows, metrics, gen_order, "metric", "generator", "ratio")
    plot_heatmap_matrix(
        data,
        metrics,
        gen_order,
        "Generated-to-Human Summary Error Ratio",
        output_dir / "summary_error" / "generated_to_human_summary_ratio",
        formats,
        dpi,
        "median generated/human baseline ratio",
        "summary_error",
        "Median generated-to-summary error divided by human-to-summary baseline error.",
        cmap="coolwarm",
        norm=centered_ratio_norm(data),
        annotate=True,
    )

    combined = combined_dir_for(input_dir)
    grouped_ratios, baseline_grouped = class_metric_ratios_from_stats(combined, LOWER_IS_BETTER_METRICS)
    grouped_ratios = {gen: grouped_ratios.get(gen, {metric: [] for metric in LOWER_IS_BETTER_METRICS}) for gen in gen_order}
    plot_boxplot_panels(
        grouped_ratios,
        REPRESENTATIVE_METRICS,
        "Generated Error Relative to Human Summary",
        output_dir / "summary_error" / "representative_metrics" / "summary_error_boxplots",
        formats,
        dpi,
        "summary_error",
        "Class-level generated/human summary error ratio boxplots.",
        "generated / human baseline ratio",
        log_y=True,
    )
    plot_boxplot_panels(
        grouped_ratios,
        LOWER_IS_BETTER_METRICS,
        "Generated Error Relative to Human Summary: All Metrics",
        output_dir / "summary_error" / "all_metrics" / "summary_error_boxplots_all_metrics",
        formats,
        dpi,
        "summary_error",
        "Class-level generated/human summary error ratio boxplots for all lower-is-better metrics.",
        "generated / human baseline ratio",
        log_y=True,
    )

    datasets = dataset_order(tables["dataset"])
    baseline_grouped = {dataset: baseline_grouped.get(dataset, {metric: [] for metric in LOWER_IS_BETTER_METRICS}) for dataset in datasets}
    plot_boxplot_panels(
        baseline_grouped,
        REPRESENTATIVE_METRICS,
        "Human Baseline Variation by Dataset",
        output_dir / "summary_error" / "representative_metrics" / "human_baseline_boxplots",
        formats,
        dpi,
        "summary_error",
        "Human sample-to-summary median metric values by dataset.",
        "median metric value",
        log_y=True,
    )
    plot_boxplot_panels(
        baseline_grouped,
        LOWER_IS_BETTER_METRICS,
        "Human Baseline Variation by Dataset: All Metrics",
        output_dir / "summary_error" / "all_metrics" / "human_baseline_boxplots_all_metrics",
        formats,
        dpi,
        "summary_error",
        "Human sample-to-summary median metric values by dataset for all lower-is-better metrics.",
        "median metric value",
        log_y=True,
    )

    scatter_rows = []
    for row in tables["metric"]:
        if row.get("metric") not in REPRESENTATIVE_METRICS:
            continue
        ratio = to_float(row.get("summaryErrorRatioMedian"))
        if ratio is None:
            continue
        scatter_rows.append(row)
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(9, 6.2))
        color_map = {gen: plt.cm.tab10(i % 10) for i, gen in enumerate(gen_order)}
        for row in scatter_rows:
            ratio = to_float(row.get("summaryErrorRatioMedian"))
            raw = to_float(row.get("summaryErrorRatioMedian"))
            if ratio is None or raw is None:
                continue
            ax.scatter(
                raw,
                ratio,
                color=color_map.get(str(row["generator"]), "#777777"),
                alpha=0.65,
                s=28,
                label=str(row["generator"]),
            )
        ax.axline((1, 1), slope=1, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("median human baseline ratio proxy")
        ax.set_ylabel("median generated/human summary ratio")
        ax.set_title("Dataset Difficulty vs Generator Error")
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=7, ncols=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "summary_error" / "human_baseline_vs_generator_scatter",
            formats,
            dpi,
            "Dataset Difficulty vs Generator Error",
            "summary_error",
            "Scatter of dataset/generator summary-relative ratio observations.",
        )


def plot_variability_outputs(
    input_dir: Path,
    tables: dict[str, list[dict[str, object]]],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    gen_order = generator_order(tables["overall"])
    ratio_rows = []
    for row in tables["metric"]:
        value = to_float(row.get("variabilityLogRatioMedian"))
        if value is None:
            continue
        ratio_rows.append({"metric": row.get("metric"), "generator": row.get("generator"), "ratio": math.exp(value)})
    metrics = [metric for metric in LOWER_IS_BETTER_METRICS if any(row["metric"] == metric for row in ratio_rows)]
    data = matrix_from_rows(ratio_rows, metrics, gen_order, "metric", "generator", "ratio")
    plot_heatmap_matrix(
        data,
        metrics,
        gen_order,
        "Generated Internal Variability vs Human Internal Variability",
        output_dir / "variability" / "generated_internal_variability_ratio",
        formats,
        dpi,
        "median generated/human internal variability ratio",
        "variability",
        "Within-generated median variability divided by within-human median variability. Color scale is centered at 1 and saturated at 4 so extreme outliers remain annotated without washing out other cells.",
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=4.0),
        annotate=True,
    )

    combined = combined_dir_for(input_dir)
    grouped: dict[str, dict[str, list[float]]] = {gen: {metric: [] for metric in REPRESENTATIVE_METRICS} for gen in gen_order}
    for row in read_csv_rows(combined / "pairwise.csv"):
        gen = row_label(row)
        if gen not in grouped:
            continue
        for metric in REPRESENTATIVE_METRICS:
            value = to_float(row.get(metric))
            if value is not None and value >= -EPSILON:
                grouped[gen][metric].append(max(0.0, value))
    plot_boxplot_panels(
        grouped,
        REPRESENTATIVE_METRICS,
        "Raw Pairwise Metrics by Generator",
        output_dir / "variability" / "raw_pairwise_metric_boxplots",
        formats,
        dpi,
        "variability",
        "Raw generated-to-summary row-level metric distributions by generator.",
        "metric value",
        log_y=True,
    )


def plot_tradeoff_scatters(
    tables: dict[str, list[dict[str, object]]],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    gen_order = generator_order(tables["overall"])
    color_map = {gen: plt.cm.tab10(i % 10) for i, gen in enumerate(gen_order)}
    scatter_specs = [
        ("summaryErrorScore", "variabilityScore", "Summary Error vs Variability Match"),
        ("summaryErrorScore", "distributionScore", "Summary Error vs Raw Distribution Match"),
        ("summaryErrorScore", "summaryDistributionScore", "Summary Error vs Summary Distribution Match"),
    ]
    for x_field, y_field, title in scatter_specs:
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        any_point = False
        for row in tables["metric"]:
            x = to_float(row.get(x_field))
            y = to_float(row.get(y_field))
            if x is None or y is None:
                continue
            gen = str(row.get("generator"))
            ax.scatter(x, y, s=28, alpha=0.55, color=color_map.get(gen, "#777777"), label=gen)
            any_point = True
        if not any_point:
            plt.close(fig)
            continue
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7, ncols=2)
        ax.set_xlabel(f"{COMPONENT_LABELS.get(x_field, x_field)} normalized rank")
        ax.set_ylabel(f"{COMPONENT_LABELS.get(y_field, y_field)} normalized rank")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "tradeoffs" / safe_name(title.lower()),
            formats,
            dpi,
            title,
            "tradeoffs",
            f"Metric-level tradeoff scatter of {x_field} against {y_field}; lower-left is better.",
        )


def plot_human_generated_metric_scatters(
    input_dir: Path,
    tables: dict[str, list[dict[str, object]]],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    combined = combined_dir_for(input_dir)
    metric_set = set(LOWER_IS_BETTER_METRICS)
    pairwise_stats = load_stats_medians(combined / "stats.csv", metric_set)
    baseline_stats = load_stats_medians(combined / "baseline_stats.csv", metric_set)
    baseline_by_class_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_source, dataset, _variant, class_key, metric), value in baseline_stats.items():
        baseline_by_class_metric[(dataset, class_key, metric)].append(value)

    rows_by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key, generated in pairwise_stats.items():
        source, dataset, variant, class_key, metric = key
        human = baseline_stats.get(key)
        if human is None:
            human = median_or_none(baseline_by_class_metric.get((dataset, class_key, metric), []))
        if human is None or generated <= EPSILON or human <= EPSILON:
            continue
        rows_by_family[metric_family(metric)].append(
            {
                "generator": label_from_parts(source, variant),
                "dataset": dataset,
                "classKey": class_key,
                "metric": metric,
                "humanMedian": human,
                "generatedMedian": generated,
                "ratio": generated / human,
            }
        )
    flat_rows = [row for rows in rows_by_family.values() for row in rows]
    write_csv(output_dir / "tables" / "human_vs_generated_metric_points.csv", flat_rows)

    gen_order = generator_order(tables["overall"])
    color_map = {gen: plt.cm.tab10(i % 10) for i, gen in enumerate(gen_order)}
    for family in ("core", "shape", "timing", "movement", "stroke", "dtw"):
        rows = rows_by_family.get(family, [])
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 6.8))
        for gen in gen_order:
            points = [row for row in rows if row["generator"] == gen]
            if len(points) > 3500:
                indices = np.linspace(0, len(points) - 1, 3500).astype(int)
                points = [points[index] for index in indices]
            if not points:
                continue
            ax.scatter(
                [float(row["humanMedian"]) for row in points],
                [float(row["generatedMedian"]) for row in points],
                s=15,
                alpha=0.35,
                color=color_map.get(gen, "#777777"),
                label=gen,
            )
        values = [
            float(value)
            for row in rows
            for value in (row["humanMedian"], row["generatedMedian"])
            if to_float(value) is not None and float(value) > 0
        ]
        if values:
            low = min(values)
            high = max(values)
            ax.plot([low, high], [low, high], color="#333333", linewidth=0.9, linestyle="--")
            ax.set_xlim(left=low * 0.8, right=high * 1.2)
            ax.set_ylim(bottom=low * 0.8, top=high * 1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("human-to-summary class median metric")
        ax.set_ylabel("generated-to-summary class median metric")
        ax.set_title(f"Human vs Generated Metrics: {family.title()}")
        ax.legend(fontsize=7, ncols=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "summary_error" / "human_vs_generated_scatter" / f"{family}_metrics",
            formats,
            dpi,
            f"Human vs Generated Metrics: {family.title()}",
            "summary_error",
            "Each point is a dataset/class/metric/generator median. The dashed line is generated equals human baseline.",
        )


def load_distribution_scope_medians(
    path: Path,
    metrics: set[str],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if not path.exists():
        return {}
    for row in read_csv_rows(path):
        metric = str(row.get("metric") or "")
        if metric not in metrics:
            continue
        generator = row_label(row)
        dataset = str(row.get("dataset") or "unknown")
        key = (generator, dataset, metric, str(row.get("summary") or ""))
        for scope, field in (
            ("human_human", "withinReferenceMdn"),
            ("generated_generated", "withinComparisonMdn"),
            ("human_generated", "betweenGroupsMdn"),
        ):
            value = to_float(row.get(field))
            if value is not None and value >= -EPSILON:
                grouped[key][scope].append(max(0.0, value))
    medians: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, scope_values in grouped.items():
        reduced: dict[str, float] = {}
        for scope, values in scope_values.items():
            median = median_or_none(values)
            if median is not None:
                reduced[scope] = median
        if reduced:
            medians[key] = reduced
    return medians


def load_summary_pairwise_medians(
    path: Path,
    metrics: set[str],
    *,
    human_source_name: str = "human",
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    if not path.exists():
        return {}
    for row in read_csv_rows(path):
        metric = str(row.get("metric") or "")
        if metric not in metrics:
            continue
        source = str(row.get("source") or "unknown")
        variant = str(row.get("variant") or "root")
        generator = human_source_name if source == human_source_name else label_from_parts(source, variant)
        dataset = str(row.get("dataset") or "unknown")
        value = to_float(row.get("mdn"))
        if value is not None and value >= -EPSILON:
            grouped[(generator, dataset, metric)].append(max(0.0, value))
    return {
        key: float(statistics.median(values))
        for key, values in grouped.items()
        if values
    }


def affinity_matrix_for_metric(
    generator: str,
    dataset: str,
    metric: str,
    distribution_medians: dict[tuple[str, str, str, str], dict[str, float]],
    pairwise_summary_medians: dict[tuple[str, str, str], float],
    baseline_summary_medians: dict[tuple[str, str, str], float],
    summary_label: str,
) -> np.ndarray:
    matrix = np.full((3, 3), np.nan)
    scope_values = distribution_medians.get((generator, dataset, metric, summary_label), {})
    if not scope_values:
        scope_values = distribution_medians.get((generator, dataset, metric, ""), {})
    human_human = scope_values.get("human_human")
    generated_generated = scope_values.get("generated_generated")
    human_generated = scope_values.get("human_generated")
    human_summary = baseline_summary_medians.get(("human", dataset, metric))
    generated_summary = pairwise_summary_medians.get((generator, dataset, metric))

    if human_human is not None:
        matrix[0, 0] = human_human
    if generated_generated is not None:
        matrix[1, 1] = generated_generated
    if human_generated is not None:
        matrix[0, 1] = human_generated
        matrix[1, 0] = human_generated
    if human_summary is not None:
        matrix[0, 2] = human_summary
        matrix[2, 0] = human_summary
    if generated_summary is not None:
        matrix[1, 2] = generated_summary
        matrix[2, 1] = generated_summary
    return matrix


def plot_affinity_matrix_panels(
    matrices: Sequence[tuple[str, np.ndarray]],
    title: str,
    path_base: Path,
    formats: Sequence[str],
    dpi: int,
    category: str,
    description: str,
) -> None:
    if not matrices:
        return
    cols = min(3, max(1, len(matrices)))
    rows = int(math.ceil(len(matrices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.1 * rows), squeeze=False)
    labels = ["human", "generated", "human summary"]
    any_plot = False
    for ax, (metric, matrix) in zip(axes.ravel(), matrices):
        if not np.isfinite(matrix).any():
            ax.axis("off")
            continue
        cmap_obj = plt.colormaps["viridis_r"].copy()
        cmap_obj.set_bad(color="#f2f2f2")
        norm, _scale_key, _scale_label = heatmap_color_scale(matrix)
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap_obj, norm=norm)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(metric, fontsize=10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(matrix[i, j]):
                    ax.text(j, i, compact_number(float(matrix[i, j])), ha="center", va="center", fontsize=8)
        any_plot = True
    for ax in axes.ravel()[len(matrices):]:
        ax.axis("off")
    if not any_plot:
        plt.close(fig)
        return
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path_base, formats, dpi, title, category, description)


def plot_pairwise_affinity_outputs(
    input_dir: Path,
    tables: dict[str, list[dict[str, object]]],
    metrics: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    combined = combined_dir_for(input_dir)
    metric_set = set(metrics)
    distribution_path = combined / "distribution.csv"
    summary_distribution_path = combined / "summary_distribution.csv"
    distribution_medians = load_distribution_scope_medians(distribution_path, metric_set)
    pairwise_summary_medians = load_summary_pairwise_medians(combined / "stats.csv", metric_set)
    baseline_summary_medians = load_summary_pairwise_medians(combined / "baseline_stats.csv", metric_set)
    summary_names = Counter()
    if summary_distribution_path.exists():
        for row in read_csv_rows(summary_distribution_path):
            value = str(row.get("summary") or "").strip()
            if value:
                summary_names[value] += 1
    summary_label = summary_names.most_common(1)[0][0] if summary_names else ""

    selected_plot_metrics = [metric for metric in metrics if metric in REPRESENTATIVE_METRICS]
    if not selected_plot_metrics:
        selected_plot_metrics = list(metrics[:6])

    table_rows = []
    for row in tables["dataset"]:
        generator = str(row.get("generator") or "")
        dataset = str(row.get("dataset") or "")
        panel_matrices = []
        for metric in selected_plot_metrics:
            matrix = affinity_matrix_for_metric(
                generator,
                dataset,
                metric,
                distribution_medians,
                pairwise_summary_medians,
                baseline_summary_medians,
                summary_label,
            )
            if not np.isfinite(matrix).any():
                continue
            panel_matrices.append((metric, matrix))
            table_rows.append(
                {
                    "generator": generator,
                    "dataset": dataset,
                    "metric": metric,
                    "humanHuman": matrix[0, 0] if math.isfinite(matrix[0, 0]) else "",
                    "generatedGenerated": matrix[1, 1] if math.isfinite(matrix[1, 1]) else "",
                    "humanGenerated": matrix[0, 1] if math.isfinite(matrix[0, 1]) else "",
                    "humanToSummary": matrix[0, 2] if math.isfinite(matrix[0, 2]) else "",
                    "generatedToSummary": matrix[1, 2] if math.isfinite(matrix[1, 2]) else "",
                    "summaryLabel": summary_label,
                }
            )
        if not panel_matrices:
            continue
        subtitle = f"Pairwise Affinity Matrix: {generator} on {dataset}"
        if summary_label:
            subtitle += f"\nsummary = {summary_label}"
        plot_affinity_matrix_panels(
            panel_matrices,
            subtitle,
            output_dir / "pairwise_affinity" / safe_name(dataset) / safe_name(generator),
            formats,
            dpi,
            "pairwise_affinity",
            "Annotated pairwise comparison matrices linking human-human, generated-generated, human-generated, and summary-relative medians.",
        )
    if table_rows:
        write_csv(output_dir / "tables" / "pairwise_affinity_values.csv", table_rows)


def plot_distribution_outputs(
    input_dir: Path,
    tables: dict[str, list[dict[str, object]]],
    formats: Sequence[str],
    dpi: int,
    output_dir: Path,
) -> None:
    gen_order = generator_order(tables["overall"])
    combined = combined_dir_for(input_dir)
    for filename, title, out_name in (
        ("distribution.csv", "Raw Distribution Distance from Human Variability", "raw_distribution_distance_heatmap"),
        ("summary_distribution.csv", "Summary Distribution Distance from Human Baseline", "summary_distribution_distance_heatmap"),
    ):
        rows = []
        if not (combined / filename).exists():
            continue
        summary_values = Counter()
        for row in read_csv_rows(combined / filename):
            metric = str(row.get("metric") or "")
            if metric not in LOWER_IS_BETTER_METRICS:
                continue
            value = to_float(row.get("jensenShannonDivergence"))
            if value is None:
                continue
            rows.append({"metric": metric, "generator": row_label(row), "value": value})
            if row.get("summary"):
                summary_values[str(row["summary"])] += 1
        metrics = [metric for metric in LOWER_IS_BETTER_METRICS if any(r["metric"] == metric for r in rows)]
        data = matrix_from_rows(rows, metrics, gen_order, "metric", "generator", "value")
        subtitle = ""
        if filename == "summary_distribution.csv" and summary_values:
            subtitle = f"\nsummary = {summary_values.most_common(1)[0][0]}"
        plot_heatmap_matrix(
            data,
            metrics,
            gen_order,
            title + subtitle,
            output_dir / "distributions" / out_name,
            formats,
            dpi,
            "median Jensen-Shannon divergence",
            "distributions",
            f"{filename} median Jensen-Shannon divergence by metric and generator.",
            annotate=True,
        )

    small_rows_by_file: dict[str, dict[str, dict[str, list[float]]]] = {
        "distribution": defaultdict(lambda: defaultdict(list)),
        "summary_distribution": defaultdict(lambda: defaultdict(list)),
    }
    for file_key, filename in (("distribution", "distribution.csv"), ("summary_distribution", "summary_distribution.csv")):
        path = combined / filename
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            gen = row_label(row)
            for metric in DISTANCE_METRICS:
                value = to_float(row.get(metric))
                if value is not None and value >= 0:
                    small_rows_by_file[file_key][metric][gen].append(value)
    fig, axes = plt.subplots(len(DISTANCE_METRICS), 2, figsize=(13, 3.0 * len(DISTANCE_METRICS)), squeeze=False)
    any_plot = False
    for col, file_key in enumerate(("distribution", "summary_distribution")):
        for row_index, metric in enumerate(DISTANCE_METRICS):
            ax = axes[row_index][col]
            medians = []
            errors_low = []
            errors_high = []
            labels = []
            for gen in gen_order:
                q = quantiles(small_rows_by_file[file_key][metric].get(gen, []))
                if q["median"] is None:
                    continue
                labels.append(gen)
                medians.append(float(q["median"]))
                errors_low.append(float(q["median"]) - float(q["q1"]))
                errors_high.append(float(q["q3"]) - float(q["median"]))
            if not medians:
                ax.axis("off")
                continue
            x = np.arange(len(labels))
            ax.bar(x, medians, yerr=[errors_low, errors_high], color="#4C78A8", error_kw={"capsize": 2, "elinewidth": 0.8})
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{file_key}: {metric}")
            ax.grid(axis="y", alpha=0.25)
            any_plot = True
    if any_plot:
        fig.suptitle("Distribution Distance Metrics")
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "distributions" / "distribution_distance_small_multiples",
            formats,
            dpi,
            "Distribution Distance Metrics",
            "distributions",
            "Median and IQR for distribution distance metrics.",
        )
    else:
        plt.close(fig)


def plot_diagnostics(input_dir: Path, tables: dict[str, list[dict[str, object]]], formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    gen_order = generator_order(tables["overall"])
    combined = combined_dir_for(input_dir)
    for filename, prefix in (("distribution.csv", "raw"), ("summary_distribution.csv", "summary")):
        path = combined / filename
        if not path.exists():
            continue
        normality_counts: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        shape_points = []
        for row in read_csv_rows(path):
            metric = str(row.get("metric") or "")
            if metric not in LOWER_IS_BETTER_METRICS:
                continue
            gen = row_label(row)
            for scope in ("withinReference", "withinComparison", "betweenGroups"):
                pvalue = to_float(row.get(f"{scope}NormalityPValue"))
                if pvalue is not None:
                    normality_counts[(scope, metric, gen)].append(1 if pvalue < NORMALITY_ALPHA else 0)
            for scope in ("withinReference", "withinComparison", "betweenGroups"):
                skew = to_float(row.get(f"{scope}Skewness"))
                kurt = to_float(row.get(f"{scope}Kurtosis"))
                if skew is not None and kurt is not None and metric in REPRESENTATIVE_METRICS:
                    shape_points.append((scope, metric, gen, skew, kurt))
        for scope in ("withinReference", "withinComparison", "betweenGroups"):
            rows = []
            for (scope_key, metric, gen), values in normality_counts.items():
                if scope_key == scope and values:
                    rows.append({"metric": metric, "generator": gen, "share": statistics.fmean(values)})
            metrics = [metric for metric in LOWER_IS_BETTER_METRICS if any(row["metric"] == metric for row in rows)]
            data = matrix_from_rows(rows, metrics, gen_order, "metric", "generator", "share")
            plot_heatmap_matrix(
                data,
                metrics,
                gen_order,
                f"Distribution Normality Diagnostics: {prefix} {scope}",
                output_dir / "distributions" / f"normality_diagnostics_{prefix}_{safe_name(scope)}",
                formats,
                dpi,
                "share with p < 0.05",
                "distributions",
                f"Share of class-runs where {scope} normality p-value is below 0.05.",
                cmap="magma",
                annotate=True,
            )
        if shape_points:
            fig, ax = plt.subplots(figsize=(9, 6.5))
            color_map = {gen: plt.cm.tab10(i % 10) for i, gen in enumerate(gen_order)}
            by_generator: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for _scope, _metric, gen, skew, kurt in shape_points:
                by_generator[gen].append((skew, kurt))
            for gen in gen_order:
                points = by_generator.get(gen, [])
                if not points:
                    continue
                if len(points) > 6000:
                    indices = np.linspace(0, len(points) - 1, 6000).astype(int)
                    points = [points[index] for index in indices]
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                ax.scatter(xs, ys, s=12, alpha=0.28, color=color_map.get(gen, "#777777"), label=gen)
            ax.legend(fontsize=7, ncols=2)
            ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
            ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("skewness")
            ax.set_ylabel("excess kurtosis")
            ax.set_title(f"Distribution Shape Diagnostics: {prefix}")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            save_figure(
                fig,
                output_dir / "distributions" / f"skewness_kurtosis_diagnostics_{prefix}",
                formats,
                dpi,
                f"Distribution Shape Diagnostics: {prefix}",
                "distributions",
                "Skewness versus kurtosis for representative metrics.",
            )


def plot_class_outputs(input_dir: Path, tables: dict[str, list[dict[str, object]]], args: argparse.Namespace, formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    gen_order = generator_order(tables["overall"])
    combined = combined_dir_for(input_dir)
    baseline_stats = load_stats_medians(combined / "baseline_stats.csv", set(CORE_METRICS))
    pairwise_stats = load_stats_medians(combined / "stats.csv", set(CORE_METRICS))
    baseline_by_class_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_source, dataset, _variant, class_key, metric), value in baseline_stats.items():
        baseline_by_class_metric[(dataset, class_key, metric)].append(value)
    class_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    class_gen_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for key, value in pairwise_stats.items():
        den = baseline_stats.get(key)
        if den is None:
            _source, dataset, _variant, class_key, metric = key
            den = median_or_none(baseline_by_class_metric.get((dataset, class_key, metric), []))
        if den is None or den <= EPSILON:
            continue
        source, dataset, variant, class_key, _metric = key
        gen = label_from_parts(source, variant)
        ratio = value / den
        class_values[(dataset, class_key, f"{dataset}/{class_key}")].append(ratio)
        class_gen_values[(dataset, class_key, f"{dataset}/{class_key}", gen)].append(ratio)
    ranked = []
    for (dataset, class_key, label), values in class_values.items():
        q = quantiles(values)
        ranked.append({"dataset": dataset, "classKey": class_key, "classLabel": label, "medianNormalizedError": q["median"], "n": q["n"]})
    ranked.sort(key=lambda row: (to_float(row.get("medianNormalizedError")) or -math.inf), reverse=True)
    top_ranked = ranked[: args.top_classes]
    write_csv(output_dir / "tables" / "class_difficulty_scores.csv", ranked)
    if top_ranked:
        fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(top_ranked) + 1.5)))
        labels = [str(row["classLabel"]) for row in reversed(top_ranked)]
        values = [to_float(row.get("medianNormalizedError")) or 0 for row in reversed(top_ranked)]
        ax.barh(labels, values, color="#4C78A8")
        ax.set_xlabel("median generated/human baseline ratio")
        ax.set_title("Most Difficult Classes")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "classes" / "most_difficult_classes",
            formats,
            dpi,
            "Most Difficult Classes",
            "classes",
            "Top classes by median normalized generated summary error.",
        )
    for dataset in sorted({row["dataset"] for row in top_ranked})[:6]:
        class_labels = [row["classLabel"] for row in top_ranked if row["dataset"] == dataset][: args.top_classes]
        rows = []
        for (ds, class_key, label, gen), values in class_gen_values.items():
            if ds == dataset and label in class_labels:
                rows.append({"classLabel": label, "generator": gen, "value": median_or_none(values)})
        data = matrix_from_rows(rows, class_labels, gen_order, "classLabel", "generator", "value")
        plot_heatmap_matrix(
            data,
            class_labels,
            gen_order,
            f"Class-Level Difficulty: {dataset}",
            output_dir / "classes" / f"class_difficulty_heatmap_{safe_name(dataset)}",
            formats,
            dpi,
            "median generated/human baseline ratio",
            "classes",
            f"Top difficult classes for {dataset}.",
            annotate=True,
        )


def plot_dtw_and_variants(tables: dict[str, list[dict[str, object]]], formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    gen_order = generator_order(tables["overall"])
    rows = [row for row in tables["metric"] if row.get("metric") in DTW_METRICS]
    if rows:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        width = 0.8 / max(1, len(gen_order))
        x = np.arange(len(DTW_METRICS))
        for idx, gen in enumerate(gen_order):
            values = []
            for metric in DTW_METRICS:
                values.append(median_or_none(to_float(row.get("summaryErrorRatioMedian")) for row in rows if row.get("generator") == gen and row.get("metric") == metric))
            ax.bar(x + (idx - (len(gen_order) - 1) / 2) * width, [np.nan if v is None else v for v in values], width, label=gen)
        ax.set_xticks(x)
        ax.set_xticklabels(DTW_METRICS, rotation=30, ha="right")
        finite_values = [
            to_float(row.get("summaryErrorRatioMedian"))
            for row in rows
            if to_float(row.get("summaryErrorRatioMedian")) is not None
        ]
        if any(value is not None and value > 0 for value in finite_values):
            ax.set_yscale("log")
        ax.set_ylabel("median generated/human summary ratio")
        ax.set_title("DTW Metric Variant Agreement")
        ax.legend(fontsize=7, ncols=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "dtw" / "dtw_variant_comparison",
            formats,
            dpi,
            "DTW Metric Variant Agreement",
            "dtw",
            "Grouped bars comparing DTW variant summary-relative ratios.",
        )
    scriptstudio_rows = [row for row in tables["family"] if str(row.get("generator")) in ("scriptstudio/recoTO", "scriptstudio/syntTO")]
    if scriptstudio_rows:
        datasets = sorted({str(row["dataset"]) for row in scriptstudio_rows})
        families = ["shape", "timing", "movement", "stroke", "dtw"]
        fig, axes = plt.subplots(len(families), 1, figsize=(11, 2.7 * len(families)), sharex=True)
        for ax, family in zip(np.ravel(axes), families):
            x = np.arange(len(datasets))
            for offset, gen in ((-0.18, "scriptstudio/recoTO"), (0.18, "scriptstudio/syntTO")):
                values = []
                for dataset in datasets:
                    values.append(median_or_none(to_float(row.get("aggregateScore")) for row in scriptstudio_rows if row.get("generator") == gen and row.get("dataset") == dataset and row.get("family") == family))
                ax.bar(x + offset, [np.nan if v is None else v for v in values], 0.34, label=gen)
            ax.set_ylabel(family)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(np.arange(len(datasets)))
        axes[-1].set_xticklabels(datasets, rotation=45, ha="right")
        axes[0].legend(fontsize=8)
        fig.suptitle("ScriptStudio Variant Comparison")
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "variants" / "scriptstudio_variant_comparison",
            formats,
            dpi,
            "ScriptStudio Variant Comparison",
            "variants",
            "Family score comparison for scriptstudio/recoTO and scriptstudio/syntTO.",
        )


def plot_leaderboard_tables(tables: dict[str, list[dict[str, object]]], output_dir: Path) -> None:
    metric_rows = []
    for metric in sorted({str(row["metric"]) for row in tables["metric"]}):
        rows = [row for row in tables["metric"] if row.get("metric") == metric and to_float(row.get("aggregateScore")) is not None]
        rows.sort(key=lambda row: to_float(row.get("aggregateScore")) or math.inf)
        if not rows:
            continue
        metric_rows.append(
            {
                "metric": metric,
                "bestGenerator": rows[0]["generator"],
                "secondGenerator": rows[1]["generator"] if len(rows) > 1 else "",
                "bestMedianScore": rows[0].get("aggregateScore"),
                "classRunCount": rows[0].get("componentCellCount", ""),
                "notes": "lower aggregate score is better; twoThirdsPowerLawR2 excluded from default ranking" if metric == "twoThirdsPowerLawR2" else "",
            }
        )
    write_csv(output_dir / "tables" / "metric_leaderboard.csv", metric_rows)
    dataset_rows = []
    for dataset in sorted({str(row["dataset"]) for row in tables["dataset"]}):
        rows = [row for row in tables["dataset"] if row.get("dataset") == dataset]
        rows.sort(key=lambda row: to_float(row.get("aggregateScore")) or math.inf)
        if not rows:
            continue
        def best_for(field: str) -> str:
            candidates = [row for row in rows if to_float(row.get(field)) is not None]
            candidates.sort(key=lambda row: to_float(row.get(field)) or math.inf)
            return str(candidates[0]["generator"]) if candidates else ""
        dataset_rows.append(
            {
                "dataset": dataset,
                "bestOverallGenerator": rows[0]["generator"],
                "bestSummaryErrorGenerator": best_for("summaryErrorScore"),
                "bestVariabilityGenerator": best_for("variabilityScore"),
                "bestDistributionGenerator": best_for("distributionScore"),
                "metricCount": rows[0].get("metricCount", ""),
            }
        )
    write_csv(output_dir / "tables" / "dataset_leaderboard.csv", dataset_rows)


def parse_warning_summary(input_dir: Path) -> list[dict[str, object]]:
    rows = []
    manifest = load_json(input_dir / "manifest.json")
    warning_items = []
    for key in ("planningWarnings", "warnings"):
        value = manifest.get(key)
        if isinstance(value, list):
            warning_items.extend(value)
    for item in warning_items:
        rows.append({"source": "", "dataset": "", "variant": "", "classKey": "", "type": "manifest", "message": str(item), "count": 1})
    patterns = [
        ("warning", re.compile(r"\bwarn(?:ing)?\b[:\s-]*(.*)", re.IGNORECASE)),
        ("skip", re.compile(r"\bskip(?:ped|ping)?\b[:\s-]*(.*)", re.IGNORECASE)),
    ]
    for log_name in ("run.log", "stderr.log", "stdout.log"):
        path = input_dir / log_name
        if not path.exists():
            continue
        counter: Counter[tuple[str, str]] = Counter()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for kind, pattern in patterns:
                match = pattern.search(line)
                if match:
                    message = (match.group(1) or line).strip()
                    counter[(kind, message[:220])] += 1
        for (kind, message), count in counter.items():
            rows.append({"source": "", "dataset": "", "variant": "", "classKey": "", "type": f"{log_name}:{kind}", "message": message, "count": count})
    return rows


def build_coverage_outputs(input_dir: Path, tables: dict[str, list[dict[str, object]]], formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    manifest = load_json(input_dir / "manifest.json")
    run = load_json(input_dir / "run.json")
    combined = combined_dir_for(input_dir)
    rows = []
    run_entries = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    for run_entry in run_entries:
        source = run_entry.get("source", "")
        dataset = run_entry.get("dataset", "")
        variant = run_entry.get("variant", "root")
        rows.append(
            {
                "source": source,
                "dataset": dataset,
                "variant": variant,
                "generator": label_from_parts(source, variant),
                "classCount": run_entry.get("classCount", ""),
                "pairwiseRows": run_entry.get("pairwiseRows", ""),
                "baselineRows": run_entry.get("baselineRows", ""),
                "withinReferenceRows": run_entry.get("withinReferenceRows", ""),
                "withinComparisonRows": run_entry.get("withinComparisonRows", ""),
                "betweenGroupsRows": run_entry.get("betweenGroupsRows", ""),
                "distributionRows": run_entry.get("distributionRows", ""),
                "summaryDistributionRows": run_entry.get("summaryDistributionRows", ""),
            }
        )
    if not rows:
        for row in tables["dataset"]:
            rows.append(
                {
                    "source": str(row["generator"]).split("/", 1)[0],
                    "dataset": row.get("dataset", ""),
                    "variant": str(row["generator"]).split("/", 1)[1] if "/" in str(row["generator"]) else "root",
                    "generator": row.get("generator", ""),
                    "classCount": "",
                    "pairwiseRows": "",
                    "baselineRows": "",
                    "withinReferenceRows": "",
                    "withinComparisonRows": "",
                    "betweenGroupsRows": "",
                    "distributionRows": "",
                    "summaryDistributionRows": "",
                }
            )
    file_count_rows = []
    for filename in (
        "aggregate_summaries.csv",
        "baseline.csv",
        "baseline_stats.csv",
        "between_groups.csv",
        "between_groups_stats.csv",
        "distribution.csv",
        "pairwise.csv",
        "stats.csv",
        "summary_distribution.csv",
        "within_comparison.csv",
        "within_comparison_stats.csv",
        "within_reference.csv",
        "within_reference_stats.csv",
    ):
        path = combined / filename
        if path.exists():
            with path.open("rb") as fh:
                line_count = sum(1 for _ in fh)
            file_count_rows.append({"file": str(path), "dataRows": max(0, line_count - 1)})
    coverage_summary = [
        {"field": "inputDir", "value": str(input_dir)},
        {"field": "combinedDir", "value": str(combined)},
        {"field": "plannedRunCount", "value": manifest.get("plannedRunCount", "")},
        {"field": "plannedCandidateCount", "value": manifest.get("plannedCandidateCount", "")},
        {"field": "rate", "value": manifest.get("rate", run.get("effectiveConfig", {}).get("rate", ""))},
        {"field": "summary", "value": manifest.get("summary", run.get("effectiveConfig", {}).get("summary", ""))},
        {"field": "sampleLimitPerClass", "value": manifest.get("sampleLimitPerClass", run.get("effectiveConfig", {}).get("sampleLimitPerClass", ""))},
        {"field": "distributionSampleLimitPerClass", "value": manifest.get("distributionSampleLimitPerClass", run.get("effectiveConfig", {}).get("distributionSampleLimitPerClass", ""))},
        {"field": "rawJsonl", "value": manifest.get("rawJsonl", run.get("effectiveConfig", {}).get("rawJsonl", ""))},
    ]
    for file_row in file_count_rows:
        coverage_summary.append({"field": f"rows:{Path(str(file_row['file'])).name}", "value": file_row["dataRows"]})
    write_csv(output_dir / "tables" / "coverage_summary.csv", coverage_summary, ["field", "value"])
    write_csv(output_dir / "tables" / "coverage_by_run.csv", rows)
    warning_rows = parse_warning_summary(input_dir)
    write_csv(output_dir / "tables" / "warning_summary.csv", warning_rows, ["source", "dataset", "variant", "classKey", "type", "message", "count"])

    if rows:
        gen_order = generator_order(tables["overall"])
        datasets = sorted({str(row["dataset"]) for row in rows})
        matrix_rows = []
        for row in rows:
            value = to_float(row.get("classCount")) or to_float(row.get("distributionRows")) or 0.0
            matrix_rows.append({"dataset": row["dataset"], "generator": row["generator"], "value": value})
        data = matrix_from_rows(matrix_rows, datasets, gen_order, "dataset", "generator", "value")
        plot_heatmap_matrix(
            data,
            datasets,
            gen_order,
            "Evaluation Coverage",
            output_dir / "coverage" / "evaluation_coverage",
            formats,
            dpi,
            "completed classes or distribution rows",
            "coverage",
            "Coverage by dataset and generator from manifest/run metadata.",
            cmap="viridis",
            annotate=True,
        )
    if warning_rows:
        top = sorted(warning_rows, key=lambda row: int(row.get("count") or 0), reverse=True)[:18]
        fig, ax = plt.subplots(figsize=(11, max(4, 0.34 * len(top) + 1.5)))
        labels = [f"{row['type']}: {row['message'][:70]}" for row in reversed(top)]
        counts = [int(row.get("count") or 0) for row in reversed(top)]
        ax.barh(labels, counts, color="#E45756")
        ax.set_title("Warnings and Skipped Classes")
        ax.set_xlabel("count")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "coverage" / "warning_skip_summary",
            formats,
            dpi,
            "Warnings and Skipped Classes",
            "coverage",
            "Top warning/skip messages found in manifest and logs.",
        )


def selected_histogram_cases(
    input_dir: Path,
    limit: int,
    selection: str,
    output_dir: Path,
) -> list[tuple[str, str, str, str, str]]:
    combined = combined_dir_for(input_dir)
    rows = []
    path = combined / "distribution.csv"
    if path.exists():
        for row in read_csv_rows(path):
            metric = str(row.get("metric") or "")
            if metric not in HISTOGRAM_METRICS:
                continue
            divergence = to_float(row.get("jensenShannonDivergence"))
            count = to_float(row.get("withinComparisonN")) or 0
            score = (divergence or 0.0, count)
            rows.append((score, str(row.get("source") or ""), str(row.get("dataset") or ""), str(row.get("variant") or "root"), str(row.get("classKey") or ""), metric))
    rows.sort(reverse=True, key=lambda item: item[0])
    all_index_rows = []
    selected_keys = set()
    seen = set()
    cases = []
    unique_rows = []
    for _score, source, dataset, variant, class_key, metric in rows:
        key = (source, dataset, variant, class_key, metric)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append((_score, source, dataset, variant, class_key, metric))
        all_index_rows.append(
            {
                "source": source,
                "dataset": dataset,
                "variant": variant,
                "classKey": class_key,
                "metric": metric,
                "jensenShannonDivergence": _score[0],
                "withinComparisonN": _score[1],
            }
        )

    if selection == "all":
        cases = [(source, dataset, variant, class_key, metric) for _score, source, dataset, variant, class_key, metric in unique_rows]
    elif selection == "top":
        for _score, source, dataset, variant, class_key, metric in unique_rows:
            key = (source, dataset, variant, class_key, metric)
            class_key_short = (source, dataset, variant, class_key)
            if sum(1 for case in cases if case[:4] == class_key_short) >= 2:
                continue
            cases.append(key)
            if len(cases) >= limit:
                break
    else:
        by_generator_metric: dict[tuple[str, str, str], list[tuple[tuple[float, float], str, str, str, str, str]]] = defaultdict(list)
        for item in unique_rows:
            _score, source, dataset, variant, class_key, metric = item
            by_generator_metric[(source, variant, metric)].append(item)

        def add_case(item: tuple[tuple[float, float], str, str, str, str, str]) -> None:
            _score, source, dataset, variant, class_key, metric = item
            key = (source, dataset, variant, class_key, metric)
            if key not in cases:
                cases.append(key)

        generators = sorted({(source, variant) for _score, source, _dataset, variant, _class_key, _metric in unique_rows})
        for source, variant in generators:
            for metric in HISTOGRAM_METRICS:
                bucket = by_generator_metric.get((source, variant, metric), [])
                if not bucket:
                    continue
                add_case(bucket[0])
                if len(cases) >= limit:
                    break
            if len(cases) >= limit:
                break
        if len(cases) < limit:
            for source, variant in generators:
                for metric in HISTOGRAM_METRICS:
                    bucket = by_generator_metric.get((source, variant, metric), [])
                    if len(bucket) >= 5:
                        add_case(bucket[len(bucket) // 2])
                    if len(cases) >= limit:
                        break
                if len(cases) >= limit:
                    break
        if len(cases) < limit:
            for item in unique_rows:
                add_case(item)
                if len(cases) >= limit:
                    break

    selected_keys.update(cases)
    for row in all_index_rows:
        key = (row["source"], row["dataset"], row["variant"], row["classKey"], row["metric"])
        row["selected"] = "yes" if key in selected_keys else "no"
        row["selectionMode"] = selection
        row["selectionLimit"] = limit if selection != "all" else "all"
    write_csv(output_dir / "tables" / "appendix_histogram_candidate_index.csv", all_index_rows)
    return cases


def collect_case_metrics(
    path: Path,
    cases: Sequence[tuple[str, str, str, str, str]],
    *,
    reference_scope: bool = False,
) -> dict[tuple[str, str, str, str, str], list[float]]:
    values = {case: [] for case in cases}
    if not path.exists() or not cases:
        return values
    metrics = {case[4] for case in cases}
    available_metrics = metrics & set(csv_header(path))
    if not available_metrics:
        return values
    lookup: dict[tuple[str, ...], set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for source, dataset, variant, class_key, metric in cases:
        if reference_scope:
            lookup[(dataset, class_key, metric)].add((source, dataset, variant, class_key, metric))
        else:
            lookup[(source, dataset, variant, class_key, metric)].add((source, dataset, variant, class_key, metric))
    for row in read_csv_rows(path):
        dataset = str(row.get("dataset") or "")
        class_key = str(row.get("classKey") or "")
        if reference_scope:
            candidate_keys = [
                key
                for metric in available_metrics
                for key in lookup.get((dataset, class_key, metric), ())
            ]
        else:
            source = str(row.get("source") or "")
            variant = str(row.get("variant") or "root")
            candidate_keys = [
                key
                for metric in available_metrics
                for key in lookup.get((source, dataset, variant, class_key, metric), ())
            ]
        for key in candidate_keys:
            metric = key[4]
            value = to_float(row.get(metric))
            if value is not None and value >= -EPSILON:
                values[key].append(max(0.0, value))
    return values


def chunked(items: Sequence[tuple[str, str, str, str, str]], size: int) -> Iterator[Sequence[tuple[str, str, str, str, str]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def plot_hist_pair(
    ax: plt.Axes,
    reference: Sequence[float],
    comparison: Sequence[float],
    metric: str,
    title: str,
    comparison_label: str,
    comparison_color: str,
) -> None:
    if not reference or not comparison:
        ax.axis("off")
        return
    combined = [*reference, *comparison]
    finite = [value for value in combined if math.isfinite(value)]
    if not finite:
        ax.axis("off")
        return
    low = min(finite)
    high = max(finite)
    bins = np.linspace(low, high, 24) if high > low else 12
    ax.hist(reference, bins=bins, density=True, alpha=0.48, label="human-human", color="#4C78A8")
    ax.hist(comparison, bins=bins, density=True, alpha=0.48, label=comparison_label, color=comparison_color)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(metric, fontsize=6.8)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(axis="y", alpha=0.2)


def plot_histogram_appendix(input_dir: Path, args: argparse.Namespace, formats: Sequence[str], dpi: int, output_dir: Path) -> None:
    if not args.include_appendix:
        return
    combined = combined_dir_for(input_dir)
    cases = selected_histogram_cases(input_dir, args.max_histogram_classes, args.histogram_selection, output_dir)
    within_reference_values = collect_case_metrics(combined / "within_reference.csv", cases, reference_scope=True)
    within_comparison_values = collect_case_metrics(combined / "within_comparison.csv", cases)
    between_groups_values = collect_case_metrics(combined / "between_groups.csv", cases)
    manifest_rows = []
    page_size = max(1, args.histogram_page_size)
    cases_per_row = max(1, args.histogram_cases_per_row)
    cases_by_dataset: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for case in cases:
        cases_by_dataset[case[1]].append(case)
    global_page_index = 0
    for dataset in sorted(cases_by_dataset):
        dataset_cases = cases_by_dataset[dataset]
        dataset_page_count = math.ceil(len(dataset_cases) / page_size)
        for dataset_page_index, page_cases in enumerate(chunked(dataset_cases, page_size), start=1):
            global_page_index += 1
            grid_rows = int(math.ceil(len(page_cases) / cases_per_row))
            grid_cols = cases_per_row * 2
            fig, axes = plt.subplots(
                grid_rows,
                grid_cols,
                figsize=(6.1 * cases_per_row, max(3.0, grid_rows * 2.25)),
                squeeze=False,
            )
            for ax in axes.ravel():
                ax.axis("off")
            page_path_base = (
                output_dir
                / "appendix"
                / "histogram_pages"
                / safe_name(dataset)
                / f"histogram_{safe_name(dataset)}_page_{dataset_page_index:04d}"
            )
            for case_index, (source, dataset, variant, class_key, metric) in enumerate(page_cases):
                row_index = case_index // cases_per_row
                case_col = case_index % cases_per_row
                left_ax = axes[row_index][case_col * 2]
                right_ax = axes[row_index][case_col * 2 + 1]
                left_ax.axis("on")
                right_ax.axis("on")
                case = (source, dataset, variant, class_key, metric)
                within_reference = within_reference_values.get(case, [])
                within_comparison = within_comparison_values.get(case, [])
                between_groups = between_groups_values.get(case, [])
                title_prefix = f"{label_from_parts(source, variant)} | {class_key}"
                plot_hist_pair(
                    left_ax,
                    within_reference,
                    within_comparison,
                    metric,
                    f"{title_prefix}\nwithin generated, n={len(within_comparison)}",
                    "generated-generated",
                    "#F58518",
                )
                plot_hist_pair(
                    right_ax,
                    within_reference,
                    between_groups,
                    metric,
                    f"{title_prefix}\nhuman-generated, n={len(between_groups)}",
                    "human-generated",
                    "#E45756",
                )
                if case_index == 0:
                    left_ax.legend(fontsize=6.0, loc="upper right")
                    right_ax.legend(fontsize=6.0, loc="upper right")
                manifest_rows.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "variant": variant,
                        "classKey": class_key,
                        "metric": metric,
                        "plotType": "dataset_paged_within_and_between_histograms",
                        "referenceN": len(within_reference),
                        "withinComparisonN": len(within_comparison),
                        "betweenGroupsN": len(between_groups),
                        "globalPage": global_page_index,
                        "datasetPage": dataset_page_index,
                        "datasetPageCount": dataset_page_count,
                        "panelRow": row_index + 1,
                        "panelColumn": case_col + 1,
                        "path": str(page_path_base.with_suffix(f".{formats[0]}")),
                    }
                )
            fig.suptitle(
                f"Raw Pairwise Histogram Appendix - {dataset} - page {dataset_page_index} of {dataset_page_count}",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            save_figure(
                fig,
                page_path_base,
                formats,
                dpi,
                f"Raw Metric Distribution Appendix: {dataset}",
                "appendix",
                f"Paged histogram appendix for {dataset} with {len(page_cases)} class/metric cases.",
            )
    write_csv(output_dir / "tables" / "appendix_histogram_manifest.csv", manifest_rows)


def write_readme(input_dir: Path, output_dir: Path, formats: Sequence[str]) -> None:
    lines = [
        "RELACC Evaluation Plots v2",
        "===========================",
        "",
        f"Input directory: {input_dir}",
        f"Output formats: {', '.join(formats)}",
        "",
        "This directory contains derived plotting tables and static figures generated",
        "from combined evaluation CSVs. The workflow does not require raw_metrics.jsonl",
        "and does not generate ECDF plots.",
        "",
        "Key tables live under tables/. The top-level plot_manifest.csv lists every",
        "figure produced by this run.",
    ]
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot_guide(input_dir: Path, output_dir: Path) -> None:
    manifest = load_json(input_dir / "manifest.json")
    run = load_json(input_dir / "run.json")
    config = run.get("effectiveConfig", {}) if isinstance(run.get("effectiveConfig"), dict) else {}
    sample_limit = manifest.get("sampleLimitPerClass", config.get("sampleLimitPerClass", "unknown"))
    distribution_limit = manifest.get(
        "distributionSampleLimitPerClass",
        config.get("distributionSampleLimitPerClass", "unknown"),
    )
    rate = manifest.get("rate", config.get("rate", "unknown"))
    summary = manifest.get("summary", config.get("summary", "unknown"))
    raw_jsonl = manifest.get("rawJsonl", config.get("rawJsonl", "unknown"))
    text = f"""# RELACC Evaluation Plot Guide

This guide explains the plots generated by `evaluation_plot_scripts/plot_evaluation_results_v2.py`.
It is written for a reader who has not worked with this project before.

## Run Context

- Input folder: `{input_dir}`
- Combined CSV folder: `{combined_dir_for(input_dir)}`
- Summary gesture method: `{summary}`
- Resampling rate: `{rate}` points per gesture, when resampling was enabled by the evaluator
- Candidate/sample limit per class: `{sample_limit}`
- Direct distribution sample limit per class: `{distribution_limit}`
- Raw JSONL available: `{raw_jsonl}`

The current plotting workflow uses the combined CSV files. It does not require `raw_metrics.jsonl`.

## Core Vocabulary

- **Human/reference samples** are real gestures from the dataset.
- **Generated/comparison samples** are gestures produced by a generator.
- **Summary gesture** is the representative human gesture for a class, usually the medoid in this run.
- **Class** means a gesture label such as `arrow`, `star`, or a character ID.
- **Dataset** means a source dataset such as `1dollar`, `MCYT`, or `MobileTouchDB`.
- **Generator** means the generated source. Variants are shown only when meaningful, for example `scriptstudio/recoTO`; `root` variants are displayed as just the generator name.

## Main CSV Inputs

- `baseline.csv`: human samples compared with the human summary gesture. This estimates intrinsic human-to-summary variation.
- `pairwise.csv`: generated samples compared with the human summary gesture. This estimates generated-to-summary error.
- `within_reference.csv`: human sample pairs compared with other human samples from the same class.
- `within_comparison.csv`: generated sample pairs compared with other generated samples from the same generator/class.
- `between_groups.csv`: human samples compared directly with generated samples from the same class.
- `distribution.csv`: per class and metric distribution summaries comparing within-reference, within-comparison, and between-groups distributions.
- `summary_distribution.csv`: distribution summaries for generated-vs-summary behavior compared with human baseline-vs-summary behavior.
- `*_stats.csv`: compact per class/metric summaries used for fast ranking and boxplot ratios.

## Metric Direction

Almost all metrics are lower-is-better distances or errors. The script excludes `twoThirdsPowerLawR2` from generic lower-is-better rankings because R2 is usually higher-is-better.

## Ranking Tables

The ranking tables live in `tables/`.

- `generator_overall_scores.csv`: one row per generator. Lower aggregate score is better.
- `generator_dataset_scores.csv`: one row per generator and dataset.
- `generator_dataset_metric_scores.csv`: one row per generator, dataset, and metric.
- `metric_family_scores.csv`: one row per generator, dataset, and metric family.

The overall score combines four components:

1. **Summary error score**: generated-to-summary median divided by human-to-summary median, from `pairwise.csv`/`stats.csv` versus `baseline.csv`/`baseline_stats.csv`.
2. **Variability score**: absolute log ratio of generated internal variability to human internal variability, from `within_comparison_stats.csv` versus `within_reference_stats.csv`.
3. **Raw distribution score**: distribution-distance agreement between human-human and generated-generated variability, from `distribution.csv`.
4. **Summary distribution score**: distribution-distance agreement between generated-to-summary and human-to-summary behavior, from `summary_distribution.csv`.

Component scores are normalized ranks within dataset/metric before aggregation. That prevents large-scale metrics like DTW from dominating smaller numeric metrics.

## Ranking Plots

### `ranking/overall_generator_ranking.png`

Horizontal stacked bar chart. Each row is a generator. The x-axis is the sum of normalized component ranks; lower is better. Bar colors show how much each component contributes to the final score.

### `ranking/generator_score_components.png`

Heatmap of generators by score component. Each cell is a normalized rank score. Lower numbers and darker lower-is-better colors indicate better behavior.

### `ranking/dataset_generator_heatmap.png`

Rows are datasets, columns are generators. Each cell is the dataset-level aggregate score. This answers: “Which generator is best for this dataset?”

### `ranking/dataset_generator_rank_heatmap.png`

Same structure as above, but cells are ranks. `1` means the best generator for that dataset.

### `ranking/metric_family_*.png`

Each plot focuses on one metric family: shape, timing, movement, stroke, or DTW. Rows are datasets and columns are generators. Values are family-level aggregate scores.

### `ranking/metric_family_generator_scores.png`

Rows are metric families and columns are generators. This summarizes whether a generator is strong in one family but weak in another.

## Summary Error Plots

These plots compare generated samples against the human summary gesture and normalize by human baseline behavior.

### `summary_error/generated_to_human_summary_ratio.png`

Rows are metrics and columns are generators. A value of `1.0` means generated samples are about as far from the human summary as human samples are. Values above `1.0` mean generated samples are farther from the summary. Values below `1.0` can mean the generator is unusually close to the medoid and may be under-diverse.

### `summary_error/representative_metrics/summary_error_boxplots.png`

Boxplots of class-level generated/human summary error ratios for representative metrics. X-axis is generator. Y-axis is the ratio. The box shows the distribution across classes/datasets.

### `summary_error/all_metrics/summary_error_boxplots_all_metrics.png`

Same as above, but includes every lower-is-better metric. This is detailed and taller, intended for inspection rather than a paper figure.

### `summary_error/representative_metrics/human_baseline_boxplots.png`

Human-to-summary variation by dataset for representative metrics. X-axis is dataset. Y-axis is the median metric value from human samples to the human summary.

### `summary_error/all_metrics/human_baseline_boxplots_all_metrics.png`

All lower-is-better metrics for the same human baseline view.

### `summary_error/human_vs_generated_scatter/*_metrics.png`

Scatter plots by metric family. Each point is a dataset/class/metric/generator median.

- X-axis: human-to-summary class median.
- Y-axis: generated-to-summary class median.
- Dashed diagonal: generated equals human baseline.
- Points above the line indicate generated samples are worse/farther than human baseline for that class/metric.
- Points below the line indicate generated samples are closer to the summary than human samples.

These plots are useful for spotting bias: for example, a generator may look good on easy classes but degrade sharply when the human baseline is already difficult.

## Pairwise Affinity Plots

### `pairwise_affinity/<dataset>/<generator>.png`

Each figure is a small-multiple set of 3x3 annotated affinity matrices for representative metrics.

- Axes: `human`, `generated`, and `human summary`
- `human` ↔ `human`: median within-human distance from `distribution.csv`
- `generated` ↔ `generated`: median within-generated distance from `distribution.csv`
- `human` ↔ `generated`: median direct cross-group distance from `distribution.csv`
- `human` ↔ `human summary`: median human-to-summary baseline from `baseline_stats.csv`
- `generated` ↔ `human summary`: median generated-to-summary value from `stats.csv`

This gives one compact view per dataset and generator that combines “within dataset”, “generated vs human”, and “generated vs human summary” comparisons in the same layout.

## Variability Plots

### `variability/generated_internal_variability_ratio.png`

Rows are metrics and columns are generators. Values are generated internal variability divided by human internal variability.

- Near `1.0`: generated diversity matches human diversity.
- Below `1.0`: generated samples may be too similar to each other.
- Above `1.0`: generated samples may be too noisy or inconsistent.

The color scale is clipped at the 95th percentile so one extreme outlier does not make all other cells visually grey. The printed cell values are still the real unclipped ratios.

### `variability/raw_pairwise_metric_boxplots.png`

Raw row-level generated-to-summary metric distributions from `pairwise.csv` for representative metrics.

### `variability/between_groups_histograms/`

Histogram comparisons of human-human distances against direct human-generated distances. These come from `within_reference.csv` and `between_groups.csv`.

## Distribution Plots

### `distributions/raw_distribution_distance_heatmap.png`

Compares human-human variability distributions with generated-generated variability distributions using Jensen-Shannon divergence from `distribution.csv`.

### `distributions/summary_distribution_distance_heatmap.png`

Compares generated-to-summary distributions against human-to-summary baseline distributions using `summary_distribution.csv`.

### `distributions/distribution_distance_small_multiples.png`

Shows several distribution-distance metrics side-by-side: Jensen-Shannon, Wasserstein, energy distance, KS statistic, and total variation. This checks whether conclusions depend on one distance metric.

### `distributions/normality_diagnostics_*.png`

Heatmaps showing the share of class-runs whose normality test p-value is below 0.05. High values mean the distributions are often non-normal, which supports using medians/IQRs and non-parametric summaries.

### `distributions/skewness_kurtosis_diagnostics_*.png`

Scatter plots of skewness versus kurtosis. These diagnose asymmetric or heavy-tailed metric distributions.

## Class Difficulty Plots

### `classes/most_difficult_classes.png`

Horizontal bar chart of the classes with the highest median generated/human baseline ratio across core metrics.

### `classes/class_difficulty_heatmap_<dataset>.png`

Rows are difficult classes for a dataset. Columns are generators. Values are median generated/human baseline ratios.

## DTW and Variant Plots

### `dtw/dtw_variant_comparison.png`

Grouped bars for DTW variants. This checks whether `dtwDistance`, `ldtwDistance`, `ddtwDistance`, `wdtwDistance`, and `wddtwDistance` agree on generator quality.

### `variants/scriptstudio_variant_comparison.png`

Compares `scriptstudio/recoTO` and `scriptstudio/syntTO` by dataset and metric family.

## Tradeoff Plots

The `tradeoffs/` folder contains metric-level scatter plots comparing ranking components.

- Lower-left is better.
- A generator with low summary error but high variability/distribution score may match the medoid while failing to match human diversity.
- A generator with better distribution score but worse summary error may have human-like spread but be shifted away from the correct summary.

## Appendix Histograms

The appendix histograms are compact dataset-grouped contact sheets. The default `curated` mode selects a balanced set across generators and representative metrics: highly divergent examples plus typical examples. Each dataset gets its own appendix page sequence.

Each selected class/metric case gets two adjacent panels:

- Left panel: blue bars are human-human distances from `within_reference.csv`; orange bars are generated-generated distances from `within_comparison.csv`.
- Right panel: blue bars are the same human-human distances from `within_reference.csv`; red bars are human-generated distances from `between_groups.csv`.

The x-axis is the metric value named below the panel, for example `shapeError` or `dtwDistance`. Lower x-values mean the compared gestures are closer under that metric. The y-axis is histogram density, not number of samples; this lets distributions with different pair counts be overlaid fairly. The `n=...` in each title is the number of raw pairwise rows in the generated or human-generated comparison.

How to read the panels:

- Orange close to blue in the left panel means generated samples have similar internal diversity to humans.
- Orange much narrower than blue means generated samples may be too uniform or collapsed.
- Orange much wider than blue means generated samples may be too noisy internally.
- Red close to blue in the right panel means generated samples sit near the human sample cloud.
- Red shifted far right means generated samples are far from human samples for that class/metric.
- Red far right while orange is narrow often means a generator is internally consistent but consistently wrong.

- `tables/appendix_histogram_candidate_index.csv` lists all candidate histogram cases and whether they were selected.
- `tables/appendix_histogram_manifest.csv` lists the dataset page, panel row, and panel column for each plotted case.
- Use `--max-histogram-classes N` to increase the curated/top cap.
- Use `--histogram-selection top` to focus only on the most divergent examples.
- Use `--histogram-selection all` only when you truly need exhaustive histogram generation; it can still take a long time even though the output is paged.
- Use `--histogram-page-size N` to control how many class/metric cases are placed on each page.

Each appendix histogram uses raw row-level CSVs, not JSONL:

- Human vs generated internal variability: `within_reference.csv` versus `within_comparison.csv`.
- Human-to-generated pairwise distance: `within_reference.csv` versus `between_groups.csv`.

## Caching

Derived ranking tables are cached in `tables/`. On later runs, the script reuses these tables by default and skips recomputing the ranking aggregation. Use `--refresh-cache` after changing the input evaluation folder, scoring logic, or metric-family selection.

Some plots still stream row-level CSVs because they need raw distributions, especially raw boxplots and histograms.
"""
    (output_dir / "PLOT_GUIDE.md").write_text(text, encoding="utf-8")


def write_plot_manifest(output_dir: Path) -> None:
    rows = [
        {
            "path": str(record.path),
            "title": record.title,
            "category": record.category,
            "description": record.description,
        }
        for record in PLOT_MANIFEST
    ]
    write_csv(output_dir / "plot_manifest.csv", rows, ["path", "title", "category", "description"])
    write_csv(output_dir / "tables" / "plot_manifest.csv", rows, ["path", "title", "category", "description"])
    write_csv(output_dir / "tables" / "heatmap_scale_manifest.csv", HEATMAP_SCALE_ROWS)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    formats = tuple(fmt.strip().lower() for fmt in args.format.split(",") if fmt.strip())
    if not formats:
        formats = DEFAULT_FORMATS
    ensure_dir(output_dir)
    metrics = selected_metrics(args)
    tables = None if args.refresh_cache else load_cached_ranking_tables(output_dir)
    if tables is None:
        tables = build_ranking_tables(input_dir, output_dir, metrics)
    else:
        print(f"Using cached derived ranking tables from {output_dir / 'tables'}")
    plot_overall_ranking(tables["overall"], formats, args.dpi, output_dir)
    plot_ranking_heatmaps(tables, formats, args.dpi, output_dir)
    plot_summary_error_outputs(input_dir, tables, formats, args.dpi, output_dir)
    plot_variability_outputs(input_dir, tables, formats, args.dpi, output_dir)
    plot_pairwise_affinity_outputs(input_dir, tables, metrics, formats, args.dpi, output_dir)
    plot_tradeoff_scatters(tables, formats, args.dpi, output_dir)
    plot_human_generated_metric_scatters(input_dir, tables, formats, args.dpi, output_dir)
    plot_distribution_outputs(input_dir, tables, formats, args.dpi, output_dir)
    plot_diagnostics(input_dir, tables, formats, args.dpi, output_dir)
    plot_class_outputs(input_dir, tables, args, formats, args.dpi, output_dir)
    plot_dtw_and_variants(tables, formats, args.dpi, output_dir)
    plot_leaderboard_tables(tables, output_dir)
    build_coverage_outputs(input_dir, tables, formats, args.dpi, output_dir)
    plot_histogram_appendix(input_dir, args, formats, args.dpi, output_dir)
    write_readme(input_dir, output_dir, formats)
    write_plot_guide(input_dir, output_dir)
    write_plot_manifest(output_dir)
    print(f"Wrote {len(PLOT_MANIFEST)} plots and derived tables to {output_dir}")


if __name__ == "__main__":
    main()
