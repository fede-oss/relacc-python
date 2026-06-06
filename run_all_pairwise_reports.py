#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from scipy import stats as scipy_stats

from relacc.distribution_metrics import compute_distribution_metrics

from relacc.dtw import DEFAULT_EXACT_RATE_THRESHOLD, recommended_window
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import METRIC_NAMES, compute_metrics
from relacc.pipeline._common import (
    compute_pair_metrics_from_points,
    format_csv_rows,
    list_csv_files,
    normalize_summary_shape,
    read_points,
    sampling_rate_for_sets,
)
from relacc.pipeline.reporting import (
    CLASS_SCHEME_AUTO,
    CLASS_SCHEMES,
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    ReportingEntry,
    _dataset_and_class_for_relative_path,
    _input_dataset_hint,
    _normalize_class_scheme,
    _normalize_group_by,
)
from relacc.utils.math import MathUtil
from relacc.utils.runlog import (
    add_run_logging_arguments,
    append_run_log,
    build_run_metadata,
    record_effective_config,
    run_logged_experiment,
    sidecar_paths,
    verbosity_from_opt,
    write_run_metadata,
)


DEFAULT_DATASETS_ROOT = Path("/Users/fede/S6-Project/datasets")
DEFAULT_RATE = 24

PAIRWISE_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "pairKey",
    "candidateFile",
    "referenceInput",
    "mode",
    "referenceCount",
    "candidateCount",
    "rate",
    "requestedRate",
    "alignment",
    "summary",
    "popular",
    "dtwWindow",
    "exactDtw",
    *METRIC_NAMES,
)

BASELINE_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "sampleKey",
    "sampleFile",
    "referenceInput",
    "mode",
    "referenceCount",
    "rate",
    "requestedRate",
    "alignment",
    "summary",
    "popular",
    "dtwWindow",
    "exactDtw",
    *METRIC_NAMES,
)

STATS_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "metric",
    "n",
    "finiteN",
    "mean",
    "mdn",
    "sd",
    "min",
    "max",
)

DISTRIBUTION_COLUMNS = (
    "runId",
    "source",
    "dataset",
    "variant",
    "classKey",
    "metric",
    "withinReferenceN",
    "withinReferenceFiniteN",
    "withinReferenceMean",
    "withinReferenceMdn",
    "withinReferenceSd",
    "withinReferenceVariance",
    "withinReferenceMin",
    "withinReferenceMax",
    "withinReferenceQ05",
    "withinReferenceQ25",
    "withinReferenceQ50",
    "withinReferenceQ75",
    "withinReferenceQ95",
    "withinReferenceSkewness",
    "withinReferenceKurtosis",
    "withinReferenceNormalityPValue",
    "withinComparisonN",
    "withinComparisonFiniteN",
    "withinComparisonMean",
    "withinComparisonMdn",
    "withinComparisonSd",
    "withinComparisonVariance",
    "withinComparisonMin",
    "withinComparisonMax",
    "withinComparisonQ05",
    "withinComparisonQ25",
    "withinComparisonQ50",
    "withinComparisonQ75",
    "withinComparisonQ95",
    "withinComparisonSkewness",
    "withinComparisonKurtosis",
    "withinComparisonNormalityPValue",
    "betweenGroupsN",
    "betweenGroupsFiniteN",
    "betweenGroupsMean",
    "betweenGroupsMdn",
    "betweenGroupsSd",
    "betweenGroupsVariance",
    "betweenGroupsMin",
    "betweenGroupsMax",
    "betweenGroupsQ05",
    "betweenGroupsQ25",
    "betweenGroupsQ50",
    "betweenGroupsQ75",
    "betweenGroupsQ95",
    "betweenGroupsSkewness",
    "betweenGroupsKurtosis",
    "betweenGroupsNormalityPValue",
    "wassersteinDistance",
    "normalizedWassersteinDistance",
    "energyDistance",
    "ksStatistic",
    "ksPValue",
    "betweenGroupsMeanDelta",
    "betweenGroupsMdnDelta",
    "betweenGroupsSdDelta",
    "withinComparisonToReferenceMeanDelta",
    "withinComparisonToReferenceMeanRatio",
    "withinComparisonToReferenceMdnDelta",
    "withinComparisonToReferenceMdnRatio",
    "withinComparisonToReferenceSdDelta",
    "withinComparisonToReferenceSdRatio",
)


def _int_cast(value):
    if value is None or value == "":
        return None
    return int(value)


def _candidate_groups(source_root: Path, dataset_name: str):
    dataset_root = source_root / dataset_name
    if not dataset_root.exists():
        return []

    groups = []
    if sorted(dataset_root.glob("*.csv")):
        groups.append((dataset_root, None))

    for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if sorted(child.glob("*.csv")):
            groups.append((child, child.name))
    return groups


def _run_id(source_name: str, dataset_name: str, variant: str | None) -> str:
    if variant:
        return f"{source_name}/{dataset_name}/{variant}"
    return f"{source_name}/{dataset_name}"


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(seconds + 0.5)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


class ProgressReporter:
    def __init__(self, total_candidates: int, total_runs: int, verbosity: int = 0):
        self.total_candidates = total_candidates
        self.total_runs = total_runs
        self.verbosity = verbosity
        self.completed_candidates = 0
        self.completed_runs = 0
        self.started_at = time.perf_counter()

    def _line(self, prefix: str) -> str:
        elapsed = time.perf_counter() - self.started_at
        if self.completed_candidates > 0:
            rate = self.completed_candidates / elapsed
            remaining = max(0, self.total_candidates - self.completed_candidates)
            eta = remaining / rate if rate > 0 else float("nan")
        else:
            rate = 0.0
            eta = float("nan")
        return (
            f"{prefix} progress={self.completed_candidates}/{self.total_candidates} "
            f"candidates runs={self.completed_runs}/{self.total_runs} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
            f"rate={rate:.2f}/s"
        )

    def start_run(self, run_id: str, candidate_count: int) -> None:
        if self.verbosity < 2:
            return
        print(
            self._line(
                f"starting {run_id} ({candidate_count} candidates);"
            ),
            flush=True,
        )

    def finish_class(self, run_id: str, class_key: str, candidate_count: int) -> None:
        self.completed_candidates += candidate_count
        if self.verbosity < 2:
            return
        print(
            self._line(
                f"finished {run_id} class={class_key} rows={candidate_count};"
            ),
            flush=True,
        )

    def skip_class(self, run_id: str, class_key: str, reason: str) -> None:
        if self.verbosity < 1:
            return
        print(f"skipped {run_id} class={class_key}: {reason}", flush=True)

    def finish_run(self, run_id: str, row_count: int) -> None:
        self.completed_runs += 1
        if self.verbosity < 2:
            return
        print(
            self._line(f"finished {run_id} rows={row_count};"),
            flush=True,
        )


def _safe_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "_"


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


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_csv_rows(rows, columns), encoding="utf-8")


def _effective_dtw_window(rate: int, requested_window: int | None, exact_dtw: bool):
    if exact_dtw:
        return None
    if requested_window is not None:
        return requested_window
    if rate <= DEFAULT_EXACT_RATE_THRESHOLD:
        return None
    return recommended_window(rate)


def _load_reporting_entries(
    input_path: Path,
    group_by: str,
    class_scheme: str,
    warnings: List[Dict[str, str]],
) -> List[ReportingEntry]:
    dataset_hint = _input_dataset_hint(str(input_path))
    entries: List[ReportingEntry] = []
    for key, path in sorted(list_csv_files(input_path).items()):
        try:
            points = read_points(str(path))
            dataset_key, class_key = _dataset_and_class_for_relative_path(
                key,
                group_by,
                class_scheme,
                dataset_hint,
            )
            entries.append(
                ReportingEntry(
                    key=key,
                    path=str(path),
                    dataset_key=dataset_key,
                    class_key=class_key,
                    points=points,
                )
            )
        except Exception as exc:
            warnings.append(
                {
                    "input": str(input_path),
                    "file": str(path),
                    "error": str(exc),
                }
            )
    return sorted(entries, key=lambda entry: (entry.class_key, entry.key))


def _entries_by_class(entries: Iterable[ReportingEntry]):
    grouped: Dict[str, List[ReportingEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.class_key].append(entry)
    return dict(grouped)


def _validate_bounded_stats(
    stats_row: Dict[str, object],
    stat_names: Sequence[str],
    context: str,
) -> Dict[str, object]:
    minimum = _numeric_value(stats_row.get("min"))
    maximum = _numeric_value(stats_row.get("max"))
    if minimum is None or maximum is None:
        return stats_row

    for stat_name in stat_names:
        value = _numeric_value(stats_row.get(stat_name))
        if value is not None and not minimum <= value <= maximum:
            raise ValueError(
                "Invalid summary statistics for %s: %s=%s is outside min=%s and max=%s."
                % (context, stat_name, value, minimum, maximum)
            )
    return stats_row


def _summary_stats(
    rows: Sequence[Dict[str, object]],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    round_precision: int | None,
):
    stats_rows = []
    for metric_name in METRIC_NAMES:
        values = [row.get(metric_name) for row in rows]
        finite_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        finite_n = len(finite_values)
        if finite_n == 0:
            mean = mdn = sd = minimum = maximum = None
        else:
            mean = statistics.fmean(finite_values)
            mdn = statistics.median(finite_values)
            sd = statistics.stdev(finite_values) if finite_n > 1 else 0.0
            minimum = min(finite_values)
            maximum = max(finite_values)

        def rounded(value):
            if value is None:
                return None
            return MathUtil.roundTo(value, round_precision)

        stats_row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": variant,
            "classKey": class_key,
            "metric": metric_name,
            "n": len(values),
            "finiteN": finite_n,
            "mean": rounded(mean),
            "mdn": rounded(mdn),
            "sd": rounded(sd),
            "min": rounded(minimum),
            "max": rounded(maximum),
        }
        stats_rows.append(
            _validate_bounded_stats(
                stats_row,
                ("mean", "mdn"),
                f"{run_id}/{class_key}/{metric_name}",
            )
        )
    return stats_rows


def _numeric_value(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_metric_values(rows: Sequence[Dict[str, object]], metric_name: str) -> List[float]:
    values = []
    for row in rows:
        value = _numeric_value(row.get(metric_name))
        if value is not None:
            values.append(value)
    return values


def _quantile(sorted_values: Sequence[float], fraction: float):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _shape_statistic(values: Sequence[float], fn):
    if len(values) < 3 or len(set(values)) <= 1:
        return None
    value = float(fn(values))
    return value if math.isfinite(value) else None


def _normality_p_value(values: Sequence[float]):
    if len(values) < 8 or len(set(values)) <= 1:
        return None
    value = float(scipy_stats.normaltest(values).pvalue)
    return value if math.isfinite(value) else None


def _distribution_summary(values: Sequence[float], total_n: int, round_precision: int | None):
    finite_values = list(values)

    def rounded(value):
        if value is None:
            return None
        return MathUtil.roundTo(value, round_precision)

    if not finite_values:
        return {
            "n": total_n,
            "finiteN": 0,
            "mean": None,
            "mdn": None,
            "sd": None,
            "variance": None,
            "min": None,
            "max": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "skewness": None,
            "kurtosis": None,
            "normalityPValue": None,
        }

    sorted_values = sorted(finite_values)
    variance = statistics.variance(finite_values) if len(finite_values) > 1 else 0.0
    sd = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0
    summary = {
        "n": total_n,
        "finiteN": len(finite_values),
        "mean": rounded(statistics.fmean(finite_values)),
        "mdn": rounded(statistics.median(finite_values)),
        "sd": rounded(sd),
        "variance": rounded(variance),
        "min": rounded(min(finite_values)),
        "max": rounded(max(finite_values)),
        "q05": rounded(_quantile(sorted_values, 0.05)),
        "q25": rounded(_quantile(sorted_values, 0.25)),
        "q50": rounded(_quantile(sorted_values, 0.50)),
        "q75": rounded(_quantile(sorted_values, 0.75)),
        "q95": rounded(_quantile(sorted_values, 0.95)),
        "skewness": rounded(
            _shape_statistic(
                finite_values,
                lambda series: scipy_stats.skew(series, bias=False),
            )
        ),
        "kurtosis": rounded(
            _shape_statistic(
                finite_values,
                lambda series: scipy_stats.kurtosis(series, fisher=True, bias=False),
            )
        ),
        "normalityPValue": rounded(_normality_p_value(finite_values)),
    }
    return _validate_bounded_stats(
        summary,
        ("mean", "mdn", "q05", "q25", "q50", "q75", "q95"),
        "distribution summary",
    )


def _ratio(numerator: float | None, denominator: float | None):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _delta(left: float | None, right: float | None):
    if left is None or right is None:
        return None
    return left - right


def _lightweight_distribution_rows(
    candidate_rows: Sequence[Dict[str, object]],
    candidate_stats_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    baseline_stats_rows: Sequence[Dict[str, object]],
    within_comparison_rows: Sequence[Dict[str, object]],
    within_comparison_stats_rows: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
    round_precision: int | None,
) -> List[Dict[str, object]]:
    def rounded(value):
        if value is None:
            return None
        return MathUtil.roundTo(value, round_precision)

    distribution_rows = []
    metadata_row = (
        candidate_rows[0]
        if candidate_rows
        else within_comparison_rows[0]
        if within_comparison_rows
        else baseline_rows[0]
        if baseline_rows
        else {}
    )
    for metric_name in metric_names:
        within_reference_values = _finite_metric_values(baseline_rows, metric_name)
        within_comparison_values = _finite_metric_values(
            within_comparison_rows,
            metric_name,
        )
        between_group_values = _finite_metric_values(candidate_rows, metric_name)
        within_reference_stats = _distribution_summary(
            within_reference_values,
            len(baseline_rows),
            round_precision,
        )
        within_comparison_stats = _distribution_summary(
            within_comparison_values,
            len(within_comparison_rows),
            round_precision,
        )
        between_group_stats = _distribution_summary(
            between_group_values,
            len(candidate_rows),
            round_precision,
        )
        distribution_metrics = (
            compute_distribution_metrics(
                within_reference_values,
                between_group_values,
                round_precision=round_precision,
            )
            if within_reference_values and between_group_values
            else {}
        )
        within_reference_mean = within_reference_stats.get("mean")
        within_comparison_mean = within_comparison_stats.get("mean")
        between_group_mean = between_group_stats.get("mean")
        within_reference_mdn = within_reference_stats.get("mdn")
        within_comparison_mdn = within_comparison_stats.get("mdn")
        between_group_mdn = between_group_stats.get("mdn")
        within_reference_sd = within_reference_stats.get("sd")
        within_comparison_sd = within_comparison_stats.get("sd")
        between_group_sd = between_group_stats.get("sd")
        wasserstein = distribution_metrics.get("wassersteinDistance")

        distribution_rows.append(
            {
                "runId": metadata_row.get("runId"),
                "source": metadata_row.get("source"),
                "dataset": metadata_row.get("dataset"),
                "variant": metadata_row.get("variant"),
                "classKey": metadata_row.get("classKey"),
                "metric": metric_name,
                "withinReferenceN": within_reference_stats.get("n"),
                "withinReferenceFiniteN": within_reference_stats.get("finiteN"),
                "withinReferenceMean": within_reference_mean,
                "withinReferenceMdn": within_reference_mdn,
                "withinReferenceSd": within_reference_sd,
                "withinReferenceVariance": within_reference_stats.get("variance"),
                "withinReferenceMin": within_reference_stats.get("min"),
                "withinReferenceMax": within_reference_stats.get("max"),
                "withinReferenceQ05": within_reference_stats.get("q05"),
                "withinReferenceQ25": within_reference_stats.get("q25"),
                "withinReferenceQ50": within_reference_stats.get("q50"),
                "withinReferenceQ75": within_reference_stats.get("q75"),
                "withinReferenceQ95": within_reference_stats.get("q95"),
                "withinReferenceSkewness": within_reference_stats.get("skewness"),
                "withinReferenceKurtosis": within_reference_stats.get("kurtosis"),
                "withinReferenceNormalityPValue": within_reference_stats.get(
                    "normalityPValue"
                ),
                "withinComparisonN": within_comparison_stats.get("n"),
                "withinComparisonFiniteN": within_comparison_stats.get("finiteN"),
                "withinComparisonMean": within_comparison_mean,
                "withinComparisonMdn": within_comparison_mdn,
                "withinComparisonSd": within_comparison_sd,
                "withinComparisonVariance": within_comparison_stats.get("variance"),
                "withinComparisonMin": within_comparison_stats.get("min"),
                "withinComparisonMax": within_comparison_stats.get("max"),
                "withinComparisonQ05": within_comparison_stats.get("q05"),
                "withinComparisonQ25": within_comparison_stats.get("q25"),
                "withinComparisonQ50": within_comparison_stats.get("q50"),
                "withinComparisonQ75": within_comparison_stats.get("q75"),
                "withinComparisonQ95": within_comparison_stats.get("q95"),
                "withinComparisonSkewness": within_comparison_stats.get("skewness"),
                "withinComparisonKurtosis": within_comparison_stats.get("kurtosis"),
                "withinComparisonNormalityPValue": within_comparison_stats.get(
                    "normalityPValue"
                ),
                "betweenGroupsN": between_group_stats.get("n"),
                "betweenGroupsFiniteN": between_group_stats.get("finiteN"),
                "betweenGroupsMean": between_group_mean,
                "betweenGroupsMdn": between_group_mdn,
                "betweenGroupsSd": between_group_sd,
                "betweenGroupsVariance": between_group_stats.get("variance"),
                "betweenGroupsMin": between_group_stats.get("min"),
                "betweenGroupsMax": between_group_stats.get("max"),
                "betweenGroupsQ05": between_group_stats.get("q05"),
                "betweenGroupsQ25": between_group_stats.get("q25"),
                "betweenGroupsQ50": between_group_stats.get("q50"),
                "betweenGroupsQ75": between_group_stats.get("q75"),
                "betweenGroupsQ95": between_group_stats.get("q95"),
                "betweenGroupsSkewness": between_group_stats.get("skewness"),
                "betweenGroupsKurtosis": between_group_stats.get("kurtosis"),
                "betweenGroupsNormalityPValue": between_group_stats.get(
                    "normalityPValue"
                ),
                "wassersteinDistance": wasserstein,
                "normalizedWassersteinDistance": rounded(
                    _ratio(wasserstein, within_reference_sd)
                ),
                "energyDistance": distribution_metrics.get("energyDistance"),
                "ksStatistic": distribution_metrics.get("ksStatistic"),
                "ksPValue": distribution_metrics.get("ksPValue"),
                "betweenGroupsMeanDelta": rounded(
                    _delta(between_group_mean, within_reference_mean)
                ),
                "betweenGroupsMdnDelta": rounded(
                    _delta(between_group_mdn, within_reference_mdn)
                ),
                "betweenGroupsSdDelta": rounded(
                    _delta(between_group_sd, within_reference_sd)
                ),
                "withinComparisonToReferenceMeanDelta": rounded(
                    _delta(within_comparison_mean, within_reference_mean)
                ),
                "withinComparisonToReferenceMeanRatio": rounded(
                    _ratio(within_comparison_mean, within_reference_mean)
                ),
                "withinComparisonToReferenceMdnDelta": rounded(
                    _delta(within_comparison_mdn, within_reference_mdn)
                ),
                "withinComparisonToReferenceMdnRatio": rounded(
                    _ratio(within_comparison_mdn, within_reference_mdn)
                ),
                "withinComparisonToReferenceSdDelta": rounded(
                    _delta(within_comparison_sd, within_reference_sd)
                ),
                "withinComparisonToReferenceSdRatio": rounded(
                    _ratio(within_comparison_sd, within_reference_sd)
                ),
            }
        )
    return distribution_rows


def _compare_class(
    reference_entries: Sequence[ReportingEntry],
    candidate_entries: Sequence[ReportingEntry],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    reference_input: Path,
    rate: int | None,
    alignment: int,
    summary_shape: str | None,
    popular: bool,
    round_precision: int | None,
    metric_names: Sequence[str],
    dtw_window: int | None,
    exact_dtw: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    reference_points = [entry.points for entry in reference_entries]
    effective_rate = sampling_rate_for_sets(reference_points, rate)
    selected_dtw_window = _effective_dtw_window(effective_rate, dtw_window, exact_dtw)
    summary_label = f"{dataset_name}/{class_key}/human-summary"
    reference_gestures = [
        Gesture(entry.points, summary_label, effective_rate)
        for entry in reference_entries
    ]
    reference_summary = SummaryGesture(
        reference_gestures,
        alignment,
        summary_shape,
        popular,
    )

    rows = []
    for candidate_entry in candidate_entries:
        candidate = Gesture(candidate_entry.points, summary_label, effective_rate)
        metric_values = compute_metrics(
            candidate,
            reference_summary,
            round_precision=round_precision,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
        )
        row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": variant,
            "classKey": class_key,
            "pairKey": Path(candidate_entry.key).with_suffix("").as_posix(),
            "candidateFile": candidate_entry.path,
            "referenceInput": str(reference_input),
            "mode": "reference-summary",
            "referenceCount": len(reference_entries),
            "candidateCount": len(candidate_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(metric_values)
        rows.append(row)

    within_comparison_rows = []
    for left_entry, right_entry in combinations(candidate_entries, 2):
        forward_values = compute_pair_metrics_from_points(
            left_entry.points,
            right_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        backward_values = compute_pair_metrics_from_points(
            right_entry.points,
            left_entry.points,
            class_key,
            effective_rate,
            alignment_type=alignment,
            summary_shape=summary_shape,
            popular_shape=popular,
            round_precision=None,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
            exact_dtw=exact_dtw,
        )
        row = {
            "runId": run_id,
            "source": source_name,
            "dataset": dataset_name,
            "variant": variant,
            "classKey": class_key,
            "pairKey": "%s::%s"
            % (
                Path(left_entry.key).with_suffix("").as_posix(),
                Path(right_entry.key).with_suffix("").as_posix(),
            ),
            "candidateFile": "%s::%s" % (left_entry.path, right_entry.path),
            "referenceInput": str(reference_input),
            "mode": "within-comparison",
            "referenceCount": len(reference_entries),
            "candidateCount": len(candidate_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        for metric_name in metric_names:
            row[metric_name] = MathUtil.roundTo(
                (forward_values[metric_name] + backward_values[metric_name]) / 2.0,
                round_precision,
            )
        within_comparison_rows.append(row)

    stats_rows = _summary_stats(
        rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        round_precision,
    )
    within_comparison_stats_rows = _summary_stats(
        within_comparison_rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        round_precision,
    )
    metadata = {
        "runId": run_id,
        "source": source_name,
        "dataset": dataset_name,
        "variant": variant,
        "classKey": class_key,
        "mode": "reference-summary",
        "referenceCount": len(reference_entries),
        "candidateCount": len(candidate_entries),
        "withinComparisonPairs": len(within_comparison_rows),
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment,
        "summary": summary_shape,
        "popular": bool(popular),
        "roundPrecision": round_precision,
        "metricNames": list(metric_names),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }
    return rows, stats_rows, within_comparison_rows, within_comparison_stats_rows, metadata


def _baseline_stats(
    rows: Sequence[Dict[str, object]],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    round_precision: int | None,
):
    return _summary_stats(
        rows,
        run_id,
        source_name,
        dataset_name,
        variant,
        class_key,
        round_precision,
    )


def _compare_human_baseline_class(
    reference_entries: Sequence[ReportingEntry],
    run_id: str,
    source_name: str,
    dataset_name: str,
    variant: str | None,
    class_key: str,
    reference_input: Path,
    rate: int | None,
    alignment: int,
    summary_shape: str | None,
    popular: bool,
    round_precision: int | None,
    metric_names: Sequence[str],
    dtw_window: int | None,
    exact_dtw: bool,
) -> tuple[list[dict], list[dict], dict]:
    baseline_source_name = "human"
    reference_points = [entry.points for entry in reference_entries]
    effective_rate = sampling_rate_for_sets(reference_points, rate)
    selected_dtw_window = _effective_dtw_window(effective_rate, dtw_window, exact_dtw)
    summary_label = f"{dataset_name}/{class_key}/human-summary"
    reference_gestures = [
        Gesture(entry.points, summary_label, effective_rate)
        for entry in reference_entries
    ]
    reference_summary = SummaryGesture(
        reference_gestures,
        alignment,
        summary_shape,
        popular,
    )

    rows = []
    for reference_entry, reference_gesture in zip(reference_entries, reference_gestures):
        metric_values = compute_metrics(
            reference_gesture,
            reference_summary,
            round_precision=round_precision,
            metric_names=metric_names,
            dtw_window=selected_dtw_window,
        )
        row = {
            "runId": run_id,
            "source": baseline_source_name,
            "dataset": dataset_name,
            "variant": variant,
            "classKey": class_key,
            "sampleKey": Path(reference_entry.key).with_suffix("").as_posix(),
            "sampleFile": reference_entry.path,
            "referenceInput": str(reference_input),
            "mode": "human-summary-baseline",
            "referenceCount": len(reference_entries),
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment,
            "summary": summary_shape,
            "popular": bool(popular),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        }
        row.update(metric_values)
        rows.append(row)

    stats_rows = _baseline_stats(
        rows,
        run_id,
        baseline_source_name,
        dataset_name,
        variant,
        class_key,
        round_precision,
    )
    metadata = {
        "runId": run_id,
        "source": source_name,
        "baselineSource": baseline_source_name,
        "dataset": dataset_name,
        "variant": variant,
        "classKey": class_key,
        "mode": "human-summary-baseline",
        "referenceCount": len(reference_entries),
        "baselineRows": len(rows),
        "rate": effective_rate,
        "requestedRate": rate,
        "alignment": alignment,
        "summary": summary_shape,
        "popular": bool(popular),
        "roundPrecision": round_precision,
        "metricNames": list(metric_names),
        "dtwWindow": selected_dtw_window,
        "exactDtw": bool(exact_dtw),
    }
    return rows, stats_rows, metadata


def _output_run_dir(output_root: Path, source_name: str, dataset_name: str, variant: str | None):
    run_dir = output_root / source_name / dataset_name
    if variant:
        run_dir = run_dir / variant
    return run_dir


def _count_run_candidates(
    candidate_input: Path,
    group_by: str,
    class_scheme: str,
    warnings: List[Dict[str, str]],
) -> tuple[int, Dict[str, int]]:
    dataset_hint = _input_dataset_hint(str(candidate_input))
    counts: Dict[str, int] = defaultdict(int)
    for key, path in sorted(list_csv_files(candidate_input).items()):
        try:
            _, class_key = _dataset_and_class_for_relative_path(
                key,
                group_by,
                class_scheme,
                dataset_hint,
            )
            counts[class_key] += 1
        except Exception as exc:
            warnings.append(
                {
                    "input": str(candidate_input),
                    "file": str(path),
                    "phase": "planning",
                    "error": str(exc),
                }
            )
    return sum(counts.values()), dict(counts)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run all-files human-summary comparisons for every generated "
            "source/dataset/class, plus a lightweight human-summary baseline. "
            "This intentionally does not compute distribution cross-products."
        )
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory containing humans and generated dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where pairwise report artifacts will be written.",
    )
    parser.add_argument(
        "--rate",
        default=str(DEFAULT_RATE),
        help="Fixed resampling rate for all comparisons. Default: 24.",
    )
    parser.add_argument("--round", default="3", help="Decimal precision for metrics.")
    parser.add_argument(
        "--alignment",
        default=str(PtAlignType.CHRONOLOGICAL),
        help="Alignment mode value accepted by the existing pipelines.",
    )
    parser.add_argument("--summary", default=None, help="Optional summary-shape mode.")
    parser.add_argument("--popular", action="store_true", help="Use popular stroke count.")
    parser.add_argument("--exact-dtw", action="store_true", help="Disable DTW windowing.")
    parser.add_argument("--dtw-window", default=None, help="Optional explicit DTW window.")
    parser.add_argument(
        "--group-by",
        default=GROUP_BY_FILENAME_LABEL,
        choices=GROUP_BY_MODES,
        help="Grouping mode for deriving dataset/class keys.",
    )
    parser.add_argument(
        "--class-scheme",
        default=CLASS_SCHEME_AUTO,
        choices=CLASS_SCHEMES,
        help="Class-label extraction scheme. Default: auto.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Optional comma-separated generated source names to run.",
    )
    add_run_logging_arguments(parser)
    return parser


def _run_reports(opt, output_root: Path, paths=None, metadata=None):
    datasets_root = Path(opt.datasets_root)
    humans_root = datasets_root / "humans"

    rate = _int_cast(opt.rate)
    round_precision = _int_cast(opt.round)
    alignment = _int_cast(opt.alignment)
    dtw_window = _int_cast(opt.dtw_window)
    summary_shape = normalize_summary_shape(opt.summary)
    group_by = _normalize_group_by(opt.group_by)
    class_scheme = _normalize_class_scheme(opt.class_scheme)
    metric_names = METRIC_NAMES

    if dtw_window is not None and opt.exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")
    if not humans_root.exists():
        raise FileNotFoundError(f"Missing humans dataset root: {humans_root}")

    selected_datasets = (
        {item.strip() for item in opt.datasets.split(",") if item.strip()}
        if opt.datasets
        else None
    )
    selected_sources = (
        {item.strip() for item in opt.sources.split(",") if item.strip()}
        if opt.sources
        else None
    )
    record_effective_config(
        paths or {},
        metadata,
        {
            "datasetsRoot": str(datasets_root),
            "outputDir": str(output_root),
            "rate": rate,
            "roundPrecision": round_precision,
            "alignment": alignment,
            "summary": summary_shape,
            "popular": bool(opt.popular),
            "exactDtw": bool(opt.exact_dtw),
            "dtwWindow": dtw_window,
            "groupBy": group_by,
            "classScheme": class_scheme,
            "datasets": sorted(selected_datasets) if selected_datasets else None,
            "sources": sorted(selected_sources) if selected_sources else None,
            "verbosity": verbosity_from_opt(opt),
        },
    )

    manifest = {
        "datasetsRoot": str(datasets_root),
        "outputDir": str(output_root),
        "mode": "all-files-reference-summary-pairwise",
        "baselineMode": "human-summary-baseline",
        "rate": rate,
        "roundPrecision": round_precision,
        "alignment": alignment,
        "summary": summary_shape,
        "popular": bool(opt.popular),
        "exactDtw": bool(opt.exact_dtw),
        "dtwWindow": dtw_window,
        "groupBy": group_by,
        "classScheme": class_scheme,
        "metricNames": list(metric_names),
        "runs": [],
        "warnings": [],
    }

    human_dataset_dirs = [
        path
        for path in sorted(humans_root.iterdir())
        if path.is_dir() and (selected_datasets is None or path.name in selected_datasets)
    ]
    source_roots = [
        path
        for path in sorted(datasets_root.iterdir())
        if path.is_dir()
        and path.name != "humans"
        and (selected_sources is None or path.name in selected_sources)
    ]

    planned_runs = []
    planning_warnings: List[Dict[str, str]] = []
    for human_dataset_dir in human_dataset_dirs:
        dataset_name = human_dataset_dir.name
        reference_input = human_dataset_dir / "realTO"
        if not reference_input.exists():
            continue
        for source_root in source_roots:
            source_name = source_root.name
            for candidate_input, variant in _candidate_groups(source_root, dataset_name):
                candidate_count, class_counts = _count_run_candidates(
                    candidate_input,
                    group_by,
                    class_scheme,
                    planning_warnings,
                )
                planned_runs.append(
                    {
                        "runId": _run_id(source_name, dataset_name, variant),
                        "dataset": dataset_name,
                        "source": source_name,
                        "variant": variant,
                        "candidateInput": candidate_input,
                        "candidateCount": candidate_count,
                        "classCounts": class_counts,
                    }
                )
    manifest["planningWarnings"] = planning_warnings
    manifest["plannedRunCount"] = len(planned_runs)
    manifest["plannedCandidateCount"] = sum(
        item["candidateCount"] for item in planned_runs
    )

    progress = ProgressReporter(
        total_candidates=manifest["plannedCandidateCount"],
        total_runs=manifest["plannedRunCount"],
        verbosity=verbosity_from_opt(opt),
    )
    all_distribution_rows: List[Dict[str, object]] = []
    if verbosity_from_opt(opt) >= 2:
        print(
            (
                f"planned {manifest['plannedRunCount']} runs and "
                f"{manifest['plannedCandidateCount']} candidate comparisons"
            ),
            flush=True,
        )

    for human_dataset_dir in human_dataset_dirs:
        dataset_name = human_dataset_dir.name
        reference_input = human_dataset_dir / "realTO"
        if not reference_input.exists():
            manifest["warnings"].append(
                {
                    "dataset": dataset_name,
                    "input": str(reference_input),
                    "error": "missing reference realTO directory",
                }
            )
            continue

        reference_entries = _load_reporting_entries(
            reference_input,
            group_by,
            class_scheme,
            manifest["warnings"],
        )
        references_by_class = _entries_by_class(reference_entries)

        for source_root in source_roots:
            source_name = source_root.name
            for candidate_input, variant in _candidate_groups(source_root, dataset_name):
                run_id = _run_id(source_name, dataset_name, variant)
                run_dir = _output_run_dir(output_root, source_name, dataset_name, variant)
                class_dir = run_dir / "classes"
                run_dir.mkdir(parents=True, exist_ok=True)

                planned_candidate_count, _ = _count_run_candidates(
                    candidate_input,
                    group_by,
                    class_scheme,
                    manifest["warnings"],
                )
                progress.start_run(run_id, planned_candidate_count)

                candidate_entries = _load_reporting_entries(
                    candidate_input,
                    group_by,
                    class_scheme,
                    manifest["warnings"],
                )
                candidates_by_class = _entries_by_class(candidate_entries)

                run_rows: List[Dict[str, object]] = []
                run_stats: List[Dict[str, object]] = []
                run_baseline_rows: List[Dict[str, object]] = []
                run_baseline_stats: List[Dict[str, object]] = []
                run_within_comparison_rows: List[Dict[str, object]] = []
                run_within_comparison_stats: List[Dict[str, object]] = []
                run_distribution_rows: List[Dict[str, object]] = []
                class_manifests = []
                skipped_classes = []

                for class_key in sorted(
                    set(references_by_class.keys()) | set(candidates_by_class.keys())
                ):
                    class_references = references_by_class.get(class_key, [])
                    class_candidates = candidates_by_class.get(class_key, [])
                    if len(class_references) == 0 or len(class_candidates) == 0:
                        reason = (
                            "missingReference"
                            if len(class_references) == 0
                            else "missingComparison"
                        )
                        skipped_classes.append(
                            {
                                "classKey": class_key,
                                "referenceCount": len(class_references),
                                "candidateCount": len(class_candidates),
                                "reason": reason,
                            }
                        )
                        progress.skip_class(run_id, class_key, reason)
                        continue

                    (
                        rows,
                        stats_rows,
                        within_comparison_rows,
                        within_comparison_stats_rows,
                        class_metadata,
                    ) = _compare_class(
                        class_references,
                        class_candidates,
                        run_id,
                        source_name,
                        dataset_name,
                        variant,
                        class_key,
                        reference_input,
                        rate,
                        alignment,
                        summary_shape,
                        bool(opt.popular),
                        round_precision,
                        metric_names,
                        dtw_window,
                        bool(opt.exact_dtw),
                    )
                    baseline_rows, baseline_stats_rows, baseline_metadata = (
                        _compare_human_baseline_class(
                            class_references,
                            run_id,
                            source_name,
                            dataset_name,
                            variant,
                            class_key,
                            reference_input,
                            rate,
                            alignment,
                            summary_shape,
                            bool(opt.popular),
                            round_precision,
                            metric_names,
                            dtw_window,
                            bool(opt.exact_dtw),
                        )
                    )

                    safe_class_key = _safe_path_part(class_key)
                    class_output_dir = class_dir / safe_class_key
                    _write_json(
                        class_output_dir / "pairwise.json",
                        {
                            "metadata": class_metadata,
                            "pairs": rows,
                            "stats": stats_rows,
                        },
                    )
                    _write_csv(class_output_dir / "pairwise.csv", rows, PAIRWISE_COLUMNS)
                    _write_csv(class_output_dir / "stats.csv", stats_rows, STATS_COLUMNS)
                    _write_json(
                        class_output_dir / "baseline.json",
                        {
                            "metadata": baseline_metadata,
                            "baseline": baseline_rows,
                            "stats": baseline_stats_rows,
                        },
                    )
                    _write_csv(
                        class_output_dir / "baseline.csv",
                        baseline_rows,
                        BASELINE_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "baseline_stats.csv",
                        baseline_stats_rows,
                        STATS_COLUMNS,
                    )
                    class_distribution_rows = _lightweight_distribution_rows(
                        rows,
                        stats_rows,
                        baseline_rows,
                        baseline_stats_rows,
                        within_comparison_rows,
                        within_comparison_stats_rows,
                        metric_names,
                        round_precision,
                    )
                    _write_csv(
                        class_output_dir / "distribution.csv",
                        class_distribution_rows,
                        DISTRIBUTION_COLUMNS,
                    )

                    run_rows.extend(rows)
                    run_stats.extend(stats_rows)
                    run_baseline_rows.extend(baseline_rows)
                    run_baseline_stats.extend(baseline_stats_rows)
                    run_within_comparison_rows.extend(within_comparison_rows)
                    run_within_comparison_stats.extend(within_comparison_stats_rows)
                    run_distribution_rows.extend(class_distribution_rows)
                    class_manifests.append(
                        {
                            **class_metadata,
                            "baselineMode": baseline_metadata["mode"],
                            "outputDir": str(class_output_dir),
                            "pairwiseRows": len(rows),
                            "statsRows": len(stats_rows),
                            "withinComparisonRows": len(within_comparison_rows),
                            "withinComparisonStatsRows": len(
                                within_comparison_stats_rows
                            ),
                            "baselineRows": len(baseline_rows),
                            "baselineStatsRows": len(baseline_stats_rows),
                            "distributionRows": len(class_distribution_rows),
                        }
                    )
                    progress.finish_class(run_id, class_key, len(rows))

                run_manifest = {
                    "id": run_id,
                    "source": source_name,
                    "dataset": dataset_name,
                    "variant": variant,
                    "referenceInput": str(reference_input),
                    "candidateInput": str(candidate_input),
                    "outputDir": str(run_dir),
                    "classCount": len(class_manifests),
                    "pairwiseRows": len(run_rows),
                    "statsRows": len(run_stats),
                    "baselineRows": len(run_baseline_rows),
                    "baselineStatsRows": len(run_baseline_stats),
                    "withinComparisonRows": len(run_within_comparison_rows),
                    "withinComparisonStatsRows": len(run_within_comparison_stats),
                    "distributionRows": len(run_distribution_rows),
                    "classes": class_manifests,
                    "skippedClasses": skipped_classes,
                }

                _write_json(
                    run_dir / "pairwise.json",
                    {
                        "metadata": run_manifest,
                        "pairs": run_rows,
                        "stats": run_stats,
                    },
                )
                _write_csv(run_dir / "pairwise.csv", run_rows, PAIRWISE_COLUMNS)
                _write_csv(run_dir / "stats.csv", run_stats, STATS_COLUMNS)
                _write_json(
                    run_dir / "baseline.json",
                    {
                        "metadata": {
                            **run_manifest,
                            "mode": "human-summary-baseline",
                        },
                        "baseline": run_baseline_rows,
                        "stats": run_baseline_stats,
                    },
                )
                _write_csv(run_dir / "baseline.csv", run_baseline_rows, BASELINE_COLUMNS)
                _write_csv(
                    run_dir / "baseline_stats.csv",
                    run_baseline_stats,
                    STATS_COLUMNS,
                )
                _write_csv(
                    run_dir / "distribution.csv",
                    run_distribution_rows,
                    DISTRIBUTION_COLUMNS,
                )
                _write_json(run_dir / "manifest.json", run_manifest)
                manifest["runs"].append(run_manifest)
                all_distribution_rows.extend(run_distribution_rows)
                progress.finish_run(run_id, len(run_rows))

    _write_csv(output_root / "distribution.csv", all_distribution_rows, DISTRIBUTION_COLUMNS)
    _write_json(output_root / "manifest.json", manifest)
    return 0


def main(argv=None):
    parser = _build_parser()
    opt = parser.parse_args(argv)

    output_root = Path(opt.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sidecar_paths(
        output_root,
        opt.log_dir,
        stem="run-all-pairwise-reports",
        output_is_dir=True,
    )
    metadata = build_run_metadata(parser, opt, argv, "run-all-pairwise-reports")
    write_run_metadata(paths, metadata)
    return run_logged_experiment(
        paths,
        lambda: _run_reports(opt, output_root, paths, metadata),
    )


if __name__ == "__main__":
    raise SystemExit(main())
