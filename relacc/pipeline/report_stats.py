from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Dict, List, Sequence

from scipy import stats as scipy_stats

from relacc.distribution_metrics import (
    DISTRIBUTION_METRIC_NAMES,
    compute_distribution_metrics,
)
from relacc.metrics import METRIC_NAMES
from relacc.pipeline.report_schema import (
    DISTRIBUTION_OUTPUT_VALUE_COLUMNS,
    statistical_contract_csv_fields,
    statistical_contract_fields,
)
from relacc.utils.math import MathUtil


DEFAULT_VARIANT_LABEL = "root"


def _variant_label(variant: str | None) -> str:
    return variant or DEFAULT_VARIANT_LABEL


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
    summary_shape: str | None,
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
            "variant": _variant_label(variant),
            "classKey": class_key,
            "summary": summary_shape,
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


def _aggregate_summary_stats(
    rows: Sequence[Dict[str, object]],
    record_set: str,
    scope: str,
    source_name: str | None,
    dataset_name: str | None,
    variant: str | None,
    summary_shape: str | None,
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

        stats_rows.append(
            {
                "recordSet": record_set,
                "scope": scope,
                "source": source_name,
                "dataset": dataset_name,
                "variant": _variant_label(variant),
                "summary": summary_shape,
                "metric": metric_name,
                "n": len(values),
                "finiteN": finite_n,
                "mean": rounded(mean),
                "mdn": rounded(mdn),
                "sd": rounded(sd),
                "min": rounded(minimum),
                "max": rounded(maximum),
            }
        )
    return stats_rows


def _aggregate_rows_for_record_set(
    rows: Sequence[Dict[str, object]],
    record_set: str,
    round_precision: int | None,
) -> List[Dict[str, object]]:
    aggregate_rows: List[Dict[str, object]] = []
    grouping_specs = [
        ("overall", ()),
        ("source", ("source",)),
        ("dataset", ("dataset",)),
        ("source-dataset", ("source", "dataset")),
        ("run", ("source", "dataset", "variant")),
    ]

    for scope, keys in grouping_specs:
        grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
        if not keys:
            grouped[()] = list(rows)
        else:
            for row in rows:
                grouped[tuple(row.get(key) for key in keys)].append(row)

        for group_key in sorted(
            grouped.keys(),
            key=lambda value: tuple("" if item is None else str(item) for item in value),
        ):
            values = dict(zip(keys, group_key))
            summaries = sorted(
                {
                    str(row.get("summary"))
                    for row in grouped[group_key]
                    if row.get("summary") not in (None, "")
                }
            )
            summary_shape = summaries[0] if len(summaries) == 1 else "mixed"
            aggregate_rows.extend(
                _aggregate_summary_stats(
                    grouped[group_key],
                    record_set,
                    scope,
                    values.get("source"),
                    values.get("dataset"),
                    values.get("variant"),
                    summary_shape,
                    round_precision,
                )
            )
    return aggregate_rows


def build_combined_aggregate_summaries(
    pairwise_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    within_reference_rows: Sequence[Dict[str, object]],
    within_comparison_rows: Sequence[Dict[str, object]],
    between_group_rows: Sequence[Dict[str, object]],
    round_precision: int | None,
) -> List[Dict[str, object]]:
    return [
        *_aggregate_rows_for_record_set(
            pairwise_rows,
            "comparison-to-reference-summary",
            round_precision,
        ),
        *_aggregate_rows_for_record_set(
            baseline_rows,
            "human-baseline",
            round_precision,
        ),
        *_aggregate_rows_for_record_set(
            within_reference_rows,
            "within-reference",
            round_precision,
        ),
        *_aggregate_rows_for_record_set(
            within_comparison_rows,
            "within-comparison",
            round_precision,
        ),
        *_aggregate_rows_for_record_set(
            between_group_rows,
            "between-groups",
            round_precision,
        ),
    ]


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

        row = {
            **statistical_contract_csv_fields(),
            "runId": metadata_row.get("runId"),
            "source": metadata_row.get("source"),
            "dataset": metadata_row.get("dataset"),
            "variant": metadata_row.get("variant"),
            "classKey": metadata_row.get("classKey"),
            "summary": metadata_row.get("summary"),
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
            "normalizedWassersteinDistance": rounded(
                _ratio(wasserstein, within_reference_sd)
            ),
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
        for distribution_metric_name in DISTRIBUTION_METRIC_NAMES:
            row[distribution_metric_name] = distribution_metrics.get(
                distribution_metric_name
            )
        distribution_rows.append(row)
    return distribution_rows


def _raw_distribution_outputs_from_rows(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    raw_rows = []
    for row in rows:
        base = {
            "schemaVersion": 1,
            "recordType": "rawDistributionOutput",
            **statistical_contract_fields(),
            "runId": row.get("runId"),
            "source": row.get("source"),
            "dataset": row.get("dataset"),
            "variant": row.get("variant"),
            "classKey": row.get("classKey"),
            "summary": row.get("summary"),
            "gestureMetric": row.get("metric"),
            "withinReferenceN": row.get("withinReferenceN"),
            "withinReferenceFiniteN": row.get("withinReferenceFiniteN"),
            "withinComparisonN": row.get("withinComparisonN"),
            "withinComparisonFiniteN": row.get("withinComparisonFiniteN"),
            "betweenGroupsN": row.get("betweenGroupsN"),
            "betweenGroupsFiniteN": row.get("betweenGroupsFiniteN"),
        }
        for distribution_metric in DISTRIBUTION_OUTPUT_VALUE_COLUMNS:
            raw_rows.append(
                {
                    **base,
                    "distributionMetric": distribution_metric,
                    "value": row.get(distribution_metric),
                }
            )
    return raw_rows
