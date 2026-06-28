from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import METRIC_NAMES
from relacc.utils.math import MathUtil

from ._common import compute_pair_metrics_from_points, effective_dtw_window


FORWARD_DIRECTION = "forward"
BACKWARD_DIRECTION = "backward"
REFERENCE_TO_CANDIDATE_DIRECTION = "reference-to-candidate"

WITHIN_REFERENCE_SAMPLE_KIND = "withinReference"
WITHIN_COMPARISON_SAMPLE_KIND = "withinComparison"
BETWEEN_GROUPS_SAMPLE_KIND = "betweenGroups"

WITHIN_REFERENCE_RECORD_SET = "within-reference"
WITHIN_COMPARISON_RECORD_SET = "within-comparison"
BETWEEN_GROUPS_RECORD_SET = "between-groups"


@dataclass(frozen=True)
class PairEndpoint:
    key: str
    path: str
    points: object


@dataclass(frozen=True)
class PairMetricOptions:
    label: str
    effective_rate: int
    alignment_type: int = PtAlignType.CHRONOLOGICAL
    summary_shape: str | None = None
    popular_shape: bool = False
    metric_names: Sequence[str] | None = None
    dtw_window: int | None = None
    exact_dtw: bool = False

    @property
    def selected_metric_names(self) -> tuple[str, ...]:
        return tuple(self.metric_names or METRIC_NAMES)

    @property
    def selected_dtw_window(self) -> int | None:
        return effective_dtw_window(
            self.effective_rate,
            self.dtw_window,
            self.exact_dtw,
        )


@dataclass(frozen=True)
class PairMetricEvidence:
    left: PairEndpoint
    right: PairEndpoint
    values: Dict[str, float]
    direction: str | None = None
    forward_values: Dict[str, float] | None = None
    backward_values: Dict[str, float] | None = None


def endpoint_for(entry) -> PairEndpoint:
    return PairEndpoint(key=entry.key, path=entry.path, points=entry.points)


def joined_pair_key(left: PairEndpoint, right: PairEndpoint) -> str:
    return "%s::%s" % (
        Path(left.key).with_suffix("").as_posix(),
        Path(right.key).with_suffix("").as_posix(),
    )


def joined_pair_path(left: PairEndpoint, right: PairEndpoint) -> str:
    return "%s::%s" % (left.path, right.path)


def rounded_metric_values(
    values: Mapping[str, float],
    round_precision: int | None,
) -> Dict[str, float]:
    if round_precision is None:
        return dict(values)
    return {
        metric_name: MathUtil.roundTo(value, round_precision)
        for metric_name, value in values.items()
    }


def compute_directional_pair_evidence(
    left: PairEndpoint,
    right: PairEndpoint,
    options: PairMetricOptions,
    direction: str | None = REFERENCE_TO_CANDIDATE_DIRECTION,
) -> PairMetricEvidence:
    values = compute_pair_metrics_from_points(
        left.points,
        right.points,
        options.label,
        options.effective_rate,
        alignment_type=options.alignment_type,
        summary_shape=options.summary_shape,
        popular_shape=options.popular_shape,
        round_precision=None,
        metric_names=options.selected_metric_names,
        dtw_window=options.selected_dtw_window,
        exact_dtw=options.exact_dtw,
    )
    return PairMetricEvidence(
        left=left,
        right=right,
        values=values,
        direction=direction,
    )


def directed_pair_evidences(
    left: PairEndpoint,
    right: PairEndpoint,
    options: PairMetricOptions,
) -> tuple[PairMetricEvidence, PairMetricEvidence]:
    return (
        compute_directional_pair_evidence(
            left,
            right,
            options,
            direction=FORWARD_DIRECTION,
        ),
        compute_directional_pair_evidence(
            right,
            left,
            options,
            direction=BACKWARD_DIRECTION,
        ),
    )


def compute_bidirectional_pair_evidence(
    left: PairEndpoint,
    right: PairEndpoint,
    options: PairMetricOptions,
) -> PairMetricEvidence:
    forward, backward = directed_pair_evidences(left, right, options)
    values = {
        metric_name: (
            forward.values[metric_name] + backward.values[metric_name]
        ) / 2.0
        for metric_name in options.selected_metric_names
    }
    return PairMetricEvidence(
        left=left,
        right=right,
        values=values,
        forward_values=forward.values,
        backward_values=backward.values,
    )


def metric_output_rows(
    base_fields: Mapping[str, object],
    evidence: PairMetricEvidence,
    include_direction_values: bool = False,
) -> list[dict[str, object]]:
    rows = metric_value_rows(base_fields, evidence.values)
    if include_direction_values:
        for row in rows:
            metric_name = str(row["metric"])
            row["forwardValue"] = evidence.forward_values[metric_name]
            row["backwardValue"] = evidence.backward_values[metric_name]
    return rows


def metric_value_rows(
    base_fields: Mapping[str, object],
    values: Mapping[str, float],
) -> list[dict[str, object]]:
    rows = []
    for metric_name, value in values.items():
        rows.append({**base_fields, "metric": metric_name, "value": value})
    return rows


def distribution_within_reference_rows(
    evidence: PairMetricEvidence,
    base_fields: Mapping[str, object],
) -> list[dict[str, object]]:
    return metric_output_rows(
        {
            **base_fields,
            "sampleKind": WITHIN_REFERENCE_SAMPLE_KIND,
            "leftReferenceKey": evidence.left.key,
            "leftReferenceFile": evidence.left.path,
            "rightReferenceKey": evidence.right.key,
            "rightReferenceFile": evidence.right.path,
        },
        evidence,
        include_direction_values=True,
    )


def distribution_within_comparison_rows(
    evidence: PairMetricEvidence,
    base_fields: Mapping[str, object],
) -> list[dict[str, object]]:
    return metric_output_rows(
        {
            **base_fields,
            "sampleKind": WITHIN_COMPARISON_SAMPLE_KIND,
            "leftComparisonKey": evidence.left.key,
            "leftComparisonFile": evidence.left.path,
            "rightComparisonKey": evidence.right.key,
            "rightComparisonFile": evidence.right.path,
        },
        evidence,
        include_direction_values=True,
    )


def distribution_between_groups_rows(
    evidence: PairMetricEvidence,
    base_fields: Mapping[str, object],
) -> list[dict[str, object]]:
    return metric_output_rows(
        {
            **base_fields,
            "sampleKind": BETWEEN_GROUPS_SAMPLE_KIND,
            "referenceKey": evidence.left.key,
            "referenceFile": evidence.left.path,
            "comparisonKey": evidence.right.key,
            "comparisonFile": evidence.right.path,
        },
        evidence,
    )
