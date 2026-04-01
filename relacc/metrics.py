"""Shared metric definitions and evaluators for gesture comparisons."""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

from relacc import relacc as RelAcc
from relacc.gestures.gesture import Gesture
from relacc.gestures.summarygesture import SummaryGesture
from relacc.utils.math import MathUtil


MetricFn = Callable[..., float]

BASE_METRIC_DEFINITIONS: Tuple[Tuple[str, MetricFn], ...] = (
    ("shapeError", RelAcc.shapeError),
    ("shapeVariability", RelAcc.shapeVariability),
    ("lengthError", RelAcc.lengthError),
    ("sizeError", RelAcc.sizeError),
    ("bendingError", RelAcc.bendingError),
    ("bendingVariability", RelAcc.bendingVariability),
    ("timeError", RelAcc.timeError),
    ("timeVariability", RelAcc.timeVariability),
    ("velocityError", RelAcc.velocityError),
    ("velocityVariability", RelAcc.velocityVariability),
    ("strokeError", RelAcc.strokeError),
    ("strokeOrderError", RelAcc.strokeOrderError),
)

DTW_METRIC_DEFINITIONS: Tuple[Tuple[str, MetricFn], ...] = (
    ("dtwDistance", RelAcc.dtwDistance),
    ("ldtwDistance", RelAcc.ldtwDistance),
    ("ddtwDistance", RelAcc.ddtwDistance),
    ("wdtwDistance", RelAcc.wdtwDistance),
    ("wddtwDistance", RelAcc.wddtwDistance),
)

_METRIC_DEFINITIONS: Tuple[Tuple[str, MetricFn], ...] = (
    *BASE_METRIC_DEFINITIONS,
    *DTW_METRIC_DEFINITIONS,
)

BASE_METRIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in BASE_METRIC_DEFINITIONS)
DTW_METRIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in DTW_METRIC_DEFINITIONS)
METRIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in _METRIC_DEFINITIONS)
_METRIC_DEFINITION_MAP: Dict[str, MetricFn] = {name: fn for name, fn in _METRIC_DEFINITIONS}


def get_metric_names(include_dtw: bool = True) -> Tuple[str, ...]:
    if include_dtw:
        return METRIC_NAMES
    return BASE_METRIC_NAMES


def compute_metrics(
    gesture: Gesture,
    summary: SummaryGesture,
    round_precision: int | None = None,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
) -> Dict[str, float]:
    selected_metric_names = metric_names or METRIC_NAMES
    values: Dict[str, float] = {}
    for name in selected_metric_names:
        metric_fn = _METRIC_DEFINITION_MAP.get(name)
        if metric_fn is None:
            raise ValueError("Unknown metric: %s" % name)

        if name in DTW_METRIC_NAMES:
            value = metric_fn(gesture, summary, window=dtw_window)
        else:
            value = metric_fn(gesture, summary)
        if round_precision is not None:
            value = MathUtil.roundTo(value, round_precision)
        values[name] = value
    return values
