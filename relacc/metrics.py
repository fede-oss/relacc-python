"""Shared metric definitions and evaluators for gesture comparisons."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from relacc import relacc as RelAcc
from relacc.gestures.gesture import Gesture
from relacc.gestures.summarygesture import SummaryGesture
from relacc.utils.math import MathUtil


MetricFn = Callable[[Gesture, SummaryGesture], float]

_METRIC_DEFINITIONS: Tuple[Tuple[str, MetricFn], ...] = (
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

METRIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in _METRIC_DEFINITIONS)


def compute_metrics(
    gesture: Gesture,
    summary: SummaryGesture,
    round_precision: int | None = None,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, metric_fn in _METRIC_DEFINITIONS:
        value = metric_fn(gesture, summary)
        if round_precision is not None:
            value = MathUtil.roundTo(value, round_precision)
        values[name] = value
    return values
