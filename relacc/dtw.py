"""Dynamic Time Warping helpers for gesture trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeAlias

from relacc.geom.measure import Measure


# Keep a small custom DTW implementation so the project can work directly with
# its point objects and avoid adding heavy numeric dependencies. aeon (dtw,
# ddtw, wdtw, wddtw) is a viable future replacement if we decide the extra
# array-conversion layer and different library defaults are worth the trade-off.

# Project heuristic; tune empirically if a different gesture dataset needs a
# stricter or looser phase penalty.
DEFAULT_WARPING_PENALTY = 0.25
DEFAULT_WINDOW_FRACTION = 0.1
DEFAULT_EXACT_RATE_THRESHOLD = 128
MIN_RECOMMENDED_WINDOW = 8
MAX_RECOMMENDED_WINDOW = 64


class SupportsXY(Protocol):
    """Minimal protocol for point-like objects used by the DTW helpers."""

    X: float
    Y: float


PointSequence: TypeAlias = Sequence[SupportsXY]
Vector2: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class DTWResult:
    """Final DTW cost together with the chosen warping-path length."""

    cost: float
    path_length: int


def _vector_distance(a: Vector2, b: Vector2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _validate_points(points: PointSequence, label: str) -> None:
    if len(points) == 0:
        raise ValueError(f"{label} must contain at least one point.")


def _validate_penalty_g(penalty_g: float) -> None:
    if penalty_g < 0:
        raise ValueError("penalty_g must be >= 0.")


def _validate_window(window: int | None) -> int | None:
    if window is None:
        return None
    if window < 0:
        raise ValueError("window must be >= 0.")
    return window


def recommended_window(series_length: int) -> int:
    if series_length < 1:
        raise ValueError("series_length must be >= 1.")
    scaled_window = int(series_length * DEFAULT_WINDOW_FRACTION)
    return max(MIN_RECOMMENDED_WINDOW, min(MAX_RECOMMENDED_WINDOW, scaled_window))


def _derivative_at(points: PointSequence, index: int) -> Vector2:
    if len(points) == 1:
        return (0.0, 0.0)

    if index == 0:
        return (
            points[1].X - points[0].X,
            points[1].Y - points[0].Y,
        )

    if index == len(points) - 1:
        return (
            points[-1].X - points[-2].X,
            points[-1].Y - points[-2].Y,
        )

    dx = (
        (points[index].X - points[index - 1].X)
        + ((points[index + 1].X - points[index - 1].X) / 2.0)
    ) / 2.0
    dy = (
        (points[index].Y - points[index - 1].Y)
        + ((points[index + 1].Y - points[index - 1].Y) / 2.0)
    ) / 2.0
    return (dx, dy)


def _derivative_sequence(points: PointSequence) -> list[Vector2]:
    return [_derivative_at(points, index) for index in range(len(points))]


def _logistic_weight(index_gap: int, series_length: int, penalty_g: float) -> float:
    """Return the WDTW logistic weight for a given index gap.

    Larger ``penalty_g`` values make the weight rise more steeply as the
    alignment moves away from the diagonal, so off-phase matches are penalized
    more aggressively.
    """

    # This is the paper's modified logistic weight function with an implicit
    # wmax=1. Jeong et al. report that changing wmax did not affect
    # classification performance because it only rescales the positive weights.
    exponent = penalty_g * (index_gap - (series_length / 2.0))
    if exponent >= 0:
        scaled = math.exp(-exponent)
        return 1.0 / (1.0 + scaled)

    scaled = math.exp(exponent)
    return scaled / (1.0 + scaled)


def _best_predecessor(
    candidates: list[tuple[float, int, int]],
    prefer_longer_path: bool = False,
) -> tuple[float, int, int]:
    if prefer_longer_path:
        return min(candidates, key=lambda item: (item[0], -item[1], item[2]))

    return min(candidates, key=lambda item: (item[0], item[1], item[2]))


def _run_dtw(
    rows,
    cols,
    local_cost: Callable[[int, int], float],
    prefer_longer_path: bool = False,
    window: int | None = None,
) -> DTWResult:
    window = _validate_window(window)
    if window is None:
        window = max(rows, cols)
    else:
        window = max(window, abs(rows - cols))

    prev_costs = [float("inf")] * cols
    prev_lengths = [0] * cols
    for i in range(rows):
        current_costs = [float("inf")] * cols
        current_lengths = [0] * cols
        j_start = max(0, i - window)
        j_end = min(cols - 1, i + window)
        for j in range(j_start, j_end + 1):
            step_cost = local_cost(i, j)
            if i == 0 and j == 0:
                current_costs[j] = step_cost
                current_lengths[j] = 1
                continue

            candidates = []
            if i > 0 and not math.isinf(prev_costs[j]):
                candidates.append((prev_costs[j], prev_lengths[j], 1))
            if j > 0 and not math.isinf(current_costs[j - 1]):
                candidates.append((current_costs[j - 1], current_lengths[j - 1], 1))
            if i > 0 and j > 0 and not math.isinf(prev_costs[j - 1]):
                candidates.append((prev_costs[j - 1], prev_lengths[j - 1], 0))
            if not candidates:
                continue

            prev_cost, prev_length, _ = _best_predecessor(
                candidates,
                prefer_longer_path=prefer_longer_path,
            )
            current_costs[j] = prev_cost + step_cost
            current_lengths[j] = prev_length + 1
        prev_costs = current_costs
        prev_lengths = current_lengths

    if math.isinf(prev_costs[-1]):
        raise ValueError("window is too small to align the sequences.")

    return DTWResult(cost=prev_costs[-1], path_length=prev_lengths[-1])


def dtw(
    points_a: PointSequence,
    points_b: PointSequence,
    window: int | None = None,
) -> DTWResult:
    """Return the classic DTW total alignment cost and chosen path length."""

    _validate_points(points_a, "points_a")
    _validate_points(points_b, "points_b")
    return _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: Measure.distance(points_a[i], points_b[j]),
        window=window,
    )


def length_independent_dtw(
    points_a: PointSequence,
    points_b: PointSequence,
    window: int | None = None,
) -> float:
    """Return DTW normalized by the selected warping-path length.

    When multiple warping paths have the same minimum DTW cost, prefer the
    longest such path so the normalization does not depend on an arbitrary
    shortest-path tie break. This still follows the paper's LDTW definition,
    ``DTW / T``: the ambiguity is only in choosing ``T`` when several optimal
    paths share the same DTW cost.
    """

    _validate_points(points_a, "points_a")
    _validate_points(points_b, "points_b")
    result = _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: Measure.distance(points_a[i], points_b[j]),
        prefer_longer_path=True,
        window=window,
    )
    return result.cost / result.path_length


def ddtw(
    points_a: PointSequence,
    points_b: PointSequence,
    window: int | None = None,
) -> DTWResult:
    """Return derivative DTW, comparing local trajectory trend vectors."""

    _validate_points(points_a, "points_a")
    _validate_points(points_b, "points_b")
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    return _run_dtw(
        len(derivatives_a),
        len(derivatives_b),
        lambda i, j: _vector_distance(derivatives_a[i], derivatives_b[j]),
        window=window,
    )


def weighted_dtw(
    points_a: PointSequence,
    points_b: PointSequence,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
    window: int | None = None,
) -> DTWResult:
    """Return weighted DTW using a logistic phase-offset penalty.

    The returned cost uses the same local Euclidean distance as ``dtw()``, but
    scales it by a logistic weight in ``(0, 1)`` that increases with ``|i-j|``.
    Because of that weighting, weighted DTW scores are not directly comparable
    to classic DTW scores.
    """

    _validate_points(points_a, "points_a")
    _validate_points(points_b, "points_b")
    _validate_penalty_g(penalty_g)
    series_length = max(len(points_a), len(points_b))
    return _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: _logistic_weight(abs(i - j), series_length, penalty_g)
        * Measure.distance(points_a[i], points_b[j]),
        window=window,
    )


def weighted_derivative_dtw(
    points_a: PointSequence,
    points_b: PointSequence,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
    window: int | None = None,
) -> DTWResult:
    """Return weighted derivative DTW with the same logistic penalty.

    ``penalty_g`` controls how strongly alignments far from the diagonal are
    discouraged. The default is a project heuristic and should be tuned if a
    different gesture dataset needs a stricter or looser phase penalty.
    """

    _validate_points(points_a, "points_a")
    _validate_points(points_b, "points_b")
    _validate_penalty_g(penalty_g)
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    series_length = max(len(derivatives_a), len(derivatives_b))
    return _run_dtw(
        len(derivatives_a),
        len(derivatives_b),
        lambda i, j: _logistic_weight(abs(i - j), series_length, penalty_g)
        * _vector_distance(derivatives_a[i], derivatives_b[j]),
        window=window,
    )
