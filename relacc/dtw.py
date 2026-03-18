"""Dynamic Time Warping helpers for gesture trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from relacc.geom.measure import Measure


DEFAULT_WARPING_PENALTY = 0.25


@dataclass(frozen=True)
class DTWResult:
    """Final DTW cost together with the chosen warping-path length."""

    cost: float
    path_length: int


def _vector_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _derivative_at(points, index):
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


def _derivative_sequence(points):
    return [_derivative_at(points, index) for index in range(len(points))]


def _logistic_weight(index_gap, series_length, penalty_g):
    exponent = penalty_g * (index_gap - (series_length / 2.0))
    if exponent >= 0:
        scaled = math.exp(-exponent)
        return 1.0 / (1.0 + scaled)

    scaled = math.exp(exponent)
    return scaled / (1.0 + scaled)


def _best_predecessor(costs, path_lengths, i, j, prefer_longer_path=False):
    candidates = []
    if i > 0:
        candidates.append((costs[i - 1][j], path_lengths[i - 1][j], 1))
    if j > 0:
        candidates.append((costs[i][j - 1], path_lengths[i][j - 1], 1))
    if i > 0 and j > 0:
        candidates.append((costs[i - 1][j - 1], path_lengths[i - 1][j - 1], 0))
    if prefer_longer_path:
        return min(candidates, key=lambda item: (item[0], -item[1], item[2]))
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))


def _run_dtw(rows, cols, local_cost: Callable[[int, int], float], prefer_longer_path=False) -> DTWResult:
    costs = [[float("inf")] * cols for _ in range(rows)]
    path_lengths = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            step_cost = local_cost(i, j)
            if i == 0 and j == 0:
                costs[i][j] = step_cost
                path_lengths[i][j] = 1
                continue

            prev_cost, prev_length, _ = _best_predecessor(
                costs,
                path_lengths,
                i,
                j,
                prefer_longer_path=prefer_longer_path,
            )
            costs[i][j] = prev_cost + step_cost
            path_lengths[i][j] = prev_length + 1

    return DTWResult(cost=costs[-1][-1], path_length=path_lengths[-1][-1])


def dtw(points_a, points_b) -> DTWResult:
    return _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: Measure.distance(points_a[i], points_b[j]),
    )


def length_independent_dtw(points_a, points_b) -> float:
    result = _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: Measure.distance(points_a[i], points_b[j]),
        prefer_longer_path=True,
    )
    return result.cost / result.path_length


def ddtw(points_a, points_b) -> DTWResult:
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    return _run_dtw(
        len(derivatives_a),
        len(derivatives_b),
        lambda i, j: _vector_distance(derivatives_a[i], derivatives_b[j]),
    )


def weighted_dtw(
    points_a,
    points_b,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
) -> DTWResult:
    series_length = max(len(points_a), len(points_b))
    return _run_dtw(
        len(points_a),
        len(points_b),
        lambda i, j: _logistic_weight(abs(i - j), series_length, penalty_g)
        * Measure.distance(points_a[i], points_b[j]),
    )


def weighted_derivative_dtw(
    points_a,
    points_b,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
) -> DTWResult:
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    series_length = max(len(derivatives_a), len(derivatives_b))
    return _run_dtw(
        len(derivatives_a),
        len(derivatives_b),
        lambda i, j: _logistic_weight(abs(i - j), series_length, penalty_g)
        * _vector_distance(derivatives_a[i], derivatives_b[j]),
    )
