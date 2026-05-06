import math

import pytest

from relacc import dtw as Dtw
from relacc.geom.measure import Measure
from relacc.geom.point import Point


def p(x, y):
    return Point(x, y, 0, 0)


def _reference_dtw_result(rows, cols, local_cost, prefer_longer_path=False):
    costs = [[float("inf")] * cols for _ in range(rows)]
    path_lengths = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            step_cost = local_cost(i, j)
            if i == 0 and j == 0:
                costs[i][j] = step_cost
                path_lengths[i][j] = 1
                continue

            candidates = []
            if i > 0:
                candidates.append((costs[i - 1][j], path_lengths[i - 1][j], 1))
            if j > 0:
                candidates.append((costs[i][j - 1], path_lengths[i][j - 1], 1))
            if i > 0 and j > 0:
                candidates.append((costs[i - 1][j - 1], path_lengths[i - 1][j - 1], 0))

            if prefer_longer_path:
                prev_cost, prev_length, _ = min(candidates, key=lambda item: (item[0], -item[1], item[2]))
            else:
                prev_cost, prev_length, _ = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            costs[i][j] = prev_cost + step_cost
            path_lengths[i][j] = prev_length + 1

    return Dtw.DTWResult(cost=costs[-1][-1], path_length=path_lengths[-1][-1])


def test_standard_and_length_independent_dtw_for_simple_sequences():
    first = [p(0, 0), p(1, 0), p(2, 0)]
    second = [p(0, 0), p(2, 0), p(3, 0)]

    result = Dtw.dtw(first, second)

    assert result.cost == pytest.approx(2.0)
    assert result.path_length == 3
    assert Dtw.length_independent_dtw(first, second) == pytest.approx(0.5)


def test_length_independent_dtw_prefers_longer_equal_cost_path():
    first = [p(0, 0), p(0, 0)]
    second = [p(0, 0), p(1, 0)]

    assert Dtw.dtw(first, second) == Dtw.DTWResult(cost=1.0, path_length=2)
    assert Dtw.length_independent_dtw(first, second) == pytest.approx(1.0 / 3.0)


def test_exact_dtw_matches_independent_full_matrix_reference():
    first = [p(0, 0), p(1, 0), p(3, 0), p(4, 0)]
    second = [p(0, 0), p(2, 0), p(2.5, 0), p(4, 0)]

    reference = _reference_dtw_result(
        len(first),
        len(second),
        lambda i, j: Measure.distance(first[i], second[j]),
    )

    result = Dtw.dtw(first, second)
    assert result.cost == pytest.approx(reference.cost)
    assert result.path_length == reference.path_length
    assert Dtw.length_independent_dtw(first, second) == pytest.approx(
        reference.cost / reference.path_length
    )


def test_exact_weighted_and_derivative_variants_match_reference_costs():
    first = [p(0, 0), p(1, 1), p(3, 1), p(4, 0)]
    second = [p(0, 0), p(2, 1), p(2.5, 1), p(4, 0)]

    derivatives_a = Dtw._derivative_sequence(first)
    derivatives_b = Dtw._derivative_sequence(second)
    penalty_g = 0.25
    series_length = max(len(first), len(second))

    weighted_reference = _reference_dtw_result(
        len(first),
        len(second),
        lambda i, j: Dtw._logistic_weight(abs(i - j), series_length, penalty_g)
        * Measure.distance(first[i], second[j]),
    )
    derivative_reference = _reference_dtw_result(
        len(derivatives_a),
        len(derivatives_b),
        lambda i, j: math.hypot(
            derivatives_a[i][0] - derivatives_b[j][0],
            derivatives_a[i][1] - derivatives_b[j][1],
        ),
    )

    weighted_result = Dtw.weighted_dtw(first, second, penalty_g=penalty_g)
    derivative_result = Dtw.ddtw(first, second)

    assert weighted_result.cost == pytest.approx(weighted_reference.cost)
    assert weighted_result.path_length == weighted_reference.path_length
    assert derivative_result.cost == pytest.approx(derivative_reference.cost)
    assert derivative_result.path_length == derivative_reference.path_length


def test_public_dtw_variants_reject_empty_sequences():
    points = [p(0, 0)]

    with pytest.raises(ValueError, match="points_a must contain at least one point\\."):
        Dtw.dtw([], points)

    with pytest.raises(ValueError, match="points_b must contain at least one point\\."):
        Dtw.length_independent_dtw(points, [])

    with pytest.raises(ValueError, match="points_a must contain at least one point\\."):
        Dtw.ddtw([], points)

    with pytest.raises(ValueError, match="points_b must contain at least one point\\."):
        Dtw.weighted_dtw(points, [])

    with pytest.raises(ValueError, match="points_a must contain at least one point\\."):
        Dtw.weighted_derivative_dtw([], points)


def test_derivative_and_weighted_variants_handle_edge_cases():
    single_a = [p(1, 1)]
    single_b = [p(4, 5)]

    assert Dtw.ddtw(single_a, single_b).cost == 0
    assert Dtw.ddtw(single_a, single_b).path_length == 1
    assert Dtw.weighted_derivative_dtw(single_a, single_b).cost == 0

    first = [p(0, 0), p(1, 1), p(2, 2)]
    second = [p(0, 0), p(2, 2), p(4, 4)]

    assert Dtw.ddtw(first, first).cost == 0
    assert Dtw.weighted_dtw(first, second).cost < Dtw.dtw(first, second).cost
    assert Dtw.weighted_derivative_dtw(first, second).cost < Dtw.ddtw(first, second).cost


def test_weighted_variants_respond_to_penalty_strength():
    first = [p(0, 0), p(1, 1), p(2, 2)]
    second = [p(0, 0), p(2, 2), p(4, 4)]

    weak_penalty = Dtw.weighted_dtw(first, second, penalty_g=0.05).cost
    strong_penalty = Dtw.weighted_dtw(first, second, penalty_g=1.0).cost
    weak_derivative = Dtw.weighted_derivative_dtw(first, second, penalty_g=0.05).cost
    strong_derivative = Dtw.weighted_derivative_dtw(first, second, penalty_g=1.0).cost

    assert weak_penalty != pytest.approx(strong_penalty)
    assert weak_derivative != pytest.approx(strong_derivative)


def test_weighted_variants_reject_negative_penalty():
    first = [p(0, 0), p(1, 1)]
    second = [p(0, 0), p(1, 1)]

    with pytest.raises(ValueError, match="penalty_g must be >= 0\\."):
        Dtw.weighted_dtw(first, second, penalty_g=-0.1)

    with pytest.raises(ValueError, match="penalty_g must be >= 0\\."):
        Dtw.weighted_derivative_dtw(first, second, penalty_g=-0.1)


def test_windowed_dtw_can_trade_accuracy_for_speed():
    first = [p(0, 0), p(0, 0), p(10, 0)]
    second = [p(0, 0), p(10, 0), p(10, 0)]

    assert Dtw.dtw(first, second).cost == pytest.approx(0.0)
    assert Dtw.dtw(first, second, window=0).cost == pytest.approx(10.0)


def test_recommended_window_rejects_invalid_series_length():
    with pytest.raises(ValueError, match="series_length must be >= 1\\."):
        Dtw.recommended_window(0)


def test_dtw_variants_reject_negative_window():
    first = [p(0, 0), p(1, 0)]
    second = [p(0, 0), p(1, 0)]

    with pytest.raises(ValueError, match="window must be >= 0\\."):
        Dtw.dtw(first, second, window=-1)

    with pytest.raises(ValueError, match="window must be >= 0\\."):
        Dtw.length_independent_dtw(first, second, window=-1)

    with pytest.raises(ValueError, match="window must be >= 0\\."):
        Dtw.ddtw(first, second, window=-1)

    with pytest.raises(ValueError, match="window must be >= 0\\."):
        Dtw.weighted_dtw(first, second, window=-1)

    with pytest.raises(ValueError, match="window must be >= 0\\."):
        Dtw.weighted_derivative_dtw(first, second, window=-1)


def test_derivative_estimates_cover_start_middle_and_end():
    points = [p(0, 0), p(2, 2), p(4, 0)]

    assert Dtw._derivative_at(points, 0) == pytest.approx((2.0, 2.0))
    assert Dtw._derivative_at(points, 1) == pytest.approx((2.0, 1.0))
    assert Dtw._derivative_at(points, 2) == pytest.approx((2.0, -2.0))
    assert Dtw._logistic_weight(0, 3, Dtw.DEFAULT_WARPING_PENALTY) < Dtw._logistic_weight(
        2, 3, Dtw.DEFAULT_WARPING_PENALTY
    )


def test_logistic_weight_stays_finite_for_long_series():
    near_diagonal = Dtw._logistic_weight(0, 40320, Dtw.DEFAULT_WARPING_PENALTY)
    far_offset = Dtw._logistic_weight(40319, 40320, Dtw.DEFAULT_WARPING_PENALTY)

    assert math.isfinite(near_diagonal)
    assert math.isfinite(far_offset)
    assert near_diagonal < far_offset
