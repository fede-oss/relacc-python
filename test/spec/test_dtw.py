import math

import pytest

from relacc import dtw as Dtw
from relacc.geom.point import Point


def p(x, y):
    return Point(x, y, 0, 0)


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
