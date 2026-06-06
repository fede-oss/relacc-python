import pytest

from relacc import dtw as Dtw


def test_classic_dtw_uses_a_visible_alignment_example(p):
    first = [p(0), p(1), p(2)]
    second = [p(0), p(2), p(3)]

    result = Dtw.dtw(first, second)

    # The diagonal alignment costs |0-0| + |1-2| + |2-3| = 2.
    assert result == Dtw.DTWResult(cost=2.0, path_length=3)

    # There is an equal-cost path with four steps, so length-independent DTW uses 2 / 4.
    assert Dtw.length_independent_dtw(first, second) == pytest.approx(0.5)


def test_windowed_dtw_shows_the_accuracy_tradeoff(p):
    first = [p(0), p(0), p(10)]
    second = [p(0), p(10), p(10)]

    assert Dtw.dtw(first, second).cost == pytest.approx(0)
    # A zero-width window only permits diagonal matches: 0->0, 0->10, 10->10.
    assert Dtw.dtw(first, second, window=0).cost == pytest.approx(10)


def test_derivative_dtw_compares_slopes_not_absolute_position(p):
    flat_low = [p(0, 0), p(1, 0), p(2, 0)]
    flat_high = [p(0, 5), p(1, 5), p(2, 5)]

    assert Dtw.dtw(flat_low, flat_high).cost == pytest.approx(15)
    # The two lines have the same local direction, so derivative DTW sees no difference.
    assert Dtw.ddtw(flat_low, flat_high) == Dtw.DTWResult(cost=0.0, path_length=3)


def test_weighted_dtw_has_a_concrete_penalty_value(p):
    first = [p(0), p(1), p(2)]
    second = [p(0), p(2), p(3)]

    weighted = Dtw.weighted_dtw(first, second, penalty_g=0.25)

    # The best path is still three steps; the local distances are scaled by logistic weights.
    assert weighted.path_length == 3
    assert weighted.cost == pytest.approx(0.8146668000918604)


def test_dtw_validation_errors_are_plain(p):
    with pytest.raises(ValueError, match="points_a must contain at least one point"):
        Dtw.dtw([], [p(0)])

    with pytest.raises(ValueError, match="window must be >= 0"):
        Dtw.dtw([p(0)], [p(0)], window=-1)

    with pytest.raises(ValueError, match="penalty_g must be >= 0"):
        Dtw.weighted_dtw([p(0)], [p(0)], penalty_g=-0.1)
