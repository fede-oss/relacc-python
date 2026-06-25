import pytest

from relacc.geom.point import Point
from relacc.geom.pointset import PointSet, _js_round


def _points():
    return [
        Point(0, 0, 0, 0),
        Point(10, 0, 10, 0),
        Point(10, 10, 20, 1),
        Point(20, 10, 30, 1),
    ]


def test_null_safe_helpers():
    assert PointSet.clone(None) == Point()
    assert PointSet.centroid(None) == Point()
    assert PointSet.minPt(None) == Point()
    assert PointSet.maxPt(None) == Point()


def test_invalid_path_ranges():
    points = _points()
    assert PointSet.pathLength(None) == 0
    assert PointSet.pathLength(points, 2, 1) == 0


def test_scale_non_finite():
    same = [Point(1, 1, 0, 0), Point(1, 1, 1, 0)]
    scaled = PointSet.scale(same)
    assert scaled == same
    assert scaled[0] is not same[0]


def test_scale_to_and_translate_helpers():
    points = _points()
    scaled = PointSet.scaleTo(points, 0.5)
    moved = PointSet.translateBy(points, Point(1, 1, 0, 0))
    assert scaled[0].X == 0
    assert scaled[1].X == 5
    assert moved[0] == Point(-1, -1, 0, 0)


def test_zero_length_unif_resampling():
    same = [Point(1, 1, 0, 0), Point(1, 1, 0, 0)]
    rs = PointSet.unifResampling(same, 4)
    assert len(rs) == 4
    assert rs[0] == Point(1, 1, 0, 0)


def test_unif_resampling_cross_strokes():
    rs = PointSet.unifResampling(_points(), 5)
    assert len(rs) == 5


def test_count_strokes_transition_logic():
    multi = [
        Point(0, 0, 0, 0),
        Point(1, 0, 1, 1),
        Point(2, 0, 2, 0),
        Point(3, 0, 3, 1),
        Point(4, 0, 4, 1),
    ]
    assert PointSet.countStrokes(multi) == 4


def test_count_strokes_counts_contiguous_runs():
    assert PointSet.countStrokes([]) == 0
    assert PointSet.countStrokes([Point(0, 0, 0, 4)]) == 1
    assert PointSet.countStrokes(_points()) == 2
    assert PointSet.countStrokes(
        [
            Point(0, 0, 0, 0),
            Point(1, 0, 1, 1),
            Point(2, 0, 2, 0),
        ]
    ) == 3


def test_cum_distance_and_index_lookup():
    points = _points()
    cum = PointSet.cumDistances(points)
    assert len(cum) == len(points)
    assert cum[0] == 0
    assert PointSet.indexOfDistance(cum, 5) == 0
    assert PointSet.indexOfDistance(cum, 999) == -1


def test_ma_resampling():
    rs = PointSet.maResampling(_points(), 3)
    assert len(rs) == 3


def test_zero_length_ma_resampling():
    same = [Point(2, 2, 0, 0), Point(2, 2, 0, 0)]
    rs = PointSet.maResampling(same, 4)
    assert len(rs) == 4


def test_ma_resampling_index_fallback(monkeypatch):
    points = _points()
    monkeypatch.setattr(PointSet, "indexOfDistance", staticmethod(lambda *_: -1))
    rs = PointSet.maResampling(points, 3)
    assert len(rs) == 3


def test_eq_dist_strokes_split():
    strokes = PointSet.eqDistStrokes(_points(), 2)
    assert len(strokes) == 2
    assert len(strokes[0]) > 0
    assert len(strokes[1]) > 0


def test_eq_dist_strokes_empty_and_single():
    assert PointSet.eqDistStrokes([], 2) == []
    one = [Point(0, 0, 0, 0)]
    strokes = PointSet.eqDistStrokes(one, 1)
    assert len(strokes) == 1
    assert strokes[0] == one


def test_js_round_negative_branch():
    assert _js_round(-1.4) == -1


def test_eq_resample_delegates_to_multistroke_policy():
    assert PointSet.eqResample(_points(), 5) == PointSet.resample(_points(), 5)


def test_ensure_resampling_raises_value_error():
    PointSet.ensureResampling([Point()], 1)
    with pytest.raises(ValueError, match="Expected 2 resampled points"):
        PointSet.ensureResampling([Point()], 2)


def test_resample_requires_one_point_per_stroke_and_valid_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        PointSet.resample([], 1)
    with pytest.raises(ValueError, match="integer"):
        PointSet.resample(_points(), 2.5)
    with pytest.raises(ValueError, match="at least the number of strokes"):
        PointSet.resample(_points(), 1)


def test_resample_zero_length_strokes_distributes_round_robin():
    points = [
        Point(1, 1, 0, 7),
        Point(1, 1, 1, 7),
        Point(5, 5, 2, 8),
        Point(5, 5, 3, 8),
    ]

    resampled = PointSet.resample(points, 5)

    assert len(resampled) == 5
    assert [point.StrokeID for point in resampled] == [7, 7, 7, 8, 8]


def test_resample_largest_remainder_ties_follow_stroke_order():
    points = [
        Point(0, 0, 0, 0),
        Point(2, 0, 2, 0),
        Point(10, 0, 10, 1),
        Point(12, 0, 12, 1),
    ]

    resampled = PointSet.resample(points, 5)

    assert [point.StrokeID for point in resampled] == [0, 0, 0, 1, 1]
    assert resampled[1] == Point(1, 0, 1, 0)
