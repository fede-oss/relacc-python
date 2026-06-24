from relacc.geom.point import Point
from relacc.geom.pointset import PointSet

from test.helpers.math import round_to


def _make_set():
    pt1 = Point(1, 2, 100, 0)
    pt2 = Point(3, 4, 200, 0)
    pt3 = Point(5, 6, 300, 0)
    return pt1, pt2, pt3, [pt1, pt2, pt3]


def test_clone_set():
    _, _, _, s = _make_set()
    cloned = PointSet.clone(s)
    assert cloned == s


def test_centroid_set():
    _, _, _, s = _make_set()
    assert PointSet.centroid(s) == Point(3, 4, 0, 0)


def test_min_max_and_bbox():
    _, _, _, s = _make_set()
    assert PointSet.minPt(s) == Point(1, 2, 0, 0)
    assert PointSet.maxPt(s) == Point(5, 6, 0, 0)
    bb = PointSet.boundingBox(s)
    assert bb.topLeft == Point(1, 2, 0, 0)
    assert bb.bottomRight == Point(5, 6, 0, 0)


def test_path_length_ranges():
    _, _, _, s = _make_set()
    assert round_to(PointSet.pathLength(s)) == 5.66
    assert round_to(PointSet.pathLength(s, 0, 1)) == 2.83
    assert round_to(PointSet.pathLength(s, 1, 2)) == 2.83


def test_scale():
    _, _, _, s = _make_set()
    assert PointSet.scale(s) == [
        Point(0, 0, 100, 0),
        Point(0.5, 0.5, 200, 0),
        Point(1, 1, 300, 0),
    ]


def test_translate_by():
    pt1, _, _, s = _make_set()
    assert PointSet.translateBy(s, pt1) == [
        Point(0, 0, 100, 0),
        Point(2, 2, 200, 0),
        Point(4, 4, 300, 0),
    ]


def test_resample_length_2():
    pt1, _, pt3, s = _make_set()
    assert PointSet.resample(s, 2) == [pt1, pt3]


def test_resample_length_3():
    pt1, pt2, pt3, s = _make_set()
    assert PointSet.resample(s, 3) == [pt1, pt2, pt3]


def test_resample_length_gt_3():
    _, _, _, s = _make_set()
    rs = PointSet.resample(s, 4)
    assert len(rs) > 3


def test_resample_preserves_single_stroke_uniform_behavior_and_input():
    _, _, _, points = _make_set()
    original = PointSet.clone(points)

    assert PointSet.resample(points, 3) == original
    assert points == original


def test_resample_multiple_strokes_never_interpolates_pen_up_gap():
    points = [
        Point(0, 0, 0, 1),
        Point(1, 0, 10, 1),
        Point(100, 0, 20, 2),
        Point(101, 0, 30, 2),
    ]

    resampled = PointSet.resample(points, 5)

    assert resampled == [
        Point(0, 0, 0, 1),
        Point(0.5, 0, 5, 1),
        Point(1, 0, 10, 1),
        Point(100, 0, 20, 2),
        Point(101, 0, 30, 2),
    ]
    assert all(not (1 < point.X < 100) for point in resampled)
