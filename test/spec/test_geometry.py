import math

import pytest

from relacc.geom.measure import Measure
from relacc.geom.point import Point
from relacc.geom.pointset import PointSet
from relacc.geom.rectangle import Rectangle
from relacc.geom.vector import Vector


def test_point_creation_and_arithmetic_match_the_js_examples():
    assert Point() == Point(0, 0, 0, 0)
    assert Point({"X": 1, "Y": 2, "T": 3, "StrokeID": 4}) == Point(1, 2, 3, 4)

    first = Point(1, 2, 100, 0)
    second = Point(3, 4, 200, 1)

    assert first.add(second) == Point(4, 6, 100, 0)
    assert first.subtract(second) == Point(-2, -2, 100, 0)
    assert first.multiplyBy(2) == Point(2, 4, 100, 0)
    assert first.divideBy(2) == Point(0.5, 1, 100, 0)

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        first.divideBy(0)


def test_measurements_use_small_right_triangle_examples(p):
    origin = p(0, 0)
    east = p(3, 0)
    north = p(0, 4)

    assert Measure.sqDistance(origin, east) == 9
    assert Measure.distance(origin, north) == 4
    assert Measure.taxicab(east, north) == 7
    assert Measure.shortAngle(Vector(origin, east), Vector(origin, north)) == pytest.approx(math.pi / 2)
    assert Rectangle(p(0, 0), p(3, 4)).area() == 12


def test_pointset_helpers_show_the_geometry_being_computed(p):
    points = [p(0, 0, 0), p(3, 0, 3), p(3, 4, 7)]

    assert PointSet.centroid(points) == Point(2, 4 / 3, 0, 0)
    assert PointSet.pathLength(points) == 7
    assert PointSet.boundingBox(points).width() == 3
    assert PointSet.boundingBox(points).height() == 4

    # The path is seven units long, so five equally spaced samples land every 1.75 units.
    resampled = PointSet.resample(PointSet.clone(points), 5)
    assert [(pt.X, pt.Y) for pt in resampled] == pytest.approx(
        [(0, 0), (1.75, 0), (3, 0.5), (3, 2.25), (3, 4)]
    )


def test_pointset_counts_stroke_breaks_as_in_the_js_port(p):
    points = [p(0, 0, stroke=0), p(1, 0, stroke=0), p(1, 1, stroke=1), p(2, 1, stroke=1)]

    assert PointSet.countStrokes(points) == 1
    assert [[pt.StrokeID for pt in group] for group in PointSet.eqDistStrokes(points, 2)] == [[0, 0], [1, 1]]
