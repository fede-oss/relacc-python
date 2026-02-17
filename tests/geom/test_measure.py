import pytest

from relacc.geom.measure import Measure
from relacc.geom.point import Point
from relacc.geom.vector import Vector

from tests.helpers.math import round_to


pt1 = Point(1, 2, 100, 0)
pt2 = Point(3, 4, 200, 0)
pt3 = Point(5, 6, 300, 0)
v1 = Vector(pt1, pt2)
v2 = Vector(pt2, pt3)
v3 = Vector(pt1, pt3)


def test_sq_distance():
    assert Measure.sqDistance(pt1, pt2) == pytest.approx(8)
    assert Measure.sqDistance(pt2, pt3) == pytest.approx(8)
    assert Measure.sqDistance(pt1, pt3) == pytest.approx(32)


def test_distance():
    assert round_to(Measure.distance(pt1, pt2)) == 2.83
    assert round_to(Measure.distance(pt2, pt3)) == 2.83
    assert round_to(Measure.distance(pt1, pt3)) == 5.66


def test_short_angle():
    assert Measure.shortAngle(v1, v2) == pytest.approx(0, abs=1e-6)
    assert Measure.shortAngle(v2, v3) == pytest.approx(0, abs=1e-6)
    assert Measure.shortAngle(v1, v3) == pytest.approx(0, abs=1e-6)


def test_angle():
    assert Measure.angle(v1, v2) == pytest.approx(0, abs=1e-6)
    assert Measure.angle(v2, v3) == pytest.approx(0, abs=1e-6)
    assert Measure.angle(v1, v3) == pytest.approx(0, abs=1e-6)


def test_trigonometric_order():
    assert Measure.trigonometricOrder(v1, v2) is True
    assert Measure.trigonometricOrder(v2, v3) is True
    assert Measure.trigonometricOrder(v1, v3) is True
