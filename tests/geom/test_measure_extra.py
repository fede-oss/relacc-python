import math

from relacc.geom.measure import Measure
from relacc.geom.point import Point
from relacc.geom.vector import Vector


def test_taxicab_distance():
    a = Point(1, 2, 0, 0)
    b = Point(4, 7, 0, 0)
    assert Measure.taxicab(a, b) == 8


def test_short_angle_zero_vector():
    p = Point(0, 0, 0, 0)
    q = Point(1, 0, 0, 0)
    zero = Vector(p, p)
    non_zero = Vector(p, q)
    assert Measure.shortAngle(zero, non_zero) == 0


def test_short_angle_cosine_edges():
    p = Point(0, 0, 0, 0)
    right = Point(1, 0, 0, 0)
    right2 = Point(2, 0, 0, 0)
    left = Point(-1, 0, 0, 0)

    assert Measure.shortAngle(Vector(p, right), Vector(p, right2)) == 0
    assert math.isclose(Measure.shortAngle(Vector(p, right), Vector(p, left)), math.pi, abs_tol=1e-10)


def test_trigonometric_angle_non_ordered():
    p = Point(0, 0, 0, 0)
    a = Point(1, 0, 0, 0)
    b = Point(-1, 1, 0, 0)

    v = Vector(p, a)
    u = Vector(p, b)

    assert Measure.trigonometricOrder(v, u) is False
    assert Measure.angle(v, u) > math.pi
