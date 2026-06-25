from relacc.geom.point import Point
from relacc.geom.vector import Vector

from test.helpers.math import round_to


pt1 = Point(1, 2, 100, 0)
pt2 = Point(3, 4, 200, 0)
pt3 = Point(5, 6, 300, 0)
v1 = Vector(pt1, pt2)
v2 = Vector(pt2, pt3)
v3 = Vector(pt1, pt3)


def test_vector_length():
    assert round_to(v1.length()) == 2.83
    assert round_to(v2.length()) == 2.83
    assert round_to(v3.length()) == 5.66


def test_vector_dot_product():
    assert Vector.dotProduct(v1, v2) == 8
    assert Vector.dotProduct(v2, v3) == 16
    assert Vector.dotProduct(v1, v3) == 16


def test_vector_cross_product_cardinal_directions():
    p = Point(0, 0, 0, 0)
    right = Vector(p, Point(1, 0, 0, 0))
    up = Vector(p, Point(0, 1, 0, 0))
    down = Vector(p, Point(0, -1, 0, 0))
    left = Vector(p, Point(-1, 0, 0, 0))

    assert Vector.crossProduct(right, right) == 0
    assert Vector.crossProduct(right, up) == 1
    assert Vector.crossProduct(right, down) == -1
    assert Vector.crossProduct(right, left) == 0
