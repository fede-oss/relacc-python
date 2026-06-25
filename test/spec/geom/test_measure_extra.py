import math

import pytest

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


@pytest.mark.parametrize(
    ("incoming_point", "outgoing_point", "expected_angle"),
    [
        (Point(1, 0, 0, 0), Point(0, 1, 0, 0), math.pi / 2),
        (Point(1, 0, 0, 0), Point(0, -1, 0, 0), 3 * math.pi / 2),
        (Point(1, 0, 0, 0), Point(2, 0, 0, 0), 0),
        (Point(1, 0, 0, 0), Point(-1, 0, 0, 0), math.pi),
        (Point(1, 0, 0, 0), Point(0, 0, 0, 0), 0),
    ],
)
def test_angle_uses_incoming_to_outgoing_orientation(
    incoming_point, outgoing_point, expected_angle
):
    p = Point(0, 0, 0, 0)
    incoming = Vector(p, incoming_point)
    outgoing = Vector(p, outgoing_point)

    assert Measure.angle(outgoing, incoming) == pytest.approx(expected_angle)
