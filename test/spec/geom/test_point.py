import pytest

from relacc.geom.point import Point


def test_point_default_constructor():
    assert Point() == Point(0, 0, 0, 0)


def test_point_copy_constructor():
    assert Point({"X": 1, "Y": 1, "T": 0, "StrokeID": 0}) == Point(1, 1, 0, 0)


def test_point_all_args_constructor():
    assert Point(1, 1, 0, 0) == Point(1, 1, 0, 0)


def test_point_max_xy():
    assert Point(1, 2, 100, 0).maxXY() == 2
    assert Point(3, 4, 200, 0).maxXY() == 4


def test_point_min_xy():
    assert Point(1, 2, 100, 0).minXY() == 1
    assert Point(3, 4, 200, 0).minXY() == 3


def test_point_add():
    assert Point(1, 2, 100, 0).add(Point(3, 4, 200, 0)) == Point(4, 6, 100, 0)


def test_point_subtract():
    assert Point(1, 2, 100, 0).subtract(Point(3, 4, 200, 0)) == Point(-2, -2, 100, 0)


def test_point_divide_by():
    assert Point(1, 2, 100, 0).divideBy(2) == Point(0.5, 1, 100, 0)


def test_point_divide_by_zero_raises():
    with pytest.raises(ValueError, match="Cannot divide by zero\\."):
        Point(1, 2, 100, 0).divideBy(0)


def test_point_multiply_by():
    assert Point(1, 2, 100, 0).multiplyBy(2) == Point(2, 4, 100, 0)
