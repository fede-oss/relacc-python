from relacc.geom.point import Point


def test_point_constructor_xy_only():
    assert Point(3, 4) == Point(3, 4, 0, 0)


def test_point_abs_min_max():
    min_pt = Point.absMin()
    max_pt = Point.absMax()

    assert min_pt.X == float("-inf")
    assert min_pt.Y == float("-inf")
    assert max_pt.X == float("inf")
    assert max_pt.Y == float("inf")


def test_point_repr_and_non_point_equality():
    pt = Point(1, 2, 3, 4)
    assert (pt == object()) is False
    assert "Point(" in repr(pt)
