from relacc.geom.point import Point
from relacc.geom.rectangle import Rectangle


def test_rectangle_fields_and_metrics():
    pt1 = Point(1, 2, 100, 0)
    pt2 = Point(3, 4, 200, 0)
    rectangle = Rectangle(pt1, pt2)

    assert rectangle.topLeft == pt1
    assert rectangle.bottomRight == pt2
    assert rectangle.width() == 2
    assert rectangle.height() == 2
    assert rectangle.area() == 4
