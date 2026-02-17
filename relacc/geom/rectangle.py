from relacc.geom.point import Point


class Rectangle:
    """Create a rectangle."""

    def __init__(self, topLeft, bottomRight):
        self.topLeft = Point(topLeft)
        self.bottomRight = Point(bottomRight)

    def width(self):
        return abs(self.bottomRight.X - self.topLeft.X)

    def height(self):
        return abs(self.bottomRight.Y - self.topLeft.Y)

    def area(self):
        return self.width() * self.height()
