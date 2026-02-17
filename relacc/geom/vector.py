from relacc.geom.point import Point


class Vector:
    """Create a vector."""

    def __init__(self, a, b):
        self.vec = b.subtract(a)

    def length(self):
        orig = Point()
        from relacc.geom.measure import Measure

        return Measure.distance(self.vec, orig)

    @staticmethod
    def dotProduct(v, u):
        return v.vec.X * u.vec.X + v.vec.Y * u.vec.Y
