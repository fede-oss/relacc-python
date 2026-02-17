import math

from relacc.geom.vector import Vector


class Measure:
    """Gesture geometry measurements."""

    @staticmethod
    def sqDistance(a, b):
        return (a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y)

    @staticmethod
    def distance(a, b):
        return math.sqrt(Measure.sqDistance(a, b))

    @staticmethod
    def taxicab(a, b):
        return abs(a.X - b.X) + abs(a.Y - b.Y)

    @staticmethod
    def shortAngle(v, u):
        vLength = v.length()
        uLength = u.length()
        if abs(vLength * uLength) <= 5e-324:
            return 0
        cosAngle = Vector.dotProduct(v, u) / (vLength * uLength)
        if cosAngle <= -1:
            return math.pi
        if cosAngle >= 1:
            return 0
        return math.acos(cosAngle)

    @staticmethod
    def angle(v, u):
        angle = Measure.shortAngle(v, u)
        if not Measure.trigonometricOrder(v, u):
            angle = 2 * math.pi - angle
        return angle

    @staticmethod
    def trigonometricOrder(v, u):
        return Vector.dotProduct(v, u) >= 0
