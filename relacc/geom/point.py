class Point:
    """Create a point with coordinates and stroke info."""

    def __init__(self, x=None, y=None, t=None, sid=None):
        self.X = 0
        self.Y = 0
        self.T = 0
        self.StrokeID = 0

        def loadMemberData(px, py, pt, psid):
            self.X = px
            self.Y = py
            self.T = pt
            self.StrokeID = psid

        if x is None:
            loadMemberData(0, 0, 0, 0)
        elif y is None and t is None and sid is None and hasattr(x, "X"):
            loadMemberData(x.X, x.Y, x.T, x.StrokeID)
        elif y is None and t is None and sid is None and isinstance(x, dict):
            loadMemberData(x["X"], x["Y"], x["T"], x["StrokeID"])
        elif t is None and sid is None:
            loadMemberData(x, y, 0, 0)
        else:
            loadMemberData(x, y, t, sid)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return (
            self.X == other.X
            and self.Y == other.Y
            and self.T == other.T
            and self.StrokeID == other.StrokeID
        )

    def __repr__(self):
        return f"Point(X={self.X}, Y={self.Y}, T={self.T}, StrokeID={self.StrokeID})"

    def maxXY(self):
        return max(self.X, self.Y)

    def minXY(self):
        return min(self.X, self.Y)

    def add(self, p2):
        p1 = self
        return Point(p1.X + p2.X, p1.Y + p2.Y, p1.T, p1.StrokeID)

    def subtract(self, p2):
        p1 = self
        return Point(p1.X - p2.X, p1.Y - p2.Y, p1.T, p1.StrokeID)

    def divideBy(self, scalar):
        if abs(scalar) < 10e-6:
            raise ValueError("Cannot divide by zero.")
        return Point(self.X / scalar, self.Y / scalar, self.T, self.StrokeID)

    def multiplyBy(self, scalar):
        return Point(scalar * self.X, scalar * self.Y, self.T, self.StrokeID)

    @staticmethod
    def absMin():
        return Point(float("-inf"), float("-inf"), 0, 0)

    @staticmethod
    def absMax():
        return Point(float("inf"), float("inf"), 0, 0)
