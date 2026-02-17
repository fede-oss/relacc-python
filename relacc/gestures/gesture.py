from relacc.geom.point import Point
from relacc.geom.pointset import PointSet


class Gesture:
    """Create a gesture from a set of points."""

    def __init__(self, points, name, samplingRate=None):
        self.originalPoints = None
        self.points = None
        self.name = name
        self.samplingRate = samplingRate or 32

        if not name:
            raise ValueError("Gesture name cannot be empty.")
        if not points or len(points) == 0:
            raise ValueError("Gesture points cannot be empty.")

        self.originalPoints = []
        for i in range(len(points)):
            self.originalPoints.append(Point(points[i]))
        self.preprocess(self.samplingRate)

    def preprocess(self, rate):
        self.samplingRate = rate
        self.points = PointSet.resample(PointSet.clone(self.originalPoints), rate)
        self.points = PointSet.translateBy(self.points, PointSet.centroid(self.points))
