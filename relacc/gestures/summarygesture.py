from relacc.geom.measure import Measure
from relacc.geom.point import Point
from relacc.geom.pointset import PointSet
from relacc.gestures.gesture import Gesture
from relacc.gestures.pdollaralt import PDollarAlt
from relacc.gestures.ptaligntype import PtAlignType


def numericSort(a, b):
    return a - b


def getPointsForAlignment(gesture):
    points = PointSet.clone(gesture.points)
    center = Point()
    return PointSet.translateBy(points, center)


def computeSummaryShapes(self, gestures, popularStrokeNum):
    centroid = []
    medoid = []
    xPoints = []
    yPoints = []

    collectionLen = len(gestures)
    includedCount = 0
    for _ in range(self.refGesture.samplingRate):
        centroid.append(Point())
        medoid.append(Point())

    for g in range(collectionLen):
        gesture = gestures[g]
        numStrk = PointSet.countStrokes(gesture.points)
        if popularStrokeNum > 0 and numStrk > popularStrokeNum:
            continue
        includedCount += 1
        points = self.alignGesture(gesture, self.alignmentType)
        for i in range(self.refGesture.samplingRate):
            pt = points[i]
            centroid[i] = pt.add(centroid[i])
            if i >= len(xPoints):
                xPoints.append([])
                yPoints.append([])
            xPoints[i].append(pt.X)
            yPoints[i].append(pt.Y)

    if includedCount == 0:
        raise ValueError("No gestures available to compute summary shapes.")

    pivot = int(includedCount / 2)
    for i in range(self.refGesture.samplingRate):
        centroid[i] = centroid[i].divideBy(includedCount)
        xPoints[i].sort()
        yPoints[i].sort()
        xval = xPoints[i][pivot] if pivot < len(xPoints[i]) else None
        yval = yPoints[i][pivot] if pivot < len(yPoints[i]) else None
        medoid[i] = Point(xval, yval, centroid[i].T, centroid[i].StrokeID)

    return {"centroid": centroid, "medoid": medoid}


class SummaryGesture(Gesture):
    """Compute summary gesture from a set of gestures."""

    def __init__(self, gestures, alignmentType=None, summaryShape=None, usePopularStrokeNum=None):
        refg = gestures[0]
        super().__init__(refg.points, refg.name, refg.samplingRate)

        collectionLen = len(gestures)
        for i in range(1, collectionLen):
            if gestures[i].name != refg.name:
                raise ValueError("Gesture names cannot be different.")

        self.refGesture = refg
        self.alignmentType = alignmentType or PtAlignType.CHRONOLOGICAL

        popularStrokeNum = 0
        if usePopularStrokeNum:
            strokeHist = {}
            for g in range(collectionLen):
                gesture = gestures[g]
                numStrk = PointSet.countStrokes(gesture.points)
                if numStrk not in strokeHist:
                    strokeHist[numStrk] = 0
                strokeHist[numStrk] += 1

            popularStrokeVal = 0
            for key, val in strokeHist.items():
                if val > popularStrokeVal:
                    popularStrokeVal = val
                    popularStrokeNum = int(key)

        shapes = computeSummaryShapes(self, gestures, popularStrokeNum)

        def knn(referenceGesture):
            idx = -1
            minimum = float("inf")
            for g in range(collectionLen):
                points = self.alignGesture(gestures[g], self.alignmentType)
                distance = 0
                for i in range(refg.samplingRate):
                    distance += Measure.sqDistance(referenceGesture[i], points[i])
                if distance < minimum:
                    minimum = distance
                    idx = g
            return idx

        if summaryShape == "centroid":
            self.originalPoints = shapes["centroid"]
            self.closestIndex = None
        elif summaryShape == "medoid":
            self.originalPoints = shapes["medoid"]
            self.closestIndex = None
        elif summaryShape == "kmedoid":
            closestIndex = knn(shapes["medoid"])
            self.originalPoints = gestures[closestIndex].originalPoints
            self.closestIndex = closestIndex
        elif summaryShape == "kcentroid":
            closestIndex = knn(shapes["centroid"])
            self.originalPoints = gestures[closestIndex].originalPoints
            self.closestIndex = closestIndex

        self.preprocess(self.samplingRate)

    def alignGesture(self, gesture, alignmentType=None):
        points = getPointsForAlignment(gesture)
        if alignmentType == PtAlignType.CHRONOLOGICAL:
            return points

        alignment = PDollarAlt.match(getPointsForAlignment(self.refGesture), points)
        newPoints = []
        for i in range(self.refGesture.samplingRate):
            pt = points[alignment[i]]
            pt.StrokeID = 0
            newPoints.append(pt)
        return newPoints

    def getPoints(self):
        return getPointsForAlignment(self)


SummaryGesture.getPointsForAlignment = staticmethod(getPointsForAlignment)
SummaryGesture.computeSummaryShapes = staticmethod(computeSummaryShapes)
