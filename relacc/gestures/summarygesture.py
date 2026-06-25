import math
import statistics
from collections import Counter

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


def _stroke_id_mode(stroke_ids):
    counts = Counter(stroke_ids)
    most_common_count = max(counts.values())
    return min(
        stroke_id
        for stroke_id, count in counts.items()
        if count == most_common_count
    )


def _validate_aggregate_point(point):
    if not (
        math.isfinite(point.X)
        and math.isfinite(point.Y)
        and math.isfinite(point.T)
    ):
        raise ValueError("Non-finite aggregate point value.")


def computeSummaryShapes(self, gestures, popularStrokeNum):
    xPoints = []
    yPoints = []
    tPoints = []
    strokeIds = []

    collectionLen = len(gestures)
    includedCount = 0
    for _ in range(self.refGesture.samplingRate):
        xPoints.append([])
        yPoints.append([])
        tPoints.append([])
        strokeIds.append([])

    for g in range(collectionLen):
        gesture = gestures[g]
        numStrk = PointSet.countStrokes(gesture.points)
        if popularStrokeNum > 0 and numStrk > popularStrokeNum:
            continue
        includedCount += 1
        points = self.alignGesture(gesture, self.alignmentType)
        for i in range(self.refGesture.samplingRate):
            pt = points[i]
            xPoints[i].append(pt.X)
            yPoints[i].append(pt.Y)
            tPoints[i].append(pt.T)
            strokeIds[i].append(pt.StrokeID)

    if includedCount == 0:
        raise ValueError("No gestures available to compute summary shapes.")

    centroid = []
    medoid = []
    for i in range(self.refGesture.samplingRate):
        stroke_id = _stroke_id_mode(strokeIds[i])
        centroid_point = Point(
            statistics.fmean(xPoints[i]),
            statistics.fmean(yPoints[i]),
            statistics.fmean(tPoints[i]),
            stroke_id,
        )
        medoid_point = Point(
            statistics.median(xPoints[i]),
            statistics.median(yPoints[i]),
            statistics.median(tPoints[i]),
            stroke_id,
        )
        _validate_aggregate_point(centroid_point)
        _validate_aggregate_point(medoid_point)
        centroid.append(centroid_point)
        medoid.append(medoid_point)

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
        selected_alignment = (
            PtAlignType.CHRONOLOGICAL if alignmentType is None else alignmentType
        )
        self.alignmentType = PtAlignType.normalize(selected_alignment)

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
        if alignmentType is None:
            alignmentType = self.alignmentType
        alignmentType = PtAlignType.normalize(alignmentType)
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
