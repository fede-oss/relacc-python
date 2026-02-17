import math

from relacc.geom.measure import Measure
from relacc.geom.pointset import PointSet
from relacc.geom.vector import Vector
from relacc.gestures.ptaligntype import PtAlignType


def shapeError(gesture, summaryShape):
    errors = localShapeErrors(gesture, summaryShape)
    return mean(errors)


def shapeVariability(gesture, summaryShape):
    errors = localShapeErrors(gesture, summaryShape)
    return stdev(errors)


def localShapeErrors(gesture, summaryShape):
    summaryPts = summaryShape.getPoints()
    gesturePts = summaryShape.alignGesture(gesture)
    errors = []
    for i in range(len(gesturePts)):
        err = Measure.distance(summaryPts[i], gesturePts[i])
        errors.append(err)
    return errors


def lengthError(gesture, summaryShape):
    gestureLength = PointSet.pathLength(gesture.points)
    summaryLength = PointSet.pathLength(summaryShape.points)
    error = abs(gestureLength - summaryLength)
    return error


def sizeError(gesture, summaryShape):
    gestureArea = PointSet.boundingBox(gesture.points).area()
    summaryArea = PointSet.boundingBox(summaryShape.points).area()
    error = abs(gestureArea - summaryArea)
    return error


def bendingError(gesture, summaryShape):
    errors = localBendingErrors(gesture, summaryShape)
    return mean(errors)


def bendingVariability(gesture, summaryShape):
    errors = localBendingErrors(gesture, summaryShape)
    return stdev(errors)


def localBendingErrors(gesture, summaryShape):
    summaryPts = summaryShape.getPoints()
    gesturePts = summaryShape.alignGesture(gesture)
    kSummary = turningAngleArray(summaryPts)
    kGesture = turningAngleArray(gesturePts)
    errors = []
    for i in range(len(gesturePts)):
        err = abs(kGesture[i] - kSummary[i])
        errors.append(err)
    return errors


def turningAngleArray(points):
    n = len(points)
    k = []
    r = 1
    for i in range(n):
        angle = 0
        if i - r >= 0 and i + r < n:
            angle = Measure.angle(
                Vector(points[i], points[i + r]),
                Vector(points[i - r], points[i]),
            )
            if angle > math.pi:
                angle -= 2 * math.pi
        k.append(angle)
    return k


def timeError(gesture, summaryShape):
    summaryTime = productionTime(summaryShape)
    gestureTime = productionTime(gesture)
    error = abs(gestureTime - summaryTime)
    return error


def timeVariability(gesture, summaryShape):
    summaryPts = summaryShape.getPoints()
    gesturePts = summaryShape.alignGesture(gesture)
    errors = []
    for i in range(len(gesturePts)):
        errors.append(abs(gesturePts[i].T - summaryPts[i].T))
    return stdev(errors)


def velocityError(gesture, summaryShape):
    errors = localSpeedErrors(gesture, summaryShape)
    return mean(errors)


def velocityVariability(gesture, summaryShape):
    errors = localSpeedErrors(gesture, summaryShape)
    return stdev(errors)


def localSpeedErrors(gesture, summaryShape):
    summaryPts = summaryShape.getPoints()
    gesturePts = summaryShape.alignGesture(gesture)
    vSummary = speedArray(summaryPts)
    vGesture = speedArray(gesturePts)
    errors = []
    for i in range(len(gesturePts)):
        errors.append(abs(vGesture[i] - vSummary[i]))
    return errors


def productionTime(gesture):
    points = gesture.points
    if len(points) <= 1:
        return 0
    return points[len(points) - 1].T - points[0].T


def speedArray(points):
    n = len(points)
    v = []
    r = 1
    for i in range(n):
        index1 = max(0, i - r)
        index2 = min(i + r, n - 1)
        distance = PointSet.pathLength(points, index1, index2)
        time = points[index2].T - points[index1].T
        if abs(time) < 1e-5:
            v.append(0)
        else:
            v.append(distance / time)
    return v


def strokeError(gesture, summaryShape):
    return abs(numStrokes(gesture) - numStrokes(summaryShape))


def numStrokes(gesture):
    points = gesture.points
    numStk = 1 if len(points) > 0 else 0
    for i in range(1, len(points)):
        if points[i].StrokeID != points[i - 1].StrokeID:
            numStk += 1
    return numStk


def strokeOrderError(gesture, summaryShape):
    summaryPts = summaryShape.getPoints()

    gesturePts = summaryShape.alignGesture(gesture, PtAlignType.CHRONOLOGICAL)
    oDollarCost = 0
    for i in range(len(gesturePts)):
        oDollarCost += Measure.distance(gesturePts[i], summaryPts[i])

    gesturePts = summaryShape.alignGesture(gesture, PtAlignType.CLOUD_MATCH)
    pDollarCost = 0
    for i in range(len(gesturePts)):
        pDollarCost += Measure.distance(gesturePts[i], summaryPts[i])

    return abs(oDollarCost - pDollarCost)


def mean(arr):
    if len(arr) == 0:
        raise ValueError("Input set cannot be empty.")
    if len(arr) == 1:
        return arr[0]

    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total / len(arr)


def stdev(arr):
    if len(arr) == 0:
        raise ValueError("Input set cannot be empty.")
    if len(arr) == 1:
        return 0

    avg = mean(arr)
    sd = 0
    for i in range(len(arr)):
        item = arr[i]
        sd += (item - avg) * (item - avg)
    return math.sqrt(sd / (len(arr) - 1))
