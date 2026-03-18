import math

from relacc.dtw import (
    ddtw as derivative_dtw,
    dtw as classic_dtw,
    length_independent_dtw,
    weighted_derivative_dtw,
    weighted_dtw,
)
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


def _dtwComparisonPoints(gesture, summaryShape):
    return (
        summaryShape.alignGesture(gesture, PtAlignType.CHRONOLOGICAL),
        summaryShape.getPoints(),
    )


def dtwDistance(gesture, summaryShape):
    """Classic DTW total cost on chronological point sequences.

    Source: Sakoe and Chiba, "Dynamic programming algorithm optimization
    for spoken word recognition" (IEEE TASSP, 1978).
    https://doi.org/10.1109/TASSP.1978.1163055
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return classic_dtw(gesturePts, summaryPts).cost


def ldtwDistance(gesture, summaryShape):
    """Length-independent DTW, i.e. DTW normalized by warping-path length.

    Source: Chen et al., "Complex Handwriting Trajectory Recovery:
    Evaluation Metrics and Algorithm" (ACCV 2022).
    https://openaccess.thecvf.com/content/ACCV2022/papers/Chen_Complex_Handwriting_Trajectory_Recovery_Evaluation_Metrics_and_Algorithm_ACCV_2022_paper.pdf
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return length_independent_dtw(gesturePts, summaryPts)


def ddtwDistance(gesture, summaryShape):
    """Derivative DTW compares local trajectory trends instead of raw points.

    Source: Keogh and Pazzani, "Derivative Dynamic Time Warping" (SDM 2001).
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return derivative_dtw(gesturePts, summaryPts).cost


def wdtwDistance(gesture, summaryShape):
    """Weighted DTW penalizes alignments with larger phase offsets.

    Source: Jeong et al., "Weighted dynamic time warping for time series
    classification" (Pattern Recognition, 2011).
    https://doi.org/10.1016/j.patcog.2010.09.022
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return weighted_dtw(gesturePts, summaryPts).cost


def wddtwDistance(gesture, summaryShape):
    """Weighted derivative DTW combines slope-based matching and phase penalties.

    Source: Jeong et al., "Weighted dynamic time warping for time series
    classification" (Pattern Recognition, 2011).
    https://doi.org/10.1016/j.patcog.2010.09.022
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return weighted_derivative_dtw(gesturePts, summaryPts).cost


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


# Pythonic aliases for public API ergonomics while preserving original names.
shape_error = shapeError
shape_variability = shapeVariability
local_shape_errors = localShapeErrors
length_error = lengthError
size_error = sizeError
bending_error = bendingError
bending_variability = bendingVariability
local_bending_errors = localBendingErrors
turning_angle_array = turningAngleArray
time_error = timeError
time_variability = timeVariability
velocity_error = velocityError
velocity_variability = velocityVariability
local_speed_errors = localSpeedErrors
production_time = productionTime
speed_array = speedArray
stroke_error = strokeError
num_strokes = numStrokes
stroke_order_error = strokeOrderError
dtw_distance = dtwDistance
ldtw_distance = ldtwDistance
ddtw_distance = ddtwDistance
wdtw_distance = wdtwDistance
wddtw_distance = wddtwDistance
