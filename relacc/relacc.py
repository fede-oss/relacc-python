import math

from scipy.stats import wasserstein_distance

from relacc.dtw import (
    DEFAULT_WARPING_PENALTY,
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


EPSILON = 1e-12


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


def _chronologicalComparisonPoints(gesture, summaryShape):
    return (
        summaryShape.alignGesture(gesture, PtAlignType.CHRONOLOGICAL),
        summaryShape.getPoints(),
    )


def _finiteMean(values, fallback=0.0):
    finite_values = [value for value in values if math.isfinite(value)]
    if len(finite_values) == 0:
        return fallback
    return mean(finite_values)


def curvatureArray(points):
    n = len(points)
    curvatures = [0.0] * n
    for i in range(1, n - 1):
        prevPoint = points[i - 1]
        currPoint = points[i]
        nextPoint = points[i + 1]
        if (
            currPoint.StrokeID != prevPoint.StrokeID
            or currPoint.StrokeID != nextPoint.StrokeID
        ):
            continue

        localLength = Measure.distance(prevPoint, currPoint) + Measure.distance(
            currPoint,
            nextPoint,
        )
        if localLength <= EPSILON:
            continue

        angle = Measure.shortAngle(
            Vector(prevPoint, currPoint),
            Vector(currPoint, nextPoint),
        )
        curvatures[i] = angle / localLength
    return curvatures


def curvature(gesture, summaryShape):
    gesturePts, summaryPts = _chronologicalComparisonPoints(gesture, summaryShape)
    gestureCurvature = curvatureArray(gesturePts)
    summaryCurvature = curvatureArray(summaryPts)
    if len(gestureCurvature) == 0 or len(summaryCurvature) == 0:
        return 0.0
    return float(wasserstein_distance(summaryCurvature, gestureCurvature))


def cornerSlowdownRatio(points):
    curvatures = curvatureArray(points)
    speeds = speedArray(points)
    globalSpeeds = [
        speed
        for speed in speeds
        if math.isfinite(speed) and speed > EPSILON
    ]
    if len(globalSpeeds) == 0:
        return 1.0

    cornerCurvatures = [
        curvature
        for curvature in curvatures
        if math.isfinite(curvature) and curvature > EPSILON
    ]
    if len(cornerCurvatures) == 0:
        return 1.0

    threshold = sorted(cornerCurvatures)[int(0.75 * (len(cornerCurvatures) - 1))]
    cornerSpeeds = [
        speed
        for curvature, speed in zip(curvatures, speeds)
        if curvature >= threshold and math.isfinite(speed) and speed > EPSILON
    ]
    if len(cornerSpeeds) == 0:
        return 1.0

    return _finiteMean(cornerSpeeds) / _finiteMean(globalSpeeds)


def cornerSlowdown(gesture, summaryShape):
    gesturePts, summaryPts = _chronologicalComparisonPoints(gesture, summaryShape)
    return abs(cornerSlowdownRatio(gesturePts) - cornerSlowdownRatio(summaryPts))


def twoThirdsPowerLawR2Value(points):
    curvatures = curvatureArray(points)
    speeds = speedArray(points)
    xs = []
    ys = []
    for curvature, speed in zip(curvatures, speeds):
        if (
            math.isfinite(curvature)
            and math.isfinite(speed)
            and curvature > EPSILON
            and speed > EPSILON
        ):
            xs.append(math.log(curvature))
            ys.append(math.log(speed))

    if len(xs) < 2:
        return 0.0

    xMean = mean(xs)
    yMean = mean(ys)
    ssX = sum((value - xMean) * (value - xMean) for value in xs)
    ssY = sum((value - yMean) * (value - yMean) for value in ys)
    if ssX <= EPSILON or ssY <= EPSILON:
        return 0.0

    covariance = sum((x - xMean) * (y - yMean) for x, y in zip(xs, ys))
    slope = covariance / ssX
    intercept = yMean - slope * xMean
    ssResidual = sum(
        (y - (slope * x + intercept)) * (y - (slope * x + intercept))
        for x, y in zip(xs, ys)
    )
    r2 = 1.0 - (ssResidual / ssY)
    return min(1.0, max(0.0, r2))


def twoThirdsPowerLawR2(gesture, summaryShape):
    gesturePts, summaryPts = _chronologicalComparisonPoints(gesture, summaryShape)
    return abs(
        twoThirdsPowerLawR2Value(gesturePts)
        - twoThirdsPowerLawR2Value(summaryPts)
    )


def _dftFrequencyEnergy(xs, ys, frequency):
    realX = 0.0
    imagX = 0.0
    realY = 0.0
    imagY = 0.0
    n = len(xs)
    for i in range(n):
        angle = 2.0 * math.pi * frequency * i / n
        cosine = math.cos(angle)
        sine = math.sin(angle)
        realX += xs[i] * cosine
        imagX -= xs[i] * sine
        realY += ys[i] * cosine
        imagY -= ys[i] * sine
    return realX * realX + imagX * imagX + realY * realY + imagY * imagY


def highFrequencyRatioValue(points):
    n = len(points)
    if n < 4:
        return 0.0

    xMean = _finiteMean([point.X for point in points])
    yMean = _finiteMean([point.Y for point in points])
    xs = [point.X - xMean for point in points]
    ys = [point.Y - yMean for point in points]
    maxFrequency = n // 2
    splitFrequency = max(1, maxFrequency // 2)
    lowEnergy = 0.0
    highEnergy = 0.0
    for frequency in range(1, maxFrequency + 1):
        energy = _dftFrequencyEnergy(xs, ys, frequency)
        if frequency <= splitFrequency:
            lowEnergy += energy
        else:
            highEnergy += energy

    totalEnergy = lowEnergy + highEnergy
    if totalEnergy <= EPSILON:
        return 0.0
    return highEnergy / totalEnergy


def highFrequencyRatio(gesture, summaryShape):
    gesturePts, summaryPts = _chronologicalComparisonPoints(gesture, summaryShape)
    return abs(
        highFrequencyRatioValue(gesturePts)
        - highFrequencyRatioValue(summaryPts)
    )


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
        if time <= 1e-5:
            v.append(0)
        else:
            v.append(distance / time)
    return v


def strokeError(gesture, summaryShape):
    return abs(numStrokes(gesture) - numStrokes(summaryShape))


def strokeGroups(points):
    if len(points) == 0:
        return []

    groups = [[points[0]]]
    for point in points[1:]:
        if point.StrokeID == groups[-1][-1].StrokeID:
            groups[-1].append(point)
        else:
            groups.append([point])
    return groups


def strokeLengths(points):
    return [PointSet.pathLength(stroke) for stroke in strokeGroups(points)]


def strokeDurations(points):
    durations = []
    for stroke in strokeGroups(points):
        times = [point.T for point in stroke]
        durations.append(max(times) - min(times))
    return durations


def strokeLengthStdValue(points):
    lengths = strokeLengths(points)
    if len(lengths) < 2:
        return 0.0
    return stdev(lengths)


def strokeLengthStd(gesture, summaryShape):
    return abs(
        strokeLengthStdValue(gesture.points)
        - strokeLengthStdValue(summaryShape.points)
    )


def meanStrokeDurationValue(points):
    durations = strokeDurations(points)
    if len(durations) == 0:
        return 0.0
    return mean(durations)


def meanStrokeDuration(gesture, summaryShape):
    return abs(
        meanStrokeDurationValue(gesture.points)
        - meanStrokeDurationValue(summaryShape.points)
    )


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


def dtwDistance(gesture, summaryShape, window=None):
    """Classic DTW total cost on chronological point sequences.

    Source: Sakoe and Chiba, "Dynamic programming algorithm optimization
    for spoken word recognition" (IEEE TASSP, 1978).
    https://doi.org/10.1109/TASSP.1978.1163055
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return classic_dtw(gesturePts, summaryPts, window=window).cost


def ldtwDistance(gesture, summaryShape, window=None):
    """Length-independent DTW, i.e. DTW normalized by warping-path length.

    Source: Chen et al., "Complex Handwriting Trajectory Recovery:
    Evaluation Metrics and Algorithm" (ACCV 2022).
    https://openaccess.thecvf.com/content/ACCV2022/papers/Chen_Complex_Handwriting_Trajectory_Recovery_Evaluation_Metrics_and_Algorithm_ACCV_2022_paper.pdf
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return length_independent_dtw(gesturePts, summaryPts, window=window)


def ddtwDistance(gesture, summaryShape, window=None):
    """Derivative DTW compares local trajectory trends instead of raw points.

    Source: Keogh and Pazzani, "Derivative Dynamic Time Warping" (SDM 2001).
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return derivative_dtw(gesturePts, summaryPts, window=window).cost


def wdtwDistance(gesture, summaryShape, penalty_g=DEFAULT_WARPING_PENALTY, window=None):
    """Weighted DTW penalizes alignments with larger phase offsets.

    Source: Jeong et al., "Weighted dynamic time warping for time series
    classification" (Pattern Recognition, 2011).
    https://doi.org/10.1016/j.patcog.2010.09.022

    ``penalty_g`` controls the steepness of the logistic phase penalty.
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return weighted_dtw(gesturePts, summaryPts, penalty_g=penalty_g, window=window).cost


def wddtwDistance(gesture, summaryShape, penalty_g=DEFAULT_WARPING_PENALTY, window=None):
    """Weighted derivative DTW combines slope-based matching and phase penalties.

    Source: Jeong et al., "Weighted dynamic time warping for time series
    classification" (Pattern Recognition, 2011).
    https://doi.org/10.1016/j.patcog.2010.09.022

    ``penalty_g`` controls the steepness of the logistic phase penalty.
    """

    gesturePts, summaryPts = _dtwComparisonPoints(gesture, summaryShape)
    return weighted_derivative_dtw(
        gesturePts,
        summaryPts,
        penalty_g=penalty_g,
        window=window,
    ).cost


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
curvature_array = curvatureArray
corner_slowdown_ratio = cornerSlowdownRatio
corner_slowdown = cornerSlowdown
two_thirds_power_law_r2_value = twoThirdsPowerLawR2Value
two_thirds_power_law_r2 = twoThirdsPowerLawR2
high_frequency_ratio_value = highFrequencyRatioValue
high_frequency_ratio = highFrequencyRatio
production_time = productionTime
speed_array = speedArray
stroke_error = strokeError
stroke_groups = strokeGroups
stroke_lengths = strokeLengths
stroke_durations = strokeDurations
stroke_length_std_value = strokeLengthStdValue
stroke_length_std = strokeLengthStd
mean_stroke_duration_value = meanStrokeDurationValue
mean_stroke_duration = meanStrokeDuration
num_strokes = numStrokes
stroke_order_error = strokeOrderError
dtw_distance = dtwDistance
ldtw_distance = ldtwDistance
ddtw_distance = ddtwDistance
wdtw_distance = wdtwDistance
wddtw_distance = wddtwDistance
