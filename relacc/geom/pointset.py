from relacc.geom.measure import Measure
from relacc.geom.point import Point
from relacc.geom.rectangle import Rectangle


def _js_round(value):
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


class PointSet:
    """Point set geometry utilities."""

    @staticmethod
    def clone(points):
        pt = Point()
        if points is None:
            return pt

        newPoints = []
        for i in range(len(points)):
            newPoints.append(Point(points[i]))
        return newPoints

    @staticmethod
    def centroid(points):
        pt = Point()
        if points is None:
            return pt

        for i in range(len(points)):
            pt = pt.add(points[i])
        return pt.divideBy(len(points))

    @staticmethod
    def minPt(points):
        pt = Point()
        if points is None:
            return pt

        min_pt = Point.absMax()
        for p in points:
            if min_pt.X > p.X:
                min_pt.X = p.X
            if min_pt.Y > p.Y:
                min_pt.Y = p.Y
        return min_pt

    @staticmethod
    def maxPt(points):
        pt = Point()
        if points is None:
            return pt

        max_pt = Point.absMin()
        for p in points:
            if max_pt.X < p.X:
                max_pt.X = p.X
            if max_pt.Y < p.Y:
                max_pt.Y = p.Y
        return max_pt

    @staticmethod
    def boundingBox(points):
        return Rectangle(PointSet.minPt(points), PointSet.maxPt(points))

    @staticmethod
    def pathLength(points, startIndex=None, endIndex=None):
        if points is None:
            return 0
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(points) - 1

        if startIndex >= endIndex:
            return 0

        length = 0
        for i in range(startIndex + 1, endIndex + 1):
            length += Measure.distance(points[i], points[i - 1])
        return length

    @staticmethod
    def scale(points):
        min_pt = PointSet.minPt(points)
        max_pt = PointSet.maxPt(points)
        denom = max_pt.subtract(min_pt).maxXY()
        scaleFactor = float("inf") if denom == 0 else (1 / denom)
        newPoints = []
        if scaleFactor == scaleFactor and abs(scaleFactor) != float("inf"):
            for i in range(len(points)):
                newPoints.append(points[i].subtract(min_pt).multiplyBy(scaleFactor))
        else:
            for i in range(len(points)):
                newPoints.append(Point(points[i]))
        return newPoints

    @staticmethod
    def scaleTo(points, scaleFactor):
        PointSet.minPt(points)
        newPoints = []
        for i in range(len(points)):
            newPoints.append(points[i].multiplyBy(scaleFactor))
        return newPoints

    @staticmethod
    def translateBy(points, offset):
        newPoints = []
        for i in range(len(points)):
            newPoints.append(points[i].subtract(offset))
        return newPoints

    @staticmethod
    def unifResampling(points, n):
        if not points:
            raise ValueError("Points must not be empty.")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("Sampling rate must be a positive integer.")

        workingPoints = PointSet.clone(points)
        if n == 1:
            return [Point(workingPoints[0])]

        newPoints = []
        pathLen = PointSet.pathLength(workingPoints)
        if pathLen == 0:
            for i in range(n):
                newPoints.append(Point(workingPoints[0]))
            return newPoints

        intervalLen = pathLen / (n - 1)
        D = 0.0
        newPoints = [workingPoints[0]]
        i = 1
        while i < len(workingPoints):
            prevPoint = workingPoints[i - 1]
            currPoint = workingPoints[i]
            if currPoint.StrokeID == prevPoint.StrokeID:
                d = Measure.distance(prevPoint, currPoint)
                if d > 0 and (D + d) >= intervalLen:
                    s = (intervalLen - D) / d
                    qx = prevPoint.X + s * (currPoint.X - prevPoint.X)
                    qy = prevPoint.Y + s * (currPoint.Y - prevPoint.Y)
                    qt = prevPoint.T + s * (currPoint.T - prevPoint.T)
                    q = Point(qx, qy, qt, currPoint.StrokeID)
                    newPoints.append(q)
                    workingPoints.insert(i, q)
                    D = 0.0
                else:
                    D += d
            i += 1

        if len(newPoints) < n:
            for i in range(len(newPoints), n):
                newPoints.append(Point(workingPoints[len(workingPoints) - 1]))
        return newPoints

    @staticmethod
    def _contiguousStrokes(points):
        if not points:
            return []

        strokes = [[points[0]]]
        for point in points[1:]:
            if point.StrokeID != strokes[-1][-1].StrokeID:
                strokes.append([])
            strokes[-1].append(point)
        return strokes

    @staticmethod
    def _allocateResamplingPoints(strokes, n):
        stroke_count = len(strokes)
        remaining = n - stroke_count
        allocations = [1] * stroke_count
        if remaining == 0:
            return allocations

        lengths = [PointSet.pathLength(stroke) for stroke in strokes]
        total_length = sum(lengths)
        if total_length == 0:
            for index in range(remaining):
                allocations[index % stroke_count] += 1
            return allocations

        quotas = [remaining * length / total_length for length in lengths]
        floors = [int(quota) for quota in quotas]
        for index, floor in enumerate(floors):
            allocations[index] += floor

        unallocated = remaining - sum(floors)
        remainder_order = sorted(
            range(stroke_count),
            key=lambda index: (-(quotas[index] - floors[index]), index),
        )
        for index in remainder_order[:unallocated]:
            allocations[index] += 1
        return allocations

    @staticmethod
    def countStrokes(points):
        return len(PointSet._contiguousStrokes(points))

    @staticmethod
    def cumDistances(points, startIndex=None):
        if startIndex is None:
            startIndex = 0
        cum = 0
        lst = [0.0] * len(points)
        for i in range(startIndex + 1, len(points)):
            prevPoint = points[i - 1]
            currPoint = points[i]
            dist = Measure.distance(prevPoint, currPoint)
            cum += dist
            lst[i] = cum
        return lst

    @staticmethod
    def indexOfDistance(cumDistList, queryDist, startIndex=None):
        if startIndex is None:
            startIndex = 0
        index = -1
        for i in range(startIndex, len(cumDistList)):
            if cumDistList[i] > queryDist:
                index = i - 1
                break
        return index

    @staticmethod
    def maResampling(points, n):
        newPoints = [Point(points[0])]
        pathLen = PointSet.pathLength(points)
        if pathLen == 0:
            for i in range(1, n):
                newPoints.append(Point(points[0]))
            return newPoints

        intervalLen = pathLen / (n - 1)
        cumDistList = PointSet.cumDistances(points)
        lastSeenIndex = 0

        for i in range(1, n - 1):
            seenPtIndex = PointSet.indexOfDistance(cumDistList, intervalLen, lastSeenIndex)
            if seenPtIndex == -1:
                seenPtIndex = max(0, len(points) - 2)

            currPoint = points[lastSeenIndex]
            nextPoint = points[seenPtIndex]
            morePoint = points[seenPtIndex + 1]
            distCurrNext = intervalLen - PointSet.pathLength(points, lastSeenIndex, seenPtIndex)
            distNextMore = PointSet.pathLength(points, seenPtIndex, seenPtIndex + 1)
            t = distCurrNext / distNextMore
            q = Point(
                (1 - t) * nextPoint.X + t * morePoint.X,
                (1 - t) * nextPoint.Y + t * morePoint.Y,
                (1 - t) * nextPoint.T + t * morePoint.T,
                currPoint.StrokeID,
            )
            newPoints.append(q)
            lastSeenIndex = seenPtIndex + 1
            points.insert(lastSeenIndex, q)
            cumDistList = PointSet.cumDistances(points, lastSeenIndex)

        newPoints.append(Point(points[len(points) - 1]))
        return newPoints

    @staticmethod
    def eqDistStrokes(points, strkNum=None):
        return PointSet._contiguousStrokes(points)

    @staticmethod
    def eqResample(points, n):
        return PointSet.resample(points, n)

    @staticmethod
    def ensureResampling(newPoints, n):
        if len(newPoints) != n:
            raise ValueError(
                "Expected %s resampled points, received %s." % (n, len(newPoints))
            )

    @staticmethod
    def resample(points, n):
        if not points:
            raise ValueError("Points must not be empty.")
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError("Sampling rate must be an integer.")

        strokes = PointSet._contiguousStrokes(points)
        if n < len(strokes):
            raise ValueError(
                "Sampling rate must be at least the number of strokes (%s)."
                % len(strokes)
            )

        allocations = PointSet._allocateResamplingPoints(strokes, n)
        newPoints = []
        for stroke, allocation in zip(strokes, allocations):
            newPoints.extend(PointSet.unifResampling(stroke, allocation))
        PointSet.ensureResampling(newPoints, n)
        return newPoints
