import traceback

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
        newPoints = []
        pathLen = PointSet.pathLength(points)
        if pathLen == 0:
            for i in range(1, n):
                newPoints.append(Point(points[0]))
            return newPoints

        intervalLen = pathLen / (n - 1)
        D = 0.0
        newPoints = [points[0]]
        i = 1
        while i < len(points):
            prevPoint = points[i - 1]
            currPoint = points[i]
            if currPoint.StrokeID == prevPoint.StrokeID:
                d = Measure.distance(prevPoint, currPoint)
                if d > 0 and (D + d) >= intervalLen:
                    s = (intervalLen - D) / d
                    qx = prevPoint.X + s * (currPoint.X - prevPoint.X)
                    qy = prevPoint.Y + s * (currPoint.Y - prevPoint.Y)
                    qt = prevPoint.T + s * (currPoint.T - prevPoint.T)
                    q = Point(qx, qy, qt, currPoint.StrokeID)
                    newPoints.append(q)
                    points.insert(i, q)
                    D = 0.0
                else:
                    D += d
            i += 1

        if len(newPoints) < n:
            for i in range(len(newPoints), n):
                newPoints.append(Point(points[len(points) - 1]))
        return newPoints

    @staticmethod
    def countStrokes(points):
        arr = []
        for i in range(1, len(points) - 1):
            prevPoint = points[i - 1]
            nextPoint = points[i]
            if nextPoint.StrokeID != prevPoint.StrokeID:
                arr.append(prevPoint.StrokeID)

        uniq = []
        for i, elem in enumerate(arr):
            if len(arr) - 1 - arr[::-1].index(elem) == i:
                uniq.append(elem)
        return len(uniq)

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
        if not points or len(points) == 0:
            return []

        strokes = [None] * (strkNum or 0)
        c = 0
        if len(points) == 1:
            strokes[c] = [points[0]]
            return strokes

        strokes[c] = []
        for i in range(0, len(points) - 1):
            currPoint = points[i]
            nextPoint = points[i + 1]
            strokes[c].append(currPoint)
            if currPoint.StrokeID != nextPoint.StrokeID:
                c += 1
                strokes[c] = []

        strokes[c].append(nextPoint)
        return strokes

    @staticmethod
    def eqResample(points, n):
        strkNum = PointSet.countStrokes(points)
        strokes = PointSet.eqDistStrokes(points, strkNum)
        pointsPerStroke = _js_round(n / strkNum)
        newPoints = []
        for stroke in strokes:
            resampled = PointSet.unifResampling(stroke, pointsPerStroke)
            newPoints = newPoints + resampled
        PointSet.ensureResampling(newPoints, n)
        return newPoints

    @staticmethod
    def ensureResampling(newPoints, n):
        if len(newPoints) != n:
            traceback.print_stack()
            raise SystemExit(1)

    @staticmethod
    def resample(points, n):
        newPoints = PointSet.unifResampling(points, n)
        PointSet.ensureResampling(newPoints, n)
        return newPoints
