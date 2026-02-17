from relacc.geom.measure import Measure


def _greedyCloudMatch(points1, points2):
    bestAlignment = []
    e = 0.5
    step = int(pow(len(points1), 1 - e))
    for i in range(0, len(points1), step):
        alignment1 = []
        alignment2 = []
        d1 = _cloudDistance(points1, points2, i, alignment1)
        d2 = _cloudDistance(points2, points1, i, alignment2)
        bestAlignment = alignment1[:] if d1 < d2 else alignment2[:]
    return bestAlignment


def _cloudDistance(pts1, pts2, start, arr):
    matched = []
    for _ in range(len(pts1)):
        matched.append(False)

    total = 0
    i = start
    while True:
        index = -1
        minimum = float("inf")
        for j in range(len(matched)):
            if not matched[j]:
                d = Measure.distance(pts1[i], pts2[j])
                if d < minimum:
                    minimum = d
                    index = j
        arr.append(index)
        matched[index] = True
        weight = 1 - (((i - start + len(pts1)) % len(pts1)) / len(pts1))
        total += weight * minimum
        i = (i + 1) % len(pts1)
        if i == start:
            break
    return total


def _hungarianMatch(weights):
    NOT_FOUND = -1
    NOT_MATCHED = -1

    n = len(weights[0])

    labelsLeft = [0] * n
    labelsRight = [0] * n
    for i in range(n):
        labelsRight[i] = 0
        labelsLeft[i] = weights[i][0]
        for j in range(1, n):
            labelsLeft[i] = max(labelsLeft[i], weights[i][j])

    matchingCount = 0
    matchingLeft = []
    matchingRight = []
    for i in range(n):
        matchingLeft.append(NOT_MATCHED)
        matchingRight.append(NOT_MATCHED)

    parent = [0] * n

    while matchingCount < n:
        u = 0
        while matchingLeft[u] != NOT_MATCHED:
            u += 1

        y = NOT_FOUND
        while y == NOT_FOUND:
            visitedLeft = [0.0] * n
            visitedRight = [0.0] * n

            queue = [u]
            visitedLeft[u] = True
            while len(queue) > 0 and y == NOT_FOUND:
                vertex = queue.pop(0)
                for j in range(n):
                    if not visitedRight[j]:
                        diff = weights[vertex][j] - (labelsLeft[vertex] + labelsRight[j])
                        if diff < 0:
                            diff = -diff
                        if diff < 10e-4:
                            parent[j] = vertex
                            visitedRight[j] = True
                            z = matchingRight[j]
                            if z == NOT_MATCHED:
                                y = j
                                break
                            queue.append(z)
                            visitedLeft[z] = True

            if y == NOT_FOUND:
                alpha = float("inf")
                for i in range(n):
                    if visitedLeft[i]:
                        for j in range(n):
                            if not visitedRight[j]:
                                diff = labelsLeft[i] + labelsRight[j] - weights[i][j]
                                if alpha > diff:
                                    alpha = diff

                for i in range(n):
                    if visitedLeft[i]:
                        labelsLeft[i] -= alpha
                    if visitedRight[i]:
                        labelsRight[i] += alpha
            else:
                index = y
                while index != NOT_MATCHED:
                    t = matchingLeft[parent[index]]
                    matchingLeft[parent[index]] = index
                    matchingRight[index] = parent[index]
                    index = t
                matchingCount += 1

    return matchingLeft


class PDollarAlt:
    """Alternative $P recognizer."""

    @staticmethod
    def match(points1, points2):
        m = _hungarianMatch(PDollarAlt.weights(points1, points2))
        return m

    @staticmethod
    def weights(points1, points2):
        n = len(points1)
        weights = [None] * n
        for i in range(n):
            weights[i] = []
            for j in range(n):
                weights[i].append(-Measure.sqDistance(points1[i], points2[j]))
        return weights

    @staticmethod
    def cost(matching, weights):
        cost = 0
        n = len(matching)
        for i in range(n):
            cost += -weights[i][matching[i]]
        return cost


PDollarAlt._greedyCloudMatch = _greedyCloudMatch
PDollarAlt._cloudDistance = _cloudDistance
PDollarAlt._hungarianMatch = _hungarianMatch
