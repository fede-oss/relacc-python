from relacc.geom.point import Point
from relacc.gestures.pdollaralt import PDollarAlt


def pointsA():
    return [Point(0, 0, 0, 0), Point(1, 0, 1, 0), Point(2, 0, 2, 0)]


def pointsB():
    return [Point(0, 0, 0, 0), Point(2, 0, 1, 0), Point(1, 0, 2, 0)]


def test_pdollar_weights():
    weights = PDollarAlt.weights(pointsA(), pointsB())
    assert len(weights) == 3
    assert len(weights[0]) == 3
    assert weights[0][0] == 0
    assert weights[0][1] < 0


def test_pdollar_uneven_sizes():
    weights = PDollarAlt.weights([Point(0, 0, 0, 0)], pointsB())
    assert len(weights) == 1
    assert len(weights[0]) == 1


def test_pdollar_hungarian_match():
    matching = PDollarAlt.match(pointsA(), pointsB())
    sorted_match = sorted(matching)
    assert len(matching) == 3
    assert sorted_match == [0, 1, 2]


def test_pdollar_cloud_distance_and_greedy_internals():
    arr = []
    distance = PDollarAlt._cloudDistance(pointsA(), pointsB(), 0, arr)
    greedy = PDollarAlt._greedyCloudMatch(pointsA(), pointsB())

    assert distance >= 0
    assert len(arr) == 3
    assert len(greedy) == 3


def test_pdollar_hungarian_internal_and_cost():
    weights = [[5, 1, 0], [0, 6, 1], [1, 0, 7]]
    matching = PDollarAlt._hungarianMatch(weights)
    assert matching == [0, 1, 2]

    cost = PDollarAlt.cost([0, 1], [[-1, -5], [-3, -2]])
    assert cost == 3


def test_pdollar_hungarian_label_update_path():
    # Repeated row maxima force the algorithm to update labels.
    weights = [[10, 9, 8], [10, 9, 8], [10, 9, 8]]
    matching = PDollarAlt._hungarianMatch(weights)
    assert sorted(matching) == [0, 1, 2]
