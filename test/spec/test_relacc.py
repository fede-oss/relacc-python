import pytest

from relacc import relacc as RelAcc
from relacc.gestures.ptaligntype import PtAlignType


class GestureShape:
    def __init__(self, points):
        self.points = points


class SummaryShape:
    def __init__(self, summary_points, chronological_points, cloud_points=None):
        self.points = summary_points
        self._chronological_points = chronological_points
        self._cloud_points = cloud_points or chronological_points

    def getPoints(self):
        return self.points

    def alignGesture(self, gesture, alignmentType=None):
        if alignmentType == PtAlignType.CLOUD_MATCH:
            return self._cloud_points
        return self._chronological_points


def test_shape_time_and_speed_metrics_are_based_on_small_arrays(p):
    summary_points = [p(0, 0, 0), p(2, 2, 10), p(4, 0, 20)]
    gesture_points = [p(0, 0, 0), p(2, 1, 10), p(4, 1, 20)]
    gesture = GestureShape(gesture_points)
    summary = SummaryShape(summary_points, gesture_points)

    assert RelAcc.localShapeErrors(gesture, summary) == [0, 1, 1]
    assert RelAcc.shapeError(gesture, summary) == pytest.approx(2 / 3)
    assert RelAcc.productionTime(gesture) == 20
    assert RelAcc.speedArray([p(0, 0, 10), p(1, 0, 10), p(2, 0, 10)]) == [0, 0, 0]


def test_bending_and_turning_angles_use_a_known_right_turn(p):
    points = [p(0, 0), p(1, 0), p(1, 1)]

    angles = RelAcc.turningAngleArray(points)

    assert angles[0] == 0
    assert angles[1] == pytest.approx(1.57079632679)
    assert angles[2] == 0
    assert RelAcc.curvatureArray(points) == pytest.approx([0, 1.57079632679 / 2, 0])


def test_stroke_metrics_are_readable_counts_and_lengths(p):
    points = [p(0, 0, 0, 0), p(3, 4, 10, 0), p(3, 4, 20, 1), p(6, 8, 40, 1)]
    gesture = GestureShape(points)

    assert RelAcc.numStrokes(gesture) == 2
    assert RelAcc.strokeLengths(points) == [5, 5]
    assert RelAcc.strokeDurations(points) == [10, 20]
    assert RelAcc.meanStrokeDurationValue(points) == 15
    assert RelAcc.strokeLengthStdValue(points) == 0


def test_dtw_metric_wrappers_use_chronological_alignment(p):
    summary_points = [p(0), p(1), p(2)]
    chronological_points = [p(0), p(2), p(3)]
    cloud_points = summary_points
    gesture = GestureShape(chronological_points)
    summary = SummaryShape(summary_points, chronological_points, cloud_points)

    assert RelAcc.dtwDistance(gesture, summary) == pytest.approx(2)
    assert RelAcc.ldtwDistance(gesture, summary) == pytest.approx(0.5)
    assert RelAcc.strokeOrderError(gesture, summary) == pytest.approx(2)


def test_mean_and_stdev_edge_cases_match_the_js_tests():
    assert RelAcc.mean([2, 4, 6]) == 4
    assert RelAcc.stdev([0, 2, 4]) == pytest.approx(2)

    with pytest.raises(ValueError, match="Input set cannot be empty"):
        RelAcc.mean([])
