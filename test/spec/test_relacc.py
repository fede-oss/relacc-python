import math

import pytest

from relacc import relacc as RelAcc
from relacc.geom.point import Point
from relacc.gestures.ptaligntype import PtAlignType


def p(x, y, t, sid):
    return Point(x, y, t, sid)


def _fixture_shapes():
    summaryPts = [p(0, 0, 0, 0), p(2, 2, 10, 0), p(4, 0, 20, 1)]
    chronoPts = [p(0, 0, 0, 0), p(2, 1, 10, 0), p(4, 1, 20, 1)]
    cloudPts = [p(0, 0, 0, 0), p(2, 2, 10, 0), p(4, 0, 20, 1)]

    gesture = type("GestureObj", (), {"points": chronoPts})()

    class SummaryShape:
        points = summaryPts

        @staticmethod
        def getPoints():
            return summaryPts

        @staticmethod
        def alignGesture(_, alignmentType=None):
            if alignmentType == PtAlignType.CLOUD_MATCH:
                return cloudPts
            return chronoPts

    return summaryPts, chronoPts, cloudPts, gesture, SummaryShape()


def test_local_and_aggregate_shape_errors():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    local = RelAcc.localShapeErrors(gesture, summaryShape)
    assert local == [0, 1, 1]
    assert RelAcc.shapeError(gesture, summaryShape) == pytest.approx(RelAcc.mean(local))
    assert RelAcc.shapeVariability(gesture, summaryShape) == pytest.approx(RelAcc.stdev(local))


def test_geometric_metrics():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    assert RelAcc.lengthError(gesture, summaryShape) > 0
    assert RelAcc.sizeError(gesture, summaryShape) == 4
    assert RelAcc.bendingError(gesture, summaryShape) >= 0
    assert RelAcc.bendingVariability(gesture, summaryShape) >= 0
    assert len(RelAcc.localBendingErrors(gesture, summaryShape)) == 3


def test_turning_angles_normalization():
    points = [p(0, 0, 0, 0), p(1, 0, 1, 0), p(0.5, -0.866, 2, 0)]
    angles = RelAcc.turningAngleArray(points)
    assert angles[0] == 0
    assert angles[2] == 0
    assert angles[1] < 0


def test_kinematic_metrics():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    assert RelAcc.timeError(gesture, summaryShape) == 0
    assert RelAcc.timeVariability(gesture, summaryShape) == 0
    assert RelAcc.velocityError(gesture, summaryShape) > 0
    assert RelAcc.velocityVariability(gesture, summaryShape) > 0
    assert len(RelAcc.localSpeedErrors(gesture, summaryShape)) == 3


def test_production_time_and_speed_arrays():
    _, _, _, gesture, _ = _fixture_shapes()
    assert RelAcc.productionTime(type("G", (), {"points": [p(1, 1, 7, 0)]})()) == 0
    assert RelAcc.productionTime(gesture) == 20

    zeroTime = [p(0, 0, 10, 0), p(1, 0, 10, 0), p(2, 0, 10, 0)]
    speed = RelAcc.speedArray(zeroTime)
    assert speed == [0, 0, 0]


def test_articulation_metrics():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    assert RelAcc.strokeError(gesture, summaryShape) == 0
    assert RelAcc.numStrokes(type("G", (), {"points": []})()) == 0
    assert RelAcc.numStrokes(gesture) == 2
    assert RelAcc.strokeOrderError(gesture, summaryShape) > 0


def test_dtw_family_metrics():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    assert RelAcc.dtwDistance(gesture, summaryShape) == pytest.approx(2.0)
    assert RelAcc.ldtwDistance(gesture, summaryShape) == pytest.approx(2.0 / 3.0)
    assert RelAcc.ddtwDistance(gesture, summaryShape) == pytest.approx(3.25)
    assert RelAcc.wdtwDistance(gesture, summaryShape) < RelAcc.dtwDistance(gesture, summaryShape)
    assert RelAcc.wddtwDistance(gesture, summaryShape) < RelAcc.ddtwDistance(gesture, summaryShape)


def test_weighted_dtw_metric_wrappers_accept_custom_penalty():
    _, _, _, gesture, summaryShape = _fixture_shapes()

    weak_penalty = RelAcc.wdtwDistance(gesture, summaryShape, penalty_g=0.05)
    strong_penalty = RelAcc.wdtwDistance(gesture, summaryShape, penalty_g=1.0)
    weak_derivative = RelAcc.wddtwDistance(gesture, summaryShape, penalty_g=0.05)
    strong_derivative = RelAcc.wddtwDistance(gesture, summaryShape, penalty_g=1.0)

    assert weak_penalty != pytest.approx(strong_penalty)
    assert weak_derivative != pytest.approx(strong_derivative)


def test_weighted_dtw_metric_wrappers_reject_negative_penalty():
    _, _, _, gesture, summaryShape = _fixture_shapes()

    with pytest.raises(ValueError, match="penalty_g must be >= 0\\."):
        RelAcc.wdtwDistance(gesture, summaryShape, penalty_g=-0.1)

    with pytest.raises(ValueError, match="penalty_g must be >= 0\\."):
        RelAcc.wddtwDistance(gesture, summaryShape, penalty_g=-0.1)


def test_dtw_family_metrics_ignore_summary_alignment_type():
    summaryPts = [p(0, 0, 0, 0), p(10, 0, 10, 0)]
    chronoPts = [p(0, 0, 0, 0), p(20, 0, 10, 0)]
    cloudPts = [p(0, 0, 0, 0), p(10, 0, 10, 0)]
    gesture = type("GestureObj", (), {"points": chronoPts})()

    class SummaryShape:
        points = summaryPts
        alignmentType = PtAlignType.CLOUD_MATCH

        @staticmethod
        def getPoints():
            return summaryPts

        @staticmethod
        def alignGesture(_, alignmentType=None):
            if alignmentType == PtAlignType.CLOUD_MATCH:
                return cloudPts
            return chronoPts

    summaryShape = SummaryShape()
    assert summaryShape.alignGesture(gesture, summaryShape.alignmentType) == cloudPts
    assert summaryShape.alignGesture(gesture, PtAlignType.CHRONOLOGICAL) == chronoPts
    assert RelAcc.dtwDistance(gesture, summaryShape) == 10
    assert RelAcc.ldtwDistance(gesture, summaryShape) == 5


def test_mean_stdev_edge_cases():
    with pytest.raises(ValueError, match="Input set cannot be empty\\."):
        RelAcc.mean([])

    assert RelAcc.mean([4]) == 4
    assert RelAcc.mean([2, 4, 6]) == 4

    with pytest.raises(ValueError, match="Input set cannot be empty\\."):
        RelAcc.stdev([])

    assert RelAcc.stdev([4]) == 0
    assert math.isclose(RelAcc.stdev([0, 2, 4]), 2, abs_tol=1e-10)


def test_snake_case_aliases_match_original_api():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    assert RelAcc.shape_error(gesture, summaryShape) == RelAcc.shapeError(gesture, summaryShape)
    assert RelAcc.length_error(gesture, summaryShape) == RelAcc.lengthError(gesture, summaryShape)
    assert RelAcc.stroke_order_error(gesture, summaryShape) == RelAcc.strokeOrderError(gesture, summaryShape)
    assert RelAcc.num_strokes(gesture) == RelAcc.numStrokes(gesture)
    assert RelAcc.dtw_distance(gesture, summaryShape) == RelAcc.dtwDistance(gesture, summaryShape)
    assert RelAcc.ldtw_distance(gesture, summaryShape) == RelAcc.ldtwDistance(gesture, summaryShape)
    assert RelAcc.ddtw_distance(gesture, summaryShape) == RelAcc.ddtwDistance(gesture, summaryShape)
    assert RelAcc.wdtw_distance(gesture, summaryShape) == RelAcc.wdtwDistance(gesture, summaryShape)
    assert RelAcc.wddtw_distance(gesture, summaryShape) == RelAcc.wddtwDistance(gesture, summaryShape)
