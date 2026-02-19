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
