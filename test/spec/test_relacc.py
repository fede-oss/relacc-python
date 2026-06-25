import math

import pytest

from relacc import relacc as RelAcc
from relacc.geom.point import Point
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture


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


def _summary_for(summaryPts, gesturePts):
    class SummaryShape:
        points = summaryPts

        @staticmethod
        def getPoints():
            return summaryPts

        @staticmethod
        def alignGesture(_, alignmentType=None):
            return gesturePts

    return SummaryShape()


def test_local_and_aggregate_shape_errors():
    _, _, _, gesture, summaryShape = _fixture_shapes()
    local = RelAcc.localShapeErrors(gesture, summaryShape)
    assert local == [0, 1, 1]
    assert RelAcc.shapeError(gesture, summaryShape) == pytest.approx(RelAcc.mean(local))
    assert RelAcc.shapeVariability(gesture, summaryShape) == pytest.approx(RelAcc.stdev(local))


def test_shape_error_uses_summary_stored_alignment(monkeypatch):
    points = [p(0, 0, 0, 0), p(1, 0, 10, 0), p(2, 0, 20, 0)]
    gestures = [Gesture(points, "line", 3), Gesture(points, "line", 3)]
    summary = SummaryGesture(gestures, PtAlignType.CHRONOLOGICAL)

    monkeypatch.setattr(
        "relacc.gestures.summarygesture.PDollarAlt.match",
        lambda reference, candidate: [2, 1, 0],
    )

    summary.alignmentType = PtAlignType.CLOUD_MATCH
    cloud_error = RelAcc.shapeError(gestures[1], summary)
    summary.alignmentType = PtAlignType.CHRONOLOGICAL
    chronological_error = RelAcc.shapeError(gestures[1], summary)

    assert cloud_error > 0
    assert chronological_error == 0


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


@pytest.mark.parametrize("summary_shape", ["centroid", "medoid"])
def test_timing_metrics_are_invariant_to_summary_collection_order(summary_shape):
    gestures = [
        Gesture(
            [p(0, 0, 0, 0), p(1, 0, 10, 0), p(2, 0, 20, 0)],
            "line",
            3,
        ),
        Gesture(
            [p(0, 0, 0, 0), p(1, 0, 100, 0), p(2, 0, 200, 0)],
            "line",
            3,
        ),
        Gesture(
            [p(0, 0, 0, 0), p(1, 0, 70, 0), p(2, 0, 140, 0)],
            "line",
            3,
        ),
    ]
    original_summary = SummaryGesture(
        gestures,
        PtAlignType.CHRONOLOGICAL,
        summary_shape,
    )
    reordered_summary = SummaryGesture(
        [gestures[2], gestures[0], gestures[1]],
        PtAlignType.CHRONOLOGICAL,
        summary_shape,
    )

    metric_functions = [
        RelAcc.timeError,
        RelAcc.timeVariability,
        RelAcc.velocityError,
        RelAcc.meanStrokeDuration,
    ]
    for gesture in gestures:
        for metric_function in metric_functions:
            assert metric_function(gesture, original_summary) == pytest.approx(
                metric_function(gesture, reordered_summary)
            )


def test_curvature_and_corner_slowdown_metrics():
    summaryPts = [
        p(0, 0, 0, 0),
        p(1, 0, 10, 0),
        p(1, 1, 30, 0),
        p(2, 1, 40, 0),
        p(3, 1, 50, 0),
    ]
    gesturePts = [
        p(0, 0, 0, 0),
        p(1, 0, 10, 0),
        p(1, 2, 20, 0),
        p(2, 2, 30, 0),
        p(3, 2, 40, 0),
    ]
    gesture = type("GestureObj", (), {"points": gesturePts})()
    summaryShape = _summary_for(summaryPts, gesturePts)

    assert RelAcc.curvatureArray(summaryPts)[0] == 0
    assert RelAcc.curvatureArray(summaryPts)[1] > 0
    assert RelAcc.curvature(gesture, summaryShape) > 0
    assert RelAcc.cornerSlowdownRatio(summaryPts) < 1
    assert RelAcc.cornerSlowdown(gesture, summaryShape) > 0

    straight = [p(0, 0, 0, 0), p(1, 0, 10, 0), p(2, 0, 20, 0)]
    still = [p(0, 0, 0, 0), p(0, 0, 0, 0), p(0, 0, 0, 0)]
    broken_stroke = [p(0, 0, 0, 0), p(1, 0, 10, 1), p(1, 1, 20, 0)]
    assert RelAcc.cornerSlowdownRatio(straight) == 1
    assert RelAcc.cornerSlowdownRatio(still) == 1
    assert RelAcc.curvatureArray(broken_stroke) == [0.0, 0.0, 0.0]
    assert RelAcc.curvatureArray(still) == [0.0, 0.0, 0.0]


def test_new_metric_defensive_paths(monkeypatch):
    assert RelAcc._finiteMean([float("nan")], fallback=2.0) == 2.0

    empty_summary = _summary_for([], [])
    gesture = type("GestureObj", (), {"points": []})()
    assert RelAcc.curvature(gesture, empty_summary) == 0

    monkeypatch.setattr(RelAcc, "curvatureArray", lambda points: [1.0, 0.0])
    monkeypatch.setattr(RelAcc, "speedArray", lambda points: [0.0, 1.0])
    assert RelAcc.cornerSlowdownRatio([p(0, 0, 0, 0)]) == 1


def test_power_law_frequency_and_stroke_metrics():
    summaryPts = [
        p(0, 0, 0, 0),
        p(1, 0, 10, 0),
        p(1, 1, 25, 0),
        p(2, 1, 35, 0),
        p(2, 3, 65, 1),
        p(4, 3, 95, 1),
    ]
    gesturePts = [
        p(0, 0, 0, 0),
        p(1, 0, 8, 0),
        p(1, 2, 18, 0),
        p(2, 2, 28, 0),
        p(1, 3, 38, 1),
        p(5, 3, 58, 1),
    ]
    gesture = type("GestureObj", (), {"points": gesturePts})()
    summaryShape = _summary_for(summaryPts, gesturePts)

    assert 0 <= RelAcc.twoThirdsPowerLawR2Value(summaryPts) <= 1
    assert RelAcc.twoThirdsPowerLawR2Value([p(0, 0, 0, 0), p(1, 0, 10, 0)]) == 0
    assert RelAcc.twoThirdsPowerLawR2Value(
        [
            p(0, 0, 0, 0),
            p(1, 0, 10, 0),
            p(1, 1, 20, 0),
            p(2, 1, 30, 0),
            p(2, 2, 40, 0),
        ]
    ) == 0
    assert RelAcc.twoThirdsPowerLawR2(gesture, summaryShape) >= 0

    smooth = [p(0, 0, 0, 0), p(1, 0, 1, 0), p(2, 0, 2, 0)]
    constant = [p(1, 1, 0, 0), p(1, 1, 1, 0), p(1, 1, 2, 0), p(1, 1, 3, 0)]
    jittery = [
        p(0, 0, 0, 0),
        p(1, 1, 1, 0),
        p(2, -1, 2, 0),
        p(3, 1, 3, 0),
        p(4, -1, 4, 0),
        p(5, 0, 5, 0),
    ]
    assert RelAcc.highFrequencyRatioValue(smooth) == 0
    assert RelAcc.highFrequencyRatioValue(constant) == 0
    assert RelAcc.highFrequencyRatioValue(jittery) > 0
    assert RelAcc.highFrequencyRatio(gesture, summaryShape) >= 0

    assert RelAcc.strokeGroups([]) == []
    assert len(RelAcc.strokeGroups(summaryPts)) == 2
    assert RelAcc.strokeLengths(summaryPts)[0] > 0
    assert RelAcc.strokeDurations(summaryPts) == [35, 30]
    assert RelAcc.strokeLengthStdValue([p(0, 0, 0, 0)]) == 0
    assert RelAcc.strokeLengthStd(gesture, summaryShape) > 0
    assert RelAcc.meanStrokeDurationValue([]) == 0
    assert RelAcc.meanStrokeDurationValue(summaryPts) == pytest.approx(32.5)
    assert RelAcc.meanStrokeDuration(gesture, summaryShape) > 0


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
    assert RelAcc.corner_slowdown(gesture, summaryShape) == RelAcc.cornerSlowdown(gesture, summaryShape)
    assert RelAcc.two_thirds_power_law_r2(gesture, summaryShape) == RelAcc.twoThirdsPowerLawR2(gesture, summaryShape)
    assert RelAcc.high_frequency_ratio(gesture, summaryShape) == RelAcc.highFrequencyRatio(gesture, summaryShape)
    assert RelAcc.stroke_length_std(gesture, summaryShape) == RelAcc.strokeLengthStd(gesture, summaryShape)
    assert RelAcc.mean_stroke_duration(gesture, summaryShape) == RelAcc.meanStrokeDuration(gesture, summaryShape)
    assert RelAcc.curvature_array(gesture.points) == RelAcc.curvatureArray(gesture.points)
    assert RelAcc.corner_slowdown_ratio(gesture.points) == RelAcc.cornerSlowdownRatio(gesture.points)
    assert RelAcc.two_thirds_power_law_r2_value(gesture.points) == RelAcc.twoThirdsPowerLawR2Value(gesture.points)
    assert RelAcc.high_frequency_ratio_value(gesture.points) == RelAcc.highFrequencyRatioValue(gesture.points)
    assert RelAcc.stroke_groups(gesture.points) == RelAcc.strokeGroups(gesture.points)
    assert RelAcc.stroke_lengths(gesture.points) == RelAcc.strokeLengths(gesture.points)
    assert RelAcc.stroke_durations(gesture.points) == RelAcc.strokeDurations(gesture.points)
    assert RelAcc.stroke_length_std_value(gesture.points) == RelAcc.strokeLengthStdValue(gesture.points)
    assert RelAcc.mean_stroke_duration_value(gesture.points) == RelAcc.meanStrokeDurationValue(gesture.points)
    assert RelAcc.stroke_order_error(gesture, summaryShape) == RelAcc.strokeOrderError(gesture, summaryShape)
    assert RelAcc.num_strokes(gesture) == RelAcc.numStrokes(gesture)
    assert RelAcc.dtw_distance(gesture, summaryShape) == RelAcc.dtwDistance(gesture, summaryShape)
    assert RelAcc.ldtw_distance(gesture, summaryShape) == RelAcc.ldtwDistance(gesture, summaryShape)
    assert RelAcc.ddtw_distance(gesture, summaryShape) == RelAcc.ddtwDistance(gesture, summaryShape)
    assert RelAcc.wdtw_distance(gesture, summaryShape) == RelAcc.wdtwDistance(gesture, summaryShape)
    assert RelAcc.wddtw_distance(gesture, summaryShape) == RelAcc.wddtwDistance(gesture, summaryShape)
