from relacc.geom.point import Point
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture, numericSort


def p(x, y, t, sid):
    return Point(x, y, t, sid)


def _gestures():
    setA = [p(0, 0, 0, 0), p(1, 0, 1, 0), p(2, 0, 2, 1), p(3, 0, 3, 1)]
    setB = [p(0, 0, 0, 0), p(1, 1, 1, 1), p(2, 0, 2, 0), p(3, 1, 3, 1)]
    return Gesture(setA, "shape", 4), Gesture(setB, "shape", 4)


def test_points_for_alignment_export():
    gestureA, _ = _gestures()
    aligned = SummaryGesture.getPointsForAlignment(gestureA)
    assert len(aligned) == len(gestureA.points)


def test_cloud_alignment_sets_stroke_id_to_zero():
    gestureA, gestureB = _gestures()
    summary = SummaryGesture([gestureA, gestureB], PtAlignType.CLOUD_MATCH)
    aligned = summary.alignGesture(gestureB, PtAlignType.CLOUD_MATCH)

    assert len(aligned) == summary.refGesture.samplingRate
    for pt in aligned:
        assert pt.StrokeID == 0


def test_popular_stroke_summary():
    gestureA, gestureB = _gestures()
    summary = SummaryGesture([gestureA, gestureB, gestureA], PtAlignType.CHRONOLOGICAL, "centroid", True)
    assert len(summary.points) == gestureA.samplingRate
    assert len(summary.getPoints()) == gestureA.samplingRate


def _stroke_gesture(strokes, name="modal shape"):
    points = []
    timestamp = 0
    for stroke_id, xs in enumerate(strokes):
        for x in xs:
            points.append(p(x, 0, timestamp, stroke_id))
            timestamp += 1
    return Gesture(points, name, 4)


def _popular_collection():
    lower = _stroke_gesture([[-150, -50, 50, 150]])
    modal_a = _stroke_gesture([[-2, -1], [1, 2]])
    modal_b = _stroke_gesture([[-4, -3], [3, 4]])
    higher = _stroke_gesture([[-300, -200], [0], [200]])
    return lower, modal_a, modal_b, higher


def _x_values(points):
    return [point.X for point in points]


def test_popular_centroid_uses_only_exact_modal_stroke_count():
    lower, modal_a, modal_b, higher = _popular_collection()

    summary = SummaryGesture(
        [lower, modal_a, modal_b, higher],
        PtAlignType.CHRONOLOGICAL,
        "centroid",
        True,
    )

    assert _x_values(summary.originalPoints) == [-3, -2, 2, 3]


def test_popular_medoid_uses_only_exact_modal_stroke_count():
    lower, modal_a, modal_b, higher = _popular_collection()

    summary = SummaryGesture(
        [higher, modal_b, lower, modal_a],
        PtAlignType.CHRONOLOGICAL,
        "medoid",
        True,
    )

    assert _x_values(summary.originalPoints) == [-3, -2, 2, 3]


def test_popular_stroke_tie_uses_numeric_minimum_across_permutations():
    one_stroke = _stroke_gesture([[-10, 0, 10, 20]])
    two_stroke = _stroke_gesture([[-200, -100], [100, 200]])
    three_stroke = _stroke_gesture([[-300, -200], [0], [200]])
    four_stroke = _stroke_gesture([[-400], [-100], [100], [400]])
    permutations = [
        [one_stroke, two_stroke, three_stroke, four_stroke],
        [two_stroke, three_stroke, four_stroke, one_stroke],
        [four_stroke, three_stroke, two_stroke, one_stroke],
    ]

    summaries = [
        SummaryGesture(
            permutation,
            PtAlignType.CHRONOLOGICAL,
            "centroid",
            True,
        )
        for permutation in permutations
    ]

    assert [_x_values(summary.originalPoints) for summary in summaries] == [
        [-15, -5, 5, 15],
        [-15, -5, 5, 15],
        [-15, -5, 5, 15],
    ]


def test_numeric_sort_helper():
    assert numericSort(1, 2) == -1
