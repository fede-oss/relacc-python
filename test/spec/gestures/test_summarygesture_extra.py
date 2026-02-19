from relacc.geom.point import Point
from relacc.geom.pointset import PointSet
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


def test_compute_summary_shapes_skips_above_popular_threshold(monkeypatch):
    gestureA, gestureB = _gestures()

    calls = {"i": 0}

    def _count_strokes(_):
        calls["i"] += 1
        return 1 if calls["i"] == 1 else 2

    monkeypatch.setattr(PointSet, "countStrokes", staticmethod(_count_strokes))

    class SelfObj:
        refGesture = gestureA
        alignmentType = PtAlignType.CHRONOLOGICAL

        def __init__(self):
            self.calls = 0

        def alignGesture(self, gesture, _):
            self.calls += 1
            return gesture.points

    self_obj = SelfObj()
    shapes = SummaryGesture.computeSummaryShapes(self_obj, [gestureA, gestureB], 1)

    assert self_obj.calls == 1
    assert len(shapes["centroid"]) == gestureA.samplingRate
    assert len(shapes["medoid"]) == gestureA.samplingRate


def test_numeric_sort_helper():
    assert numericSort(1, 2) == -1
