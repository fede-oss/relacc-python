import pytest
from types import SimpleNamespace

from relacc.geom.point import Point
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture


def _collection():
    pt1 = Point(1, 2, 100, 0)
    pt2 = Point(3, 4, 200, 0)
    pt3 = Point(5, 6, 300, 0)
    s = [pt1, pt2, pt3]
    return s, [Gesture(s, "my label", 3), Gesture(s, "my label", 3)]


def test_summarygesture_rejects_different_names():
    s, _ = _collection()
    with pytest.raises(ValueError, match="Gesture names cannot be different\\."):
        SummaryGesture([Gesture(s, "my label"), Gesture(s, "different label")])


def test_summarygesture_default_alignment():
    _, collection = _collection()
    summary = SummaryGesture(collection)
    assert summary.alignmentType == PtAlignType.CHRONOLOGICAL


def test_summarygesture_custom_alignment():
    _, collection = _collection()
    summary = SummaryGesture(collection, PtAlignType.CLOUD_MATCH)
    assert summary.alignmentType == PtAlignType.CLOUD_MATCH


def test_summarygesture_centroid_mode():
    _, collection = _collection()
    summary = SummaryGesture(collection, PtAlignType.CHRONOLOGICAL, "centroid")
    assert summary.originalPoints == [Point(-2, -2, 100, 0), Point(0, 0, 200, 0), Point(2, 2, 300, 0)]


def test_summarygesture_kcentroid_mode():
    s, collection = _collection()
    summary = SummaryGesture(collection, PtAlignType.CHRONOLOGICAL, "kcentroid")
    assert summary.originalPoints == s


def test_summarygesture_medoid_mode():
    _, collection = _collection()
    summary = SummaryGesture(collection, PtAlignType.CHRONOLOGICAL, "medoid")
    assert summary.originalPoints == [Point(-2, -2, 100, 0), Point(0, 0, 200, 0), Point(2, 2, 300, 0)]


def test_summarygesture_kmedoid_mode():
    s, collection = _collection()
    summary = SummaryGesture(collection, PtAlignType.CHRONOLOGICAL, "kmedoid")
    assert summary.originalPoints == s


def test_summarygesture_knn_uses_effective_alignment(monkeypatch):
    _, collection = _collection()
    calls = []
    original = SummaryGesture.alignGesture

    def spy(self, gesture, alignmentType=None):
        calls.append(alignmentType)
        return original(self, gesture, alignmentType)

    monkeypatch.setattr(SummaryGesture, "alignGesture", spy)
    SummaryGesture(collection, None, "kcentroid")
    assert None not in calls


def test_compute_summary_shapes_divides_by_included_gestures():
    class DummySummary:
        def __init__(self):
            self.refGesture = SimpleNamespace(samplingRate=2)
            self.alignmentType = PtAlignType.CHRONOLOGICAL

        @staticmethod
        def alignGesture(gesture, alignmentType=None):
            return gesture.aligned

    included = SimpleNamespace(
        points=[
            Point(0, 0, 0, 0),
            Point(1, 0, 1, 1),
            Point(2, 0, 2, 1),
            Point(3, 0, 3, 1),
        ],
        aligned=[Point(10, 0, 0, 0), Point(20, 0, 0, 0)],
    )
    filtered = SimpleNamespace(
        points=[
            Point(0, 0, 0, 0),
            Point(1, 0, 1, 1),
            Point(2, 0, 2, 2),
            Point(3, 0, 3, 2),
        ],
        aligned=[Point(30, 0, 0, 0), Point(40, 0, 0, 0)],
    )

    shapes = SummaryGesture.computeSummaryShapes(DummySummary(), [included, filtered], popularStrokeNum=1)
    centroid = shapes["centroid"]
    assert [pt.X for pt in centroid] == [10, 20]


def test_compute_summary_shapes_rejects_empty_filtered_set():
    class DummySummary:
        def __init__(self):
            self.refGesture = SimpleNamespace(samplingRate=2)
            self.alignmentType = PtAlignType.CHRONOLOGICAL

        @staticmethod
        def alignGesture(gesture, alignmentType=None):
            return gesture.aligned

    filtered = SimpleNamespace(
        points=[
            Point(0, 0, 0, 0),
            Point(1, 0, 1, 1),
            Point(2, 0, 2, 2),
            Point(3, 0, 3, 2),
        ],
        aligned=[Point(30, 0, 0, 0), Point(40, 0, 0, 0)],
    )

    with pytest.raises(ValueError, match="No gestures available to compute summary shapes\\."):
        SummaryGesture.computeSummaryShapes(DummySummary(), [filtered], popularStrokeNum=1)
