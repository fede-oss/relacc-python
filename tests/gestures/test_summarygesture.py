import pytest

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
