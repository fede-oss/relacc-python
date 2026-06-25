from types import SimpleNamespace

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


def _timed_line(times, stroke_ids=(0, 0, 0), name="timed line"):
    points = [
        Point(0, 0, times[0], stroke_ids[0]),
        Point(1, 0, times[1], stroke_ids[1]),
        Point(2, 0, times[2], stroke_ids[2]),
    ]
    return Gesture(points, name, 3)


def _permutation_fixture():
    return [
        _timed_line([0, 10, 20], [6, 6, 6]),
        _timed_line([0, 100, 200], [2, 2, 2]),
        _timed_line([0, 70, 140], [4, 4, 4]),
    ]


def _point_fields(points):
    return [(point.X, point.Y, point.T, point.StrokeID) for point in points]


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


def test_public_alignment_values_are_canonical():
    assert PtAlignType.CHRONOLOGICAL == 0
    assert PtAlignType.CLOUD_MATCH == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ("0", 0),
        ("chronological", 0),
        (1, 1),
        ("1", 1),
        ("cloud", 1),
        ("cloud-match", 1),
    ],
)
def test_alignment_normalization(value, expected):
    assert PtAlignType.normalize(value) == expected


@pytest.mark.parametrize("value", [-1, 2, 0.0, 1.0, "other", True, False, None])
def test_alignment_normalization_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="alignment"):
        PtAlignType.normalize(value)


def test_aligngesture_uses_stored_alignment_and_allows_override(monkeypatch):
    _, collection = _collection()
    summary = SummaryGesture(collection, "cloud")
    calls = []

    def match(reference, points):
        calls.append((reference, points))
        return list(reversed(range(len(points))))

    monkeypatch.setattr("relacc.gestures.summarygesture.PDollarAlt.match", match)
    implicit = summary.alignGesture(collection[1])
    explicit_chronological = summary.alignGesture(collection[1], "chronological")
    explicit_cloud = summary.alignGesture(collection[1], 1)

    assert len(calls) == 2
    assert [point.X for point in implicit] == [2, 0, -2]
    assert [point.X for point in explicit_chronological] == [-2, 0, 2]
    assert [point.X for point in explicit_cloud] == [2, 0, -2]


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


@pytest.mark.parametrize("summary_shape", ["centroid", "medoid"])
def test_synthetic_summary_points_are_permutation_invariant(summary_shape):
    gestures = _permutation_fixture()
    permutations = [
        gestures,
        [gestures[1], gestures[2], gestures[0]],
        [gestures[2], gestures[0], gestures[1]],
    ]

    summaries = [
        SummaryGesture(permutation, PtAlignType.CHRONOLOGICAL, summary_shape)
        for permutation in permutations
    ]
    point_fields = [_point_fields(summary.originalPoints) for summary in summaries]

    assert point_fields[0] == point_fields[1] == point_fields[2]
    assert [point.T for point in summaries[0].originalPoints] == [0, 70, 140]
    assert [point.StrokeID for point in summaries[0].originalPoints] == [2, 2, 2]


def test_medoid_uses_coordinate_wise_median_timestamp_for_even_collection_size():
    gestures = [
        _timed_line([0, 10, 20]),
        _timed_line([0, 30, 60]),
        _timed_line([0, 50, 100]),
        _timed_line([0, 70, 140]),
    ]

    summary = SummaryGesture(gestures, PtAlignType.CHRONOLOGICAL, "medoid")

    assert [point.T for point in summary.originalPoints] == [0, 40, 80]


def test_synthetic_summary_stroke_id_uses_mode_with_numeric_tie_break():
    gestures = [
        _timed_line([0, 10, 20], [3, 5, 8]),
        _timed_line([0, 20, 40], [3, 5, 6]),
        _timed_line([0, 30, 60], [4, 7, 6]),
        _timed_line([0, 40, 80], [4, 9, 8]),
    ]

    summary = SummaryGesture(gestures, PtAlignType.CHRONOLOGICAL, "centroid")

    assert [point.StrokeID for point in summary.originalPoints] == [3, 5, 6]


@pytest.mark.parametrize("summary_shape", ["kcentroid", "kmedoid"])
def test_k_summary_modes_retain_real_gesture_timing(summary_shape):
    gestures = [
        _timed_line([0, 10, 20]),
        _timed_line([0, 100, 200]),
    ]

    summary = SummaryGesture(gestures, PtAlignType.CHRONOLOGICAL, summary_shape)

    assert [point.T for point in summary.originalPoints] == [0, 10, 20]


def test_compute_summary_shapes_rejects_non_finite_aggregate_timestamp():
    class DummySummary:
        def __init__(self):
            self.refGesture = SimpleNamespace(samplingRate=1)
            self.alignmentType = PtAlignType.CHRONOLOGICAL

        @staticmethod
        def alignGesture(gesture, alignmentType=None):
            return gesture.aligned

    gestures = [
        SimpleNamespace(points=[Point(0, 0, 0, 0)], aligned=[Point(1, 1, 0, 0)]),
        SimpleNamespace(
            points=[Point(0, 0, 0, 0)],
            aligned=[Point(1, 1, float("inf"), 0)],
        ),
    ]

    with pytest.raises(ValueError, match="aggregate"):
        SummaryGesture.computeSummaryShapes(DummySummary(), gestures, popularStrokeNum=0)


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

    shapes = SummaryGesture.computeSummaryShapes(DummySummary(), [included, filtered], popularStrokeNum=2)
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
