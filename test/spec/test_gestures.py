import pytest

from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture


def test_gesture_requires_a_name_and_preprocesses_to_the_requested_rate(p):
    raw_points = [p(0, 0, 0), p(2, 0, 10), p(4, 0, 20)]

    with pytest.raises(ValueError, match="Gesture name cannot be empty"):
        Gesture(raw_points, "")

    gesture = Gesture(raw_points, "line", samplingRate=3)

    assert gesture.originalPoints == raw_points
    assert len(gesture.points) == 3
    # Preprocessing translates the gesture to its centroid, so the middle point sits at the origin.
    assert [(pt.X, pt.Y) for pt in gesture.points] == pytest.approx([(-2, 0), (0, 0), (2, 0)])


def test_summary_gesture_centroid_averages_aligned_points(p):
    first = Gesture([p(0, 0, 0), p(2, 0, 10), p(4, 0, 20)], "line", samplingRate=3)
    second = Gesture([p(0, 2, 0), p(2, 2, 10), p(4, 2, 20)], "line", samplingRate=3)

    summary = SummaryGesture([first, second], PtAlignType.CHRONOLOGICAL, summaryShape="centroid")

    assert summary.name == "line"
    assert len(summary.getPoints()) == 3
    # Both gestures are centered before the summary is built; their y offsets cancel out.
    assert [(pt.X, pt.Y) for pt in summary.getPoints()] == pytest.approx([(-2, 0), (0, 0), (2, 0)])


def test_summary_gesture_rejects_mixed_labels(p):
    line = Gesture([p(0, 0), p(1, 0)], "line", samplingRate=2)
    arc = Gesture([p(0, 0), p(1, 1)], "arc", samplingRate=2)

    with pytest.raises(ValueError, match="Gesture names cannot be different"):
        SummaryGesture([line, arc])
