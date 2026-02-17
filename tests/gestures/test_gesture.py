import pytest

from relacc.geom.point import Point
from relacc.gestures.gesture import Gesture


def _set():
    pt1 = Point(1, 2, 100, 0)
    pt2 = Point(3, 4, 200, 0)
    pt3 = Point(5, 6, 300, 0)
    return [pt1, pt2, pt3]


def test_gesture_name_required():
    with pytest.raises(ValueError, match="Gesture name cannot be empty\\."):
        Gesture(_set(), None)


def test_gesture_interpolates_default_sampling_rate():
    gesture = Gesture(_set(), "my label")
    assert len(gesture.points) == gesture.samplingRate
    assert len(gesture.originalPoints) != gesture.samplingRate
    assert len(gesture.originalPoints) == len(_set())
