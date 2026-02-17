import pytest

from relacc.gestures.gesture import Gesture


def test_gesture_empty_points_raises():
    with pytest.raises(ValueError, match="Gesture points cannot be empty\\."):
        Gesture([], "label")
