import json

import pytest

from relacc.geom.point import Point
from relacc.pipeline import _common as Common


def test_read_gesture_points_accepts_json_and_rejects_empty_or_unknown(tmp_path):
    json_file = tmp_path / "gesture.json"
    json_file.write_text(
        json.dumps({"strokes": [[[1, 2, 0], [3, 4, 5]]]}),
        encoding="utf-8",
    )

    points = Common.read_gesture_points(str(json_file))
    assert len(points) == 2
    assert points[0].X == 1
    assert points[1].T == 5

    empty_json = tmp_path / "empty.json"
    empty_json.write_text(json.dumps({"strokes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="No points parsed"):
        Common.read_gesture_points(str(empty_json))

    unknown = tmp_path / "gesture.txt"
    unknown.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid input file format"):
        Common.read_gesture_points(str(unknown))


def test_output_format_uses_output_extension_then_requested_or_default():
    assert Common.output_format("/tmp/report.csv", None) == "csv"
    assert Common.output_format(None, "xml") == "xml"
    assert Common.output_format(None, None, default="text") == "text"


def _stroke_runs(count):
    return [Point(index, 0, index, index) for index in range(count)]


@pytest.mark.parametrize(
    ("stroke_count", "expected_rate"),
    [(1, 24), (4, 32), (20, 160), (32, 256), (33, 256)],
)
def test_sampling_rate_for_sets_uses_bounded_points_per_stroke_policy(
    stroke_count, expected_rate
):
    assert Common.sampling_rate_for_sets([_stroke_runs(stroke_count)], None) == expected_rate


def test_sampling_rate_for_sets_rejects_automatic_gestures_over_cap():
    with pytest.raises(ValueError, match="more than 256 stroke runs"):
        Common.sampling_rate_for_sets([_stroke_runs(257)], None)


def test_sampling_rate_for_sets_validates_explicit_rate_against_strokes():
    points = _stroke_runs(4)
    with pytest.raises(ValueError, match="at least the maximum stroke count"):
        Common.sampling_rate_for_sets([points], 3)
    assert Common.sampling_rate_for_sets([points], 300) == 300


def test_summary_sampling_rate_keeps_reference_auto_rate_when_candidate_is_below_it():
    reference_points = [_stroke_runs(4)]
    candidate_points = [_stroke_runs(5)]

    assert Common.summary_sampling_rate(reference_points, candidate_points, None) == 32


def test_summary_sampling_rate_validates_explicit_rate_against_candidates():
    reference_points = [_stroke_runs(2)]
    candidate_points = [_stroke_runs(5)]

    with pytest.raises(ValueError, match="at least the maximum stroke count"):
        Common.summary_sampling_rate(reference_points, candidate_points, 4)


def test_summary_sampling_rate_rejects_automatic_candidates_over_cap():
    reference_points = [_stroke_runs(2)]
    candidate_points = [_stroke_runs(257)]

    with pytest.raises(ValueError, match="more than 256 stroke runs"):
        Common.summary_sampling_rate(reference_points, candidate_points, None)
