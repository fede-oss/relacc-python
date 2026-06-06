import json

from relacc.distribution_metrics import compute_distribution_metrics
from relacc.metrics import compute_metrics
from relacc.pipeline.one_vs_many import run_one_vs_many_comparison
from relacc.pipeline.pairwise import DIRECT_MODE, discover_pairs, run_pairwise_comparison
from relacc.utils.csv import CSVUtil
from relacc.gestures.gesture import Gesture
from relacc.gestures.summarygesture import SummaryGesture
import relacc_cli
import relacc_canvas_cli


SIMPLE_ROWS = [
    (0, 0, 0, 0),
    (0, 1, 0, 10),
    (0, 2, 0, 20),
]


def _shifted_rows(dx):
    return [(stroke, x + dx, y, time) for stroke, x, y, time in SIMPLE_ROWS]


def test_one_vs_many_pipeline_reads_generated_csv_files(write_gesture_csv):
    first = write_gesture_csv("samples/s01-zigzag-t01.csv", SIMPLE_ROWS)
    second = write_gesture_csv("samples/s02-zigzag-t02.csv", _shifted_rows(1))

    payload = run_one_vs_many_comparison(
        [str(first), str(second)],
        label="zigzag",
        rate=3,
        metric_names=["shapeError", "dtwDistance"],
    )

    assert payload["metadata"]["sampleCount"] == 2
    assert payload["metadata"]["label"] == "zigzag"
    assert [row["file"] for row in payload["samples"]] == ["s01-zigzag-t01", "s02-zigzag-t02"]
    assert set(payload["samples"][0]) >= {"shapeError", "dtwDistance"}


def test_pairwise_pipeline_matches_files_by_relative_name(write_gesture_csv, tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    write_gesture_csv("reference/user-a/s01-zigzag-t01.csv", SIMPLE_ROWS)
    write_gesture_csv("candidate/user-a/s01-zigzag-t01.csv", _shifted_rows(1))

    pairs, missing_candidate, missing_reference = discover_pairs(str(reference_dir), str(candidate_dir))
    payload = run_pairwise_comparison(
        str(reference_dir),
        str(candidate_dir),
        label="zigzag",
        rate=3,
        metric_names=["shapeError"],
    )

    assert [pair.key for pair in pairs] == ["user-a/s01-zigzag-t01"]
    assert missing_candidate == []
    assert missing_reference == []
    assert payload["metadata"]["comparisonMode"] == DIRECT_MODE
    assert payload["metadata"]["pairCount"] == 1
    assert payload["pairs"][0]["shapeError"] >= 0


def test_metrics_registry_computes_only_the_requested_metric(write_gesture_csv):
    path = write_gesture_csv("samples/s01-zigzag-t01.csv", SIMPLE_ROWS)
    parsed = {}
    CSVUtil.readGesture(path, lambda points: parsed.setdefault("points", points))

    gesture = Gesture(parsed["points"], "zigzag", samplingRate=3)
    summary = SummaryGesture([gesture], summaryShape="centroid")

    assert compute_metrics(gesture, summary, metric_names=["shapeError"]) == {"shapeError": 0.0}
    assert compute_distribution_metrics([1, 2, 3], [2, 3, 4], round_precision=3)["wassersteinDistance"] == 1.0


def test_relacc_cli_outputs_json_from_generated_csvs(write_gesture_csv, capsys):
    first = write_gesture_csv("samples/s01-zigzag-t01.csv", SIMPLE_ROWS)
    second = write_gesture_csv("samples/s02-zigzag-t02.csv", _shifted_rows(1))

    assert relacc_cli.main(["--label", "zigzag", "--rate", "3", "--format", "json", str(first), str(second)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["label"] == "zigzag"
    assert len(payload["results"]) == 2


def test_canvas_cli_writes_an_image_from_generated_csvs(write_gesture_csv, tmp_path):
    first = write_gesture_csv("samples/s01-zigzag-t01.csv", SIMPLE_ROWS)
    second = write_gesture_csv("samples/s02-zigzag-t02.csv", _shifted_rows(1))
    image = tmp_path / "summary.png"

    assert relacc_canvas_cli.main(
        ["--label", "zigzag", "--rate", "3", "--output", str(image), "--format", "png", str(first), str(second)]
    ) == 0

    assert image.read_bytes().startswith(b"\x89PNG")
