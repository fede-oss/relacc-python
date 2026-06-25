import json
import math
from pathlib import Path

import pytest

from relacc.metrics import METRIC_NAMES
from relacc.pipeline import one_vs_many as OneVsMany


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _sample_rows(offset=0, stroke_1=0, stroke_2=1):
    return [
        "stroke_id x y time is_writing",
        f"{stroke_1} {10+offset} 20 0 1",
        f"{stroke_1} {12+offset} 22 10 1",
        f"{stroke_2} {14+offset} 24 20 1",
        f"{stroke_2} {16+offset} 25 30 1",
    ]


def _timed_line_rows(times):
    return [
        "stroke_id x y time is_writing",
        f"0 0 0 {times[0]} 1",
        f"0 1 0 {times[1]} 1",
        f"0 2 0 {times[2]} 1",
    ]


def _stroke_rows(strokes):
    rows = ["stroke_id x y time is_writing"]
    timestamp = 0
    for stroke_id, xs in enumerate(strokes):
        for x in xs:
            rows.append(f"{stroke_id} {x} 0 {timestamp} 1")
            timestamp += 1
    return rows


def _normalize_ordered_payload(payload):
    return {
        "metadata": payload["metadata"],
        "results": payload["results"],
        "samples": sorted(payload["samples"], key=lambda row: row["file"]),
        "rawMetricOutputs": sorted(
            payload["rawMetricOutputs"],
            key=lambda row: (row["sampleKey"], row["metric"]),
        ),
    }


def test_run_one_vs_many_comparison_outputs_samples_and_stats(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    f2 = tmp_path / "s1-arrow-t2.csv"
    _write_csv(f1, _sample_rows(0))
    _write_csv(f2, _sample_rows(1))

    payload = OneVsMany.run_one_vs_many_comparison(
        [str(f1), str(f2)],
        rate=5,
        summary_shape="Centroid",
        stats=True,
        dtw_window=2,
    )

    assert payload["metadata"]["comparisonMode"] == "one-vs-many"
    assert payload["metadata"]["sampleCount"] == 2
    assert payload["metadata"]["label"] == "arrow"
    assert payload["metadata"]["summary"] == "centroid"
    assert payload["metadata"]["rate"] == 5
    assert payload["metadata"]["dtwWindow"] == 2
    assert payload["metadata"]["alignment"] == 0
    assert payload["metadata"]["alignmentName"] == "chronological"
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["file"] == "s1-arrow-t1"
    assert set(payload["results"].keys()) == set(METRIC_NAMES)
    assert len(payload["rawMetricOutputs"]) == 2 * len(METRIC_NAMES)
    assert payload["rawMetricOutputs"][0]["recordType"] == "rawMetricOutput"
    assert payload["rawMetricOutputs"][0]["comparisonMode"] == "one-vs-many"
    assert payload["rawMetricOutputs"][0]["alignmentName"] == "chronological"


@pytest.mark.parametrize("summary_shape", ["centroid", "medoid"])
def test_run_one_vs_many_timing_metrics_are_invariant_to_file_order(tmp_path, summary_shape):
    f1 = tmp_path / "s1-line-t1.csv"
    f2 = tmp_path / "s1-line-t2.csv"
    f3 = tmp_path / "s1-line-t3.csv"
    _write_csv(f1, _timed_line_rows([0, 10, 20]))
    _write_csv(f2, _timed_line_rows([0, 100, 200]))
    _write_csv(f3, _timed_line_rows([0, 70, 140]))

    metric_names = [
        "timeError",
        "timeVariability",
        "velocityError",
        "meanStrokeDuration",
    ]
    original = OneVsMany.run_one_vs_many_comparison(
        [str(f1), str(f2), str(f3)],
        label="line",
        rate=3,
        summary_shape=summary_shape,
        stats=True,
        metric_names=metric_names,
        round_precision=None,
    )
    reordered = OneVsMany.run_one_vs_many_comparison(
        [str(f3), str(f1), str(f2)],
        label="line",
        rate=3,
        summary_shape=summary_shape,
        stats=True,
        metric_names=metric_names,
        round_precision=None,
    )

    assert _normalize_ordered_payload(original) == _normalize_ordered_payload(reordered)


def test_run_one_vs_many_popular_summary_filters_to_exact_modal_count(tmp_path):
    lower = tmp_path / "s1-modal-t1.csv"
    modal_a = tmp_path / "s1-modal-t2.csv"
    modal_b = tmp_path / "s1-modal-t3.csv"
    higher = tmp_path / "s1-modal-t4.csv"
    _write_csv(lower, _stroke_rows([[-150, -50, 50, 150]]))
    _write_csv(modal_a, _stroke_rows([[-2, -1], [1, 2]]))
    _write_csv(modal_b, _stroke_rows([[-4, -3], [3, 4]]))
    _write_csv(higher, _stroke_rows([[-300, -200], [0], [200]]))

    payload = OneVsMany.run_one_vs_many_comparison(
        [str(lower), str(modal_a), str(modal_b), str(higher)],
        label="modal",
        rate=4,
        summary_shape="centroid",
        popular_shape=True,
        stats=False,
        metric_names=["shapeError"],
        round_precision=None,
    )

    modal_a_sample = next(
        sample
        for sample in payload["samples"]
        if sample["file"] == "s1-modal-t2"
    )
    assert modal_a_sample["shapeError"] == 1.0


def test_run_one_vs_many_stats_aggregate_raw_values_before_rounding(monkeypatch, tmp_path):
    files = [
        str(tmp_path / "s1-arrow-t1.csv"),
        str(tmp_path / "s1-arrow-t2.csv"),
        str(tmp_path / "s1-arrow-t3.csv"),
    ]

    class FakeGesture:
        def __init__(self, points, label, rate):
            self.points = points
            self.label = label
            self.rate = rate

    class FakeSummaryGesture:
        def __init__(self, gestures, alignment_type, summary_shape, popular_shape):
            self.gestures = gestures

    metric_results = iter(
        [
            {"shapeError": 1.24},
            {"shapeError": 1.25},
            {"shapeError": 1.25},
        ]
    )

    def fake_compute_metrics(*args, round_precision, **kwargs):
        assert round_precision is None
        return next(metric_results)

    monkeypatch.setattr(OneVsMany, "read_gesture_points", lambda file_path: [object()])
    monkeypatch.setattr(OneVsMany, "Gesture", FakeGesture)
    monkeypatch.setattr(OneVsMany, "SummaryGesture", FakeSummaryGesture)
    monkeypatch.setattr(OneVsMany, "compute_metrics", fake_compute_metrics)

    payload = OneVsMany.run_one_vs_many_comparison(
        files,
        label="arrow",
        rate=1,
        stats=True,
        round_precision=1,
        metric_names=["shapeError"],
    )

    assert [sample["shapeError"] for sample in payload["samples"]] == [1.2, 1.3, 1.3]
    assert payload["results"]["shapeError"]["mean"] == 1.2
    assert [row["value"] for row in payload["rawMetricOutputs"]] == [1.24, 1.25, 1.25]


def test_run_one_vs_many_rejects_bad_rate_and_bad_inferred_label(tmp_path):
    f1 = tmp_path / "arrow.csv"
    _write_csv(f1, _sample_rows(0))

    with pytest.raises(ValueError, match="Cannot derive gesture label"):
        OneVsMany.run_one_vs_many_comparison([str(f1)])

    with pytest.raises(ValueError, match="Sampling rate must be >= 1"):
        OneVsMany.run_one_vs_many_comparison([str(f1)], label="arrow", rate=0)


def test_run_one_vs_many_rejects_empty_input_and_conflicting_dtw_options(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    with pytest.raises(ValueError, match="Please provide some gesture files"):
        OneVsMany.run_one_vs_many_comparison([])

    with pytest.raises(ValueError, match="cannot be combined"):
        OneVsMany.run_one_vs_many_comparison(
            [str(f1)],
            dtw_window=2,
            exact_dtw=True,
        )


def test_one_vs_many_summary_stats_handles_empty_and_non_finite_values():
    empty_stats = OneVsMany.summary_stats([])
    assert empty_stats == {
        "mean": 0,
        "mdn": 0,
        "sd": 0,
        "min": 0,
        "max": 0,
        "n": 0,
    }

    non_finite_stats = OneVsMany.summary_stats([1.0, float("nan")])
    assert math.isnan(non_finite_stats["mean"])
    assert math.isnan(non_finite_stats["mdn"])
    assert math.isnan(non_finite_stats["sd"])
    assert math.isnan(non_finite_stats["min"])
    assert math.isnan(non_finite_stats["max"])
    assert non_finite_stats["n"] == 2


def test_one_vs_many_formatters_are_json_safe_and_csv_consistent(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=False)
    json_payload = json.loads(OneVsMany.format_one_vs_many_result(payload, "json"))
    assert json_payload["metadata"]["comparisonMode"] == "one-vs-many"
    assert len(json_payload["results"]) == 1

    csv_output = OneVsMany.format_one_vs_many_result(payload, "csv")
    assert csv_output.splitlines()[0].startswith("file,inputFile,label,rate")
    assert ",alignment,alignmentName,summary," in csv_output.splitlines()[0]
    assert ",0,chronological,," in csv_output.splitlines()[1]

    stats_payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=True)
    stats_csv = OneVsMany.format_one_vs_many_result(stats_payload, "csv")
    assert stats_csv.splitlines()[0] == "measure,n,mean,mdn,sd,min,max"

    xml_output = OneVsMany.format_one_vs_many_result(stats_payload, "xml")
    assert xml_output.startswith("<?xml")


def test_one_vs_many_text_xml_and_legacy_metadata_formatters(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=False)
    legacy_args = OneVsMany.legacy_args_from_metadata(
        payload,
        output="samples.txt",
        fmt="text",
    )
    assert legacy_args["output"] == "samples.txt"
    assert legacy_args["format"] == "text"

    json_payload = json.loads(
        OneVsMany.format_one_vs_many_json(payload, legacy_args=legacy_args)
    )
    assert json_payload["metadata"]["args"]["format"] == "text"

    samples_text = OneVsMany.format_one_vs_many_result(payload, "text")
    assert samples_text.splitlines()[0].startswith("file shapeError")

    sample_xml = OneVsMany.format_one_vs_many_result(
        payload,
        "xml",
        legacy_args=legacy_args,
    )
    assert "<args" in sample_xml
    assert "<sample " in sample_xml
    assert 'alignment="0" alignmentName="chronological"' in sample_xml

    non_finite_payload = {
        **payload,
        "metadata": {**payload["metadata"], "rate": float("inf")},
    }
    non_finite_xml = OneVsMany.format_one_vs_many_result(non_finite_payload, "xml")
    assert 'rate=""' in non_finite_xml

    stats_payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=True)
    stats_text = OneVsMany.format_one_vs_many_result(stats_payload, "txt")
    assert stats_text.splitlines()[0] == "measure n mean mdn sd min max"

    with pytest.raises(ValueError, match="Invalid output format"):
        OneVsMany.format_one_vs_many_result(payload, "yaml")
