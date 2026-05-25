import json
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
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["file"] == "s1-arrow-t1"
    assert set(payload["results"].keys()) == set(METRIC_NAMES)


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


def test_run_one_vs_many_rejects_bad_rate_and_bad_inferred_label(tmp_path):
    f1 = tmp_path / "arrow.csv"
    _write_csv(f1, _sample_rows(0))

    with pytest.raises(ValueError, match="Cannot derive gesture label"):
        OneVsMany.run_one_vs_many_comparison([str(f1)])

    with pytest.raises(ValueError, match="Sampling rate must be >= 1"):
        OneVsMany.run_one_vs_many_comparison([str(f1)], label="arrow", rate=0)


def test_one_vs_many_formatters_are_json_safe_and_csv_consistent(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=False)
    json_payload = json.loads(OneVsMany.format_one_vs_many_result(payload, "json"))
    assert json_payload["metadata"]["comparisonMode"] == "one-vs-many"
    assert len(json_payload["results"]) == 1

    csv_output = OneVsMany.format_one_vs_many_result(payload, "csv")
    assert csv_output.splitlines()[0].startswith("file,inputFile,label,rate")

    stats_payload = OneVsMany.run_one_vs_many_comparison([str(f1)], stats=True)
    stats_csv = OneVsMany.format_one_vs_many_result(stats_payload, "csv")
    assert stats_csv.splitlines()[0] == "measure,n,mean,mdn,sd,min,max"

    xml_output = OneVsMany.format_one_vs_many_result(stats_payload, "xml")
    assert xml_output.startswith("<?xml")
