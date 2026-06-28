from pathlib import Path

from relacc.pipeline import report_compare as ReportCompare
from relacc.pipeline.reporting import ReportingEntry


def _entry(key, points):
    return ReportingEntry(
        key=key,
        path=f"/tmp/{key}",
        dataset_key="dataset-a",
        class_key="arrow",
        points=points,
    )


def _metric_dict(value):
    return {"shapeError": value}


def test_compare_class_uses_summary_sampling_rate_with_candidates(monkeypatch):
    reference_entries = (_entry("r1.csv", ["ref-1"]), _entry("r2.csv", ["ref-2"]))
    candidate_entries = (_entry("c1.csv", ["cand-1"]),)
    captured = {}

    def fake_summary_sampling_rate(reference_points, candidate_points, rate):
        captured["reference_points"] = list(reference_points)
        captured["candidate_points"] = list(candidate_points)
        captured["rate"] = rate
        return 31

    monkeypatch.setattr(ReportCompare, "summary_sampling_rate", fake_summary_sampling_rate)
    monkeypatch.setattr(ReportCompare, "Gesture", lambda *args, **kwargs: object())
    monkeypatch.setattr(ReportCompare, "SummaryGesture", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ReportCompare,
        "compute_metrics",
        lambda *args, **kwargs: _metric_dict(1.0),
    )

    rows = ReportCompare._compare_class(
        reference_entries,
        candidate_entries,
        (),
        run_id="run-1",
        source_name="generated",
        dataset_name="dataset-a",
        variant="syntTO",
        class_key="arrow",
        reference_input=Path("/tmp/reference"),
        rate=None,
        alignment=0,
        summary_shape="medoid",
        popular=False,
        round_precision=3,
        metric_names=("shapeError",),
        dtw_window=None,
        exact_dtw=False,
    )[0]

    assert captured == {
        "reference_points": [["ref-1"], ["ref-2"]],
        "candidate_points": [["cand-1"]],
        "rate": None,
    }
    assert rows[0]["rate"] == 31


def test_direct_distribution_rate_selection_includes_candidates(monkeypatch):
    reference_entries = (_entry("r1.csv", ["ref-1"]), _entry("r2.csv", ["ref-2"]))
    candidate_entries = (_entry("c1.csv", ["cand-1"]),)
    captured = {}

    def fake_sampling_rate_for_sets(point_sets, rate):
        captured["point_sets"] = list(point_sets)
        captured["rate"] = rate
        return 37

    monkeypatch.setattr(ReportCompare, "sampling_rate_for_sets", fake_sampling_rate_for_sets)
    monkeypatch.setattr(
        ReportCompare.PairEvidence,
        "compute_pair_metrics_from_points",
        lambda *args, **kwargs: _metric_dict(2.0),
    )

    rows = ReportCompare._compare_direct_distribution_pairs_class(
        reference_entries,
        candidate_entries,
        run_id="run-1",
        source_name="generated",
        dataset_name="dataset-a",
        variant="syntTO",
        class_key="arrow",
        reference_input=Path("/tmp/reference"),
        rate=None,
        alignment=0,
        summary_shape="medoid",
        popular=False,
        round_precision=3,
        metric_names=("shapeError",),
        dtw_window=None,
        exact_dtw=False,
    )[0]

    assert captured == {
        "point_sets": [["ref-1"], ["ref-2"], ["cand-1"]],
        "rate": None,
    }
    assert rows[0]["rate"] == 37


def test_human_baseline_rate_selection_stays_reference_only(monkeypatch):
    reference_entries = (_entry("r1.csv", ["ref-1"]), _entry("r2.csv", ["ref-2"]))
    captured = {}

    def fake_sampling_rate_for_sets(point_sets, rate):
        captured["point_sets"] = list(point_sets)
        captured["rate"] = rate
        return 41

    monkeypatch.setattr(ReportCompare, "sampling_rate_for_sets", fake_sampling_rate_for_sets)
    monkeypatch.setattr(ReportCompare, "Gesture", lambda *args, **kwargs: object())
    monkeypatch.setattr(ReportCompare, "SummaryGesture", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ReportCompare,
        "compute_metrics",
        lambda *args, **kwargs: _metric_dict(3.0),
    )

    rows = ReportCompare._compare_human_baseline_class(
        reference_entries,
        run_id="run-1",
        source_name="generated",
        dataset_name="dataset-a",
        variant="syntTO",
        class_key="arrow",
        reference_input=Path("/tmp/reference"),
        rate=None,
        alignment=0,
        summary_shape="medoid",
        popular=False,
        round_precision=3,
        metric_names=("shapeError",),
        dtw_window=None,
        exact_dtw=False,
    )[0]

    assert captured == {
        "point_sets": [["ref-1"], ["ref-2"]],
        "rate": None,
    }
    assert rows[0]["rate"] == 41
