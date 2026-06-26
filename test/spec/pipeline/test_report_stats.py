import pytest

from relacc.metrics import METRIC_NAMES
from relacc.pipeline import report_stats as ReportStats


def test_validate_bounded_stats_rejects_values_outside_min_max():
    with pytest.raises(ValueError, match="outside min=0.0 and max=1.0"):
        ReportStats._validate_bounded_stats(
            {"mean": 1.5, "mdn": 0.5, "min": 0.0, "max": 1.0},
            ("mean", "mdn"),
            "shapeError",
        )


def test_distribution_summary_handles_empty_and_non_empty_values():
    empty = ReportStats._distribution_summary([], total_n=3, round_precision=3)
    assert empty["n"] == 3
    assert empty["finiteN"] == 0
    assert empty["mean"] is None
    assert empty["q50"] is None

    summary = ReportStats._distribution_summary([1.0, 2.0, 3.0], 4, 3)
    assert summary["n"] == 4
    assert summary["finiteN"] == 3
    assert summary["mean"] == 2.0
    assert summary["mdn"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_aggregate_rows_for_record_set_groups_small_record_set():
    metric_name = METRIC_NAMES[0]
    rows = [
        {
            "source": "generated",
            "dataset": "1dollar",
            "variant": "syntTO",
            "summary": "medoid",
            metric_name: 1.0,
        },
        {
            "source": "generated",
            "dataset": "1dollar",
            "variant": "syntTO",
            "summary": "medoid",
            metric_name: 3.0,
        },
    ]

    aggregate_rows = ReportStats._aggregate_rows_for_record_set(
        rows,
        "comparison-to-reference-summary",
        round_precision=3,
    )
    metric_rows = [row for row in aggregate_rows if row["metric"] == metric_name]

    assert {row["scope"] for row in metric_rows} == {
        "overall",
        "source",
        "dataset",
        "source-dataset",
        "run",
    }
    overall = next(row for row in metric_rows if row["scope"] == "overall")
    assert overall["recordSet"] == "comparison-to-reference-summary"
    assert overall["summary"] == "medoid"
    assert overall["n"] == 2
    assert overall["finiteN"] == 2
    assert overall["mean"] == 2.0
