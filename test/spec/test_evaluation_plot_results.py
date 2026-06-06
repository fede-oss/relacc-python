import importlib.util
import sys
from pathlib import Path

import pytest


def _load_plot_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "evaluation_plot_scripts" / "plot_evaluation_results.py"
    spec = importlib.util.spec_from_file_location("plot_evaluation_results", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_aggregate_stats_and_labels_include_variability():
    plots = _load_plot_module()

    stats = plots.aggregate_stats([1.0, 2.0, 3.0])

    assert stats.n == 3
    assert stats.mean == pytest.approx(2.0)
    assert stats.sd == pytest.approx(1.0)
    assert stats.ci_low == pytest.approx(0.8683934723883335)
    assert stats.ci_high == pytest.approx(3.1316065276116665)
    assert plots.mean_std_label(stats) == "2.00 +/- 1.00"
    assert plots.ci_label(stats) == "95% CI 0.87-3.13"


def test_matrix_from_rows_returns_cell_level_spread():
    plots = _load_plot_module()

    rows = [
        {"sourceLabel": "generated", "dataset": "arrows", "score": 1.0},
        {"sourceLabel": "generated", "dataset": "arrows", "score": 3.0},
        {"sourceLabel": "generated", "dataset": "circles", "score": ""},
    ]

    row_labels, col_labels, data, stats_by_key = plots.matrix_from_rows(
        rows,
        "sourceLabel",
        "dataset",
        "score",
    )

    assert row_labels == ["generated"]
    assert col_labels == ["arrows"]
    assert data[0, 0] == pytest.approx(2.0)
    assert stats_by_key[("generated", "arrows")].sd == pytest.approx(1.41421356237)


def test_matrix_from_rows_uses_explicit_preaggregated_spread():
    plots = _load_plot_module()

    rows = [
        {
            "sourceLabel": "generated",
            "dataset": "arrows",
            "score": 2.0,
            "scoreN": 6,
            "scoreSd": 0.5,
            "scoreCi95Low": 1.6,
            "scoreCi95High": 2.4,
        }
    ]

    _, _, data, stats_by_key = plots.matrix_from_rows(
        rows,
        "sourceLabel",
        "dataset",
        "score",
    )

    stats = stats_by_key[("generated", "arrows")]
    assert data[0, 0] == pytest.approx(2.0)
    assert stats.n == 6
    assert stats.sd == pytest.approx(0.5)
    assert stats.ci_low == pytest.approx(1.6)
    assert stats.ci_high == pytest.approx(2.4)
