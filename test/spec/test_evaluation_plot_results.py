import importlib.util
import sys
from pathlib import Path

import numpy as np
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


def test_heatmap_color_scale_uses_log_for_large_positive_ranges():
    plots = _load_plot_module()

    _, scale_key, scale_label = plots.heatmap_color_scale(
        plots.np.asarray([[0.1, 1.0, 100.0]])
    )

    assert scale_key == "log"
    assert "log10" in scale_label


def test_pairwise_matrix_from_combined_pair_keys_is_symmetric_with_empty_diagonal():
    plots = _load_plot_module()

    rows = [
        {"pairKey": "a::b", "shapeError": "2.0"},
        {"pairKey": "b::c", "shapeError": "4.0"},
    ]

    labels, _, data = plots.pairwise_matrix_from_combined_pair_keys(
        rows,
        "shapeError",
    )

    assert labels == ["a", "b", "c"]
    assert np.isnan(data[0, 0])
    assert data[0, 1] == pytest.approx(2.0)
    assert data[1, 0] == pytest.approx(2.0)
    assert data[1, 2] == pytest.approx(4.0)


def test_pairwise_matrix_from_raw_rows_supports_rectangular_between_group_data():
    plots = _load_plot_module()

    rows = [
        {"metric": "dtwDistance", "referenceKey": "ref-a", "candidateKey": "gen-a", "value": "1.5"},
        {"metric": "dtwDistance", "referenceKey": "ref-b", "candidateKey": "gen-a", "value": "2.5"},
    ]

    row_labels, col_labels, data = plots.pairwise_matrix_from_rows(
        rows,
        "dtwDistance",
        "referenceKey",
        "candidateKey",
        symmetric=False,
        neutral_diagonal=False,
    )

    assert row_labels == ["ref-a", "ref-b"]
    assert col_labels == ["gen-a"]
    assert data[:, 0].tolist() == pytest.approx([1.5, 2.5])


def test_quantile_summary_reports_iqr():
    plots = _load_plot_module()

    summary = plots.quantile_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["q1"] == pytest.approx(1.75)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["q3"] == pytest.approx(3.25)
    assert summary["iqr"] == pytest.approx(1.5)
    assert plots.iqr_label(summary) == "2.50 [1.75-3.25]"


def test_plot_pairwise_distance_heatmaps_writes_index_for_within_comparison(tmp_path):
    plots = _load_plot_module()
    input_dir = tmp_path / "report"
    output_dir = tmp_path / "plots"
    run_dir = input_dir / "source-a" / "dataset-a"
    run_dir.mkdir(parents=True)
    (run_dir / "within_comparison.csv").write_text(
        "\n".join(
            [
                "runId,source,dataset,variant,classKey,pairKey,shapeError,dtwDistance,velocityError",
                "source-a/dataset-a,source-a,dataset-a,,arrow,a::b,1.0,2.0,3.0",
                "source-a/dataset-a,source-a,dataset-a,,arrow,b::c,4.0,5.0,6.0",
            ]
        ),
        encoding="utf-8",
    )

    rows = plots.plot_pairwise_distance_heatmaps(
        input_dir,
        output_dir,
        metrics=("shapeError",),
    )

    assert len(rows) == 1
    assert rows[0]["pairType"] == "within-comparison"
    assert Path(rows[0]["path"]).exists()
    assert (output_dir / "tables" / "pairwise_heatmaps.csv").exists()
