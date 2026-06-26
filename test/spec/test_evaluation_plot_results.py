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


def test_tukey_upper_whisker_uses_iqr_rule():
    plots = _load_plot_module()

    whisker = plots.tukey_upper_whisker([1.0, 2.0, 3.0, 4.0])

    assert whisker == pytest.approx(5.5)


def test_run_label_hides_only_empty_and_root_variants():
    plots = _load_plot_module()

    assert plots.run_label({"source": "scriptstudio", "variant": "root"}) == "scriptstudio"
    assert plots.run_label({"source": "scriptstudio", "variant": ""}) == "scriptstudio"
    assert (
        plots.run_label({"source": "scriptstudio", "variant": "recoTO"})
        == "scriptstudio/recoTO"
    )


def test_load_distribution_rows_infers_legacy_report_context(tmp_path):
    plots = _load_plot_module()
    input_dir = tmp_path / "report"
    run_dir = input_dir / "DHG" / "1dollar"
    run_dir.mkdir(parents=True)
    (run_dir / "distribution.csv").write_text(
        "\n".join(
            [
                "scope,classKey,gestureMetric,baselineMean,candidateMean,meanRatio,wassersteinDistance,ksStatistic",
                "class,arrow,shapeError,1.0,2.0,2.0,0.5,0.25",
            ]
        ),
        encoding="utf-8",
    )

    rows = plots.load_distribution_rows(input_dir)

    assert rows[0]["source"] == "DHG"
    assert rows[0]["dataset"] == "1dollar"
    assert rows[0]["variant"] == ""
    assert rows[0]["metric"] == "shapeError"
    assert rows[0]["_runLabel"] == "DHG"


def test_load_distribution_rows_preserves_root_variant_but_hides_label_suffix(tmp_path):
    plots = _load_plot_module()
    input_dir = tmp_path / "report"
    root_dir = input_dir / "generated" / "1dollar"
    variant_dir = input_dir / "generated" / "1dollar" / "recoTO"
    root_dir.mkdir(parents=True)
    variant_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "source,dataset,variant,scope,classKey,metric,withinReferenceMean,withinComparisonMean,withinComparisonToReferenceMeanRatio,normalizedWassersteinDistance,ksStatistic",
            "generated,1dollar,{variant},class,arrow,shapeError,1.0,2.0,2.0,0.5,0.25",
        ]
    )
    (root_dir / "distribution.csv").write_text(
        csv_text.format(variant="root"),
        encoding="utf-8",
    )
    (variant_dir / "distribution.csv").write_text(
        csv_text.format(variant="recoTO"),
        encoding="utf-8",
    )

    rows = plots.load_distribution_rows(input_dir)
    rows_by_variant = {row["variant"]: row for row in rows}

    assert rows_by_variant["root"]["variant"] == "root"
    assert rows_by_variant["root"]["_runLabel"] == "generated"
    assert rows_by_variant["recoTO"]["_runLabel"] == "generated/recoTO"


def test_build_overview_tables_writes_root_and_meaningful_variant_labels(tmp_path):
    plots = _load_plot_module()
    output_dir = tmp_path / "plots"
    distribution_rows = []
    for variant in ("root", "recoTO"):
        row = {
            "source": "generated",
            "dataset": "1dollar",
            "variant": variant,
            "metric": "shapeError",
            "withinReferenceMean": "1.0",
            "withinComparisonMean": "2.0",
            "withinComparisonToReferenceMeanRatio": "2.0",
            "normalizedWassersteinDistance": "0.5",
            "ksStatistic": "0.25",
        }
        row["_runLabel"] = plots.run_label(row)
        distribution_rows.append(row)

    overview_rows, metric_rows = plots.build_overview_tables(distribution_rows, output_dir)
    overview_csv = plots.read_csv_rows(output_dir / "tables" / "source_dataset_scores.csv")
    metric_csv = plots.read_csv_rows(output_dir / "tables" / "source_dataset_metric_scores.csv")

    assert (output_dir / "tables" / "source_dataset_scores.csv").exists()
    assert {row["sourceLabel"] for row in overview_rows} == {
        "generated",
        "generated/recoTO",
    }
    assert {row["sourceLabel"] for row in overview_csv} == {
        "generated",
        "generated/recoTO",
    }
    assert {row["sourceLabel"] for row in metric_rows} == {
        "generated",
        "generated/recoTO",
    }
    assert "withinComparisonToReferenceMeanRatio" in metric_csv[0]


def test_plot_ratio_boxplots_writes_main_and_outlier_views(tmp_path):
    plots = _load_plot_module()
    output_dir = tmp_path / "plots"
    rows = []
    for value in [1.0, 1.1, 1.2, 1.3, 10.0]:
        rows.append(
            {
                "metric": "shapeError",
                "_runLabel": "DHG",
                "withinReferenceMean": "1.0",
                "withinComparisonMean": str(value),
                "withinComparisonToReferenceMeanRatio": str(value),
            }
        )

    plots.plot_ratio_boxplots(rows, output_dir)

    assert (output_dir / "boxplots" / "core_within_variability_ratio_by_source.png").exists()
    assert (output_dir / "boxplots" / "core_within_variability_ratio_outliers_by_source.png").exists()
    assert (output_dir / "tables" / "core_within_variability_ratio_iqr.csv").exists()


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
