import csv
import math
from pathlib import Path

import pytest

from relacc.distribution_metrics import (
    ASYMMETRIC,
    DISTRIBUTION_METRIC_NAMES,
    DISTRIBUTION_METRIC_SYMMETRY,
    SYMMETRIC,
)
from relacc.metrics import METRIC_NAMES
from relacc.pipeline import distribution as Distribution

REMOVED_INFERENTIAL_FIELDS = [
    "meanCi95Low",
    "meanCi95High",
    "normalityPValue",
    "ksPValue",
]


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


def _metric_dict(value):
    return {name: float(value) for name in METRIC_NAMES}


def test_grouping_helpers_and_pair_generation():
    assert Distribution._normalize_group_by(None) == "filename-label"
    assert Distribution._normalize_group_by("Parent-Dir") == "parent-dir"
    with pytest.raises(ValueError, match="Invalid group-by mode"):
        Distribution._normalize_group_by("broken")

    assert Distribution._filename_label_class_key("s01-arrow-01.csv") == "arrow"
    with pytest.raises(ValueError, match="Cannot derive class label"):
        Distribution._filename_label_class_key("arrow.csv")

    assert Distribution._parent_dir_class_key("arrow/sample.csv") == "arrow"
    assert Distribution._parent_dir_class_key("sample.csv") == "."
    assert (
        Distribution._class_key_for_relative_path("arrow/sample.csv", "parent-dir")
        == "arrow"
    )

    ref_a = Distribution.GestureEntry("r1", "r1", "arrow", ["r1"])
    ref_b = Distribution.GestureEntry("r2", "r2", "arrow", ["r2"])
    ref_c = Distribution.GestureEntry("r3", "r3", "arrow", ["r3"])
    cand_a = Distribution.GestureEntry("c1", "c1", "arrow", ["c1"])
    cand_b = Distribution.GestureEntry("c2", "c2", "arrow", ["c2"])

    assert len(Distribution._unordered_pairs([ref_a, ref_b, ref_c])) == 3
    assert Distribution._between_group_pairs([ref_a, ref_b], [cand_a, cand_b]) == [
        (ref_a, cand_a),
        (ref_a, cand_b),
        (ref_b, cand_a),
        (ref_b, cand_b),
    ]


def test_discover_class_comparisons_reports_valid_invalid_and_skipped(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "arrow" / "r1.csv", _sample_rows(0))
    _write_csv(reference_dir / "arrow" / "r2.csv", _sample_rows(1))
    _write_csv(reference_dir / "circle" / "r1.csv", _sample_rows(2))
    _write_csv(reference_dir / "square" / "r1.csv", _sample_rows(3))
    _write_csv(reference_dir / "square" / "r2.csv", _sample_rows(4))

    _write_csv(candidate_dir / "arrow" / "c1.csv", _sample_rows(5))
    _write_csv(candidate_dir / "circle" / "c1.csv", _sample_rows(6))
    _write_csv(candidate_dir / "triangle" / "c1.csv", _sample_rows(7))

    valid_classes, skipped_classes, invalid_classes = Distribution.discover_class_comparisons(
        str(reference_dir),
        str(candidate_dir),
        Distribution.GROUP_BY_PARENT_DIR,
    )

    assert [spec.class_key for spec in valid_classes] == ["arrow"]
    assert skipped_classes == [
        {
            "classKey": "square",
            "reason": "missingComparison",
            "referenceGroupCount": 2,
            "comparisonGroupCount": 0,
        },
        {
            "classKey": "triangle",
            "reason": "missingReference",
            "referenceGroupCount": 0,
            "comparisonGroupCount": 1,
        },
    ]
    assert invalid_classes == [
        {
            "classKey": "circle",
            "reason": "needAtLeastTwoReferenceSamples",
            "referenceGroupCount": 1,
            "comparisonGroupCount": 1,
        }
    ]


def test_metric_samples_for_class_use_reference_only_for_auto_rate_and_generate_expected_pairs(
    monkeypatch,
):
    spec = Distribution.ClassComparisonSpec(
        class_key="arrow",
        reference_entries=(
            Distribution.GestureEntry("r1", "r1", "arrow", ["ref-1"]),
            Distribution.GestureEntry("r2", "r2", "arrow", ["ref-2"]),
        ),
        candidate_entries=(
            Distribution.GestureEntry("c1", "c1", "arrow", ["cand-1"]),
            Distribution.GestureEntry("c2", "c2", "arrow", ["cand-2"]),
        ),
    )

    captured_point_sets = []
    metric_values = iter([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])

    def fake_sampling_rate_for_sets(point_sets, rate):
        captured_point_sets.append(list(point_sets))
        return 9

    def fake_compute_pair_metrics(*args, **kwargs):
        return _metric_dict(next(metric_values))

    monkeypatch.setattr(Distribution, "sampling_rate_for_sets", fake_sampling_rate_for_sets)
    monkeypatch.setattr(Distribution, "compute_pair_metrics_from_points", fake_compute_pair_metrics)

    (
        samples,
        effective_rate,
        selected_dtw_window,
    ) = Distribution._metric_samples_for_class(
        spec,
        rate=None,
        alignment_type=0,
        summary_shape=None,
        popular_shape=False,
    )

    assert effective_rate == 9
    assert selected_dtw_window is None
    assert captured_point_sets == [[["ref-1"], ["ref-2"]]]
    assert samples[Distribution.WITHIN_REFERENCE_GROUP]["shapeError"] == [2.0]
    assert samples[Distribution.WITHIN_COMPARISON_GROUP]["shapeError"] == [6.0]
    assert samples[Distribution.BETWEEN_GROUPS]["shapeError"] == [9.0, 11.0, 13.0, 15.0]


def test_run_distribution_comparison_filename_grouping_and_no_valid_classes(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(1))

    with pytest.raises(ValueError, match="No valid classes found"):
        Distribution.run_distribution_comparison(str(reference_dir), str(candidate_dir))


def test_run_distribution_comparison_outputs_per_class_and_overall(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "s01-arrow-01.csv", _sample_rows(0))
    _write_csv(reference_dir / "s02-arrow-02.csv", _sample_rows(1))
    _write_csv(reference_dir / "s03-circle-01.csv", _sample_rows(2))
    _write_csv(reference_dir / "s04-circle-02.csv", _sample_rows(3))
    _write_csv(reference_dir / "s05-circle-03.csv", _sample_rows(4))
    _write_csv(reference_dir / "s06-square-01.csv", _sample_rows(5))
    _write_csv(reference_dir / "s07-square-02.csv", _sample_rows(6))

    _write_csv(candidate_dir / "g01-arrow-01.csv", _sample_rows(7))
    _write_csv(candidate_dir / "g05-arrow-02.csv", _sample_rows(11))
    _write_csv(candidate_dir / "g02-circle-01.csv", _sample_rows(8))
    _write_csv(candidate_dir / "g03-square-01.csv", _sample_rows(9))
    _write_csv(candidate_dir / "g04-triangle-01.csv", _sample_rows(10))

    payload = Distribution.run_distribution_comparison(
        str(reference_dir),
        str(candidate_dir),
        round_precision=4,
    )

    assert payload["metadata"]["comparisonMode"] == "distribution"
    assert payload["metadata"]["statisticalMode"] == "descriptive-pair-distances"
    assert payload["metadata"]["independentUnit"] == "gesture-file"
    assert payload["metadata"]["pairValuesIndependent"] is False
    assert payload["metadata"]["statisticsSchemaVersion"] == 2
    assert payload["metadata"]["removedInferentialFields"] == REMOVED_INFERENTIAL_FIELDS
    assert payload["metadata"]["groupBy"] == "filename-label"
    assert payload["metadata"]["comparisonGroups"][Distribution.WITHIN_REFERENCE_GROUP][
        "groupName"
    ] == "reference"
    assert payload["metadata"]["comparisonGroups"][Distribution.WITHIN_COMPARISON_GROUP][
        "groupName"
    ] == "comparison"
    assert payload["metadata"]["validClassCount"] == 3
    assert payload["metadata"]["dtwWindow"] is None
    assert payload["metadata"]["exactDtw"] is False
    assert {
        entry["name"]: entry["symmetry"]
        for entry in payload["metadata"]["distributionMetricSemantics"]
    } == DISTRIBUTION_METRIC_SYMMETRY
    assert payload["metadata"]["skippedClasses"] == [
        {
            "classKey": "triangle",
            "reason": "missingReference",
            "referenceGroupCount": 0,
            "comparisonGroupCount": 1,
        }
    ]
    assert payload["metadata"]["invalidClasses"] == []

    per_class_results = payload["results"]["perClass"]
    overall_results = payload["results"]["overall"]
    assert len(per_class_results) == 3 * len(METRIC_NAMES)
    assert len(overall_results) == len(METRIC_NAMES)

    shape_rows = [row for row in per_class_results if row["gestureMetric"] == "shapeError"]
    sample_counts = {
        row["classKey"]: (
            row["withinReferenceSampleCount"],
            row["withinComparisonSampleCount"],
            row["betweenGroupsSampleCount"],
        )
        for row in shape_rows
    }
    assert sample_counts == {
        "arrow": (1, 1, 4),
        "circle": (3, 0, 3),
        "square": (1, 0, 2),
    }
    assert set(shape_rows[0]["distributionMetrics"].keys()) == set(DISTRIBUTION_METRIC_NAMES)
    assert "ksStatistic" in shape_rows[0]["distributionMetrics"]
    assert "ksPValue" not in shape_rows[0]["distributionMetrics"]
    assert shape_rows[0]["distributionMetrics"]["earthMoverDistance"] == (
        shape_rows[0]["distributionMetrics"]["wassersteinDistance"]
    )
    assert payload["metadata"]["distributionMetricSemantics"][1]["symmetry"] == SYMMETRIC
    assert any(
        entry["name"] == "klDivergenceReferenceToCandidate"
        and entry["symmetry"] == ASYMMETRIC
        for entry in payload["metadata"]["distributionMetricSemantics"]
    )
    assert "baselineStats" not in shape_rows[0]
    assert "candidateStats" not in shape_rows[0]
    assert shape_rows[0]["statisticalMode"] == "descriptive-pair-distances"
    assert shape_rows[0]["independentUnit"] == "gesture-file"
    assert shape_rows[0]["pairValuesIndependent"] is False
    assert shape_rows[0]["statisticsSchemaVersion"] == 2
    assert shape_rows[0]["removedInferentialFields"] == REMOVED_INFERENTIAL_FIELDS
    for stats_name in (
        "withinReferenceStats",
        "withinComparisonStats",
        "betweenGroupsStats",
    ):
        stats_keys = set(shape_rows[0][stats_name])
        assert {
            "mean",
            "mdn",
            "sd",
            "variance",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "skewness",
            "kurtosis",
            "n",
        } <= stats_keys
        assert not stats_keys.intersection(REMOVED_INFERENTIAL_FIELDS)

    overall_shape = next(row for row in overall_results if row["gestureMetric"] == "shapeError")
    assert overall_shape["scope"] == "overall"
    assert overall_shape["classKey"] is None
    assert overall_shape["withinReferenceSampleCount"] == 5
    assert overall_shape["withinComparisonSampleCount"] == 1
    assert overall_shape["betweenGroupsSampleCount"] == 9
    assert overall_shape["withinReferenceStats"]["n"] == 5
    assert overall_shape["withinComparisonStats"]["n"] == 1
    assert overall_shape["betweenGroupsStats"]["n"] == 9
    assert len(payload["rawMetricOutputs"]) == 15 * len(METRIC_NAMES)
    assert {row["recordType"] for row in payload["rawMetricOutputs"]} == {
        "rawMetricOutput"
    }
    assert {row["sampleKind"] for row in payload["rawMetricOutputs"]} == {
        Distribution.WITHIN_REFERENCE_GROUP,
        Distribution.WITHIN_COMPARISON_GROUP,
        Distribution.BETWEEN_GROUPS,
    }
    assert all("candidateKey" not in row for row in payload["rawMetricOutputs"])
    assert len(payload["rawDistributionOutputs"]) > 0
    assert {row["recordType"] for row in payload["rawDistributionOutputs"]} == {
        "rawDistributionOutput"
    }
    assert {
        "withinComparisonToReferenceMeanRatio",
        "withinComparisonToReferenceMdnRatio",
        "withinComparisonToReferenceSdRatio",
    }.issubset(
        {row["distributionMetric"] for row in payload["rawDistributionOutputs"]}
    )
    assert all(
        row["statisticsSchemaVersion"] == 2
        and row["statisticalMode"] == "descriptive-pair-distances"
        and row["independentUnit"] == "gesture-file"
        and row["pairValuesIndependent"] is False
        and row["removedInferentialFields"] == REMOVED_INFERENTIAL_FIELDS
        for row in payload["rawDistributionOutputs"]
    )


def test_run_distribution_comparison_reports_dependent_pair_contract(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    for index in range(4):
        _write_csv(
            reference_dir / "arrow" / f"ref-{index}.csv",
            _sample_rows(index),
        )
    for index in range(2):
        _write_csv(
            candidate_dir / "arrow" / f"cand-{index}.csv",
            _sample_rows(index + 10),
        )

    payload = Distribution.run_distribution_comparison(
        str(reference_dir),
        str(candidate_dir),
        group_by="parent-dir",
    )

    shape_row = next(
        row
        for row in payload["results"]["perClass"]
        if row["classKey"] == "arrow" and row["gestureMetric"] == "shapeError"
    )

    assert shape_row["referenceGroupCount"] == 4
    assert shape_row["withinReferenceSampleCount"] == 6
    assert shape_row["withinReferenceStats"]["n"] == 6
    assert shape_row["independentUnit"] == "gesture-file"
    assert shape_row["pairValuesIndependent"] is False
    assert shape_row["statisticalMode"] == "descriptive-pair-distances"
    assert shape_row["statisticsSchemaVersion"] == 2


def test_run_distribution_comparison_keeps_stroke_error_stats_coherent(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "arrow" / "ref-two-a.csv", _sample_rows(0, 0, 1))
    _write_csv(reference_dir / "arrow" / "ref-two-b.csv", _sample_rows(1, 0, 1))
    _write_csv(candidate_dir / "arrow" / "cand-two.csv", _sample_rows(2, 0, 1))
    _write_csv(candidate_dir / "arrow" / "cand-one.csv", _sample_rows(3, 0, 0))

    payload = Distribution.run_distribution_comparison(
        str(reference_dir),
        str(candidate_dir),
        group_by="parent-dir",
        rate=4,
    )

    stroke_row = next(
        row
        for row in payload["results"]["overall"]
        if row["gestureMetric"] == "strokeError"
    )
    between_group_stats = stroke_row["betweenGroupsStats"]
    assert between_group_stats["min"] <= between_group_stats["mean"] <= between_group_stats["max"]
    assert between_group_stats["min"] <= between_group_stats["mdn"] <= between_group_stats["max"]
    assert between_group_stats["mean"] == 0.5
    assert between_group_stats["mdn"] == 0.5
    assert between_group_stats["min"] == 0.0
    assert between_group_stats["max"] == 1.0


def test_run_distribution_comparison_parent_dir_grouping_and_validation(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "arrow" / "sample-a.csv", _sample_rows(0))
    _write_csv(reference_dir / "arrow" / "sample-b.csv", _sample_rows(1))
    _write_csv(candidate_dir / "arrow" / "sample-c.csv", _sample_rows(2))

    payload = Distribution.run_distribution_comparison(
        str(reference_dir),
        str(candidate_dir),
        rate=11,
        alignment_type=1,
        summary_shape="Centroid",
        popular_shape=True,
        round_precision=2,
        group_by="parent-dir",
        dtw_window=4,
    )

    assert payload["metadata"]["groupBy"] == "parent-dir"
    assert payload["metadata"]["rate"] == 11
    assert payload["metadata"]["alignment"] == 1
    assert payload["metadata"]["alignmentName"] == "cloud-match"
    assert payload["metadata"]["summary"] == "centroid"
    assert payload["metadata"]["popular"] is True
    assert payload["metadata"]["dtwWindow"] == 4
    first_row = payload["results"]["perClass"][0]
    assert first_row["referenceGroupCount"] == 2
    assert first_row["comparisonGroupCount"] == 1
    assert first_row["withinComparisonSampleCount"] == 0

    with pytest.raises(ValueError, match="Invalid group-by mode"):
        Distribution.run_distribution_comparison(
            str(reference_dir),
            str(candidate_dir),
            group_by="broken",
        )


def test_summary_stats_include_shape_quantiles_and_small_n_behavior():
    stats = Distribution._summary_stats(
        [1.0, 2.0, 3.0, 4.0, 5.0, float("inf"), float("nan")],
        2,
    )

    expected_stats = {
        "mean": 3.0,
        "mdn": 3.0,
        "sd": 1.58,
        "variance": 2.5,
        "min": 1.0,
        "max": 5.0,
        "q05": 1.2,
        "q25": 2.0,
        "q50": 3.0,
        "q75": 4.0,
        "q95": 4.8,
        "skewness": 0.0,
        "kurtosis": -1.2,
        "n": 5,
    }
    for key, expected_value in expected_stats.items():
        assert stats[key] == expected_value
    assert not set(stats).intersection(REMOVED_INFERENTIAL_FIELDS)

    one_value_stats = Distribution._summary_stats([2.0], 2)
    assert one_value_stats["variance"] == 0.0
    assert one_value_stats["q05"] == 2.0
    assert one_value_stats["q95"] == 2.0
    assert math.isnan(one_value_stats["skewness"])
    assert math.isnan(one_value_stats["kurtosis"])


def test_summary_stats_remain_descriptive_when_sample_is_large_enough():
    stats = Distribution._summary_stats([1, 2, 3, 4, 5, 6, 7, 8], 3)

    assert stats["skewness"] == 0.0
    assert stats["kurtosis"] == -1.2
    assert not set(stats).intersection(REMOVED_INFERENTIAL_FIELDS)


def test_summary_stats_reject_impossible_aggregate_values():
    with pytest.raises(ValueError, match="outside min=0.0 and max=1.0"):
        Distribution._validate_summary_stats(
            {
                "mean": 17.0,
                "mdn": 0.0,
                "sd": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 1.0,
                "q05": 0.0,
                "q25": 0.0,
                "q50": 0.0,
                "q75": 0.0,
                "q95": 1.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "n": 3,
            }
        )


def test_summary_stats_return_nan_shape_fields_for_empty_samples():
    stats = Distribution._summary_stats([float("nan"), float("inf")], 2)

    assert stats["n"] == 0
    assert math.isnan(stats["variance"])
    assert math.isnan(stats["q50"])
    assert math.isnan(stats["skewness"])
    assert math.isnan(stats["kurtosis"])
    assert not set(stats).intersection(REMOVED_INFERENTIAL_FIELDS)


def test_discover_class_comparisons_require_reference_and_candidate_csvs(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()

    with pytest.raises(ValueError, match="No reference CSV files found"):
        Distribution.discover_class_comparisons(
            str(reference_dir),
            str(candidate_dir),
            Distribution.GROUP_BY_PARENT_DIR,
        )

    _write_csv(reference_dir / "arrow" / "r1.csv", _sample_rows(0))
    _write_csv(reference_dir / "arrow" / "r2.csv", _sample_rows(1))

    with pytest.raises(ValueError, match="No comparison CSV files found"):
        Distribution.discover_class_comparisons(
            str(reference_dir),
            str(candidate_dir),
            Distribution.GROUP_BY_PARENT_DIR,
        )


def test_format_distribution_rows_csv_with_escaping_and_overall_row():
    results = {
        "perClass": [
            {
                "scope": "class",
                "classKey": "arrow,fast",
                "gestureMetric": "shapeError",
                "referenceGroupCount": 2,
                "comparisonGroupCount": 2,
                "withinReferenceSampleCount": 1,
                "withinComparisonSampleCount": 1,
                "betweenGroupsSampleCount": 4,
                "withinReferenceStats": {
                    "mean": 1.0,
                    "mdn": 1.0,
                    "sd": 0.5,
                    "min": 1.0,
                    "max": 1.0,
                },
                "withinComparisonStats": {
                    "mean": 3.0,
                    "mdn": 3.0,
                    "sd": 1.5,
                    "min": 3.0,
                    "max": 3.0,
                },
                "betweenGroupsStats": {
                    "mean": 2.0,
                    "mdn": 2.0,
                    "sd": 0.0,
                    "min": 2.0,
                    "max": 2.0,
                },
                "withinComparisonToReferenceRatios": {
                    "mean": 3.0,
                    "mdn": 3.0,
                    "sd": 3.0,
                },
                "distributionMetrics": {
                    "wassersteinDistance": 1.0,
                    "energyDistance": 1.0,
                    "ksStatistic": 1.0,
                },
            }
        ],
        "overall": [
            {
                "scope": "overall",
                "classKey": None,
                "gestureMetric": "shapeError",
                "referenceGroupCount": 2,
                "comparisonGroupCount": 2,
                "withinReferenceSampleCount": 1,
                "withinComparisonSampleCount": 1,
                "betweenGroupsSampleCount": 4,
                "withinReferenceStats": {"mean": 1.0, "mdn": 1.0, "sd": 0.5},
                "withinComparisonStats": {"mean": 3.0, "mdn": 3.0, "sd": 1.5},
                "betweenGroupsStats": {"mean": 2.0, "mdn": 2.0, "sd": 0.0},
                "withinComparisonToReferenceRatios": {
                    "mean": 3.0,
                    "mdn": 3.0,
                    "sd": 3.0,
                },
                "distributionMetrics": {
                    "wassersteinDistance": 1.0,
                    "energyDistance": 1.0,
                    "ksStatistic": 1.0,
                },
            }
        ],
    }

    output = Distribution.format_distribution_rows_csv(results)
    lines = output.splitlines()
    columns = lines[0].split(",")
    assert columns[:8] == [
        "scope",
        "classKey",
        "gestureMetric",
        "statisticalMode",
        "independentUnit",
        "pairValuesIndependent",
        "statisticsSchemaVersion",
        "removedInferentialFields",
    ]
    assert "referenceGroupCount" in columns
    assert "comparisonGroupCount" in columns
    assert "withinReferenceFiniteSampleCount" in columns
    assert "withinComparisonFiniteSampleCount" in columns
    assert "betweenGroupsFiniteSampleCount" in columns
    assert "withinReferenceVariance" in columns
    assert "withinReferenceQ95" in columns
    assert "withinReferenceSkewness" in columns
    assert "withinReferenceKurtosis" in columns
    assert "withinComparisonVariance" in columns
    assert "withinComparisonQ95" in columns
    assert "betweenGroupsVariance" in columns
    assert "betweenGroupsQ95" in columns
    assert "ksStatistic" in columns
    assert "ksPValue" not in columns
    assert not any("MeanCi95" in column for column in columns)
    assert not any("NormalityPValue" in column for column in columns)
    assert "withinComparisonToReferenceMeanRatio" in columns
    assert "baselineMean" not in columns
    assert "candidateMean" not in columns
    assert '"arrow,fast"' in lines[1]
    parsed_rows = list(csv.DictReader(lines))
    assert parsed_rows[1]["scope"] == "overall"
    assert parsed_rows[0]["statisticalMode"] == "descriptive-pair-distances"
    assert parsed_rows[0]["independentUnit"] == "gesture-file"
    assert parsed_rows[0]["pairValuesIndependent"] == "False"
    assert parsed_rows[0]["statisticsSchemaVersion"] == "2"
    assert parsed_rows[0]["removedInferentialFields"] == (
        '["meanCi95Low","meanCi95High","normalityPValue","ksPValue"]'
    )

    legacy_output = Distribution.format_distribution_rows_csv(
        results,
        legacy_column_names=True,
    )
    legacy_columns = legacy_output.splitlines()[0].split(",")
    assert "baselineFiniteSampleCount" in legacy_columns
    assert "candidateFiniteSampleCount" in legacy_columns
    assert "baselineMean" in legacy_columns
    assert "candidateMean" in legacy_columns
    assert not any("MeanCi95" in column for column in legacy_columns)
    assert not any("NormalityPValue" in column for column in legacy_columns)
    assert "withinReferenceMean" not in legacy_columns
