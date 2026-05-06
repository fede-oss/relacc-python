from pathlib import Path

import pytest

from relacc.distribution_metrics import DISTRIBUTION_METRIC_NAMES
from relacc.metrics import METRIC_NAMES
from relacc.pipeline import distribution as Distribution


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

    assert len(Distribution._unordered_reference_pairs([ref_a, ref_b, ref_c])) == 3
    assert Distribution._candidate_reference_pairs([ref_a, ref_b], [cand_a, cand_b]) == [
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
            "reason": "missingCandidate",
            "referenceCount": 2,
            "candidateCount": 0,
        },
        {
            "classKey": "triangle",
            "reason": "missingReference",
            "referenceCount": 0,
            "candidateCount": 1,
        },
    ]
    assert invalid_classes == [
        {
            "classKey": "circle",
            "reason": "needAtLeastTwoReferenceSamples",
            "referenceCount": 1,
            "candidateCount": 1,
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
        ),
    )

    captured_point_sets = []
    metric_values = iter([1.0, 3.0, 5.0, 7.0])

    def fake_sampling_rate_for_sets(point_sets, rate):
        captured_point_sets.append(list(point_sets))
        return 9

    def fake_compute_pair_metrics(*args, **kwargs):
        return _metric_dict(next(metric_values))

    monkeypatch.setattr(Distribution, "sampling_rate_for_sets", fake_sampling_rate_for_sets)
    monkeypatch.setattr(Distribution, "compute_pair_metrics_from_points", fake_compute_pair_metrics)

    (
        baseline_samples,
        candidate_samples,
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
    assert baseline_samples["shapeError"] == [2.0]
    assert candidate_samples["shapeError"] == [5.0, 7.0]


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
    _write_csv(candidate_dir / "g02-circle-01.csv", _sample_rows(8))
    _write_csv(candidate_dir / "g03-square-01.csv", _sample_rows(9))
    _write_csv(candidate_dir / "g04-triangle-01.csv", _sample_rows(10))

    payload = Distribution.run_distribution_comparison(
        str(reference_dir),
        str(candidate_dir),
        round_precision=4,
    )

    assert payload["metadata"]["comparisonMode"] == "distribution"
    assert payload["metadata"]["groupBy"] == "filename-label"
    assert payload["metadata"]["validClassCount"] == 3
    assert payload["metadata"]["dtwWindow"] is None
    assert payload["metadata"]["exactDtw"] is False
    assert payload["metadata"]["skippedClasses"] == [
        {
            "classKey": "triangle",
            "reason": "missingReference",
            "referenceCount": 0,
            "candidateCount": 1,
        }
    ]
    assert payload["metadata"]["invalidClasses"] == []

    per_class_results = payload["results"]["perClass"]
    overall_results = payload["results"]["overall"]
    assert len(per_class_results) == 3 * len(METRIC_NAMES)
    assert len(overall_results) == len(METRIC_NAMES)

    shape_rows = [row for row in per_class_results if row["gestureMetric"] == "shapeError"]
    sample_counts = {
        row["classKey"]: (row["baselineSampleCount"], row["candidateSampleCount"])
        for row in shape_rows
    }
    assert sample_counts == {
        "arrow": (1, 2),
        "circle": (3, 3),
        "square": (1, 2),
    }
    assert set(shape_rows[0]["distributionMetrics"].keys()) == set(DISTRIBUTION_METRIC_NAMES)

    overall_shape = next(row for row in overall_results if row["gestureMetric"] == "shapeError")
    assert overall_shape["scope"] == "overall"
    assert overall_shape["classKey"] is None
    assert overall_shape["baselineSampleCount"] == 5
    assert overall_shape["candidateSampleCount"] == 7
    assert overall_shape["baselineStats"]["n"] == 5
    assert overall_shape["candidateStats"]["n"] == 7


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
    assert payload["metadata"]["summary"] == "centroid"
    assert payload["metadata"]["popular"] is True
    assert payload["metadata"]["dtwWindow"] == 4
    first_row = payload["results"]["perClass"][0]
    assert first_row["referenceCount"] == 2
    assert first_row["candidateCount"] == 1

    with pytest.raises(ValueError, match="Invalid group-by mode"):
        Distribution.run_distribution_comparison(
            str(reference_dir),
            str(candidate_dir),
            group_by="broken",
        )


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

    with pytest.raises(ValueError, match="No candidate CSV files found"):
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
                "referenceCount": 2,
                "candidateCount": 1,
                "baselineSampleCount": 1,
                "candidateSampleCount": 2,
                "baselineStats": {"mean": 1.0, "mdn": 1.0, "sd": 0.0, "min": 1.0, "max": 1.0},
                "candidateStats": {"mean": 2.0, "mdn": 2.0, "sd": 0.0, "min": 2.0, "max": 2.0},
                "distributionMetrics": {
                    "wassersteinDistance": 1.0,
                    "energyDistance": 1.0,
                    "ksStatistic": 1.0,
                    "ksPValue": 0.5,
                },
            }
        ],
        "overall": [
            {
                "scope": "overall",
                "classKey": None,
                "gestureMetric": "shapeError",
                "referenceCount": 2,
                "candidateCount": 1,
                "baselineSampleCount": 1,
                "candidateSampleCount": 2,
                "baselineStats": {"mean": 1.0, "mdn": 1.0, "sd": 0.0, "min": 1.0, "max": 1.0},
                "candidateStats": {"mean": 2.0, "mdn": 2.0, "sd": 0.0, "min": 2.0, "max": 2.0},
                "distributionMetrics": {
                    "wassersteinDistance": 1.0,
                    "energyDistance": 1.0,
                    "ksStatistic": 1.0,
                    "ksPValue": 0.5,
                },
            }
        ],
    }

    output = Distribution.format_distribution_rows_csv(results)
    lines = output.splitlines()
    assert lines[0].startswith("scope,classKey,gestureMetric")
    assert '"arrow,fast"' in lines[1]
    assert lines[2].startswith("overall,")
