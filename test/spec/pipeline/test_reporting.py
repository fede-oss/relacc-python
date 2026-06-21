from pathlib import Path

import pytest

from relacc.pipeline import reporting as Reporting
from relacc.pipeline import reporting_raw as ReportingRaw


def _write_csv(path: Path, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "stroke_id x y time is_writing",
                f"0 {10 + offset} 20 0 1",
                f"0 {12 + offset} 22 10 1",
                f"1 {14 + offset} 24 20 1",
                f"1 {16 + offset} 25 30 1",
            ]
        ),
        encoding="utf-8",
    )


def _manifest_keys(payload, source):
    return [row["key"] for row in payload["samples"] if row["source"] == source]


def test_select_reporting_samples_keeps_all_files_under_limit(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    for index in range(3):
        _write_csv(reference_dir / f"s{index:02d}-arrow-01.csv", index)
        _write_csv(candidate_dir / f"g{index:02d}-arrow-01.csv", index)

    payload = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=16,
    )

    assert payload["metadata"]["sampleLimit"] == 16
    assert payload["metadata"]["samplingMode"] == "stable"
    assert payload["metadata"]["groupCount"] == 1
    assert payload["metadata"]["sourceNames"] == {
        "reference": "human",
        "candidate": "generated",
    }
    assert len(payload["samples"]) == 6
    assert _manifest_keys(payload, "human") == [
        "s00-arrow-01.csv",
        "s01-arrow-01.csv",
        "s02-arrow-01.csv",
    ]
    assert payload["samples"][0]["sourceRole"] == "reference"


def test_select_reporting_samples_limits_each_source_to_sixteen(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    for index in range(20):
        _write_csv(reference_dir / f"s{index:02d}-arrow-01.csv", index)
        _write_csv(candidate_dir / f"g{index:02d}-arrow-01.csv", index)

    payload = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
    )

    reference_rows = [
        row for row in payload["samples"] if row["sourceRole"] == "reference"
    ]
    candidate_rows = [
        row for row in payload["samples"] if row["sourceRole"] == "candidate"
    ]
    assert len(reference_rows) == 16
    assert len(candidate_rows) == 16
    assert reference_rows[-1]["key"] == "s15-arrow-01.csv"
    assert candidate_rows[-1]["key"] == "g15-arrow-01.csv"


def test_sample_order_is_deterministic_for_stable_and_seeded_modes(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    for index in [9, 1, 6, 3, 0, 8, 4, 2, 7, 5]:
        _write_csv(reference_dir / f"s{index:02d}-arrow-01.csv", index)
        _write_csv(candidate_dir / f"g{index:02d}-arrow-01.csv", index)

    stable_a = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=4,
    )
    stable_b = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=4,
    )
    seeded_a = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=4,
        random_seed=7,
    )
    seeded_b = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=4,
        random_seed=7,
    )

    assert stable_a["samples"] == stable_b["samples"]
    assert _manifest_keys(stable_a, "human") == [
        "s00-arrow-01.csv",
        "s01-arrow-01.csv",
        "s02-arrow-01.csv",
        "s03-arrow-01.csv",
    ]
    assert seeded_a["samples"] == seeded_b["samples"]
    assert seeded_a["metadata"]["samplingMode"] == "seeded-random"
    assert _manifest_keys(seeded_a, "human") != _manifest_keys(
        stable_a,
        "human",
    )


def test_select_reporting_samples_allows_custom_source_names(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s00-arrow-01.csv", 0)
    _write_csv(candidate_dir / "g00-arrow-01.csv", 1)

    payload = Reporting.select_reporting_samples(
        str(reference_dir),
        str(candidate_dir),
        reference_source_name="real",
        candidate_source_name="synthetic",
    )

    assert payload["metadata"]["sourceNames"] == {
        "reference": "real",
        "candidate": "synthetic",
    }
    assert [row["source"] for row in payload["samples"]] == ["real", "synthetic"]
    assert [row["sourceRole"] for row in payload["samples"]] == [
        "reference",
        "candidate",
    ]


def test_filename_label_grouping_tracks_dataset_and_class(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "dataset-a" / "s01-arrow-01.csv", 1)
    _write_csv(reference_dir / "dataset-a" / "s02-circle-01.csv", 2)
    _write_csv(candidate_dir / "dataset-a" / "g01-arrow-01.csv", 3)
    _write_csv(candidate_dir / "dataset-b" / "g01-arrow-01.csv", 4)

    groups = Reporting.discover_reporting_sample_groups(
        str(reference_dir),
        str(candidate_dir),
        group_by="filename-label",
    )

    assert [(group.dataset_key, group.class_key) for group in groups] == [
        ("dataset-a", "arrow"),
        ("dataset-a", "circle"),
        ("dataset-b", "arrow"),
    ]
    first_group = groups[0]
    assert [entry.key for entry in first_group.reference_entries] == [
        "dataset-a/s01-arrow-01.csv"
    ]
    assert [entry.key for entry in first_group.candidate_entries] == [
        "dataset-a/g01-arrow-01.csv"
    ]


def test_auto_grouping_strips_source_folders_for_root_to_root_reports(tmp_path):
    reference_dir = tmp_path / "humans"
    candidate_dir = tmp_path / "scriptstudio"
    _write_csv(
        reference_dir / "1dollar" / "realTO" / "s01-arrow-fast-t01.csv",
        1,
    )
    _write_csv(
        candidate_dir / "1dollar" / "syntTO" / "s02-arrow-fast-t02.csv",
        2,
    )

    groups = Reporting.discover_reporting_sample_groups(
        str(reference_dir),
        str(candidate_dir),
    )

    assert [(group.dataset_key, group.class_key) for group in groups] == [
        ("1dollar", "arrow")
    ]
    assert groups[0].reference_entries[0].key == "1dollar/realTO/s01-arrow-fast-t01.csv"
    assert groups[0].candidate_entries[0].key == "1dollar/syntTO/s02-arrow-fast-t02.csv"


def test_auto_class_scheme_handles_known_dataset_filename_families(tmp_path):
    reference_dir = tmp_path / "humans"
    candidate_dir = tmp_path / "generated"

    _write_csv(
        reference_dir / "MobileTouchDB" / "realTO" / "user1-Sesion1-u1_s1_mayA.csv",
        1,
    )
    _write_csv(
        candidate_dir / "MobileTouchDB" / "syntTO" / "user2-Sesion1-u2_s1_mayA.csv",
        2,
    )
    _write_csv(reference_dir / "MCYT" / "realTO" / "001g01.csv", 3)
    _write_csv(candidate_dir / "MCYT" / "syntTO" / "001g02.csv", 4)
    _write_csv(
        reference_dir / "projected3Dsignatures" / "realTO" / "01_01.csv",
        5,
    )
    _write_csv(
        candidate_dir / "projected3Dsignatures" / "syntTO" / "01_02.csv",
        6,
    )
    _write_csv(
        reference_dir / "raton" / "realTO" / "weiziang-104_201281516847_logfile.csv",
        7,
    )
    _write_csv(candidate_dir / "raton" / "syntTO" / "weiziang-008.csv", 8)

    groups = Reporting.discover_reporting_sample_groups(
        str(reference_dir),
        str(candidate_dir),
    )

    assert [(group.dataset_key, group.class_key) for group in groups] == [
        ("MCYT", "001"),
        ("MobileTouchDB", "mayA"),
        ("projected3Dsignatures", "01"),
        ("raton", "weiziang"),
    ]


def test_class_scheme_can_force_legacy_filename_label_behavior(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "MobileTouchDB" / "user1-Sesion1-u1_s1_mayA.csv", 1)
    _write_csv(candidate_dir / "MobileTouchDB" / "user2-Sesion1-u2_s1_mayA.csv", 2)

    groups = Reporting.discover_reporting_sample_groups(
        str(reference_dir),
        str(candidate_dir),
        class_scheme="filename-label",
    )

    assert [(group.dataset_key, group.class_key) for group in groups] == [
        ("MobileTouchDB", "Sesion1")
    ]


def test_single_file_input_uses_parent_dataset_hint_for_auto_class_scheme(tmp_path):
    csv_file = tmp_path / "1dollar" / "realTO" / "s01-arrow-fast-t01.csv"
    _write_csv(csv_file, 1)

    entries = Reporting.load_reporting_entries(str(csv_file))

    assert [(entry.dataset_key, entry.class_key) for entry in entries] == [
        (".", "arrow")
    ]


def test_parent_dir_class_scheme_can_be_used_with_filename_dataset_grouping(tmp_path):
    assert (
        Reporting._class_key_for_relative_path(
            "dataset-a/arrow/r1.csv",
            "parent-dir",
            None,
        )
        == "arrow"
    )


def test_parent_dir_grouping_tracks_dataset_and_class(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "root.csv", 0)
    _write_csv(candidate_dir / "root.csv", 0)
    _write_csv(reference_dir / "dataset-a" / "arrow" / "r1.csv", 1)
    _write_csv(reference_dir / "circle" / "r1.csv", 2)
    _write_csv(candidate_dir / "dataset-a" / "arrow" / "c1.csv", 3)

    groups = Reporting.discover_reporting_sample_groups(
        str(reference_dir),
        str(candidate_dir),
        group_by="parent-dir",
    )

    assert [(group.dataset_key, group.class_key) for group in groups] == [
        (".", "."),
        (".", "circle"),
        ("dataset-a", "arrow"),
    ]


def test_sampling_rejects_invalid_limit_and_grouping(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s01-arrow-01.csv", 1)
    _write_csv(candidate_dir / "g01-arrow-01.csv", 2)

    with pytest.raises(ValueError, match="Sample limit"):
        Reporting.select_reporting_samples(
            str(reference_dir),
            str(candidate_dir),
            sample_limit=0,
        )

    with pytest.raises(ValueError, match="Invalid group-by mode"):
        Reporting.select_reporting_samples(
            str(reference_dir),
            str(candidate_dir),
            group_by="broken",
        )

    with pytest.raises(ValueError, match="Invalid class scheme"):
        Reporting.select_reporting_samples(
            str(reference_dir),
            str(candidate_dir),
            class_scheme="broken",
        )


def test_auto_class_scheme_reports_unsupported_filenames(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "plain.csv", 1)
    _write_csv(candidate_dir / "plain.csv", 2)

    assert Reporting._raton_class_key("plain.csv") is None

    with pytest.raises(ValueError, match="Cannot derive class label"):
        Reporting.select_reporting_samples(
            str(reference_dir),
            str(candidate_dir),
        )


def test_build_raw_comparison_tables_preserves_pair_directions(monkeypatch):
    group = Reporting.ReportingSampleGroup(
        dataset_key="dataset-a",
        class_key="arrow",
        reference_entries=(
            Reporting.ReportingEntry("r1.csv", "/tmp/r1.csv", "dataset-a", "arrow", ["r1"]),
            Reporting.ReportingEntry("r2.csv", "/tmp/r2.csv", "dataset-a", "arrow", ["r2"]),
        ),
        candidate_entries=(
            Reporting.ReportingEntry("c1.csv", "/tmp/c1.csv", "dataset-a", "arrow", ["c1"]),
        ),
    )
    values_by_points = {
        ("r1", "r2"): 10.0,
        ("r2", "r1"): 20.0,
        ("r1", "c1"): 30.0,
        ("r2", "c1"): 40.0,
    }

    def fake_compute_pair_metrics(reference_points, candidate_points, *args, **kwargs):
        return {
            "shapeError": values_by_points[
                (reference_points[0], candidate_points[0])
            ]
        }

    monkeypatch.setattr(
        ReportingRaw,
        "compute_pair_metrics_from_points",
        fake_compute_pair_metrics,
    )
    monkeypatch.setattr(ReportingRaw, "sampling_rate_for_sets", lambda point_sets, rate: 24)

    payload = Reporting.build_raw_comparison_tables(
        [group],
        metric_names=["shapeError"],
        reference_source_name="human",
        candidate_source_name="generated",
    )

    baseline_rows = payload["rawBaselinePairs"]
    candidate_rows = payload["rawCandidatePairs"]
    assert [row["direction"] for row in baseline_rows] == ["forward", "backward"]
    assert [row["value"] for row in baseline_rows] == [10.0, 20.0]
    assert [row["value"] for row in candidate_rows] == [30.0, 40.0]
    assert baseline_rows[0]["referenceSourceRole"] == "reference"
    assert baseline_rows[0]["candidateSourceRole"] == "reference"
    assert candidate_rows[0]["candidateSourceRole"] == "candidate"
    assert payload["metadata"]["baselineRowCount"] == 2
    assert payload["metadata"]["candidateRowCount"] == 2
    assert payload["metadata"]["alignmentName"] == "chronological"
    assert baseline_rows[0]["alignmentName"] == "chronological"


def test_export_raw_comparison_tables_writes_sampled_and_full_outputs(
    monkeypatch,
    tmp_path,
):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    for index in range(3):
        _write_csv(reference_dir / f"r{index:02d}-arrow-01.csv", index)
        _write_csv(candidate_dir / f"c{index:02d}-arrow-01.csv", index + 10)

    monkeypatch.setattr(
        ReportingRaw,
        "compute_pair_metrics_from_points",
        lambda *args, **kwargs: {"shapeError": 1.25},
    )
    load_calls = []
    original_load_reporting_entries = Reporting.load_reporting_entries

    def counted_load_reporting_entries(*args, **kwargs):
        load_calls.append(args[0])
        return original_load_reporting_entries(*args, **kwargs)

    monkeypatch.setattr(
        Reporting,
        "load_reporting_entries",
        counted_load_reporting_entries,
    )

    sampled_output_dir = tmp_path / "sampled-output"
    sampled_payload = Reporting.export_raw_comparison_tables(
        str(reference_dir),
        str(candidate_dir),
        output_dir=sampled_output_dir,
        sample_limit=2,
        metric_names=["shapeError"],
    )

    assert sampled_payload["metadata"]["effectiveSampleLimit"] == 2
    assert sampled_payload["metadata"]["baselineRowCount"] == 2
    assert sampled_payload["metadata"]["candidateRowCount"] == 4
    assert (sampled_output_dir / "raw_baseline_pairs.csv").exists()
    assert (sampled_output_dir / "raw_candidate_pairs.csv").exists()
    assert (
        sampled_output_dir / "raw_baseline_pairs.csv"
    ).read_text(encoding="utf-8").splitlines()[0] == ",".join(
        Reporting.RAW_COMPARISON_COLUMNS
    )

    full_payload = Reporting.export_raw_comparison_tables(
        str(reference_dir),
        str(candidate_dir),
        sample_limit=None,
        metric_names=["shapeError"],
    )

    assert full_payload["metadata"]["sampleLimit"] is None
    assert full_payload["metadata"]["effectiveSampleLimit"] is None
    assert full_payload["metadata"]["baselineRowCount"] == 6
    assert full_payload["metadata"]["candidateRowCount"] == 9
    assert load_calls == [
        str(reference_dir),
        str(candidate_dir),
        str(reference_dir),
        str(candidate_dir),
    ]
