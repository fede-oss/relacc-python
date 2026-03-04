from pathlib import Path

import pytest

from relacc.pipeline import pairwise as Pairwise


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


def _sample_rows_with_stroke_count(stroke_count: int, offset=0):
    rows = ["stroke_id x y time is_writing"]
    time = 0
    x = 10 + offset
    for stroke_id in range(stroke_count):
        rows.append(f"{stroke_id} {x} 20 {time} 1")
        time += 10
        x += 2
        rows.append(f"{stroke_id} {x} 22 {time} 1")
        time += 10
        x += 2
    return rows


def _metric_dict(value=0.0):
    return {name: value for name in Pairwise.METRIC_NAMES}


def test_list_csv_files_modes(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        Pairwise._list_csv_files(missing)

    txt = tmp_path / "sample.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected a \.csv file"):
        Pairwise._list_csv_files(txt)

    one = tmp_path / "single.csv"
    _write_csv(one, _sample_rows())
    one_map = Pairwise._list_csv_files(one)
    assert one_map == {"single.csv": one}

    root = tmp_path / "root"
    _write_csv(root / "a.csv", _sample_rows())
    _write_csv(root / "nested" / "b.csv", _sample_rows(1))
    listed = Pairwise._list_csv_files(root)
    assert sorted(listed.keys()) == ["a.csv", "nested/b.csv"]


def test_discover_pairs_file_and_path_shape_validation(tmp_path):
    reference = tmp_path / "ref.csv"
    candidate = tmp_path / "cand.csv"
    _write_csv(reference, _sample_rows())
    _write_csv(candidate, _sample_rows(1))

    pairs, missing_candidate, missing_reference = Pairwise.discover_pairs(
        str(reference), str(candidate)
    )
    assert len(pairs) == 1
    assert pairs[0].key == "ref"
    assert missing_candidate == []
    assert missing_reference == []

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ValueError, match="Both inputs must be files or both must be directories"):
        Pairwise.discover_pairs(str(reference), str(directory))


def test_discover_pairs_directory_modes(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "a.csv", _sample_rows())
    _write_csv(reference_dir / "nested" / "c.csv", _sample_rows(1))
    _write_csv(reference_dir / "only_ref.csv", _sample_rows(2))

    _write_csv(candidate_dir / "a.csv", _sample_rows(3))
    _write_csv(candidate_dir / "nested" / "c.csv", _sample_rows(4))
    _write_csv(candidate_dir / "only_cand.csv", _sample_rows(5))

    with pytest.raises(ValueError, match="Unmatched files found"):
        Pairwise.discover_pairs(str(reference_dir), str(candidate_dir), strict=True)

    pairs, missing_candidate, missing_reference = Pairwise.discover_pairs(
        str(reference_dir), str(candidate_dir), strict=False
    )
    assert [p.key for p in pairs] == ["a", "nested/c"]
    assert missing_candidate == ["only_ref.csv"]
    assert missing_reference == ["only_cand.csv"]


def test_discover_pairs_no_matching_files(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "a.csv", _sample_rows())
    _write_csv(candidate_dir / "b.csv", _sample_rows(1))

    with pytest.raises(ValueError, match="No matching CSV files found"):
        Pairwise.discover_pairs(str(reference_dir), str(candidate_dir))


def test_read_points_and_sampling_rate(tmp_path):
    csv_file = tmp_path / "gesture.csv"
    _write_csv(csv_file, _sample_rows())
    points = Pairwise._read_points(str(csv_file))
    assert len(points) == 4

    empty = tmp_path / "empty.csv"
    _write_csv(empty, ["stroke_id x y time is_writing"])
    with pytest.raises(ValueError, match="No points parsed"):
        Pairwise._read_points(str(empty))

    explicit_rate = Pairwise._sampling_rate(points, points, 9)
    assert explicit_rate == 9

    with pytest.raises(ValueError, match="Sampling rate must be >= 1"):
        Pairwise._sampling_rate(points, points, 0)

    smart_rate = Pairwise._sampling_rate(points, points, None)
    assert smart_rate == 24


def test_sampling_rate_for_sets_uses_max_strokes_when_rate_missing(tmp_path):
    csv_file = tmp_path / "many_strokes.csv"
    _write_csv(csv_file, _sample_rows_with_stroke_count(5))
    points = Pairwise._read_points(str(csv_file))

    inferred_rate = Pairwise._sampling_rate_for_sets([points], None)
    assert inferred_rate == 24


def test_compute_compare_and_run_pairwise(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows())
    _write_csv(cand, _sample_rows(1))

    pair = Pairwise.PairSpec(key="pair-1", reference_file=str(ref), candidate_file=str(cand))
    row = Pairwise.compare_pair(
        pair,
        label=None,
        rate=8,
        alignment_type=1,
        summary_shape="centroid",
        popular_shape=False,
        round_precision=4,
    )

    assert row["pairKey"] == "pair-1"
    assert row["label"] == "pair-1"
    assert row["rate"] == 8
    assert row["summary"] == "centroid"
    for metric_name in Pairwise.METRIC_NAMES:
        assert metric_name in row

    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "s1.csv", _sample_rows())
    _write_csv(reference_dir / "s2.csv", _sample_rows(2))
    _write_csv(candidate_dir / "s1.csv", _sample_rows(1))
    _write_csv(candidate_dir / "extra.csv", _sample_rows(3))

    payload = Pairwise.run_pairwise_comparison(
        str(reference_dir),
        str(candidate_dir),
        label="gesture",
        rate=7,
        alignment_type=1,
        summary_shape=None,
        popular_shape=False,
        strict=False,
        round_precision=2,
    )
    assert payload["metadata"]["comparisonMode"] == "direct"
    assert payload["metadata"]["pairCount"] == 1
    assert payload["metadata"]["referenceCount"] == 1
    assert payload["metadata"]["missingInCandidate"] == ["s2.csv"]
    assert payload["metadata"]["missingInReference"] == ["extra.csv"]
    assert payload["metadata"]["rate"] == 7
    assert payload["metadata"]["label"] == "gesture"
    assert payload["metadata"]["roundPrecision"] == 2
    assert payload["pairs"][0]["label"] == "gesture"
    assert payload["pairs"][0]["mode"] == "direct"
    assert payload["pairs"][0]["referenceCount"] == 1


def test_run_pairwise_summary_mode_compares_all_candidates(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "ref_a.csv", _sample_rows(0))
    _write_csv(reference_dir / "nested" / "ref_b.csv", _sample_rows(2))

    _write_csv(candidate_dir / "cand_1.csv", _sample_rows(1))
    _write_csv(candidate_dir / "nested" / "cand_2.csv", _sample_rows(3))

    payload = Pairwise.run_pairwise_comparison(
        str(reference_dir),
        str(candidate_dir),
        summary_shape="centroid",
        popular_shape=True,
        strict=True,
        comparison_mode="summary",
    )

    assert payload["metadata"]["comparisonMode"] == "summary"
    assert payload["metadata"]["pairCount"] == 2
    assert payload["metadata"]["referenceCount"] == 2
    assert payload["metadata"]["missingInCandidate"] == []
    assert payload["metadata"]["missingInReference"] == []
    assert all(row["mode"] == "summary" for row in payload["pairs"])
    assert all(row["referenceCount"] == 2 for row in payload["pairs"])


def test_run_pairwise_summary_mode_auto_rate_uses_reference_only(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    _write_csv(reference_dir / "ref_a.csv", _sample_rows_with_stroke_count(2, 0))
    _write_csv(reference_dir / "ref_b.csv", _sample_rows_with_stroke_count(2, 3))

    # Candidate has many strokes; summary-mode auto-rate should not depend on this.
    _write_csv(candidate_dir / "cand_many_strokes.csv", _sample_rows_with_stroke_count(5, 1))

    payload = Pairwise.run_pairwise_comparison(
        str(reference_dir),
        str(candidate_dir),
        comparison_mode="summary",
    )

    assert payload["metadata"]["comparisonMode"] == "summary"
    assert payload["pairs"][0]["rate"] == 24


def test_run_pairwise_invalid_mode_and_summary_validation(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    _write_csv(ref, _sample_rows())
    _write_csv(cand, _sample_rows(1))

    with pytest.raises(ValueError, match="Invalid comparison mode"):
        Pairwise.run_pairwise_comparison(str(ref), str(cand), comparison_mode="broken")

    with pytest.raises(ValueError, match="Invalid summary shape"):
        Pairwise.run_pairwise_comparison(str(ref), str(cand), summary_shape="average")


def test_run_pairwise_summary_mode_requires_reference_csvs(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    _write_csv(candidate_dir / "cand.csv", _sample_rows())

    with pytest.raises(ValueError, match="No reference CSV files found"):
        Pairwise.run_pairwise_comparison(
            str(reference_dir),
            str(candidate_dir),
            comparison_mode="summary",
        )


def test_run_pairwise_summary_mode_requires_candidate_csvs(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    _write_csv(reference_dir / "ref.csv", _sample_rows())
    candidate_dir.mkdir()

    with pytest.raises(ValueError, match="No candidate CSV files found"):
        Pairwise.run_pairwise_comparison(
            str(reference_dir),
            str(candidate_dir),
            comparison_mode="summary",
        )


def test_format_pair_rows_csv_with_escaping_and_missing_values():
    row = {
        "pairKey": "k1",
        "label": "label,with,comma",
        "referenceFile": 'ref"x.csv',
        "candidateFile": "cand.csv",
        "mode": "direct",
        "referenceCount": 1,
        "rate": 24,
        "alignment": 1,
        "summary": None,
        "popular": False,
    }
    row.update(_metric_dict(0.0))

    output = Pairwise.format_pair_rows_csv([row])
    lines = output.splitlines()
    assert lines[0].startswith("pairKey,label,referenceFile")
    assert '"label,with,comma"' in lines[1]
    assert '"ref""x.csv"' in lines[1]
    assert ",," in lines[1]
