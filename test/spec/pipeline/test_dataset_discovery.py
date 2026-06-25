from relacc.pipeline.dataset_discovery import (
    CLASS_SCHEME_AUTO,
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_PARENT_DIR,
    class_key_for_relative_path,
    dataset_and_class_for_relative_path,
    input_dataset_hint,
    match_relative_csv_keys,
    unmatched_keys,
)


def test_dataset_and_class_discovery_uses_dataset_specific_filename_schemes():
    assert dataset_and_class_for_relative_path(
        "MobileTouchDB/realTO/user1-Sesion1-u1_s1_mayA.csv",
        GROUP_BY_FILENAME_LABEL,
        CLASS_SCHEME_AUTO,
        None,
    ) == ("MobileTouchDB", "mayA")

    assert dataset_and_class_for_relative_path(
        "MCYT/realTO/001g01.csv",
        GROUP_BY_FILENAME_LABEL,
        CLASS_SCHEME_AUTO,
        None,
    ) == ("MCYT", "001")

    assert dataset_and_class_for_relative_path(
        "dataset-a/arrow/r1.csv",
        GROUP_BY_PARENT_DIR,
    ) == ("dataset-a", "arrow")


def test_input_dataset_hint_handles_source_folder_inputs(tmp_path):
    input_dir = tmp_path / "MobileTouchDB" / "realTO"
    input_dir.mkdir(parents=True)

    assert input_dataset_hint(str(input_dir)) == "MobileTouchDB"


def test_class_key_for_relative_path_preserves_distribution_defaults():
    assert (
        class_key_for_relative_path("s01-arrow-01.csv", GROUP_BY_FILENAME_LABEL)
        == "arrow"
    )
    assert (
        class_key_for_relative_path("arrow/sample.csv", GROUP_BY_PARENT_DIR)
        == "arrow"
    )


def test_match_relative_csv_keys_uses_exact_top_level_and_unique_filename_matches():
    matches = match_relative_csv_keys(
        [
            "exact.csv",
            "alpha/realTO/same.csv",
            "beta/realTO/same.csv",
            "reference-only-layout/a.csv",
            "unmatched-reference.csv",
        ],
        [
            "exact.csv",
            "alpha/same.csv",
            "beta/same.csv",
            "candidate-only-layout/a.csv",
            "unmatched-candidate.csv",
        ],
    )

    assert [(match.reference_key, match.candidate_key) for match in matches] == [
        ("exact.csv", "exact.csv"),
        ("alpha/realTO/same.csv", "alpha/same.csv"),
        ("beta/realTO/same.csv", "beta/same.csv"),
        ("reference-only-layout/a.csv", "candidate-only-layout/a.csv"),
    ]
    assert unmatched_keys(
        [
            "exact.csv",
            "alpha/realTO/same.csv",
            "beta/realTO/same.csv",
            "reference-only-layout/a.csv",
            "unmatched-reference.csv",
        ],
        [
            "exact.csv",
            "alpha/same.csv",
            "beta/same.csv",
            "candidate-only-layout/a.csv",
            "unmatched-candidate.csv",
        ],
        matches,
    ) == (["unmatched-reference.csv"], ["unmatched-candidate.csv"])
