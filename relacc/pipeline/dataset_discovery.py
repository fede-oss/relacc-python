from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from ._common import infer_label_from_filename


GROUP_BY_FILENAME_LABEL = "filename-label"
GROUP_BY_PARENT_DIR = "parent-dir"
GROUP_BY_MODES: Tuple[str, str] = (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_PARENT_DIR,
)

CLASS_SCHEME_AUTO = "auto"
CLASS_SCHEME_FILENAME_LABEL = "filename-label"
CLASS_SCHEME_PARENT_DIR = "parent-dir"
CLASS_SCHEMES: Tuple[str, str, str] = (
    CLASS_SCHEME_AUTO,
    CLASS_SCHEME_FILENAME_LABEL,
    CLASS_SCHEME_PARENT_DIR,
)

SOURCE_FOLDER_NAMES: Tuple[str, str, str] = ("realTO", "syntTO", "recoTO")
SIGNATURE_DATASETS = {
    "biosecurid",
    "ebiosig_finger",
    "ebiosig_stylus",
    "mcyt",
    "projected3dsignatures",
    "visual",
}


@dataclass(frozen=True)
class RelativeFileMatch:
    reference_key: str
    candidate_key: str


def normalize_group_by(group_by: str | None) -> str:
    mode = (group_by or GROUP_BY_FILENAME_LABEL).strip().lower()
    if mode not in GROUP_BY_MODES:
        raise ValueError(
            "Invalid group-by mode (%s). Supported values: filename-label, parent-dir."
            % group_by
        )
    return mode


def normalize_class_scheme(class_scheme: str | None) -> str:
    scheme = (class_scheme or CLASS_SCHEME_AUTO).strip().lower()
    if scheme not in CLASS_SCHEMES:
        raise ValueError(
            "Invalid class scheme (%s). Supported values: auto, filename-label, parent-dir."
            % class_scheme
        )
    return scheme


def filename_label_class_key(relative_csv_path: str) -> str:
    return infer_label_from_filename(relative_csv_path, "class")


def parent_dir_class_key(relative_csv_path: str) -> str:
    parts = relative_csv_path.split("/")
    if len(parts) == 1:
        return "."
    return "/".join(parts[:-1])


def strip_source_folder_parts(parts: Sequence[str]) -> Tuple[str, ...]:
    return tuple(part for part in parts if part not in SOURCE_FOLDER_NAMES)


def dataset_key_from_parts(parts: Sequence[str]) -> str:
    if len(parts) == 0:
        return "."
    return "/".join(parts)


def input_dataset_hint(input_path: str) -> str | None:
    path = Path(input_path)
    if path.is_file():
        path = path.parent
    if path.name in SOURCE_FOLDER_NAMES:
        return path.parent.name
    return path.name or None


def parent_dir_dataset_and_class_from_parts(
    parent_parts: Sequence[str],
) -> Tuple[str, str]:
    if len(parent_parts) == 0:
        return ".", "."
    if len(parent_parts) == 1:
        return ".", parent_parts[0]
    return parent_parts[0], "/".join(parent_parts[1:])


def parent_dir_dataset_and_class(relative_csv_path: str) -> Tuple[str, str]:
    parts = relative_csv_path.split("/")
    parent_parts = strip_source_folder_parts(parts[:-1])
    return parent_dir_dataset_and_class_from_parts(parent_parts)


def mobiletouchdb_class_key(filename: str) -> str | None:
    match = re.search(r"_s\d+_([^_]+)$", Path(filename).stem)
    if match:
        return match.group(1)
    return None


def signature_class_key(filename: str) -> str | None:
    stem = Path(filename).stem
    match = re.match(r"^(?:W\d+_)?(\d+)[gv]\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(\d+)_\d+$", stem)
    if match:
        return match.group(1)
    return None


def raton_class_key(filename: str) -> str | None:
    stem = Path(filename).stem
    if "-" not in stem:
        return None
    return stem.split("-", 1)[0]


def dataset_name_for_class(dataset_key: str, dataset_hint: str | None) -> str | None:
    if dataset_key != ".":
        return dataset_key.split("/", 1)[0]
    return dataset_hint


def safe_filename_label_class_key(relative_csv_path: str) -> str | None:
    try:
        return filename_label_class_key(relative_csv_path)
    except ValueError:
        return None


def auto_class_key(relative_csv_path: str, dataset_name: str | None) -> str:
    filename = relative_csv_path.rsplit("/", 1)[-1]
    normalized_dataset = (dataset_name or "").lower()

    if normalized_dataset == "mobiletouchdb":
        class_key = mobiletouchdb_class_key(filename)
        if class_key is not None:
            return class_key

    if normalized_dataset == "raton":
        class_key = raton_class_key(filename)
        if class_key is not None:
            return class_key

    if normalized_dataset in SIGNATURE_DATASETS:
        class_key = signature_class_key(filename)
        if class_key is not None:
            return class_key

    for class_key in [
        mobiletouchdb_class_key(filename),
        signature_class_key(filename),
        safe_filename_label_class_key(relative_csv_path),
    ]:
        if class_key is not None:
            return class_key

    raise ValueError(
        "Cannot derive class label from filename (%s). Use class_scheme='parent-dir' "
        "for directory-structured classes or provide files with a supported naming pattern."
        % relative_csv_path
    )


def class_key_for_relative_path(
    relative_csv_path: str,
    group_by: str,
    class_scheme: str = CLASS_SCHEME_FILENAME_LABEL,
    dataset_name: str | None = None,
) -> str:
    normalized_group_by = normalize_group_by(group_by)
    if normalized_group_by == GROUP_BY_PARENT_DIR:
        return parent_dir_class_key(relative_csv_path)

    normalized_class_scheme = normalize_class_scheme(class_scheme)
    if normalized_class_scheme == CLASS_SCHEME_FILENAME_LABEL:
        return filename_label_class_key(relative_csv_path)
    if normalized_class_scheme == CLASS_SCHEME_AUTO:
        return auto_class_key(relative_csv_path, dataset_name)
    return parent_dir_dataset_and_class(relative_csv_path)[1]


def dataset_and_class_for_relative_path(
    relative_csv_path: str,
    group_by: str,
    class_scheme: str = CLASS_SCHEME_AUTO,
    dataset_hint: str | None = None,
) -> Tuple[str, str]:
    normalized_group_by = normalize_group_by(group_by)
    parts = relative_csv_path.split("/")
    parent_parts = strip_source_folder_parts(parts[:-1])

    if normalized_group_by == GROUP_BY_FILENAME_LABEL:
        dataset_key = dataset_key_from_parts(parent_parts)
        dataset_name = dataset_name_for_class(dataset_key, dataset_hint)
        return (
            dataset_key,
            class_key_for_relative_path(
                relative_csv_path,
                normalized_group_by,
                class_scheme,
                dataset_name,
            ),
        )
    return parent_dir_dataset_and_class_from_parts(parent_parts)


def top_level_filename_key(relative_csv_path: str) -> str:
    parts = relative_csv_path.split("/")
    if len(parts) < 2:
        return parts[0]
    return "/".join([parts[0], parts[-1]])


def filename_key(relative_csv_path: str) -> str:
    return relative_csv_path.split("/")[-1]


def unique_index(
    keys: Sequence[str],
    key_func: Callable[[str], str],
) -> Dict[str, str]:
    grouped: Dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(key_func(key), []).append(key)
    return {
        match_key: values[0]
        for match_key, values in grouped.items()
        if len(values) == 1
    }


def match_relative_csv_keys(
    reference_keys: Sequence[str],
    candidate_keys: Sequence[str],
) -> Tuple[RelativeFileMatch, ...]:
    ref_remaining = set(reference_keys)
    cand_remaining = set(candidate_keys)
    matches: list[RelativeFileMatch] = []

    exact_keys = sorted(ref_remaining & cand_remaining)
    for key in exact_keys:
        matches.append(RelativeFileMatch(reference_key=key, candidate_key=key))
    ref_remaining -= set(exact_keys)
    cand_remaining -= set(exact_keys)

    for key_func in [top_level_filename_key, filename_key]:
        ref_index = unique_index(sorted(ref_remaining), key_func)
        cand_index = unique_index(sorted(cand_remaining), key_func)
        matched_lookup_keys = sorted(set(ref_index.keys()) & set(cand_index.keys()))
        if not matched_lookup_keys:
            continue

        matched_refs = set()
        matched_cands = set()
        for match_key in matched_lookup_keys:
            ref_key = ref_index[match_key]
            cand_key = cand_index[match_key]
            matches.append(
                RelativeFileMatch(reference_key=ref_key, candidate_key=cand_key)
            )
            matched_refs.add(ref_key)
            matched_cands.add(cand_key)
        ref_remaining -= matched_refs
        cand_remaining -= matched_cands

    return tuple(matches)


def unmatched_keys(
    reference_keys: Sequence[str],
    candidate_keys: Sequence[str],
    matches: Sequence[RelativeFileMatch],
) -> Tuple[list[str], list[str]]:
    matched_ref_keys = {match.reference_key for match in matches}
    matched_cand_keys = {match.candidate_key for match in matches}
    return (
        sorted(set(reference_keys) - matched_ref_keys),
        sorted(set(candidate_keys) - matched_cand_keys),
    )
