from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ._common import load_csv_entries
from .distribution import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    GROUP_BY_PARENT_DIR,
    _filename_label_class_key,
    _normalize_group_by,
)


REPORTING_MODE = "report"
DEFAULT_SAMPLE_LIMIT = 16
REFERENCE_SOURCE = "reference"
CANDIDATE_SOURCE = "candidate"
STABLE_SAMPLING_MODE = "stable"
SEEDED_RANDOM_SAMPLING_MODE = "seeded-random"
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
class ReportingEntry:
    key: str
    path: str
    dataset_key: str
    class_key: str
    points: list


@dataclass(frozen=True)
class ReportingSampleGroup:
    dataset_key: str
    class_key: str
    reference_entries: Tuple[ReportingEntry, ...]
    candidate_entries: Tuple[ReportingEntry, ...]


def _validate_sample_limit(sample_limit: int) -> int:
    parsed_limit = int(sample_limit)
    if parsed_limit < 1:
        raise ValueError("Sample limit must be >= 1.")
    return parsed_limit


def _normalize_class_scheme(class_scheme: str | None) -> str:
    scheme = (class_scheme or CLASS_SCHEME_AUTO).strip().lower()
    if scheme not in CLASS_SCHEMES:
        raise ValueError(
            "Invalid class scheme (%s). Supported values: auto, filename-label, parent-dir."
            % class_scheme
        )
    return scheme


def _strip_source_folder_parts(parts: Sequence[str]) -> Tuple[str, ...]:
    return tuple(part for part in parts if part not in SOURCE_FOLDER_NAMES)


def _dataset_key_from_parts(parts: Sequence[str]) -> str:
    if len(parts) == 0:
        return "."
    return "/".join(parts)


def _input_dataset_hint(input_path: str) -> str | None:
    path = Path(input_path)
    if path.is_file():
        path = path.parent
    if path.name in SOURCE_FOLDER_NAMES:
        return path.parent.name
    return path.name or None


def _parent_dir_dataset_and_class_from_parts(
    parent_parts: Sequence[str],
) -> Tuple[str, str]:
    if len(parent_parts) == 0:
        return ".", "."
    if len(parent_parts) == 1:
        return ".", parent_parts[0]
    return parent_parts[0], "/".join(parent_parts[1:])


def _parent_dir_dataset_and_class(relative_csv_path: str) -> Tuple[str, str]:
    parts = relative_csv_path.split("/")
    parent_parts = _strip_source_folder_parts(parts[:-1])
    return _parent_dir_dataset_and_class_from_parts(parent_parts)


def _mobiletouchdb_class_key(filename: str) -> str | None:
    match = re.search(r"_s\d+_([^_]+)$", Path(filename).stem)
    if match:
        return match.group(1)
    return None


def _signature_class_key(filename: str) -> str | None:
    stem = Path(filename).stem
    match = re.match(r"^(?:W\d+_)?(\d+)[gv]\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(\d+)_\d+$", stem)
    if match:
        return match.group(1)
    return None


def _raton_class_key(filename: str) -> str | None:
    stem = Path(filename).stem
    if "-" not in stem:
        return None
    return stem.split("-", 1)[0]


def _dataset_name_for_class(dataset_key: str, dataset_hint: str | None) -> str | None:
    if dataset_key != ".":
        return dataset_key.split("/", 1)[0]
    return dataset_hint


def _safe_filename_label_class_key(relative_csv_path: str) -> str | None:
    try:
        return _filename_label_class_key(relative_csv_path)
    except ValueError:
        return None


def _auto_class_key(relative_csv_path: str, dataset_name: str | None) -> str:
    filename = relative_csv_path.rsplit("/", 1)[-1]
    normalized_dataset = (dataset_name or "").lower()

    if normalized_dataset == "mobiletouchdb":
        class_key = _mobiletouchdb_class_key(filename)
        if class_key is not None:
            return class_key

    if normalized_dataset == "raton":
        class_key = _raton_class_key(filename)
        if class_key is not None:
            return class_key

    if normalized_dataset in SIGNATURE_DATASETS:
        class_key = _signature_class_key(filename)
        if class_key is not None:
            return class_key

    for class_key in [
        _mobiletouchdb_class_key(filename),
        _signature_class_key(filename),
        _safe_filename_label_class_key(relative_csv_path),
    ]:
        if class_key is not None:
            return class_key

    raise ValueError(
        "Cannot derive class label from filename (%s). Use class_scheme='parent-dir' "
        "for directory-structured classes or provide files with a supported naming pattern."
        % relative_csv_path
    )


def _class_key_for_relative_path(
    relative_csv_path: str,
    class_scheme: str,
    dataset_name: str | None,
) -> str:
    if class_scheme == CLASS_SCHEME_FILENAME_LABEL:
        return _filename_label_class_key(relative_csv_path)
    if class_scheme == CLASS_SCHEME_AUTO:
        return _auto_class_key(relative_csv_path, dataset_name)
    return _parent_dir_dataset_and_class(relative_csv_path)[1]


def _dataset_and_class_for_relative_path(
    relative_csv_path: str,
    group_by: str,
    class_scheme: str = CLASS_SCHEME_AUTO,
    dataset_hint: str | None = None,
) -> Tuple[str, str]:
    parts = relative_csv_path.split("/")
    parent_parts = _strip_source_folder_parts(parts[:-1])

    if group_by == GROUP_BY_FILENAME_LABEL:
        dataset_key = _dataset_key_from_parts(parent_parts)
        dataset_name = _dataset_name_for_class(dataset_key, dataset_hint)
        return (
            dataset_key,
            _class_key_for_relative_path(
                relative_csv_path,
                class_scheme,
                dataset_name,
            ),
        )
    return _parent_dir_dataset_and_class_from_parts(parent_parts)


def load_reporting_entries(
    input_path: str,
    group_by: str = GROUP_BY_FILENAME_LABEL,
    class_scheme: str = CLASS_SCHEME_AUTO,
) -> Tuple[ReportingEntry, ...]:
    normalized_group_by = _normalize_group_by(group_by)
    normalized_class_scheme = _normalize_class_scheme(class_scheme)
    dataset_hint = _input_dataset_hint(input_path)
    entries = []
    for key, path, points in load_csv_entries(input_path):
        dataset_key, class_key = _dataset_and_class_for_relative_path(
            key,
            normalized_group_by,
            normalized_class_scheme,
            dataset_hint,
        )
        entries.append(
            ReportingEntry(
                key=key,
                path=path,
                dataset_key=dataset_key,
                class_key=class_key,
                points=points,
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (entry.dataset_key, entry.class_key, entry.key),
        )
    )


def _group_entries_by_dataset_class(
    entries: Sequence[ReportingEntry],
) -> Dict[Tuple[str, str], List[ReportingEntry]]:
    grouped: Dict[Tuple[str, str], List[ReportingEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.dataset_key, entry.class_key), []).append(entry)
    return grouped


def _seed_for_group(
    random_seed: int | str,
    source: str,
    dataset_key: str,
    class_key: str,
) -> int:
    digest = hashlib.sha256(
        ("%s:%s:%s:%s" % (random_seed, source, dataset_key, class_key)).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _select_entries(
    entries: Sequence[ReportingEntry],
    sample_limit: int,
    random_seed: int | str | None,
    source: str,
    dataset_key: str,
    class_key: str,
) -> Tuple[ReportingEntry, ...]:
    sorted_entries = tuple(sorted(entries, key=lambda entry: entry.key))
    if len(sorted_entries) <= sample_limit:
        return sorted_entries

    if random_seed is None:
        return sorted_entries[:sample_limit]

    rng = random.Random(_seed_for_group(random_seed, source, dataset_key, class_key))
    sampled_entries = rng.sample(list(sorted_entries), sample_limit)
    return tuple(sorted(sampled_entries, key=lambda entry: entry.key))


def discover_reporting_sample_groups(
    reference_input: str,
    candidate_input: str,
    group_by: str = GROUP_BY_FILENAME_LABEL,
    class_scheme: str = CLASS_SCHEME_AUTO,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    random_seed: int | str | None = None,
) -> Tuple[ReportingSampleGroup, ...]:
    normalized_group_by = _normalize_group_by(group_by)
    normalized_class_scheme = _normalize_class_scheme(class_scheme)
    selected_limit = _validate_sample_limit(sample_limit)
    reference_groups = _group_entries_by_dataset_class(
        load_reporting_entries(
            reference_input,
            normalized_group_by,
            normalized_class_scheme,
        )
    )
    candidate_groups = _group_entries_by_dataset_class(
        load_reporting_entries(
            candidate_input,
            normalized_group_by,
            normalized_class_scheme,
        )
    )

    sample_groups: List[ReportingSampleGroup] = []
    for dataset_key, class_key in sorted(
        set(reference_groups.keys()) | set(candidate_groups.keys())
    ):
        reference_entries = _select_entries(
            reference_groups.get((dataset_key, class_key), []),
            selected_limit,
            random_seed,
            REFERENCE_SOURCE,
            dataset_key,
            class_key,
        )
        candidate_entries = _select_entries(
            candidate_groups.get((dataset_key, class_key), []),
            selected_limit,
            random_seed,
            CANDIDATE_SOURCE,
            dataset_key,
            class_key,
        )
        sample_groups.append(
            ReportingSampleGroup(
                dataset_key=dataset_key,
                class_key=class_key,
                reference_entries=reference_entries,
                candidate_entries=candidate_entries,
            )
        )

    return tuple(sample_groups)


def build_sample_manifest(
    sample_groups: Sequence[ReportingSampleGroup],
) -> List[Dict[str, object]]:
    manifest = []
    for group in sample_groups:
        for source, entries in [
            (REFERENCE_SOURCE, group.reference_entries),
            (CANDIDATE_SOURCE, group.candidate_entries),
        ]:
            for index, entry in enumerate(entries, start=1):
                manifest.append(
                    {
                        "datasetKey": group.dataset_key,
                        "classKey": group.class_key,
                        "source": source,
                        "sampleIndex": index,
                        "key": entry.key,
                        "file": entry.path,
                    }
                )
    return manifest


def select_reporting_samples(
    reference_input: str,
    candidate_input: str,
    group_by: str = GROUP_BY_FILENAME_LABEL,
    class_scheme: str = CLASS_SCHEME_AUTO,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    random_seed: int | str | None = None,
) -> Dict[str, object]:
    normalized_group_by = _normalize_group_by(group_by)
    normalized_class_scheme = _normalize_class_scheme(class_scheme)
    selected_limit = _validate_sample_limit(sample_limit)
    sample_groups = discover_reporting_sample_groups(
        reference_input,
        candidate_input,
        group_by=normalized_group_by,
        class_scheme=normalized_class_scheme,
        sample_limit=selected_limit,
        random_seed=random_seed,
    )
    return {
        "metadata": {
            "reportingMode": REPORTING_MODE,
            "groupBy": normalized_group_by,
            "classScheme": normalized_class_scheme,
            "sampleLimit": selected_limit,
            "samplingMode": (
                SEEDED_RANDOM_SAMPLING_MODE
                if random_seed is not None
                else STABLE_SAMPLING_MODE
            ),
            "randomSeed": random_seed,
            "groupCount": len(sample_groups),
        },
        "samples": build_sample_manifest(sample_groups),
    }


__all__ = [
    "CANDIDATE_SOURCE",
    "CLASS_SCHEME_AUTO",
    "CLASS_SCHEME_FILENAME_LABEL",
    "CLASS_SCHEME_PARENT_DIR",
    "CLASS_SCHEMES",
    "DEFAULT_SAMPLE_LIMIT",
    "GROUP_BY_FILENAME_LABEL",
    "GROUP_BY_MODES",
    "GROUP_BY_PARENT_DIR",
    "REFERENCE_SOURCE",
    "REPORTING_MODE",
    "ReportingEntry",
    "ReportingSampleGroup",
    "SOURCE_FOLDER_NAMES",
    "build_sample_manifest",
    "discover_reporting_sample_groups",
    "load_reporting_entries",
    "select_reporting_samples",
]
