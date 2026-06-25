from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ._common import load_csv_entries
from .dataset_discovery import (
    CLASS_SCHEME_AUTO,
    CLASS_SCHEME_FILENAME_LABEL,
    CLASS_SCHEME_PARENT_DIR,
    CLASS_SCHEMES,
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    GROUP_BY_PARENT_DIR,
    SIGNATURE_DATASETS,
    SOURCE_FOLDER_NAMES,
    auto_class_key,
    class_key_for_relative_path,
    dataset_and_class_for_relative_path,
    dataset_key_from_parts,
    dataset_name_for_class,
    filename_label_class_key,
    input_dataset_hint,
    mobiletouchdb_class_key,
    normalize_class_scheme,
    normalize_group_by,
    parent_dir_dataset_and_class,
    parent_dir_dataset_and_class_from_parts,
    raton_class_key,
    safe_filename_label_class_key,
    signature_class_key,
    strip_source_folder_parts,
)


REPORTING_MODE = "report"
DEFAULT_SAMPLE_LIMIT = 16
REFERENCE_SOURCE = "reference"
CANDIDATE_SOURCE = "candidate"
DEFAULT_REFERENCE_SOURCE_NAME = "human"
DEFAULT_CANDIDATE_SOURCE_NAME = "generated"
STABLE_SAMPLING_MODE = "stable"
SEEDED_RANDOM_SAMPLING_MODE = "seeded-random"


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
    return normalize_class_scheme(class_scheme)


def _normalize_group_by(group_by: str | None) -> str:
    return normalize_group_by(group_by)


def _strip_source_folder_parts(parts: Sequence[str]) -> Tuple[str, ...]:
    return strip_source_folder_parts(parts)


def _dataset_key_from_parts(parts: Sequence[str]) -> str:
    return dataset_key_from_parts(parts)


def _input_dataset_hint(input_path: str) -> str | None:
    return input_dataset_hint(input_path)


def _parent_dir_dataset_and_class_from_parts(
    parent_parts: Sequence[str],
) -> Tuple[str, str]:
    return parent_dir_dataset_and_class_from_parts(parent_parts)


def _parent_dir_dataset_and_class(relative_csv_path: str) -> Tuple[str, str]:
    return parent_dir_dataset_and_class(relative_csv_path)


def _mobiletouchdb_class_key(filename: str) -> str | None:
    return mobiletouchdb_class_key(filename)


def _signature_class_key(filename: str) -> str | None:
    return signature_class_key(filename)


def _raton_class_key(filename: str) -> str | None:
    return raton_class_key(filename)


def _dataset_name_for_class(dataset_key: str, dataset_hint: str | None) -> str | None:
    return dataset_name_for_class(dataset_key, dataset_hint)


def _safe_filename_label_class_key(relative_csv_path: str) -> str | None:
    return safe_filename_label_class_key(relative_csv_path)


def _auto_class_key(relative_csv_path: str, dataset_name: str | None) -> str:
    return auto_class_key(relative_csv_path, dataset_name)


def _class_key_for_relative_path(
    relative_csv_path: str,
    class_scheme: str,
    dataset_name: str | None,
) -> str:
    return class_key_for_relative_path(
        relative_csv_path,
        GROUP_BY_FILENAME_LABEL,
        class_scheme,
        dataset_name,
    )


def _dataset_and_class_for_relative_path(
    relative_csv_path: str,
    group_by: str,
    class_scheme: str = CLASS_SCHEME_AUTO,
    dataset_hint: str | None = None,
) -> Tuple[str, str]:
    return dataset_and_class_for_relative_path(
        relative_csv_path,
        group_by,
        class_scheme,
        dataset_hint,
    )


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
    sample_limit: int | None,
    random_seed: int | str | None,
    source: str,
    dataset_key: str,
    class_key: str,
) -> Tuple[ReportingEntry, ...]:
    sorted_entries = tuple(sorted(entries, key=lambda entry: entry.key))
    if sample_limit is None or len(sorted_entries) <= sample_limit:
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
    sample_limit: int | None = DEFAULT_SAMPLE_LIMIT,
    random_seed: int | str | None = None,
) -> Tuple[ReportingSampleGroup, ...]:
    normalized_group_by = _normalize_group_by(group_by)
    normalized_class_scheme = _normalize_class_scheme(class_scheme)
    selected_limit = None if sample_limit is None else _validate_sample_limit(sample_limit)
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
    reference_source_name: str = DEFAULT_REFERENCE_SOURCE_NAME,
    candidate_source_name: str = DEFAULT_CANDIDATE_SOURCE_NAME,
) -> List[Dict[str, object]]:
    manifest = []
    for group in sample_groups:
        for source_role, source_name, entries in [
            (REFERENCE_SOURCE, reference_source_name, group.reference_entries),
            (CANDIDATE_SOURCE, candidate_source_name, group.candidate_entries),
        ]:
            for index, entry in enumerate(entries, start=1):
                manifest.append(
                    {
                        "datasetKey": group.dataset_key,
                        "classKey": group.class_key,
                        "source": source_name,
                        "sourceRole": source_role,
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
    reference_source_name: str = DEFAULT_REFERENCE_SOURCE_NAME,
    candidate_source_name: str = DEFAULT_CANDIDATE_SOURCE_NAME,
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
            "sourceNames": {
                REFERENCE_SOURCE: reference_source_name,
                CANDIDATE_SOURCE: candidate_source_name,
            },
        },
        "samples": build_sample_manifest(
            sample_groups,
            reference_source_name=reference_source_name,
            candidate_source_name=candidate_source_name,
        ),
    }


from .reporting_raw import (  # noqa: E402
    BASELINE_PAIR_TYPE,
    CANDIDATE_PAIR_TYPE,
    RAW_BASELINE_PAIRS_FILENAME,
    RAW_CANDIDATE_PAIRS_FILENAME,
    RAW_COMPARISON_COLUMNS,
    build_raw_comparison_tables,
    export_raw_comparison_tables,
    format_raw_comparison_rows_csv,
    write_raw_comparison_exports,
)


__all__ = [
    "CANDIDATE_SOURCE",
    "CLASS_SCHEME_AUTO",
    "CLASS_SCHEME_FILENAME_LABEL",
    "CLASS_SCHEME_PARENT_DIR",
    "CLASS_SCHEMES",
    "DEFAULT_CANDIDATE_SOURCE_NAME",
    "DEFAULT_REFERENCE_SOURCE_NAME",
    "DEFAULT_SAMPLE_LIMIT",
    "BASELINE_PAIR_TYPE",
    "CANDIDATE_PAIR_TYPE",
    "GROUP_BY_FILENAME_LABEL",
    "GROUP_BY_MODES",
    "GROUP_BY_PARENT_DIR",
    "RAW_BASELINE_PAIRS_FILENAME",
    "RAW_CANDIDATE_PAIRS_FILENAME",
    "RAW_COMPARISON_COLUMNS",
    "REFERENCE_SOURCE",
    "REPORTING_MODE",
    "ReportingEntry",
    "ReportingSampleGroup",
    "SOURCE_FOLDER_NAMES",
    "build_sample_manifest",
    "build_raw_comparison_tables",
    "discover_reporting_sample_groups",
    "export_raw_comparison_tables",
    "format_raw_comparison_rows_csv",
    "load_reporting_entries",
    "select_reporting_samples",
    "write_raw_comparison_exports",
]
