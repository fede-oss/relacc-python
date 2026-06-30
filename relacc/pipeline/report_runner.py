from __future__ import annotations

import hashlib
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from relacc.distribution_metrics import DISTRIBUTION_METRIC_NAMES
from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import METRIC_NAMES
from relacc.pipeline._common import (
    list_csv_files,
    normalize_summary_shape,
    read_points,
)
from relacc.pipeline.dataset_discovery import (
    dataset_and_class_for_relative_path,
    input_dataset_hint,
    normalize_class_scheme,
    normalize_group_by,
)
from relacc.pipeline.reporting import ReportingEntry
from relacc.pipeline.report_compare import (
    _compare_class,
    _compare_direct_distribution_pairs_class,
    _compare_human_baseline_class,
)
from relacc.pipeline.report_exports import _write_csv, _write_json, _write_readme, write_combined_report_exports
from relacc.pipeline.report_schema import (
    BASELINE_COLUMNS,
    DISTRIBUTION_COLUMNS,
    PAIRWISE_COLUMNS,
    STATS_COLUMNS,
    statistical_contract_fields,
)
from relacc.pipeline.report_stats import (
    _lightweight_distribution_rows,
    _raw_distribution_outputs_from_rows,
    _summary_stats,
    _variant_label,
    build_combined_aggregate_summaries,
)
from relacc.utils.runlog import record_effective_config, verbosity_from_opt


DEFAULT_VARIANT_LABEL = "root"


def _int_cast(value):
    if value is None or value == "":
        return None
    return int(value)


def _optional_int_cast(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "none", "all", "full"):
        return None
    parsed = int(text)
    if parsed < 1:
        raise ValueError("Limits must be >= 1, or use 'none' for no limit.")
    return parsed


def _candidate_groups(source_root: Path, dataset_name: str):
    dataset_root = source_root / dataset_name
    if not dataset_root.exists():
        return []

    groups = []
    if sorted(dataset_root.glob("*.csv")):
        groups.append((dataset_root, None))

    for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if sorted(child.glob("*.csv")):
            groups.append((child, child.name))
    return groups


def _run_id(source_name: str, dataset_name: str, variant: str | None) -> str:
    if variant:
        return f"{source_name}/{dataset_name}/{variant}"
    return f"{source_name}/{dataset_name}"


def _variant_label(variant: str | None) -> str:
    return variant or DEFAULT_VARIANT_LABEL


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(seconds + 0.5)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _statistical_contract_fields() -> Dict[str, object]:
    return statistical_contract_fields()


class ProgressReporter:
    def __init__(self, total_candidates: int, total_runs: int, verbosity: int = 0):
        self.total_candidates = total_candidates
        self.total_runs = total_runs
        self.verbosity = verbosity
        self.completed_candidates = 0
        self.completed_runs = 0
        self.started_at = time.perf_counter()

    def _line(self, prefix: str) -> str:
        elapsed = time.perf_counter() - self.started_at
        if self.completed_candidates > 0:
            rate = self.completed_candidates / elapsed
            remaining = max(0, self.total_candidates - self.completed_candidates)
            eta = remaining / rate if rate > 0 else float("nan")
        else:
            rate = 0.0
            eta = float("nan")
        return (
            f"{prefix} progress={self.completed_candidates}/{self.total_candidates} "
            f"candidates runs={self.completed_runs}/{self.total_runs} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
            f"rate={rate:.2f}/s"
        )

    def start_run(self, run_id: str, candidate_count: int) -> None:
        if self.verbosity < 2:
            return
        print(
            self._line(
                f"starting {run_id} ({candidate_count} candidates);"
            ),
            flush=True,
        )

    def finish_class(self, run_id: str, class_key: str, candidate_count: int) -> None:
        self.completed_candidates += candidate_count
        if self.verbosity < 2:
            return
        print(
            self._line(
                f"finished {run_id} class={class_key} rows={candidate_count};"
            ),
            flush=True,
        )

    def skip_class(self, run_id: str, class_key: str, reason: str) -> None:
        if self.verbosity < 1:
            return
        print(f"skipped {run_id} class={class_key}: {reason}", flush=True)

    def finish_run(self, run_id: str, row_count: int) -> None:
        self.completed_runs += 1
        if self.verbosity < 2:
            return
        print(
            self._line(f"finished {run_id} rows={row_count};"),
            flush=True,
        )


def _safe_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "_"


def _seed_for_entries(
    random_seed: int | str,
    source_role: str,
    run_id: str,
    class_key: str,
) -> int:
    digest = hashlib.sha256(
        f"{random_seed}:{source_role}:{run_id}:{class_key}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _select_entries(
    entries: Sequence[ReportingEntry],
    sample_limit: int | None,
    random_seed: int | str | None,
    source_role: str,
    run_id: str,
    class_key: str,
) -> tuple[ReportingEntry, ...]:
    sorted_entries = tuple(sorted(entries, key=lambda entry: entry.key))
    if sample_limit is None or len(sorted_entries) <= sample_limit:
        return sorted_entries
    if random_seed is None:
        return sorted_entries[:sample_limit]

    rng = random.Random(
        _seed_for_entries(random_seed, source_role, run_id, class_key)
    )
    sampled_entries = rng.sample(list(sorted_entries), sample_limit)
    return tuple(sorted(sampled_entries, key=lambda entry: entry.key))


def _load_reporting_entries(
    input_path: Path,
    group_by: str,
    class_scheme: str,
    warnings: List[Dict[str, str]],
) -> List[ReportingEntry]:
    dataset_hint = input_dataset_hint(str(input_path))
    entries: List[ReportingEntry] = []
    for key, path in sorted(list_csv_files(input_path).items()):
        try:
            points = read_points(str(path))
            dataset_key, class_key = dataset_and_class_for_relative_path(
                key,
                group_by,
                class_scheme,
                dataset_hint,
            )
            entries.append(
                ReportingEntry(
                    key=key,
                    path=str(path),
                    dataset_key=dataset_key,
                    class_key=class_key,
                    points=points,
                )
            )
        except Exception as exc:
            warnings.append(
                {
                    "input": str(input_path),
                    "file": str(path),
                    "error": str(exc),
                }
            )
    return sorted(entries, key=lambda entry: (entry.class_key, entry.key))


def _entries_by_class(entries: Iterable[ReportingEntry]):
    grouped: Dict[str, List[ReportingEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.class_key].append(entry)
    return dict(grouped)


def _output_run_dir(output_root: Path, source_name: str, dataset_name: str, variant: str | None):
    run_dir = output_root / source_name / dataset_name
    if variant:
        run_dir = run_dir / variant
    return run_dir


def _count_run_candidates(
    candidate_input: Path,
    group_by: str,
    class_scheme: str,
    warnings: List[Dict[str, str]],
) -> tuple[int, Dict[str, int]]:
    dataset_hint = input_dataset_hint(str(candidate_input))
    counts: Dict[str, int] = defaultdict(int)
    for key, path in sorted(list_csv_files(candidate_input).items()):
        try:
            _, class_key = dataset_and_class_for_relative_path(
                key,
                group_by,
                class_scheme,
                dataset_hint,
            )
            counts[class_key] += 1
        except Exception as exc:
            warnings.append(
                {
                    "input": str(candidate_input),
                    "file": str(path),
                    "phase": "planning",
                    "error": str(exc),
                }
            )
    return sum(counts.values()), dict(counts)


def _run_reports(opt, output_root: Path, paths=None, metadata=None):
    datasets_root = Path(opt.datasets_root)
    humans_root = datasets_root / "humans"

    rate = _int_cast(opt.rate)
    round_precision = _int_cast(opt.round)
    alignment = PtAlignType.normalize(opt.alignment)
    dtw_window = _int_cast(opt.dtw_window)
    summary_shape = normalize_summary_shape(opt.summary)
    group_by = normalize_group_by(opt.group_by)
    class_scheme = normalize_class_scheme(opt.class_scheme)
    metric_names = METRIC_NAMES
    sample_limit_per_class = _optional_int_cast(opt.sample_limit_per_class)
    distribution_sample_limit_per_class = _optional_int_cast(
        opt.distribution_sample_limit_per_class
    )
    class_limit_per_run = _optional_int_cast(opt.class_limit_per_run)
    collect_raw_outputs = not bool(opt.skip_raw_jsonl)

    if dtw_window is not None and opt.exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")
    if not humans_root.exists():
        raise FileNotFoundError(f"Missing humans dataset root: {humans_root}")

    selected_datasets = (
        {item.strip() for item in opt.datasets.split(",") if item.strip()}
        if opt.datasets
        else None
    )
    selected_sources = (
        {item.strip() for item in opt.sources.split(",") if item.strip()}
        if opt.sources
        else None
    )
    record_effective_config(
        paths or {},
        metadata,
        {
            "datasetsRoot": str(datasets_root),
            "outputDir": str(output_root),
            "rate": rate,
            "roundPrecision": round_precision,
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": summary_shape,
            "popular": bool(opt.popular),
            "exactDtw": bool(opt.exact_dtw),
            "dtwWindow": dtw_window,
            "groupBy": group_by,
            "classScheme": class_scheme,
            "datasets": sorted(selected_datasets) if selected_datasets else None,
            "sources": sorted(selected_sources) if selected_sources else None,
            "sampleLimitPerClass": sample_limit_per_class,
            "distributionSampleLimitPerClass": distribution_sample_limit_per_class,
            "sampleSeed": opt.sample_seed,
            "classLimitPerRun": class_limit_per_run,
            "directDistributionPairs": not bool(opt.skip_direct_distribution_pairs),
            "rawJsonl": collect_raw_outputs,
            "verbosity": verbosity_from_opt(opt),
        },
    )

    manifest = {
        "datasetsRoot": str(datasets_root),
        "outputDir": str(output_root),
        "mode": "all-files-reference-summary-pairwise",
        "baselineMode": "human-summary-baseline",
        **_statistical_contract_fields(),
        "rate": rate,
        "roundPrecision": round_precision,
        "alignment": alignment,
        "alignmentName": PtAlignType.name(alignment),
        "summary": summary_shape,
        "popular": bool(opt.popular),
        "exactDtw": bool(opt.exact_dtw),
        "dtwWindow": dtw_window,
        "groupBy": group_by,
        "classScheme": class_scheme,
        "sampleLimitPerClass": sample_limit_per_class,
        "distributionSampleLimitPerClass": distribution_sample_limit_per_class,
        "sampleSeed": opt.sample_seed,
        "samplingMode": "seeded-random" if opt.sample_seed is not None else "stable",
        "classLimitPerRun": class_limit_per_run,
        "directDistributionPairs": not bool(opt.skip_direct_distribution_pairs),
        "rawJsonl": collect_raw_outputs,
        "metricNames": list(metric_names),
        "distributionMetricNames": list(DISTRIBUTION_METRIC_NAMES),
        "runs": [],
        "warnings": [],
    }

    human_dataset_dirs = [
        path
        for path in sorted(humans_root.iterdir())
        if path.is_dir() and (selected_datasets is None or path.name in selected_datasets)
    ]
    source_roots = [
        path
        for path in sorted(datasets_root.iterdir())
        if path.is_dir()
        and path.name != "humans"
        and (selected_sources is None or path.name in selected_sources)
    ]

    planned_runs = []
    planning_warnings: List[Dict[str, str]] = []
    for human_dataset_dir in human_dataset_dirs:
        dataset_name = human_dataset_dir.name
        reference_input = human_dataset_dir / "realTO"
        if not reference_input.exists():
            continue
        for source_root in source_roots:
            source_name = source_root.name
            for candidate_input, variant in _candidate_groups(source_root, dataset_name):
                candidate_count, class_counts = _count_run_candidates(
                    candidate_input,
                    group_by,
                    class_scheme,
                    planning_warnings,
                )
                planned_runs.append(
                    {
                        "runId": _run_id(source_name, dataset_name, variant),
                        "dataset": dataset_name,
                        "source": source_name,
                        "variant": _variant_label(variant),
                        "candidateInput": candidate_input,
                        "candidateCount": candidate_count,
                        "classCounts": class_counts,
                    }
                )
    manifest["planningWarnings"] = planning_warnings
    manifest["plannedRunCount"] = len(planned_runs)
    manifest["plannedCandidateCount"] = sum(
        item["candidateCount"] for item in planned_runs
    )

    progress = ProgressReporter(
        total_candidates=manifest["plannedCandidateCount"],
        total_runs=manifest["plannedRunCount"],
        verbosity=verbosity_from_opt(opt),
    )
    all_distribution_rows: List[Dict[str, object]] = []
    combined_pairwise_rows: List[Dict[str, object]] = []
    combined_stats_rows: List[Dict[str, object]] = []
    combined_baseline_rows: List[Dict[str, object]] = []
    combined_baseline_stats_rows: List[Dict[str, object]] = []
    combined_within_reference_rows: List[Dict[str, object]] = []
    combined_within_reference_stats_rows: List[Dict[str, object]] = []
    combined_within_comparison_rows: List[Dict[str, object]] = []
    combined_within_comparison_stats_rows: List[Dict[str, object]] = []
    combined_between_group_rows: List[Dict[str, object]] = []
    combined_between_group_stats_rows: List[Dict[str, object]] = []
    combined_summary_distribution_rows: List[Dict[str, object]] = []
    combined_raw_metric_outputs: List[Dict[str, object]] = []
    combined_raw_distribution_outputs: List[Dict[str, object]] = []
    if verbosity_from_opt(opt) >= 2:
        print(
            (
                f"planned {manifest['plannedRunCount']} runs and "
                f"{manifest['plannedCandidateCount']} candidate comparisons"
            ),
            flush=True,
        )

    for human_dataset_dir in human_dataset_dirs:
        dataset_name = human_dataset_dir.name
        reference_input = human_dataset_dir / "realTO"
        if not reference_input.exists():
            manifest["warnings"].append(
                {
                    "dataset": dataset_name,
                    "input": str(reference_input),
                    "error": "missing reference realTO directory",
                }
            )
            continue

        reference_entries = _load_reporting_entries(
            reference_input,
            group_by,
            class_scheme,
            manifest["warnings"],
        )
        references_by_class = _entries_by_class(reference_entries)

        for source_root in source_roots:
            source_name = source_root.name
            for candidate_input, variant in _candidate_groups(source_root, dataset_name):
                run_id = _run_id(source_name, dataset_name, variant)
                run_dir = _output_run_dir(output_root, source_name, dataset_name, variant)
                class_dir = run_dir / "classes"
                run_dir.mkdir(parents=True, exist_ok=True)

                planned_candidate_count, _ = _count_run_candidates(
                    candidate_input,
                    group_by,
                    class_scheme,
                    manifest["warnings"],
                )
                progress.start_run(run_id, planned_candidate_count)

                candidate_entries = _load_reporting_entries(
                    candidate_input,
                    group_by,
                    class_scheme,
                    manifest["warnings"],
                )
                candidates_by_class = _entries_by_class(candidate_entries)

                run_rows: List[Dict[str, object]] = []
                run_stats: List[Dict[str, object]] = []
                run_baseline_rows: List[Dict[str, object]] = []
                run_baseline_stats: List[Dict[str, object]] = []
                run_within_reference_rows: List[Dict[str, object]] = []
                run_within_reference_stats: List[Dict[str, object]] = []
                run_within_comparison_rows: List[Dict[str, object]] = []
                run_within_comparison_stats: List[Dict[str, object]] = []
                run_between_group_rows: List[Dict[str, object]] = []
                run_between_group_stats: List[Dict[str, object]] = []
                run_distribution_rows: List[Dict[str, object]] = []
                run_summary_distribution_rows: List[Dict[str, object]] = []
                run_raw_metric_outputs: List[Dict[str, object]] = []
                run_raw_distribution_outputs: List[Dict[str, object]] = []
                class_manifests = []
                skipped_classes = []

                class_keys = sorted(
                    set(references_by_class.keys()) | set(candidates_by_class.keys())
                )
                if class_limit_per_run is not None:
                    class_keys = class_keys[:class_limit_per_run]

                for class_key in class_keys:
                    class_references = references_by_class.get(class_key, [])
                    class_candidates = candidates_by_class.get(class_key, [])
                    if len(class_references) == 0 or len(class_candidates) == 0:
                        reason = (
                            "missingReference"
                            if len(class_references) == 0
                            else "missingComparison"
                        )
                        skipped_classes.append(
                            {
                                "classKey": class_key,
                                "referenceCount": len(class_references),
                                "candidateCount": len(class_candidates),
                                "reason": reason,
                            }
                        )
                        progress.skip_class(run_id, class_key, reason)
                        continue

                    full_reference_count = len(class_references)
                    full_candidate_count = len(class_candidates)
                    class_references = list(
                        _select_entries(
                            class_references,
                            sample_limit_per_class,
                            opt.sample_seed,
                            "reference",
                            run_id,
                            class_key,
                        )
                    )
                    class_candidates = list(
                        _select_entries(
                            class_candidates,
                            sample_limit_per_class,
                            opt.sample_seed,
                            "candidate",
                            run_id,
                            class_key,
                        )
                    )
                    distribution_references = list(
                        _select_entries(
                            class_references,
                            distribution_sample_limit_per_class,
                            opt.sample_seed,
                            "distribution-reference",
                            run_id,
                            class_key,
                        )
                    )
                    distribution_candidates = list(
                        _select_entries(
                            class_candidates,
                            distribution_sample_limit_per_class,
                            opt.sample_seed,
                            "distribution-candidate",
                            run_id,
                            class_key,
                        )
                    )

                    (
                        rows,
                        stats_rows,
                        within_comparison_rows,
                        within_comparison_stats_rows,
                        raw_metric_outputs,
                        class_metadata,
                    ) = _compare_class(
                        class_references,
                        class_candidates,
                        distribution_candidates,
                        run_id,
                        source_name,
                        dataset_name,
                        variant,
                        class_key,
                        reference_input,
                        rate,
                        alignment,
                        summary_shape,
                        bool(opt.popular),
                        round_precision,
                        metric_names,
                        dtw_window,
                        bool(opt.exact_dtw),
                        collect_raw_outputs,
                    )
                    (
                        baseline_rows,
                        baseline_stats_rows,
                        baseline_raw_metric_outputs,
                        baseline_metadata,
                    ) = (
                        _compare_human_baseline_class(
                            class_references,
                            run_id,
                            source_name,
                            dataset_name,
                            variant,
                            class_key,
                            reference_input,
                            rate,
                            alignment,
                            summary_shape,
                            bool(opt.popular),
                            round_precision,
                            metric_names,
                            dtw_window,
                            bool(opt.exact_dtw),
                            collect_raw_outputs,
                        )
                    )

                    within_reference_rows = []
                    within_reference_stats_rows = _summary_stats(
                        [],
                        run_id,
                        "human",
                        dataset_name,
                        variant,
                        class_key,
                        summary_shape,
                        round_precision,
                    )
                    between_group_rows = []
                    between_group_stats_rows = _summary_stats(
                        [],
                        run_id,
                        source_name,
                        dataset_name,
                        variant,
                        class_key,
                        summary_shape,
                        round_precision,
                    )
                    direct_raw_metric_outputs = []
                    direct_metadata = {
                        "mode": "direct-distribution-pairs",
                        **_statistical_contract_fields(),
                        "referenceCount": len(distribution_references),
                        "candidateCount": len(distribution_candidates),
                        "withinReferencePairs": 0,
                        "betweenGroupPairs": 0,
                    }
                    if not opt.skip_direct_distribution_pairs:
                        (
                            within_reference_rows,
                            within_reference_stats_rows,
                            between_group_rows,
                            between_group_stats_rows,
                            direct_raw_metric_outputs,
                            direct_metadata,
                        ) = _compare_direct_distribution_pairs_class(
                            distribution_references,
                            distribution_candidates,
                            run_id,
                            source_name,
                            dataset_name,
                            variant,
                            class_key,
                            reference_input,
                            rate,
                            alignment,
                            summary_shape,
                            bool(opt.popular),
                            round_precision,
                            metric_names,
                            dtw_window,
                            bool(opt.exact_dtw),
                            collect_raw_outputs,
                        )

                    safe_class_key = _safe_path_part(class_key)
                    class_output_dir = class_dir / safe_class_key
                    _write_json(
                        class_output_dir / "pairwise.json",
                        {
                            "metadata": class_metadata,
                            "pairs": rows,
                            "stats": stats_rows,
                        },
                    )
                    _write_csv(class_output_dir / "pairwise.csv", rows, PAIRWISE_COLUMNS)
                    _write_csv(class_output_dir / "stats.csv", stats_rows, STATS_COLUMNS)
                    _write_csv(
                        class_output_dir / "within_comparison.csv",
                        within_comparison_rows,
                        PAIRWISE_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "within_comparison_stats.csv",
                        within_comparison_stats_rows,
                        STATS_COLUMNS,
                    )
                    _write_json(
                        class_output_dir / "baseline.json",
                        {
                            "metadata": baseline_metadata,
                            "baseline": baseline_rows,
                            "stats": baseline_stats_rows,
                        },
                    )
                    _write_csv(
                        class_output_dir / "baseline.csv",
                        baseline_rows,
                        BASELINE_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "baseline_stats.csv",
                        baseline_stats_rows,
                        STATS_COLUMNS,
                    )
                    class_summary_distribution_rows = _lightweight_distribution_rows(
                        rows,
                        stats_rows,
                        baseline_rows,
                        baseline_stats_rows,
                        within_comparison_rows,
                        within_comparison_stats_rows,
                        metric_names,
                        round_precision,
                    )
                    class_distribution_rows = _lightweight_distribution_rows(
                        between_group_rows,
                        between_group_stats_rows,
                        within_reference_rows,
                        within_reference_stats_rows,
                        within_comparison_rows,
                        within_comparison_stats_rows,
                        metric_names,
                        round_precision,
                    )
                    class_raw_distribution_outputs = (
                        _raw_distribution_outputs_from_rows(class_distribution_rows)
                        if collect_raw_outputs
                        else []
                    )
                    _write_csv(
                        class_output_dir / "distribution.csv",
                        class_distribution_rows,
                        DISTRIBUTION_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "summary_distribution.csv",
                        class_summary_distribution_rows,
                        DISTRIBUTION_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "within_reference.csv",
                        within_reference_rows,
                        PAIRWISE_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "within_reference_stats.csv",
                        within_reference_stats_rows,
                        STATS_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "between_groups.csv",
                        between_group_rows,
                        PAIRWISE_COLUMNS,
                    )
                    _write_csv(
                        class_output_dir / "between_groups_stats.csv",
                        between_group_stats_rows,
                        STATS_COLUMNS,
                    )
                    _write_readme(
                        class_output_dir,
                        f"Evaluation Outputs: {run_id} / {class_key}",
                        [
                            f"Class: {class_key}",
                            f"Run: {run_id}",
                            f"Selected reference samples: {len(class_references)} of {full_reference_count}.",
                            f"Selected candidate samples: {len(class_candidates)} of {full_candidate_count}.",
                            f"Direct distribution reference samples: {len(distribution_references)}.",
                            f"Direct distribution candidate samples: {len(distribution_candidates)}.",
                        ],
                    )

                    run_rows.extend(rows)
                    run_stats.extend(stats_rows)
                    run_baseline_rows.extend(baseline_rows)
                    run_baseline_stats.extend(baseline_stats_rows)
                    run_within_reference_rows.extend(within_reference_rows)
                    run_within_reference_stats.extend(within_reference_stats_rows)
                    run_within_comparison_rows.extend(within_comparison_rows)
                    run_within_comparison_stats.extend(within_comparison_stats_rows)
                    run_between_group_rows.extend(between_group_rows)
                    run_between_group_stats.extend(between_group_stats_rows)
                    run_distribution_rows.extend(class_distribution_rows)
                    run_summary_distribution_rows.extend(class_summary_distribution_rows)
                    run_raw_metric_outputs.extend(raw_metric_outputs)
                    run_raw_metric_outputs.extend(baseline_raw_metric_outputs)
                    run_raw_metric_outputs.extend(direct_raw_metric_outputs)
                    run_raw_distribution_outputs.extend(class_raw_distribution_outputs)
                    class_manifests.append(
                        {
                            **class_metadata,
                            **_statistical_contract_fields(),
                            "baselineMode": baseline_metadata["mode"],
                            "directDistributionMode": direct_metadata["mode"],
                            "outputDir": str(class_output_dir),
                            "fullReferenceCount": full_reference_count,
                            "fullCandidateCount": full_candidate_count,
                            "selectedReferenceCount": len(class_references),
                            "selectedCandidateCount": len(class_candidates),
                            "distributionReferenceCount": len(distribution_references),
                            "distributionCandidateCount": len(distribution_candidates),
                            "pairwiseRows": len(rows),
                            "statsRows": len(stats_rows),
                            "withinReferenceRows": len(within_reference_rows),
                            "withinReferenceStatsRows": len(within_reference_stats_rows),
                            "withinComparisonRows": len(within_comparison_rows),
                            "withinComparisonStatsRows": len(
                                within_comparison_stats_rows
                            ),
                            "betweenGroupsRows": len(between_group_rows),
                            "betweenGroupsStatsRows": len(between_group_stats_rows),
                            "baselineRows": len(baseline_rows),
                            "baselineStatsRows": len(baseline_stats_rows),
                            "distributionRows": len(class_distribution_rows),
                            "summaryDistributionRows": len(
                                class_summary_distribution_rows
                            ),
                            "rawMetricRows": (
                                len(raw_metric_outputs)
                                + len(baseline_raw_metric_outputs)
                                + len(direct_raw_metric_outputs)
                            ),
                            "rawDistributionRows": len(class_raw_distribution_outputs),
                        }
                    )
                    progress.finish_class(run_id, class_key, len(rows))

                run_manifest = {
                    **_statistical_contract_fields(),
                    "id": run_id,
                    "source": source_name,
                    "dataset": dataset_name,
                    "variant": _variant_label(variant),
                    "referenceInput": str(reference_input),
                    "candidateInput": str(candidate_input),
                    "outputDir": str(run_dir),
                    "classCount": len(class_manifests),
                    "pairwiseRows": len(run_rows),
                    "statsRows": len(run_stats),
                    "baselineRows": len(run_baseline_rows),
                    "baselineStatsRows": len(run_baseline_stats),
                    "withinReferenceRows": len(run_within_reference_rows),
                    "withinReferenceStatsRows": len(run_within_reference_stats),
                    "withinComparisonRows": len(run_within_comparison_rows),
                    "withinComparisonStatsRows": len(run_within_comparison_stats),
                    "betweenGroupsRows": len(run_between_group_rows),
                    "betweenGroupsStatsRows": len(run_between_group_stats),
                    "distributionRows": len(run_distribution_rows),
                    "summaryDistributionRows": len(run_summary_distribution_rows),
                    "rawMetricRows": len(run_raw_metric_outputs),
                    "rawDistributionRows": len(run_raw_distribution_outputs),
                    "classes": class_manifests,
                    "skippedClasses": skipped_classes,
                }

                _write_json(
                    run_dir / "pairwise.json",
                    {
                        "metadata": run_manifest,
                        "pairs": run_rows,
                        "stats": run_stats,
                    },
                )
                _write_csv(run_dir / "pairwise.csv", run_rows, PAIRWISE_COLUMNS)
                _write_csv(run_dir / "stats.csv", run_stats, STATS_COLUMNS)
                _write_csv(
                    run_dir / "within_reference.csv",
                    run_within_reference_rows,
                    PAIRWISE_COLUMNS,
                )
                _write_csv(
                    run_dir / "within_reference_stats.csv",
                    run_within_reference_stats,
                    STATS_COLUMNS,
                )
                _write_csv(
                    run_dir / "within_comparison.csv",
                    run_within_comparison_rows,
                    PAIRWISE_COLUMNS,
                )
                _write_csv(
                    run_dir / "within_comparison_stats.csv",
                    run_within_comparison_stats,
                    STATS_COLUMNS,
                )
                _write_csv(
                    run_dir / "between_groups.csv",
                    run_between_group_rows,
                    PAIRWISE_COLUMNS,
                )
                _write_csv(
                    run_dir / "between_groups_stats.csv",
                    run_between_group_stats,
                    STATS_COLUMNS,
                )
                _write_json(
                    run_dir / "baseline.json",
                    {
                        "metadata": {
                            **run_manifest,
                            "mode": "human-summary-baseline",
                        },
                        "baseline": run_baseline_rows,
                        "stats": run_baseline_stats,
                    },
                )
                _write_csv(run_dir / "baseline.csv", run_baseline_rows, BASELINE_COLUMNS)
                _write_csv(
                    run_dir / "baseline_stats.csv",
                    run_baseline_stats,
                    STATS_COLUMNS,
                )
                _write_csv(
                    run_dir / "distribution.csv",
                    run_distribution_rows,
                    DISTRIBUTION_COLUMNS,
                )
                _write_csv(
                    run_dir / "summary_distribution.csv",
                    run_summary_distribution_rows,
                    DISTRIBUTION_COLUMNS,
                )
                _write_json(run_dir / "manifest.json", run_manifest)
                _write_readme(
                    run_dir,
                    f"Evaluation Outputs: {run_id}",
                    [
                        f"Run: {run_id}",
                        f"Classes written: {len(class_manifests)}.",
                        "Per-class outputs are under classes/<classKey>/.",
                    ],
                )
                manifest["runs"].append(run_manifest)
                all_distribution_rows.extend(run_distribution_rows)
                combined_pairwise_rows.extend(run_rows)
                combined_stats_rows.extend(run_stats)
                combined_baseline_rows.extend(run_baseline_rows)
                combined_baseline_stats_rows.extend(run_baseline_stats)
                combined_within_reference_rows.extend(run_within_reference_rows)
                combined_within_reference_stats_rows.extend(run_within_reference_stats)
                combined_within_comparison_rows.extend(run_within_comparison_rows)
                combined_within_comparison_stats_rows.extend(
                    run_within_comparison_stats
                )
                combined_between_group_rows.extend(run_between_group_rows)
                combined_between_group_stats_rows.extend(run_between_group_stats)
                combined_summary_distribution_rows.extend(run_summary_distribution_rows)
                combined_raw_metric_outputs.extend(run_raw_metric_outputs)
                combined_raw_distribution_outputs.extend(run_raw_distribution_outputs)
                progress.finish_run(run_id, len(run_rows))

    _write_csv(output_root / "distribution.csv", all_distribution_rows, DISTRIBUTION_COLUMNS)
    _write_csv(
        output_root / "summary_distribution.csv",
        combined_summary_distribution_rows,
        DISTRIBUTION_COLUMNS,
    )
    combined_aggregate_rows = build_combined_aggregate_summaries(
        combined_pairwise_rows,
        combined_baseline_rows,
        combined_within_reference_rows,
        combined_within_comparison_rows,
        combined_between_group_rows,
        round_precision,
    )
    manifest["combinedOutputs"] = write_combined_report_exports(
        output_root,
        combined_pairwise_rows,
        combined_stats_rows,
        combined_baseline_rows,
        combined_baseline_stats_rows,
        combined_within_reference_rows,
        combined_within_reference_stats_rows,
        combined_within_comparison_rows,
        combined_within_comparison_stats_rows,
        combined_between_group_rows,
        combined_between_group_stats_rows,
        all_distribution_rows,
        combined_summary_distribution_rows,
        combined_aggregate_rows,
        combined_raw_metric_outputs,
        combined_raw_distribution_outputs,
        manifest,
        write_raw_jsonl=collect_raw_outputs,
    )
    _write_json(output_root / "manifest.json", manifest)
    _write_readme(
        output_root,
        "Evaluation Outputs",
        [
            "Top-level distribution.csv and summary_distribution.csv pool every completed run.",
            "The combined/ folder contains concatenated CSV and JSONL files intended for plotting.",
        ],
    )
    return 0
