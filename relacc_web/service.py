from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any
from urllib.parse import quote, unquote

from relacc.canvas import OverlayGroup, render_overlay_svg
from relacc.metrics import METRIC_NAMES
from relacc.gestures.ptaligntype import PtAlignType
from relacc.pipeline.distribution import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_PARENT_DIR,
    format_distribution_rows_csv,
    run_distribution_comparison,
)
from relacc.pipeline.dataset_discovery import (
    class_key_for_relative_path,
)
from relacc.pipeline.pairwise import (
    DIRECT_MODE,
    SUMMARY_MODE,
    discover_pairs,
    format_pair_rows_csv,
    run_pairwise_comparison,
)
from relacc.utils.csv import CSVUtil


IGNORED_NAMES = {".ds_store", "thumbs.db"}
MAX_OVERLAY_FILES = 18
MIB = 1024 * 1024
MAX_UPLOAD_BYTES = 25 * MIB
MAX_TOTAL_UPLOAD_BYTES = 100 * MIB
MAX_COMPARISON_ARCHIVES = 8
MAX_ZIP_MEMBERS = 10_000
MAX_CSV_FILES = 5_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 10 * MIB
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 250 * MIB
MAX_COMPRESSION_RATIO = 1_000
MAX_ACTIVE_JOBS = 4
MAX_RETAINED_TERMINAL_JOBS = 100
JOB_TTL_SECONDS = 86_400
TEMP_ROOT = Path(tempfile.gettempdir())
WORKDIR_PREFIX = "relacc-web-"


@dataclass(frozen=True)
class EvaluationConfig:
    mode: str = SUMMARY_MODE
    summary: str | None = "centroid"
    alignment: int = 0
    popular: bool = False
    strict: bool = True
    rate: int | None = None
    round_precision: int = 3
    dtw_window: int | None = None
    exact_dtw: bool = False
    group_by: str = GROUP_BY_FILENAME_LABEL


JOBS: dict[str, dict[str, Any]] = {}
LOCK = RLock()


class ResourceLimitError(ValueError):
    pass


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_config(config_text: str | None) -> EvaluationConfig:
    raw = json.loads(config_text or "{}")
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object.")

    def as_optional_int(key):
        value = raw.get(key)
        if value in (None, ""):
            return None
        return int(value)

    mode = str(raw.get("mode") or SUMMARY_MODE).strip().lower()
    if mode not in {DIRECT_MODE, SUMMARY_MODE, "distribution"}:
        mode = SUMMARY_MODE

    summary = raw.get("summary", "centroid")
    if summary in ("", "first", "none", None):
        summary = None

    return EvaluationConfig(
        mode=mode,
        summary=summary,
        alignment=PtAlignType.normalize(
            raw.get("alignment", PtAlignType.CHRONOLOGICAL)
        ),
        popular=bool(raw.get("popular", False)),
        strict=bool(raw.get("strict", True)),
        rate=as_optional_int("rate"),
        round_precision=int(raw.get("roundPrecision") or raw.get("round_precision") or 3),
        dtw_window=as_optional_int("dtwWindow"),
        exact_dtw=bool(raw.get("exactDtw", False)),
        group_by=str(raw.get("groupBy") or GROUP_BY_FILENAME_LABEL),
    )


def _issue(severity: str, scope: str, message: str, path: str | None = None):
    return {
        "severity": severity,
        "scope": scope,
        "message": message,
        "path": path,
    }


def _is_ignored_zip_member(name: str):
    path = PurePosixPath(name)
    return (
        any(part == "__MACOSX" for part in path.parts)
        or path.name.lower() in IGNORED_NAMES
        or path.name.startswith("._")
    )


def _zip_member_is_symlink(info: zipfile.ZipInfo):
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _rollback_paths(paths):
    for path in reversed(paths):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def extract_zip(upload: bytes, target_dir: Path, scope: str):
    issues = []
    extracted = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(upload))
    except zipfile.BadZipFile:
        return [_issue("error", scope, "Upload is not a readable zip archive.")], []

    with archive:
        members = archive.infolist()
        if not members:
            return [_issue("error", scope, "Zip archive is empty.")], []
        if len(members) > MAX_ZIP_MEMBERS:
            return [_issue("error", scope, f"Zip archive exceeds member limit ({MAX_ZIP_MEMBERS}).")], []

        csv_infos = []
        total_declared = 0
        for info in members:
            if info.is_dir() or _is_ignored_zip_member(info.filename):
                continue

            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                issues.append(_issue("error", scope, "Zip member has an unsafe path.", info.filename))
                continue
            if info.flag_bits & 0x1:
                issues.append(_issue("error", scope, "Encrypted ZIP members are not supported.", info.filename))
                continue
            if _zip_member_is_symlink(info):
                issues.append(_issue("error", scope, "Symlink ZIP members are not supported.", info.filename))
                continue
            if rel.suffix.lower() != ".csv":
                issues.append(_issue("warning", scope, "Ignored non-CSV file.", info.filename))
                continue
            if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                issues.append(_issue("error", scope, f"CSV member exceeds uncompressed size limit ({MAX_MEMBER_UNCOMPRESSED_BYTES} bytes).", info.filename))
                continue
            compressed_size = max(info.compress_size, 1)
            if info.file_size / compressed_size > MAX_COMPRESSION_RATIO:
                issues.append(_issue("error", scope, f"CSV member exceeds compression ratio limit ({MAX_COMPRESSION_RATIO}:1).", info.filename))
                continue
            total_declared += info.file_size
            if total_declared > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                issues.append(_issue("error", scope, f"Zip archive exceeds uncompressed size limit ({MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes).", info.filename))
                continue
            csv_infos.append((info, rel))

        if len(csv_infos) > MAX_CSV_FILES:
            return [_issue("error", scope, f"Zip archive exceeds CSV file limit ({MAX_CSV_FILES}).")], []
        if any(issue["severity"] == "error" for issue in issues):
            return issues, []
        for info, rel in csv_infos:
            destination = target_dir.joinpath(*rel.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            actual_member = 0
            try:
                with archive.open(info) as source, open(destination, "wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        actual_member += len(chunk)
                        if actual_member > info.file_size or actual_member > MAX_MEMBER_UNCOMPRESSED_BYTES:
                            raise ResourceLimitError("CSV member exceeded declared or configured size during extraction.")
                        output.write(chunk)
                extracted.append(destination)
            except (zipfile.BadZipFile, RuntimeError, OSError, ResourceLimitError) as exc:
                destination.unlink(missing_ok=True)
                _rollback_paths(extracted)
                return [_issue("error", scope, str(exc), info.filename)], []

    if not extracted:
        issues.append(_issue("error", scope, "Zip archive contains no CSV gesture files."))

    return issues, extracted


def _relative_csv_keys(root: Path):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.csv"))


def _class_key_safe(relative_path: str, group_by: str):
    try:
        return class_key_for_relative_path(relative_path, group_by)
    except ValueError as exc:
        return None, str(exc)


def _dataset_summary(root: Path, scope: str, group_by: str):
    issues = []
    files = _relative_csv_keys(root)
    total_points = 0
    class_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}

    for rel in files:
        full_path = root / rel
        try:
            state = {}

            def done(points):
                state["points"] = points

            CSVUtil.readGesture(str(full_path), done)
            point_count = len(state.get("points") or [])
            if point_count == 0:
                issues.append(_issue("error", scope, "CSV parsed no gesture points.", rel))
            total_points += point_count
        except Exception as exc:  # noqa: BLE001 - validation should report parser details.
            issues.append(_issue("error", scope, str(exc), rel))

        parent_key = class_key_for_relative_path(rel, GROUP_BY_PARENT_DIR)
        parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1
        class_key = _class_key_safe(rel, group_by)
        if isinstance(class_key, tuple):
            issues.append(_issue("error", scope, class_key[1], rel))
        else:
            class_counts[class_key] = class_counts.get(class_key, 0) + 1

    return {
        "fileCount": len(files),
        "pointCount": total_points,
        "examples": files[:6],
        "classCounts": class_counts,
        "parentCounts": parent_counts,
    }, issues


def validate_inputs(reference_dir: Path, candidate_dir: Path, config: EvaluationConfig):
    class_group_by = config.group_by if config.mode == "distribution" else GROUP_BY_PARENT_DIR
    reference_summary, reference_issues = _dataset_summary(reference_dir, "reference", class_group_by)
    candidate_summary, candidate_issues = _dataset_summary(candidate_dir, "candidate", class_group_by)
    issues = [*reference_issues, *candidate_issues]

    if config.dtw_window is not None and config.exact_dtw:
        issues.append(
            _issue("error", "configuration", "--dtw-window cannot be combined with exact DTW.")
        )

    if config.mode == DIRECT_MODE:
        try:
            pairs, missing_candidate, missing_reference = discover_pairs(
                str(reference_dir),
                str(candidate_dir),
                strict=False,
            )
        except ValueError as exc:
            pairs = []
            missing_candidate = []
            missing_reference = []
            issues.append(_issue("error", "matching", str(exc)))
        if config.strict and (missing_candidate or missing_reference):
            issues.append(
                _issue(
                    "error",
                    "matching",
                    "Strict mode found unmatched files.",
                )
            )
        elif missing_candidate or missing_reference:
            issues.append(
                _issue(
                    "warning",
                    "matching",
                    "Unmatched files will be ignored in non-strict direct mode.",
                )
            )
        mode_summary = {
            "matchedPairCount": len(pairs),
            "missingInCandidate": missing_candidate[:50],
            "missingInReference": missing_reference[:50],
        }
    elif config.mode == "distribution":
        ref_counts = reference_summary["classCounts"]
        cand_counts = candidate_summary["classCounts"]
        valid = []
        skipped = []
        invalid = []
        for class_key in sorted(set(ref_counts) | set(cand_counts)):
            ref_count = ref_counts.get(class_key, 0)
            cand_count = cand_counts.get(class_key, 0)
            if ref_count == 0:
                skipped.append({"classKey": class_key, "reason": "missingReference"})
            elif cand_count == 0:
                skipped.append({"classKey": class_key, "reason": "missingCandidate"})
            elif ref_count < 2:
                invalid.append({"classKey": class_key, "reason": "needAtLeastTwoReferenceSamples"})
            else:
                valid.append(class_key)
        if not valid:
            issues.append(_issue("error", "distribution", "No valid classes for distribution comparison."))
        mode_summary = {
            "validClassCount": len(valid),
            "validClasses": valid[:100],
            "skippedClasses": skipped[:100],
            "invalidClasses": invalid[:100],
        }
    else:
        mode_summary = {
            "referenceCount": reference_summary["fileCount"],
            "candidateCount": candidate_summary["fileCount"],
        }

    return {
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "reference": reference_summary,
        "candidate": candidate_summary,
        "mode": mode_summary,
        "issues": issues,
    }


def create_job(reference_zip: bytes, candidate_zip: bytes, config: EvaluationConfig):
    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix=f"{WORKDIR_PREFIX}{job_id}-", dir=TEMP_ROOT))
    reference_dir = workdir / "reference"
    candidate_dir = workdir / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()

    issues, _ = extract_zip(reference_zip, reference_dir, "reference")
    candidate_issues, _ = extract_zip(candidate_zip, candidate_dir, "candidate")
    issues.extend(candidate_issues)

    validation = {
        "ok": False,
        "reference": {"fileCount": 0, "pointCount": 0, "examples": [], "classCounts": {}, "parentCounts": {}},
        "candidate": {"fileCount": 0, "pointCount": 0, "examples": [], "classCounts": {}, "parentCounts": {}},
        "mode": {},
        "issues": issues,
    }
    if not any(issue["severity"] == "error" for issue in issues):
        validation = validate_inputs(reference_dir, candidate_dir, config)

    status = "queued" if validation["ok"] else "failed"
    job = {
        "id": job_id,
        "status": status,
        "phase": "Queued" if validation["ok"] else "Validation failed",
        "progress": 8 if validation["ok"] else 100,
        "createdAt": time.time(),
        "updatedAt": time.time(),
        "config": config.__dict__,
        "validation": validation,
        "workdir": str(workdir),
        "referenceDir": str(reference_dir),
        "candidateDir": str(candidate_dir),
        "result": None,
        "error": None if validation["ok"] else "Validation failed.",
    }
    with LOCK:
        JOBS[job_id] = job
        if status == "failed":
            _delete_job_workdir_locked(job)
    return public_job(job)


def _safe_group_name(name: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (name or "").strip())
    return cleaned.strip("-") or fallback


def _is_terminal(job: dict[str, Any]):
    return job.get("status") in {"completed", "failed"}


def _safe_workdir_path(path_text: str | None):
    if not path_text:
        return None
    try:
        path = Path(path_text).resolve()
        root = TEMP_ROOT.resolve()
    except OSError:
        return None
    if path.parent != root or not path.name.startswith(WORKDIR_PREFIX):
        return None
    return path


def _delete_job_workdir_locked(job: dict[str, Any]):
    path = _safe_workdir_path(job.get("workdir"))
    if path is None:
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def prune_terminal_jobs_locked(now: float | None = None):
    now = time.time() if now is None else now
    for job_id, job in list(JOBS.items()):
        if _is_terminal(job) and now - float(job.get("updatedAt", 0)) > JOB_TTL_SECONDS:
            _delete_job_workdir_locked(job)
            JOBS.pop(job_id, None)

    terminal = sorted(
        [(job_id, job) for job_id, job in JOBS.items() if _is_terminal(job)],
        key=lambda item: float(item[1].get("updatedAt", 0)),
    )
    while len(terminal) > MAX_RETAINED_TERMINAL_JOBS:
        job_id, job = terminal.pop(0)
        _delete_job_workdir_locked(job)
        JOBS.pop(job_id, None)


def prune_terminal_jobs(now: float | None = None):
    with LOCK:
        prune_terminal_jobs_locked(now=now)


def count_active_jobs():
    with LOCK:
        prune_terminal_jobs_locked()
        return sum(1 for job in JOBS.values() if job.get("status") in {"queued", "running"})


def has_active_job_capacity():
    return count_active_jobs() < MAX_ACTIVE_JOBS


def cleanup_orphan_workdirs(now: float | None = None):
    now = time.time() if now is None else now
    try:
        children = list(TEMP_ROOT.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir() or not child.name.startswith(WORKDIR_PREFIX):
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError:
            continue
        if age > JOB_TTL_SECONDS:
            safe = _safe_workdir_path(str(child))
            if safe is not None:
                shutil.rmtree(safe, ignore_errors=True)


def create_group_job(
    reference_zip: bytes,
    comparisons: list[tuple[str, bytes]],
    config: EvaluationConfig,
):
    if len(comparisons) == 1:
        return create_job(reference_zip, comparisons[0][1], config)

    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix=f"{WORKDIR_PREFIX}{job_id}-", dir=TEMP_ROOT))
    reference_dir = workdir / "reference"
    candidates_root = workdir / "candidates"
    reference_dir.mkdir()
    candidates_root.mkdir()

    issues, _ = extract_zip(reference_zip, reference_dir, "reference")
    candidate_dirs: dict[str, str] = {}
    candidate_summaries = {}
    validation = {
        "ok": False,
        "reference": {"fileCount": 0, "pointCount": 0, "examples": [], "classCounts": {}, "parentCounts": {}},
        "candidate": {"fileCount": 0, "pointCount": 0, "examples": [], "classCounts": {}, "parentCounts": {}},
        "candidates": {},
        "mode": {"groupCount": len(comparisons)},
        "issues": issues,
    }

    used_names = set()
    for index, (raw_name, upload) in enumerate(comparisons, start=1):
        name = _safe_group_name(raw_name, f"comparison-{index}")
        while name in used_names:
            name = f"{name}-{index}"
        used_names.add(name)
        candidate_dir = candidates_root / name
        candidate_dir.mkdir()
        candidate_issues, _ = extract_zip(upload, candidate_dir, name)
        issues.extend(candidate_issues)
        candidate_dirs[name] = str(candidate_dir)

    if not any(issue["severity"] == "error" for issue in issues):
        for name, path in candidate_dirs.items():
            group_validation = validate_inputs(reference_dir, Path(path), config)
            validation["candidates"][name] = group_validation
            candidate_summaries[name] = group_validation["candidate"]
            for issue in group_validation["issues"]:
                scoped = dict(issue)
                if scoped["scope"] != "reference":
                    scoped["scope"] = f"{name}:{scoped['scope']}"
                issues.append(scoped)
        validation.update(
            {
                "reference": next(iter(validation["candidates"].values()))["reference"] if validation["candidates"] else validation["reference"],
                "candidate": {
                    "fileCount": sum(item["fileCount"] for item in candidate_summaries.values()),
                    "pointCount": sum(item["pointCount"] for item in candidate_summaries.values()),
                    "examples": [
                        example
                        for summary in candidate_summaries.values()
                        for example in summary.get("examples", [])[:2]
                    ][:8],
                    "classCounts": {},
                    "parentCounts": {},
                },
                "candidates": validation["candidates"],
            }
        )

    validation["issues"] = issues
    validation["ok"] = not any(issue["severity"] == "error" for issue in issues)
    status = "queued" if validation["ok"] else "failed"
    job = {
        "id": job_id,
        "status": status,
        "phase": "Queued" if validation["ok"] else "Validation failed",
        "progress": 8 if validation["ok"] else 100,
        "createdAt": time.time(),
        "updatedAt": time.time(),
        "config": config.__dict__,
        "validation": validation,
        "workdir": str(workdir),
        "referenceDir": str(reference_dir),
        "candidateDirs": candidate_dirs,
        "result": None,
        "error": None if validation["ok"] else "Validation failed.",
    }
    with LOCK:
        JOBS[job_id] = job
        if status == "failed":
            _delete_job_workdir_locked(job)
    return public_job(job)


def public_job(job: dict[str, Any]):
    safe = {
        key: value
        for key, value in job.items()
        if key not in {"workdir", "referenceDir", "candidateDir", "candidateDirs"}
    }
    if "config" in safe:
        safe["config"] = {
            **safe["config"],
            "alignmentName": PtAlignType.name(safe["config"]["alignment"]),
        }
    return _json_safe(safe)


def get_job(job_id: str):
    with LOCK:
        prune_terminal_jobs_locked()
        job = JOBS.get(job_id)
        if job is None:
            return None
        return job


def _update_job(job_id: str, **updates):
    with LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updatedAt"] = time.time()


def encode_group_overlay_key(group: str, result_key: str):
    return f"{quote(str(group), safe='')}::{quote(str(result_key), safe='')}"


def _decode_group_overlay_key(key: str):
    if "::" not in key:
        return None
    encoded_group, encoded_result = key.split("::", 1)
    if not encoded_group or not encoded_result:
        return None
    return unquote(encoded_group), unquote(encoded_result)


def run_job(job_id: str):
    job = get_job(job_id)
    if job is None or job["status"] != "queued":
        return

    config = EvaluationConfig(**job["config"])
    reference_dir = job["referenceDir"]

    _update_job(job_id, status="running", phase="Preparing pipeline", progress=22)
    start = time.time()
    try:
        candidate_dirs = job.get("candidateDirs")
        if candidate_dirs:
            results = []
            pair_rows = []
            for index, (name, candidate_dir) in enumerate(candidate_dirs.items(), start=1):
                _update_job(
                    job_id,
                    phase=f"Computing group {name}",
                    progress=min(90, 20 + int(index / max(len(candidate_dirs), 1) * 65)),
                )
                if config.mode == "distribution":
                    group_result = run_distribution_comparison(
                        reference_dir,
                        candidate_dir,
                        rate=config.rate,
                        alignment_type=config.alignment,
                        summary_shape=config.summary,
                        popular_shape=config.popular,
                        round_precision=config.round_precision,
                        group_by=config.group_by,
                        dtw_window=config.dtw_window,
                        exact_dtw=config.exact_dtw,
                    )
                    group_result.setdefault("metadata", {})["comparisonGroup"] = name
                    group_result["metadata"]["overlayKeys"] = [
                        encode_group_overlay_key(name, class_key)
                        for class_key in overlay_keys_for_result(config.mode, group_result)
                    ]
                    results.append(group_result)
                else:
                    group_result = run_pairwise_comparison(
                        reference_dir,
                        candidate_dir,
                        rate=config.rate,
                        alignment_type=config.alignment,
                        summary_shape=config.summary,
                        popular_shape=config.popular,
                        strict=config.strict,
                        round_precision=config.round_precision,
                        comparison_mode=config.mode,
                        metric_names=METRIC_NAMES,
                        dtw_window=config.dtw_window,
                        exact_dtw=config.exact_dtw,
                    )
                    for row in group_result.get("pairs", []):
                        row["comparisonGroup"] = name
                        row["overlayKey"] = encode_group_overlay_key(name, row.get("pairKey"))
                        pair_rows.append(row)
                    results.append(group_result)
            result = {
                "metadata": {
                    "comparisonMode": config.mode,
                    "alignment": config.alignment,
                    "alignmentName": PtAlignType.name(config.alignment),
                    "comparisonGroups": list(candidate_dirs.keys()),
                    "pairCount": len(pair_rows),
                    "runSeconds": round(time.time() - start, 3),
                    "overlayKeys": [row.get("overlayKey") for row in pair_rows[:200]],
                },
                "pairs": pair_rows,
                "groupResults": results,
            }
        elif config.mode == "distribution":
            candidate_dir = job["candidateDir"]
            _update_job(job_id, phase="Computing human and candidate distributions", progress=48)
            result = run_distribution_comparison(
                reference_dir,
                candidate_dir,
                rate=config.rate,
                alignment_type=config.alignment,
                summary_shape=config.summary,
                popular_shape=config.popular,
                round_precision=config.round_precision,
                group_by=config.group_by,
                dtw_window=config.dtw_window,
                exact_dtw=config.exact_dtw,
            )
        else:
            candidate_dir = job["candidateDir"]
            _update_job(job_id, phase="Computing pairwise gesture metrics", progress=52)
            result = run_pairwise_comparison(
                reference_dir,
                candidate_dir,
                rate=config.rate,
                alignment_type=config.alignment,
                summary_shape=config.summary,
                popular_shape=config.popular,
                strict=config.strict,
                round_precision=config.round_precision,
                comparison_mode=config.mode,
                metric_names=METRIC_NAMES,
                dtw_window=config.dtw_window,
                exact_dtw=config.exact_dtw,
            )

        result.setdefault("metadata", {})
        result["metadata"]["runSeconds"] = round(time.time() - start, 3)
        result["metadata"]["overlayKeys"] = overlay_keys_for_result(config.mode, result)
        _update_job(
            job_id,
            status="completed",
            phase="Completed",
            progress=100,
            result=_json_safe(result),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - jobs should surface domain errors.
        _update_job(
            job_id,
            status="failed",
            phase="Run failed",
            progress=100,
            error=str(exc),
        )
        with LOCK:
            failed_job = JOBS.get(job_id)
            if failed_job is not None:
                _delete_job_workdir_locked(failed_job)


def overlay_keys_for_result(mode: str, result: dict[str, Any]):
    group_results = result.get("groupResults")
    if group_results is not None:
        if mode == "distribution":
            keys = []
            for group_result in group_results:
                group = group_result.get("metadata", {}).get("comparisonGroup")
                for class_key in overlay_keys_for_result(mode, group_result):
                    keys.append(encode_group_overlay_key(group, class_key))
            return sorted(keys)
        return [row.get("overlayKey") for row in result.get("pairs", [])[:200]]
    if mode == "distribution":
        rows = result.get("results", {}).get("perClass", [])
        return sorted({row.get("classKey") for row in rows if row.get("classKey") is not None})
    return [row.get("pairKey") for row in result.get("pairs", [])[:200]]


def export_result(job_id: str, fmt: str):
    job = get_job(job_id)
    if job is None or not job.get("result"):
        return None
    result = job["result"]
    config = EvaluationConfig(**job["config"])
    if fmt == "json":
        return json.dumps(_json_safe(result), indent=2), "application/json"
    if config.mode == "distribution":
        return format_distribution_rows_csv(result["results"]), "text/csv"
    return format_pair_rows_csv(result["pairs"]), "text/csv"


def _files_for_class(root: Path, class_key: str, group_by: str):
    files = []
    for rel in _relative_csv_keys(root):
        try:
            key = class_key_for_relative_path(rel, group_by)
        except ValueError:
            continue
        if key == class_key:
            files.append(str(root / rel))
    return files


def render_job_overlay(job_id: str, key: str):
    job = get_job(job_id)
    if job is None or not job.get("result"):
        return None

    config = EvaluationConfig(**job["config"])
    result = job["result"]
    reference_dir = Path(job["referenceDir"])
    candidate_dirs = job.get("candidateDirs")

    if candidate_dirs:
        decoded = _decode_group_overlay_key(key)
        if decoded is None:
            return None
        group, inner_key = decoded
        candidate_dir_text = candidate_dirs.get(group)
        if candidate_dir_text is None:
            return None
        candidate_dir = Path(candidate_dir_text)
        if config.mode == "distribution":
            group_result = next(
                (
                    item
                    for item in result.get("groupResults", [])
                    if item.get("metadata", {}).get("comparisonGroup") == group
                ),
                None,
            )
            if group_result is None:
                return None
            rows = [
                row
                for row in group_result.get("results", {}).get("perClass", [])
                if row.get("classKey") == inner_key
            ]
            if not rows:
                return None
            reference_files = _files_for_class(reference_dir, inner_key, config.group_by)[:MAX_OVERLAY_FILES]
            candidate_files = _files_for_class(candidate_dir, inner_key, config.group_by)[:MAX_OVERLAY_FILES]
            label = f"{group}: {inner_key}"
        else:
            rows = [row for row in result.get("pairs", []) if row.get("overlayKey") == key]
            if not rows:
                return None
            row = rows[0]
            label = f"{group}: {row.get('label') or inner_key}"
            if config.mode == DIRECT_MODE:
                reference_files = [row["referenceFile"]]
            else:
                reference_files = [str(path) for path in sorted(reference_dir.rglob("*.csv"))[:MAX_OVERLAY_FILES]]
            candidate_files = [row["candidateFile"]]
    else:
        candidate_dir = Path(job["candidateDir"])

        if config.mode == "distribution":
            reference_files = _files_for_class(reference_dir, key, config.group_by)[:MAX_OVERLAY_FILES]
            candidate_files = _files_for_class(candidate_dir, key, config.group_by)[:MAX_OVERLAY_FILES]
            label = key
        else:
            rows = [row for row in result.get("pairs", []) if row.get("pairKey") == key]
            if not rows:
                return None
            row = rows[0]
            label = row.get("label") or key
            if config.mode == DIRECT_MODE:
                reference_files = [row["referenceFile"]]
            else:
                reference_files = [str(path) for path in sorted(reference_dir.rglob("*.csv"))[:MAX_OVERLAY_FILES]]
            candidate_files = [row["candidateFile"]]

    if not reference_files and not candidate_files:
        return None

    return render_overlay_svg(
        [
            OverlayGroup("Human", reference_files, "#126a73", width=1.55, alpha=0.58),
            OverlayGroup("Generated", candidate_files, "#b94722", width=1.8, alpha=0.72),
        ],
        label=str(label),
        rate=config.rate,
        alignment_type=config.alignment,
        summary_shape=config.summary,
        popular_shape=config.popular,
        include_reference_summary=True,
    )
