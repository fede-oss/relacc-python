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
from threading import Lock
from typing import Any

from relacc.canvas import OverlayGroup, render_overlay_svg
from relacc.metrics import METRIC_NAMES
from relacc.pipeline.distribution import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_PARENT_DIR,
    _class_key_for_relative_path,
    format_distribution_rows_csv,
    run_distribution_comparison,
)
from relacc.pipeline.pairwise import (
    DIRECT_MODE,
    SUMMARY_MODE,
    format_pair_rows_csv,
    run_pairwise_comparison,
)
from relacc.utils.csv import CSVUtil


IGNORED_NAMES = {".ds_store", "thumbs.db"}
MAX_OVERLAY_FILES = 18


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
LOCK = Lock()


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
        alignment=int(raw.get("alignment") or 0),
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

        for info in members:
            if info.is_dir() or _is_ignored_zip_member(info.filename):
                continue

            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                issues.append(_issue("error", scope, "Zip member has an unsafe path.", info.filename))
                continue
            if rel.suffix.lower() != ".csv":
                issues.append(_issue("warning", scope, "Ignored non-CSV file.", info.filename))
                continue

            destination = target_dir.joinpath(*rel.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(destination)

    if not extracted:
        issues.append(_issue("error", scope, "Zip archive contains no CSV gesture files."))

    return issues, extracted


def _relative_csv_keys(root: Path):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.csv"))


def _class_key_safe(relative_path: str, group_by: str):
    try:
        return _class_key_for_relative_path(relative_path, group_by)
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

        parent_key = _class_key_for_relative_path(rel, GROUP_BY_PARENT_DIR)
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
        ref_keys = set(_relative_csv_keys(reference_dir))
        cand_keys = set(_relative_csv_keys(candidate_dir))
        common = sorted(ref_keys & cand_keys)
        missing_candidate = sorted(ref_keys - cand_keys)
        missing_reference = sorted(cand_keys - ref_keys)
        if not common:
            issues.append(_issue("error", "matching", "No matching CSV paths found for direct pairwise mode."))
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
            "matchedPairCount": len(common),
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
    workdir = Path(tempfile.mkdtemp(prefix=f"relacc-web-{job_id}-"))
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
    return public_job(job)


def public_job(job: dict[str, Any]):
    safe = {
        key: value
        for key, value in job.items()
        if key not in {"workdir", "referenceDir", "candidateDir"}
    }
    return _json_safe(safe)


def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return job


def _update_job(job_id: str, **updates):
    with LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updatedAt"] = time.time()


def run_job(job_id: str):
    job = get_job(job_id)
    if job is None or job["status"] != "queued":
        return

    config = EvaluationConfig(**job["config"])
    reference_dir = job["referenceDir"]
    candidate_dir = job["candidateDir"]

    _update_job(job_id, status="running", phase="Preparing pipeline", progress=22)
    start = time.time()
    try:
        if config.mode == "distribution":
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


def overlay_keys_for_result(mode: str, result: dict[str, Any]):
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
            key = _class_key_for_relative_path(rel, group_by)
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
