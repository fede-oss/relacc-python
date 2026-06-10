from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from relacc.canvas import OverlayGroup, render_overlay_svg


PROJECT_REPORT_ROOT = Path("/Users/fede/S6-Project/relacc-python")
REPORT_GLOB = "report-output-eval-*"
COMBINED_FILES = {
    "pairwise": "pairwise.csv",
    "baseline": "baseline.csv",
    "within_reference": "within_reference.csv",
    "within_comparison": "within_comparison.csv",
    "between_groups": "between_groups.csv",
    "distribution": "distribution.csv",
    "summary_distribution": "summary_distribution.csv",
    "aggregate_summaries": "aggregate_summaries.csv",
    "stats": "stats.csv",
    "baseline_stats": "baseline_stats.csv",
    "within_reference_stats": "within_reference_stats.csv",
    "within_comparison_stats": "within_comparison_stats.csv",
    "between_groups_stats": "between_groups_stats.csv",
}

METRIC_FAMILIES = {
    "core": ("shapeError", "lengthError", "bendingError", "velocityError", "dtwDistance", "curvature"),
    "shape": ("shapeError", "shapeVariability", "lengthError", "sizeError", "bendingError", "bendingVariability"),
    "timing": ("timeError", "timeVariability", "velocityError", "velocityVariability", "meanStrokeDuration"),
    "movement": ("cornerSlowdown", "twoThirdsPowerLawR2", "highFrequencyRatio", "curvature"),
    "stroke": ("strokeError", "strokeOrderError", "strokeLengthStd"),
    "dtw": ("dtwDistance", "ldtwDistance", "ddtwDistance", "wdtwDistance", "wddtwDistance"),
}


@dataclass(frozen=True)
class ReportCache:
    id: str
    root: Path
    combined: Path
    manifest: dict[str, Any]
    run: dict[str, Any]


def _json_safe(value: Any):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _report_search_roots() -> list[Path]:
    roots = [PROJECT_REPORT_ROOT]
    env_roots = os.environ.get("RELACC_REPORT_ROOTS", "")
    for raw in env_roots.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser())
    return list(dict.fromkeys(root for root in roots if root.exists()))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _report_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.name}--{digest}"


def _is_report_root(path: Path) -> bool:
    combined = path / "combined"
    return (path / "manifest.json").exists() and combined.exists() and (combined / "distribution.csv").exists()


def _discover_reports() -> list[ReportCache]:
    reports: list[ReportCache] = []
    for root in _report_search_roots():
        for path in sorted(root.glob(REPORT_GLOB), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_dir() or not _is_report_root(path):
                continue
            reports.append(
                ReportCache(
                    id=_report_id(path),
                    root=path,
                    combined=path / "combined",
                    manifest=_read_json(path / "manifest.json"),
                    run=_read_json(path / "run.json"),
                )
            )
    return reports


def list_report_caches() -> list[dict[str, Any]]:
    return [_report_card(report) for report in _discover_reports()]


def get_report_cache(report_id: str) -> ReportCache | None:
    for report in _discover_reports():
        if report.id == report_id or report.root.name == report_id:
            return report
    return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _report_card(report: ReportCache) -> dict[str, Any]:
    manifest = report.manifest
    run = report.run
    combined_outputs = manifest.get("combinedOutputs") or {}
    warnings = manifest.get("planningWarnings") or []
    completed_runs = [run for run in manifest.get("runs", []) if run.get("classCount")]
    datasets = sorted({item.get("dataset") for item in completed_runs if item.get("dataset")})
    sources = sorted({source_label(item) for item in completed_runs if item.get("source")})
    classes = sum(len(item.get("classes") or []) for item in completed_runs)
    size_bytes = sum(_file_size(report.combined / filename) for filename in COMBINED_FILES.values())
    return _json_safe(
        {
            "id": report.id,
            "name": report.root.name,
            "path": str(report.root),
            "summary": manifest.get("summary") or run.get("effectiveConfig", {}).get("summary"),
            "rate": manifest.get("rate") or run.get("effectiveConfig", {}).get("rate"),
            "alignment": manifest.get("alignment") or run.get("effectiveConfig", {}).get("alignment"),
            "sources": sources,
            "datasets": datasets,
            "classCount": classes,
            "warningCount": len(warnings),
            "plannedRunCount": manifest.get("plannedRunCount"),
            "plannedCandidateCount": manifest.get("plannedCandidateCount"),
            "createdUtc": run.get("createdUtc"),
            "gitHead": run.get("source", {}).get("gitHead"),
            "combinedOutputs": sorted(combined_outputs.keys()),
            "sizeBytes": size_bytes,
        }
    )


def source_label(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").strip()
    variant = str(row.get("variant") or "").strip()
    if variant and variant != "root":
        return f"{source}/{variant}"
    return source or "unknown"


def _csv_path(report: ReportCache, record_set: str) -> Path | None:
    filename = COMBINED_FILES.get(record_set)
    if not filename:
        return None
    path = report.combined / filename
    return path if path.exists() else None


def _read_rows(path: Path, limit: int | None = None) -> Iterable[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key in ("source", "dataset", "classKey", "metric"):
        value = filters.get(key)
        if value and str(row.get(key) or "") != str(value):
            return False
    variant = filters.get("variant")
    if variant and str(row.get("variant") or "root") != str(variant):
        return False
    return True


def _metric_family(filters: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, ...]:
    metric = filters.get("metric")
    if metric:
        return (str(metric),)
    family = filters.get("metricFamily") or "core"
    if family == "all":
        return tuple(manifest.get("metricNames") or [])
    return METRIC_FAMILIES.get(str(family), METRIC_FAMILIES["core"])


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return statistics.fmean(finite)


def _median(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return statistics.median(finite)


def report_summary(report: ReportCache) -> dict[str, Any]:
    card = _report_card(report)
    manifest = report.manifest
    run = report.run
    coverage = {
        "plannedRunCount": manifest.get("plannedRunCount"),
        "completedRunCount": len(manifest.get("runs") or []),
        "plannedCandidateCount": manifest.get("plannedCandidateCount"),
        "distributionSampleLimitPerClass": manifest.get("distributionSampleLimitPerClass"),
        "rawJsonl": manifest.get("rawJsonl"),
    }
    return _json_safe(
        {
            **card,
            "coverage": coverage,
            "metricNames": manifest.get("metricNames") or [],
            "distributionMetricNames": manifest.get("distributionMetricNames") or [],
            "warnings": (manifest.get("planningWarnings") or [])[:50],
            "command": run.get("execution", {}).get("command"),
            "effectiveConfig": run.get("effectiveConfig") or {},
        }
    )


def report_filters(report: ReportCache) -> dict[str, Any]:
    runs = report.manifest.get("runs") or []
    sources = sorted({source_label(row) for row in runs if row.get("source")})
    datasets = sorted({row.get("dataset") for row in runs if row.get("dataset")})
    variants = sorted({row.get("variant") or "root" for row in runs})
    class_by_dataset: dict[str, set[str]] = {}
    for run in runs:
        dataset = run.get("dataset")
        if not dataset:
            continue
        class_by_dataset.setdefault(str(dataset), set()).update(
            str(klass.get("classKey"))
            for klass in (run.get("classes") or [])
            if klass.get("classKey")
        )
    classes = sorted(
        {
            klass.get("classKey")
            for run in runs
            for klass in (run.get("classes") or [])
            if klass.get("classKey")
        }
    )
    return {
        "sources": sources,
        "datasets": datasets,
        "variants": variants,
        "classes": classes,
        "classByDataset": {dataset: sorted(values) for dataset, values in class_by_dataset.items()},
        "metrics": report.manifest.get("metricNames") or [],
        "distributionMetrics": report.manifest.get("distributionMetricNames") or [],
        "metricFamilies": sorted([*METRIC_FAMILIES.keys(), "all"]),
    }


def _ranking_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    path = _csv_path(report, "distribution")
    if path is None:
        return {"kind": "ranking", "rows": []}
    metrics = set(_metric_family(filters, report.manifest))
    grouped: dict[str, list[float]] = {}
    for row in _read_rows(path):
        if row.get("metric") not in metrics or not _matches(row, filters):
            continue
        score = _num(row.get("normalizedWassersteinDistance")) or _num(row.get("wassersteinDistance"))
        if score is None:
            continue
        grouped.setdefault(source_label(row), []).append(score)
    rows = [
        {"source": source, "score": _mean(values), "metricCount": len(values)}
        for source, values in grouped.items()
    ]
    rows.sort(key=lambda row: (row["score"] is None, row["score"] or 0, row["source"]))
    return {"kind": "ranking", "lowerIsBetter": True, "rows": _json_safe(rows[:20])}


def _heatmap_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    path = _csv_path(report, "distribution")
    if path is None:
        return {"kind": "heatmap", "sources": [], "metrics": [], "values": []}
    metrics = list(_metric_family(filters, report.manifest))[:12]
    metric_set = set(metrics)
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in _read_rows(path):
        if row.get("metric") not in metric_set or not _matches(row, filters):
            continue
        score = _num(row.get("normalizedWassersteinDistance")) or _num(row.get("wassersteinDistance"))
        if score is None:
            continue
        grouped.setdefault((source_label(row), row["metric"]), []).append(score)

    sources = sorted({source for source, _ in grouped})
    raw_values = {(source, metric): _mean(values) for (source, metric), values in grouped.items()}
    metric_values = [value for value in raw_values.values() if value is not None]
    center = _mean(metric_values) or 0.0
    spread = statistics.pstdev(metric_values) if len(metric_values) > 1 else 1.0
    spread = spread or 1.0
    values = []
    for source_index, source in enumerate(sources):
        for metric_index, metric in enumerate(metrics):
            raw = raw_values.get((source, metric))
            z_score = None if raw is None else (raw - center) / spread
            values.append(
                {
                    "source": source,
                    "metric": metric,
                    "sourceIndex": source_index,
                    "metricIndex": metric_index,
                    "value": z_score,
                    "raw": raw,
                }
            )
    return {"kind": "heatmap", "sources": sources, "metrics": metrics, "values": _json_safe(values)}


def _scatter_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    baseline_path = _csv_path(report, "baseline_stats")
    generated_path = _csv_path(report, "stats")
    if baseline_path is None or generated_path is None:
        return {"kind": "scatter", "points": []}
    metrics = set(_metric_family(filters, report.manifest))
    baseline: dict[tuple[str, str], float] = {}
    for row in _read_rows(baseline_path):
        if row.get("metric") not in metrics:
            continue
        key = (str(row.get("dataset")), str(row.get("classKey")), str(row.get("metric")))
        baseline[key] = _num(row.get("mdn")) or 0.0

    points = []
    for row in _read_rows(generated_path):
        if row.get("metric") not in metrics or not _matches(row, filters):
            continue
        key = (str(row.get("dataset")), str(row.get("classKey")), str(row.get("metric")))
        human = baseline.get(key)
        generated = _num(row.get("mdn"))
        if human is None or generated is None:
            continue
        points.append(
            {
                "source": source_label(row),
                "dataset": row.get("dataset"),
                "classKey": row.get("classKey"),
                "metric": row.get("metric"),
                "humanMedian": human,
                "generatedMedian": generated,
                "ratio": None if human == 0 else generated / human,
            }
        )
        if len(points) >= 1200:
            break
    return {"kind": "scatter", "points": _json_safe(points)}


def _distribution_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    path = _csv_path(report, "distribution")
    if path is None:
        return {"kind": "distribution", "rows": []}
    metric = filters.get("metric") or "shapeError"
    rows = []
    for row in _read_rows(path):
        if row.get("metric") != metric or not _matches(row, filters):
            continue
        rows.append(
            {
                "source": source_label(row),
                "dataset": row.get("dataset"),
                "classKey": row.get("classKey"),
                "withinReferenceMean": _num(row.get("withinReferenceMean")),
                "withinComparisonMean": _num(row.get("withinComparisonMean")),
                "betweenGroupsMean": _num(row.get("betweenGroupsMean")),
                "withinReferenceQ25": _num(row.get("withinReferenceQ25")),
                "withinReferenceQ75": _num(row.get("withinReferenceQ75")),
                "betweenGroupsQ25": _num(row.get("betweenGroupsQ25")),
                "betweenGroupsQ75": _num(row.get("betweenGroupsQ75")),
                "ratio": _num(row.get("withinComparisonToReferenceMeanRatio")),
                "wasserstein": _num(row.get("wassersteinDistance")),
            }
        )
        if len(rows) >= 500:
            break
    return {"kind": "distribution", "metric": metric, "rows": _json_safe(rows)}


def report_chart_payload(report: ReportCache, chart_kind: str, filters: dict[str, Any]) -> dict[str, Any] | None:
    if chart_kind == "ranking":
        return _ranking_payload(report, filters)
    if chart_kind == "heatmap":
        return _heatmap_payload(report, filters)
    if chart_kind == "scatter":
        return _scatter_payload(report, filters)
    if chart_kind == "distribution":
        return _distribution_payload(report, filters)
    return None


def report_table(
    report: ReportCache,
    record_set: str,
    filters: dict[str, Any],
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any] | None:
    path = _csv_path(report, record_set)
    if path is None:
        return None
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    rows = []
    matched = 0
    columns: list[str] = []
    for row in _read_rows(path):
        if not columns:
            columns = list(row.keys())
        if not _matches(row, filters):
            continue
        if matched >= offset and len(rows) < limit:
            rows.append(row)
        matched += 1
        if len(rows) >= limit and matched >= offset + limit:
            break
    return _json_safe(
        {
            "recordSet": record_set,
            "columns": columns,
            "rows": rows,
            "matched": matched,
            "limit": limit,
            "offset": offset,
        }
    )


def _resolve_report_path(report: ReportCache, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    candidates = [report.root.parent / path, report.root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _class_dir(report: ReportCache, source: str, dataset: str, variant: str | None, class_key: str) -> Path | None:
    variant = variant or "root"
    for run in report.manifest.get("runs") or []:
        if source_label(run) != source or run.get("dataset") != dataset:
            continue
        if str(run.get("variant") or "root") != variant and "/" not in source:
            continue
        for klass in run.get("classes") or []:
            if klass.get("classKey") == class_key:
                return _resolve_report_path(report, klass.get("outputDir"))
    return None


def _csv_values(path: Path, column: str) -> list[str]:
    if not path.exists():
        return []
    values = []
    for row in _read_rows(path):
        value = row.get(column)
        if value:
            values.append(value)
    return values


def report_overlay_svg(
    report: ReportCache,
    source: str,
    dataset: str,
    variant: str | None,
    class_key: str,
    comparison: str | None,
    sample_count: int,
    summary: str | None,
    reference_color: str,
    comparison_color: str,
) -> str | None:
    class_dir = _class_dir(report, source, dataset, variant, class_key)
    if class_dir is None:
        return None
    reference_files = _csv_values(class_dir / "baseline.csv", "sampleFile")[:sample_count]
    candidate_files = _csv_values(class_dir / "pairwise.csv", "candidateFile")[:sample_count]
    if comparison:
        candidate_files = [path for path in candidate_files if comparison in path][:sample_count] or candidate_files
    if not reference_files and not candidate_files:
        return None
    chosen_summary = summary or str(report.manifest.get("summary") or "medoid")
    return render_overlay_svg(
        [
            OverlayGroup("Reference", reference_files, reference_color, width=1.35, alpha=0.48, limit=sample_count),
            OverlayGroup("Comparison", candidate_files, comparison_color, width=1.65, alpha=0.62, limit=sample_count),
        ],
        label=class_key,
        rate=int(report.manifest.get("rate") or 24),
        alignment_type=int(report.manifest.get("alignment") or 0),
        summary_shape=chosen_summary,
        popular_shape=bool(report.manifest.get("popular")),
        include_reference_summary=True,
    )
