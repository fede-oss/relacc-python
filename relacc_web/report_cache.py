from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
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

DISCOVERY_CACHE_TTL_SECONDS = 2.0

_CACHE_LOCK = RLock()
_DISCOVERY_CACHE: dict[str, Any] = {"key": None, "expires_at": 0.0, "reports": []}
_ROW_CACHE: dict[tuple[str, int, int], tuple[dict[str, str], ...]] = {}
_ROW_INDEX_CACHE: dict[
    tuple[tuple[str, int, int], tuple[str, ...]],
    dict[tuple[str, str], tuple[dict[str, str], ...]],
] = {}


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


def _discovery_cache_key(roots: Iterable[Path]) -> tuple[tuple[str, int], ...]:
    key = []
    for root in roots:
        try:
            resolved = root.resolve()
            stat = resolved.stat()
        except OSError:
            continue
        key.append((str(resolved), stat.st_mtime_ns))
    return tuple(key)


def clear_report_cache() -> None:
    with _CACHE_LOCK:
        _DISCOVERY_CACHE["key"] = None
        _DISCOVERY_CACHE["expires_at"] = 0.0
        _DISCOVERY_CACHE["reports"] = []
        _ROW_CACHE.clear()
        _ROW_INDEX_CACHE.clear()


def _cache_stats() -> dict[str, int]:
    with _CACHE_LOCK:
        return {
            "reports": len(_DISCOVERY_CACHE["reports"]),
            "rowFiles": len(_ROW_CACHE),
            "rowIndexes": len(_ROW_INDEX_CACHE),
        }


def _overlay_file_roots() -> list[Path]:
    roots = []
    env_roots = os.environ.get("RELACC_OVERLAY_FILE_ROOTS", "")
    for raw in env_roots.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser())
    return list(dict.fromkeys(root for root in roots if root.exists()))


def _resolve_existing(path: Path) -> Path | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return resolved if resolved.exists() else None


def _is_under(path: Path, roots: Iterable[Path]) -> bool:
    resolved_path = _resolve_existing(path)
    if resolved_path is None:
        return False
    for root in roots:
        resolved_root = _resolve_existing(root)
        if resolved_root is None:
            continue
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            continue
        return True
    return False


def _allowed_report_roots(report: ReportCache) -> list[Path]:
    return [report.root]


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


def _cached_discover_reports() -> list[ReportCache]:
    roots = _report_search_roots()
    cache_key = _discovery_cache_key(roots)
    now = time.monotonic()
    with _CACHE_LOCK:
        if _DISCOVERY_CACHE["key"] == cache_key and now < _DISCOVERY_CACHE["expires_at"]:
            return list(_DISCOVERY_CACHE["reports"])
        reports = _discover_reports()
        _DISCOVERY_CACHE["key"] = cache_key
        _DISCOVERY_CACHE["expires_at"] = now + DISCOVERY_CACHE_TTL_SECONDS
        _DISCOVERY_CACHE["reports"] = tuple(reports)
        return reports


def list_report_caches() -> list[dict[str, Any]]:
    return [_report_card(report) for report in _cached_discover_reports()]


def get_report_cache(report_id: str) -> ReportCache | None:
    for report in _cached_discover_reports():
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


def _row_cache_key(report: ReportCache, record_set: str) -> tuple[Path, tuple[str, int, int]] | None:
    if record_set not in COMBINED_FILES:
        return None
    path = _csv_path(report, record_set)
    if path is None:
        return None
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return resolved, (str(resolved), stat.st_mtime_ns, stat.st_size)


def _cached_rows(report: ReportCache, record_set: str) -> tuple[dict[str, str], ...]:
    key_parts = _row_cache_key(report, record_set)
    if key_parts is None:
        return ()
    resolved, cache_key = key_parts
    with _CACHE_LOCK:
        cached = _ROW_CACHE.get(cache_key)
        if cached is not None:
            return cached
        for key in [key for key in _ROW_CACHE if key[0] == str(resolved)]:
            _ROW_CACHE.pop(key, None)
        for key in [key for key in _ROW_INDEX_CACHE if key[0][0] == str(resolved)]:
            _ROW_INDEX_CACHE.pop(key, None)
        rows = tuple(_read_rows(resolved))
        _ROW_CACHE[cache_key] = rows
        return rows


def _row_index(
    report: ReportCache,
    record_set: str,
    fields: tuple[str, ...],
) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    key_parts = _row_cache_key(report, record_set)
    if key_parts is None:
        return {}
    _, row_key = key_parts
    index_key = (row_key, fields)
    with _CACHE_LOCK:
        cached = _ROW_INDEX_CACHE.get(index_key)
        if cached is not None:
            return cached
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _cached_rows(report, record_set):
        for field in fields:
            value = str(row.get(field) or "")
            if value:
                grouped.setdefault((field, value), []).append(row)
    index = {key: tuple(rows) for key, rows in grouped.items()}
    with _CACHE_LOCK:
        _ROW_INDEX_CACHE[index_key] = index
    return index


def _indexed_candidate_rows(
    report: ReportCache,
    record_set: str,
    filters: dict[str, Any],
    metrics: set[str],
    fields: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    rows = _cached_rows(report, record_set)
    if not rows:
        return ()
    index = _row_index(report, record_set, fields)
    candidates: list[tuple[dict[str, str], ...]] = []
    if "metric" in fields and metrics:
        metric_rows = []
        for metric in metrics:
            metric_rows.extend(index.get(("metric", metric), ()))
        if not metric_rows:
            return ()
        candidates.append(tuple(metric_rows))
    for field, filter_key in (("source", "source"), ("dataset", "dataset"), ("classKey", "classKey")):
        value = filters.get(filter_key)
        if field in fields and value:
            bucket = index.get((field, str(value)), ())
            if not bucket:
                return ()
            candidates.append(bucket)
    if not candidates:
        return rows
    return min(candidates, key=len)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _list_filter(filters: dict[str, Any], key: str) -> list[str]:
    value = filters.get(key)
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _matches(row: dict[str, Any], filters: dict[str, Any], *, metric: bool = True) -> bool:
    for key in ("source", "dataset", "classKey"):
        value = filters.get(key)
        if value and str(row.get(key) or "") != str(value):
            return False
    variant = filters.get("variant")
    if variant and str(row.get("variant") or "root") != str(variant):
        return False
    if metric:
        metrics = set(_list_filter(filters, "metrics"))
        if not metrics and filters.get("metric"):
            metrics = {str(filters["metric"])}
        if metrics and str(row.get("metric") or "") not in metrics:
            return False
    return True


def _metric_family(filters: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, ...]:
    metrics = _list_filter(filters, "metrics")
    if metrics:
        return tuple(metric for metric in metrics if metric in set(manifest.get("metricNames") or metrics))
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


def _weighted_accumulate(
    groups: dict[Any, dict[str, float]],
    key: Any,
    value: float | None,
    weight: float | None,
):
    if value is None:
        return
    weight = weight if weight and weight > 0 else 1.0
    bucket = groups.setdefault(key, {"weighted": 0.0, "weight": 0.0, "count": 0.0})
    bucket["weighted"] += value * weight
    bucket["weight"] += weight
    bucket["count"] += 1


def _weighted_rows(groups: dict[Any, dict[str, float]]) -> dict[Any, float | None]:
    return {
        key: None if bucket["weight"] <= 0 else bucket["weighted"] / bucket["weight"]
        for key, bucket in groups.items()
    }


def _distribution_score(row: dict[str, Any], filters: dict[str, Any]) -> float | None:
    metric_name = str(filters.get("distributionMetric") or "normalizedWassersteinDistance")
    score = _num(row.get(metric_name))
    return score if score is not None else _num(row.get("wassersteinDistance"))


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
    if _csv_path(report, "distribution") is None:
        return {"kind": "ranking", "rows": []}
    metrics = set(_metric_family(filters, report.manifest))
    rows = _indexed_candidate_rows(report, "distribution", filters, metrics, ("metric", "source", "dataset", "classKey"))
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("metric") not in metrics or not _matches(row, filters, metric=False):
            continue
        score = _distribution_score(row, filters)
        _weighted_accumulate(grouped, source_label(row), score, _num(row.get("betweenGroupsFiniteN")))
    scores = _weighted_rows(grouped)
    rows = [
        {
            "source": source,
            "score": score,
            "rowCount": int(grouped[source]["count"]),
            "weight": int(grouped[source]["weight"]),
        }
        for source, score in scores.items()
    ]
    rows.sort(key=lambda row: (row["score"] is None, row["score"] or 0, row["source"]))
    return {
        "kind": "ranking",
        "metric": filters.get("distributionMetric") or "normalizedWassersteinDistance",
        "gestureMetrics": sorted(metrics),
        "lowerIsBetter": True,
        "rows": _json_safe(rows[:20]),
    }


def _heatmap_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    if _csv_path(report, "distribution") is None:
        return {"kind": "heatmap", "sources": [], "metrics": [], "values": []}
    metrics = list(_metric_family(filters, report.manifest))[:18]
    metric_set = set(metrics)
    rows = _indexed_candidate_rows(report, "distribution", filters, metric_set, ("metric", "source", "dataset", "classKey"))
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("metric") not in metric_set or not _matches(row, filters, metric=False):
            continue
        score = _distribution_score(row, filters)
        _weighted_accumulate(grouped, (source_label(row), row["metric"]), score, _num(row.get("betweenGroupsFiniteN")))

    sources = sorted({source for source, _ in grouped})
    raw_values = _weighted_rows(grouped)
    metric_values = [value for value in raw_values.values() if value is not None]
    min_value = min(metric_values) if metric_values else 0.0
    max_value = max(metric_values) if metric_values else 1.0
    values = []
    for source_index, source in enumerate(sources):
        for metric_index, metric in enumerate(metrics):
            raw = raw_values.get((source, metric))
            values.append(
                {
                    "source": source,
                    "metric": metric,
                    "sourceIndex": source_index,
                    "metricIndex": metric_index,
                    "value": raw,
                    "raw": raw,
                    "rowCount": int(grouped.get((source, metric), {}).get("count", 0)),
                }
            )
    return {
        "kind": "heatmap",
        "metric": filters.get("distributionMetric") or "normalizedWassersteinDistance",
        "sources": sources,
        "metrics": metrics,
        "min": min_value,
        "max": max_value,
        "values": _json_safe(values),
    }


def _scatter_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    if _csv_path(report, "baseline_stats") is None or _csv_path(report, "stats") is None:
        return {"kind": "scatter", "points": []}
    selected_metrics = list(_metric_family(filters, report.manifest))
    metric = selected_metrics[0] if selected_metrics else "shapeError"
    metric_set = {metric}
    baseline_rows = _indexed_candidate_rows(
        report,
        "baseline_stats",
        {**filters, "source": ""},
        metric_set,
        ("metric", "dataset", "classKey", "source"),
    )
    generated_rows = _indexed_candidate_rows(report, "stats", filters, metric_set, ("metric", "dataset", "classKey", "source"))
    baseline: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in baseline_rows:
        if row.get("metric") != metric or not _matches(row, {**filters, "source": ""}, metric=False):
            continue
        key = (str(row.get("dataset")), str(row.get("classKey")), str(row.get("metric")))
        value = _num(row.get("mean"))
        if value is not None:
            baseline[key] = {"mean": value, "mdn": _num(row.get("mdn"))}

    points = []
    for row in generated_rows:
        if row.get("metric") != metric or not _matches(row, filters, metric=False):
            continue
        key = (str(row.get("dataset")), str(row.get("classKey")), str(row.get("metric")))
        human_stats = baseline.get(key)
        human = human_stats.get("mean") if human_stats else None
        generated = _num(row.get("mean"))
        if human is None or generated is None:
            continue
        generated_median = _num(row.get("mdn"))
        points.append(
            {
                "source": source_label(row),
                "dataset": row.get("dataset"),
                "classKey": row.get("classKey"),
                "metric": row.get("metric"),
                "humanMean": human,
                "generatedMean": generated,
                "humanMedian": human_stats.get("mdn"),
                "generatedMedian": generated_median,
                "n": _num(row.get("finiteN")) or _num(row.get("n")),
                "ratio": None if human == 0 else generated / human,
            }
        )
        if len(points) >= 1800:
            break
    return {"kind": "scatter", "metric": metric, "points": _json_safe(points)}


def _pairwise_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    if _csv_path(report, "baseline_stats") is None or _csv_path(report, "stats") is None:
        return {"kind": "pairwise", "rows": []}
    metrics = set(_metric_family(filters, report.manifest))
    baseline_rows = _indexed_candidate_rows(
        report,
        "baseline_stats",
        {**filters, "source": ""},
        metrics,
        ("metric", "dataset", "classKey", "source"),
    )
    generated_rows = _indexed_candidate_rows(report, "stats", filters, metrics, ("metric", "dataset", "classKey", "source"))
    baseline_groups: dict[str, dict[str, float]] = {}
    generated_groups: dict[tuple[str, str], dict[str, float]] = {}
    for row in baseline_rows:
        if row.get("metric") not in metrics or not _matches(row, {**filters, "source": ""}, metric=False):
            continue
        _weighted_accumulate(baseline_groups, row["metric"], _num(row.get("mean")), _num(row.get("finiteN")))
    baseline_means = _weighted_rows(baseline_groups)
    for row in generated_rows:
        if row.get("metric") not in metrics or not _matches(row, filters, metric=False):
            continue
        _weighted_accumulate(
            generated_groups,
            (source_label(row), row["metric"]),
            _num(row.get("mean")),
            _num(row.get("finiteN")),
        )
    generated_means = _weighted_rows(generated_groups)
    rows = []
    for (source, metric), generated in generated_means.items():
        human = baseline_means.get(metric)
        rows.append(
            {
                "source": source,
                "metric": metric,
                "humanMean": human,
                "candidateMean": generated,
                "ratio": None if not human else generated / human if generated is not None else None,
                "classCount": int(generated_groups[(source, metric)]["count"]),
                "n": int(generated_groups[(source, metric)]["weight"]),
            }
        )
    rows.sort(key=lambda row: (row["metric"], row["source"]))
    return {"kind": "pairwise", "rows": _json_safe(rows)}


def _distribution_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    if _csv_path(report, "distribution") is None:
        return {"kind": "distribution", "rows": []}
    metrics = set(_metric_family(filters, report.manifest))
    rows = _indexed_candidate_rows(report, "distribution", filters, metrics, ("metric", "source", "dataset", "classKey"))
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for row in rows:
        if row.get("metric") not in metrics or not _matches(row, filters, metric=False):
            continue
        key = (source_label(row), row["metric"])
        bucket = grouped.setdefault(key, {})
        for field, weight_field in (
            ("withinReferenceMean", "withinReferenceFiniteN"),
            ("withinComparisonMean", "withinComparisonFiniteN"),
            ("betweenGroupsMean", "betweenGroupsFiniteN"),
            ("withinComparisonToReferenceMeanRatio", "withinComparisonFiniteN"),
            ("normalizedWassersteinDistance", "betweenGroupsFiniteN"),
            ("wassersteinDistance", "betweenGroupsFiniteN"),
            ("energyDistance", "betweenGroupsFiniteN"),
            ("ksStatistic", "betweenGroupsFiniteN"),
        ):
            _weighted_accumulate(bucket, field, _num(row.get(field)), _num(row.get(weight_field)))
    means = {key: _weighted_rows(bucket) for key, bucket in grouped.items()}
    rows = []
    for (source, metric), values in means.items():
        rows.append(
            {
                "source": source,
                "metric": metric,
                "withinReferenceMean": values.get("withinReferenceMean"),
                "withinComparisonMean": values.get("withinComparisonMean"),
                "betweenGroupsMean": values.get("betweenGroupsMean"),
                "ratio": values.get("withinComparisonToReferenceMeanRatio"),
                "normalizedWassersteinDistance": values.get("normalizedWassersteinDistance"),
                "wassersteinDistance": values.get("wassersteinDistance"),
                "energyDistance": values.get("energyDistance"),
                "ksStatistic": values.get("ksStatistic"),
                "rowCount": int(grouped[(source, metric)].get("withinReferenceMean", {}).get("count", 0)),
            }
        )
    rows.sort(key=lambda row: (row["metric"], row["source"]))
    return {"kind": "distribution", "gestureMetrics": sorted(metrics), "rows": _json_safe(rows)}


def _histogram_payload(report: ReportCache, filters: dict[str, Any]) -> dict[str, Any]:
    metric = (list(_metric_family(filters, report.manifest)) or ["shapeError"])[0]
    dataset = filters.get("dataset")
    class_key = filters.get("classKey")
    source = filters.get("source")
    if not dataset or not class_key:
        return {"kind": "histogram", "metric": metric, "series": []}

    values_by_series: dict[str, list[float]] = {
        "within reference": [],
        "within comparison": [],
        "between groups": [],
    }
    record_sets = [
        ("within_reference", "within reference", False),
        ("within_comparison", "within comparison", True),
        ("between_groups", "between groups", True),
    ]
    for record_set, label, match_source in record_sets:
        cached_rows = _cached_rows(report, record_set)
        if not cached_rows:
            continue
        for row in cached_rows:
            if row.get("dataset") != dataset or row.get("classKey") != class_key:
                continue
            if filters.get("variant") and str(row.get("variant") or "root") != str(filters["variant"]):
                continue
            if match_source and source and source_label(row) != source:
                continue
            value = _num(row.get(metric))
            if value is not None:
                values_by_series[label].append(value)
            if len(values_by_series[label]) >= 12000:
                break

    finite_values = [value for values in values_by_series.values() for value in values]
    if not finite_values:
        return {"kind": "histogram", "metric": metric, "series": []}
    lo = min(finite_values)
    hi = max(finite_values)
    if lo == hi:
        hi = lo + 1
    bin_count = min(30, max(8, int(math.sqrt(len(finite_values)))))
    width = (hi - lo) / bin_count
    bins = [lo + width * index for index in range(bin_count + 1)]
    series = []
    for label, values in values_by_series.items():
        counts = [0] * bin_count
        for value in values:
            index = min(bin_count - 1, max(0, int((value - lo) / width)))
            counts[index] += 1
        series.append(
            {
                "name": label,
                "n": len(values),
                "mean": _mean(values),
                "bins": [
                    {
                        "x0": bins[index],
                        "x1": bins[index + 1],
                        "count": counts[index],
                    }
                    for index in range(bin_count)
                ],
            }
        )
    return {
        "kind": "histogram",
        "metric": metric,
        "dataset": dataset,
        "classKey": class_key,
        "source": source,
        "series": _json_safe(series),
    }


def report_chart_payload(report: ReportCache, chart_kind: str, filters: dict[str, Any]) -> dict[str, Any] | None:
    if chart_kind == "ranking":
        return _ranking_payload(report, filters)
    if chart_kind == "heatmap":
        return _heatmap_payload(report, filters)
    if chart_kind == "scatter":
        return _scatter_payload(report, filters)
    if chart_kind == "pairwise":
        return _pairwise_payload(report, filters)
    if chart_kind == "distribution":
        return _distribution_payload(report, filters)
    if chart_kind == "histogram":
        return _histogram_payload(report, filters)
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
    for row in _cached_rows(report, record_set):
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
    allowed_roots = _allowed_report_roots(report)
    if path.is_absolute():
        resolved = _resolve_existing(path)
        if resolved is None or not _is_under(resolved, allowed_roots):
            return None
        return resolved
    if ".." in path.parts:
        return None
    candidates = [report.root.parent / path, report.root / path]
    for candidate in candidates:
        resolved = _resolve_existing(candidate)
        if resolved is not None and _is_under(resolved, allowed_roots):
            return resolved
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


def _resolve_overlay_files(report: ReportCache, class_dir: Path, raw_paths: Iterable[str]) -> list[str]:
    allowed_roots = _allowed_report_roots(report) + [report.root.parent] + _overlay_file_roots()
    resolved_paths = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_absolute():
            candidates = [path]
        elif ".." in path.parts:
            candidates = []
        else:
            candidates = [class_dir / path, report.root.parent / path, report.root / path]
        for candidate in candidates:
            resolved = _resolve_existing(candidate)
            if resolved is not None and _is_under(resolved, allowed_roots):
                resolved_paths.append(str(resolved))
                break
    return resolved_paths


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
    summary_color: str = "#111514",
    show_reference: bool = True,
    show_comparison: bool = True,
    show_summary: bool = True,
    summary_source: str = "reference",
) -> str | None:
    class_dir = _class_dir(report, source, dataset, variant, class_key)
    if class_dir is None:
        return None
    reference_files = _resolve_overlay_files(
        report,
        class_dir,
        _csv_values(class_dir / "baseline.csv", "sampleFile"),
    )[:sample_count]
    candidate_files = _resolve_overlay_files(
        report,
        class_dir,
        _csv_values(class_dir / "pairwise.csv", "candidateFile"),
    )[:sample_count]
    if comparison:
        candidate_files = [path for path in candidate_files if comparison in path][:sample_count] or candidate_files
    if not reference_files and not candidate_files:
        return None
    chosen_summary = summary or str(report.manifest.get("summary") or "medoid")
    groups: list[OverlayGroup] = []
    needs_reference_for_summary = show_summary and summary_source in {"reference", "human", ""}
    if show_reference or needs_reference_for_summary:
        groups.append(
            OverlayGroup(
                "Reference",
                reference_files,
                reference_color,
                width=1.35 if show_reference else 0.01,
                alpha=0.48 if show_reference else 0.0,
                limit=sample_count,
            )
        )
    if show_comparison or (show_summary and summary_source == "comparison"):
        groups.append(
            OverlayGroup(
                "Comparison",
                candidate_files,
                comparison_color,
                width=1.65 if show_comparison else 0.01,
                alpha=0.62 if show_comparison else 0.0,
                limit=sample_count,
            )
        )
    if not groups:
        return None
    return render_overlay_svg(
        groups,
        label=class_key,
        rate=int(report.manifest.get("rate") or 24),
        alignment_type=int(report.manifest.get("alignment") or 0),
        summary_shape=chosen_summary,
        popular_shape=bool(report.manifest.get("popular")),
        include_reference_summary=show_summary,
        summary_source="comparison" if summary_source == "comparison" else "all" if summary_source == "all" else "reference",
        summary_color=summary_color,
    )
