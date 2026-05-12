from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence
from xml.sax.saxutils import escape, quoteattr

from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import METRIC_NAMES, compute_metrics
from relacc.utils.date import DateUtil
from relacc.utils.math import MathUtil

from ._common import (
    effective_dtw_window,
    format_csv_rows,
    infer_label_from_filename,
    normalize_summary_shape,
    read_gesture_points,
    sampling_rate_for_sets,
)


ONE_VS_MANY_MODE = "one-vs-many"
STATS_COLUMNS = ("measure", "n", "mean", "mdn", "sd", "min", "max")


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def summary_stats(values: Sequence[float], round_precision: int | None = 3):
    n = len(values)
    if values and any(not math.isfinite(value) for value in values):
        mean = float("nan")
        mdn = float("nan")
        sd = float("nan")
        minimum = float("nan")
        maximum = float("nan")
    elif n > 0:
        mean = statistics.fmean(values)
        mdn = statistics.median(values)
        sd = statistics.stdev(values) if n > 1 else 0
        minimum = min(values)
        maximum = max(values)
    else:
        mean = 0
        mdn = 0
        sd = 0
        minimum = 0
        maximum = 0

    return {
        "mean": MathUtil.roundTo(mean, round_precision),
        "mdn": MathUtil.roundTo(mdn, round_precision),
        "sd": MathUtil.roundTo(sd, round_precision),
        "min": MathUtil.roundTo(minimum, round_precision),
        "max": MathUtil.roundTo(maximum, round_precision),
        "n": n,
    }


def _file_key(file_path: str) -> str:
    return Path(file_path).with_suffix("").name


def _load_gesture_entries(files: Sequence[str]):
    if not files:
        raise ValueError("Please provide some gesture files as input.")

    entries = []
    for file_path in files:
        entries.append(
            {
                "key": _file_key(file_path),
                "path": str(file_path),
                "points": read_gesture_points(str(file_path)),
            }
        )
    return entries


def _effective_label(files: Sequence[str], label: str | None) -> str:
    if label:
        return label
    return infer_label_from_filename(Path(files[0]).name, "gesture")


def run_one_vs_many_comparison(
    files: Sequence[str],
    label: str | None = None,
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    stats: bool = False,
    round_precision: int = 3,
    metric_names: Sequence[str] | None = None,
    dtw_window: int | None = None,
    exact_dtw: bool = False,
):
    if dtw_window is not None and exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")

    summary_shape = normalize_summary_shape(summary_shape)
    entries = _load_gesture_entries(files)
    selected_label = _effective_label(files, label)
    selected_metric_names = tuple(metric_names or METRIC_NAMES)
    point_sets = [entry["points"] for entry in entries]
    effective_rate = sampling_rate_for_sets(point_sets, rate)
    selected_dtw_window = effective_dtw_window(effective_rate, dtw_window, exact_dtw)

    gestures = [
        Gesture(entry["points"], selected_label, effective_rate)
        for entry in entries
    ]
    summary = SummaryGesture(
        gestures,
        alignment_type,
        summary_shape,
        popular_shape,
    )

    metric_values = {name: [] for name in selected_metric_names}
    rows = []
    for entry, gesture in zip(entries, gestures):
        values = compute_metrics(
            gesture,
            summary,
            round_precision=round_precision,
            metric_names=selected_metric_names,
            dtw_window=selected_dtw_window,
        )
        for name, value in values.items():
            metric_values[name].append(value)

        row = {
            "file": entry["key"],
            "inputFile": entry["path"],
            "label": selected_label,
            "rate": effective_rate,
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "dtwWindow": selected_dtw_window,
        }
        row.update(values)
        rows.append(row)

    aggregate_stats = {
        name: summary_stats(metric_values[name], round_precision)
        for name in selected_metric_names
    }

    return {
        "metadata": {
            "comparisonMode": ONE_VS_MANY_MODE,
            "sampleCount": len(rows),
            "label": selected_label,
            "rate": effective_rate,
            "requestedRate": rate,
            "alignment": alignment_type,
            "summary": summary_shape,
            "popular": bool(popular_shape),
            "stats": bool(stats),
            "roundPrecision": round_precision,
            "metricNames": list(selected_metric_names),
            "dtwWindow": selected_dtw_window,
            "exactDtw": bool(exact_dtw),
        },
        "samples": rows,
        "results": aggregate_stats if stats else rows,
    }


def legacy_args_from_metadata(payload: Dict[str, object], output=None, fmt: str = "json"):
    metadata = payload["metadata"]
    return {
        "label": metadata["label"],
        "rate": metadata["rate"],
        "alignment": metadata["alignment"],
        "summary": metadata["summary"],
        "popular": metadata["popular"],
        "stats": metadata["stats"],
        "output": output,
        "format": fmt,
        "exact_dtw": metadata["exactDtw"],
        "dtw_window": metadata["dtwWindow"],
    }


def format_one_vs_many_json(payload, legacy_args: Dict[str, object] | None = None):
    if legacy_args is not None:
        metadata = {
            "date": DateUtil.utc(),
            "time": DateUtil.now(),
            "args": legacy_args,
            **payload["metadata"],
        }
        payload = {**payload, "metadata": metadata}
    return json.dumps(json_safe(payload), allow_nan=False)


def format_one_vs_many_stats_csv(results: Dict[str, Dict[str, object]]) -> str:
    rows = [{"measure": name, **stats} for name, stats in results.items()]
    return format_csv_rows(rows, STATS_COLUMNS)


def format_one_vs_many_samples_csv(
    rows: Sequence[Dict[str, object]],
    metric_names: Sequence[str] | None = None,
) -> str:
    selected_metric_names = tuple(metric_names or METRIC_NAMES)
    columns: List[str] = [
        "file",
        "inputFile",
        "label",
        "rate",
        "alignment",
        "summary",
        "popular",
        "dtwWindow",
        *selected_metric_names,
    ]
    return format_csv_rows(rows, columns)


def _xml_attr(name, value):
    if value is None:
        value = ""
    if isinstance(value, float) and not math.isfinite(value):
        value = ""
    return '%s=%s' % (name, quoteattr(str(value)))


def format_one_vs_many_xml(payload, legacy_args: Dict[str, object] | None = None) -> str:
    metadata = payload["metadata"]
    metric_names = tuple(metadata["metricNames"])
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<root>"]
    lines.append(
        '  <metadata %s />'
        % " ".join(_xml_attr(key, value) for key, value in metadata.items() if key != "metricNames")
    )

    if legacy_args is not None:
        lines.append(
            '  <args %s />'
            % " ".join(_xml_attr(key, value) for key, value in legacy_args.items())
        )

    lines.append("  <results>")
    if metadata["stats"]:
        for measure, stats in payload["results"].items():
            attrs = [_xml_attr("measure", measure)]
            attrs.extend(_xml_attr(key, stats[key]) for key in STATS_COLUMNS if key != "measure")
            lines.append("    <metric %s />" % " ".join(attrs))
    else:
        for row in payload["results"]:
            attrs = [
                _xml_attr("file", row.get("file")),
                _xml_attr("inputFile", row.get("inputFile")),
                _xml_attr("label", row.get("label")),
                _xml_attr("rate", row.get("rate")),
                _xml_attr("alignment", row.get("alignment")),
                _xml_attr("summary", row.get("summary")),
                _xml_attr("popular", row.get("popular")),
                _xml_attr("dtwWindow", row.get("dtwWindow")),
            ]
            lines.append("    <sample %s>" % " ".join(attrs))
            for metric_name in metric_names:
                lines.append(
                    "      <metric name=%s value=%s />"
                    % (
                        quoteattr(escape(metric_name)),
                        quoteattr(str(row.get(metric_name, ""))),
                    )
                )
            lines.append("    </sample>")
    lines.append("  </results>")
    lines.append("</root>")
    return "\n".join(lines)


def format_one_vs_many_result(
    payload,
    fmt: str,
    legacy_args: Dict[str, object] | None = None,
) -> str:
    if fmt == "json":
        return format_one_vs_many_json(payload, legacy_args=legacy_args)
    if fmt == "csv":
        if payload["metadata"]["stats"]:
            return format_one_vs_many_stats_csv(payload["results"])
        return format_one_vs_many_samples_csv(
            payload["results"],
            metric_names=payload["metadata"]["metricNames"],
        )
    if fmt == "xml":
        return format_one_vs_many_xml(payload, legacy_args=legacy_args)
    raise ValueError("Invalid output format (%s). Supported formats: json, csv, xml." % fmt)

