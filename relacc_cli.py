#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
from typing import Dict, List

from relacc.geom.pointset import PointSet
from relacc.dtw import (
    DEFAULT_EXACT_RATE_THRESHOLD,
    recommended_window as recommended_dtw_window,
)
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.metrics import get_metric_names, compute_metrics
from relacc.utils.args import Args
from relacc.utils.csv import CSVUtil
from relacc.utils.date import DateUtil
from relacc.utils.debug import Debug
from relacc.utils.json import JSONUtil
from relacc.utils.math import MathUtil


def _bool_cast(value):
    return bool(value)


def _int_cast(value):
    if value is None:
        return None
    return int(value)


def getStats(arr):
    finite_values = [value for value in arr if math.isfinite(value)]
    n = len(finite_values)
    mean = 0
    mdn = 0
    sd = 0
    minimum = 0
    maximum = 0
    if finite_values:
        mean = statistics.fmean(finite_values)
        mdn = statistics.median(finite_values)
        sd = statistics.stdev(finite_values) if n > 1 else 0
        minimum = min(finite_values)
        maximum = max(finite_values)
    elif arr:
        mean = float("nan")
        mdn = float("nan")
        sd = float("nan")
        minimum = float("nan")
        maximum = float("nan")

    return {
        "mean": MathUtil.roundTo(mean),
        "mdn": MathUtil.roundTo(mdn),
        "sd": MathUtil.roundTo(sd),
        "min": MathUtil.roundTo(minimum),
        "max": MathUtil.roundTo(maximum),
        "n": n,
    }


def _resolve_dtw_window(rate, exact_dtw, requested_window):
    if exact_dtw:
        return None
    if requested_window is not None:
        return requested_window
    if rate <= DEFAULT_EXACT_RATE_THRESHOLD:
        return None
    return recommended_dtw_window(rate)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def toJSON(obj, defaults):
    meta = {"date": DateUtil.utc(), "time": DateUtil.now(), "args": defaults}
    payload = {"metadata": meta, "results": _json_safe(obj)}
    return json.dumps(payload, allow_nan=False)


def toCSV(obj):
    sep = " "
    lines = ["measure n mean mdn sd min max".split(" ")]
    csv_lines = [sep.join(lines[0])]
    for key in obj:
        val = obj[key]
        csv_lines.append(
            "%s %d %s %s %s %s %s"
            % (key, val["n"], val["mean"], val["mdn"], val["sd"], val["min"], val["max"])
        )
    return "\n".join(csv_lines)


def toXML(obj, defaults):
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']

    def indent(level=2):
        return "  " * (level - 1)

    xml.append("<root>")
    xml.append('%s<metadata date="%s" time="%d" />' % (indent(), DateUtil.utc(), DateUtil.now()))

    strArgs = []
    for a in defaults:
        strArgs.append('%s="%s"' % (a, defaults[a]))
    xml.append("%s<args %s />" % (indent(), " ".join(strArgs)))

    xml.append("%s<results>" % indent())
    for key in obj:
        val = obj[key]
        entry = (
            '%s<%s n="%d" mean="%s" mdn="%s" sd="%s" min="%s" max="%s" />'
            % (indent(3), key, val["n"], val["mean"], val["mdn"], val["sd"], val["min"], val["max"])
        )
        xml.append(entry)
    xml.append("%s</results>" % indent())
    xml.append("</root>")
    return "\n".join(xml)


def displayResults(res, output, debug):
    if output:
        try:
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(res)
            debug.fmt("Results were saved in %s", output)
        except OSError:
            debug.fmt("Cannot write to %s", output)
    else:
        print(res)


def evaluate(collection, label, rate, argParser, defaults, output, fmt, debug):
    alignmentType = argParser.get("alignment", defaults["alignment"], _int_cast)
    summaryShape = argParser.get("summary", defaults["summary"])
    popularShape = argParser.get("popular", defaults["popular"])
    displayStats = argParser.get("stats", defaults["stats"], _bool_cast)
    exactDtw = argParser.get("exact_dtw", defaults["exact_dtw"], _bool_cast)
    requestedDtwWindow = argParser.get("dtw_window", defaults["dtw_window"], _int_cast)
    dtwWindow = _resolve_dtw_window(rate, exactDtw, requestedDtwWindow)

    defaults["alignment"] = alignmentType
    defaults["summary"] = summaryShape
    defaults["popular"] = popularShape
    defaults["stats"] = displayStats
    defaults["exact_dtw"] = exactDtw
    defaults["dtw_window"] = dtwWindow

    gestures = []
    files = list(collection.keys())
    for file in files:
        points = collection[file]
        gestures.append(Gesture(points, label, rate))

    taskAxis = SummaryGesture(gestures, alignmentType, summaryShape, popularShape)
    metric_names = get_metric_names()
    metric_values = {name: [] for name in metric_names}
    for gesture in gestures:
        values = compute_metrics(
            gesture,
            taskAxis,
            metric_names=metric_names,
            dtw_window=dtwWindow,
        )
        for name, value in values.items():
            metric_values[name].append(value)

    if displayStats:
        stats = {name: getStats(metric_values[name]) for name in metric_names}
        if fmt == "json":
            res = toJSON(stats, defaults)
        elif fmt == "csv":
            res = toCSV(stats)
        elif fmt == "xml":
            res = toXML(stats, defaults)
        else:
            raise ValueError(
                "Invalid output format (%s). Supported formats: json, csv, xml." % fmt
            )
        displayResults(res, output, debug)
    else:
        print("file", *metric_names)
        for i in range(len(files)):
            rounded_values = [MathUtil.roundTo(metric_values[name][i]) for name in metric_names]
            print(files[i], *rounded_values)


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-l", "--label")
    parser.add_argument("-r", "--rate")
    parser.add_argument("-a", "--alignment")
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("-s", "--stats", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("-f", "--format")
    parser.add_argument("--exact-dtw", action="store_true")
    parser.add_argument("--dtw-window")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)

    if opt.help:
        parser.print_help()
        return 0

    args = vars(opt)
    argParser = Args(args)
    debug = Debug({"verbose": bool(args.get("verbose"))})

    defaults = {
        "label": None,
        "rate": None,
        "alignment": PtAlignType.CHRONOLOGICAL,
        "summary": None,
        "popular": False,
        "stats": False,
        "output": None,
        "format": "json",
        "exact_dtw": False,
        "dtw_window": None,
    }

    files = opt.files
    nfiles = len(files)
    label = argParser.get("label")

    rate = argParser.get("rate")
    rate = int(rate) if rate not in (None, "") else None

    output = argParser.get("output")
    dtw_window = argParser.get("dtw_window", defaults["dtw_window"], _int_cast)
    if output:
        fmt = os.path.splitext(output)[1][1:].lower()
    else:
        fmt = (argParser.get("format", defaults["format"]) or defaults["format"]).lower()

    if dtw_window is not None and args.get("exact_dtw"):
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")

    defaults["label"] = label
    defaults["rate"] = rate
    defaults["output"] = output
    defaults["format"] = fmt
    defaults["exact_dtw"] = bool(args.get("exact_dtw"))
    defaults["dtw_window"] = dtw_window

    if not nfiles:
        parser.print_help()
        raise ValueError("Please provide some gesture files as input.")

    if not label:
        label = os.path.basename(files[0]).split("-")[1]
        defaults["label"] = label
        debug.fmt("Notice: No gesture label provided, I'll assume that all samples are '%s'.", label)

    maxStrokeCount = 1
    collection: Dict[str, List] = {}

    def doneParsing(file, points):
        nonlocal maxStrokeCount, rate
        strokeCount = PointSet.countStrokes(points)
        if strokeCount > maxStrokeCount:
            maxStrokeCount = strokeCount
        ext = os.path.splitext(file)[1]
        collection[os.path.basename(file[: -len(ext)]) if ext else os.path.basename(file)] = points

        if len(collection.keys()) == nfiles:
            debug.fmt("Processed %s files", nfiles)
            if not rate or (isinstance(rate, float) and math.isnan(rate)):
                smartRate = max(24, MathUtil.factorial(maxStrokeCount))
                debug.fmt("Notice: Setting sampling rate to %s points per gesture.", smartRate)
                rate = smartRate
                defaults["rate"] = smartRate
            evaluate(collection, label, rate, argParser, defaults, output, fmt, debug)

    for file in files:
        ext = os.path.splitext(file)[1]
        if ext == ".csv":
            CSVUtil.readGesture(file, lambda points, file=file: doneParsing(file, points))
        elif ext == ".json":
            JSONUtil.readGesture(file, lambda points, file=file: doneParsing(file, points))
        else:
            raise ValueError(
                "Invalid input file format (%s). Supported formats: json, csv." % ext
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
