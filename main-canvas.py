#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relacc.geom.point import Point
from relacc.geom.pointset import PointSet
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.utils.args import Args
from relacc.utils.csv import CSVUtil
from relacc.utils.debug import Debug
from relacc.utils.math import MathUtil


def _bool_cast(value):
    return bool(value)


def _int_cast(value):
    if value is None:
        return None
    return int(value)


def _readJsonStrokes(strokes, callback):
    points = []
    for i, stroke in enumerate(strokes):
        for j in range(len(stroke)):
            x = stroke[j][0]
            y = stroke[j][1]
            t = stroke[j][2]
            points.append(Point(x, y, t, i))
    callback(points)


def _readJsonFile(file, callback):
    with open(file, "r", encoding="utf-8") as fh:
        strokes = json.load(fh)
    _readJsonStrokes(strokes, callback)


def _parse_color(color):
    if isinstance(color, str) and color.startswith("rgba(") and color.endswith(")"):
        body = color[5:-1]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) == 4:
            r = float(parts[0]) / 255.0
            g = float(parts[1]) / 255.0
            b = float(parts[2]) / 255.0
            a = float(parts[3])
            return (r, g, b, a)
    return color


def _make_canvas(imsize, fmt):
    dpi = 100
    fig = plt.figure(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, imsize)
    ax.set_ylim(imsize, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if fmt in ["jpg", "jpeg"]:
        fig.patch.set_facecolor("white")
    return fig, ax


def _drawGesture(ax, points, canvasSize, lineWidth, lineColor):
    trPts = PointSet.clone(points)
    bounds = PointSet.boundingBox(trPts)
    boundsDiff = max(canvasSize - bounds.width(), canvasSize - bounds.height())
    scale = boundsDiff / canvasSize if boundsDiff > 0 else canvasSize / boundsDiff
    padScale = 0.9 * (1 + scale)
    trPts = PointSet.scaleTo(trPts, padScale)
    offPt = Point(-canvasSize / 2, -canvasSize / 2)
    trPts = PointSet.translateBy(trPts, offPt)

    if len(trPts) == 0:
        return
    lineColor = _parse_color(lineColor)

    strokeX = [trPts[0].X]
    strokeY = [trPts[0].Y]
    currStroke = trPts[0].StrokeID
    for i in range(1, len(trPts)):
        pt = trPts[i]
        if pt.StrokeID != currStroke:
            if len(strokeX) > 1:
                ax.plot(strokeX, strokeY, linewidth=lineWidth, color=lineColor, solid_capstyle="round")
            strokeX = [pt.X]
            strokeY = [pt.Y]
            currStroke = pt.StrokeID
        else:
            strokeX.append(pt.X)
            strokeY.append(pt.Y)

    if len(strokeX) > 1:
        ax.plot(strokeX, strokeY, linewidth=lineWidth, color=lineColor, solid_capstyle="round")


def _displayResult(fig, output, fmt, debug):
    if output:
        if fmt in ["png", "jpg", "jpeg", "pdf", "svg"]:
            save_fmt = "jpeg" if fmt == "jpg" else fmt
            fig.savefig(output, format=save_fmt, dpi=100, bbox_inches="tight", pad_inches=0)
            debug.fmt("Results were saved in %s", output)
        else:
            raise ValueError(
                "Invalid image format (%s). Supported formats: jpg, jpeg, png, pdf, svg." % fmt
            )
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        print('<img src="data:image/png;base64,' + b64 + '" />')


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-l", "--label")
    parser.add_argument("-r", "--rate")
    parser.add_argument("-a", "--alignment")
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("-f", "--format")
    parser.add_argument("-s", "--size")
    parser.add_argument("-t", "--thickness")
    parser.add_argument("-c", "--color")
    parser.add_argument("-T", "--summary-thickness")
    parser.add_argument("-C", "--summary-color")
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
        "size": None,
        "alignment": PtAlignType.CHRONOLOGICAL,
        "summary": None,
        "popular": False,
        "output": None,
        "format": "csv",
        "thickness": 1,
        "color": "rgba(0,0,0, .5)",
        "summaryThickness": 10,
        "summaryColor": "#F00",
    }

    files = opt.files
    nfiles = len(files)
    label = defaults["label"] = argParser.get("label")

    rate = argParser.get("rate")
    rate = int(rate) if rate not in (None, "") else None
    defaults["rate"] = rate

    output = defaults["output"] = argParser.get("output", defaults["output"])
    fmt = defaults["format"] = (
        os.path.splitext(output)[1][1:].lower() if output else (argParser.get("format", "img") or "img").lower()
    )

    imsize = defaults["size"] = argParser.get("size", 500, _int_cast)
    alignmentType = defaults["alignment"] = argParser.get("alignment", defaults["alignment"], _int_cast)
    summaryShape = defaults["summary"] = argParser.get("summary", defaults["summary"])
    popularShape = defaults["popular"] = argParser.get("popular", defaults["popular"], _bool_cast)
    lineWidth = defaults["thickness"] = argParser.get("thickness", defaults["thickness"], _int_cast)
    lineColor = defaults["color"] = argParser.get("color", defaults["color"])
    summaryWidth = defaults["summaryThickness"] = argParser.get(
        "summary_thickness", defaults["summaryThickness"], _int_cast
    )
    summaryColor = defaults["summaryColor"] = argParser.get("summary_color", defaults["summaryColor"])

    collection = []

    if not nfiles:
        parser.print_help()
        raise ValueError("Please provide some gesture files as input.")

    if not label:
        label = os.path.basename(files[0]).split("-")[1]
        defaults["label"] = label
        debug.fmt("Notice: No gesture label provided, I'll assume that all samples are '%s'.", label)

    maxStrokeCount = 1

    def evaluate():
        fig, ax = _make_canvas(imsize, fmt)

        gestures = [Gesture(points, label, rate) for points in collection]
        summaryGesture = SummaryGesture(gestures, alignmentType, summaryShape, popularShape)
        for gesture in gestures:
            gesturePts = summaryGesture.alignGesture(gesture, alignmentType)
            _drawGesture(ax, gesturePts, imsize, lineWidth, lineColor)

        if summaryShape:
            summaryPts = summaryGesture.getPoints()
            _drawGesture(ax, summaryPts, imsize, summaryWidth, summaryColor)

        _displayResult(fig, output, fmt, debug)
        plt.close(fig)

    def doneParsing(points):
        nonlocal maxStrokeCount, rate
        strokeCount = PointSet.countStrokes(points)
        if strokeCount > maxStrokeCount:
            maxStrokeCount = strokeCount

        collection.append(points)
        if len(collection) == nfiles:
            debug.fmt("Processed %s gesture files.", nfiles)
            if not rate:
                smartRate = max(24, MathUtil.factorial(maxStrokeCount))
                debug.fmt("Notice: Setting sampling rate to %s points per gesture.", smartRate)
                rate = smartRate
                defaults["rate"] = smartRate
            print(defaults, file=sys.stderr)
            evaluate()

    if files[0] == "-":
        pending = []
        nfiles_local = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            strokes = json.loads(line)
            pending.append(strokes)
            nfiles_local += 1

        nfiles = nfiles_local
        for strokes in pending:
            _readJsonStrokes(strokes, doneParsing)
    else:
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext == ".csv":
                CSVUtil.readGesture(file, doneParsing)
            elif ext == ".json":
                _readJsonFile(file, doneParsing)
            else:
                debug.fmt("Unknown file extension: %s", ext)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
