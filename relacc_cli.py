#!/usr/bin/env python3
import argparse
import json

from relacc.gestures.ptaligntype import PtAlignType
from relacc.pipeline._common import effective_dtw_window, output_format
from relacc.pipeline.one_vs_many import (
    format_one_vs_many_result,
    json_safe,
    legacy_args_from_metadata,
    run_one_vs_many_comparison,
    summary_stats,
)
from relacc.utils.date import DateUtil
from relacc.utils.debug import Debug


def _bool_cast(value):
    return bool(value)


def _int_cast(value):
    if value is None:
        return None
    return int(value)


def getStats(arr):
    return summary_stats(arr)


def _resolve_dtw_window(rate, exact_dtw, requested_window):
    return effective_dtw_window(rate, requested_window, exact_dtw)


def toJSON(obj, defaults):
    meta = {"date": DateUtil.utc(), "time": DateUtil.now(), "args": defaults}
    payload = {"metadata": meta, "results": json_safe(obj)}
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

    str_args = []
    for arg in defaults:
        str_args.append('%s="%s"' % (arg, defaults[arg]))
    xml.append("%s<args %s />" % (indent(), " ".join(str_args)))

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
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(res)
        debug.fmt("Results were saved in %s", output)
    else:
        print(res)


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
    parser.add_argument("--round")
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

    if not opt.files:
        parser.print_help()
        raise ValueError("Please provide some gesture files as input.")

    debug = Debug({"verbose": bool(opt.verbose)})
    fmt = output_format(opt.output, opt.format)
    if fmt not in ["json", "csv", "xml"]:
        raise ValueError("Invalid output format (%s). Supported formats: json, csv, xml." % fmt)

    rate = _int_cast(opt.rate)
    parsed_alignment = _int_cast(opt.alignment)
    alignment = PtAlignType.CHRONOLOGICAL if parsed_alignment is None else parsed_alignment
    parsed_round = _int_cast(opt.round)
    round_precision = 3 if parsed_round is None else parsed_round
    dtw_window = _int_cast(opt.dtw_window)

    payload = run_one_vs_many_comparison(
        opt.files,
        label=opt.label,
        rate=rate,
        alignment_type=alignment,
        summary_shape=opt.summary,
        popular_shape=opt.popular,
        stats=bool(opt.stats),
        round_precision=round_precision,
        dtw_window=dtw_window,
        exact_dtw=bool(opt.exact_dtw),
    )

    if opt.label is None:
        debug.fmt(
            "Notice: No gesture label provided, I'll assume that all samples are '%s'.",
            payload["metadata"]["label"],
        )
    if rate is None:
        debug.fmt(
            "Notice: Setting sampling rate to %s points per gesture.",
            payload["metadata"]["rate"],
        )

    legacy_args = legacy_args_from_metadata(payload, output=opt.output, fmt=fmt)
    result = format_one_vs_many_result(payload, fmt, legacy_args=legacy_args)
    displayResults(result, opt.output, debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
