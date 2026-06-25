#!/usr/bin/env python3
import argparse
import json

from relacc.gestures.ptaligntype import PtAlignType
from relacc.pipeline._common import (
    default_raw_output_path,
    effective_dtw_window,
    output_format,
    write_jsonl_rows,
)
from relacc.pipeline.one_vs_many import (
    TEXT_FORMATS,
    format_one_vs_many_result,
    json_safe,
    legacy_args_from_metadata,
    run_one_vs_many_comparison,
    summary_stats,
)
from relacc.utils.date import DateUtil
from relacc.utils.debug import Debug
from relacc.utils.runlog import (
    add_run_logging_arguments,
    append_run_log,
    build_run_metadata,
    record_effective_config,
    run_logged_experiment,
    sidecar_paths,
    verbosity_from_opt,
    write_run_metadata,
)


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


def _get_format(output, requested_format, stats):
    if output is None and requested_format is None and not stats:
        return "text"
    return output_format(output, requested_format)


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
    parser.add_argument("-a", "--alignment", type=PtAlignType.normalize)
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("-s", "--stats", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("--raw-output")
    parser.add_argument("-f", "--format")
    parser.add_argument("--round")
    parser.add_argument("--exact-dtw", action="store_true")
    parser.add_argument("--dtw-window")
    add_run_logging_arguments(parser)
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

    paths = sidecar_paths(opt.output, opt.log_dir, stem="one-vs-many")
    metadata = build_run_metadata(parser, opt, argv, "one-vs-many")
    write_run_metadata(paths, metadata)
    return run_logged_experiment(paths, lambda: _run_experiment(opt, paths, metadata))


def _run_experiment(opt, paths=None, metadata=None):
    debug = Debug({"verbose": verbosity_from_opt(opt)})
    fmt = _get_format(opt.output, opt.format, opt.stats)
    if fmt not in ["json", "csv", "xml", *TEXT_FORMATS]:
        raise ValueError(
            "Invalid output format (%s). Supported formats: json, csv, xml, text." % fmt
        )

    rate = _int_cast(opt.rate)
    alignment = PtAlignType.normalize(
        PtAlignType.CHRONOLOGICAL if opt.alignment is None else opt.alignment
    )
    parsed_round = _int_cast(opt.round)
    round_precision = 3 if parsed_round is None else parsed_round
    dtw_window = _int_cast(opt.dtw_window)
    effective_config = {
        "format": fmt,
        "label": opt.label,
        "rate": rate,
        "alignment": alignment,
        "alignmentName": PtAlignType.name(alignment),
        "summary": opt.summary,
        "popular": bool(opt.popular),
        "stats": bool(opt.stats),
        "roundPrecision": round_precision,
        "dtwWindow": dtw_window,
        "exactDtw": bool(opt.exact_dtw),
        "output": opt.output,
        "files": list(opt.files),
        "verbosity": verbosity_from_opt(opt),
    }

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
        effective_config["label"] = payload["metadata"]["label"]
        debug.fmt(
            "Notice: No gesture label provided, I'll assume that all samples are '%s'.",
            payload["metadata"]["label"],
        )
    if rate is None:
        effective_config["rate"] = payload["metadata"]["rate"]
        debug.fmt(
            "Notice: Setting sampling rate to %s points per gesture.",
            payload["metadata"]["rate"],
        )
    effective_config["dtwWindow"] = payload["metadata"]["dtwWindow"]
    record_effective_config(paths or {}, metadata, effective_config)

    legacy_args = legacy_args_from_metadata(payload, output=opt.output, fmt=fmt)
    result = format_one_vs_many_result(payload, fmt, legacy_args=legacy_args)
    displayResults(result, opt.output, debug)
    raw_output = opt.raw_output or default_raw_output_path(opt.output)
    if raw_output:
        write_jsonl_rows(raw_output, payload["rawMetricOutputs"])
        debug.fmt("Raw metric outputs were saved in %s", raw_output)
    append_run_log(paths or {}, "Output format: %s" % fmt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
