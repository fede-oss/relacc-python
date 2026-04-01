#!/usr/bin/env python3
import argparse
import json
import os

from relacc.gestures.ptaligntype import PtAlignType
from relacc.metrics import get_metric_names
from relacc.pipeline.pairwise import (
    COMPARISON_MODES,
    DIRECT_MODE,
    format_pair_rows_csv,
    run_pairwise_comparison,
)
from relacc.utils.debug import Debug


def _int_cast(value):
    if value is None:
        return None
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("reference", nargs="?")
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("-l", "--label")
    parser.add_argument("-r", "--rate")
    parser.add_argument("-a", "--alignment")
    parser.add_argument("--mode", default=DIRECT_MODE, choices=COMPARISON_MODES)
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    parser.add_argument("-f", "--format")
    parser.add_argument("-o", "--output")
    parser.add_argument("--round")
    parser.add_argument("--exact-dtw", action="store_true")
    parser.add_argument("--dtw-window")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def _get_format(output, requested_format):
    if output:
        ext = os.path.splitext(output)[1][1:].lower()
        if ext:
            return ext
    return (requested_format or "json").lower()


def _display_result(text, output, debug):
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
        debug.fmt("Results were saved in %s", output)
    else:
        print(text)


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)

    if opt.help:
        parser.print_help()
        return 0

    if not opt.reference or not opt.candidate:
        parser.print_help()
        raise ValueError("Please provide both reference and candidate inputs.")

    debug = Debug({"verbose": bool(opt.verbose)})

    fmt = _get_format(opt.output, opt.format)
    if fmt not in ["json", "csv"]:
        raise ValueError("Invalid output format (%s). Supported formats: json, csv." % fmt)

    rate = _int_cast(opt.rate)
    parsed_alignment = _int_cast(opt.alignment)
    alignment = PtAlignType.CHRONOLOGICAL if parsed_alignment is None else parsed_alignment
    parsed_round = _int_cast(opt.round)
    round_precision = 3 if parsed_round is None else parsed_round
    dtw_window = _int_cast(opt.dtw_window)

    if dtw_window is not None and opt.exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")

    metric_names = get_metric_names()

    payload = run_pairwise_comparison(
        opt.reference,
        opt.candidate,
        label=opt.label,
        rate=rate,
        alignment_type=alignment,
        summary_shape=opt.summary,
        popular_shape=opt.popular,
        strict=opt.strict,
        round_precision=round_precision,
        comparison_mode=opt.mode,
        metric_names=metric_names,
        dtw_window=dtw_window,
        exact_dtw=bool(opt.exact_dtw),
    )

    if fmt == "json":
        result = json.dumps(payload)
    else:
        result = format_pair_rows_csv(payload["pairs"], metric_names=metric_names)

    _display_result(result, opt.output, debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
