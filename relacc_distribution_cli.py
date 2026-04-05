#!/usr/bin/env python3
import argparse
import json
import os

from relacc.gestures.ptaligntype import PtAlignType
from relacc.pipeline.distribution import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    format_distribution_rows_csv,
    run_distribution_comparison,
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
    parser.add_argument("--group-by", default=GROUP_BY_FILENAME_LABEL, choices=GROUP_BY_MODES)
    parser.add_argument("-r", "--rate")
    parser.add_argument("-a", "--alignment")
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("-f", "--format")
    parser.add_argument("-o", "--output")
    parser.add_argument("--round")
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

    payload = run_distribution_comparison(
        opt.reference,
        opt.candidate,
        rate=rate,
        alignment_type=alignment,
        summary_shape=opt.summary,
        popular_shape=opt.popular,
        round_precision=round_precision,
        group_by=opt.group_by,
    )

    if fmt == "json":
        result = json.dumps(payload)
    else:
        result = format_distribution_rows_csv(payload["results"])

    _display_result(result, opt.output, debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
