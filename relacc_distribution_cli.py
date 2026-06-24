#!/usr/bin/env python3
import argparse
import json

from relacc.gestures.ptaligntype import PtAlignType
from relacc.pipeline._common import (
    default_raw_output_path,
    output_format,
    write_jsonl_rows,
)
from relacc.pipeline.distribution import (
    GROUP_BY_FILENAME_LABEL,
    GROUP_BY_MODES,
    format_distribution_rows_csv,
    run_distribution_comparison,
)
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


def _int_cast(value):
    if value is None:
        return None
    return int(value)


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("reference", nargs="?")
    parser.add_argument("candidate", nargs="?", metavar="comparison")
    parser.add_argument("--group-by", default=GROUP_BY_FILENAME_LABEL, choices=GROUP_BY_MODES)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--comparison-name", default="comparison")
    parser.add_argument("--legacy-column-names", action="store_true")
    parser.add_argument("--legacy-json-fields", action="store_true")
    parser.add_argument("-r", "--rate")
    parser.add_argument("-a", "--alignment", type=PtAlignType.normalize)
    parser.add_argument("-m", "--summary")
    parser.add_argument("-p", "--popular", action="store_true")
    parser.add_argument("-f", "--format")
    parser.add_argument("-o", "--output")
    parser.add_argument("--raw-output")
    parser.add_argument("--round")
    parser.add_argument("--exact-dtw", action="store_true")
    parser.add_argument("--dtw-window")
    add_run_logging_arguments(parser)
    parser.add_argument("-h", "--help", action="store_true")
    return parser


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
        raise ValueError("Please provide both reference and comparison inputs.")

    paths = sidecar_paths(opt.output, opt.log_dir, stem="distribution")
    metadata = build_run_metadata(parser, opt, argv, "distribution")
    write_run_metadata(paths, metadata)
    return run_logged_experiment(paths, lambda: _run_experiment(opt, paths, metadata))


def _run_experiment(opt, paths=None, metadata=None):
    debug = Debug({"verbose": verbosity_from_opt(opt)})
    fmt = output_format(opt.output, opt.format)
    if fmt not in ["json", "csv"]:
        raise ValueError("Invalid output format (%s). Supported formats: json, csv." % fmt)

    rate = _int_cast(opt.rate)
    alignment = PtAlignType.normalize(
        PtAlignType.CHRONOLOGICAL if opt.alignment is None else opt.alignment
    )
    parsed_round = _int_cast(opt.round)
    round_precision = 3 if parsed_round is None else parsed_round
    dtw_window = _int_cast(opt.dtw_window)

    if dtw_window is not None and opt.exact_dtw:
        raise ValueError("--dtw-window cannot be combined with --exact-dtw.")

    payload = run_distribution_comparison(
        opt.reference,
        opt.candidate,
        rate=rate,
        alignment_type=alignment,
        summary_shape=opt.summary,
        popular_shape=opt.popular,
        round_precision=round_precision,
        group_by=opt.group_by,
        dtw_window=dtw_window,
        exact_dtw=bool(opt.exact_dtw),
        reference_group_name=opt.reference_name,
        comparison_group_name=opt.comparison_name,
        include_legacy_fields=bool(opt.legacy_json_fields),
    )
    record_effective_config(
        paths or {},
        metadata,
        {
            "format": fmt,
            "rate": payload["metadata"]["rate"],
            "alignment": alignment,
            "alignmentName": PtAlignType.name(alignment),
            "summary": opt.summary,
            "popular": bool(opt.popular),
            "roundPrecision": round_precision,
            "groupBy": opt.group_by,
            "dtwWindow": payload["metadata"]["dtwWindow"],
            "exactDtw": bool(opt.exact_dtw),
            "output": opt.output,
            "reference": opt.reference,
            "candidate": opt.candidate,
            "verbosity": verbosity_from_opt(opt),
        },
    )

    if fmt == "json":
        result = json.dumps(payload)
    else:
        result = format_distribution_rows_csv(
            payload["results"],
            legacy_column_names=bool(opt.legacy_column_names),
        )

    _display_result(result, opt.output, debug)
    raw_output = opt.raw_output or default_raw_output_path(opt.output)
    if raw_output:
        write_jsonl_rows(
            raw_output,
            [
                *payload["rawMetricOutputs"],
                *payload["rawDistributionOutputs"],
            ],
        )
        debug.fmt("Raw distribution outputs were saved in %s", raw_output)
    append_run_log(paths or {}, "Output format: %s" % fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
