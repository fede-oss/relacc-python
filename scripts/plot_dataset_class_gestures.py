#!/usr/bin/env python3
"""Plot gesture samples, human summaries, and metric scores by dataset/class."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.utils.csv import CSVUtil
from relacc_canvas_cli import _drawGesture


DEFAULT_METRICS = [
    "shapeError",
    "shapeVariability",
    "lengthError",
    "sizeError",
    "bendingError",
    "timeError",
    "velocityError",
    "strokeError",
    "dtwDistance",
    "wdtwDistance",
]


def read_points(path):
    result = []

    def done(points):
        result.extend(points)

    CSVUtil.readGesture(str(path), done)
    return result


def class_key_from_filename(dataset, path):
    name = Path(path).stem

    if dataset in {"1dollar", "ndollar_finger", "ndollar_stylus", "char74k"}:
        parts = name.split("-")
        if len(parts) >= 3:
            return parts[1]

    if dataset == "MobileTouchDB":
        match = re.search(r"_s\d+_([^_]+)$", name)
        if match:
            return match.group(1)

    if dataset in {"MCYT", "Visual"}:
        match = re.match(r"(\d+)g\d+$", name)
        if match:
            return match.group(1)

    if dataset in {"ebiosig_finger", "ebiosig_stylus"}:
        match = re.match(r"W\d+_(\d+)g\d+$", name)
        if match:
            return match.group(1)

    if dataset == "BiosecurID":
        match = re.match(r"(\d+)v\d+$", name)
        if match:
            return match.group(1)

    if dataset == "projected3Dsignatures":
        match = re.match(r"(\d+)_(\d+)$", name)
        if match:
            return match.group(1).zfill(2)

    if "-" in name:
        return name.split("-", 1)[0]

    if "_" in name:
        return name.split("_", 1)[0]

    return name


def index_csvs_by_class(dataset, folder):
    by_class = defaultdict(list)
    if not folder or not Path(folder).exists():
        return by_class

    for path in sorted(Path(folder).glob("*.csv")):
        by_class[class_key_from_filename(dataset, path)].append(path)
    return by_class


def read_stats(path):
    rows = {}
    if not path.exists():
        return rows

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            metric = row.get("metric")
            if metric:
                rows[metric] = row
    return rows


def read_class_stats(run_dir, class_key, filename):
    class_file = run_dir / "classes" / class_key / filename
    if class_file.exists():
        return read_stats(class_file)

    root_file = run_dir / filename
    if not root_file.exists():
        return {}

    rows = {}
    with root_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("classKey") == class_key and row.get("metric"):
                rows[row["metric"]] = row
    return rows


def parse_float(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def metric_lines(candidate_stats, human_stats, metrics):
    lines = []
    deltas = []

    for metric in metrics:
        cand = parse_float(candidate_stats.get(metric, {}).get("mean"))
        human = parse_float(human_stats.get(metric, {}).get("mean"))
        if cand is None and human is None:
            continue
        if cand is not None and human not in (None, 0):
            ratio = cand / human
            deltas.append(abs(math.log(max(ratio, 1e-12))))
            lines.append(f"{metric}: {cand:.3g} vs H {human:.3g} ({ratio:.2f}x)")
        elif cand is not None:
            lines.append(f"{metric}: {cand:.3g}")
        else:
            lines.append(f"{metric}: H {human:.3g}")

    aggregate = sum(deltas) / len(deltas) if deltas else None
    return lines, aggregate


def make_gestures(files, label, limit, rate):
    gestures = []
    for path in files[:limit]:
        try:
            gestures.append(Gesture(read_points(path), label, rate))
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}")
    return gestures


def draw_canvas(ax, sample_gestures, human_gestures, rate, alignment, summary_shape, summary_width=5.2):
    ax.set_xlim(0, 500)
    ax.set_ylim(500, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    if sample_gestures:
        aligner = sample_gestures[0]
        if len(sample_gestures) > 1:
            try:
                aligner = SummaryGesture(sample_gestures, alignment, None, False)
            except Exception:
                aligner = sample_gestures[0]

        for gesture in sample_gestures:
            points = aligner.alignGesture(gesture, alignment) if hasattr(aligner, "alignGesture") else gesture.points
            _drawGesture(ax, points, 500, 1.2, "rgba(0,0,0,.22)")

    if human_gestures:
        try:
            summary = SummaryGesture(human_gestures, alignment, summary_shape, False)
            _drawGesture(ax, summary.getPoints(), 500, summary_width, "#d62728")
        except Exception as exc:
            print(f"warning: could not draw human summary: {exc}")


def add_metrics_text(ax, lines, max_lines=10):
    ax.axis("off")
    if not lines:
        ax.text(0, 1, "No metric rows found", va="top", fontsize=7, color="#777")
        return

    text = "\n".join(lines[:max_lines])
    ax.text(
        0,
        1,
        text,
        va="top",
        ha="left",
        fontsize=6.8,
        family="monospace",
        color="#252525",
        linespacing=1.2,
    )


def draw_detail_page(page_items, output, title, metrics):
    rows = max(1, math.ceil(len(page_items) / 2))
    fig = plt.figure(figsize=(15, 3.7 * rows + 0.7), dpi=160)
    outer = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.34, wspace=0.22)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)

    for idx, item in enumerate(page_items):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[idx], height_ratios=[3.2, 1.15], hspace=0.03
        )
        ax_plot = fig.add_subplot(inner[0])
        ax_text = fig.add_subplot(inner[1])
        draw_canvas(
            ax_plot,
            item["sample_gestures"],
            item["human_gestures"],
            item["rate"],
            item["alignment"],
            item["summary_shape"],
            summary_width=5.6,
        )
        ax_plot.set_title(
            f"{item['class_key']}  samples={item['sample_count']} human={item['human_count']}",
            fontsize=9,
            pad=2,
        )
        add_metrics_text(ax_text, item["metric_lines"], len(metrics))

    for idx in range(len(page_items), rows * 2):
        ax = fig.add_subplot(outer[idx])
        ax.axis("off")

    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_summary_page(page_items, output, title):
    rows = len(page_items)
    fig = plt.figure(figsize=(15, max(5, rows * 1.4)), dpi=160)
    grid = gridspec.GridSpec(rows, 3, figure=fig, width_ratios=[1.15, 2.9, 2.2], hspace=0.2, wspace=0.12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    for row_idx, item in enumerate(page_items):
        ax_label = fig.add_subplot(grid[row_idx, 0])
        ax_plot = fig.add_subplot(grid[row_idx, 1])
        ax_metrics = fig.add_subplot(grid[row_idx, 2])

        ax_label.axis("off")
        score = item["aggregate"]
        score_text = "score n/a" if score is None else f"delta {score:.2f}"
        ax_label.text(0, 0.68, item["class_key"], fontsize=9, fontweight="bold")
        ax_label.text(0, 0.38, f"{item['sample_count']} samples", fontsize=7, color="#555")
        ax_label.text(0, 0.16, score_text, fontsize=7, color="#555")

        draw_canvas(
            ax_plot,
            item["sample_gestures"],
            item["human_gestures"],
            item["rate"],
            item["alignment"],
            item["summary_shape"],
            summary_width=4.2,
        )
        add_metrics_text(ax_metrics, item["metric_lines"], 5)

    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def iter_run_dirs(metrics_root, sources, datasets):
    for manifest in sorted(Path(metrics_root).glob("**/manifest.json")):
        run_dir = manifest.parent
        rel = run_dir.relative_to(metrics_root)
        parts = rel.parts
        if len(parts) < 2:
            continue
        source, dataset = parts[0], parts[1]
        if sources and source not in sources:
            continue
        if datasets and dataset not in datasets:
            continue
        yield run_dir, manifest


def build_items(run_dir, manifest_path, args, metrics):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = manifest.get("classes", [])
    if args.max_classes:
        classes = classes[: args.max_classes]

    candidate_input = manifest.get("candidateInput")
    reference_input = manifest.get("referenceInput")
    dataset = manifest.get("dataset") or (classes[0].get("dataset") if classes else run_dir.name)

    candidate_by_class = index_csvs_by_class(dataset, candidate_input)
    reference_by_class = index_csvs_by_class(dataset, reference_input)

    items = []
    for klass in classes:
        class_key = str(klass["classKey"])
        candidate_files = candidate_by_class.get(class_key, [])
        reference_files = reference_by_class.get(class_key, [])
        if not candidate_files and not reference_files:
            continue

        rate = int(klass.get("rate") or klass.get("requestedRate") or args.rate)
        alignment = int(klass.get("alignment") or args.alignment)
        candidate_stats = read_class_stats(run_dir, class_key, "stats.csv")
        human_stats = read_class_stats(run_dir, class_key, "baseline_stats.csv")
        lines, aggregate = metric_lines(candidate_stats, human_stats, metrics)

        items.append(
            {
                "class_key": class_key,
                "sample_count": len(candidate_files),
                "human_count": len(reference_files),
                "sample_gestures": make_gestures(candidate_files, class_key, args.samples, rate),
                "human_gestures": make_gestures(reference_files, class_key, args.human_samples, rate),
                "metric_lines": lines,
                "aggregate": aggregate,
                "rate": rate,
                "alignment": alignment,
                "summary_shape": args.summary_shape,
            }
        )

    if args.sort_summary_by_delta:
        items.sort(key=lambda item: item["aggregate"] if item["aggregate"] is not None else -1, reverse=True)

    return manifest, items


def chunked(items, size):
    for index in range(0, len(items), size):
        yield index // size + 1, items[index : index + size]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", default="report-output-full-metrics")
    parser.add_argument("--output-dir", default="report-output-full-metrics/gesture-plots")
    parser.add_argument("--source", action="append", help="Limit to a source, e.g. SDT. Can be repeated.")
    parser.add_argument("--dataset", action="append", help="Limit to a dataset, e.g. raton. Can be repeated.")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--human-samples", type=int, default=64)
    parser.add_argument("--max-classes", type=int)
    parser.add_argument("--detail-classes-per-page", type=int, default=6)
    parser.add_argument("--summary-classes-per-page", type=int, default=14)
    parser.add_argument("--summary-shape", default="centroid", choices=["centroid", "medoid", "kcentroid", "kmedoid"])
    parser.add_argument("--alignment", type=int, default=PtAlignType.CHRONOLOGICAL)
    parser.add_argument("--rate", type=int, default=24)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--detail-only", action="store_true")
    parser.add_argument("--sort-summary-by-delta", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = args.metrics or DEFAULT_METRICS
    output_root = Path(args.output_dir)
    detail_root = output_root / "detail"
    summary_root = output_root / "summary"
    detail_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    written = []
    for run_dir, manifest_path in iter_run_dirs(Path(args.metrics_root), set(args.source or []), set(args.dataset or [])):
        manifest, items = build_items(run_dir, manifest_path, args, metrics)
        if not items:
            print(f"warning: no plottable classes for {run_dir}")
            continue

        run_id = manifest.get("runId") or "/".join(run_dir.parts[-2:])
        slug = "__".join(run_dir.relative_to(args.metrics_root).parts)

        if not args.summary_only:
            for page_no, page_items in chunked(items, args.detail_classes_per_page):
                output = detail_root / f"{slug}__detail_{page_no:03d}.png"
                draw_detail_page(page_items, output, f"{run_id} detail page {page_no}", metrics)
                written.append(output)

        if not args.detail_only:
            for page_no, page_items in chunked(items, args.summary_classes_per_page):
                output = summary_root / f"{slug}__summary_{page_no:03d}.png"
                draw_summary_page(page_items, output, f"{run_id} compact summary page {page_no}")
                written.append(output)

    print(f"wrote {len(written)} png files to {output_root}")
    for path in written[:12]:
        print(path)
    if len(written) > 12:
        print(f"... {len(written) - 12} more")


if __name__ == "__main__":
    main()
