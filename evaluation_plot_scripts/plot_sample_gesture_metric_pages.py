#!/usr/bin/env python3
"""Plot generated gesture samples with human summaries and metric comparisons.

The script consumes RELACC evaluation output directories. For each generator
run and dataset, it draws one panel per class: generated samples are shown in
black using the relacc-canvas drawing primitive, and the human summary gesture
is shown in red. A compact metric table below each canvas compares the selected
generated samples against the human-to-summary baseline for the same class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.utils.csv import CSVUtil
from relacc_canvas_cli import _drawGesture


DEFAULT_METRICS = (
    "shapeError",
    "shapeVariability",
    "lengthError",
    "sizeError",
    "bendingError",
    "bendingVariability",
    "timeError",
    "timeVariability",
    "velocityError",
    "velocityVariability",
    "cornerSlowdown",
    "twoThirdsPowerLawR2",
    "highFrequencyRatio",
    "curvature",
    "strokeError",
    "strokeOrderError",
    "strokeLengthStd",
    "meanStrokeDuration",
    "dtwDistance",
    "ldtwDistance",
    "ddtwDistance",
    "wdtwDistance",
    "wddtwDistance",
)
DEFAULT_FORMATS = ("png",)


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    generated_median: float | None
    human_median: float | None
    ratio: float | None


@dataclass(frozen=True)
class ClassPanel:
    source: str
    dataset: str
    variant: str
    class_key: str
    summary_shape: str
    generated_gestures: tuple[Gesture, ...]
    human_gestures: tuple[Gesture, ...]
    metric_summaries: tuple[MetricSummary, ...]
    rate: int
    alignment: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="report-output-eval-detailed-s24-20260608",
        help="Evaluation output root containing generator/dataset manifest.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation-plots-detailed-s24/sample-gesture-metric-pages-one-page",
        help="Directory where sample gesture pages will be written.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Limit to a source/generator, for example DHG. Can be repeated.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Limit to a dataset, for example 1dollar. Can be repeated.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Limit to a variant, for example root, recoTO, or syntTO. Can be repeated.",
    )
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        help="Limit to a class key. Can be repeated.",
    )
    parser.add_argument("--samples", type=int, default=16, help="Generated samples to draw per class.")
    parser.add_argument(
        "--human-samples",
        type=int,
        default=64,
        help="Human samples used to compute and draw the summary gesture.",
    )
    parser.add_argument(
        "--classes-per-page",
        type=int,
        default=None,
        help="Deprecated; classes now default to one page per generator/dataset run.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        help="Override the auto-selected number of class columns on each page.",
    )
    parser.add_argument("--max-runs", type=int, help="Stop after this many generator/dataset runs.")
    parser.add_argument("--max-classes", type=int, help="Limit classes per run.")
    parser.add_argument(
        "--sample-selection",
        choices=("stable", "random"),
        default="stable",
        help="How generated samples are selected from pairwise.csv.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for random sample selection.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metrics shown below each gesture canvas.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Comma-separated output formats, for example png,pdf.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=220,
        help="Canvas size passed to the relacc-canvas drawing primitive.",
    )
    parser.add_argument("--rate", type=int, default=24)
    parser.add_argument("--alignment", type=int, default=PtAlignType.CHRONOLOGICAL)
    parser.add_argument(
        "--summary-shape",
        default=None,
        choices=("centroid", "medoid", "kcentroid", "kmedoid"),
        help="Override the summary shape from the evaluation manifest.",
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    text = str(value if value not in (None, "") else "none")
    chars = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text]
    return "".join(chars) or "_"


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def compact_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    if magnitude >= 0.01:
        return f"{value:.4f}"
    return f"{value:.2g}"


def compact_metric_name(metric: str) -> str:
    aliases = {
        "shapeError": "shape",
        "shapeVariability": "shapeVar",
        "lengthError": "length",
        "sizeError": "size",
        "bendingError": "bend",
        "bendingVariability": "bendVar",
        "timeError": "time",
        "timeVariability": "timeVar",
        "velocityError": "vel",
        "velocityVariability": "velVar",
        "cornerSlowdown": "corner",
        "twoThirdsPowerLawR2": "2/3R2",
        "highFrequencyRatio": "hiFreq",
        "curvature": "curve",
        "strokeError": "strokes",
        "strokeOrderError": "order",
        "strokeLengthStd": "strokeLen",
        "meanStrokeDuration": "strokeDur",
        "dtwDistance": "dtw",
        "ldtwDistance": "ldtw",
        "ddtwDistance": "ddtw",
        "wdtwDistance": "wdtw",
        "wddtwDistance": "wddtw",
    }
    return aliases.get(metric, metric)


def median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(statistics.median(finite))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_points(path: Path):
    points = []

    def done(parsed_points):
        points.extend(parsed_points)

    CSVUtil.readGesture(str(path), done)
    return points


def make_gesture(path: Path, label: str, rate: int) -> Gesture | None:
    try:
        return Gesture(read_points(path), label, rate)
    except Exception as exc:
        print(f"warning: skipped gesture {path}: {exc}")
        return None


def make_gestures(paths: Sequence[Path], label: str, rate: int) -> tuple[Gesture, ...]:
    gestures = [make_gesture(path, label, rate) for path in paths]
    return tuple(gesture for gesture in gestures if gesture is not None)


def output_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(fmt.strip().lower().lstrip(".") for fmt in raw.split(",") if fmt.strip())
    return formats or DEFAULT_FORMATS


def iter_run_manifests(input_dir: Path) -> Iterator[Path]:
    for manifest in sorted(input_dir.glob("**/manifest.json")):
        if manifest.parent == input_dir or manifest.parent.name == "combined":
            continue
        yield manifest


def manifest_matches(
    manifest: dict,
    source_filter: set[str],
    dataset_filter: set[str],
    variant_filter: set[str],
) -> bool:
    source = str(manifest.get("source") or "")
    dataset = str(manifest.get("dataset") or "")
    variant = str(manifest.get("variant") or "root")
    return (
        (not source_filter or source in source_filter)
        and (not dataset_filter or dataset in dataset_filter)
        and (not variant_filter or variant in variant_filter)
    )


def class_dir_for(run_dir: Path, klass: dict) -> Path:
    output_dir = klass.get("outputDir")
    if output_dir:
        candidate = Path(str(output_dir))
        if candidate.exists():
            return candidate
    return run_dir / "classes" / str(klass.get("classKey"))


def row_file_path(row: dict[str, str], *fields: str) -> Path | None:
    for field in fields:
        value = row.get(field)
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def select_rows(
    rows: Sequence[dict[str, str]],
    limit: int,
    mode: str,
    seed_text: str,
    seed: int,
) -> list[dict[str, str]]:
    usable = [row for row in rows if row_file_path(row, "candidateFile", "sampleFile")]
    if mode == "random" and len(usable) > limit:
        rng = random.Random(f"{seed}:{seed_text}")
        usable = sorted(rng.sample(usable, limit), key=lambda row: row.get("pairKey") or row.get("sampleKey") or "")
    return usable[:limit]


def build_metric_summaries(
    generated_rows: Sequence[dict[str, str]],
    human_rows: Sequence[dict[str, str]],
    metrics: Sequence[str],
) -> tuple[MetricSummary, ...]:
    summaries = []
    for metric in metrics:
        generated_median = median(to_float(row.get(metric)) for row in generated_rows)
        human_median = median(to_float(row.get(metric)) for row in human_rows)
        ratio = None
        if generated_median is not None and human_median not in (None, 0):
            ratio = generated_median / human_median
        summaries.append(MetricSummary(metric, generated_median, human_median, ratio))
    return tuple(summaries)


def build_panels_for_run(
    run_dir: Path,
    manifest: dict,
    args: argparse.Namespace,
) -> list[ClassPanel]:
    class_filter = set(args.classes or [])
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    classes = list(manifest.get("classes") or [])
    if args.max_classes:
        classes = classes[: args.max_classes]

    panels = []
    for klass in classes:
        class_key = str(klass.get("classKey") or "")
        if class_filter and class_key not in class_filter:
            continue

        class_dir = class_dir_for(run_dir, klass)
        generated_rows = read_csv_rows(class_dir / "pairwise.csv")
        human_rows = read_csv_rows(class_dir / "baseline.csv")
        if not generated_rows or not human_rows:
            continue

        rate = int(klass.get("rate") or klass.get("requestedRate") or args.rate)
        alignment = int(klass.get("alignment") or args.alignment)
        summary_shape = str(args.summary_shape or klass.get("summary") or "medoid")
        source = str(klass.get("source") or manifest.get("source") or "")
        dataset = str(klass.get("dataset") or manifest.get("dataset") or "")
        variant = str(klass.get("variant") or manifest.get("variant") or "root")

        selected_rows = select_rows(
            generated_rows,
            args.samples,
            args.sample_selection,
            f"{source}/{dataset}/{variant}/{class_key}",
            args.seed,
        )
        generated_paths = [
            path
            for row in selected_rows
            if (path := row_file_path(row, "candidateFile", "sampleFile")) is not None
        ]
        human_paths = [
            path
            for row in human_rows[: args.human_samples]
            if (path := row_file_path(row, "sampleFile", "candidateFile")) is not None
        ]
        generated_gestures = make_gestures(generated_paths, class_key, rate)
        human_gestures = make_gestures(human_paths, class_key, rate)
        if not generated_gestures or not human_gestures:
            continue

        metric_summaries = build_metric_summaries(selected_rows, human_rows, metrics)
        panels.append(
            ClassPanel(
                source=source,
                dataset=dataset,
                variant=variant,
                class_key=class_key,
                summary_shape=summary_shape,
                generated_gestures=generated_gestures,
                human_gestures=human_gestures,
                metric_summaries=metric_summaries,
                rate=rate,
                alignment=alignment,
            )
        )

    return panels


def align_points_for_canvas(gesture: Gesture, aligner, alignment: int):
    if hasattr(aligner, "alignGesture"):
        return aligner.alignGesture(gesture, alignment)
    return gesture.points


def draw_gesture_canvas(ax, panel: ClassPanel, canvas_size: int) -> None:
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    aligner = panel.generated_gestures[0]
    if len(panel.generated_gestures) > 1:
        try:
            aligner = SummaryGesture(list(panel.generated_gestures), panel.alignment, None, False)
        except Exception:
            aligner = panel.generated_gestures[0]

    for gesture in panel.generated_gestures:
        points = align_points_for_canvas(gesture, aligner, panel.alignment)
        _drawGesture(ax, points, canvas_size, 1.15, "rgba(0,0,0,.24)")

    try:
        summary = SummaryGesture(list(panel.human_gestures), panel.alignment, panel.summary_shape, False)
        _drawGesture(ax, summary.getPoints(), canvas_size, 4.8, "#d62728")
    except Exception as exc:
        print(f"warning: could not draw human summary for {panel.class_key}: {exc}")


def draw_metric_text(ax, panel: ClassPanel) -> None:
    ax.axis("off")
    lines = []
    for summary in panel.metric_summaries:
        ratio = "n/a" if summary.ratio is None else f"{summary.ratio:.2f}x"
        lines.append(
            f"{compact_metric_name(summary.metric):<9} "
            f"{compact_number(summary.generated_median):>7}/"
            f"{compact_number(summary.human_median):<7} {ratio:>6}"
        )
    ax.text(
        0,
        1,
        "metric        gen/human ratio\n" + "\n".join(lines),
        va="top",
        ha="left",
        fontsize=4.6,
        family="monospace",
        color="#202020",
        linespacing=0.92,
    )


def auto_columns(class_count: int) -> int:
    if class_count <= 6:
        return 2
    if class_count <= 20:
        return 4
    if class_count <= 60:
        return 6
    if class_count <= 100:
        return 8
    return 10


def draw_page(
    panels: Sequence[ClassPanel],
    output: Path,
    title: str,
    fmt: str,
    dpi: int,
    canvas_size: int,
    columns: int | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = max(1, columns or auto_columns(len(panels)))
    rows = max(1, math.ceil(len(panels) / cols))
    panel_width = 2.2 if cols >= 8 else 2.55
    panel_height = 2.95 if cols >= 8 else 3.15
    fig = plt.figure(figsize=(panel_width * cols, panel_height * rows + 0.45), dpi=dpi)
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.34, wspace=0.16)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.997)
    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.018, top=0.965)

    for idx, panel in enumerate(panels):
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[idx],
            height_ratios=[1.0, 1.22],
            hspace=0.04,
        )
        ax_canvas = fig.add_subplot(inner[0])
        ax_metrics = fig.add_subplot(inner[1])
        draw_gesture_canvas(ax_canvas, panel, canvas_size)

        ax_canvas.set_title(panel.class_key, fontsize=6.4, pad=3)
        draw_metric_text(ax_metrics, panel)

    for idx in range(len(panels), rows * cols):
        ax = fig.add_subplot(outer[idx])
        ax.axis("off")

    fig.savefig(output, format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chunked(items: Sequence[ClassPanel], size: int) -> Iterator[tuple[int, Sequence[ClassPanel]]]:
    for index in range(0, len(items), size):
        yield index // size + 1, items[index : index + size]


def write_manifest(output_dir: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.csv"
    fieldnames = [
        "path",
        "source",
        "dataset",
        "variant",
        "page",
        "classCount",
        "classes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if args.human_samples < 1:
        raise ValueError("--human-samples must be >= 1")
    if args.columns is not None and args.columns < 1:
        raise ValueError("--columns must be >= 1")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_filter = set(args.source or [])
    dataset_filter = set(args.dataset or [])
    variant_filter = set(args.variant or [])
    formats = output_formats(args.format)

    written_rows: list[dict[str, object]] = []
    processed_runs = 0
    for manifest_path in iter_run_manifests(input_dir):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest_matches(manifest, source_filter, dataset_filter, variant_filter):
            continue

        panels = build_panels_for_run(run_dir, manifest, args)
        if not panels:
            print(f"warning: no plottable classes for {run_dir}")
            continue

        processed_runs += 1
        source = str(manifest.get("source") or panels[0].source or "")
        dataset = str(manifest.get("dataset") or panels[0].dataset or "")
        variant = str(manifest.get("variant") or panels[0].variant or "root")
        run_label = "/".join(part for part in (source, dataset, variant) if part)
        run_slug = "__".join(safe_name(part) for part in (source, dataset, variant) if part)

        title = source
        for fmt in formats:
            output = output_dir / f"{run_slug}__sample_metrics.{fmt}"
            draw_page(panels, output, title, fmt, args.dpi, args.canvas_size, args.columns)
            written_rows.append(
                {
                    "path": str(output),
                    "source": source,
                    "dataset": dataset,
                    "variant": variant,
                    "page": 1,
                    "classCount": len(panels),
                    "classes": ";".join(panel.class_key for panel in panels),
                }
            )

        if args.max_runs is not None and processed_runs >= args.max_runs:
            break

    write_manifest(output_dir, written_rows)
    print(f"Wrote {len(written_rows)} plot files to {output_dir}")
    for row in written_rows[:12]:
        print(row["path"])
    if len(written_rows) > 12:
        print(f"... {len(written_rows) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
