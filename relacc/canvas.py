from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_mpl_config_dir = Path(tempfile.gettempdir()) / "relacc-matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relacc.geom.point import Point
from relacc.geom.pointset import PointSet
from relacc.gestures.gesture import Gesture
from relacc.gestures.ptaligntype import PtAlignType
from relacc.gestures.summarygesture import SummaryGesture
from relacc.pipeline._common import sampling_rate_for_sets
from relacc.utils.csv import CSVUtil


@dataclass(frozen=True)
class OverlayGroup:
    name: str
    files: Sequence[str]
    color: str
    width: float = 1.4
    alpha: float = 0.72
    limit: int = 18


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


def _read_points(csv_file: str):
    state = {}

    def _done(points):
        state["points"] = points

    CSVUtil.readGesture(csv_file, _done)
    return state.get("points") or []


def _draw_gesture(ax, points, canvas_size, line_width, line_color, alpha=1.0):
    tr_pts = PointSet.clone(points)
    if len(tr_pts) == 0:
        return

    bounds = PointSet.boundingBox(tr_pts)
    largest = max(bounds.width(), bounds.height(), 1)
    pad_scale = (canvas_size * 0.82) / largest
    tr_pts = PointSet.scaleTo(tr_pts, pad_scale)
    tr_pts = PointSet.translateBy(tr_pts, Point(-canvas_size / 2, -canvas_size / 2))
    line_color = _parse_color(line_color)

    stroke_x = [tr_pts[0].X]
    stroke_y = [tr_pts[0].Y]
    curr_stroke = tr_pts[0].StrokeID
    for point in tr_pts[1:]:
        if point.StrokeID != curr_stroke:
            if len(stroke_x) > 1:
                ax.plot(
                    stroke_x,
                    stroke_y,
                    linewidth=line_width,
                    color=line_color,
                    alpha=alpha,
                    solid_capstyle="round",
                )
            stroke_x = [point.X]
            stroke_y = [point.Y]
            curr_stroke = point.StrokeID
        else:
            stroke_x.append(point.X)
            stroke_y.append(point.Y)

    if len(stroke_x) > 1:
        ax.plot(
            stroke_x,
            stroke_y,
            linewidth=line_width,
            color=line_color,
            alpha=alpha,
            solid_capstyle="round",
        )


def render_overlay_svg(
    groups: Sequence[OverlayGroup],
    label: str = "gesture",
    rate: int | None = None,
    alignment_type: int = PtAlignType.CHRONOLOGICAL,
    summary_shape: str | None = None,
    popular_shape: bool = False,
    canvas_size: int = 640,
    include_reference_summary: bool = True,
    summary_source: str = "reference",
    summary_color: str = "#111514",
) -> str:
    """Render reference/candidate gesture groups as an SVG overlay."""
    all_points = []
    reference_points = []
    summary_points = []
    normalized_summary_source = (summary_source or "reference").strip().lower()

    for group_index, group in enumerate(groups):
        selected_files = list(group.files)[: group.limit]
        for file in selected_files:
            points = _read_points(file)
            if not points:
                continue
            all_points.append((group, points))
            if group_index == 0:
                reference_points.append(points)
            group_name = group.name.strip().lower()
            if normalized_summary_source == "all":
                summary_points.append(points)
            elif normalized_summary_source in group_name:
                summary_points.append(points)

    if not summary_points and normalized_summary_source == "reference":
        summary_points = reference_points

    rate = sampling_rate_for_sets([points for _, points in all_points], rate)

    fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#f8faf8")

    gestures = [Gesture(points, label, rate) for _, points in all_points]
    summary = SummaryGesture(gestures, alignment_type, None, False) if gestures else None
    for group, points in all_points:
        gesture = Gesture(points, label, rate)
        aligned = summary.alignGesture(gesture, alignment_type) if summary else points
        _draw_gesture(ax, aligned, canvas_size, group.width, group.color, group.alpha)

    if include_reference_summary and summary_shape and summary_points:
        reference_gestures = [Gesture(points, label, rate) for points in summary_points]
        reference_summary = SummaryGesture(
            reference_gestures,
            alignment_type,
            summary_shape,
            popular_shape,
        )
        _draw_gesture(
            ax,
            reference_summary.getPoints(),
            canvas_size,
            3.0,
            summary_color,
            0.92,
        )

    buf = io.StringIO()
    fig.savefig(buf, format="svg", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
