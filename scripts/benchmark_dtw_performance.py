#!/usr/bin/env python3
"""Compare relacc's DTW helpers with FastDTW approximations.

The benchmark reports runtime, speedup, and score differences for the DTW-family
metrics used by relacc. By default it uses deterministic synthetic trajectories,
but two CSV gesture files can also be supplied.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fastdtw import fastdtw
except ImportError as exc:  # pragma: no cover - depends on the local environment.
    fastdtw = None
    FASTDTW_IMPORT_ERROR = exc
else:
    FASTDTW_IMPORT_ERROR = None

from relacc.dtw import (  # noqa: E402
    DEFAULT_WARPING_PENALTY,
    DTWResult,
    _derivative_sequence,
    _logistic_weight,
    ddtw,
    dtw,
    length_independent_dtw,
    weighted_derivative_dtw,
    weighted_dtw,
)
from relacc.geom.point import Point  # noqa: E402
from relacc.pipeline._common import read_points  # noqa: E402


PointSequence = Sequence[Point]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    current: Callable[[PointSequence, PointSequence, int | None], float]
    fast: Callable[[PointSequence, PointSequence, int], float]


@dataclass(frozen=True)
class BenchmarkRow:
    size: str
    metric: str
    current_value: float
    fast_value: float
    abs_error: float
    rel_error_pct: float
    current_ms: float
    fast_ms: float
    fastdtw_x: float
    faster: str


@dataclass(frozen=True)
class FastItem:
    """Numeric sequence item that FastDTW can average while downsampling."""

    values: tuple[float, ...]

    def __add__(self, other: "FastItem") -> "FastItem":
        return FastItem(tuple(a + b for a, b in zip(self.values, other.values)))

    def __truediv__(self, divisor: float) -> "FastItem":
        return FastItem(tuple(value / divisor for value in self.values))


def _fast_points(points: PointSequence) -> list[FastItem]:
    return [FastItem((point.X, point.Y)) for point in points]


def _fast_indexed_points(points: PointSequence) -> list[FastItem]:
    return [FastItem((float(index), point.X, point.Y)) for index, point in enumerate(points)]


def _fast_vectors(vectors: Sequence[tuple[float, float]]) -> list[FastItem]:
    return [FastItem(vector) for vector in vectors]


def _fast_indexed_vectors(vectors: Sequence[tuple[float, float]]) -> list[FastItem]:
    return [FastItem((float(index), vector[0], vector[1])) for index, vector in enumerate(vectors)]


def _point_distance(a: Point, b: Point) -> float:
    return math.hypot(a.X - b.X, a.Y - b.Y)


def _fast_point_distance(a: FastItem, b: FastItem) -> float:
    return math.hypot(a.values[0] - b.values[0], a.values[1] - b.values[1])


def _vector_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fast_vector_distance(a: FastItem, b: FastItem) -> float:
    return math.hypot(a.values[0] - b.values[0], a.values[1] - b.values[1])


def _indexed_point_distance(
    a: FastItem,
    b: FastItem,
    series_length: int,
    penalty_g: float,
) -> float:
    return (
        _logistic_weight(abs(a.values[0] - b.values[0]), series_length, penalty_g)
        * math.hypot(a.values[1] - b.values[1], a.values[2] - b.values[2])
    )


def _indexed_vector_distance(
    a: FastItem,
    b: FastItem,
    series_length: int,
    penalty_g: float,
) -> float:
    return (
        _logistic_weight(abs(a.values[0] - b.values[0]), series_length, penalty_g)
        * math.hypot(a.values[1] - b.values[1], a.values[2] - b.values[2])
    )


def _fast_dtw(points_a: PointSequence, points_b: PointSequence, radius: int) -> DTWResult:
    cost, path = fastdtw(
        _fast_points(points_a),
        _fast_points(points_b),
        radius=radius,
        dist=_fast_point_distance,
    )
    return DTWResult(cost=cost, path_length=len(path))


def _fast_ldtw(points_a: PointSequence, points_b: PointSequence, radius: int) -> float:
    result = _fast_dtw(points_a, points_b, radius)
    return result.cost / result.path_length


def _fast_ddtw(points_a: PointSequence, points_b: PointSequence, radius: int) -> DTWResult:
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    cost, path = fastdtw(
        _fast_vectors(derivatives_a),
        _fast_vectors(derivatives_b),
        radius=radius,
        dist=_fast_vector_distance,
    )
    return DTWResult(cost=cost, path_length=len(path))


def _fast_wdtw(
    points_a: PointSequence,
    points_b: PointSequence,
    radius: int,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
) -> DTWResult:
    series_length = max(len(points_a), len(points_b))
    indexed_a = _fast_indexed_points(points_a)
    indexed_b = _fast_indexed_points(points_b)
    cost, path = fastdtw(
        indexed_a,
        indexed_b,
        radius=radius,
        dist=lambda a, b: _indexed_point_distance(a, b, series_length, penalty_g),
    )
    return DTWResult(cost=cost, path_length=len(path))


def _fast_wddtw(
    points_a: PointSequence,
    points_b: PointSequence,
    radius: int,
    penalty_g: float = DEFAULT_WARPING_PENALTY,
) -> DTWResult:
    derivatives_a = _derivative_sequence(points_a)
    derivatives_b = _derivative_sequence(points_b)
    series_length = max(len(derivatives_a), len(derivatives_b))
    indexed_a = _fast_indexed_vectors(derivatives_a)
    indexed_b = _fast_indexed_vectors(derivatives_b)
    cost, path = fastdtw(
        indexed_a,
        indexed_b,
        radius=radius,
        dist=lambda a, b: _indexed_vector_distance(a, b, series_length, penalty_g),
    )
    return DTWResult(cost=cost, path_length=len(path))


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("dtwDistance", lambda a, b, w: dtw(a, b, window=w).cost, lambda a, b, r: _fast_dtw(a, b, r).cost),
    MetricSpec("ldtwDistance", length_independent_dtw, _fast_ldtw),
    MetricSpec("ddtwDistance", lambda a, b, w: ddtw(a, b, window=w).cost, lambda a, b, r: _fast_ddtw(a, b, r).cost),
    MetricSpec("wdtwDistance", lambda a, b, w: weighted_dtw(a, b, window=w).cost, lambda a, b, r: _fast_wdtw(a, b, r).cost),
    MetricSpec("wddtwDistance", lambda a, b, w: weighted_derivative_dtw(a, b, window=w).cost, lambda a, b, r: _fast_wddtw(a, b, r).cost),
)


def _synthetic_pair(size: int) -> tuple[list[Point], list[Point]]:
    first: list[Point] = []
    second: list[Point] = []
    for i in range(size):
        t = i / max(1, size - 1)
        x = 220.0 * t
        y = 55.0 * math.sin(2.0 * math.pi * t) + 18.0 * math.sin(7.0 * math.pi * t)
        first.append(Point(x, y, i * 10, 0))

        warped_t = min(1.0, max(0.0, t + 0.035 * math.sin(2.0 * math.pi * t)))
        warped_x = 220.0 * warped_t + 2.5 * math.sin(11.0 * math.pi * t)
        warped_y = (
            55.0 * math.sin(2.0 * math.pi * warped_t)
            + 18.0 * math.sin(7.0 * math.pi * warped_t)
            + 3.0 * math.cos(5.0 * math.pi * t)
        )
        second.append(Point(warped_x, warped_y, i * 10, 0))
    return first, second


def _time_call(fn: Callable[[], float], repeat: int) -> tuple[float, float]:
    fn()
    timings = []
    value = 0.0
    for _ in range(repeat):
        started = time.perf_counter()
        value = fn()
        timings.append((time.perf_counter() - started) * 1000.0)
    return value, statistics.median(timings)


def _benchmark_pair(
    size_label: str,
    points_a: PointSequence,
    points_b: PointSequence,
    metrics: Iterable[MetricSpec],
    repeat: int,
    radius: int,
    window: int | None,
) -> list[BenchmarkRow]:
    rows = []
    for metric in metrics:
        current_value, current_ms = _time_call(
            lambda metric=metric: metric.current(points_a, points_b, window),
            repeat,
        )
        fast_value, fast_ms = _time_call(
            lambda metric=metric: metric.fast(points_a, points_b, radius),
            repeat,
        )
        abs_error = abs(fast_value - current_value)
        rel_error_pct = 0.0 if current_value == 0 else (abs_error / abs(current_value)) * 100.0
        fastdtw_x = float("inf") if fast_ms == 0 else current_ms / fast_ms
        if current_ms <= fast_ms:
            faster = f"current {fast_ms / current_ms:.2f}x"
        else:
            faster = f"fastdtw {current_ms / fast_ms:.2f}x"
        rows.append(
            BenchmarkRow(
                size=size_label,
                metric=metric.name,
                current_value=current_value,
                fast_value=fast_value,
                abs_error=abs_error,
                rel_error_pct=rel_error_pct,
                current_ms=current_ms,
                fast_ms=fast_ms,
                fastdtw_x=fastdtw_x,
                faster=faster,
            )
        )
    return rows


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be a comma-separated list of positive integers")
    return sizes


def _print_table(rows: Sequence[BenchmarkRow]) -> None:
    headers = (
        "size",
        "metric",
        "current_ms",
        "fast_ms",
        "fastdtw_x",
        "faster",
        "current",
        "fast",
        "abs_error",
        "rel_error",
    )
    print(
        "{:<8} {:<17} {:>11} {:>10} {:>9} {:>14} {:>13} {:>13} {:>13} {:>10}".format(
            *headers
        )
    )
    print("-" * 127)
    for row in rows:
        fastdtw_x = "inf" if math.isinf(row.fastdtw_x) else f"{row.fastdtw_x:.2f}x"
        print(
            "{:<8} {:<17} {:>11.3f} {:>10.3f} {:>9} {:>14} {:>13.6g} {:>13.6g} {:>13.6g} {:>9.3f}%".format(
                row.size,
                row.metric,
                row.current_ms,
                row.fast_ms,
                fastdtw_x,
                row.faster,
                row.current_value,
                row.fast_value,
                row.abs_error,
                row.rel_error_pct,
            )
        )


def _summarize(rows: Sequence[BenchmarkRow]) -> None:
    finite_fastdtw_ratios = [row.fastdtw_x for row in rows if math.isfinite(row.fastdtw_x)]
    if not finite_fastdtw_ratios:
        return
    print()
    print("Summary")
    print("-------")
    print(f"Median FastDTW ratio: {statistics.median(finite_fastdtw_ratios):.2f}x")
    print(f"Best FastDTW ratio:   {max(finite_fastdtw_ratios):.2f}x")
    print(f"Median relative error: {statistics.median(row.rel_error_pct for row in rows):.3f}%")
    print(f"Max relative error:    {max(row.rel_error_pct for row in rows):.3f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark relacc DTW metrics against FastDTW approximations."
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=[32, 64, 128, 256],
        help="Comma-separated synthetic sequence lengths. Default: 32,64,128,256",
    )
    parser.add_argument("--csv-a", help="Optional first gesture CSV. Uses CSV input instead of synthetic data.")
    parser.add_argument("--csv-b", help="Optional second gesture CSV. Required when --csv-a is set.")
    parser.add_argument("--repeat", type=int, default=5, help="Timing repetitions per metric. Default: 5")
    parser.add_argument("--radius", type=int, default=8, help="FastDTW search radius. Default: 8")
    parser.add_argument("--window", type=int, default=None, help="Optional window for current relacc DTW.")
    parser.add_argument(
        "--metrics",
        default=",".join(metric.name for metric in METRICS),
        help="Comma-separated metrics to run. Default: all DTW-family metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
        category=RuntimeWarning,
    )
    if FASTDTW_IMPORT_ERROR is not None:
        raise SystemExit(
            "fastdtw is not installed. Install it with `python -m pip install fastdtw` "
            f"and rerun this script. Original error: {FASTDTW_IMPORT_ERROR}"
        )
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.radius < 1:
        raise SystemExit("--radius must be >= 1")
    if args.window is not None and args.window < 0:
        raise SystemExit("--window must be >= 0")
    if bool(args.csv_a) != bool(args.csv_b):
        raise SystemExit("--csv-a and --csv-b must be provided together")

    selected_names = {name.strip() for name in args.metrics.split(",") if name.strip()}
    metric_lookup = {metric.name: metric for metric in METRICS}
    unknown = sorted(selected_names - set(metric_lookup))
    if unknown:
        raise SystemExit("Unknown metric(s): %s" % ", ".join(unknown))
    selected_metrics = [metric for metric in METRICS if metric.name in selected_names]

    rows: list[BenchmarkRow] = []
    if args.csv_a:
        points_a = read_points(args.csv_a)
        points_b = read_points(args.csv_b)
        label = f"{len(points_a)}x{len(points_b)}"
        rows.extend(
            _benchmark_pair(label, points_a, points_b, selected_metrics, args.repeat, args.radius, args.window)
        )
    else:
        for size in args.sizes:
            points_a, points_b = _synthetic_pair(size)
            rows.extend(
                _benchmark_pair(str(size), points_a, points_b, selected_metrics, args.repeat, args.radius, args.window)
            )

    _print_table(rows)
    _summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
