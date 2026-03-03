# Relative Accuracy Measures for Stroke Gestures (Python)

This repository contains a Python port of the Gesture Relative Accuracy Toolkit (GREAT).

## Requirements

- Python 3.11+
- `matplotlib` (for canvas rendering CLI)

## Installation

```bash
python3 -m pip install -e .
```

If your system Python is marked as "externally managed" (common on newer Debian/Ubuntu-based setups), use:

```bash
python3 -m pip install --break-system-packages -e .
```

## Core Concept: One-vs-Many Evaluation

Both `relacc` and `relacc-canvas` work the same way conceptually:

1. You pass **multiple CSV files** for the same gesture (e.g. 10 trials of an arrow gesture).
2. A single **summary gesture** (the reference) is computed from the whole collection.
3. Every individual sample is measured against that reference, producing 12 metrics per sample.

The metrics cover:

| Metric | What it measures |
|---|---|
| `shapeError` | Mean spatial distance from the reference trajectory |
| `shapeVariability` | Consistency of that deviation along the path |
| `lengthError` | Difference in total path length |
| `sizeError` | Difference in bounding-box area |
| `bendingError` | Mean difference in curvature (turning angles) |
| `bendingVariability` | Consistency of curvature deviation |
| `timeError` | Difference in total production time |
| `timeVariability` | Consistency of per-point timing |
| `velocityError` | Mean speed difference along the path |
| `velocityVariability` | Consistency of speed deviation |
| `strokeError` | Difference in number of strokes |
| `strokeOrderError` | Degree of stroke-order violation |

Lower values always mean closer to the reference. Passing a single CSV produces all zeros (reference = the file itself).

---

## The Summary Gesture (`-m`)

The summary gesture is the computed reference all samples are measured against. It is built by first resampling every input gesture to the same number of points (`-r`), aligning them, and then aggregating point-by-point across the collection. The `-m` flag controls the aggregation strategy:

| `-m` value | Description | Real gesture? |
|---|---|---|
| *(omitted)* | First input file's resampled points — an arbitrary pick | Yes |
| `centroid` | Coordinate-wise **mean** at each sample index — a smooth synthetic average | No |
| `medoid` | Coordinate-wise **median** at each sample index — more robust to outliers than centroid | No |
| `kcentroid` | The actual gesture in the collection **nearest to the centroid** (1-NN search) | Yes |
| `kmedoid` | The actual gesture in the collection **nearest to the medoid** (1-NN search) | Yes |

`centroid` and `medoid` produce a synthetic shape that may not correspond to any real recording. `kcentroid` and `kmedoid` always return a real gesture from the input — useful when you want a natural, plausible reference with realistic timing and stroke structure.

Use `-p` (`--popular`) to filter out gestures whose stroke count differs from the most common one before building the summary — handy when a subset of samples accidentally has an extra stroke.

---

## Usage

### `relacc` — metric report

Computes metrics for each gesture relative to the summary and prints the results.

**Per-gesture output** (default): one row per input file.

```bash
relacc -s -r 32 -a 1 -m centroid -f json /path/to/*gesture_name*.csv

# All arrow-fast trials for subject 01, one row per trial
relacc datasets/becaptcha/1dollar/s01-arrow-fast-*.csv
```

**Aggregate stats** (`-s`): one row per metric, summarising the whole collection (mean, median, sd, min, max).

```bash
# Aggregate stats, centroid reference, 32 sample points, JSON output
relacc -s -m centroid -r 32 -f json \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv

# Save to file (format inferred from extension)
relacc -s -m centroid -r 32 \
  -o /tmp/arrow-fast-stats.csv \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv

# Use a real gesture as reference (nearest to the medoid)
relacc -s -m kmedoid -r 32 -f csv \
  datasets/becaptcha/1dollar/*-arrow-fast-*.csv

# Cloud-match alignment (stroke-order agnostic), XML output
relacc -s -m centroid -r 32 -a 1 -f xml \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `-m, --summary` | *(first file)* | Summary strategy: `centroid`, `medoid`, `kcentroid`, `kmedoid` |
| `-r, --rate` | auto | Resampling rate (points per gesture). Auto-estimated from stroke count if omitted |
| `-a, --alignment` | `0` | Point alignment: `0` = chronological, `1` = cloud-match (unordered) |
| `-p, --popular` | off | Filter to most common stroke count before building summary |
| `-s, --stats` | off | Output aggregate stats instead of per-file rows |
| `-f, --format` | `json` | Output format: `json`, `csv`, `xml` |
| `-o, --output` | *(stdout)* | Write output to file (format inferred from extension) |
| `-l, --label` | *(from filename)* | Gesture label — inferred from the second `-`-separated segment of the first filename if omitted |
| `-v, --verbose` | off | Debug logging |

---

### `relacc-canvas` — visual overlay

Same reference-building logic as `relacc`, but renders all gestures and the summary as an image instead of computing numbers. Every input gesture is drawn in one colour; the summary is drawn on top in another **only when `-m` is provided** — without it, only the raw gesture overlays are rendered.

```bash
# Render all arrow-fast trials, centroid in red, samples in semi-transparent blue
relacc-canvas -m centroid \
  -c "rgba(0,0,255,0.3)" -C red \
  -o /tmp/arrow-fast.png \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv

# Larger canvas, thicker summary line, SVG output
relacc-canvas -m kmedoid -r 32 \
  -s 800 -T 3 -t 1 \
  -o /tmp/arrow-fast.svg \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv

# No output file — base64-encoded image printed to stdout
relacc-canvas -m centroid \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv
```

**Flags** (inherits all of `relacc`'s flags except `-s`/`-f`, plus):

| Flag | Default | Description |
|---|---|---|
| `-s, --size` | `500` | Canvas size in pixels (square) |
| `-t, --thickness` | `1` | Line thickness for individual gestures |
| `-c, --color` | `rgba(0,0,0,0.5)` | Colour for individual gestures (`rgba(r,g,b,a)` or named colour) |
| `-T, --summary-thickness` | `10` | Line thickness for the summary shape |
| `-C, --summary-color` | `#F00` (red) | Colour for the summary shape |
| `-f, --format` | `png` | Image format: `png`, `jpg`, `jpeg`, `pdf`, `svg` |
| `-o, --output` | *(stdout, base64)* | Output image file |

---

### Other commands

- `relacc-pairwise` — pairwise one-vs-one comparison between a reference file/directory and a candidate file/directory. See `relacc-pairwise -h`.
- Legacy entry points: `python3 main.py`, `python3 main-canvas.py`, `python3 main-pairwise.py`.

## Run tests

```bash
python3 -m pytest --cov=relacc --cov-fail-under=100 --cov-report=term-missing
```

## Gestures CSV format

Input gesture files are space-separated CSV with header:

```text
stroke_id x y time is_writing
```

Where:
- `stroke_id`: stroke index, starting at `0`
- `x`: horizontal coordinate
- `y`: vertical coordinate
- `time`: absolute or relative timestamp (milliseconds)
- `is_writing`: pen down/up flag

Expected filename pattern:
`subject_name-gesture_name-trial_number.csv`

## License

This software is distributed under the MIT License.
