# Relative Accuracy Measures for Stroke Gestures (Python)

This repository contains a Python port of the Gesture Relative Accuracy Toolkit (GREAT).

## Requirements

- Python 3.11+
- `matplotlib` (for canvas rendering CLI)
- `scipy` (for distribution-comparison metrics)

## Installation

```bash
python3 -m pip install -e .
```

If your system Python is marked as "externally managed" (common on newer Debian/Ubuntu-based setups), use:

```bash
python3 -m pip install --break-system-packages -e .
```

## One-vs-Many Evaluation

Both `relacc` and `relacc-canvas` work the same way conceptually:

1. You pass **multiple CSV files** for the same gesture (e.g. 10 trials of an arrow gesture).
2. A single **summary gesture** (the reference) is computed from the whole collection.
3. Every individual sample is measured against that reference, producing metrics per sample.

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
| `dtwDistance` | Classic Dynamic Time Warping total alignment cost |
| `ldtwDistance` | DTW normalized by warping-path length |
| `ddtwDistance` | Derivative DTW, comparing local trajectory trends |
| `wdtwDistance` | Weighted DTW, penalizing larger phase offsets |
| `wddtwDistance` | Weighted derivative DTW |

Lower values always mean closer to the reference. Passing a single CSV produces all zeros (reference = the file itself).

The DTW-family metrics are computed on the chronological point sequences after the same resampling and translation steps used elsewhere in the toolkit.
The Python API uses the exact DTW dynamic program, so runtime is quadratic in the resampled point count.
For the weighted variants, the logistic phase-penalty slope defaults to `0.25` (`penalty_g`) and can be overridden from the Python API when stricter or looser off-diagonal penalties are needed.
On the CLI, the DTW-family metrics are included by default. Smaller resampling rates stay exact; larger resampling rates automatically switch to a Sakoe-Chiba-style band for faster approximate runs. Use `--exact-dtw` to force exact DTW, or `--dtw-window N` to choose your own window radius.

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

`centroid` and `medoid` produce a synthetic shape that may not correspond to any real recording. `kcentroid` and `kmedoid` always return a real gesture from the input.
Use `-p` (`--popular`) to filter out gestures whose stroke count differs from the most common one before building the summary.

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

# Use a custom DTW band of 8 points for faster approximate DTW
relacc -s -m centroid -r 32 --dtw-window 8 -f json \
  datasets/becaptcha/1dollar/s01-arrow-fast-*.csv

# Disable the DTW band and run exact DTW
relacc -s -m centroid -r 32 --exact-dtw -f json \
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
| `--dtw-window` | auto for larger rates | Optional Sakoe-Chiba band radius for faster approximate DTW runs |
| `--exact-dtw` | off | Force exact DTW-family metrics even at larger resampling rates |
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

### `relacc-pairwise` — direct and summary-reference comparison

`relacc-pairwise` supports 2 modes:

- `--mode direct` (default): strict reference-vs-candidate pairing.
  - file vs file: one pair
  - directory vs directory: matched first by relative CSV path (`nested/a.csv` ↔ `nested/a.csv`), then by unambiguous layout-tolerant keys (`dataset/realTO/a.csv` ↔ `dataset/a.csv`, or unique `a.csv` filenames)
  - `--strict/--no-strict` controls whether unmatched files fail or are skipped
- `--mode summary`: candidate-vs-summary-reference.
  - all CSVs in `reference` are used to build one summary gesture
  - every candidate CSV in `candidate` is compared against that summary
  - this transfers the `relacc` summary-gesture logic to cross-dataset evaluation

Examples:

```bash
# Strict one-to-one pairwise (default mode)
relacc-pairwise references/ candidates/ -f csv -o /tmp/direct.csv

# Candidate-vs-summary-reference (multiple references -> one summary)
relacc-pairwise --mode summary -m centroid references/ generated/ -f json

# Same, but use a custom DTW band of 8 points
relacc-pairwise --mode summary -m centroid --dtw-window 8 \
  references/ generated/ -f json

# Or disable the band and run exact DTW
relacc-pairwise --mode summary -m centroid --exact-dtw \
  references/ generated/ -f json
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | `direct` | Comparison mode: `direct` or `summary` |
| `-m, --summary` | *(first reference gesture)* | Summary strategy: `centroid`, `medoid`, `kcentroid`, `kmedoid` |
| `-p, --popular` | off | Filter to most common stroke count when building summary |
| `--strict / --no-strict` | strict | Only used by `direct` mode for directory matching |
| `-r, --rate` | auto | Resampling rate (auto-estimated when omitted) |
| `-a, --alignment` | `0` | Point alignment: `0` chronological, `1` cloud-match |
| `--round` | `3` | Decimal precision in output metrics |
| `-f, --format` | `json` | Output format: `json`, `csv` |
| `-o, --output` | *(stdout)* | Write output to file |
| `--dtw-window` | auto for larger rates | Optional Sakoe-Chiba band radius for faster approximate DTW runs |
| `--exact-dtw` | off | Force exact DTW-family metrics even at larger resampling rates |

---

### `relacc-distribution` — class-aware distribution comparison

`relacc-distribution` compares **metric distributions** instead of individual pairs.

V1 workflow:

- Group samples by class.
- Build a **human-human baseline** distribution inside each class from unordered reference-reference pairs.
- Build a **generated-human** distribution inside each class from all reference-candidate pairs.
- Compare the two scalar distributions for every gesture metric, then pool an overall summary across valid classes.

Grouping modes:

- `--group-by filename-label` (default): derive the class from the second `-`-separated filename segment.
  - Example: `s01-arrow-01.csv` and `g03-arrow-02.csv` are both grouped as `arrow`.
- `--group-by parent-dir`: derive the class from the file's relative parent directory.
  - Example: `arrow/sample-1.csv` and `arrow/sample-2.csv` are grouped as `arrow`.

A class is considered valid when it has:

- at least 2 reference CSVs
- at least 1 candidate CSV

Output shape:

- JSON: `metadata` plus `results.perClass` and `results.overall`
- CSV: one flattened row per `scope x gestureMetric`

Examples:

```bash
# Default grouping by filename label
relacc-distribution references/ generated/ -f json

# Group by directory structure instead of filename label
relacc-distribution --group-by parent-dir references/ generated/ -o /tmp/distribution.csv
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--group-by` | `filename-label` | Class grouping mode: `filename-label` or `parent-dir` |
| `-m, --summary` | *(first reference gesture)* | Summary strategy reused from pairwise comparison |
| `-p, --popular` | off | Filter to most common stroke count when building the per-reference summary |
| `-r, --rate` | auto | Resampling rate; auto-estimated from reference samples only |
| `-a, --alignment` | `0` | Point alignment: `0` chronological, `1` cloud-match |
| `--round` | `3` | Decimal precision in output metrics |
| `-f, --format` | `json` | Output format: `json`, `csv` |
| `-o, --output` | *(stdout)* | Write output to file |
| `--dtw-window` | auto for larger rates | Optional Sakoe-Chiba band radius for faster approximate DTW runs |
| `--exact-dtw` | off | Force exact DTW-family metrics even at larger resampling rates |

Current distribution metrics:

- `wassersteinDistance`
- `energyDistance`
- `ksStatistic`
- `ksPValue`

---

### Other commands

- Legacy entry points: `python3 main.py`, `python3 main-canvas.py`, `python3 main-pairwise.py`, `python3 main-distribution.py`.

## Add a new metric

There are two metric families:

- Gesture metrics (`relacc/metrics.py`): used by both `relacc` and `relacc-pairwise`.
- Distribution metrics (`relacc/distribution_metrics.py`): used by `relacc-distribution`

### 1) Add a gesture metric (shared by `relacc` and `relacc-pairwise`)

1. Implement the metric function in `relacc/relacc.py` (signature: `(gesture, summaryShape) -> float`).
2. Register it in `relacc/metrics.py` by appending an entry to `BASE_METRIC_DEFINITIONS` or `DTW_METRIC_DEFINITIONS`, for example:
   `("myMetric", RelAcc.myMetric)`.

After step 2, it is automatically included in:
- `relacc` per-file output and `-s` stats output
- `relacc-pairwise` JSON/CSV output

### 2) Add a distribution metric

1. Implement a function in `relacc/distribution_metrics.py` with signature:
   `(reference_values, candidate_values) -> float`.
2. Register it in `_DISTRIBUTION_METRIC_DEFINITIONS`, for example:
   `("myDistributionMetric", my_distribution_metric)`.

This keeps distribution logic modular and independent from pairwise evaluation.



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
