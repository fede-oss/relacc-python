# Relative Accuracy Measures for Stroke Gestures (Python)

This repository contains a Python port of the Gesture Relative Accuracy Toolkit (GREAT).

## Requirements

- Python 3.11+
- `matplotlib` (for canvas rendering CLI)

## Installation

```bash
python3 -m pip install -e .
```

## Usage

1. Generate metric reports (`relacc`):

```bash
relacc -s -r 32 -a 1 -m centroid -f json /path/to/*gesture_name*.csv
```

2. Render gesture visualizations (`relacc-canvas`):

```bash
relacc-canvas -r 32 -a 1 -m centroid -o /tmp/sample.png /path/to/*gesture_name*.csv
```

3. Show CLI help:

```bash
relacc -h
relacc-canvas -h
```

Legacy script entry points are still available:

```bash
python3 main.py -h
python3 main-canvas.py -h
```

## Common Flags

### `main.py` (metrics)

- `-l, --label`: gesture label (if omitted, inferred from first filename)
- `-r, --rate`: sampling rate
- `-a, --alignment`: point alignment mode
- `-m, --summary`: summary shape strategy
- `-p, --popular`: use popular summary shape
- `-s, --stats`: output aggregate stats instead of per-file rows
- `-f, --format`: output format (`json`, `csv`, `xml`)
- `-o, --output`: write output to file
- `-v, --verbose`: verbose logs

### `main-canvas.py` (rendering)

- `-l, --label`: gesture label (if omitted, inferred from first filename)
- `-r, --rate`: sampling rate
- `-a, --alignment`: point alignment mode
- `-m, --summary`: summary shape strategy
- `-p, --popular`: use popular summary shape
- `-s, --size`: image canvas size in pixels
- `-t, --thickness`: gesture line thickness
- `-c, --color`: gesture color
- `-T, --summary-thickness`: summary line thickness
- `-C, --summary-color`: summary color
- `-f, --format`: output image format (`png`, `jpg`, `jpeg`, `pdf`, `svg`)
- `-o, --output`: write image to file (format inferred from extension if present)
- `-v, --verbose`: verbose logs

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
