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

1. Generate metric reports:

```bash
python3 main.py -s -r 32 -a 1 -m centroid -f json /path/to/*gesture_name*.csv
```

2. Render gesture visualizations:

```bash
python3 main-canvas.py -r 32 -a 1 -m centroid -o /tmp/sample.png /path/to/*gesture_name*.csv
```

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
