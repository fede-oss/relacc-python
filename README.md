# Relative Accuracy Measures for Stroke Gestures

This project is a JavaScript port of the C# Gesture RElative Accuracy Toolkit (GREAT) by Radu-Daniel Vatavu, Lisa Anthony, and Jacob O. Wobbrock.

Over time, new features have been incorporated to the code, so the project is now more mature.

## Installation

`npm install`

Note: The [node-canvas lib](https://github.com/Automattic/node-canvas/wiki) requires some OS dependencies.
Please be sure your OS meets them all prior to installing this software.

## Usage

Note: Input gesture files must be in a [particular CSV format](#gestures-csv-format).

1. Generate a report for later analysis:

```
./main.js -v -r 32 -a 1 -m centroid -o /tmp/sample.json /path/to/*gesture_name*.csv
```

2. Visualize all gesture samples of a given class:

```
./main-canvas.js -v -r 32 -a 1 -m centroid -o /tmp/sample.png /path/to/*gesture_name*.csv
```

### In the browser

```
npm install browserify -g
browserify lib/relacc.js -o relacc.js
```

Then include the generated file on your page like any other JS file:

`<script src="relacc.js"></script>`

## Run tests

```
npm install jasmine-node -g
npm test
```

## Generate documentation

```
npm install jsdoc -g
npm run docs
```

## License

This software is distributed under the [MIT License](LICENSE) (MIT).
Copyright (c) 2015 Luis A. Leiva.

## Appendix

### Gestures CSV format

This software assumes that gesture files are in CSV format, space separated, with the following header:

```
stroke_id x y time is_writing
```

where:

* `stroke_id` is an integer denoting stroke index, starting at 0.
* `x` is an integer denoting the horizontal point coordinate.
* `y` is an integer denoting the vertical point coordinate.
* `time` is an integer denoting the time of the point coordinate,
either as absolute (millisecond-based) timestamps
or relative milliseconds, where the first point of the first stroke has time 0.
* `is_writing` is an integer denoting a pen down (0) or pen up (1) stroke.

Example of a gesture file with absolute timestamps:
```
stroke_id x y time is_writing
0 69 228 1341355148128 1
0 67 228 1341355148141 1
0 70 225 1341355148182 1
0 73 223 1341355148231 1
0 75 222 1341355148286 1
...
```

Example of a gesture file with relative timestamps:
```
stroke_id x y time is_writing
0 69 228 0 1
0 67 228 13 1
0 70 225 41 1
0 73 223 49 1
0 75 222 55 1
...
```

Also, this software assumes that gesture file names follow this pattern:
`subject_name-gesture_name-trial_number.csv`.

Examples of *valid* gesture names:
- :+1: `s01-arrow-t01.csv`
- :+1: `s1-arrow-fast-t5.csv`
- :+1: `john_doe-arrow-t10.csv`

Examples of *invalid* gesture names:
- :no_entry_sign: `arrow-t01.csv`
- :no_entry_sign: `s1-arrow.csv`
- :no_entry_sign: `john_doe-t10.csv`

### More Information

Visit https://luis.leiva.name/relacc/
