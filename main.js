#!/usr/bin/env node
/*
 * This project is a JavaScript port of the C# Gesture RElative Accuracy Toolkit
 * (GREAT) by Radu-Daniel Vatavu, Lisa Anthony, and Jacob O. Wobbrock.
 *
 * The JavaScript code is distributed under MIT License (MIT).
 * Copyright (c) 2015 Luis A. Leiva.
 *
 * The original C# code is distributed under the "New BSD License" agreement.
 * Copyright (c) 2013, Radu-Daniel Vatavu, Lisa Anthony, and Jacob O. Wobbrock.
 */

// Standard libs.
var fs    = require("fs")
  , util  = require("util")
  , path  = require("path")
  , Stats = require('fast-stats').Stats
  ;
// RelAcc libs.
var RelAcc         = require("./lib/relacc")
  , Point          = require("./lib/geom/point")
  , PointSet       = require("./lib/geom/pointset")
  , Gesture        = require("./lib/gestures/gesture")
  , SummaryGesture = require("./lib/gestures/summarygesture")
  , PtAlignType    = require("./lib/gestures/ptaligntype")
  ;
// Util libs.
var ArgUtil   = require("./lib/utils/args")
  , CSVUtil   = require("./lib/utils/csv")
  , JSONUtil  = require("./lib/utils/json")
  , DateUtil  = require("./lib/utils/date")
  , DebugUtil = require("./lib/utils/debug")
  , MathUtil  = require("./lib/utils/math")
  ;

// This script takes as input gesture files in a particular CSV format,
// thus you may need to convert your gesture files in the first place.
// Usage example: ./main.js -v -l arrow /path/to/csv1dollar/*arrow*slow*.csv -f json | python -mjson.tool

var getopt  = require('node-getopt')
  , cliConf = [
    ['l' , 'label=ARG'     , 'Gesture label (inferred from file names if not supplied)'],
    ['r' , 'rate=ARG'      , 'Gesture sampling rate (default: smart guess)'],
    ['a' , 'alignment=ARG' , 'Alignment type: 1 (chronological, default) or 2 (cloud match)'],
    ['m' , 'summary=ARG'   , 'Summary gesture: none (default), centroid, medoid, kcentroid, kmedoid'],
    ['p' , 'popular'       , 'Consider only popular gesture articulations (default: false)'],
    ['s' , 'stats'         , 'Compute stats (default: false), otherwise will show each observation separately'],
    ['o' , 'output=ARG'    , 'Output file (default: stdout)'],
    ['f' , 'format=ARG'    , 'Output format: JSON (default), CSV, XML'],
    ['v' , 'verbose'       , 'Display debug info'],
    ['h' , 'help'          , 'Display this help']
  ]
  , cli       = getopt.create(cliConf).bindHelp()
  , opt       = cli.parseSystem()
  , args      = opt.options
  , argParser = new ArgUtil(args)
  , debug     = new DebugUtil({ verbose: args.verbose })
  , defaults  = {
      label     : undefined
    , rate      : undefined
    , alignment : PtAlignType.CHRONOLOGICAL
    , summary   : undefined
    , popular   : false
    , stats     : false
    , output    : undefined
    , format    : "json"
  };

function getStats(arr) {
  var n    = arr.length
    , mean = 0
    , mdn  = 0
    , sd   = 0
    , min  = 0
    , max  = 0
    ;
  if (n > 0) {
    var stats = new Stats().push(arr);
    var range = stats.range();
    mean = stats.amean();
    mdn  = stats.median();
    sd   = stats.stddev();
    min  = range[0];
    max  = range[1];
  }
	// Note: roundTo uses 3 decimal places by default.
  return {
      mean : MathUtil.roundTo(mean)
    , mdn  : MathUtil.roundTo(mdn)
    , sd   : MathUtil.roundTo(sd)
    , min  : MathUtil.roundTo(min)
    , max  : MathUtil.roundTo(max)
    , n    : n
  }
};

function evaluate(collection) {
  // Compute summary gesture according to given options.
  var alignmentType = argParser.get('alignment', defaults.alignment, parseInt);
  var summaryShape  = argParser.get('summary', defaults.summary);
  var popularShape  = argParser.get('popular', defaults.popular);
  var displayStats  = argParser.get('stats', defaults.stats, Boolean);

  // Remember settings.
  defaults.alignment = alignmentType;
	defaults.summary   = summaryShape;
	defaults.popular   = popularShape;
	defaults.stats     = displayStats;

  var gestures = [];
  var files = Object.keys(collection);
  files.forEach(function(file) {
		var points = collection[file];
    gestures.push(  new Gesture(points, label, rate) );
  });

  var taskAxis = new SummaryGesture(gestures, alignmentType, summaryShape, popularShape);

  var shapeError          = []
    , shapeVariability    = []
    , lengthError         = []
    , sizeError           = []
    , bendingError        = []
    , bendingVariability  = []
    , timeError           = []
    , timeVariability     = []
    , velocityError       = []
    , velocityVariability = []
    , strokeError         = []
    , strokeOrderError    = []
    ;

  gestures.forEach(function(gesture) {
    // Compute measures: Each gesture is compared against the collection's task axis.
    var shE = RelAcc.shapeError(gesture, taskAxis)
      , shV = RelAcc.shapeVariability(gesture, taskAxis)
      , LE  = RelAcc.lengthError(gesture, taskAxis)
      , szE = RelAcc.sizeError(gesture, taskAxis)
      , BE  = RelAcc.bendingError(gesture, taskAxis)
      , BV  = RelAcc.bendingVariability(gesture, taskAxis)
      , TE  = RelAcc.timeError(gesture, taskAxis)
      , TV  = RelAcc.timeVariability(gesture, taskAxis)
      , VE  = RelAcc.velocityError(gesture, taskAxis)
      , VV  = RelAcc.velocityVariability(gesture, taskAxis)
      , skE = RelAcc.strokeError(gesture, taskAxis)
      , sOE = RelAcc.strokeOrderError(gesture, taskAxis)
      ;
    shapeError.push(shE);
    shapeVariability.push(shV);
    lengthError.push(LE);
    sizeError.push(szE);
    bendingError.push(BE);
    bendingVariability.push(BV);
    timeError.push(TE);
    timeVariability.push(TV);
    velocityError.push(VE);
    velocityVariability.push(VV);
    strokeError.push(skE);
    strokeOrderError.push(sOE);
  });

  if (displayStats) {
    var res, stats = {
        shapeError          : getStats(shapeError)
      , shapeVariability    : getStats(shapeVariability)
      , lengthError         : getStats(lengthError)
      , sizeError           : getStats(sizeError)
      , bendingError        : getStats(bendingError)
      , bendingVariability  : getStats(bendingVariability)
      , timeError           : getStats(timeError)
      , timeVariability     : getStats(timeVariability)
      , velocityError       : getStats(velocityError)
      , velocityVariability : getStats(velocityVariability)
      , strokeError         : getStats(strokeError)
      , strokeOrderError    : getStats(strokeOrderError)
    };
    switch (format) {
      case 'json':
        res = toJSON(stats);
        break;
      case 'csv':
        res = toCSV(stats);
        break;
      case 'xml':
        res = toXML(stats);
        break;
      default:
        throw new Error(util.format("Invalid output format (%s). Supported formats: json, csv, xml.", format));
        break;
    }
    displayResults(res);
  } else {
    // Display each observation separately. Useful for debugging.
    // Each gesture has been compared against the collection's task axis.
    console.log(
			"file",
      "shapeError", "shapeVariability", "lengthError", "sizeError", "bendingError", "bendingVariability",
      "timeError", "timeVariability", "velocityError", "velocityVariability",
      "strokeError", "strokeOrderError"
    );
    for (var i = 0; i < files.length; i++) {
      console.log(
				files[i],
				// Geometric measures
        MathUtil.roundTo(shapeError[i]), MathUtil.roundTo(shapeVariability[i]),
				MathUtil.roundTo(lengthError[i]), MathUtil.roundTo(sizeError[i]),
				MathUtil.roundTo(bendingError[i]), MathUtil.roundTo(bendingVariability[i]),
				// Kinematic measures
        MathUtil.roundTo(timeError[i]), MathUtil.roundTo(timeVariability[i]),
				MathUtil.roundTo(velocityError[i]), MathUtil.roundTo(velocityVariability[i]),
				// Articulation measures
        MathUtil.roundTo(strokeError[i]), MathUtil.roundTo(strokeOrderError[i])
      );
    }
  }
};

function displayResults(res) {
  if (output) {
    // Save results to file.
    fs.writeFile(output, res, function(err) {
      if (err) debug.fmt("Cannot write to %s", output);
      else debug.fmt("Results were saved in %s", output);
    });
  } else {
    // Display results in stdout.
    console.log(res);
  }
};

function toJSON(obj) {
  var meta = {
    date: DateUtil.utc(),
    time: DateUtil.now(),
    args: defaults
  };
  return JSON.stringify({ metadata: meta, results: obj });
};

function toCSV(obj) {
  var sep = " "
    , csv = ['measure n mean mdn sd min max'.split(' ').join(sep)]
    , fmt = '%s %d %d %d %d %d %d'.split(' ').join(sep)
    , val
    ;
  for (var key in obj) {
    val = obj[key];
    csv.push( util.format(fmt, key, val.n, val.mean, val.mdn, val.sd, val.min, val.max) );
  }
  return csv.join('\n');
};

function toXML(obj) {
  var val, entry, xml = ['<?xml version="1.0" encoding="UTF-8"?>'];
  // Auxiliar function to format indentation.
  function indent(level) {
    if (typeof level === 'undefined') level = 2;
    return Array(level).join("  ");
  };

  xml.push( '<root>' );
  xml.push( util.format('%s<metadata date="%s" time="%d" />', indent(), DateUtil.utc(), DateUtil.now()) );

  var strArgs = [];
  for (var a in defaults) strArgs.push( util.format('%s="%s"', a, defaults[a]) );
  xml.push( util.format('%s<args %s />', indent(), strArgs.join(' ')) );

  xml.push( util.format('%s<results>', indent()) );
  for (var key in obj) {
    val = obj[key];
    entry = util.format('%s<%s n="%d" mean="%d" mdn="%d" sd="%d" min="%d" max="%d" />',
              indent(3), key, val.n, val.mean, val.mdn, val.sd, val.min, val.max);
    xml.push(entry);
  }
  xml.push( util.format('%s</results>', indent()) );
  xml.push( '</root>' );
  return xml.join('\n');
};

// Read options.
var nfiles = opt.argv.length
  , label  = argParser.get('label')
  , rate   = argParser.get('rate', Number)
  , output = argParser.get('output')
  , format = (output ? path.extname(output).substr(1) : argParser.get('format', defaults.format)).toLowerCase()
  , collection = {} // Index gesture points by filename
  ;

// Remember settings.
defaults.label  = label;
defaults.rate   = rate;
defaults.output = output;
defaults.format = format;

if (!nfiles) {
  cli.showHelp();
  throw new Error("Please provide some gesture files as input.");
}

if (!label) {
  // Assume that all input files are named subject-label-whatever_suffix.csv.
  label = defaults.label = path.basename(opt.argv[0]).split("-")[1];
  debug.fmt("Notice: No gesture label provided, I'll assume that all samples are '%s'.", label);
}


var maxStrokeCount = 1;

function doneParsing(file, points) {
  // This is for smart guessing the most appropriate sampling rate, if not set.
  var strokeCount = PointSet.countStrokes(points);
  if (strokeCount > maxStrokeCount) maxStrokeCount = strokeCount;
  // Instead of storing gesture samples right now, save the points and estimate sampling rate.
  var ext = path.extname(file);
  collection[ path.basename(file, ext) ] = points;
  if (Object.keys(collection).length == nfiles) {
    debug.fmt("Processed %s files", nfiles);
    if (!rate) {
      var smartRate = Math.max( 24, MathUtil.factorial(maxStrokeCount) );
      debug.fmt("Notice: Setting sampling rate to %s points per gesture.", smartRate);
      rate = defaults.rate = smartRate;
    }
    //console.error(defaults);
    //console.error(args);
    evaluate(collection);
  }
};

opt.argv.forEach(function(file) {
  var ext = path.extname(file);
  if (ext == '.csv') {
    CSVUtil.readGesture(file, function(points) {
      doneParsing(file, points);
    });
  } else if (ext == '.json') {
    JSONUtil.readGesture(file, function(points) {
      doneParsing(file, points);
    });
  } else {
    throw new Error(util.format("Invalid input file format (%s). Supported formats: json, csv.", ext));
  }
});
