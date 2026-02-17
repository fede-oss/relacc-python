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
var fs   = require("fs")
  , util = require("util")
  , path = require("path")
  , readline = require('readline')
  ;
// Canvas lib. See install dependencies here:
// See: https://github.com/Automattic/node-canvas/wiki
var Canvas = require("canvas");
// RelAcc libs.
var Point          = require("./lib/geom/point")
  , PointSet       = require("./lib/geom/pointset")
  , Gesture        = require("./lib/gestures/gesture")
  , SummaryGesture = require("./lib/gestures/summarygesture")
  , PtAlignType    = require("./lib/gestures/ptaligntype")
  ;
// Util libs.
var ArgUtil   = require("./lib/utils/args")
  , CSVUtil   = require("./lib/utils/csv")
  , DateUtil  = require("./lib/utils/date")
  , DebugUtil = require("./lib/utils/debug")
  , MathUtil  = require("./lib/utils/math")
  ;

// This script takes as input gesture files in a particular CSV format,
// thus you may need to convert your gesture files in the first place.
// Usage example: ./main-canvas.js -v /path/to/csv1dollar/*arrow*slow*.csv

var getopt  = require('node-getopt')
  , cliConf = [
    ['l' , 'label=ARG'     , 'Gesture label (inferred from file names if not supplied)'],
    ['r' , 'rate=ARG'      , 'Gesture sampling rate (default: smart guess)'],
    ['a' , 'alignment=ARG' , 'Alignment type: 1 (chronological, default) or 2 (cloud match)'],
    ['m' , 'summary=ARG'   , 'Summary gesture: none (default), centroid, medoid, kcentroid, kmedoid'],
    ['p' , 'popular'       , 'Consider only popular gesture articulations (default: false)'],
    ['o' , 'output=ARG'    , 'Output file (default: stdout)'],
    ['f' , 'format=ARG'    , 'Output format: <img/> element (default if no output specified), PNG, JPG, PDF or SVG'],
    ['s' , 'size=ARG'      , 'Output image size, in px (default: 500)'],
    ['t' , 'thickness=ARG' , 'Line width of gestures, in px (default: 1)'],
    ['c' , 'color=ARG'     , 'Line color of gestures (default: rgba(0,0,0, .5))'],
    ['T' , 'summary-thickness=ARG' , 'Line width of the summary gesture, in px (default: 10)'],
    ['C' , 'summary-color=ARG'     , 'Line color of the summary gesture (default: #F00)'],
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
    , size      : undefined
    , alignment : PtAlignType.CHRONOLOGICAL
    , summary   : undefined
    , popular   : false
    , output    : null
    , format    : "csv"
    , thickness : 1
    , color     : "rgba(0,0,0, .5)"
    , summaryThickness : 10
    , summaryColor     : "#F00"
  }
  ;

// Read options.
var nfiles = opt.argv.length
  , label  = defaults.label  = argParser.get('label')
  , rate   = defaults.rate   = argParser.get('rate', Number)
  , output = defaults.output = argParser.get('output', defaults.output)
  , format = defaults.format = (output ? path.extname(output).substr(1) :
                                  argParser.get('format', "img")).toLowerCase()
  , imsize = defaults.size   = argParser.get('size', 500, parseInt)
  , alignmentType = defaults.alignment = argParser.get('alignment', defaults.alignment, parseInt)
  , summaryShape  = defaults.summary   = argParser.get('summary', defaults.summary)
  , popularShape  = defaults.popular   = argParser.get('popular', defaults.popular, Boolean)
  , lineWidth     = defaults.thickness = argParser.get('thickness', defaults.thickness, parseInt)
  , lineColor     = defaults.color     = argParser.get('color', defaults.color)
  , summaryWidth  = defaults.summaryThickness = argParser.get('summary-thickness', defaults.summaryThickness, parseInt)
  , summaryColor  = defaults.summaryColor     = argParser.get('summary-color', defaults.summaryColor)
  , collection = []
  ;

if (!nfiles) {
  cli.showHelp();
  throw new Error("Please provide some gesture files as input.");
}

if (!label) {
  // Assume that all input files are named subject-label-whatever.csv.
  label = defaults.label = path.basename(opt.argv[0]).split("-")[1];
  debug.fmt("Notice: No gesture label provided, I'll assume that all samples are '%s'.", label);
}

function evaluate() {
  var canvasType = ['pdf', 'svg'].indexOf(format) > -1 ? format : 'hey'; // for some reason, hey == image
  // In Canvas v2 there's no constructor anymore; see https://www.npmjs.com/package/canvas
  var cnv = Canvas.createCanvas(imsize, imsize, canvasType);

  // Add white background for JPG images, since they don't support transparency
  // and thus the Canvas lib will use the black color by default.
  if (['jpg', 'jpeg'].indexOf(format) > -1) {
    var ctx = cnv.getContext('2d');
    ctx.fillStyle = '#FFF';
    ctx.fillRect(0, 0, imsize, imsize);
  }

  var gestures = collection.map(function(points) {
    return new Gesture(points, label, rate);
  });
  var summaryGesture = new SummaryGesture(gestures, alignmentType, summaryShape, popularShape);
  gestures.forEach(function(gesture) {
    var gesturePts = summaryGesture.alignGesture(gesture, alignmentType);
    drawGesture(cnv, gesturePts);
  });
  if (summaryShape) {
    var summaryPts = summaryGesture.getPoints();
    drawGesture(cnv, summaryPts, {
      lineWidth   : summaryWidth,
      strokeStyle : summaryColor,
    });
  }
  displayResult(cnv);
};

function displayResult(canvas) {
  if (output) {
    // Save results to file.
    var stream;
    switch (format) {
      case 'png':
        stream = canvas.createPNGStream();
        saveRasterFile(stream);
        break;
      case 'jpg':
      case 'jpeg':
        stream = canvas.createJPEGStream({ quality: 0.9 });
        saveRasterFile(stream);
        break;
      case 'pdf':
      case 'svg':
        stream = canvas.toBuffer();
        saveVectorFile(stream);
        break;
      default:
        throw new Error(util.format("Invalid image format (%s). Supported formats: jpg, jpeg, png, pdf, svg.", format));
        break;
    }
  } else {
    // Display results in stdout.
    console.log('<img src="' + canvas.toDataURL() + '" />');
  }
};

function saveRasterFile(stream) {
  var out = fs.createWriteStream(output);
  stream.on('end', function() {
    debug.fmt("Results were saved in %s", output);
  });
  stream.on('error', function(err) {
    debug.fmt("Cannot write to %s", output);
  });
  stream.pipe(out);
};

function saveVectorFile(stream) {
  fs.writeFile(output, stream, function(err) {
    if (err) return debug.fmt("Cannot write to %s", output);
    debug.fmt("Results were saved in %s", output);
  });
};

function drawGesture(canvas, points, style) {
  var ctx = canvas.getContext('2d');
//  var r = Math.ceil( Math.random() * 255 )
//    , g = Math.ceil( Math.random() * 255 )
//    , b = Math.ceil( Math.random() * 255 )
//    ;
  ctx.lineWidth   = style && style.lineWidth   || lineWidth; // 1
  ctx.strokeStyle = style && style.strokeStyle || lineColor; // 'rgba(255,255,255,0.5)';
  var trPts = PointSet.clone(points);
  // Leverage as much space onscreen as possible.
  var bounds = PointSet.boundingBox(trPts);
  var boundsDiff = Math.max( canvas.width - bounds.width(), canvas.height - bounds.height() );
  var scale = boundsDiff > 0 ? boundsDiff/canvas.width : canvas.width/boundsDiff;
  var padScale = 0.9*(1 + scale);
  trPts = PointSet.scaleTo(trPts, padScale);
  var offPt = new Point(-canvas.width/2, -canvas.height/2);
  trPts = PointSet.translateBy(trPts, offPt);
  // Begin path for first stroke.
  ctx.beginPath();
  ctx.lineCap = "round";
  for (var i = 0; i < trPts.length - 1; i++) {
    var currPt = trPts[i];
    var nextPt = trPts[i+1];
    ctx.moveTo(currPt.X, currPt.Y);
    if (nextPt.StrokeID == currPt.StrokeID) {
      ctx.lineTo(nextPt.X, nextPt.Y);
    } else {
      ctx.stroke();
      // N-th stroke, close previous path?
      ctx.closePath();
      ctx.beginPath();
    }
  }
  // Finally close path for the last stroke.
  ctx.stroke();
  ctx.closePath();
};


var maxStrokeCount = 1;

function doneParsing(points) {
  // This is for smart guessing the most appropriate sampling rate, if not set.
  var strokeCount = PointSet.countStrokes(points);
  if (strokeCount > maxStrokeCount) maxStrokeCount = strokeCount;
  // Instead of storing gesture samples right now, save the points and estimate sampling rate.
  collection.push(points);
  if (collection.length == nfiles) {
    debug.fmt("Processed %s gesture files.", nfiles);
    if (!rate) {
      var smartRate = Math.max( 24, MathUtil.factorial(maxStrokeCount) );
      debug.fmt("Notice: Setting sampling rate to %s points per gesture.", smartRate);
      rate = defaults.rate = smartRate;
    }
    console.error(defaults);
    evaluate(collection);
  }
};

function readJsonFile(file, callback) {
  var strokes = require(file);
  readJsonStrokes(strokes, callback);
}

function readJsonStrokes(strokes, callback) {
	var points = [];
  for (var i = 0; i < strokes.length; i++) {
		var stroke = strokes[i];
    for (var j = 0; j < stroke.length; j++) {
			var x = stroke[j][0];
			var y = stroke[j][1];
			var t = stroke[j][2];
		  points.push( new Point(x,y,t, i) );
    }
  }
	callback(points);
}

// Script entry point.
if (opt.argv[0] == '-') {

  // Reset file count, since we need to read all lines.
  nfiles = 0;
  // Read from stdin.
  var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  var pending = [];
  rl.on('line', function(line) {
    // XXX: When reading from stdin, only JSON is supported.
    var strokes = JSON.parse(line);
    pending.push(strokes);

    nfiles++;
  })
  .on('close', function(line) {
    pending.forEach(function(strokes) {
      readJsonStrokes(strokes, doneParsing);
    });
  });

} else {

  // Read file list.
  opt.argv.forEach(function(file) {
    var ext = path.extname(file);
    if (ext == '.csv') {
      CSVUtil.readGesture(file, doneParsing);
    } else if (ext == '.json') {
      readJsonFile(file, doneParsing);
    } else {
      debug.fmt("Unknown file extension: %s", ext);
    }
  });

}

//// Quick Test:
//collection = [
//    new Gesture([
//      new Point(10,  10,  0, 0),
//      new Point(20,  50,100, 0),
//      new Point(100,100,200, 0),
//      new Point(110,110,300, 1),
//      new Point(20,  10,400, 1),
//      new Point(50,  50,500, 1),
//    ], label, rate),
//    new Gesture([
//      new Point(20,  20, 0,  0),
//      new Point(20,  50,100, 0),
//      new Point(120,120,200, 0),
//      new Point(120,110,300, 1),
//      new Point(40,  10,400, 1),
//      new Point(60,  50,500, 1),
//    ], label, rate),
//    new Gesture([
//      new Point(40,  40, 0,  0),
//      new Point(30,  50,100, 0),
//      new Point(140,140,200, 0),
//      new Point(100,120,300, 1),
//      new Point(20,  20,400, 1),
//      new Point(50,  30,500, 1),
//    ], label, rate)
//];
//evaluate(collection);
