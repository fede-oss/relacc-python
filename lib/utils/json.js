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

var Point = require("../geom/point")
  ;

/**
 * Process JSON gesture files.
 * @module lib/utils/json
 */
module.exports = {

  /**
   * Reads a gesture file, in json format.
   * @example JSON file example:
   * [[[69, 228, 0, 1], [67, 228, 13, 1], [70 225 41 1], [73 223 49 1], [75 222 55 1], ...]]
   * @static
   * @param {String} file - Path to JSON file.
   * @param {Function} callback - Function to call when the file is processed.
   * @return {Array} Gesture points.
   * @example Usage example: var JSON = require("./lib/utils/json");
   * var gesture = JSON.readGesture("/path/to/file.json");
   * console.log(gesture); // output: [ [0, 69, 228, 0], ... ]
   */
  readGesture: function(file, callback) {
    var points = [];
    var myTime; // Remove duplicated timestamps, if any.

    var strokes = require(file).strokes;
    for (var sid = 0; sid < strokes.length; sid++) {
      var stroke = strokes[sid];
      for (var pid = 0; pid < stroke.length; pid++) {
          var pt = stroke[pid];
          // NOTE: Some datasets might not have timestamps.
          var x = pt[0]
            , y = pt[1]
            , time = pt[2]
            ;
         if (time >= 0) {
           if (time != myTime) {
             points.push( new Point(x, y, time, sid) );
           }
           myTime = time;
         } else {
           // Some datasets don't have timestamps, so let's be consistent.
           points.push( new Point(x, y, NaN, sid) );
         }
      }
    }
    callback(points);
  }

};
