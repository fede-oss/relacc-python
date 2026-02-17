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

var csv   = require("fast-csv")
  , Point = require("../geom/point")
  ;

/**
 * Process CSV gesture files.
 * @module lib/utils/csv
 */
module.exports = {

  /**
   * Reads a gesture file, in CSV format.
   * The CSV file must have a header and use spaces as field separator.
   * @example CSV file example:
   * stroke_id x y time is_writing
   * 0 69 228 0 1
   * 0 67 228 13 1
   * 0 70 225 41 1
   * 0 73 223 49 1
   * 0 75 222 55 1
   * ...
   * @static
   * @param {String} file - Path to CSV file.
   * @param {Function} callback - Function to call when the file is processed.
   * @return {Array} Gesture points.
   * @example Usage example: var CSV = require("./lib/utils/csv");
   * var gesture = CSV.readGesture("/path/to/file.csv");
   * console.log(gesture); // output: [ [0, 69, 228, 0], ... ]
   */
  readGesture: function(file, callback) {
    var points = [];
    var myTime; // Remove duplicated timestamps, if any.

    csv.fromPath(file, { headers: true, delimiter: " " })
       .on("data", function(data) {
         var x = parseInt(data.x, 10)
           , y = parseInt(data.y, 10)
           , time = parseInt(data.time, 10)
           , sid = parseInt(data.stroke_id, 10)
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
       })
       .on("end", function() {
         callback(points);
       });
  }

};
