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

var Vector      = require("./geom/vector");
var Measure     = require("./geom/measure");
var PointSet    = require("./geom/pointset");
var PtAlignType = require("./gestures/ptaligntype");

/**
 * Computes relative accuracy measures for stroke gestures based on the notion
 * of a "gesture task axis" (aka summary/representative shape/gesture).
 * @module lib/relacc
 */
module.exports = {

  // geometric relative accuracy measures: ShE, ShV, LE, SzE, BE, BV

  /**
   * Compute Shape Error (ShE).
   * Defined as the mean of the absolute Euclidean distances computed between gesture points and task axis points.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  shapeError: function(gesture, summaryShape) {
    var errors = this.localShapeErrors(gesture, summaryShape);
    return this.mean(errors);
  },
  /**
   * Compute Shape Variability (ShV).
   * Defined as the standard deviation of of all point distances to the gesture task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  shapeVariability: function(gesture, summaryShape) {
    var errors = this.localShapeErrors(gesture, summaryShape);
    return this.stdev(errors);
  },
  /**
   * Compute absolute Euclidean distances between gesture points and task axis points.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Array}
   */
  localShapeErrors: function(gesture, summaryShape) {
    // Align gesture points with the task axis.
    var summaryPts = summaryShape.getPoints();
    var gesturePts = summaryShape.alignGesture(gesture);
    // Compute errors.
    var errors = [];
    for (var i = 0; i < gesturePts.length; i++) {
      var err = Measure.distance(summaryPts[i], gesturePts[i]);
      errors.push(err);
    }
    return errors;
  },
  /**
   * Compute Length Error (LE).
   * Defined as the absolute difference in path length between the gesture and the task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  lengthError: function(gesture, summaryShape) {
    var gestureLength = PointSet.pathLength(gesture.points);
    var summaryLength = PointSet.pathLength(summaryShape.points);
    var error = Math.abs(gestureLength - summaryLength);
    //if (computeAsPercentage) error /= summaryLength;
    return error;
  },
  /**
   * Compute Size Error (SzE).
   * Defined as the absolute difference in area size between the gesture and the task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  sizeError: function(gesture, summaryShape) {
    var gestureArea = PointSet.boundingBox(gesture.points).area();
    var summaryArea = PointSet.boundingBox(summaryShape.points).area();
    var error = Math.abs(gestureArea - summaryArea);
    //if (computeAsPercentage) error /= summaryArea;
    return error;
  },
  /**
   * Compute Bending Error (BE).
   * Defined as the average of absolute differences in turning angle of each gesture w.r.t the task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  bendingError: function(gesture, summaryShape) {
    var errors = this.localBendingErrors(gesture, summaryShape);
    return this.mean(errors);
  },
  /**
   * Compute Bending Variability (BV).
   * Defined as the standard deviation of differences in turning angle of each gesture w.r.t the task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  bendingVariability: function(gesture, summaryShape) {
    var errors = this.localBendingErrors(gesture, summaryShape);
    return this.stdev(errors);
  },
  /**
   * Compute differences in turning angle of each gesture w.r.t the task axis.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Array}
   */
  localBendingErrors: function(gesture, summaryShape) {
    // Align gesture points with the task axis.
    var summaryPts = summaryShape.getPoints();
    var gesturePts = summaryShape.alignGesture(gesture);
    // Compute errors.
    var kSummary = this.turningAngleArray(summaryPts);
    var kGesture = this.turningAngleArray(gesturePts);
    var errors = [];
    for (var i = 0; i < gesturePts.length; i++) {
      var err = Math.abs(kGesture[i] - kSummary[i]);
      errors.push(err);
    }
    return errors;
  },
  /**
   * Compute the turning angle of each gesture point.
   * @static
   * @param points {Array} Input gesture points.
   * @return {Array}
   */
  turningAngleArray: function(points) {
    var n = points.length;
    var k = [];
    var r = 1;
    for (var i = 0; i < n; i++) {
      var angle = 0;
      if (i - r >= 0 && i + r < n) {
        angle = Measure.angle(
          new Vector(points[i], points[i + r]),
          new Vector(points[i - r], points[i])
        );
        if (angle > Math.PI) angle -= 2 * Math.PI;
      }
      k.push(angle);
    }
    return k;
  },

  // kinematic relative accuracy measures: TE, TV, VE, VV

  /**
   * Compute Time Error (TE).
   * Defined as the absolute difference in articulation time.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  timeError: function(gesture, summaryShape) {
    var summaryTime = this.productionTime(summaryShape);
    var gestureTime = this.productionTime(gesture);
    var error = Math.abs(gestureTime - summaryTime);
    //if (computeAsPercentage) error /= summaryTime;
    return error;
  },
  /**
   * Compute Time Variability (TV).
   * Defined as the standard deviation of local times.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  timeVariability: function(gesture, summaryShape) {
    // Align gesture points with the task axis.
    var summaryPts = summaryShape.getPoints();
    var gesturePts = summaryShape.alignGesture(gesture);
    // Compute errors.
    var errors = []
    for (var i = 0; i < gesturePts.length; i++) {
      errors.push( Math.abs(gesturePts[i].T - summaryPts[i].T) );
    }
    return this.stdev(errors);
  },
  /**
   * Compute Velocity Error (VE).
   * Defined as the mean of local speed differences.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  velocityError: function(gesture, summaryShape) {
    var errors = this.localSpeedErrors(gesture, summaryShape);
    return this.mean(errors);
  },
  /**
   * Compute Velocity Variability.
   * Defined as the standard deviation of local speed differences.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  velocityVariability: function(gesture, summaryShape) {
    var errors = this.localSpeedErrors(gesture, summaryShape);
    return this.stdev(errors);
  },
  /**
   * Compute the local speed differences between gesture points and task axis points.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Array}
   */
  localSpeedErrors: function(gesture, summaryShape) {
    // Align gesture points with the task axis.
    var summaryPts = summaryShape.getPoints();
    var gesturePts = summaryShape.alignGesture(gesture);
    // Compute errors.
    var vSummary = this.speedArray(summaryPts);
    var vGesture = this.speedArray(gesturePts);
    var errors = [];
    for (var i = 0; i < gesturePts.length; i++) {
      errors.push( Math.abs(vGesture[i] - vSummary[i]) );
    }
    return errors;
  },
  /**
   * Compute production time of a gesture articulation.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @return {Number}
   */
  productionTime: function(gesture) {
    var points = gesture.points;
    if (points.length <= 1) return 0;
    return points[points.length - 1].T - points[0].T;
  },
  /**
   * Compute local speed values for each gesture point.
   * @static
   * @param points {Array} Input gesture points.
   * @return {Array}
   */
  speedArray: function(points) {
    var n = points.length;
    var v = [];
    var r = 1;
    for (var i = 0; i < n; i++) {
      var index1 = Math.max(0, i - r);
      var index2 = Math.min(i + r, n - 1);
      var distance = PointSet.pathLength(points, index1, index2);
      var time = points[index2].T - points[index1].T;
      if (Math.abs(time) < 1e-5) {
        v.push(0);
      } else {
        v.push(distance/time);
      }
    }
    return v;
  },

  // articulation relative accuracy measures: SkE, SkOE -------------------------

  /**
   * Compute Stroke Error (SkE).
   * Defined as the absolute difference in stroke count.
   * @static
   * @param gesture {Gesture} Input gesture.
   * @param summaryShape {SummaryGesture} Task axis.
   * @return {Number}
   */
  strokeError: function(gesture, summaryShape) {
    return Math.abs(this.numStrokes(gesture) - this.numStrokes(summaryShape));
  },
  /**
   * Compute the number of strokes of a gesture.
   * @static
   * @param gesture {Gesture} Input gesture
   * @return {Number}
   */
  numStrokes: function(gesture) {
    var points = gesture.points;
    var numStk = points.length > 0 ? 1 : 0;
    for (var i = 1; i < points.length; i++) {
      if (points[i].StrokeID != points[i - 1].StrokeID) {
        numStk++;
      }
    }
    return numStk;
  },
  /**
   * Compute Stroke Order Error (SkOE).
   * Defined as the absolute difference between the $1 and $P recognizers' distance costs.
   * @static
   * @param gesture {Gesture} Input gesture
   * @param summaryShape {SummaryGesture} Task axis
   * @return {Number}
   */
  strokeOrderError: function(gesture, summaryShape) {
    // Align gesture points with the task axis.
    var summaryPts = summaryShape.getPoints();
    // Compute $1 cost measure.
    var gesturePts = summaryShape.alignGesture(gesture, PtAlignType.CHRONOLOGICAL);
    var oDollarCost = 0;
    for (var i = 0; i < gesturePts.length; i++) {
      oDollarCost += Measure.distance(gesturePts[i], summaryPts[i]);
    }
    // Compute $P cost measure.
    gesturePts = summaryShape.alignGesture(gesture, PtAlignType.CLOUD_MATCH);
    var pDollarCost = 0;
    for (var i = 0; i < gesturePts.length; i++) {
      pDollarCost += Measure.distance(gesturePts[i], summaryPts[i]);
    }
    return Math.abs(oDollarCost - pDollarCost);
  },

  // utility functions: mean, stdev

  /**
   * Compute the mean of a set of real values.
   * @static
   * @param {Array} arr - Input set.
   * @return {Number} Average value of the input set.
   * @throws {Error} Input set cannot be empty.
   */
  mean: function(arr) {
    if (arr.length == 0) throw new Error("Input set cannot be empty.");
    else if (arr.length == 1) return arr[0];

    var sum = 0;
    for (var i = 0; i < arr.length; i++) {
      sum += arr[i];
    }
    return sum/arr.length;
  },
  /**
   * Compute the unbiased standard deviation for a set of real values.
   * @static
   * @param {Array} arr - Input set.
   * @return {Number} Standard deviation of the input set.
   * @throws {Error} Input set cannot be empty.
   */
  stdev: function(arr) {
    if (arr.length == 0) throw new Error("Input set cannot be empty.");
    else if (arr.length == 1) return 0;

    var mean  = this.mean(arr);
    var stdev = 0;
    for (var i = 0; i < arr.length; i++) {
      var item = arr[i];
      stdev += (item - mean) * (item - mean);
    }
    return Math.sqrt( stdev/(arr.length - 1) );
  }

};
