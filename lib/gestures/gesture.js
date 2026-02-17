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

var Point    = require("../geom/point");
var PointSet = require("../geom/pointset");

/**
 * Create a gesture from a set of points.
 * @constructor
 * @param {Array} points - Input gesture points.
 * @param {String} name - Gesture label.
 * @param {Number} samplingRate - Number of resampled points.
 * @throws {Error} Gesture points cannot be empty.
 * @throws {Error} Gesture name cannot be empty.
 */
var Gesture = function(points, name, samplingRate) {

  var self = this;

  /** Original gesture points. */
  self.originalPoints = null;
  /** Resampled points. */
  self.points = null;
  /** Gesture label. */
  self.name = name;
  /**
   * Preprocess gesture points.
   * @param {Number} rate - Sampling rate.
   */
  self.preprocess = function(rate) {
    this.samplingRate = rate;
    this.points = PointSet.resample(PointSet.clone(this.originalPoints), rate);
    // do not scale points when computing relative accuracy measures
    //this.points = PointSet.scale(this.points);
    this.points = PointSet.translateBy(this.points, PointSet.centroid(this.points));
  };
  /** Sampling rate. Default: 32 points. */
  self.samplingRate = samplingRate || 32;

  // Constructor begins here.
  if (!name) throw new Error("Gesture name cannot be empty.");
  if (!points || points.length == 0) {
    throw new Error("Gesture points cannot be empty.");
  } else {
    self.originalPoints = [];
    for (var i = 0; i < points.length; i++) {
      self.originalPoints.push( new Point(points[i]) );
    }
    self.preprocess(self.samplingRate);
  }

};

/**
 * Gesture module.
 * @module lib/gestures/gesture
 */
module.exports = Gesture;
