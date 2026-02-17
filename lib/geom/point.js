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

/**
 * Create a point with (X, Y) coordinates, timestamp (T), and stroke identifier (StrokeId).
 * @constructor
 */
var Point = function(x, y, t, sid) {

  var self = this;

	self.X = 0;
	self.Y = 0;
  self.T = 0;
  self.StrokeID = 0;

  // Private instance constructor.
  function loadMemberData(x, y, t, sid) {
	  self.X = x;
	  self.Y = y;
	  self.T = t;
    self.StrokeID = sid;
  };

  if (x === undefined) { // No args
    loadMemberData(0, 0, 0, 0);
  } else if (typeof x === 'object') { // Single arg given
    // Creates a point identical to the argument point.
    loadMemberData(x.X, x.Y, x.T, x.StrokeID);
  } else if (arguments.length == 2) { // Only x,y are given
    loadMemberData(x,y,0,0);
  } else { // Default constructor.
    loadMemberData(x, y, t, sid);
  }

};

/**
 * Compute the maximum value of the X and Y coordinates of this point.
 * @return {Number}
 */
Point.prototype.maxXY = function() {
  return Math.max(this.X, this.Y);
};
/**
 * Compute the minimum value of the X and Y coordinates of this point.
 * @return {Number}
 */
Point.prototype.minXY = function() {
  return Math.min(this.X, this.Y);
};
/**
 * Perform p1 + p2 on each coordinate. Copies the timestamp T and StrokeID of p1.
 * @param {Point} p2
 * @return {Point}
 */
Point.prototype.add = function(p2) {
  var p1 = this;
  return new Point(p1.X + p2.X, p1.Y + p2.Y, p1.T, p1.StrokeID);
};
/**
 * Perform p1 - p2 on each coordinate. Copies the timestamp T and StrokeID of p1.
 * @param {Point} p2
 * @return {Point}
 */
Point.prototype.subtract = function(p2) {
  var p1 = this;
  return new Point(p1.X - p2.X, p1.Y - p2.Y, p1.T, p1.StrokeID);
};
/**
 * Perform p / scalar on each coordinate.
 * @param {Number} scalar
 * @return {Point}
 * @throws {Error} Cannot divide by zero.
 */
Point.prototype.divideBy = function(scalar) {
  if (Math.abs(scalar) < 10e-6) throw new Error("Cannot divide by zero.");
  return new Point(this.X/scalar, this.Y/scalar, this.T, this.StrokeID);
};
/**
 * Perform p * scalar on each coordinate.
 * @param {Number} scalar
 * @return {Point}
 */
Point.prototype.multiplyBy = function(scalar) {
  return new Point(scalar*this.X, scalar*this.Y, this.T, this.StrokeID);
};

/**
 * @module lib/geom/point
 */
module.exports = Point;

/**
 * Compute the point with minimum coordinates on X and Y axes.
 * @static
 * @return {Point}
 */
module.exports.absMin = function() {
  return new Point(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, 0, 0);
};
/**
 * Compute the point with maximum coordinates on X and Y axes.
 * @static
 * @return {Point}
 */
module.exports.absMax = function() {
  return new Point(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0, 0);
};
