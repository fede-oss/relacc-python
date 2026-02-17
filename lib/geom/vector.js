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

var Point = require("./point");

/**
 * @module lib/geom/vector
 */

/**
 * Create a vector.
 * @constructor
 * @param {Point} a - Vector starting point.
 * @param {Point} b - Vector ending point.
 */
var Vector = function(a, b) {

  // The vector points from a to b.
  this.vec = b.subtract(a);

};

/**
 * Compute vector length.
 * @return {Number}
 */
Vector.prototype.length = function() {
  var orig = new Point();
  // Note, we require the Measure module here to avoid circular referencing,
  // since the Measure module depends on Vector, too.
  return require("./measure").distance(this.vec, orig);
};

module.exports = Vector;

/**
 * Compute the inner product between 2 vectors.
 * @static
 * @param v {Vector} Vector 1.
 * @param u {Vector} Vector 2.
 * @return {Number}
 */
module.exports.dotProduct = function(v, u) {
  return v.vec.X * u.vec.X + v.vec.Y * u.vec.Y;
};
