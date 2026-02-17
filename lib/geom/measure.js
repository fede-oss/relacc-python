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

var Vector = require("./vector");
/**
 * Computes relative accuracy measures for stroke gestures based on the notion
 * of a "gesture task axis" (aka centroid gesture).
 * @module lib/geom/measure
 */
module.exports = {

    /**
     * Compute the squared Euclidean Distance between two points.
     * @static
     * @param a {Point} Point 1
     * @param b {Point} Point 2
     * @return {Number}
     */
    sqDistance: function(a, b) {
      return (a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y);
    },
    /**
     * Compute the Euclidean distance between two points.
     * @static
     * @param a {Point} Point 1
     * @param b {Point} Point 2
     * @return {Number}
     */
    distance: function(a, b) {
      return Math.sqrt( this.sqDistance(a, b) );
    },
    /**
     * Compute the Manhatan Distance between two points.
     * @static
     * @param a {Point} Point 1
     * @param b {Point} Point 2
     * @return {Number}
     */
    taxicab: function(a, b) {
      return Math.abs(a.X - b.X) + Math.abs(a.Y - b.Y);
    },
    /**
     * Compute the smallest turning angle between vectors v and u in radians, in [0..PI].
     * @static
     * @param v {Vector} Vector 1
     * @param u {Vector} Vector 2
     * @return {Number}
     */
    shortAngle: function(v, u) {
      // compute cosine between vectors
      var vLength = v.length();
      var uLength = u.length();
      if (Math.abs(vLength * uLength) <= Number.MIN_VALUE)
        return 0;
      var cosAngle = Vector.dotProduct(v, u) / (vLength * uLength);
      // deal with special cases
      if (cosAngle <= -1) return Math.PI;
	    if (cosAngle >= 1) return 0;
      // return angle in the interval [0, PI]
      return Math.acos(cosAngle);
    },
    /**
     * Compute the trigonometric turning angle between vectors v and u in radians, in [0..2PI].
     * @static
     * @param v {Vector} Vector 1
     * @param u {Vector} Vector 2
     * @return {Number}
     */
    angle: function(v, u) {
      var angle = this.shortAngle(v, u);
      if (!this.trigonometricOrder(v, u)) {
        angle = 2 * Math.PI - angle;
      }
	    return angle;
    },
    /**
     * Check if vectors v and u are in trigonometric order. Returns false otherwise.
     * @static
     * @param v {Vector} Vector 1
     * @param u {Vector} Vector 2
     * @return {Boolean}
     */
    trigonometricOrder: function(v, u) {
      return Vector.dotProduct(v,u) >= 0;
    }

};
