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
 * Create a rectangle.
 * @constructor
 * @param {Point} topLeft - A point for the top left corner.
 * @param {Point} bottomRight - A point for the bottom right  corner.
 */
var Rectangle = function(topLeft, bottomRight) {
  // Don't lose scope.
  var self = this;
  /**
   * Top left corner point.
   * @memberof Rectangle
   */
  self.topLeft = new Point(topLeft);
  /**
   * Bottom right corner point.
   * @memberof Rectangle
   */
  self.bottomRight = new Point(bottomRight);
  /**
   * Compute rectangle width.
   * @memberof Rectangle
   * @return {Number}
   */
  self.width = function() {
    return Math.abs(this.bottomRight.X - this.topLeft.X);
  };
  /**
   * Compute rectangle height.
   * @memberof Rectangle
   * @return {Number}
   */
  self.height = function() {
    return Math.abs(this.bottomRight.Y - this.topLeft.Y);
  };
  /**
   * Compute rectangle area.
   * @memberof Rectangle
   * @return {Number}
   */
  self.area = function() {
    return this.width() * this.height();
  };

};

/**
 * @module lib/geom/rectangle
 */
module.exports = Rectangle;
