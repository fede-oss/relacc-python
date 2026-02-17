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
 * Point alignment type enum.
 * @module lib/geom/ptaligntype
 * @readonly
 */
var AlignEnum = {
  /**
   * Chronological order of input points.
   * @name CHRONOLOGICAL
   */
  CHRONOLOGICAL:  1,
  /**
   * Cloud match order. Ignores stroke number and point order.
   * @name CLOUD_MATCH
   */
  CLOUD_MATCH:    2
};

if (Object.freeze) Object.freeze(AlignEnum);

module.exports = AlignEnum;
