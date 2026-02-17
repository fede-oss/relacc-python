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
 * Process CSV gesture files.
 * @module lib/utils/date
 */
module.exports = {

  /**
   * Get current date in UTC format.
   * @static
   * @return {String} Date in UTC format.
   * @example var dat = require("./lib/utils/date");
   * console.log(dat.utc()); // output: "Fri, 18 Apr 2015 21:36:11 GMT"
   */
  utc: function() {
    return new Date().toUTCString();
  },
  /**
   * Get current timestamp.
   * @static
   * @return {Number} Timestamp.
   * @example var dat = require("./lib/utils/date");
   * console.log(dat.now()); // output: 1442612203756
   */
  now: function() {
    return new Date().getTime();
  }

};
