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

var util = require('util');

/**
 * Debug utilities to display messages on stderr.
 * @module lib/utils/debug
 */

/**
 * @constructor
 * @name Debug
 * @param {Object} opts - Configuration options.
 * @param {Boolean} opts.verbose - Verbose output. Default: false. If false, nothing is displayed.
 * @example var Debug = require('debug');
 * var dbg = new Debug({ verbose: true });
 * dbg.log("Hello world"); // output: "Hello world"
 * dbg.fmt('%d is a number', 42);  // output: "42 is a number"
 * dbg = new Debug(); // verbose is false by default.
 * dbg.log("Hello world"); // no output
 */
module.exports = function(opts) {

  var defaults = opts || {
    verbose: false
  };
  /**
   * Display a message.
   * @param {String} msg - Message to display.
   * @return {String} The message.
   */
  this.log = function(msg) {
    if (defaults.verbose) process.stderr.write(msg + "\n");
  };
  /**
   * Display a message in printf-like format.
   * @param {String} format - Formatted string. Accepted tokens:
   * %s - String
   * %d - Number
   * %j - JSON
   * @param {String} msg - Message to display.
   * @return {String} The formatted message.
   */
  this.fmt = function(format, msg) {
    this.log(util.format(format, msg));
  };

};
