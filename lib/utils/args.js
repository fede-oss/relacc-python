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
 * Process CLI arguments.
 * @module lib/utils/args
 */

/**
 * @constructor
 * @name Args
 * @param {Array} args - process.argv
 */
module.exports = function(args) {

  /**
   * Get a given CLI argument.
   * @param {String} name - Argument name.
   * @param {Mixed} defaultValue - Default argument value (if argument not provided).
   * @param {Function} castFn - Casting function to apply to argument value.
   * @return {Mixed} Argument value.
   * @example Usage example:
   * > nodejs args.js -n 1.5
   * // Contents of args.js file:
   * var ArgUtil = require("./lib/utils/args");
   * var argParser = new ArgUtil(process.argv);
   * var n = argParser.get('n'); // n = "1.5"
   * // With typecasting:
   * n = argParser.get('n', Number); // n = 1.5
   * n = argParser.get('n', parseInt); // n = 1
   * // With default value for undefined arguments:
   * var m = argParser.get('m', 42); // m = 42
   */
  this.get = function(name, defaultValue, castFn) {
    var value;
    var hasArg = Object.prototype.hasOwnProperty.call(args, name) &&
      typeof args[name] !== 'undefined';
    if (!hasArg && typeof defaultValue !== 'undefined') {
      value = defaultValue;
    } else value = args[name];
    // Check type casting.
    var thisArgs = Array.prototype.slice.call(arguments);
    if (typeof castFn === 'function') return castFn(value);
    else if (thisArgs.length === 2) {
      var reviver = thisArgs.pop();
      if (typeof reviver === 'function') return reviver(value);
    }
    return value;
  }

};
