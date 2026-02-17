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
 * Math functions.
 * @module lib/utils/math
 */
module.exports = {

  /**
   * Round a number to given precision.
   * @static
   * @param {Number} num - Input number.
   * @param {Number} prec - Decimal precision. Default: 3 decimal places.
   * @return {Number} Rounded input number.
   * @example var mat = require("./lib/utils/math");
   * console.log(mat.roundTo(Math.PI)); // output: 3.142
   */
  roundTo: function(num, prec) {
    if (prec === undefined) prec = 3;
    return Number(num.toFixed(prec));
  },
  /**
   * Compute the factorial of a number (5! = 5*4*3*2*1 = 120).
   * @static
   * @param {Number} num - Input number.
   * @return {Number} Factorial of the input number.
   * @example var mat = require("./lib/utils/math");
   * console.log(mat.factorial(5)); // output: 120
   */
  factorial: function(num) {
    var val = 1;
    for (var n = 2; n <= num; n++) val *= n;
    return val;
  }

};
