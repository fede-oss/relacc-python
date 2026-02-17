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

var util      = require("util");
var Point     = require("./point");
var Rectangle = require("./rectangle");
var Measure   = require("./measure");

/**
 * Point set geometry utilities.
 * @module lib/geom/pointset
 */
module.exports = {

  /**
   * Create a copy of a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Array}
   */
  clone: function(points) {
    var pt = new Point();
    if (!points) return pt;

    var newPoints = [];
    for (var i = 0; i < points.length; i++) {
      newPoints.push( new Point(points[i]) );
    }
    return newPoints;
  },
  /**
   * Compute the center of gravity of a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Point}
   */
  centroid: function(points) {
    var pt = new Point();
    if (!points) return pt;

    for (var i = 0; i < points.length; i++) {
      pt = pt.add(points[i]);
    }
    return pt.divideBy(points.length);
  },
  /**
   * Compute the point that has minimum values on the X and Y axes in a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Point}
   */
  minPt: function(points) {
    var pt = new Point();
    if (!points) return pt;

    var min = Point.absMax();
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (min.X > p.X) min.X = p.X;
      if (min.Y > p.Y) min.Y = p.Y;
    }
    return min;
  },
  /**
   * Compute the point that has maximum values on the X and Y axes in a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Point}
   */
  maxPt: function(points) {
    var pt = new Point();
    if (!points) return pt;

    var max = Point.absMin();
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (max.X < p.X) max.X = p.X;
      if (max.Y < p.Y) max.Y = p.Y;
    }
    return max;
  },
  /**
   * Compute the bounding box of a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Rectangle}
   */
  boundingBox: function(points) {
    return new Rectangle( this.minPt(points), this.maxPt(points) );
  },
  /**
   * Compute the path length of a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Number} startIndex - Array index to start search from.
   * @param {Number} endIndex - Array index to end search from.
   * @return {Number}
   */
  pathLength: function(points, startIndex, endIndex) {
    if (!points) return 0;
    if (typeof startIndex === 'undefined') startIndex = 0;
    if (typeof endIndex === 'undefined') endIndex = points.length - 1;

    if (startIndex >= endIndex) return 0;

    var len = 0;
    for (var i = startIndex + 1; i <= endIndex; i++) {
//      if (points[i].StrokeID == points[i - 1].StrokeID) { // Unnecessary check?
        len += Measure.distance(points[i], points[i - 1]);
//      }
    }
    return len;
  },
  /**
   * Scale a set of points with shape preservation.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Array}
   */
  scale: function(points) {
    var min = this.minPt(points);
    var max = this.maxPt(points);
    var scaleFactor = 1/max.subtract(min).maxXY();
    var newPoints = [];
    if (isFinite(scaleFactor)) {
      for (var i = 0; i < points.length; i++) {
        newPoints.push( points[i].subtract(min).multiplyBy(scaleFactor) );
      }
    } else {
      for (var i = 0; i < points.length; i++) {
        newPoints.push( new Point(points[i]) );
      }
    }
    return newPoints;
  },
  /**
   * Scale a set of points to a given scale factor.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Number} scaleFactor - Scale factor.
   * @return {Array}
   */
  scaleTo: function(points, scaleFactor) {
    var min = this.minPt(points);
    var newPoints = [];
    for (var i = 0; i < points.length; i++) {
      newPoints.push( points[i].multiplyBy(scaleFactor) );
    }
    return newPoints;
  },
  /**
   * Translate a set of points by a Point offset.
   * If offset coincides with the center of gravity, the new center will be (0, 0).
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Point} offset - Reference Point.
   * @return {Array}
   */
  translateBy: function(points, offset) {
    var newPoints = [];
    for (var i = 0; i < points.length; i++) {
      newPoints.push( points[i].subtract(offset) );
    }
    return newPoints;
  },
  /**
   * Resample a set of points uniformly into a custom number of points.
   * If offset coincides with the center of gravity, the new center will be (0, 0).
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Point} n - Number of resampled points.
   * @return {Array}
   */
  unifResampling: function(points, n) {
    var newPoints = [];
    var pathLen = this.pathLength(points);
    if (pathLen == 0) {
      // All points are the same, so exit early.
      for (var i = 1; i < n; i++) newPoints.push( new Point(points[0]) );
      return newPoints;
    }
    var I = pathLen / (n - 1);
    var D = 0.0;
    var prevPoint, currPoint;
    newPoints = new Array(points[0]);
    for (var i = 1; i < points.length; i++) {
	    prevPoint = points[i - 1];
	    currPoint = points[i];
	    if (currPoint.StrokeID == prevPoint.StrokeID) {
		    var d = Measure.distance(prevPoint, currPoint);
		    if (d > 0 && (D + d) >= I) {
		      var s = (I - D) / d;
			    var qx = prevPoint.X + s * (currPoint.X - prevPoint.X);
			    var qy = prevPoint.Y + s * (currPoint.Y - prevPoint.Y);
			    var qt = prevPoint.T + s * (currPoint.T - prevPoint.T);
			    var q = new Point(qx, qy, qt, currPoint.StrokeID);
			    newPoints.push(q);
			    points.splice(i, 0, q); // insert 'q' at position i in points s.t. 'q' will be the next i
			    D = 0.0;
		    }
		    else D += d;
	    }
    }
//    // Sometimes we fall a rounding-error short of adding the last point.
//    if (newPoints.length == n - 1) {
//      newPoints.push( new Point(points[points.length - 1]) );
//    }
    if (newPoints.length < n) {
      for (i = newPoints.length; i < n; i++) {
        newPoints.push( new Point(points[points.length - 1]) );
      }
    }
    return newPoints;
  },
  /**
   * Compute the number of strokes in a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Number}
   */
  countStrokes: function(points) {
    var arr = [];
    for (var i = 1; i < points.length - 1; i++) {
      var prevPoint = points[i - 1];
      var nextPoint = points[i];
      if (nextPoint.StrokeID != prevPoint.StrokeID) {
        arr.push(prevPoint.StrokeID);
      }
    }
    arr = arr.filter(function(elem, i) {
      return arr.lastIndexOf(elem) === i;
    });
    return arr.length;
  },
  /**
   * Compute the cummulated distance along a sequence of vectors.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Number} startIndex - Array index to start search from.
   * @return {Array}
   */
  cumDistances: function(points, startIndex) {
    if (typeof startIndex === 'undefined') startIndex = 0;
    var cum = 0
      , lst = new Array(points.length + 1).join(0).split('').map(parseFloat);
      ;
    for (var i = startIndex + 1; i < points.length; i++) {
      var prevPoint = points[i - 1];
      var currPoint = points[i];
      var dist = Measure.distance(prevPoint, currPoint);
      cum += dist;
      lst[i] = cum;
    }
    return lst;
  },
  /**
   * Compute the index of a query distance over a list of cummulated distaces.
   * @static
   * @param {cumDistList} cumDistList - Cummulated distaces.
   * @param {Number} queryDist - Distance to search for in cumDistList.
   * @param {Number} startIndex - Array index to start search from.
   * @return {Array}
   */
  indexOfDistance: function(cumDistList, queryDist, startIndex) {
    if (typeof startIndex === 'undefined') startIndex = 0;
    var index = -1;
    for (var i = startIndex; i < cumDistList.length; i++) {
      if (cumDistList[i] > queryDist) {
        index = i - 1;
        break;
      }
    }
    return index;
  },
  /**
   * Perform Martin-Albo resampling over a set of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Array} n - Number of resampled points.
   * @return {Array}
   */
  maResampling: function(points, n) {
    // Add first point.
    var newPoints = [ new Point(points[0]) ];
    var pathLen = this.pathLength(points);
    if (pathLen == 0) {
      // All points are the same, so exit early.
      for (var i = 1; i < n; i++) newPoints.push( new Point(points[0]) );
      return newPoints;
    }
    var intervalLen = pathLen / (n - 1);
    var cumDistList = this.cumDistances(points);
    var seenPtIndex, currPoint, prevPoint, nextPoint, morePoint;
    var lastSeenIndex = 0;
    for (var i = 1; i < n - 1; i++) {
      seenPtIndex = this.indexOfDistance(cumDistList, intervalLen, lastSeenIndex);
      // Fix rounding precission issue.
      if ( seenPtIndex === -1 ) seenPtIndex = Math.max(0, points.length - 2);
      currPoint = points[lastSeenIndex];
      nextPoint = points[seenPtIndex];
      morePoint = points[seenPtIndex + 1];
      var distCurrNext = intervalLen - this.pathLength(points, lastSeenIndex, seenPtIndex);
      var distNextMore = this.pathLength(points, seenPtIndex, seenPtIndex + 1);
      var t = distCurrNext/distNextMore;
      var q = new Point(
        (1 - t) * nextPoint.X + t * morePoint.X,
        (1 - t) * nextPoint.Y + t * morePoint.Y,
        (1 - t) * nextPoint.T + t * morePoint.T,
        currPoint.StrokeID
      );
      newPoints.push(q);
      lastSeenIndex = seenPtIndex + 1;
	    points.splice(lastSeenIndex, 0, q); // insert 'q' at position i in points s.t. 'q' will be the next i
	    cumDistList = this.cumDistances(points, lastSeenIndex);
    }
    // Add last point.
    newPoints.push( new Point(points[points.length - 1]) );
    return newPoints;
  },
  /**
   * Ensure that each gesture stroke get the same number of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @return {Array}
   */
  eqDistStrokes: function(points, strkNum) {
    if (!points || points.length === 0) return [];

    var strokes = new Array(strkNum || 0);
    var currPoint, nextPoint, c = 0;
    if (points.length === 1) {
      strokes[c] = [ points[0] ];
      return strokes;
    }
    // Ensure that each stroke gets the same number or points.
    strokes[c] = [];
    for (var i = 0; i < points.length - 1; i++) {
      currPoint = points[i];
      nextPoint = points[i + 1];
      strokes[c].push(currPoint);
      if (currPoint.StrokeID != nextPoint.StrokeID) {
        c++;
        strokes[c] = [];
      }
    }
    // Store last point.
    strokes[c].push(nextPoint);
    return strokes;
  },
  /**
   * Perform equidistant resampling.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Array} n - Number of resampled points.
   * @return {Array}
   */
  eqResample: function(points, n) {
    // Assign the same number of points per stroke.
    var strkNum = this.countStrokes(points);
    var strokes = this.eqDistStrokes(points, strkNum);
    // Now interpolate points in each stroke.
    var pointsPerStroke = Math.round(n/strkNum);
    var newPoints = [];
    for (var j = 0; j < strokes.length; j++) {
      var resampled = this.unifResampling(strokes[j], pointsPerStroke);
      newPoints = newPoints.concat(resampled);
    }
    this.ensureResampling(newPoints, n);
    return newPoints;
  },
  /**
   * Ensure that resampling returns the desired number of points.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Array} n - Number of resampled points.
   * @return {Array}
   * @throws {Error} Resampling error: ...
   */
  ensureResampling: function(newPoints, n) {
    if (newPoints.length != n) {
      console.trace( util.format("Resampling error: %d points requested, but got %d.", n, newPoints.length) );
      process.exit(1);
    }
  },
  /**
   * Generic resampling function.
   * @static
   * @param {Array} points - Input point set; e.g. [ new Point(), ... [vn1,...,vnN] ]
   * @param {Array} n - Number of resampled points.
   * @return {Array}
   */
  resample: function(points, n) {
    var newPoints = this.unifResampling(points, n);
    this.ensureResampling(newPoints, n);
    return newPoints;
  }

};
