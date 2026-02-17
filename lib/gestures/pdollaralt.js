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

var Measure = require("../geom/measure");

function greedyCloudMatch(points1, points2) {
	var bestAlignment = [];
  var e = 0.5, step = Math.floor(Math.pow(points1.length, 1 - e));
	for (var i = 0; i < points1.length; i += step) {
    var alignment1 = [], alignment2 = [];
	  var d1 = cloudDistance(points1, points2, i, alignment1);
	  var d2 = cloudDistance(points2, points1, i, alignment2);
	  bestAlignment = d1 < d2 ? alignment1.slice(0) : alignment2.slice(0);
	}
  return bestAlignment;
};

function cloudDistance(pts1, pts2, start, arr) {
	var matched = []; // pts1.length == pts2.length
	for (var k = 0; k < pts1.length; k++) matched.push(false);
	var sum = 0, i = start;
	do {
		var index = -1, min = +Infinity;
		for (var j = 0; j < matched.length; j++) {
			if (!matched[j]) {
				var d = Measure.distance(pts1[i], pts2[j]);
				if (d < min) {
					min = d;
					index = j;
				}
			}
		}
		arr.push(index);
		matched[index] = true;
		var weight = 1 - ((i - start + pts1.length) % pts1.length) / pts1.length;
		sum += weight * min;
		i = (i + 1) % pts1.length;
	} while (i != start);
	return sum;
};

// Hungarian algorithm for finding the maximum weighted matching for bipartite graphs.
function hungarianMatch(weights) {
  var NOT_FOUND   = -1;
  var NOT_MATCHED = -1;

  // Number of vertices for the two sets of a bipartite graph (the left and right sets).
  var i, j, n = weights[0].length;
  
  // Initialize vertex labeling for the left and right sets.
  var labelsLeft  = new Array(n);
  var labelsRight = new Array(n);
  for (i = 0; i < n; i++) {
    labelsRight[i] = 0;
    labelsLeft[i]  = weights[i][0];
    for (j = 1; j < n; j++) {
      labelsLeft[i] = Math.max(labelsLeft[i], weights[i][j]);
    }
  }

  // Initalize matching
  var matchingCount = 0;  // size of the current matching (stop when reach n)
  var matchingLeft  = [];
  var matchingRight = [];
  for (i = 0; i < n; i++) {
    matchingLeft.push(NOT_MATCHED);  // the vertex from the right set to which vertex i from the left set is matched to
    matchingRight.push(NOT_MATCHED); // the vertex from the left set to which vertex i from the right set is matched to
  }

  var visitedLeft;  // visitedLeft[i] = true if vertex i from the left set has been processed
  var visitedRight; // visitedRight[i] = true if vertex i from the right set has been processed
  var parent = new Array(n); // stores the edges of the Hungarian tree, parent[i] = the vertex from the left set that led to the discovery of vertex i from the right set
  while (matchingCount < n) {
    // Pick free vertex from the left set as the root of the Hungarian tree.
    var u = 0;
    while (matchingLeft[u] != NOT_MATCHED) u++;
    // Update the labels until a non-matched vertex y from the right set is found,
    // which will increase the cardinality of the matching.
    var y = NOT_FOUND;
    while (y == NOT_FOUND) {
      // Reset processed vertices. We use n+1 due to the join/split hack, but actually both arrays have size n.
      visitedLeft  = new Array(n+1).join(0).split('').map(parseFloat);
      visitedRight = new Array(n+1).join(0).split('').map(parseFloat);
      // Rraverse the equality graph using the breadth-first approach.
      // Traversal stops when :
      // 1) no more vertices are available for exploring or 
      // 2) we found a free vertex in the right set (denoted by y)
      var queue = [u];
      visitedLeft[u] = true;
      while (queue.length > 0 && y == NOT_FOUND) {
        var vertex = queue.shift();
        for (j = 0; j < n; j++) {
          if (!visitedRight[j]) {
            var diff = weights[vertex][j] - (labelsLeft[vertex] + labelsRight[j]);
            if (diff < 0) diff = -diff;
            if (diff < 10e-4) {
              parent[j] = vertex;
              visitedRight[j] = true;
              // Check whether j from the right set is already matched to a vertex (z) from the left set.
              var z = matchingRight[j];
              if (z == NOT_MATCHED) {
                // We found a non-matched vertex, stop traversal.
                y = j;
                break;
              } else {
                // Add z to the BFS queue and continue exploration.
                queue.push(z);
                visitedLeft[z] = true;
              }
            }
          }
        }
      }

      if (y == NOT_FOUND) {
        // Update vertex lables in order to enlarge the equality graph.
        var alpha = Number.MAX_VALUE;
        for (i = 0; i < n; i++) {
          if (visitedLeft[i]) {
            for (j = 0; j < n; j++) {
              if ( !visitedRight[j] ) {
                var diff = labelsLeft[i] + labelsRight[j] - weights[i][j];
                if (alpha > diff) alpha = diff;
              }
            }
          }
        }
        for (i = 0; i < n; i++) {
          if (visitedLeft[i])  labelsLeft[i]  -= alpha;
          if (visitedRight[i]) labelsRight[i] += alpha;
        }
      } else {
        // The path from root u (left set) to y (right set) is an augmenting path
        // and therefore we can increase the cardinality of the current matching by reversing the edges of the path.
        var index = y;
        while (index != NOT_MATCHED) {
          var t = matchingLeft[parent[index]];
          matchingLeft[parent[index]] = index;
          matchingRight[index] = parent[index];
          index = t;
        }
        matchingCount++;
      }
    }
  }
  // At this point we have the optimal alignment of the two sets of vertices.
  return matchingLeft;
};

/**
 * An alternative version of the $P recognizer.
 * @module lib/geom/pdollaralt
 */
module.exports = {

  /**
   * Compute pptimal alignment of two clouds of points using the Hungarian weighted matching algorithm for bipartite graphs.
   * @static
   * @param {Array} points1 - First point set.
   * @param {Array} points2 - Second point set.
   * @return {Array}
   */
  match: function(points1, points2) {
    var m = hungarianMatch(this.weights(points1, points2));
//		var m = greedyCloudMatch(points1, points2);
    return m;
  },
  /**
   * Compute alignment weights between 2 point sets, based on local distances.
   * @static
   * @param {Array} points1 - First point set.
   * @param {Array} points2 - Second point set.
   * @return {Array}
   */
  weights: function(points1, points2) {
    if (points1.length != points2.length) {
      // throw Error?
    }
    var n = points1.length;
    var weights = new Array(n);
    for (var i = 0; i < n; i++) {
      weights[i] = [];
      for (var j = 0; j < n; j++) {
        weights[i].push( -Measure.sqDistance(points1[i], points2[j]) );
      }
    }
    return weights;
  },
  // @deprecated Not used.
  cost: function(matching, weights) {
    var cost = 0;
    for (var i = 0, n = matching.length; i < n; i++) {
      cost += -weights[i][matching[i]];
    }
    return cost;
  }
    
};

// Expose internals for deterministic testing.
module.exports._greedyCloudMatch = greedyCloudMatch;
module.exports._cloudDistance = cloudDistance;
module.exports._hungarianMatch = hungarianMatch;
