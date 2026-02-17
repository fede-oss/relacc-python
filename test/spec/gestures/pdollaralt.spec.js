describe("PDollarAlt", function() {
  var PDollarAlt = require('../../../lib/gestures/pdollaralt');
  var Point = require('../../../lib/geom/point');

  function pointsA() {
    return [
      new Point(0, 0, 0, 0),
      new Point(1, 0, 1, 0),
      new Point(2, 0, 2, 0)
    ];
  }

  function pointsB() {
    return [
      new Point(0, 0, 0, 0),
      new Point(2, 0, 1, 0),
      new Point(1, 0, 2, 0)
    ];
  }

  it("computes alignment weights", function() {
    var weights = PDollarAlt.weights(pointsA(), pointsB());
    expect(weights.length).toBe(3);
    expect(weights[0].length).toBe(3);
    expect(weights[0][0]).toBe(0);
    expect(weights[0][1]).toBeLessThan(0);
  });

  it("handles uneven point-set sizes", function() {
    var weights = PDollarAlt.weights([new Point(0, 0, 0, 0)], pointsB());
    expect(weights.length).toBe(1);
    expect(weights[0].length).toBe(1);
  });

  it("matches clouds with the Hungarian algorithm", function() {
    var matching = PDollarAlt.match(pointsA(), pointsB());
    var sorted = matching.slice().sort(function(a, b) { return a - b; });

    expect(matching.length).toBe(3);
    expect(sorted).toEqual([0, 1, 2]);
  });

  it("exposes cloud-distance and greedy internals", function() {
    var arr = [];
    var distance = PDollarAlt._cloudDistance(pointsA(), pointsB(), 0, arr);
    var greedy = PDollarAlt._greedyCloudMatch(pointsA(), pointsB());

    expect(distance).toBeGreaterThanOrEqual(0);
    expect(arr.length).toBe(3);
    expect(greedy.length).toBe(3);
  });

  it("exposes Hungarian internals and computes matching cost", function() {
    var weights = [
      [5, 1, 0],
      [0, 6, 1],
      [1, 0, 7]
    ];
    var matching = PDollarAlt._hungarianMatch(weights);
    expect(matching).toEqual([0, 1, 2]);

    var cost = PDollarAlt.cost([0, 1], [[-1, -5], [-3, -2]]);
    expect(cost).toBe(3);
  });
});
