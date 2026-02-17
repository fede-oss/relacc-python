describe("PointSet extras", function() {
  var Point = require('../../../lib/geom/point');
  var PointSet = require('../../../lib/geom/pointset');

  var points;

  beforeEach(function() {
    points = [
      new Point(0, 0, 0, 0),
      new Point(10, 0, 10, 0),
      new Point(10, 10, 20, 1),
      new Point(20, 10, 30, 1)
    ];
  });

  it("returns zero-like points for null-safe helpers", function() {
    expect(PointSet.clone(null)).toEqual(new Point());
    expect(PointSet.centroid(null)).toEqual(new Point());
    expect(PointSet.minPt(null)).toEqual(new Point());
    expect(PointSet.maxPt(null)).toEqual(new Point());
  });

  it("returns zero for invalid path-length ranges", function() {
    expect(PointSet.pathLength(null)).toBe(0);
    expect(PointSet.pathLength(points, 2, 1)).toBe(0);
  });

  it("copies points when scaling is not finite", function() {
    var same = [
      new Point(1, 1, 0, 0),
      new Point(1, 1, 1, 0)
    ];

    var scaled = PointSet.scale(same);
    expect(scaled).toEqual(same);
    expect(scaled[0]).not.toBe(same[0]);
  });

  it("scales and translates points with explicit helpers", function() {
    var scaled = PointSet.scaleTo(points, 0.5);
    var moved = PointSet.translateBy(points, new Point(1, 1, 0, 0));

    expect(scaled[0].X).toBe(0);
    expect(scaled[1].X).toBe(5);
    expect(moved[0]).toEqual(new Point(-1, -1, 0, 0));
  });

  it("handles zero-length uniform resampling", function() {
    var same = [
      new Point(1, 1, 0, 0),
      new Point(1, 1, 0, 0)
    ];

    var resampled = PointSet.unifResampling(same, 4);
    expect(resampled.length).toBe(3);
    expect(resampled[0]).toEqual(new Point(1, 1, 0, 0));
  });

  it("handles non-zero uniform resampling across stroke boundaries", function() {
    var resampled = PointSet.unifResampling(points.slice(), 5);
    expect(resampled.length).toBe(5);
  });

  it("counts unique stroke transitions", function() {
    var multi = [
      new Point(0, 0, 0, 0),
      new Point(1, 0, 1, 1),
      new Point(2, 0, 2, 0),
      new Point(3, 0, 3, 1),
      new Point(4, 0, 4, 1)
    ];

    expect(PointSet.countStrokes(multi)).toBe(2);
  });

  it("computes cumulative distances and index lookups", function() {
    var cum = PointSet.cumDistances(points);
    expect(cum.length).toBe(points.length);
    expect(cum[0]).toBe(0);

    expect(PointSet.indexOfDistance(cum, 5)).toBe(0);
    expect(PointSet.indexOfDistance(cum, 999)).toBe(-1);
  });

  it("resamples with Martin-Albo algorithm", function() {
    var resampled = PointSet.maResampling(points.slice(), 3);
    expect(resampled.length).toBe(3);
  });

  it("handles zero-length Martin-Albo resampling", function() {
    var same = [
      new Point(2, 2, 0, 0),
      new Point(2, 2, 0, 0)
    ];
    var resampled = PointSet.maResampling(same, 4);
    expect(resampled.length).toBe(4);
  });

  it("handles Martin-Albo fallback when distance index is missing", function() {
    var original = PointSet.indexOfDistance;
    PointSet.indexOfDistance = function() {
      return -1;
    };

    var resampled = PointSet.maResampling(points.slice(), 3);
    expect(resampled.length).toBe(3);

    PointSet.indexOfDistance = original;
  });

  it("splits points by stroke", function() {
    var strokes = PointSet.eqDistStrokes(points, 2);
    expect(strokes.length).toBe(2);
    expect(strokes[0].length).toBeGreaterThan(0);
    expect(strokes[1].length).toBeGreaterThan(0);
  });

  it("resamples by equal stroke distribution", function() {
    spyOn(PointSet, 'countStrokes').and.returnValue(2);
    spyOn(PointSet, 'eqDistStrokes').and.returnValue([
      [new Point(0, 0, 0, 0), new Point(1, 0, 1, 0)],
      [new Point(2, 0, 2, 1), new Point(3, 0, 3, 1)]
    ]);
    spyOn(PointSet, 'unifResampling').and.callFake(function(stroke) {
      return stroke;
    });
    spyOn(PointSet, 'ensureResampling').and.stub();

    var out = PointSet.eqResample(points.slice(), 4);
    expect(out.length).toBe(4);
    expect(PointSet.ensureResampling).toHaveBeenCalledWith(out, 4);
  });

  it("ensures output length and triggers process exit on mismatch", function() {
    PointSet.ensureResampling([new Point()], 1);

    spyOn(console, 'trace').and.callFake(function() {});
    var originalExit = process.exit;
    process.exit = function() {
      throw new Error('forced exit');
    };

    expect(function() {
      PointSet.ensureResampling([new Point()], 2);
    }).toThrowError('forced exit');
    expect(console.trace).toHaveBeenCalled();

    process.exit = originalExit;
  });

  it("uses generic resampling", function() {
    spyOn(PointSet, 'unifResampling').and.returnValue([new Point(), new Point()]);
    spyOn(PointSet, 'ensureResampling').and.stub();

    var out = PointSet.resample(points.slice(), 2);
    expect(out.length).toBe(2);
    expect(PointSet.unifResampling).toHaveBeenCalled();
    expect(PointSet.ensureResampling).toHaveBeenCalledWith(out, 2);
  });
});
