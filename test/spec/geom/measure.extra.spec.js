describe("Measure extras", function() {
  var Measure = require('../../../lib/geom/measure');
  var Point = require('../../../lib/geom/point');
  var Vector = require('../../../lib/geom/vector');

  it("computes taxicab distance", function() {
    var a = new Point(1, 2, 0, 0);
    var b = new Point(4, 7, 0, 0);
    expect(Measure.taxicab(a, b)).toBe(8);
  });

  it("returns zero short angle when one vector has zero length", function() {
    var p = new Point(0, 0, 0, 0);
    var q = new Point(1, 0, 0, 0);
    var zero = new Vector(p, p);
    var nonZero = new Vector(p, q);
    expect(Measure.shortAngle(zero, nonZero)).toBe(0);
  });

  it("handles short-angle edge cases for cosine bounds", function() {
    var p = new Point(0, 0, 0, 0);
    var right = new Point(1, 0, 0, 0);
    var right2 = new Point(2, 0, 0, 0);
    var left = new Point(-1, 0, 0, 0);

    expect(Measure.shortAngle(new Vector(p, right), new Vector(p, right2))).toBe(0);
    expect(Measure.shortAngle(new Vector(p, right), new Vector(p, left))).toBeCloseTo(Math.PI, 10);
  });

  it("computes trigonometric angle when vectors are not in order", function() {
    var p = new Point(0, 0, 0, 0);
    var a = new Point(1, 0, 0, 0);
    var b = new Point(-1, 1, 0, 0);

    var v = new Vector(p, a);
    var u = new Vector(p, b);

    expect(Measure.trigonometricOrder(v, u)).toBe(false);
    expect(Measure.angle(v, u)).toBeGreaterThan(Math.PI);
  });
});
