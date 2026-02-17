describe("Point extras", function() {
  var Point = require('../../../lib/geom/point');

  it("supports constructor with only X and Y", function() {
    var pt = new Point(3, 4);
    expect(pt).toEqual(new Point(3, 4, 0, 0));
  });

  it("exposes absolute min and max points", function() {
    var min = Point.absMin();
    var max = Point.absMax();

    expect(min.X).toBe(Number.NEGATIVE_INFINITY);
    expect(min.Y).toBe(Number.NEGATIVE_INFINITY);
    expect(max.X).toBe(Number.POSITIVE_INFINITY);
    expect(max.Y).toBe(Number.POSITIVE_INFINITY);
  });
});
