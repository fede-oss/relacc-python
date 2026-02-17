describe("Math util", function() {
  var MathUtil = require('../../../lib/utils/math');

  it("rounds with default precision", function() {
    expect(MathUtil.roundTo(Math.PI)).toBe(3.142);
  });

  it("rounds with custom precision", function() {
    expect(MathUtil.roundTo(Math.PI, 2)).toBe(3.14);
  });

  it("computes factorial", function() {
    expect(MathUtil.factorial(0)).toBe(1);
    expect(MathUtil.factorial(5)).toBe(120);
  });
});
