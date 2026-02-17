describe("Point", function() {

  var Point = require('../../../lib/geom/point');

  describe("when an instance is created", function() {

    var zeroPt = new Point(0,0,0,0);
    var unitPt = new Point(1,1,0,0);

    it("should return a zero point if no arguments are provided", function() {
      var pt = new Point();
      expect(pt).toEqual(zeroPt);
    });

    it("should return a Point if a Point is provided as single argument", function() {
      var pt = new Point({ X:1, Y:1, T:0, StrokeID:0 });
      expect(pt).toEqual(unitPt);
    });

    it("should return a Point if all arguments are provided", function() {
      var pt = new Point(1,1,0,0);
      expect(pt).toEqual(unitPt);
    });

  });

  describe("when a method is invoked", function() {

    var pt1, pt2;
    beforeEach(function() {
      pt1 = new Point(1,2,100,0);
      pt2 = new Point(3,4,200,0);
    });

    it("should return the max point value", function() {
      expect(pt1.maxXY()).toEqual(2);
      expect(pt2.maxXY()).toEqual(4);
    });

    it("should return the min point value", function() {
      expect(pt1.minXY()).toEqual(1);
      expect(pt2.minXY()).toEqual(3);
    });

    it("should return the sum of 2 points", function() {
      var sum = pt1.add(pt2);
      var res = new Point(4,6,100,0);
      expect(sum).toEqual(res);
    });

    it("should return the difference of 2 points", function() {
      var sub = pt1.subtract(pt2);
      var res = new Point(-2,-2,100,0);
      expect(sub).toEqual(res);
    });

    it("should return the division of a point by a scalar", function() {
      var div = pt1.divideBy(2);
      var res = new Point(0.5,1,100,0);
      expect(div).toEqual(res);
    });

    it("should throw an error when dividing a point by zero", function() {
      var test = function(){ pt1.divideBy(0) };
      expect(test).toThrow(new Error("Cannot divide by zero."));
    });

    it("should return the multiplication of a point by a scalar", function() {
      var mul = pt1.multiplyBy(2);
      var res = new Point(2,4,100,0);
      expect(mul).toEqual(res);
    });

  });

});
