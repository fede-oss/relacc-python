describe("Measure", function() {

  var Measure = require('../../../lib/geom/measure');
  var Point   = require('../../../lib/geom/point');
  var Vector  = require('../../../lib/geom/vector');
  var Maths   = require('../../helpers/math');

  var pt1 = new Point(1,2,100,0);
  var pt2 = new Point(3,4,200,0);
  var pt3 = new Point(5,6,300,0);
  var v1  = new Vector(pt1, pt2);
  var v2  = new Vector(pt2, pt3);
  var v3  = new Vector(pt1, pt3);

  describe("when a static method is invoked", function() {

    it("should return the squared distance between 2 points", function() {
      var dist1 = Measure.sqDistance(pt1, pt2);
      var dist2 = Measure.sqDistance(pt2, pt3);
      var dist3 = Measure.sqDistance(pt1, pt3);
      expect(dist1).toBeCloseTo(8, 0);
      expect(dist2).toBeCloseTo(8, 0);
      expect(dist3).toBeCloseTo(32, 0);
    });

    it("should return the distance between 2 points", function() {
      var dist1 = Measure.distance(pt1, pt2);
      var dist2 = Measure.distance(pt2, pt3);
      var dist3 = Measure.distance(pt1, pt3);
      expect( Maths.roundTo(dist1) ).toBe(2.83);
      expect( Maths.roundTo(dist2) ).toBe(2.83);
      expect( Maths.roundTo(dist3) ).toBe(5.66);
    });

    it("should return the short angle between 2 vectors", function() {
      var angle1 = Measure.shortAngle(v1, v2);
      var angle2 = Measure.shortAngle(v2, v3);
      var angle3 = Measure.shortAngle(v1, v3);
      expect(angle1).toBeCloseTo(0);
      expect(angle2).toBeCloseTo(0);
      expect(angle3).toBeCloseTo(0);
    });

    it("should return the angle between 2 vectors", function() {
      var angle1 = Measure.angle(v1, v2);
      var angle2 = Measure.angle(v2, v3);
      var angle3 = Measure.angle(v1, v3);
      expect(angle1).toBeCloseTo(0);
      expect(angle2).toBeCloseTo(0);
      expect(angle3).toBeCloseTo(0);
    });

    it("should check the trigonometric order", function() {
      var order1 = Measure.trigonometricOrder(v1,v2);
      var order2 = Measure.trigonometricOrder(v2,v3);
      var order3 = Measure.trigonometricOrder(v1,v3);
      expect(order1).toBe(true);
      expect(order2).toBe(true);
      expect(order3).toBe(true);
    });

  });

});
