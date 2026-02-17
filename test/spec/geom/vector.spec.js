describe("Vector", function() {

  var Vector = require('../../../lib/geom/vector');
  var Point  = require('../../../lib/geom/point');
  var Maths  = require('../../helpers/math');

  var pt1 = new Point(1,2,100,0);
  var pt2 = new Point(3,4,200,0);
  var pt3 = new Point(5,6,300,0);
  var v1  = new Vector(pt1, pt2);
  var v2  = new Vector(pt2, pt3);
  var v3  = new Vector(pt1, pt3);

  describe("when an instance is created", function() {

    it("should return the length", function() {
      expect( Maths.roundTo(v1.length()) ).toBe(2.83);
      expect( Maths.roundTo(v2.length()) ).toBe(2.83);
      expect( Maths.roundTo(v3.length()) ).toBe(5.66);
    });

  });

  describe("when a method is called", function() {

    it("should return the dot product", function() {
      expect(Vector.dotProduct(v1,v2)).toEqual(8);
      expect(Vector.dotProduct(v2,v3)).toEqual(16);
      expect(Vector.dotProduct(v1,v3)).toEqual(16);
    });

  });

});
