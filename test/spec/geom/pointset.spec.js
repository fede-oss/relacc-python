describe("PointSet", function() {

  var Point    = require('../../../lib/geom/point');
  var PointSet = require('../../../lib/geom/pointset');
  var Maths    = require('../../helpers/math');

  var pt1, pt2, pt3, set;

  beforeEach(function() {
    pt1 = new Point(1,2,100,0);
    pt2 = new Point(3,4,200,0);
    pt3 = new Point(5,6,300,0);
    set = [ pt1, pt2, pt3 ];
  });

  describe("when a set of points is cloned", function() {

    it("should return a copy of the point set", function() {
      var cloned = PointSet.clone(set);
      expect(cloned).toEqual(set);
    });

  });

  describe("when the centroid of set of points is computed", function() {

    it("should return a Pont instance of center of gravity", function() {
      var centroid = PointSet.centroid(set);
      var gravity  = new Point(3,4,0,0);
      expect(centroid).toEqual(gravity);
    });

  });

  describe("when the bounds of set of points is computed", function() {

    it("should return the minimum point from a set", function() {
      var min = PointSet.minPt(set);
      var res = new Point(1,2,0,0);
      expect(min).toEqual(res);
    });

    it("should return the maximum point from a set", function() {
      var max = PointSet.maxPt(set);
      var res = new Point(5,6,0,0);
      expect(max).toEqual(res);
    });

    it("should return the bounding box from a set", function() {
      var bb  = PointSet.boundingBox(set);
      var res = {
        topLeft:     new Point(1,2,0,0),
        bottomRight: new Point(5,6,0,0),
      };
      expect(bb.topLeft).toEqual(res.topLeft);
      expect(bb.bottomRight).toEqual(res.bottomRight);
    });

  });

  describe("when the length of set of points is computed", function() {

    it("should return the full path length of a set", function() {
      var len = PointSet.pathLength(set);
      expect( Maths.roundTo(len) ).toBe(5.66);
    });

    it("should return the path length of a set from point 1 to point 2", function() {
      var len = PointSet.pathLength(set, 0, 1);
      expect( Maths.roundTo(len) ).toBe(2.83);
    });

    it("should return the path length of a set from point 2 to point 3", function() {
      var len = PointSet.pathLength(set, 1, 2);
      expect( Maths.roundTo(len) ).toBe(2.83);
    });

  });

  describe("when a set of points is transformed", function() {

    it("should return the scaled point set in [0,1]", function() {
      var sc  = PointSet.scale(set);
      var res = [
        new Point(0,0,100,0),
        new Point(0.5,0.5,200,0),
        new Point(1,1,300,0)
      ];
      expect(sc).toEqual(res);
    });

    it("should return the translated point set", function() {
      var tr  = PointSet.translateBy(set, pt1);
      var res = [
        new Point(0,0,100,0),
        new Point(2,2,200,0),
        new Point(4,4,300,0)
      ];
      expect(tr).toEqual(res);
    });

  });

  describe("when a set of points is resampled", function() {

    it("should return the first and last points if length is 2", function() {
      var rs  = PointSet.resample(set, 2);
      var res = [ pt1, pt3 ];
      expect(rs).toEqual(res);
    });

    it("should return the original 3 points if length is 3", function() {
      var rs  = PointSet.resample(set, 3);
      var res = [ pt1, pt2, pt3 ];
      expect(rs).toEqual(res);
    });

    it("should interpolate the original 3 points if length is greater than 3", function() {
      var rs  = PointSet.resample(set, 4);
      var res = [ pt1, pt2, pt3 ];
      expect(rs.length).toBeGreaterThan(res.length);
    });

  });

});
