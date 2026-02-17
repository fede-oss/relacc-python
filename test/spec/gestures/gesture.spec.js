describe("Gesture", function() {

  var Gesture = require('../../../lib/gestures/gesture');
  var Point   = require('../../../lib/geom/point');

  var pt1, pt2, pt3, set;

  beforeEach(function() {
    pt1 = new Point(1,2,100,0);
    pt2 = new Point(3,4,200,0);
    pt3 = new Point(5,6,300,0);
    set = [ pt1, pt2, pt3 ];
  });

  describe("when an instance is created", function() {

//    it("should return a zero gesture if no points are given", function() {
//      var gesture = new Gesture(null, 'my label');
//      expect(gesture.points.length).toBe(1);
//    });

//    it("should return a zero gesture if an empty point set is given", function() {
//      var gesture = new Gesture([], 'my label');
//      expect(gesture.points.length).toBe(1);
//    });

    it("should throw an error if no gesture label is given", function() {
      var test = function(){ new Gesture(set) };
      expect(test).toThrow(new Error("Gesture name cannot be empty."));
    });

    it("should interpolate N points if no sampling rate is given", function() {
      var gesture = new Gesture(set, 'my label');
      expect(gesture.points.length).toBe(gesture.samplingRate);
      expect(gesture.originalPoints.length).not.toBe(gesture.samplingRate);
      expect(gesture.originalPoints.length).toBe(set.length);
    });

  });

});
