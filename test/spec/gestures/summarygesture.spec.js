describe("SummaryGesture", function() {

  var SummaryGesture = require('../../../lib/gestures/summarygesture');
  var PointAlignType  = require('../../../lib/gestures/ptaligntype');
  var Gesture  = require('../../../lib/gestures/gesture');
  var Point    = require('../../../lib/geom/point');
  var PointSet = require('../../../lib/geom/pointset');

  var pt1, pt2, pt3, set, collection;

  beforeEach(function() {
    pt1 = new Point(1,2,100,0);
    pt2 = new Point(3,4,200,0);
    pt3 = new Point(5,6,300,0);
    set = [ pt1, pt2, pt3 ];
    collection = [
      new Gesture(set, 'my label', 3),
      new Gesture(set, 'my label', 3)
    ];
  });

  describe("when an instance is created", function() {

    it("should throw an error if different gesture labels are given in the same collection", function() {
      var test = function(){
        new SummaryGesture([
          new Gesture(set, 'my label'),
          new Gesture(set, 'different label')
        ]);
      };
      expect(test).toThrow(new Error("Gesture names cannot be different."));
    });

    it("should use cronological alignment if no preference is provided", function() {
      var summary = new SummaryGesture(collection);
      expect(summary.alignmentType).toBe(PointAlignType.CHRONOLOGICAL);
    });

    it("should use an alignment type if provided", function() {
      var summary = new SummaryGesture(collection, PointAlignType.CLOUD_MATCH);
      expect(summary.alignmentType).toBe(PointAlignType.CLOUD_MATCH);
    });

    it("should replace the original points by the centroid if centroid mode is specified", function() {
      var summary = new SummaryGesture(collection, PointAlignType.CHRONOLOGICAL, 'centroid');
      var result = [
        new Point(-2,-2,100,0),
        new Point(0,0,200,0),
        new Point(2,2,300,0)
      ];
      expect(summary.originalPoints).toEqual(result);
    });

    it("should use the closest gesture sample as centroid if kcentroid mode is specified", function() {
      var summary = new SummaryGesture(collection, PointAlignType.CHRONOLOGICAL, 'kcentroid');
      expect(summary.originalPoints).toEqual(set);
    });

    it("should replace the original points by the centroid if medoid mode is specified", function() {
      var summary = new SummaryGesture(collection, PointAlignType.CHRONOLOGICAL, 'medoid');
      var result = [
        new Point(-2,-2,100,0),
        new Point(0,0,200,0),
        new Point(2,2,300,0)
      ];
      expect(summary.originalPoints).toEqual(result);
    });

    it("should use the closest gesture sample as medoid if kmedoid mode is specified", function() {
      var summary = new SummaryGesture(collection, PointAlignType.CHRONOLOGICAL, 'kmedoid');
      expect(summary.originalPoints).toEqual(set);
    });

  });

});
