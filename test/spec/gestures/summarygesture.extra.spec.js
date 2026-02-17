describe("SummaryGesture extras", function() {
  var SummaryGesture = require('../../../lib/gestures/summarygesture');
  var PtAlignType = require('../../../lib/gestures/ptaligntype');
  var Gesture = require('../../../lib/gestures/gesture');
  var Point = require('../../../lib/geom/point');
  var PointSet = require('../../../lib/geom/pointset');

  function p(x, y, t, sid) {
    return new Point(x, y, t, sid);
  }

  var gestureA;
  var gestureB;

  beforeEach(function() {
    var setA = [
      p(0, 0, 0, 0),
      p(1, 0, 1, 0),
      p(2, 0, 2, 1),
      p(3, 0, 3, 1)
    ];
    var setB = [
      p(0, 0, 0, 0),
      p(1, 1, 1, 1),
      p(2, 0, 2, 0),
      p(3, 1, 3, 1)
    ];

    gestureA = new Gesture(setA, 'shape', 4);
    gestureB = new Gesture(setB, 'shape', 4);
  });

  it("exports points-for-alignment helper", function() {
    var aligned = SummaryGesture.getPointsForAlignment(gestureA);
    expect(aligned.length).toBe(gestureA.points.length);
  });

  it("aligns using cloud matching and normalizes stroke IDs", function() {
    var summary = new SummaryGesture([gestureA, gestureB], PtAlignType.CLOUD_MATCH);
    var aligned = summary.alignGesture(gestureB, PtAlignType.CLOUD_MATCH);

    expect(aligned.length).toBe(summary.refGesture.samplingRate);
    aligned.forEach(function(pt) {
      expect(pt.StrokeID).toBe(0);
    });
  });

  it("computes popular-stroke summary when requested", function() {
    var summary = new SummaryGesture([gestureA, gestureB, gestureA], PtAlignType.CHRONOLOGICAL, 'centroid', true);
    expect(summary.points.length).toBe(gestureA.samplingRate);
  });

  it("skips gestures above the popular stroke threshold", function() {
    var callIndex = 0;
    spyOn(PointSet, 'countStrokes').and.callFake(function() {
      callIndex += 1;
      return callIndex === 1 ? 1 : 2;
    });

    var self = {
      refGesture: gestureA,
      alignmentType: PtAlignType.CHRONOLOGICAL,
      alignGesture: jasmine.createSpy('alignGesture').and.callFake(function(gesture) {
        return gesture.points;
      })
    };

    var shapes = SummaryGesture.computeSummaryShapes(self, [gestureA, gestureB], 1);

    expect(self.alignGesture.calls.count()).toBe(1);
    expect(shapes.centroid.length).toBe(gestureA.samplingRate);
    expect(shapes.medoid.length).toBe(gestureA.samplingRate);
  });
});
