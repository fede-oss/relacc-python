describe("RelAcc", function() {
  var RelAcc = require('../../lib/relacc');
  var Point = require('../../lib/geom/point');
  var PtAlignType = require('../../lib/gestures/ptaligntype');

  function p(x, y, t, sid) {
    return new Point(x, y, t, sid);
  }

  var summaryPts;
  var chronoPts;
  var cloudPts;
  var gesture;
  var summaryShape;

  beforeEach(function() {
    summaryPts = [
      p(0, 0, 0, 0),
      p(2, 2, 10, 0),
      p(4, 0, 20, 1)
    ];
    chronoPts = [
      p(0, 0, 0, 0),
      p(2, 1, 10, 0),
      p(4, 1, 20, 1)
    ];
    cloudPts = [
      p(0, 0, 0, 0),
      p(2, 2, 10, 0),
      p(4, 0, 20, 1)
    ];

    gesture = { points: chronoPts };
    summaryShape = {
      points: summaryPts,
      getPoints: function() {
        return summaryPts;
      },
      alignGesture: function(_, alignmentType) {
        if (alignmentType === PtAlignType.CLOUD_MATCH) return cloudPts;
        return chronoPts;
      }
    };
  });

  it("computes local and aggregate shape errors", function() {
    var local = RelAcc.localShapeErrors(gesture, summaryShape);
    expect(local).toEqual([0, 1, 1]);
    expect(RelAcc.shapeError(gesture, summaryShape)).toBeCloseTo(RelAcc.mean(local), 10);
    expect(RelAcc.shapeVariability(gesture, summaryShape)).toBeCloseTo(RelAcc.stdev(local), 10);
  });

  it("computes geometric aggregate metrics", function() {
    expect(RelAcc.lengthError(gesture, summaryShape)).toBeGreaterThan(0);
    expect(RelAcc.sizeError(gesture, summaryShape)).toBe(4);

    expect(RelAcc.bendingError(gesture, summaryShape)).toBeGreaterThanOrEqual(0);
    expect(RelAcc.bendingVariability(gesture, summaryShape)).toBeGreaterThanOrEqual(0);
    expect(RelAcc.localBendingErrors(gesture, summaryShape).length).toBe(3);
  });

  it("computes turning angles and normalizes angles above PI", function() {
    var points = [
      p(0, 0, 0, 0),
      p(1, 0, 1, 0),
      p(0.5, -0.866, 2, 0)
    ];

    var angles = RelAcc.turningAngleArray(points);
    expect(angles[0]).toBe(0);
    expect(angles[2]).toBe(0);
    expect(angles[1]).toBeLessThan(0);
  });

  it("computes kinematic metrics", function() {
    expect(RelAcc.timeError(gesture, summaryShape)).toBe(0);
    expect(RelAcc.timeVariability(gesture, summaryShape)).toBe(0);
    expect(RelAcc.velocityError(gesture, summaryShape)).toBeGreaterThan(0);
    expect(RelAcc.velocityVariability(gesture, summaryShape)).toBeGreaterThan(0);
    expect(RelAcc.localSpeedErrors(gesture, summaryShape).length).toBe(3);
  });

  it("computes production time and speed arrays", function() {
    expect(RelAcc.productionTime({ points: [p(1, 1, 7, 0)] })).toBe(0);
    expect(RelAcc.productionTime(gesture)).toBe(20);

    var zeroTime = [
      p(0, 0, 10, 0),
      p(1, 0, 10, 0),
      p(2, 0, 10, 0)
    ];
    var speed = RelAcc.speedArray(zeroTime);
    expect(speed).toEqual([0, 0, 0]);
  });

  it("computes articulation metrics", function() {
    expect(RelAcc.strokeError(gesture, summaryShape)).toBe(0);
    expect(RelAcc.numStrokes({ points: [] })).toBe(0);
    expect(RelAcc.numStrokes(gesture)).toBe(2);
    expect(RelAcc.strokeOrderError(gesture, summaryShape)).toBeGreaterThan(0);
  });

  it("computes mean and standard deviation with edge cases", function() {
    expect(function() {
      RelAcc.mean([]);
    }).toThrow(new Error('Input set cannot be empty.'));
    expect(RelAcc.mean([4])).toBe(4);
    expect(RelAcc.mean([2, 4, 6])).toBe(4);

    expect(function() {
      RelAcc.stdev([]);
    }).toThrow(new Error('Input set cannot be empty.'));
    expect(RelAcc.stdev([4])).toBe(0);
    expect(RelAcc.stdev([0, 2, 4])).toBeCloseTo(2, 10);
  });
});
