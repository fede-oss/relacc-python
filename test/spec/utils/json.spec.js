describe("JSON util", function() {
  var fs = require('fs');
  var os = require('os');
  var path = require('path');
  var JSONUtil = require('../../../lib/utils/json');

  var tmpFile = path.join(os.tmpdir(), 'relacc-json-' + Date.now() + '.json');

  beforeAll(function() {
    var payload = {
      strokes: [
        [[1, 2, 0], [3, 4, 0], [5, 6, -1]],
        [[7, 8, 5]]
      ]
    };
    fs.writeFileSync(tmpFile, JSON.stringify(payload));
  });

  afterAll(function() {
    try {
      delete require.cache[require.resolve(tmpFile)];
    } catch (e) {
      // noop
    }
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
  });

  it("parses points, removes duplicate timestamps, and handles missing timestamps", function(done) {
    JSONUtil.readGesture(tmpFile, function(points) {
      expect(points.length).toBe(3);
      expect(points[0].X).toBe(1);
      expect(points[0].T).toBe(0);
      expect(points[1].X).toBe(5);
      expect(points[1].T).toBeNaN();
      expect(points[2].StrokeID).toBe(1);
      done();
    });
  });
});
