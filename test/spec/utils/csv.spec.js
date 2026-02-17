describe("CSV util", function() {
  var fs = require('fs');
  var os = require('os');
  var path = require('path');
  var CSVUtil = require('../../../lib/utils/csv');

  var tmpFile = path.join(os.tmpdir(), 'relacc-csv-' + Date.now() + '.csv');

  beforeAll(function() {
    var rows = [
      'stroke_id x y time is_writing',
      '0 10 20 0 1',
      '0 11 21 0 1',
      '1 12 22 -1 1'
    ];
    fs.writeFileSync(tmpFile, rows.join('\n'));
  });

  afterAll(function() {
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
  });

  it("parses points, removes duplicate timestamps, and handles missing timestamps", function(done) {
    CSVUtil.readGesture(tmpFile, function(points) {
      expect(points.length).toBe(2);
      expect(points[0].X).toBe(10);
      expect(points[0].T).toBe(0);
      expect(points[1].X).toBe(12);
      expect(points[1].T).toBeNaN();
      expect(points[1].StrokeID).toBe(1);
      done();
    });
  });
});
