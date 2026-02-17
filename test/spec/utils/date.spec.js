describe("Date util", function() {
  var DateUtil = require('../../../lib/utils/date');

  it("returns current UTC string", function() {
    expect(typeof DateUtil.utc()).toBe('string');
    expect(Date.parse(DateUtil.utc())).not.toBeNaN();
  });

  it("returns current timestamp", function() {
    var now = DateUtil.now();
    expect(typeof now).toBe('number');
    expect(now).toBeGreaterThan(0);
  });
});
