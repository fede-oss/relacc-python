describe("Args util", function() {
  var Args = require('../../../lib/utils/args');

  it("returns a default when the argument is missing", function() {
    var parser = new Args({});
    expect(parser.get('missing', 42)).toBe(42);
  });

  it("casts with a third argument", function() {
    var parser = new Args({ n: '1.9' });
    expect(parser.get('n', null, parseInt)).toBe(1);
  });

  it("casts when a function is passed as the second argument", function() {
    var parser = new Args({ n: '1.9' });
    expect(parser.get('n', Number)).toBeCloseTo(1.9, 10);
  });

  it("keeps falsy values instead of replacing them with defaults", function() {
    var parser = new Args({ zero: 0, bool: false });
    expect(parser.get('zero', 10)).toBe(0);
    expect(parser.get('bool', true)).toBe(false);
  });
});
