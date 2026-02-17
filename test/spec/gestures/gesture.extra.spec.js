describe("Gesture extras", function() {
  var Gesture = require('../../../lib/gestures/gesture');

  it("throws when point collection is empty", function() {
    expect(function() {
      new Gesture([], 'label');
    }).toThrow(new Error('Gesture points cannot be empty.'));
  });
});
