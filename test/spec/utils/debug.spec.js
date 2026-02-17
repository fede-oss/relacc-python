describe("Debug util", function() {
  var Debug = require('../../../lib/utils/debug');

  it("writes to stderr in verbose mode", function() {
    spyOn(process.stderr, 'write').and.callFake(function() {});
    var dbg = new Debug({ verbose: true });
    dbg.log('hello');
    expect(process.stderr.write).toHaveBeenCalledWith('hello\n');
  });

  it("does not write to stderr when not verbose", function() {
    spyOn(process.stderr, 'write').and.callFake(function() {});
    var dbg = new Debug({ verbose: false });
    dbg.log('hello');
    expect(process.stderr.write).not.toHaveBeenCalled();
  });

  it("formats printf-style messages", function() {
    spyOn(process.stderr, 'write').and.callFake(function() {});
    var dbg = new Debug({ verbose: true });
    dbg.fmt('%d is a number', 42);
    expect(process.stderr.write).toHaveBeenCalledWith('42 is a number\n');
  });
});
