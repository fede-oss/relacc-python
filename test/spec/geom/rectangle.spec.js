describe("Rectangle", function() {

  var Rectangle = require('../../../lib/geom/rectangle');
  var Point     = require('../../../lib/geom/point');

  describe("when an instance is created", function() {

    var pt1 = new Point(1,2,100,0);
    var pt2 = new Point(3,4,200,0);
    var rectangle  = new Rectangle(pt1, pt2);
    var mockResult = {
      topLeft:      pt1,
      bottomRight:  pt2,
      width:        2,
      height:       2,
      area:         4,
    };

    it("should return the bounds", function() {
      expect(rectangle.topLeft).toEqual(mockResult.topLeft);
      expect(rectangle.bottomRight).toEqual(mockResult.bottomRight);
    });

    it("should return the width", function() {
      expect(rectangle.width()).toEqual(2);
    });

    it("should return the height", function() {
      expect(rectangle.height()).toEqual(2);
    });

    it("should return the area", function() {
      expect(rectangle.area()).toEqual(4);
    });

  });

});
