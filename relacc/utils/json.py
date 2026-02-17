import json
import math

from relacc.geom.point import Point


class JSONUtil:
    """Process JSON gesture files."""

    @staticmethod
    def readGesture(file, callback):
        points = []
        myTime = None

        with open(file, "r", encoding="utf-8") as fh:
            strokes = json.load(fh)["strokes"]

        for sid, stroke in enumerate(strokes):
            for pt in stroke:
                x = pt[0]
                y = pt[1]
                time = pt[2]
                if time >= 0:
                    if time != myTime:
                        points.append(Point(x, y, time, sid))
                    myTime = time
                else:
                    points.append(Point(x, y, math.nan, sid))

        callback(points)
