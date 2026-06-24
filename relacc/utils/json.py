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
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                    time = float(pt[2])
                except (TypeError, ValueError, OverflowError, IndexError):
                    continue

                if not all(math.isfinite(value) for value in (x, y, time)) or time < 0:
                    continue

                if myTime is not None and time < myTime:
                    continue

                points.append(Point(x, y, time, sid))
                myTime = time

        callback(points)
