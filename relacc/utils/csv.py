import csv
import math

from relacc.geom.point import Point


class CSVUtil:
    """Process CSV gesture files."""

    @staticmethod
    def readGesture(file, callback):
        points = []
        myTime = None

        with open(file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=" ")
            for data in reader:
                x = int(data["x"])
                y = int(data["y"])
                time = int(data["time"])
                sid = int(data["stroke_id"])
                if time >= 0:
                    if time != myTime:
                        points.append(Point(x, y, time, sid))
                    myTime = time
                else:
                    points.append(Point(x, y, math.nan, sid))

        callback(points)
