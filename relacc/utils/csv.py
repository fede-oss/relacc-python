import gzip
import math
import re

from relacc.geom.point import Point


class CSVUtil:
    """Process CSV gesture files."""

    @staticmethod
    def readGesture(file, callback):
        points = []
        myTime = None

        with CSVUtil._open_text_auto(file) as fh:
            header = None
            for line in fh:
                line = line.strip()
                if line:
                    header = line.lstrip("\ufeff")
                    break

            if not header:
                callback(points)
                return

            delim = CSVUtil._detect_delimiter(header)
            headers = CSVUtil._split_fields(header, delim)
            index = {name.strip().lower(): i for i, name in enumerate(headers)}
            required = ["stroke_id", "x", "y", "time"]
            if any(name not in index for name in required):
                raise ValueError("Invalid CSV header. Expected fields: stroke_id x y time is_writing")

            for line in fh:
                line = line.strip()
                if not line:
                    continue

                fields = CSVUtil._split_fields(line, delim)
                if len(fields) <= max(index["stroke_id"], index["x"], index["y"], index["time"]):
                    continue

                x = CSVUtil._parse_number(fields[index["x"]])
                y = CSVUtil._parse_number(fields[index["y"]])
                time = CSVUtil._parse_number(fields[index["time"]])
                sid = int(CSVUtil._parse_number(fields[index["stroke_id"]]))

                if time >= 0:
                    if time != myTime:
                        points.append(Point(x, y, time, sid))
                    myTime = time
                else:
                    points.append(Point(x, y, math.nan, sid))

        callback(points)

    @staticmethod
    def _open_text_auto(file):
        with open(file, "rb") as probe:
            signature = probe.read(2)
        if signature == b"\x1f\x8b":
            return gzip.open(file, "rt", encoding="utf-8", errors="replace")
        return open(file, "r", encoding="utf-8", errors="replace")

    @staticmethod
    def _detect_delimiter(header):
        if "," in header and "\t" not in header:
            return ","
        return None

    @staticmethod
    def _split_fields(line, delim):
        if delim == ",":
            return [field.strip() for field in line.split(",")]
        return re.split(r"\s+", line.strip())

    @staticmethod
    def _parse_number(value):
        value = value.strip()
        if value == "":
            raise ValueError("Empty numeric field in CSV.")
        return float(value)
