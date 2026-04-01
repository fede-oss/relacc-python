from contextlib import ExitStack, contextmanager
import gzip
import io
import math
import re
import tarfile

from relacc.geom.point import Point


class CSVUtil:
    """Process CSV gesture files."""

    REQUIRED_HEADERS = ("stroke_id", "x", "y", "time")

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
            index = CSVUtil._header_index(headers)
            if index is None:
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
    @contextmanager
    def _open_text_auto(file):
        with open(file, "rb") as probe:
            signature = probe.read(2)

        with ExitStack() as stack:
            if signature == b"\x1f\x8b":
                text_stream = CSVUtil._open_gzip_payload(stack, file)
            else:
                text_stream = stack.enter_context(open(file, "r", encoding="utf-8", errors="replace"))

            yield text_stream

    @staticmethod
    def _open_gzip_payload(stack, file):
        try:
            archive = stack.enter_context(tarfile.open(file, mode="r:gz"))
        except tarfile.ReadError:
            return stack.enter_context(gzip.open(file, "rt", encoding="utf-8", errors="replace"))

        members = [member for member in archive if member.isfile()]
        csv_members = [member for member in members if member.name.lower().endswith(".csv")]
        candidates = csv_members + [member for member in members if member not in csv_members]

        for member in candidates:
            payload = archive.extractfile(member)
            if payload is None:
                continue

            buffer = stack.enter_context(payload)
            text = buffer.read().decode("utf-8", errors="replace")
            if member.name.lower().endswith(".csv") or CSVUtil._header_index_from_text(text) is not None:
                return stack.enter_context(io.StringIO(text))

        raise ValueError("No CSV file found in gzip tar payload.")

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
    def _header_index(headers):
        index = {name.strip().lower(): i for i, name in enumerate(headers)}
        if any(name not in index for name in CSVUtil.REQUIRED_HEADERS):
            return None
        return index

    @staticmethod
    def _header_index_from_text(text):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            header = line.lstrip("\ufeff")
            delim = CSVUtil._detect_delimiter(header)
            headers = CSVUtil._split_fields(header, delim)
            return CSVUtil._header_index(headers)
        return None

    @staticmethod
    def _parse_number(value):
        value = value.strip()
        if value == "":
            raise ValueError("Empty numeric field in CSV.")
        return float(value)
