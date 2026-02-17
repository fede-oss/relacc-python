import time
from datetime import datetime, timezone


class DateUtil:
    """Date utilities."""

    @staticmethod
    def utc():
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    @staticmethod
    def now():
        return int(time.time() * 1000)
