import json
import sys


class Debug:
    """Debug utilities to display messages on stderr."""

    def __init__(self, opts=None):
        self.defaults = opts if opts is not None else {"verbose": False}

    def log(self, msg):
        if self.defaults.get("verbose"):
            sys.stderr.write(str(msg) + "\n")

    def fmt(self, format_str, msg=None):
        if "%j" in format_str:
            format_str = format_str.replace("%j", "%s")
            if msg is not None:
                msg = json.dumps(msg)
        if msg is None:
            self.log(format_str)
        else:
            self.log(format_str % msg)
