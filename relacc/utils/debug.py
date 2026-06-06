import json
import sys


class Debug:
    """Debug utilities to display messages on stderr."""

    def __init__(self, opts=None):
        self.defaults = opts if opts is not None else {"verbose": 0}
        verbose = self.defaults.get("verbose", 0)
        self.verbosity = 2 if verbose is True else int(verbose or 0)

    def log(self, msg, level=2):
        if self.verbosity >= level:
            sys.stderr.write(str(msg) + "\n")

    def fmt(self, format_str, msg=None, level=2):
        if "%j" in format_str:
            format_str = format_str.replace("%j", "%s")
            if msg is not None:
                msg = json.dumps(msg)
        if msg is None:
            self.log(format_str, level=level)
        else:
            self.log(format_str % msg, level=level)

    def warning(self, msg):
        self.log(msg, level=1)
