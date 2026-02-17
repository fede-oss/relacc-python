class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class Args:
    """Process CLI arguments."""

    def __init__(self, args):
        self._args = args

    def get(self, name, defaultValue=_SENTINEL, castFn=None):
        hasArg = name in self._args and self._args[name] is not None
        if not hasArg and defaultValue is not _SENTINEL:
            value = defaultValue
        else:
            value = self._args.get(name)

        if callable(castFn):
            return castFn(value)
        if castFn is None and defaultValue is not _SENTINEL and callable(defaultValue):
            return defaultValue(value)
        return value
