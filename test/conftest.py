import os
import tempfile


def pytest_sessionstart(session):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))
