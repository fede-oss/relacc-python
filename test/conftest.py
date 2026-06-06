import os
import tempfile

import coverage
import pytest

from relacc.geom.point import Point


def pytest_sessionstart(session):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))
    cov_source = session.config.getoption("--cov", default=None)
    if cov_source:
        cov = coverage.Coverage(source=[cov_source])
        cov.start()
        session.config._relacc_cov = cov


def pytest_addoption(parser):
    try:
        parser.addoption("--cov", action="store", default=None, help="Coverage source package")
        parser.addoption("--cov-report", action="append", default=[], help="Coverage report type")
        parser.addoption("--cov-fail-under", action="store", default=None, type=float)
    except ValueError:
        return


def pytest_sessionfinish(session, exitstatus):
    cov = getattr(session.config, "_relacc_cov", None)
    if not cov:
        return

    cov.stop()
    cov.save()
    reports = session.config.getoption("--cov-report") or []
    total = cov.report(show_missing=any("term-missing" in report for report in reports))
    fail_under = session.config.getoption("--cov-fail-under")
    if fail_under is not None and total < fail_under and session.exitstatus == 0:
        session.exitstatus = 2


def point(x, y=0, time=0, stroke=0):
    return Point(x, y, time, stroke)


@pytest.fixture
def p():
    return point


@pytest.fixture
def write_gesture_csv(tmp_path):
    def _write(name, rows):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["stroke_id,x,y,time"]
        lines.extend("%s,%s,%s,%s" % row for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _write
