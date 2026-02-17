import os
import tempfile

import coverage

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
        parser.addoption(
            "--cov-fail-under",
            action="store",
            default=None,
            type=float,
            help="Fail if total coverage is under this threshold",
        )
        parser.addoption(
            "--cov-report",
            action="append",
            default=[],
            help="Coverage report type (supports term-missing)",
        )
    except ValueError:
        # pytest-cov is installed and already added these flags.
        return


def pytest_sessionfinish(session, exitstatus):
    cov = getattr(session.config, "_relacc_cov", None)
    if not cov:
        return

    cov.stop()
    cov.save()

    reports = session.config.getoption("--cov-report") or []
    show_missing = any("term-missing" in rep for rep in reports)
    total = cov.report(show_missing=show_missing)

    fail_under = session.config.getoption("--cov-fail-under")
    if fail_under is not None and total < fail_under and session.exitstatus == 0:
        session.exitstatus = 2
