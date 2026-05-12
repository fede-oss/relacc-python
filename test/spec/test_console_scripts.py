import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    ("script_name", "expected"),
    [
        ("relacc", "files"),
        ("relacc-pairwise", "reference"),
        ("relacc-distribution", "candidate"),
    ],
)
def test_installed_console_scripts_help_smoke(script_name, expected):
    executable = shutil.which(script_name)
    if executable is None:
        pytest.skip("%s console script is not installed" % script_name)

    res = subprocess.run(
        [executable, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected in res.stdout
