from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from relacc.utils.date import DateUtil


def verbosity_from_opt(opt) -> int:
    """Return the normalized 0/1/2 verbosity level from argparse options."""
    verbose = getattr(opt, "verbose", 0)
    if verbose is True:
        return 2
    if verbose is False or verbose is None:
        return 0
    return int(verbose)


def add_run_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared verbosity/log-output flags used by experiment CLIs."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=2,
        default=0,
        help="Enable detailed logs.",
    )
    parser.add_argument(
        "--verbosity",
        dest="verbose",
        type=int,
        choices=[0, 1, 2],
        default=argparse.SUPPRESS,
        help="Logging verbosity: 0=silent, 1=warnings, 2=detailed logs.",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for run metadata and redirected stdout/stderr logs.",
    )


def parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Extract argparse defaults so a run records what was implicit."""
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS or action.dest == "help":
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    return defaults


def parsed_args_dict(opt) -> dict[str, Any]:
    return vars(opt).copy()


def _git_command(args: list[str], cwd: str) -> str | None:
    """Run a read-only git command and return None outside git worktrees."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_source_state(cwd: str | None = None) -> dict[str, Any]:
    """Capture commit, branch, and dirty status for reproducibility."""
    cwd = cwd or os.getcwd()
    root = _git_command(["rev-parse", "--show-toplevel"], cwd)
    if root is None:
        return {"gitAvailable": False}

    status = _git_command(["status", "--short"], root) or ""
    branch = _git_command(["branch", "--show-current"], root)
    return {
        "gitAvailable": True,
        "gitRoot": root,
        "gitHead": _git_command(["rev-parse", "HEAD"], root),
        "gitBranch": branch or None,
        "gitDirty": bool(status),
        "gitStatusShort": status.splitlines(),
    }


def execution_config(argv, prog: str | None = None) -> dict[str, Any]:
    """Capture the exact command-line and Python runtime used for a run."""
    cli_args = list(sys.argv[1:] if argv is None else argv)
    program = prog or Path(sys.argv[0]).name
    return {
        "program": program,
        "argv": cli_args,
        "command": shlex.join([program, *cli_args]),
        "cwd": os.getcwd(),
        "pythonExecutable": sys.executable,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
    }


def build_run_metadata(
    parser: argparse.ArgumentParser,
    opt,
    argv,
    experiment: str,
) -> dict[str, Any]:
    """Build the initial run metadata before derived settings are known."""
    return {
        "metadataVersion": 1,
        "experiment": experiment,
        "createdUtc": DateUtil.utc(),
        "createdTime": DateUtil.now(),
        "defaults": parser_defaults(parser),
        "runtimeArgs": parsed_args_dict(opt),
        "execution": execution_config(argv, prog=Path(sys.argv[0]).name),
        "source": git_source_state(),
    }


def sidecar_paths(
    output: str | Path | None = None,
    log_dir: str | Path | None = None,
    stem: str = "run",
    output_is_dir: bool = False,
) -> dict[str, Path]:
    """Return run metadata/log/stdout/stderr paths for an output target."""
    if log_dir:
        base = Path(log_dir)
        prefix = stem
        return {
            "json": base / f"{prefix}.run.json",
            "log": base / f"{prefix}.run.log",
            "stdout": base / f"{prefix}.stdout.log",
            "stderr": base / f"{prefix}.stderr.log",
        }

    if not output:
        return {}

    output_path = Path(output)
    if output_is_dir:
        return {
            "json": output_path / "run.json",
            "log": output_path / "run.log",
            "stdout": output_path / "stdout.log",
            "stderr": output_path / "stderr.log",
        }

    return {
        "json": output_path.with_name(output_path.name + ".run.json"),
        "log": output_path.with_name(output_path.name + ".run.log"),
        "stdout": output_path.with_name(output_path.name + ".stdout.log"),
        "stderr": output_path.with_name(output_path.name + ".stderr.log"),
    }


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def write_run_metadata(paths: Mapping[str, Path], metadata: Mapping[str, Any]) -> None:
    """Write the initial JSON metadata and human-readable run log."""
    if not paths:
        return
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(_json_text(metadata), encoding="utf-8")
    paths["log"].write_text(_format_run_log(metadata), encoding="utf-8")


def record_effective_config(
    paths: Mapping[str, Path],
    metadata: dict[str, Any] | None,
    effective_config: Mapping[str, Any],
) -> None:
    """Persist resolved runtime settings such as inferred label/rate/format."""
    if metadata is not None:
        metadata["effectiveConfig"] = dict(effective_config)
    if not paths:
        return
    paths["json"].write_text(_json_text(metadata or {}), encoding="utf-8")
    with open(paths["log"], "a", encoding="utf-8") as fh:
        fh.write("\nEffective execution configuration:\n")
        fh.write(_json_text(dict(effective_config)))


def append_run_log(paths: Mapping[str, Path], message: str) -> None:
    """Append one line to the run log when run logging is active."""
    if paths:
        with open(paths["log"], "a", encoding="utf-8") as fh:
            fh.write(str(message).rstrip() + "\n")


def redirect_experiment_streams(paths: Mapping[str, Path]):
    """Redirect stdout/stderr to sidecar files when paths are configured."""
    if not paths:
        return contextlib.nullcontext()
    return _redirect_streams(paths)


def run_logged_experiment(
    paths: Mapping[str, Path],
    callback: Callable[[], int],
) -> int:
    """Run an experiment while recording start, completion, and failures."""
    with redirect_experiment_streams(paths):
        append_run_log(paths, "Run started.")
        try:
            result = callback()
        except Exception as exc:
            append_run_log(paths, f"Run failed: {type(exc).__name__}: {exc}")
            raise
        append_run_log(paths, "Run completed.")
        return result


@contextlib.contextmanager
def _redirect_streams(paths: Mapping[str, Path]):
    with open(paths["stdout"], "w", encoding="utf-8") as stdout_fh:
        with open(paths["stderr"], "w", encoding="utf-8") as stderr_fh:
            with contextlib.redirect_stdout(stdout_fh), contextlib.redirect_stderr(
                stderr_fh
            ):
                yield


def _format_run_log(metadata: Mapping[str, Any]) -> str:
    """Format metadata into the readable .run.log companion file."""
    lines = [
        "Run metadata",
        "============",
        "",
        "Execution configuration:",
        _json_text(metadata["execution"]).rstrip(),
        "",
        "Source state:",
        _json_text(metadata["source"]).rstrip(),
        "",
        "Parser defaults:",
        _json_text(metadata["defaults"]).rstrip(),
        "",
    ]
    if "resolvedDefaults" in metadata:
        lines.extend(
            [
                "Resolved defaults:",
                _json_text(metadata["resolvedDefaults"]).rstrip(),
                "",
            ]
        )
    lines.extend(
        [
            "Parsed arguments (opt):",
            _json_text(metadata["runtimeArgs"]).rstrip(),
            "",
        ]
    )
    return "\n".join(lines)
