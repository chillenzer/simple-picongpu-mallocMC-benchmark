"""Parse PIConGPU benchmark run logs into per-run records.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Shared by the analysis scripts. The redesigned run logs are
self-describing: `run_stamp.sh` writes one machine-readable metadata
line,

    # metadata: {"schema": 1, ...}

into the header of every log, followed by the run's `set -x` trace. The
metadata carries the run context (`run.setup`, `run.algorithm`,
`run.delays`, the flags-file line the run came from), the host, the git
commit, the dependency pins, the binary's hash and build, and the
hardware it ran on; the trace contributes the per-run grid (`-g`
dimensions) and the simulation times. One record is yielded per
`bin/picongpu` run: the metadata's run context plus the parse_grid /
parse_simulation_time values and the imposed
`malloc_sleeptime` / `free_sleeptime` in nanoseconds
(configuration "run-time"), the run's number of simulation steps
(`sim_steps`, from the flags-file line), the runtimes of the run's
`initialization time:` and `full simulation time:` lines
(`init_time_s`, `full_runtime_s`), and the provenance of the log (one
column per fact of the metadata: `started_utc`, `commit`,
`binary_sha256`, `picongpu`, `mallocmc`, `gpu`, `gpu_driver`,
`cuda_version`, `cpu`, `compiler`, `host`, `slurm_job`; the empty
string where the metadata carries nothing). The record is yielded when
the run's `full simulation time:` line is seen (or, if a run ends
without it, at the next run or the end of the log), so all three of the
run's times belong to the same record.

A log is new format if and only if it carries a `# metadata:` line whose
JSON has `"schema": 1` (the cut rule); the logs of the pre-redesign eras
must be archived under legacy/ (make legacy-results) instead. A
`"kind": "setup"` metadata line marks a session log (build or full-series
launch); such a file yields no records. A file in a sweep machine's
output directory without a valid metadata line raises `LegacyLogError`.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_PREFIX = "# metadata: "
SCHEMA = 1
RUN_CMD = "bin/picongpu "
MALLOC_DELAY_CMD = "MALLOCMC_MALLOC_DELAY="
FREE_DELAY_CMD = "MALLOCMC_FREE_DELAY="
# The `-s` step count of the flags-file line a run came from.
SIM_STEPS_RE = re.compile(r"(?:^|\s)-s (\d+)\b")
# The trailing `= <value> sec` of an `initialization time:` /
# `full simulation time:` line.
TIME_VALUE_RE = re.compile(r"= ([\d.]+) sec")
# The placeholder `logmeta.py` writes for an unavailable fact.
UNAVAILABLE = "unavailable"

GROUP_KEYS = ("setup", "algorithm", "x", "y", "z")
MALLOC_DELAY = "malloc_sleeptime"
FREE_DELAY = "free_sleeptime"
DELAY_COLUMNS = (MALLOC_DELAY, FREE_DELAY)


class LegacyLogError(Exception):
    """A run log of a sweep machine's output directory is not in the new format.

    The pre-redesign logs are archived under legacy/ instead (see
    legacy/README.md); a remaining one belongs to legacy/logs/.
    """


def parse_metadata(log_path: Path) -> dict:
    """Return the metadata dict of a run log's `# metadata:` line.

    Args:
        log_path: the log file.

    Returns:
        dict: the parsed metadata (schema 1).

    Raises:
        LegacyLogError: if the file carries no metadata line with
            schema 1 (a pre-redesign log).

    """
    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.startswith(METADATA_PREFIX):
                continue
            try:
                metadata = json.loads(line[len(METADATA_PREFIX) :])
            except json.JSONDecodeError:
                metadata = {}
            if isinstance(metadata, dict) and metadata.get("schema") == SCHEMA:
                return metadata
            break
    msg = f"{log_path} has no '# metadata:' line with schema {SCHEMA}"
    raise LegacyLogError(msg)


def parse_grid(line: str) -> dict[str, int]:
    """Parse the `-g` grid dimensions out of a picongpu command line.

    Returns:
        dict[str, int]: the grid dimensions, keyed by x, y, z.

    """
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            line.split(RUN_CMD, 1)[1].split("-g", 1)[1].split("-", maxsplit=1)[0].strip().split(" "),
            # 2-D grids have only two values; the zip truncates the keys to
            # the dimensions present (the missing one becomes NaN downstream).
            strict=False,
        )
    }


def parse_simulation_time(line: str) -> dict[str, float]:
    """Parse a `calculation  simulation time` line into a runtime dict.

    Returns:
        dict[str, float]: the simulation runtime in seconds.

    """
    return {"runtime in s": float(line.split("=")[1][: -len("sec")])}


def _text(value: object) -> str:
    """Return one metadata value as a stored text column.

    Args:
        value: the value from the metadata JSON.

    Returns:
        str: the value as a string, `""` for the "unavailable"
        placeholder and for a missing value.

    """
    if isinstance(value, str) and value != UNAVAILABLE:
        return value
    return ""


def _flush(pending: dict | None) -> dict | None:
    """Return the pending record when it measured a runtime, else None.

    Args:
        pending: the run's record in progress, or None.

    Returns:
        dict | None: the record, or None when it is absent or the run
        aborted before printing its calculation time.

    """
    if pending is not None and "runtime in s" in pending:
        return pending
    return None


def _time_line(value: str, pending: dict) -> bool:
    """Record one of the run's time lines on the pending record.

    The first value of each line type is the recorded one, so a log that
    carries stray time lines of other (concurrent) runs never overwrites
    the values of this run.

    Args:
        value: the stripped log line.
        pending: the run's record in progress.

    Returns:
        bool: True when the line is the run's `full simulation time:`,
        which ends the record.

    """
    if value.startswith("calculation") and "simulation time" in value and "runtime in s" not in pending:
        pending.update(parse_simulation_time(value))
    elif value.startswith("full simulation time"):
        match = TIME_VALUE_RE.search(value)
        if match is not None and "full_runtime_s" not in pending:
            pending["full_runtime_s"] = float(match[1])
        return True
    elif value.startswith("initialization time") and "init_time_s" not in pending:
        match = TIME_VALUE_RE.search(value)
        if match is not None:
            pending["init_time_s"] = float(match[1])
    return False


def _start_run(line: str, context: dict) -> dict | None:
    """Start a pending record at one traced `bin/picongpu` command line.

    Args:
        line: the stripped `set -x` trace line.
        context: the run context of the log (from its metadata).

    Returns:
        dict | None: the started record, or None when the line is no
        `bin/picongpu` command.

    """
    if RUN_CMD not in line:
        return None
    pending = dict(context) | parse_grid(line)
    steps = SIM_STEPS_RE.search(line)
    if steps is not None:
        pending["sim_steps"] = int(steps[1])
    return pending


def _metadata_source(metadata: dict) -> dict:
    """Return the provenance columns of one run log's metadata.

    Args:
        metadata: the metadata dict of the log (schema 1).

    Returns:
        dict: one value per provenance column (the empty string where the
        metadata carries nothing or the "unavailable" placeholder).

    """
    pins = metadata.get("pins", {})
    if not isinstance(pins, dict):
        pins = {}
    binary = metadata.get("binary", {})
    if not isinstance(binary, dict):
        binary = {}
    hw = metadata.get("hw", {})
    if not isinstance(hw, dict):
        hw = {}
    build = metadata.get("build", {})
    if not isinstance(build, dict):
        build = {}
    gpus = hw.get("gpu")
    commit = _text(metadata.get("commit"))
    if commit and metadata.get("dirty") is True:
        commit += " (dirty)"
    return {
        "started_utc": _text(metadata.get("ts")),
        "commit": commit,
        "binary_sha256": _text(binary.get("sha256")),
        "picongpu": _text(pins.get("picongpu")),
        "mallocmc": _text(pins.get("mallocmc")),
        "gpu": ",".join(gpus) if isinstance(gpus, list) and gpus else "",
        "gpu_driver": _text(hw.get("gpu_driver")),
        "cuda_version": _text(build.get("cuda")),
        "cpu": _text(hw.get("cpu")),
        "compiler": _text(build.get("compiler")),
        "host": _text(metadata.get("hostname")),
        "slurm_job": _text(metadata.get("slurm_job")),
    }


def parse_log(log_path: Path) -> Iterator[dict]:
    """Yield one record per picongpu run of a single run log.

    The run context (setup, algorithm, delays), the number of simulation
    steps and the provenance come from the metadata line; the grid and
    the simulation times come from the `set -x` trace. A delay value
    traced on the picongpu line that disagrees with the metadata is a
    warning, not an error (the metadata was written by the same script
    that sets the environment). A record is yielded when its
    `full simulation time:` line is seen (or, if the run ends without
    it, at the next run or the end of the log), so the initialisation,
    the calculation and the full runtimes belong to the same record. A
    `"kind": "setup"` log yields no records.

    Yields:
        dict: one record per picongpu run of the log.

    """
    metadata = parse_metadata(log_path)
    if metadata.get("kind") == "setup":
        return
    run_ctx = metadata["run"]
    malloc_delay, free_delay = run_ctx["delays"]
    steps = SIM_STEPS_RE.search(run_ctx.get("command", ""))
    context = {
        "setup": run_ctx["setup"],
        "algorithm": run_ctx["algorithm"],
        MALLOC_DELAY: malloc_delay,
        FREE_DELAY: free_delay,
        "configuration": "run-time",
    } | _metadata_source(metadata)
    if steps is not None:
        context["sim_steps"] = int(steps[1])
    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        pending = None
        for line in map(str.strip, file):
            if line.startswith("+ "):
                _check_delays(line, malloc_delay, free_delay, log_path)
                pending = _start_run(line, context) or pending
            elif pending is not None and _time_line(line, pending):
                record = _flush(pending)
                if record is not None:
                    yield record
                pending = None
        record = _flush(pending)
        if record is not None:
            yield record


def _check_delays(line: str, malloc_delay: int, free_delay: int, log_path: Path) -> None:
    """Warn when a traced delay value disagrees with the metadata's.

    Args:
        line: the `set -x` trace line carrying the delay env prefixes.
        malloc_delay: the metadata's malloc delay in nanoseconds.
        free_delay: the metadata's free delay in nanoseconds.
        log_path: the log the line came from (for the warning).

    """
    for prefix, expected in ((MALLOC_DELAY_CMD, malloc_delay), (FREE_DELAY_CMD, free_delay)):
        if prefix not in line:
            continue
        traced = int(line.split(prefix, 1)[1].split(maxsplit=1)[0])
        if traced != expected:
            warnings.warn(
                f"{log_path}: traced delay {traced} ns differs from the metadata's {expected} ns",
                stacklevel=2,
            )


def runs_to_df(runs: Iterable[dict]) -> pd.DataFrame:
    """Concatenate the per-run DataFrames, filling a missing z with NaN.

    Args:
        runs: the per-run dicts, each with a `name` and its `runs`.

    Returns:
        pd.DataFrame: the concatenated per-run frames, a missing z filled with NaN.

    """
    tmp = pd.concat([pd.DataFrame(run["runs"]).assign(name=run["name"]) for run in runs])
    return tmp.assign(z=tmp.get("z", np.nan))


def parse_logs(log_paths: Iterable[Path]) -> pd.DataFrame:
    """Parse every run log into a single DataFrame.

    Args:
        log_paths: the run log files.

    Returns:
        pd.DataFrame: every run log parsed into one frame.

    """
    log_paths = list(log_paths)
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)
