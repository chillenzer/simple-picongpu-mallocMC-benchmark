"""Parse PIConGPU benchmark run logs into per-run records.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Shared by the analysis scripts. The redesigned run logs are
self-describing: `run_stamp.sh` writes one machine-readable metadata
line,

    # metadata: {"schema": 1, ...}

into the header of every log, followed by the run's `set -x` trace. The
metadata carries the run context (`run.setup`, `run.algorithm`,
`run.delays`), the host, the git commit, the dependency pins, the
binary's hash and build, and the hardware it ran on; the trace
contributes the per-run grid (`-g` dimensions) and the runtime
(`calculation  simulation time:` line). One record is yielded per
`bin/picongpu` run: the metadata's run context plus the parse_grid /
parse_simulation_time values and the imposed
`malloc_sleeptime` / `free_sleeptime` in nanoseconds
(configuration "run-time").

A log is new format if and only if it carries a `# metadata:` line whose
JSON has `"schema": 1` (the cut rule); the logs of the pre-redesign eras
must be archived under legacy/ (make legacy-results) instead. A
`"kind": "setup"` metadata line marks a session log (build or full-series
launch); such a file yields no records. A file in a sweep machine's
output directory without a valid metadata line raises `LegacyLogError`.
"""

from __future__ import annotations

import json
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


def parse_log(log_path: Path) -> Iterator[dict]:
    """Yield one record per picongpu run of a single run log.

    The run context (setup, algorithm, delays) comes from the metadata
    line; the grid and the runtime come from the `set -x` trace. A delay
    value traced on the picongpu line that disagrees with the metadata is
    a warning, not an error (the metadata was written by the same script
    that sets the environment). A `"kind": "setup"` log yields no records.

    Yields:
        dict: one record per picongpu run of the log.

    """
    metadata = parse_metadata(log_path)
    if metadata.get("kind") == "setup":
        return
    run_ctx = metadata["run"]
    malloc_delay, free_delay = run_ctx["delays"]
    context = {
        "setup": run_ctx["setup"],
        "algorithm": run_ctx["algorithm"],
        MALLOC_DELAY: malloc_delay,
        FREE_DELAY: free_delay,
        "configuration": "run-time",
    }
    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        pending = None
        for line in map(str.strip, file):
            if line.startswith("+ "):
                _check_delays(line, malloc_delay, free_delay, log_path)
                if RUN_CMD in line:
                    pending = dict(context) | parse_grid(line)
            elif line.startswith("calculation") and "simulation time" in line and pending is not None:
                yield {**pending, **parse_simulation_time(line)}
                pending = None


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
