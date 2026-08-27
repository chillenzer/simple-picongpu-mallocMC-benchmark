"""Parse PIConGPU benchmark run logs into per-run records.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Shared by the analysis scripts: every one of them starts from the `set -x`
trace of a `run_folder.sh` run (invoked by the Makefile run targets, or by
the historical `run_all.sh`) and needs one record per
`bin/picongpu` run. This module extracts those records, independent of which
historical log layout produced the trace:

- the oldest per-variant layout (`cd build/<Example>/<Algorithm>-sleep<N>`,
  the delay compiled into the binary),
- the interim one-build-per-example layout (`cd .../build/<Example>`, which
  always ran FlatterScatter),
- the current one-build-per-(example, algorithm) layout
  (`cd .../build/<Example>/<Algorithm>`, the creation policy compiled in, the
  delays still injected at run time via the MALLOCMC_*_DELAY environment
  variables; pre-rename logs used MALLOCMC_SLEEP_TIME for the malloc delay).

A record carries the run context (`setup`, `algorithm`, the `x`/`y`/`z`
grid dimensions; a missing `z` of a 2-D run is filled with NaN by
`runs_to_df`) plus, for run-time sweeps, the imposed `malloc_sleeptime` /
`free_sleeptime` in nanoseconds and a `configuration` tag ("run-time" or
"compile-time"), the number of simulation steps from the run's `-s`
argument, and the runtimes in seconds of the run's `initialization time:`,
`calculation  simulation time:` (its own simulation time) and
`full simulation time:` (simulation plus initialisation and I/O) lines. A
log file without any parseable run (a build log) simply yields no records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

# The creation policy the interim one-build-per-example layout always ran.
ALGORITHM = "FlatterScatter"
CD_CMD = "+ cd "
RUN_CMD = "bin/picongpu "
MALLOC_DELAY_CMD = "MALLOCMC_MALLOC_DELAY="
FREE_DELAY_CMD = "MALLOCMC_FREE_DELAY="
# Pre-rename logs: the malloc delay was then injected via MALLOCMC_SLEEP_TIME.
LEGACY_MALLOC_DELAY_CMD = "MALLOCMC_SLEEP_TIME="
# `cd` trace lines that carry run context: the oldest per-variant layout
# (`cd build/<Example>/<Algorithm>-sleep<N>`, delays compiled in), the
# current one-build-per-(example, algorithm) layout
# (`cd .../build/<Example>/<Algorithm>`, delays still run-time), and the
# interim one-build-per-example layout (`cd .../build/<Example>`, which
# always ran FlatterScatter); every other `cd` (for example the `cd $WD`
# return in run_folder.sh) is ignored. The three patterns are mutually
# exclusive: `-sleep` breaks the trailing `\w+$` of the other two, and the
# per-algorithm path has one word component too many for the per-example
# one.
VARIANT_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)-sleep(\d+)$")
BUILD_ALGO_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)$")
BUILD_CD_RE = re.compile(r"(?:^|/)build/(\w+)$")
# The `-s` step count of one ran picongpu command (no other traced line
# carries it).
SIM_STEPS_RE = re.compile(r"-s (\d+)\b")
# The trailing `= <value> sec` of a `* simulation time:` /
# `initialization time:` line, in the shared format.
TIME_VALUE_RE = re.compile(r"= ([\d.]+) sec")

GROUP_KEYS = ("setup", "algorithm", "x", "y", "z")
MALLOC_DELAY = "malloc_sleeptime"
FREE_DELAY = "free_sleeptime"
DELAY_COLUMNS = (MALLOC_DELAY, FREE_DELAY)


def parse_setup(line: str) -> dict | None:
    """Parse the run context of a `cd` trace line; None if the line is unrelated.

    Returns:
        dict | None: the parsed setup context, or None if the line is unrelated.

    """
    path = line.rsplit(maxsplit=1)[-1]
    m = VARIANT_CD_RE.search(path)
    if m:
        # One build per (example, algorithm, sleeptime): the delay was
        # compiled into the binary (a malloc delay, no free delay).
        return {
            "setup": m[1],
            "algorithm": m[2],
            "malloc_sleeptime": int(m[3]),
            "free_sleeptime": 0,
            "configuration": "compile-time",
        }
    m = BUILD_ALGO_CD_RE.search(path)
    if m:
        # One build per (example, algorithm): the creation policy is
        # compiled into the binary; the delays are still injected at run
        # time via the MALLOCMC_*_DELAY environment variables.
        return {"setup": m[1], "algorithm": m[2]}
    m = BUILD_CD_RE.search(path)
    if m:
        return {"setup": m[1], "algorithm": ALGORITHM}
    return None


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


def _update_delays(line: str, malloc_delay: int | None, free_delay: int | None) -> tuple[int | None, int | None]:
    """Update the remembered delay env values from a traced `set -x` line.

    With `set -x`, the delay env prefixes are traced on their own line(s)
    before the picongpu line; the new layout puts both on the same line.
    Pre-rename logs only carry the malloc prefix, under the legacy name.

    Returns:
        tuple[int | None, int | None]: the updated (malloc_delay, free_delay).

    """
    if MALLOC_DELAY_CMD in line:
        malloc_delay = int(line.split(MALLOC_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    elif LEGACY_MALLOC_DELAY_CMD in line:
        malloc_delay = int(line.split(LEGACY_MALLOC_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    if FREE_DELAY_CMD in line:
        free_delay = int(line.split(FREE_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    return malloc_delay, free_delay


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

    The rosi logs run the combinations on several GPUs at once, so the
    time lines of the concurrent runs interleave; the first value of each
    line keeps the record of the run that finished first (the one the old
    parser recorded).

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
        time = TIME_VALUE_RE.search(value)
        if time is not None and "full_runtime_s" not in pending:
            pending["full_runtime_s"] = float(time[1])
        return True
    elif value.startswith("initialization time") and "init_time_s" not in pending:
        time = TIME_VALUE_RE.search(value)
        if time is not None:
            pending["init_time_s"] = float(time[1])
    return False


def _start_run(line: str, context: dict, malloc_delay: int | None, free_delay: int | None) -> dict | None:
    """Start a pending record at one traced `bin/picongpu` command line.

    Args:
        line: the stripped `set -x` trace line.
        context: the run context of the current `build/...` directory.
        malloc_delay: the remembered malloc delay, or None.
        free_delay: the remembered free delay, or None.

    Returns:
        dict | None: the started record, or None when the line is no
        `bin/picongpu` command of a known context.

    """
    if RUN_CMD not in line or "setup" not in context:
        return None
    # In the run-time layout the env vars override the variant sleeptime; in
    # the per-variant layout they are absent.
    pending = dict(context) | parse_grid(line)
    steps = SIM_STEPS_RE.search(line)
    if steps is not None:
        pending["sim_steps"] = int(steps[1])
    if malloc_delay is not None or free_delay is not None:
        pending |= {
            "malloc_sleeptime": (malloc_delay if malloc_delay is not None else 0),
            "free_sleeptime": (free_delay if free_delay is not None else 0),
            "configuration": "run-time",
        }
    return pending


def parse_log(log_path: Path) -> Iterator[dict]:
    """Yield one record per picongpu run of a single run log.

    The record is yielded when the run's `full simulation time:` line is
    seen (or, if a run ends without that line, at the next run context or
    at the end of the log), so the full and the initialisation runtimes,
    printed right after and long before the calculation time, belong to
    the same record. A run that aborted before printing its calculation
    time (a core-dump, an out-of-memory) carries no runtime and is not
    yielded.

    Yields:
        dict: one record per picongpu run of the log.

    """
    with log_path.open("r", encoding="utf-8") as file:
        context = {}
        pending = None
        malloc_delay = None
        free_delay = None
        for line in map(str.strip, file):
            if line.startswith(CD_CMD):
                # A new run context invalidates the remembered delay values
                # and, in case the run never printed its full time, ends
                # the pending record.
                setup = parse_setup(line)
                if setup is not None:
                    record = _flush(pending)
                    if record is not None:
                        yield record
                    pending = None
                    context = setup
                    malloc_delay = None
                    free_delay = None
            elif line.startswith("+ "):
                malloc_delay, free_delay = _update_delays(line, malloc_delay, free_delay)
                pending = _start_run(line, context, malloc_delay, free_delay) or pending
            elif pending is not None and _time_line(line, pending):
                record = _flush(pending)
                if record is not None:
                    yield record
                pending = None
        record = _flush(pending)
        if record is not None:
            yield record


def run_to_df(run: dict) -> pd.DataFrame:
    """Build a DataFrame from one run's records, tagged with its name.

    Returns:
        pd.DataFrame: the run's records, tagged with the run's name.

    """
    return pd.DataFrame(run["runs"]).assign(name=run["name"])


def runs_to_df(runs: Iterable[dict]) -> pd.DataFrame:
    """Concatenate the per-run DataFrames, filling a missing z with NaN.

    Returns:
        pd.DataFrame: the concatenated per-run frames, a missing z filled with NaN.

    """
    tmp = pd.concat(map(run_to_df, runs))
    return tmp.assign(z=tmp.get("z", np.nan))


def parse_logs(log_paths: Iterable[Path]) -> pd.DataFrame:
    """Parse every run log into a single DataFrame.

    Returns:
        pd.DataFrame: every run log parsed into one frame.

    """
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)
