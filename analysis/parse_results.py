r"""Parse pre-filtered run logs into a pandas DataFrame for seaborn.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Input is the output of, e.g.

    grep "cd build/Foil\\|cd build/Kelvin\\|bin/picongpu \\|calculation" output/hal-sleeptimes/run_*

i.e. lines of the form `<log-file>:<content>` where `<content>` is one of

    + cd <path>/build/<Example>[/<policy-sleep<N>>]
    + MALLOCMC_SLEEP_TIME=<N>
    [+ MALLOCMC_SLEEP_TIME=<N>] bin/picongpu <flags>
    [+ MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F>] bin/picongpu <flags>
    calculation  simulation time:  ... = <t> sec

The first form sets the current example (and, for the old per-variant
directories, the policy and sleep_time as well); the other forms mark a run.
In the current log layout the `set -x` trace puts the delay environment
prefixes (`MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F>`, or in
pre-rename logs `MALLOCMC_SLEEP_TIME=<N>`) before `bin/picongpu` on the same
line, so the delays are taken from there (or from the prefix when it is on
its own line, or from the variant directory in the old per-variant layout);
the run's flags are parsed (grid from `-g`, steps from `-s`); the last form
closes the most recent run with its wall time. One DataFrame row is produced
per picongpu run.

For the current log layout (one build per example, delays set via the
`MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables), the
`cd` lines are absolute paths, so use a grep that still captures them, e.g.

    grep -E "cd .*build/(FoilLCT|KelvinHelmholtz)|Using allocator|MALLOCMC_(SLEEP_TIME|MALLOC_DELAY|FREE_DELAY)=|" \
        "bin/picongpu |calculation" \
        output/hal-sleeptimes/run_*

Both the old and the new log layout parse correctly.

Usage:

    python3 analysis/parse_results.py results.txt              # print the frame
    python3 analysis/parse_results.py results.txt --csv r.csv  # also write CSV

    import sys
    sys.path.insert(0, "analysis")
    from parse_results import parse_results
    df = parse_results("results.txt")
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# `<log-file>:+ <trace>`, `<log-file>:calculation ...` or
# `<log-file>:Using allocator: ...`
# The greedy file part skips over the `:` characters inside the log file
# name (run_..._12:10:36+02:00.txt).
LINE_RE = re.compile(r"^(?P<file>.+):(?P<rest>\+.*|calculation.*|Using allocator.*)$")
CD_RE = re.compile(r"^\+ cd (?P<path>\S+)$")
BUILD_PATH_RE = re.compile(r"(?:^|/)(?:build/(?P<example>\w+)(?:/(?P<variant>[^/]+))?)$")
VARIANT_RE = re.compile(r"^(?P<policy>\w+)-sleep(?P<sleep>\d+)$")
# New combination-sweep run label; old runs used `sleep_time: <N> ns`.
ALLOCATION_RE = re.compile(
    r"^Using allocator: (?P<policy>\w+), "
    r"(?:sleep_time: (?P<sleep>\d+) ns"
    r"|malloc delay: (?P<malloc_delay>\d+) ns, free delay: (?P<free_delay>\d+) ns)$"
)
PICONGPU_RE = re.compile(r"^\+(?P<prefix>.*)bin/picongpu (?P<flags>.*)$")
SLEEP_ENV_RE = re.compile(r"MALLOCMC_SLEEP_TIME=(\d+)")
MALLOC_DELAY_ENV_RE = re.compile(r"MALLOCMC_MALLOC_DELAY=(\d+)")
FREE_DELAY_ENV_RE = re.compile(r"MALLOCMC_FREE_DELAY=(\d+)")
# `set -x` traces the delay prefixes on their own line before the picongpu
# line (pre-rename logs: a single MALLOCMC_SLEEP_TIME assignment).
SLEEP_ENV_ASSIGN_RE = re.compile(r"^\+ MALLOCMC_SLEEP_TIME=(?P<sleep>\d+)$")
GRID_RE = re.compile(r"-g\s+(?P<grid>(?:\d+\s+)+\d+)(?=\s+-|\s*$)")
STEPS_RE = re.compile(r"-s\s+(?P<steps>\d+)")
CALC_RE = re.compile(r"^calculation\s+simulation time:.*?=\s*(?P<time>[\d.]+)\s*sec$")

COLUMNS = [
    "file",
    "example",
    "grid",
    "grid_x",
    "grid_y",
    "grid_z",
    "steps",
    "policy",
    "sleep_time",
    "free_sleep_time",
    "time_seconds",
]


def _split_line(line: str) -> tuple[str | None, str]:
    """Split a pre-filtered line into its `<log-file>` prefix and content.

    Returns:
        tuple[str | None, str]: the file and the content, or (None, line)
        when the line carries no file prefix.

    """
    m = LINE_RE.match(line)
    return (m["file"], m["rest"]) if m else (None, line)


def _handle_calculation(line: str, rest: str, file: str | None, pending: dict | None, rows: list[dict]) -> None:
    """Close the pending run with the wall time of a `calculation` line.

    Prints a diagnostic instead when the line cannot be closed.
    """
    cm = CALC_RE.match(rest)
    if cm is None:
        print(f"parse_results: unrecognised calculation line: {line}", file=sys.stderr)
    elif pending is None:
        print(f"parse_results: calculation without a preceding run (dropped): {line}", file=sys.stderr)
    else:
        rows.append({"file": file, **pending, "time_seconds": float(cm["time"])})


def _handle_cd(cm: re.Match, ctx: dict) -> None:
    """Update the run context from a `cd` line into a build directory."""
    bm = BUILD_PATH_RE.search(cm["path"])
    if bm:
        ctx["example"] = bm["example"]
        vm = VARIANT_RE.match(bm["variant"]) if bm["variant"] else None
        ctx["policy"] = vm["policy"] if vm else None
        ctx["sleep_time"] = float(vm["sleep"]) if vm else None
        ctx["free_sleep_time"] = 0.0
        # A new build context invalidates a remembered run-time env.
        ctx["sleep_env"] = None
        ctx["free_env"] = None


def _handle_allocation(cm: re.Match, ctx: dict) -> None:
    """Record the run's creation policy and delays from the run label."""
    ctx["policy"] = cm["policy"]
    if cm["sleep"] is not None:
        ctx["sleep_time"] = float(cm["sleep"])
        ctx["free_sleep_time"] = 0.0
    else:
        ctx["sleep_time"] = float(cm["malloc_delay"])
        ctx["free_sleep_time"] = float(cm["free_delay"])


def _resolve_sleep_times(cm: re.Match, ctx: dict) -> tuple[float, float]:
    """Resolve a run's (malloc, free) sleep times from all known sources.

    Returns:
        tuple[float, float]: the malloc and free sleep times in nanoseconds.

    """
    prefix = cm["prefix"]
    malloc_env = MALLOC_DELAY_ENV_RE.search(prefix)
    free_env = FREE_DELAY_ENV_RE.search(prefix)
    sleep_env = SLEEP_ENV_RE.search(prefix)
    if malloc_env:
        # Current layout: both prefixes on the picongpu line.
        return float(malloc_env.group(1)), float(free_env.group(1)) if free_env else 0.0
    if sleep_env:
        # Pre-rename layout: inline prefix on the picongpu line.
        return float(sleep_env.group(1)), 0.0
    if ctx["sleep_env"] is not None:
        # Standalone assignment line of the current run.
        return ctx["sleep_env"], ctx["free_env"] if ctx["free_env"] is not None else 0.0
    # Per-variant layout: compiled into the binary.
    return ctx["sleep_time"], ctx["free_sleep_time"]


def _handle_picongpu(line: str, cm: re.Match, ctx: dict) -> dict | None:
    """Record the pending run of a `bin/picongpu` line.

    Returns:
        dict or None: the pending run for the next `calculation` line, or
        None when the context is incomplete (a diagnostic is printed).

    """
    grid = GRID_RE.search(cm["flags"])
    steps = STEPS_RE.search(cm["flags"])
    if grid is None or steps is None or ctx["example"] is None:
        print(f"parse_results: incomplete run (needs cd, -g and -s context): {line}", file=sys.stderr)
        return None
    gdims = [int(x) for x in grid["grid"].split()]
    sleep_time, free_sleep_time = _resolve_sleep_times(cm, ctx)
    return {
        "example": ctx["example"],
        "grid": "x".join(map(str, gdims)),
        "grid_x": gdims[0],
        "grid_y": gdims[1] if len(gdims) > 1 else 1,
        "grid_z": gdims[2] if len(gdims) > 2 else 1,
        "steps": int(steps["steps"]),
        "policy": ctx["policy"],
        "sleep_time": sleep_time,
        "free_sleep_time": free_sleep_time,
    }


def parse_results(source: str | Path | Iterable[str]) -> pd.DataFrame:
    """Parse the pre-filtered run log `source` (path or line iterable).

    See the module docstring for the expected input format and the columns.
    `sleep_time` (the malloc delay) and `free_sleep_time` (the free delay, `0`
    for pre-rename and compile-time runs) are in nanoseconds, `time_seconds`
    is the simulation wall time in seconds.

    Returns:
        pd.DataFrame: one row per picongpu run.

    """
    ctx = {
        "example": None,
        "policy": None,
        "sleep_time": None,
        "free_sleep_time": 0.0,
        "sleep_env": None,
        "free_env": None,
    }
    pending = None
    rows = []

    if not (hasattr(source, "readline") or isinstance(source, (list, tuple))):
        with Path(source).open(encoding="utf-8") as file:
            source = file.readlines()
    for raw in source:
        line = raw.strip()
        file, rest = _split_line(line)
        if rest.startswith("calculation"):
            _handle_calculation(line, rest, file, pending, rows)
            pending = None
        elif cm := CD_RE.match(rest):
            _handle_cd(cm, ctx)
        elif cm := SLEEP_ENV_ASSIGN_RE.match(rest):
            # `set -x` traces the MALLOCMC_SLEEP_TIME prefix on its own line
            # before the picongpu line.
            ctx["sleep_env"] = float(cm["sleep"])
        elif cm := ALLOCATION_RE.match(rest):
            _handle_allocation(cm, ctx)
        elif cm := PICONGPU_RE.match(rest):
            pending = _handle_picongpu(line, cm, ctx)

    if pending is not None:
        print("parse_results: run without a calculation line (dropped)", file=sys.stderr)
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> int:
    """Parse a pre-filtered run log and print (and optionally write) the frame.

    Returns:
        int: a process exit status; 0 on success.

    """
    parser = argparse.ArgumentParser(description="Parse pre-filtered run logs into a DataFrame")
    parser.add_argument("results", help="pre-filtered log (output of the grep)")
    parser.add_argument("--csv", metavar="OUT", help="also write the DataFrame to a CSV file")
    args = parser.parse_args(argv)

    df = parse_results(args.results)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(f"{len(df)} runs:")
        print(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
