"""Parse pre-filtered run logs into a pandas DataFrame for seaborn.

Input is the output of, e.g.

    grep "cd build/Foil\\|cd build/Kelvin\\|bin/picongpu \\|calculation" output/hal-sleeptimes/run_*

i.e. lines of the form `<log-file>:<content>` where `<content>` is one of

    + cd <path>/build/<Example>[/<policy-sleep<N>>]
    + MALLOCMC_SLEEP_TIME=<N>
    [+ MALLOCMC_SLEEP_TIME=<N>] bin/picongpu <flags>
    calculation  simulation time:  ... = <t> sec

The first form sets the current example (and, for the old per-variant
directories, the policy and sleep_time as well); the second, third, and
fourth forms mark a run. In the current log layout the `set -x` trace puts
the `MALLOCMC_SLEEP_TIME=<N>` prefix on its own line before `bin/picongpu`,
so the sleep time is taken from that line (or from the prefix when
`bin/picongpu` is on the same line, or from the variant directory in the old
layout); the run's flags are parsed (grid from `-g`, steps from `-s`); the
last form closes the most recent run with its wall time. One DataFrame row
is produced per picongpu run.

For the current log layout (one build per example, sleep time set via the
`MALLOCMC_SLEEP_TIME` environment variable on its own trace line), the `cd`
lines are absolute paths, so use a grep that still captures them, e.g.

    grep -E "cd .*build/(FoilLCT|KelvinHelmholtz)|MALLOCMC_SLEEP_TIME=|bin/picongpu |calculation" \
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

import pandas as pd

# `<log-file>:+ <trace>` or `<log-file>:calculation ...`
# The greedy file part skips over the `:` characters inside the log file
# name (run_..._12:10:36+02:00.txt).
LINE_RE = re.compile(r"^(?P<file>.+):(?P<rest>\+.*|calculation.*)$")
CD_RE = re.compile(r"^\+ cd (?P<path>\S+)$")
BUILD_PATH_RE = re.compile(r"(?:^|/)(?:build/(?P<example>\w+)(?:/(?P<variant>[^/]+))?)$")
VARIANT_RE = re.compile(r"^(?P<policy>\w+)-sleep(?P<sleep>\d+)$")
ALLOCATION_RE = re.compile(r"^Using allocator: (?P<policy>\w+), sleep_time: (?P<sleep>\d+) ns$")
PICONGPU_RE = re.compile(r"^\+(?P<prefix>.*)bin/picongpu (?P<flags>.*)$")
SLEEP_ENV_RE = re.compile(r"MALLOCMC_SLEEP_TIME=(\d+)")
# `set -x` traces the MALLOCMC_SLEEP_TIME prefix on its own line, before the
# picongpu line.
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
    "time_seconds",
]


def parse_results(source) -> pd.DataFrame:
    """Parse the pre-filtered run log `source` (path or line iterable).

    Returns a DataFrame with one row per picongpu run; see the module docstring
    for the expected input format and the columns. `sleep_time` is in
    nanoseconds, `time_seconds` is the simulation wall time in seconds.
    """
    ctx = {"example": None, "policy": None, "sleep_time": None, "sleep_env": None}
    pending = None
    rows = []

    lines = source if hasattr(source, "readline") or isinstance(source, (list, tuple)) else open(source, "r")
    for raw in lines:
        line = raw.strip()
        m = LINE_RE.match(line)
        file, rest = (m["file"], m["rest"]) if m else (None, line)

        if rest.startswith("calculation"):
            cm = CALC_RE.match(rest)
            if cm is None:
                print(f"parse_results: unrecognised calculation line: {line}", file=sys.stderr)
            elif pending is None:
                print(f"parse_results: calculation without a preceding run (dropped): {line}", file=sys.stderr)
            else:
                rows.append({"file": file, **pending, "time_seconds": float(cm["time"])})
            pending = None
            continue

        cm = CD_RE.match(rest)
        if cm:
            bm = BUILD_PATH_RE.search(cm["path"])
            if bm:
                ctx["example"] = bm["example"]
                vm = VARIANT_RE.match(bm["variant"]) if bm["variant"] else None
                ctx["policy"] = vm["policy"] if vm else None
                ctx["sleep_time"] = float(vm["sleep"]) if vm else None
                # A new build context invalidates a remembered run-time env.
                ctx["sleep_env"] = None
            continue

        cm = SLEEP_ENV_ASSIGN_RE.match(rest)
        if cm:
            # `set -x` traces the MALLOCMC_SLEEP_TIME prefix on its own line
            # before the picongpu line.
            ctx["sleep_env"] = float(cm["sleep"])
            continue

        cm = ALLOCATION_RE.match(rest)
        if cm:
            ctx["policy"] = cm["policy"]
            ctx["sleep_time"] = float(cm["sleep"])
            continue

        cm = PICONGPU_RE.match(rest)
        if cm:
            env = SLEEP_ENV_RE.search(cm["prefix"])
            grid = GRID_RE.search(cm["flags"])
            steps = STEPS_RE.search(cm["flags"])
            if grid is None or steps is None or ctx["example"] is None:
                print(f"parse_results: incomplete run (needs cd, -g and -s context): {line}", file=sys.stderr)
                continue
            gdims = [int(x) for x in grid["grid"].split()]
            if env:
                # Inline prefix on the picongpu line.
                sleep_time = float(env.group(1))
            elif ctx["sleep_env"] is not None:
                # Standalone assignment line of the current run.
                sleep_time = ctx["sleep_env"]
            else:
                # Per-variant layout: compiled into the binary.
                sleep_time = ctx["sleep_time"]
            pending = {
                "example": ctx["example"],
                "grid": "x".join(map(str, gdims)),
                "grid_x": gdims[0],
                "grid_y": gdims[1] if len(gdims) > 1 else 1,
                "grid_z": gdims[2] if len(gdims) > 2 else 1,
                "steps": int(steps["steps"]),
                "policy": ctx["policy"],
                "sleep_time": sleep_time,
            }
            continue

    if pending is not None:
        print("parse_results: run without a calculation line (dropped)", file=sys.stderr)
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> int:
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
