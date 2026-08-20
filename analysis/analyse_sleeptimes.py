from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

ALGORITHM = "FlatterScatter"
CD_CMD = "+ cd "
RUN_CMD = "bin/picongpu "
SLEEP_CMD = "MALLOCMC_SLEEP_TIME="
# `cd` trace lines that carry run context: the old per-variant layout
# (`cd build/<Example>/<Algorithm>-sleep<N>`) and the current one-build-per
# example layout (`cd .../build/<Example>`); every other `cd` (for example
# the `cd $WD` return in run_folder.sh) is ignored.
VARIANT_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)-sleep(\d+)$")
BUILD_CD_RE = re.compile(r"(?:^|/)build/(\w+)$")
LOG_PATHS = (Path("output") / "hal-sleeptimes").glob("run_*")
# The statistics and plot are computed only for runs with this configuration:
# "run-time" (delay injected via MALLOCMC_SLEEP_TIME) or "compile-time"
# (per-variant builds); None plots all of them. The parsed results are
# always complete.
CONFIGURATION = None


def parse_setup(line: str):
    # Run context of a `cd` trace line, or None if the line is unrelated.
    path = line.split()[-1]
    m = VARIANT_CD_RE.search(path)
    if m:
        # One build per (example, algorithm, sleeptime): the delay was
        # compiled into the binary.
        return {"setup": m[1], "algorithm": m[2], "sleeptime": int(m[3]), "configuration": "compile-time"}
    m = BUILD_CD_RE.search(path)
    if m:
        return {"setup": m[1], "algorithm": ALGORITHM}
    return None


def parse_grid(line: str):
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            line.split(RUN_CMD, 1)[1].split("-g", 1)[1].split("-")[0].strip().split(" "),
        )
    }


def parse_simulation_time(line: str):
    return {"runtime in s": float(line.split("=")[1][: -len("sec")])}


def parse_log(log_path: Path):
    with log_path.open("r") as file:
        context = {}
        pending = None
        sleep = None
        for line in map(str.strip, file):
            if line.startswith(CD_CMD):
                # A new run context invalidates a remembered sleep time.
                setup = parse_setup(line)
                if setup is not None:
                    context = setup
                    sleep = None
            elif line.startswith("+ "):
                # With `set -x`, the MALLOCMC_SLEEP_TIME prefix is traced on
                # its own line before the picongpu line.
                if SLEEP_CMD in line:
                    sleep = int(line.split(SLEEP_CMD, 1)[1].split()[0])
                if RUN_CMD in line and "setup" in context:
                    # In the run-time layout the env var overrides the variant
                    # sleeptime; in the per-variant layout it is absent.
                    pending = dict(context) | parse_grid(line)
                    if sleep is not None:
                        pending |= {"sleeptime": sleep, "configuration": "run-time"}
            elif line.startswith("calculation") and "simulation time" in line and pending is not None:
                yield {**pending, **parse_simulation_time(line)}
                pending = None


def run_to_df(run: dict):
    return pd.DataFrame(run["runs"]).assign(name=run["name"])


def runs_to_df(runs: Iterable[dict]):
    return pd.concat(map(run_to_df, runs))


def parse_logs(log_paths: Iterable[Path]):
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)


def simple_statistics(full_results: pd.DataFrame):
    return full_results.groupby(list(set(full_results.columns) - {"runtime in s", "name"}), dropna=False).apply(
        lambda df: df["runtime in s"].describe(), include_groups=False
    )


def label(info):
    grid_string = "x".join(map(str, map(int, np.asarray(info[1:])[~np.isnan(info[1:])])))
    return f"{info[0]} {grid_string}"


def simple_plot(simple_results: pd.DataFrame):
    results = simple_results.groupby(["setup", "x", "y", "z"], dropna=False)
    fig, ax = plt.subplots(1, 1)
    for name, result in results:
        x, ye_min, y, ye_max = result.reset_index(drop=False)[["sleeptime", "25%", "50%", "75%"]].to_numpy().T
        (line,) = ax.plot(x, y, linestyle="-", alpha=0.3)
        ax.errorbar(
            x, y, yerr=(y - ye_min, ye_max - y), linestyle="none", marker="o", color=line.get_color(), label=label(name)
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    return fig


def main(log_paths: Iterable[PathLike]):
    full_results = parse_logs(map(Path, log_paths))
    print(full_results)
    if CONFIGURATION is not None:
        # The parsed data is complete; only the plotted subset is filtered.
        full_results = full_results[full_results["configuration"] == CONFIGURATION]
    simple_results = simple_statistics(full_results)
    print(simple_results)
    _ = simple_plot(simple_results)
    plt.show()


if __name__ == "__main__":
    main(LOG_PATHS)
