"""Compute all the benchmark numbers from the run logs into one HDF5 file.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The single "numbers" entry point: it parses the run logs of every machine
(the `machines` table of `config.json` plus the legacy per-cluster output
directories below, whose labels and hardware titles are `LEGACY_HARDWARE`)
into one runs table, and computes from it

- `group_stats`: the runtime description of every (machine, setup,
  algorithm, grid, delay) group,
- `fits`: the Amdahl fit of every (machine, setup, algorithm, grid) sweep
  (the per-fit parameter vector and covariance are stored under
  `fits/cov/`),
- `baselines`: the zero-delay runtime IQR of every
  (machine, setup, grid, algorithm) group,
- `foil` / `foil_pvalue` / `khi`: the statistics behind the FoilLCT bar
  chart and the KelvinHelmholtz violin chart (distributions, Kruskal
  p-values, relative runtimes).

Everything is written to `output/results.h5`; this script prints nothing.
Print the tables with `summarize_results.py`, draw the figures with the
`plot_*.py` scripts (or run `make`).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- used to read the git commit
from datetime import UTC, datetime
from pathlib import Path

import amdahl
import numpy as np
import pandas as pd
from results_io import (
    RESULTS,
    RUN_TIME,
    grid_label,
    no_delay_mask,
    particle_memory_gb,
    write_results,
)
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY, parse_logs
from scipy.stats import kruskal

REFERENCE_ALGORITHM = "ScatterAlloc"
# The two algorithms the paper figures compare for their significance tests.
PAPER_ALGORITHMS = ("FlatterScatter", "ScatterAlloc")
# The legacy per-cluster output directories the old produce_figures.py read
# from (directory name under `output/` -> hardware title); the machines
# table of config.json takes precedence for the directories it names.
LEGACY_HARDWARE = {
    "hal": "A30",
    "hemera": "A100",
    "hemera-a100": "A100",
    "hemera-v100": "V100",
    "lumi": "MI250X (1 GCD)",
    "jedi": "GH200",
    "hal-sleeptimes": "A30",
    "rosi-sleeptimes": "V100",
}

RUNS_COLUMNS = ["machine", "hardware", *GROUP_KEYS, MALLOC_DELAY, FREE_DELAY, "configuration", RUN_TIME, "rep"]
GROUP_STATS_COLUMNS = [
    "machine",
    *GROUP_KEYS,
    MALLOC_DELAY,
    FREE_DELAY,
    "configuration",
    "count",
    "mean",
    "std",
    "min",
    "p25",
    "p50",
    "p75",
    "max",
]
FITS_COLUMNS = [
    "machine",
    *GROUP_KEYS,
    "n_runs",
    "model",
    "W",
    "T0",
    "r2",
    "N_malloc",
    "A_malloc",
    "m0_ns",
    "f_malloc",
    "f_malloc_err",
    "N_free",
    "A_free",
    "f0_ns",
    "f_free",
    "f_free_err",
    "note",
]
BASELINES_COLUMNS = ["machine", "setup", "x", "y", "z", "algorithm", "count", "p25", "p50", "p75"]
FOIL_COLUMNS = ["hardware", "algorithm", "n", "p25", "p50", "p75"]
FOIL_PVALUE_COLUMNS = ["hardware", "kruskal_p"]
KHI_COLUMNS = ["hardware", "memory_gb", "reference_runtime", "outlier_count", "kruskal_p", "flatter_median_relative"]


def repo_root() -> Path:
    """Return the repository root (the parent of the `analysis/` directory).

    Returns:
        Path: the repository root.

    """
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load the harness configuration.

    Returns:
        dict: the parsed `config.json`.

    """
    with (repo_root() / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_machines(config: dict) -> dict[str, tuple[list[Path], str]]:
    """Resolve the machines to analyze: label -> (run-log dirs, hardware title).

    The machines are the union of the config.json `machines` table (label:
    the table key, hardware: its title) and the legacy per-cluster output
    directories (label: the directory name). A directory listed by both is
    analyzed once, under the config machine's label and title, and a legacy
    directory that shares its label with a config machine merges its logs
    with that machine's.

    Args:
        config: the parsed `config.json`.

    Returns:
        dict[str, tuple[list[Path], str]]: label -> (run-log directories,
        hardware title), sorted by label.

    """
    root = repo_root()
    entries: dict[Path, tuple[str, str, bool]] = {}
    for dir_name, hardware in LEGACY_HARDWARE.items():
        entries.setdefault(root / "output" / dir_name, (dir_name, hardware, False))
    for key, machine in config["machines"].items():
        entries[root / machine["output"]] = (key, machine["hardware"], True)
    machines: dict[str, dict] = {}
    for log_dir, (label, hardware, is_config) in entries.items():
        if not log_dir.is_dir():
            continue
        entry = machines.setdefault(label, {"dirs": [], "hardware": hardware})
        entry["dirs"].append(log_dir)
        if is_config:
            entry["hardware"] = hardware
    return {label: (entry["dirs"], entry["hardware"]) for label, entry in sorted(machines.items())}


def read_all_runs(machines: dict[str, tuple[list[Path], str]]) -> pd.DataFrame:
    """Parse every machine's run logs into one runs table.

    Args:
        machines: label -> (run-log directories, hardware title), as from
        `load_machines`.

    Returns:
        pd.DataFrame: one row per parsed picongpu run, tagged with the
        machine's label and hardware title; `rep` numbers the repetitions
        of each (machine, setup, algorithm, grid, delay) group in file
        order.

    """
    frames = []
    for label, (dirs, hardware) in machines.items():
        log_paths = [path for d in dirs for path in sorted(d.glob("*")) if path.is_file()]
        if not log_paths:
            continue
        frame = parse_logs(log_paths)
        if frame.empty:
            continue
        frame = frame.drop(columns=["name"]).rename(columns={"runtime in s": RUN_TIME})
        frame["machine"] = label
        frame["hardware"] = hardware
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=RUNS_COLUMNS)
    runs = pd.concat(frames, ignore_index=True)
    runs["rep"] = runs.groupby(["machine", *GROUP_KEYS, MALLOC_DELAY, FREE_DELAY], dropna=False, sort=False).cumcount()
    return runs[RUNS_COLUMNS]


def _describe_grouped(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Describe the runtime of every `keys` group of a runs table.

    Args:
        frame: the rows to describe.
        keys: the group by columns.

    Returns:
        pd.DataFrame: one row per group with count, mean, std, min, p25,
        p50, p75, max of `RUN_TIME`.

    """
    stats = frame.groupby(keys, dropna=False)[RUN_TIME].describe()
    stats = stats.rename(columns={"25%": "p25", "50%": "p50", "75%": "p75"}).reset_index()
    stat_columns = ["count", "mean", "std", "min", "p25", "p50", "p75", "max"]
    return stats[keys + stat_columns]


def group_runtime_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the runtime description of every (machine, setup, algorithm, grid, delay) group.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per group with the runtime description.

    """
    if runs.empty:
        return _empty_table(GROUP_STATS_COLUMNS)
    keys = [
        column
        for column in ["machine", *GROUP_KEYS, MALLOC_DELAY, FREE_DELAY, "configuration"]
        if column in runs.columns
    ]
    return _describe_grouped(runs, keys)


def _to_ns(value: float) -> float:
    """Convert a seconds value to nanoseconds, keeping NaN.

    Args:
        value: the value in seconds.

    Returns:
        float: the value in nanoseconds.

    """
    value = float(value)
    return float("nan") if math.isnan(value) else value * 1e9


def _fit_cov(res: dict) -> tuple | None:
    """Extract the fitted parameter vector and its covariance for the bootstrap sleeves.

    Args:
        res: the result of `amdahl.fit_1d` / `amdahl.fit_2d`.

    Returns:
        tuple | None: (fit_params, pcov) for the bootstrap sleeves, or None
        if unavailable.

    """
    return (res["fit_params"], res["pcov"]) if res["fit_params"] is not None else None


def fit_one_group(m: pd.Series, f: pd.Series, runtimes: pd.Series, c_a: float | None) -> dict:
    """Fit one (setup, algorithm, grid) group and map the result onto a row.

    Groups spanning both delays get the two-operation model; groups spanning
    one delay fall back to the 1-D model on that delay.

    Args:
        m: the group's malloc sleeptimes in nanoseconds.
        f: the group's free sleeptimes in nanoseconds.
        runtimes: the group's runtimes in seconds.
        c_a: the A = N*c constraint in nanoseconds, or None.

    Returns:
        dict: the fitted row fields (model, parameters, note, cov).

    """
    if m.nunique() >= 2 and f.nunique() >= 2:
        res = amdahl.fit_2d(m, f, runtimes)
        return {
            "model": "2d",
            "W": res["W"],
            "T0": res["T0"],
            "r2": res["r2"],
            "N_malloc": res["N_m"],
            "A_malloc": res["A_m"],
            "m0_ns": _to_ns(res["m0"]),
            "f_malloc": res["f_malloc"],
            "f_malloc_err": res["f_malloc_err"],
            "N_free": res["N_f"],
            "A_free": res["A_f"],
            "f0_ns": _to_ns(res["f0"]),
            "f_free": res["f_free"],
            "f_free_err": res["f_free_err"],
            "note": "; ".join(res["warnings"]),
            "cov": _fit_cov(res),
        }
    if m.nunique() >= 2 or f.nunique() >= 2:
        varying = "malloc" if m.nunique() >= 2 else "free"
        delays = m if varying == "malloc" else f
        res = amdahl.fit_1d(delays, runtimes, c_a=c_a)
        row = {
            "model": f"1d-{varying}",
            "W": res["W"],
            "T0": res["T0"],
            "r2": res["r2"],
        }
        if varying == "malloc":
            row.update(
                {
                    "N_malloc": res["N"],
                    "A_malloc": res["A"],
                    "m0_ns": _to_ns(res["s0"]),
                    "f_malloc": res["f"],
                    "f_malloc_err": res["f_err"],
                }
            )
        else:
            row.update(
                {
                    "N_free": res["N"],
                    "A_free": res["A"],
                    "f0_ns": _to_ns(res["s0"]),
                    "f_free": res["f"],
                    "f_free_err": res["f_err"],
                }
            )
        notes = list(res["warnings"])
        notes.append(f"only the {varying} delay varies; fitted the 1-D model on it")
        row["note"] = "; ".join(notes)
        row["cov"] = _fit_cov(res)
        return row
    return {"note": "fewer than 2 distinct delays in each operation"}


def fit_sweep(runs: pd.DataFrame, c_a: float | None = None) -> tuple[pd.DataFrame, list[tuple[tuple, tuple, tuple]]]:
    """Fit every (machine, setup, algorithm, grid) group of a runs table.

    Groups whose runs span both the malloc and the free delay are fitted
    with the two-operation model of `amdahl.fit_2d`; groups spanning only
    one delay fall back to the 1-D model of `amdahl.fit_1d` on that delay.

    Args:
        runs: the parsed runs table.
        c_a: the A = N*c constraint in nanoseconds, or None.

    Returns:
        tuple: (the fits table, the (key, fit_params, pcov) entries to store
        under `fits/cov/`).

    """
    if runs.empty:
        return _empty_table(FITS_COLUMNS), []
    no_fit = {
        "model": None,
        "W": np.nan,
        "T0": np.nan,
        "r2": np.nan,
        "N_malloc": np.nan,
        "A_malloc": np.nan,
        "m0_ns": np.nan,
        "f_malloc": np.nan,
        "f_malloc_err": np.nan,
        "N_free": np.nan,
        "A_free": np.nan,
        "f0_ns": np.nan,
        "f_free": np.nan,
        "f_free_err": np.nan,
        "note": None,
    }
    rows = []
    covs = []
    for key, frame in runs.groupby(["machine", *GROUP_KEYS], dropna=False):
        grp = frame.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
        row = {
            "machine": key[0],
            **dict(zip(GROUP_KEYS, key[1:], strict=True)),
            "n_runs": len(grp),
            **no_fit,
        }
        try:
            row.update(fit_one_group(grp[MALLOC_DELAY], grp[FREE_DELAY], grp[RUN_TIME], c_a))
        except ValueError as err:
            row.update({**no_fit, "note": str(err)})
        cov = row.pop("cov", None)
        if cov is not None:
            covs.append(((key[0], key[1], key[2], grid_label(key[3], key[4], key[5])), cov[0], cov[1]))
        rows.append(row)
    return pd.DataFrame(rows)[FITS_COLUMNS], covs


def baseline_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the zero-delay runtime IQR of every (machine, setup, grid, algorithm) group.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per group with count, p25, p50, p75 of the
        zero-delay runtimes.

    """
    base = runs[no_delay_mask(runs)]
    if base.empty:
        return _empty_table(BASELINES_COLUMNS)
    stats = _describe_grouped(base, BASELINES_COLUMNS[:6])
    return stats[BASELINES_COLUMNS]


def _no_delay(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the baseline (no-delay) subset of a runs table.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: the no-delay rows.

    """
    return runs[no_delay_mask(runs)]


def foil_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the no-delay FoilLCT runtime distribution, per (hardware, algorithm).

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per (hardware, algorithm) with n, p25, p50,
        p75 of the no-delay FoilLCT runtimes.

    """
    foil = _no_delay(runs)
    foil = foil[foil["setup"] == "FoilLCT"]
    if foil.empty:
        return _empty_table(FOIL_COLUMNS)
    stats = _describe_grouped(foil, ["hardware", "algorithm"])
    return stats[["hardware", "algorithm", "count", "p25", "p50", "p75"]].rename(columns={"count": "n"})


def compute_significance(frame: pd.DataFrame, column: str) -> pd.Series:
    """Compute the Kruskal p-value of `column` across algorithms, per hardware group.

    Args:
        frame: the rows to test.
        column: the value column.

    Returns:
        pd.Series: the per-(hardware) p-values of the across-algorithm
        Kruskal test (NaN when fewer than two algorithms are present).

    """
    rows = []
    for hardware, group in frame.groupby("hardware", dropna=False):
        samples = group.groupby("algorithm")[column].agg(list).to_numpy()
        pvalue = (
            float(kruskal(*[np.asarray(s) for s in samples], nan_policy="omit").pvalue)
            if len(samples) >= 2
            else float("nan")
        )
        rows.append((hardware, pvalue))
    return pd.Series([p for _h, p in rows], index=pd.Index([h for h, _p in rows], name="hardware"))


def foil_pvalues(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the Kruskal significance of the FoilLCT bar chart, per hardware.

    The test compares the two paper algorithms (FlatterScatter vs.
    ScatterAlloc) over all no-delay FoilLCT runs of a hardware.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per hardware with the Kruskal p-value.

    """
    foil = _no_delay(runs)
    foil = foil[(foil["setup"] == "FoilLCT") & foil["algorithm"].isin(PAPER_ALGORITHMS)]
    if foil.empty:
        return _empty_table(FOIL_PVALUE_COLUMNS)
    pvalues = compute_significance(foil, RUN_TIME)
    return pd.DataFrame({"hardware": pvalues.index, "kruskal_p": pvalues.to_numpy()})


def _tukey_outlier_count(values: pd.Series, safety_factor: float = 1.5) -> int:
    """Count the values outside the Tukey fences of one group.

    Args:
        values: the values of the group.
        safety_factor: the IQR multiple of the Tukey fences.

    Returns:
        int: the number of outliers of the group.

    """
    if len(values) < 2:
        return 0
    desc = values.describe()
    low = desc["25%"] - safety_factor * (desc["75%"] - desc["25%"])
    high = desc["75%"] + safety_factor * (desc["75%"] - desc["25%"])
    return int(np.sum((values < low) | (values > high)))


def khi_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the statistics behind the KelvinHelmholtz violin chart.

    Per (hardware, memory): the reference runtime (median ScatterAlloc
    runtime), the outlier count of the two paper algorithms (Tukey), the
    Kruskal p-value of the relative runtime across them, and the median
    relative runtime of FlatterScatter.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per (hardware, memory) with the chart's
        metadata.

    """
    khi = _no_delay(runs)
    khi = khi[khi["setup"] == "KelvinHelmholtz"]
    if khi.empty:
        return _empty_table(KHI_COLUMNS)
    khi = khi.assign(memory_gb=[particle_memory_gb(x, y, z) for x, y, z in khi[["x", "y", "z"]].to_numpy()])
    reference = khi[khi["algorithm"] == REFERENCE_ALGORITHM].groupby(["hardware", "memory_gb"])[RUN_TIME].median()
    khi["relative runtime"] = (
        khi[RUN_TIME] / reference.reindex(khi.set_index(["hardware", "memory_gb"]).index).to_numpy()
    )
    rows = []
    for (hardware, memory), group in khi.groupby(["hardware", "memory_gb"], dropna=False):
        paper = group[group["algorithm"].isin(PAPER_ALGORITHMS)]
        pvalues = compute_significance(paper, "relative runtime")
        outliers = sum(_tukey_outlier_count(values) for _algorithm, values in paper.groupby("algorithm")[RUN_TIME])
        flatter = group.loc[group["algorithm"] == "FlatterScatter", "relative runtime"]
        rows.append(
            {
                "hardware": hardware,
                "memory_gb": memory,
                "reference_runtime": float(reference.get((hardware, memory), float("nan"))),
                "outlier_count": outliers,
                "kruskal_p": float(pvalues.get(hardware, float("nan"))),
                "flatter_median_relative": float(flatter.median()) if len(flatter) else float("nan"),
            }
        )
    return pd.DataFrame(rows)[KHI_COLUMNS]


def _empty_table(columns: list[str]) -> pd.DataFrame:
    """Build an empty results table with the right column names.

    Args:
        columns: the column names.

    Returns:
        pd.DataFrame: an empty frame with the given columns.

    """
    return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})


def _git_commit() -> str:
    """Return the short git commit of the repository, empty when unavailable.

    Returns:
        str: the short commit hash, or "".

    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except OSError, subprocess.CalledProcessError:
        return ""


def main(output: Path, configuration: str | None = None) -> None:
    """Parse all the run logs, compute every table, and write the results file.

    Prints nothing; write the tables to `output` (an HDF5 file). With
    `configuration`, the statistics and the fits are computed only from
    runs of that configuration (the runs table itself always stays
    complete).

    Args:
        output: the destination results file, e.g. `output/results.h5`.
        configuration: "run-time" or "compile-time", or None for all runs.

    """
    config = load_config()
    machines = load_machines(config)
    runs = read_all_runs(machines)
    fit_covs: list[tuple[tuple, tuple, tuple]] = []
    if runs.empty:
        tables = {
            "runs": runs,
            "group_stats": _empty_table(GROUP_STATS_COLUMNS),
            "fits": _empty_table(FITS_COLUMNS),
            "baselines": _empty_table(BASELINES_COLUMNS),
            "foil": _empty_table(FOIL_COLUMNS),
            "foil_pvalue": _empty_table(FOIL_PVALUE_COLUMNS),
            "khi": _empty_table(KHI_COLUMNS),
        }
    else:
        analyzed = runs if configuration is None else runs[runs["configuration"] == configuration]
        tables = {
            "runs": runs,
            "group_stats": group_runtime_stats(analyzed),
            "baselines": baseline_stats(runs),
            "foil": foil_stats(runs),
            "foil_pvalue": foil_pvalues(runs),
            "khi": khi_stats(runs),
        }
        tables["fits"], fit_covs = fit_sweep(analyzed)
    attrs = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "sources": ", ".join(
            f"{label}: {'; '.join(str(d) for d in dirs)}" for label, (dirs, _hardware) in machines.items()
        )
        or "none",
        "algorithm_order": ",".join(str(algorithm) for algorithm in config.get("algorithms", [])),
    }
    write_results(output, tables, attrs, fit_covs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute all the benchmark numbers from the run logs into output/results.h5 (prints nothing)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS,
        help="destination results file (default: %(default)s)",
    )
    parser.add_argument(
        "--configuration",
        choices=["run-time", "compile-time"],
        default=None,
        help="compute the statistics and the fits only from runs of that configuration (default: all of them)",
    )
    args = parser.parse_args()
    main(args.output, args.configuration)
