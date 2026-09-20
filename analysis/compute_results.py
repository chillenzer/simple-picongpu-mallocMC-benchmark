"""Compute all the benchmark numbers from the run logs into one HDF5 file.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The single "numbers" entry point. The runs live in two sources: the sweep
machine output directories of the `machines` table of `config.json` (the
new-format logs, each carrying a `# metadata:` JSON line in its header)
and the frozen legacy table `legacy/legacy_results.h5` (the pre-redesign
logs of legacy/, with their machine and paper-figure hardware attribution
applied at the freeze; the archived-but-excluded runs are marked by an
empty hardware name and dropped here at the single exclusion choke
point). The native per-call allocation costs of the microbenchmark suite
(the microbenchmarks/memmansurvey submodule) come from a third, optional
source: the frozen table under `microbench.frozen` of `config.json`,
built with `analysis/make_microbench.py` (`make microbench-results`) from
the raw CSVs under `microbench.data` (see microbenchmarks/README.md);
when the frozen table is absent (a fresh checkout without the machine
data), the analysis runs without it and `alloc_cost` comes out empty.
Superseded vintages (an append-only re-run of a stamped series)
are not filtered out of the table: they stay in every table and every
number, and the group statistics and the fits include them; a consumer
that wants the current state selects `runs[runs["superseded"] == 0]`. The
table is grouped by the short hardware name (e.g. `A30`, `V100`) the
comparison charts have always used, and computed from

- `group_stats`, `fits`, `baselines`, `absorption`: the group runtime
  descriptions, the performance-model fit (the per-fit parameter vector and
  covariance are stored under `fits/cov/`), the zero-delay runtime IQR of
  every group of the sweep machines, and the per-arm absorbed-delay slack
  (the plateau deficit d and the per-call c = d/N, computed from the raw
  runs; both are gauge-invariant),
- `foil` / `foil_pvalue` / `khi`: the statistics behind the FoilLCT bar
  chart and the KelvinHelmholtz violin chart (distributions, Kruskal
  p-values, relative runtimes), over the zero-delay runs of both
  sources, grouped by the short hardware name,
- `alloc_cost`: the native per-call allocation costs of the microbenchmark
  suite, one row per (run, allocator, operation, allocation size) in
  milliseconds per operation (the frozen table as-is; empty when the
  frozen table is absent).

Everything is written to `output/results.h5`; this script prints nothing.
Print the tables with `summarize_results.py`, draw the figures with the
`plot_*.py` scripts (or run `make`).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- used to read the git commit
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import performance_model
from make_microbench import ALLOC_COST_COLUMNS, ALLOC_COST_MIXED_COLUMNS, ALLOC_COST_SCALING_COLUMNS
from results_io import (
    RESULTS,
    RUN_DIMENSION_COLUMNS,
    RUN_METRIC_COLUMNS,
    RUN_NOMINAL_COLUMNS,
    RUN_SOURCE_COLUMNS,
    RUN_TIME,
    RUN_VINTAGE_COLUMNS,
    grid_label,
    load_results,
    no_delay_mask,
    particle_memory_gb,
    read_attrs,
    read_table,
    scenario_key,
    write_results,
)
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY, LegacyLogError, parse_logs
from scipy.stats import kruskal

REFERENCE_ALGORITHM = "ScatterAlloc"
# The two algorithms the paper figures compare for their significance tests.
PAPER_ALGORITHMS = ("FlatterScatter", "ScatterAlloc")
# The machine-level fit/baseline/absorption grouping keys: the group key
# plus the two benchmark dimension columns (`dep_commit`, `config`), so the
# sweep-machine numbers are per (commit, config) and the shared fit is per
# (commit, config, scenario).
GROUP_KEYS_MACHINE_DIM = ["machine", "dep_commit", "config", *GROUP_KEYS]
# The frozen legacy runs (the legacy cut; see legacy/README.md): the single
# source of the pre-redesign runs. Built once with `make legacy-results`.
LEGACY_H5 = Path(__file__).resolve().parent.parent / "legacy" / "legacy_results.h5"

# Beyond the run's own numbers, every row carries the two timing metrics
# the pre-redesign record dropped (`RUN_METRIC_COLUMNS`, after the runtime and
# before the repetition number), the provenance of the log file the run
# came from (`RUN_SOURCE_COLUMNS`), the repetition number the run declared
# (`RUN_NOMINAL_COLUMNS`), and the run's vintage state
# (`RUN_VINTAGE_COLUMNS`); the single source of truth for all of them is
# analysis/results_io.py.
RUNS_COLUMNS = [
    "machine",
    "hardware",
    *RUN_DIMENSION_COLUMNS,
    *GROUP_KEYS,
    MALLOC_DELAY,
    FREE_DELAY,
    "configuration",
    RUN_TIME,
    *RUN_METRIC_COLUMNS,
    "rep",
    *RUN_SOURCE_COLUMNS,
    *RUN_NOMINAL_COLUMNS,
    *RUN_VINTAGE_COLUMNS,
]
GROUP_STATS_COLUMNS = [
    "machine",
    "dep_commit",
    "config",
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
    "dep_commit",
    "config",
    *GROUP_KEYS,
    "n_runs",
    "model",
    "W",
    "W_err",
    "T0",
    "r2",
    "N_malloc",
    "N_malloc_err",
    "A_malloc",
    "m0_ns",
    "f_malloc",
    "f_malloc_err",
    "N_free",
    "N_free_err",
    "A_free",
    "f0_ns",
    "f_free",
    "f_free_err",
    # The soft comparison against the microbenchmark: the native per-call cost
    # c_a (ns) and how the fitted per-call absorbed slack (A/N) compares to it.
    "c_a_malloc_ns",
    "c_a_free_ns",
    "slack_over_native_malloc",
    "slack_over_native_free",
    "native_note",
    "note",
]
# One row per (group) of the A = N*c_a constrained (secondary) fit: the
# absorbed delays are the microbenchmark's native per-call costs (c_a) times
# the call counts, so only (W, N_malloc, N_free, m0, f0) are fitted. Present
# only for groups whose hardware has a matching microbenchmark cost.
FITS_CA_COLUMNS = [
    "machine",
    "dep_commit",
    "config",
    *GROUP_KEYS,
    "n_runs",
    "model",
    "c_a_malloc_ns",
    "c_a_free_ns",
    "W",
    "W_err",
    "T0",
    "r2",
    "N_malloc",
    "N_malloc_err",
    "A_malloc",
    "A_malloc_err",
    "m0_ns",
    "f_malloc",
    "f_malloc_err",
    "N_free",
    "N_free_err",
    "A_free",
    "A_free_err",
    "f0_ns",
    "f_free",
    "f_free_err",
    "note",
]
# One row per (group, arm): the data-pinned absorbed-delay slack. The plateau
# deficit d and the per-call slack c = d/N are computed from the raw runs (the
# baseline minus the large-delay line's intercept, over the large-delay slope),
# so they are gauge-invariant and immune to the (W, A, s0) flat direction.
ABSORPTION_COLUMNS = [
    "machine",
    "dep_commit",
    "config",
    *GROUP_KEYS,
    "arm",
    "n",
    "N",
    "d",
    "c_us",
]
# One row per (scenario, algorithm) of a combined (shared-parameter) fit;
# the shared-parameter columns are repeated on every row of the scenario.
SHARED_FITS_COLUMNS = [
    "machine",
    "dep_commit",
    "config",
    "setup",
    "x",
    "y",
    "z",
    "algorithm",
    "n_runs",
    "model",
    "W",
    "W_err",
    "N_malloc",
    "N_malloc_err",
    "N_free",
    "N_free_err",
    "r2",
    "A_malloc",
    "A_malloc_err",
    "A_free",
    "A_free_err",
    "m0_ns",
    "m0_err_ns",
    "f0_ns",
    "f0_err_ns",
    "f_malloc",
    "f_malloc_err",
    "f_free",
    "f_free_err",
    "note",
]
BASELINES_COLUMNS = [
    "machine",
    "dep_commit",
    "config",
    "setup",
    "x",
    "y",
    "z",
    "algorithm",
    "count",
    "p25",
    "p50",
    "p75",
]
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


def load_machines(config: dict) -> dict[str, dict]:
    """Resolve the sweep machines of the `machines` table of `config.json`.

    Args:
        config: the parsed `config.json`.

    Returns:
        dict: sweep machine label -> {"dir": the machine's log directory,
        "hardware": the short paper-figure hardware name (the last word of
        the machine's hardware title), "title": the config title}, in
        config order.

    """
    root = repo_root()
    sweep: dict[str, dict] = {}
    for label, machine in config["machines"].items():
        title = str(machine["hardware"])
        sweep[label] = {
            "dir": root / machine["output"],
            "hardware": title.rsplit(maxsplit=1)[-1],
            "title": title,
        }
    return sweep


def _identity_stamp_path(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    label: str,
    example: str,
    algorithm: str,
    commit: str,
    config: str,
    malloc_delay: object,
    free_delay: object,
    rep: object,
    stamps_root: Path,
) -> Path | None:
    """Return the run stamp of one run's identity, or None.

    The identity (example, algorithm, commit, config, delays, repetition)
    is taken from the log's parsed metadata columns, not its file name:
    the file name only carries the commit's short hash, and the stamp path
    is keyed on the commit's logical name (and the config) instead.

    Args:
        label: the sweep machine's label.
        example: the run's setup (the example), from `run.setup`.
        algorithm: the run's algorithm, from `run.algorithm`.
        commit: the logical commit name, from `run.commit` (empty for
            pre-multi-commit logs).
        config: the config name, from `run.config` (empty for
            pre-config logs).
        malloc_delay: the imposed malloc delay in nanoseconds (int).
        free_delay: the imposed free delay in nanoseconds (int).
        rep: the run's declared repetition number (int).
        stamps_root: the run-stamps directory (the repo root's).

    Returns:
        Path | None: the identity's stamp path (which may not exist), or
        None when the metadata does not carry a complete identity.

    """
    counts = (malloc_delay, free_delay, rep)
    if any(not isinstance(count, (int, np.integer)) or isinstance(count, bool) for count in counts):
        return None
    if not example or not algorithm:
        return None
    return (
        stamps_root
        / label
        / commit
        / example
        / algorithm
        / config
        / f"{malloc_delay}_{free_delay}"
        / f"rep-{rep}.stamp"
    )


def _superseded_flags(frame: pd.DataFrame, label: str, stamps_root: Path) -> dict[str, int]:
    """Map one machine output directory's log names to their vintage state.

    The run stamp of an identity lists the log paths of its current
    vintage (run_stamp.sh rewrites it on every re-run), so a log not
    carried by its identity's stamp is superseded. An identity without a
    stamp (fresh data, or after `make clean-runs`) has no superseded
    vintages. The frozen legacy runs live in no sweep directory and are
    superseded by nothing.

    Args:
        frame: the parsed runs frame for this machine (carries the run's
            identity columns and its log file name `name`).
        label: the sweep machine's label.
        stamps_root: the run-stamps directory (the repo root's).

    Returns:
        dict: log absolute path -> 1 (superseded by a newer vintage) or 0.

    """
    flags: dict[str, int] = {}
    root = stamps_root.parent
    for _, row in frame.iterrows():
        name = Path(row["name"])
        stamp = _identity_stamp_path(
            label,
            str(row.get("setup", "")),
            str(row.get("algorithm", "")),
            str(row.get("dep_commit", "")),
            str(row.get("config", "")),
            row.get(MALLOC_DELAY),
            row.get(FREE_DELAY),
            row.get("nominal_rep"),
            stamps_root,
        )
        if stamp is None or not stamp.is_file():
            flags[str(name)] = 0
            continue
        # The stamp lists the log paths relative to the repository root
        # (run_stamp.sh is invoked from it); resolve them against the root
        # so they compare to the parsed absolute log paths.
        listed = {root / line.strip() for line in stamp.read_text(encoding="utf-8").splitlines() if line.strip()}
        flags[str(name)] = 0 if name in listed else 1
    return flags


def _fill_run_columns(frame: pd.DataFrame) -> None:
    """Fill the runs-table columns a parsed frame does not carry at all.

    Args:
        frame: the parsed runs frame (mutated in place).

    """
    if frame.empty:
        return
    for column in RUN_METRIC_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    for column in RUN_DIMENSION_COLUMNS:
        if column not in frame:
            frame[column] = ""
    for column in RUN_SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    for column in RUN_NOMINAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan


def read_all_runs(sweep: dict[str, dict], legacy_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse the sweep machines' run logs and the frozen legacy runs.

    Args:
        sweep: as from `load_machines`.
        legacy_frame: the frozen legacy runs (from legacy/legacy_results.h5),
            already tagged with `machine` and `hardware`, in the frozen row
            order.

    Returns:
        tuple: (the runs table, the sweep machine labels, in config order).
        A sweep machine label is kept when its log directory exists or its
        runs are in the frozen table. The runs carry `machine` (the sweep
        machine's label, empty for the legacy runs stored without one) and
        `hardware` (the short paper-figure name). `rep` numbers the
        repetitions of each (machine, setup, algorithm, grid, delay)
        group in file order (a re-run of one nominal repetition is a new
        log in the same group, so `rep` is a position among all of the
        group's logs; `nominal_rep` is the repetition the run declared),
        and `superseded` marks the runs an identity's run stamp no longer
        lists (a newer vintage of the same combination and repetition was
        written; the frozen legacy runs are all 0).

    """
    frames = []
    stamps_root = repo_root() / "run-stamps"
    for label, machine in sweep.items():
        frame = _parse_dir(machine["dir"])
        if frame.empty:
            continue
        frame["machine"] = label
        frame["hardware"] = machine["hardware"]
        flags = _superseded_flags(frame, label, stamps_root)
        frame["superseded"] = frame["name"].map(lambda p: flags.get(str(p), 0)).astype(int)
        frames.append(frame)
    if not legacy_frame.empty:
        # A frozen legacy row carries no file name and none of the
        # log-derived columns the sweep machine frames have; default them
        # (the frozen runs are superseded by nothing).
        legacy_rows = legacy_frame.copy()
        _fill_run_columns(legacy_rows)
        legacy_rows["superseded"] = 0
        frames.append(legacy_rows)
    h5_machines = set() if legacy_frame.empty else set(legacy_frame["machine"])
    sweep_labels = [label for label, machine in sweep.items() if machine["dir"].is_dir() or label in h5_machines]
    if not frames:
        return pd.DataFrame(columns=RUNS_COLUMNS), sweep_labels
    runs = pd.concat(frames, ignore_index=True)
    if "name" in runs:
        runs = runs.drop(columns=["name"])
    # The single exclusion choke point: archived-but-excluded runs carry the
    # empty hardware name and are dropped here, before the rep numbering, so
    # they neither appear in any table nor shift the rep of an analyzed run.
    runs = runs[runs["hardware"] != ""]
    runs["rep"] = runs.groupby(["machine", *GROUP_KEYS, MALLOC_DELAY, FREE_DELAY], dropna=False, sort=False).cumcount()
    return runs[RUNS_COLUMNS], sweep_labels


def read_legacy_h5(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the frozen legacy runs table and the file's top-level attributes.

    Args:
        path: the frozen legacy file (legacy/legacy_results.h5).

    Returns:
        tuple: (the runs table, already tagged with `machine` and
        `hardware`, in the frozen row order, and the file's top-level
        attributes, e.g. `excluded_sources`).

    """
    with load_results(path) as file:
        return read_table(file, "runs"), read_attrs(file)


def _parse_dir(log_dir: Path) -> pd.DataFrame:
    """Parse one output directory's run logs.

    Args:
        log_dir: the directory of run logs to parse.

    Returns:
        pd.DataFrame: the parsed runs, or an empty frame.

    Raises:
        LegacyLogError: if a file is not a new-format log (pre-redesign
            logs belong under legacy/logs/, frozen by
            legacy/move_legacy_logs.sh).

    """
    log_paths = sorted(path for path in log_dir.glob("*") if path.is_file())
    if not log_paths:
        return pd.DataFrame()
    try:
        frame = parse_logs(log_paths)
    except LegacyLogError as error:
        message = (
            f"{error}\n"
            "Pre-redesign logs are no longer parsed directly: move them to "
            "legacy/logs/ (legacy/move_legacy_logs.sh) and freeze them with "
            "`make legacy-results`."
        )
        raise LegacyLogError(message) from error
    if frame.empty:
        return frame
    # `name` (the log's absolute path) is kept: the caller derives the
    # run's vintage state from its file name and the run stamps.
    frame = frame.rename(columns={"runtime in s": RUN_TIME})
    _fill_run_columns(frame)
    return frame


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
        for column in [
            "machine",
            *RUN_DIMENSION_COLUMNS,
            *GROUP_KEYS,
            MALLOC_DELAY,
            FREE_DELAY,
            "configuration",
        ]
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
        res: the result of `performance_model.fit_1d` / `performance_model.fit_2d`.

    Returns:
        tuple | None: (fit_params, pcov) for the bootstrap sleeves, or None
        if unavailable.

    """
    return (res["fit_params"], res["pcov"]) if res["fit_params"] is not None else None


# The dominant KelvinHelmholtz allocation: the particle-frame size in bytes
# (the paper's number). It is the size at which the native per-call cost c_a
# is looked up for the A = N*c_a constraint; for the other setups the same
# size is an order-of-magnitude estimate (their allocation patterns differ).
KHI_FRAME_BYTES = 7696
# The microbenchmark operation of each model arm (malloc -> alloc, free ->
# free): the value of the alloc_cost table's `operation` column.
MICROBENCH_OPERATION = {"malloc": "alloc", "free": "free"}
# The number of parallel allocations in one wave of the perf microbenchmark:
# each thread performs exactly one allocation per wave, so the per-single-call
# cost estimate divides the wave time (the frozen table's `mean_ms`) by that
# count.
MICROBENCH_WAVE_ALLOCATIONS = 1_000_000
# Benchmark algorithm -> microbenchmark allocator names, tried in order: the
# first entry is the current-survey name (the mallocMC creation-policy name),
# the following entries are legacy fallbacks that let the pre-existing frozen
# data still resolve (it used the older single-`mallocMC` and standalone
# names). The legacy "ScatterAlloc" and "Gallatin" are the pre-mallocMC
# standalone allocators (not the mallocMC policies), while the legacy
# "mallocMC" run corresponds to FlatterScatter. A missing match leaves the
# group unconstrained.
ALLOCATOR_ALIASES: dict[str, list[str]] = {
    "ScatterAlloc": ["mallocMC-Scatter", "ScatterAlloc"],
    "FlatterScatter": ["mallocMC-FlatterScatter", "mallocMC"],
    "Gallatin": ["Gallatin"],
}


def _gpu_model(hardware: object) -> str:
    """Normalize a hardware label to its GPU model (the vendor stripped).

    Args:
        hardware: a hardware label, e.g. "NVIDIA A100" or "A100".

    Returns:
        str: the GPU model, e.g. "A100" in both cases.

    """
    value = hardware.decode("utf-8", "replace") if isinstance(hardware, (bytes, bytearray)) else str(hardware)
    tokens = value.split()
    return " ".join(token for token in tokens if token.upper() not in {"NVIDIA", "AMD", "INTEL"}).strip()


def _resolve_c_a(
    alloc_cost: pd.DataFrame, hardware: object, algorithm: str, operation: str, setup: str
) -> tuple[float | None, str]:
    """Resolve the native per-single-call cost (ns) of one (hardware, allocator, operation).

    The cost is the microbenchmark mean at the representative allocation size
    (the dominant KHI frame, the nearest measured size; an order-of-magnitude
    estimate for the other setups). The frozen table's `mean_ms` is the time of
    one wave of `MICROBENCH_WAVE_ALLOCATIONS` parallel allocations (one per
    thread), so the per-single-call cost is the wave mean divided by that
    count. Without a matching row -- no microbenchmark data for the hardware,
    or no matching allocator -- the cost is None and the note says why, so the
    fit stays unconstrained.

    Args:
        alloc_cost: the microbenchmark alloc_cost table.
        hardware: the group's hardware label (the short name).
        algorithm: the benchmark algorithm (allocator) name.
        operation: the arm's microbenchmark operation ("alloc" or "free").
        setup: the scenario's setup name (KHI is precise, the rest order-of-magnitude).

    Returns:
        tuple: (the native per-single-call cost in nanoseconds, or None, the note).

    """
    if alloc_cost is None or alloc_cost.empty:
        return None, "no microbenchmark data"
    hw_model = alloc_cost["hardware"].map(_gpu_model).eq(_gpu_model(hardware))
    op_model = alloc_cost["operation"].eq(operation)
    precision = "" if str(setup) == "KelvinHelmholtz" else " (order-of-magnitude estimate)"
    for name in [str(algorithm), *ALLOCATOR_ALIASES.get(str(algorithm), [])]:
        sel = alloc_cost[hw_model & op_model & alloc_cost["allocator"].eq(name)]
        if not sel.empty:
            idx = sel["size_bytes"].sub(KHI_FRAME_BYTES).abs().idxmin()
            return float(sel.loc[idx, "mean_ms"]) * 1e6 / MICROBENCH_WAVE_ALLOCATIONS, (
                f"microbenchmark {operation} at {int(sel.loc[idx, 'size_bytes'])} B "
                f"({name} on {str(sel.loc[idx, 'hardware']).strip()}){precision}"
            )
    have = sorted(set(alloc_cost.loc[hw_model & op_model, "allocator"].astype(str)))
    if not have:
        return None, f"no microbenchmark data for this hardware ({_gpu_model(hardware)})"
    return None, f"no microbenchmark allocator matching {algorithm!r} (have: {', '.join(have)})"


def _slack_over_native(A: float, N: float, c_a: float | None) -> float:
    """Return the fitted per-call absorbed slack (A/N) over the native cost c_a.

    Args:
        A: the fitted absorbed delay per run (s).
        N: the fitted call count per run.
        c_a: the native per-call cost (ns), or None.

    Returns:
        float: (A/N) / c_a, or NaN when either is unavailable.

    """
    if c_a is None or not np.isfinite(c_a) or c_a == 0:
        return float("nan")
    if not (np.isfinite(A) and np.isfinite(N)) or N == 0:
        return float("nan")
    return (A / N) / (c_a * 1e-9)


def fit_one_group(m: pd.Series, f: pd.Series, runtimes: pd.Series) -> dict:
    """Fit one (setup, algorithm, grid) group with the unconstrained model.

    This is the primary fit (the absorbed-slack reading). Groups spanning both
    delays get the two-operation model; groups spanning one delay fall back to
    the 1-D model on that delay. The A = N*c_a constrained (native-cost)
    reading is a separate secondary fit (`fit_one_group_ca`).

    Args:
        m: the group's malloc sleeptimes in nanoseconds.
        f: the group's free sleeptimes in nanoseconds.
        runtimes: the group's runtimes in seconds.

    Returns:
        dict: the fitted row fields (model, parameters, note, cov).

    """
    if m.nunique() >= 2 and f.nunique() >= 2:
        res = performance_model.fit_2d(m, f, runtimes)
        return {
            "model": "2d",
            "W": res["W"],
            "W_err": res["W_err"],
            "T0": res["T0"],
            "r2": res["r2"],
            "N_malloc": res["N_m"],
            "N_malloc_err": res["N_m_err"],
            "N_free": res["N_f"],
            "N_free_err": res["N_f_err"],
            "A_malloc": res["A_m"],
            "m0_ns": _to_ns(res["m0"]),
            "f_malloc": res["f_malloc"],
            "f_malloc_err": res["f_malloc_err"],
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
        res = performance_model.fit_1d(delays, runtimes)
        row = {
            "model": f"1d-{varying}",
            "W": res["W"],
            "W_err": res["W_err"],
            "T0": res["T0"],
            "r2": res["r2"],
        }
        if varying == "malloc":
            row.update(
                {
                    "N_malloc": res["N"],
                    "N_malloc_err": res["N_err"],
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
                    "N_free_err": res["N_err"],
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


def fit_one_group_ca(
    m: pd.Series,
    f: pd.Series,
    runtimes: pd.Series,
    c_malloc: float | None,
    c_free: float | None,
) -> dict | None:
    """Fit one group with the A = N*c_a constrained (secondary, native-cost) model.

    The absorbed delays are the microbenchmark's native per-call costs
    (c_malloc, c_free, in nanoseconds) times the call counts, so only the
    baseline, the call counts, and the fade scale(s) are fitted.

    Args:
        m: the group's malloc sleeptimes in nanoseconds.
        f: the group's free sleeptimes in nanoseconds.
        runtimes: the group's runtimes in seconds.
        c_malloc: the native malloc per-call cost (ns), or None.
        c_free: the native free per-call cost (ns), or None.

    Returns:
        dict | None: the constrained row fields (model, parameters, note,
        cov), or None when the constraint is not applicable (a needed cost
        is missing) or the group has too few distinct delays.

    """
    nan = float("nan")
    c_m = nan if c_malloc is None else c_malloc
    c_f = nan if c_free is None else c_free
    if m.nunique() >= 2 and f.nunique() >= 2:
        if c_malloc is None or c_free is None:
            return None
        res = performance_model.fit_2d(m, f, runtimes, c_malloc=c_malloc, c_free=c_free)
        return {
            "model": "2d-ca",
            "c_a_malloc_ns": c_m,
            "c_a_free_ns": c_f,
            "W": res["W"],
            "W_err": res["W_err"],
            "T0": res["T0"],
            "r2": res["r2"],
            "N_malloc": res["N_m"],
            "N_malloc_err": res["N_m_err"],
            "A_malloc": res["A_m"],
            "A_malloc_err": res["A_m_err"],
            "m0_ns": _to_ns(res["m0"]),
            "f_malloc": res["f_malloc"],
            "f_malloc_err": res["f_malloc_err"],
            "N_free": res["N_f"],
            "N_free_err": res["N_f_err"],
            "A_free": res["A_f"],
            "A_free_err": res["A_f_err"],
            "f0_ns": _to_ns(res["f0"]),
            "f_free": res["f_free"],
            "f_free_err": res["f_free_err"],
            "note": "; ".join(res["warnings"]),
            "cov": _fit_cov(res),
        }
    if m.nunique() >= 2 or f.nunique() >= 2:
        varying = "malloc" if m.nunique() >= 2 else "free"
        c = c_malloc if varying == "malloc" else c_free
        if c is None:
            return None
        delays = m if varying == "malloc" else f
        res = performance_model.fit_1d(delays, runtimes, c_a=c)
        row = {
            "model": f"1d-{varying}-ca",
            "c_a_malloc_ns": c_m,
            "c_a_free_ns": c_f,
            "W": res["W"],
            "W_err": res["W_err"],
            "T0": res["T0"],
            "r2": res["r2"],
            "note": "; ".join(res["warnings"]),
            "cov": _fit_cov(res),
        }
        if varying == "malloc":
            row.update(
                {
                    "N_malloc": res["N"],
                    "N_malloc_err": res["N_err"],
                    "A_malloc": res["A"],
                    "A_malloc_err": res["A_err"],
                    "m0_ns": _to_ns(res["s0"]),
                    "f_malloc": res["f"],
                    "f_malloc_err": res["f_err"],
                }
            )
        else:
            row.update(
                {
                    "N_free": res["N"],
                    "N_free_err": res["N_err"],
                    "A_free": res["A"],
                    "A_free_err": res["A_err"],
                    "f0_ns": _to_ns(res["s0"]),
                    "f_free": res["f"],
                    "f_free_err": res["f_err"],
                }
            )
        return row
    return None


def fit_sweep(
    runs: pd.DataFrame, alloc_cost: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[tuple, tuple, tuple]]]:
    """Fit every (machine, setup, algorithm, grid) group of a runs table.

    The primary fit is the unconstrained model: groups whose runs span both
    the malloc and the free delay are fitted with the two-operation model of
    `performance_model.fit_2d`, groups spanning only one delay fall back to
    the 1-D model of `performance_model.fit_1d` on that delay. The primary
    row also carries the soft comparison against the microbenchmark (the
    native per-call cost c_a and the fitted per-call absorbed slack relative
    to it). Where the microbenchmark supplies a matching cost for the
    group's hardware, a secondary A = N*c_a constrained (native-cost) fit is
    recorded in the `fits_ca` table.

    Args:
        runs: the parsed runs table.
        alloc_cost: the microbenchmark alloc_cost table (or None); the source
        of the native per-call costs. Without it (or without a matching
        hardware/allocator) the fits stay unconstrained.

    Returns:
        tuple: (the fits table, the fits_ca table, the (key, fit_params,
        pcov) entries to store under `fits/cov/`).

    """
    if runs.empty:
        return _empty_table(FITS_COLUMNS), _empty_table(FITS_CA_COLUMNS), []
    no_fit = {
        "model": None,
        "W": np.nan,
        "W_err": np.nan,
        "T0": np.nan,
        "r2": np.nan,
        "N_malloc": np.nan,
        "N_malloc_err": np.nan,
        "A_malloc": np.nan,
        "m0_ns": np.nan,
        "f_malloc": np.nan,
        "f_malloc_err": np.nan,
        "N_free": np.nan,
        "N_free_err": np.nan,
        "A_free": np.nan,
        "f0_ns": np.nan,
        "f_free": np.nan,
        "f_free_err": np.nan,
        "c_a_malloc_ns": np.nan,
        "c_a_free_ns": np.nan,
        "slack_over_native_malloc": np.nan,
        "slack_over_native_free": np.nan,
        "native_note": None,
        "note": None,
    }
    rows = []
    ca_rows = []
    covs = []
    for key, frame in runs.groupby(GROUP_KEYS_MACHINE_DIM, dropna=False):
        grp = frame.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
        row = {**dict(zip(GROUP_KEYS_MACHINE_DIM, key, strict=True)), "n_runs": len(grp), **no_fit}
        hardware = frame["hardware"].iloc[0] if len(frame["hardware"]) else None
        try:
            row.update(fit_one_group(grp[MALLOC_DELAY], grp[FREE_DELAY], grp[RUN_TIME]))
        except ValueError as err:
            row.update({**no_fit, "note": str(err)})
        c_malloc, note_m = _resolve_c_a(
            alloc_cost, hardware, str(row["algorithm"]), MICROBENCH_OPERATION["malloc"], str(row["setup"])
        )
        c_free, note_f = _resolve_c_a(
            alloc_cost, hardware, str(row["algorithm"]), MICROBENCH_OPERATION["free"], str(row["setup"])
        )
        row["c_a_malloc_ns"] = c_malloc if c_malloc is not None else np.nan
        row["c_a_free_ns"] = c_free if c_free is not None else np.nan
        row["slack_over_native_malloc"] = _slack_over_native(row["A_malloc"], row["N_malloc"], c_malloc)
        row["slack_over_native_free"] = _slack_over_native(row["A_free"], row["N_free"], c_free)
        row["native_note"] = f"{note_m}; {note_f}"
        ca_row = fit_one_group_ca(grp[MALLOC_DELAY], grp[FREE_DELAY], grp[RUN_TIME], c_malloc, c_free)
        if ca_row is not None:
            ca_row.pop("cov", None)  # the standard errors are in the table columns
            ca_row = {**dict(zip(GROUP_KEYS_MACHINE_DIM, key, strict=True)), "n_runs": len(grp), **ca_row}
            ca_rows.append(ca_row)
        cov = row.pop("cov", None)
        if cov is not None:
            covs.append(
                (
                    (
                        row["machine"],
                        row["dep_commit"],
                        row["config"],
                        row["setup"],
                        str(row["algorithm"]),
                        grid_label(row["x"], row["y"], row["z"]),
                    ),
                    cov[0],
                    cov[1],
                )
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)[FITS_COLUMNS],
        pd.DataFrame(ca_rows)[FITS_CA_COLUMNS] if ca_rows else _empty_table(FITS_CA_COLUMNS),
        covs,
    )


def _arm_absorption(x_ns: pd.Series, runtimes: pd.Series, baseline: float) -> dict:
    """One arm's plateau deficit d and per-call slack c = d/N, from the raw data.

    The large-delay line is anchored on the two largest delays; the plateau
    deficit d = baseline - intercept is gauge-invariant, and c = d/N is the
    per-call absorbed slack. Neither involves the fade scale, so both are
    immune to the (W, A, s0) flat direction of the fits.

    Args:
        x_ns: the arm's delays in nanoseconds (the non-zero-delay values).
        runtimes: the arm's runtimes in seconds.
        baseline: the (0, 0) baseline runtime median in seconds.

    Returns:
        dict: n, N (calls per run), d (s), c_us (us per call); NaNs when the
        arm has too few points to anchor the large-delay line.

    """
    x = np.asarray(x_ns, dtype=float) * 1e-9  # ns -> s
    y = np.asarray(runtimes, dtype=float)
    n = int(x.size)
    if len(np.unique(x)) < 4:
        return {"n": n, "N": np.nan, "d": np.nan, "c_us": np.nan}
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    ymed = np.array([np.median(y[x == v]) for v in ux])
    N = (ymed[-1] - ymed[-2]) / (ux[-1] - ux[-2])
    b0 = ymed[-1] - N * ux[-1]
    d = baseline - b0
    c = d / N if N > 0 else float("nan")
    return {
        "n": n,
        "N": float(N),
        "d": float(d),
        "c_us": float(c * 1e6) if np.isfinite(c) else float("nan"),
    }


def absorption_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-arm plateau deficit and per-call absorbed slack of every group.

    For each (machine, setup, algorithm, grid) group and each arm (the
    malloc-delay arm with free delay 0, the free-delay arm with malloc delay
    0), the plateau deficit d and the per-call slack c = d/N are recorded from
    the raw runs. These are the defensible Tier-2 quantities of the review:
    the delay the pipeline absorbs, not the native allocation cost.

    Args:
        runs: the parsed runs table (the sweep machines).

    Returns:
        pd.DataFrame: the `absorption` table, one row per (group, arm).

    """
    if runs.empty:
        return _empty_table(ABSORPTION_COLUMNS)
    rows = []
    for key, frame in runs.groupby(GROUP_KEYS_MACHINE_DIM, dropna=False):
        grp = frame.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
        m = np.asarray(grp[MALLOC_DELAY], dtype=float)
        f = np.asarray(grp[FREE_DELAY], dtype=float)
        t = np.asarray(grp[RUN_TIME], dtype=float)
        base = (m == 0) & (f == 0)
        baseline = float(np.median(t[base])) if base.any() else float("nan")
        for arm, mask, xcol in (("malloc", (m > 0) & (f == 0), m), ("free", (m == 0) & (f > 0), f)):
            row = {**dict(zip(GROUP_KEYS_MACHINE_DIM, key, strict=True)), "arm": arm}
            row.update(_arm_absorption(pd.Series(xcol[mask]), t[mask], baseline))
            rows.append(row)
    return pd.DataFrame(rows)[ABSORPTION_COLUMNS]


def _shared_p0(row: pd.Series) -> dict[str, float]:
    """Return the individual fit's parameters in seconds, keyed for the combined fit.

    Args:
        row: one row of the `fits` table.

    Returns:
        dict[str, float]: the parameters in seconds a combined fit accepts
        as initial values (W, N_m, N_f, A_m, A_f, m0, f0); empty when the
        row has no usable values.

    """
    if not row["model"]:
        return {}
    params = {"W": row["W"]}
    if row["model"] == "2d":
        params.update({"N_m": row["N_malloc"], "N_f": row["N_free"], "A_m": row["A_malloc"], "A_f": row["A_free"]})
    elif row["model"] == "1d-malloc":
        params.update({"N_m": row["N_malloc"], "A_m": row["A_malloc"]})
    else:
        params.update({"N_f": row["N_free"], "A_f": row["A_free"]})
    for source, target in (("m0_ns", "m0"), ("f0_ns", "f0")):
        if not pd.isna(row[source]):
            params[target] = float(row[source]) * 1e-9
    return {key: float(value) for key, value in params.items() if not pd.isna(value)}


def fit_sweep_combined(
    runs: pd.DataFrame,
    fits: pd.DataFrame,
    order: list[str],
) -> tuple[pd.DataFrame, list[tuple[tuple, tuple, tuple]]]:
    """Fit every (machine, setup, grid) scenario across all of its algorithms.

    The shared parameters -- W and the malloc/free call counts -- are fit
    once on the pooled data of the scenario's algorithms, while each
    algorithm keeps its own absorbed delays and fade scales. Scenarios with
    fewer than two algorithms (or fewer than 3 pooled runs, or no varying
    delay) get no row. The individual fits' parameters seed the initial
    guess.

    Args:
        runs: the parsed runs table (the sweep machines).
        fits: the `fit_sweep` results table.
        order: the algorithms' figure order (config.json).

    Returns:
        tuple: (the `shared_fits` table, one row per scenario and
        algorithm, the (key, fit_params, pcov) entries to store under
        `shared_fit_cov/`).

    """
    indexed: dict[tuple, pd.Series] = {
        (
            row["machine"],
            row["dep_commit"],
            row["config"],
            row["algorithm"],
            scenario_key(row["setup"], row["x"], row["y"], row["z"]),
        ): row
        for _, row in fits.iterrows()
        if pd.notna(row["model"])
    }
    rows = []
    covs = []
    scen_keys = [k for k in GROUP_KEYS_MACHINE_DIM if k != "algorithm"]
    for key, frame in runs.groupby(scen_keys, dropna=False):
        group = dict(zip(scen_keys, key, strict=True))
        machine, dep_commit, config = group["machine"], group["dep_commit"], group["config"]
        scenario = scenario_key(group["setup"], group["x"], group["y"], group["z"])
        grp = frame.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
        if grp.shape[0] < 3 or len(dict.fromkeys(grp["algorithm"])) < 2:
            continue
        if grp[MALLOC_DELAY].nunique() < 2 and grp[FREE_DELAY].nunique() < 2:
            continue
        p0 = {}
        for a in dict.fromkeys(grp["algorithm"]):
            row = indexed.get((machine, dep_commit, config, a, scenario))
            if row is not None:
                p0[a] = _shared_p0(row)
        res = performance_model.fit_combined(
            performance_model.CombinedSweep(grp[MALLOC_DELAY], grp[FREE_DELAY], grp[RUN_TIME], grp["algorithm"]),
            order=order,
            p0=p0 or None,
        )
        base = {
            **group,
            "n_runs": len(grp),
            "model": res["model"],
            "W": res["W"],
            "W_err": res["W_err"],
            "N_malloc": res["N_malloc"],
            "N_malloc_err": res["N_malloc_err"],
            "N_free": res["N_free"],
            "N_free_err": res["N_free_err"],
            "r2": res["r2"],
            "note": "; ".join(res["warnings"]) or None,
        }
        for a in res["order"]:
            d = res["per_algorithm"][a]
            rows.append(
                {
                    **base,
                    "algorithm": a,
                    "A_malloc": d["A_malloc"],
                    "A_malloc_err": d["A_malloc_err"],
                    "A_free": d["A_free"],
                    "A_free_err": d["A_free_err"],
                    "m0_ns": _to_ns(d["m0"]),
                    "m0_err_ns": _to_ns(d["m0_err"]),
                    "f0_ns": _to_ns(d["f0"]),
                    "f0_err_ns": _to_ns(d["f0_err"]),
                    "f_malloc": d["f_malloc"],
                    "f_malloc_err": d["f_malloc_err"],
                    "f_free": d["f_free"],
                    "f_free_err": d["f_free_err"],
                }
            )
        if res["fit_params"] is not None:
            covs.append(
                (
                    (
                        machine,
                        dep_commit,
                        config,
                        group["setup"],
                        grid_label(group["x"], group["y"], group["z"]),
                    ),
                    np.asarray(res["fit_params"]),
                    np.asarray(res["pcov"]),
                )
            )
    if not rows:
        return _empty_table(SHARED_FITS_COLUMNS), []
    return pd.DataFrame(rows)[SHARED_FITS_COLUMNS], covs


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
    stats = _describe_grouped(base, BASELINES_COLUMNS[:8])
    return stats[BASELINES_COLUMNS]


def _no_delay(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the baseline (zero-delay) subset of a runs table.

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: the zero-delay rows.

    """
    return runs[no_delay_mask(runs)]


def foil_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute the zero-delay FoilLCT runtime distribution, per (hardware, algorithm).

    Args:
        runs: the parsed runs table.

    Returns:
        pd.DataFrame: one row per (hardware, algorithm) with n, p25, p50,
        p75 of the zero-delay FoilLCT runtimes.

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
    ScatterAlloc) over all zero-delay FoilLCT runs of a hardware.

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


def _legacy_inputs() -> dict[str, Any]:
    """Read the pre-redesign runs from the frozen table (the legacy source).

    The source is optional: when the frozen table (legacy/legacy_results.h5)
    is absent, an empty frame is returned and the analysis runs without the
    pre-redesign runs (a fresh checkout without the machine data).

    Returns:
        dict: `frame` (the frozen runs, already tagged with `machine` and
        `hardware`, or an empty frame when the table is absent),
        `excluded_sources` (directory -> reason), `excluded_runs` (the number
        of archived rows marked excluded), and `source` (the source-list
        label, "none" when the table is absent).

    """
    if not LEGACY_H5.is_file():
        print(
            "note: legacy/legacy_results.h5 not found; the analysis runs without the "
            "pre-redesign runs. Build it from the archived logs (under legacy/logs/, "
            "created by legacy/move_legacy_logs.sh) with `make legacy-results`.",
            file=sys.stderr,
        )
        return {"frame": pd.DataFrame(), "excluded_sources": {}, "excluded_runs": 0, "source": "none"}
    frame, attrs = read_legacy_h5(LEGACY_H5)
    excluded_runs = 0 if frame.empty else int(frame["hardware"].eq("").sum())
    return {
        "frame": frame,
        "excluded_sources": json.loads(attrs.get("excluded_sources", "{}")),
        "excluded_runs": excluded_runs,
        "source": f"frozen legacy: {LEGACY_H5}",
    }


def _microbench_inputs(config: dict) -> dict[str, Any]:
    """Read the frozen microbenchmark tables (the single microbench source).

    The source is optional: when the frozen table (the `microbench.frozen`
    path of `config.json`) is absent, empty tables are returned and the
    analysis runs without the microbenchmark numbers.

    Args:
        config: the parsed configuration.

    Returns:
        dict: `frames` (the frozen microbenchmark tables, by name, or empty
        tables with the right columns when the source is absent), `source`
        (the source-list label, "none" when the source is absent), `runs`
        (the frozen jobid -> hardware map) and `protocol` (the "jobid/test"
        -> {files, operations} map).

    """
    microbench = config.get("microbench", {})
    frozen = repo_root() / str(microbench.get("frozen", ""))
    empty_frames = {
        "alloc_cost": _empty_table(ALLOC_COST_COLUMNS),
        "alloc_cost_mixed": _empty_table(ALLOC_COST_MIXED_COLUMNS),
        "alloc_cost_scaling": _empty_table(ALLOC_COST_SCALING_COLUMNS),
    }
    if not frozen.is_file():
        return {"frames": empty_frames, "source": "none", "runs": {}, "protocol": {}}
    with load_results(frozen) as file:
        frames = {name: (read_table(file, name) if name in file else empty) for name, empty in empty_frames.items()}
        attrs = read_attrs(file)
    return {
        "frames": frames,
        "source": f"frozen microbench: {frozen}",
        "runs": json.loads(attrs.get("runs", "{}")),
        "protocol": json.loads(attrs.get("protocol", "{}")),
    }


def _build_tables(  # ruff: ignore[too-many-arguments]
    *,
    runs: pd.DataFrame,
    sweep_runs: pd.DataFrame,
    configuration: str | None,
    dep_commit: str | None,
    config_name: str | None,
    micro: dict[str, pd.DataFrame],
    algorithms: list,
) -> tuple[dict[str, pd.DataFrame], list[tuple[tuple, tuple, tuple]], list[tuple[tuple, tuple, tuple]]]:
    """Build the sweep-machine analysis tables for the requested slice.

    The group statistics, fits and zero-delay baselines cover the sweep
    machines only; the paper-figure statistics (foil, khi) cover the
    zero-delay runs of both sources.

    Args:
        runs: the complete parsed runs table (all sources).
        sweep_runs: the sweep-machine rows of `runs`.
        configuration: "run-time" / "compile-time" or None for the fits.
        dep_commit: commit slice for the fits, or None.
        config_name: config slice for the fits, or None.
        micro: the microbenchmark frames.
        algorithms: the config's figure order.

    Returns:
        tuple: (the tables dict, the fit covs, the shared-fit covs).

    """
    analyzed = _slice_runs(sweep_runs, configuration, dep_commit, config_name)
    tables: dict[str, pd.DataFrame] = {
        "runs": runs,
        "group_stats": group_runtime_stats(analyzed),
        "baselines": baseline_stats(_slice_runs(sweep_runs, None, dep_commit, config_name)),
        "absorption": absorption_table(analyzed),
        "foil": foil_stats(runs),
        "foil_pvalue": foil_pvalues(runs),
        "khi": khi_stats(runs),
    }
    tables.update(micro)
    tables["fits"], tables["fits_ca"], fit_covs = fit_sweep(analyzed, micro["alloc_cost"])
    tables["shared_fits"], shared_covs = fit_sweep_combined(analyzed, tables["fits"], [str(a) for a in algorithms])
    return tables, fit_covs, shared_covs


def main(
    output: Path,
    configuration: str | None = None,
    dep_commit: str | None = None,
    config_name: str | None = None,
) -> None:
    """Parse all the run logs, compute every table, and write the results file.

    Prints nothing; write the tables to `output` (an HDF5 file). With
    `configuration`, the statistics and the fits are computed only from
    runs of that configuration (the runs table itself always stays
    complete). `dep_commit` and `config_name` restrict the statistics and
    the fits to that one commit / config slice (the runs table also stays
    complete for them).

    Args:
        output: the destination results file, e.g. `output/results.h5`.
        configuration: "run-time" or "compile-time", or None for all runs.
        dep_commit: restrict the statistics and fits to this commit name.
        config_name: restrict the statistics and fits to this config name.

    """
    config = load_config()
    sweep = load_machines(config)
    legacy_input = _legacy_inputs()
    microbench_input = _microbench_inputs(config)
    runs, sweep_labels = read_all_runs(sweep, legacy_input["frame"])
    sweep_runs = runs[runs["machine"].isin(sweep_labels)]
    fit_covs: list[tuple[tuple, tuple, tuple]] = []
    shared_covs: list[tuple[tuple, tuple, tuple]] = []
    if runs.empty:
        tables = {
            "runs": runs,
            "group_stats": _empty_table(GROUP_STATS_COLUMNS),
            "fits": _empty_table(FITS_COLUMNS),
            "fits_ca": _empty_table(FITS_CA_COLUMNS),
            "shared_fits": _empty_table(SHARED_FITS_COLUMNS),
            "baselines": _empty_table(BASELINES_COLUMNS),
            "absorption": _empty_table(ABSORPTION_COLUMNS),
            "foil": _empty_table(FOIL_COLUMNS),
            "foil_pvalue": _empty_table(FOIL_PVALUE_COLUMNS),
            "khi": _empty_table(KHI_COLUMNS),
        }
        # The microbenchmark source is independent of the runs: the frozen
        # tables (or empty tables) are in either branch.
        tables.update(microbench_input["frames"])
    else:
        tables, fit_covs, shared_covs = _build_tables(
            runs=runs,
            sweep_runs=sweep_runs,
            configuration=configuration,
            dep_commit=dep_commit,
            config_name=config_name,
            micro=microbench_input["frames"],
            algorithms=list(config.get("algorithms", [])),
        )
    source_parts = [f"{label}: {sweep[label]['dir']}" for label in sweep if sweep[label]["dir"].is_dir()]
    if legacy_input["source"] != "none":
        source_parts.append(legacy_input["source"])
    attrs = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "sources": "; ".join(source_parts) or "none",
        "sweep_machines": ",".join(sweep_labels),
        "machine_titles": "; ".join(f"{label}: {sweep[label]['title']}" for label in sweep_labels),
        "algorithm_order": ",".join(str(algorithm) for algorithm in config.get("algorithms", [])),
        "commit_order": ",".join(str(c["name"]) for c in config.get("commits", []) if isinstance(c, dict)),
        "config_order": json.dumps(_config_order(config), sort_keys=True),
        "excluded_sources": json.dumps(legacy_input["excluded_sources"], sort_keys=True),
        "excluded_runs": str(legacy_input["excluded_runs"]),
        "microbench_source": microbench_input["source"],
        "microbench_runs": json.dumps(microbench_input["runs"], sort_keys=True),
        "microbench_protocol": json.dumps(microbench_input["protocol"], sort_keys=True),
    }
    write_results(output, tables, attrs, fit_covs, shared_covs)


def _slice_runs(
    runs: pd.DataFrame, configuration: str | None, dep_commit: str | None, config_name: str | None
) -> pd.DataFrame:
    """Restrict a runs frame to the requested configuration / dimension slice.

    The `configuration`, `dep_commit`, and `config_name` filters are applied
    independently; a `None` value leaves that dimension unfiltered.

    Args:
        runs: the runs frame to restrict (a copy is returned, the input
            is not mutated).
        configuration: "run-time" / "compile-time" or None.
        dep_commit: commit name or None.
        config_name: config name or None.

    Returns:
        pd.DataFrame: the filtered rows.

    """
    frame = runs
    if configuration is not None and "configuration" in frame:
        frame = frame[frame["configuration"] == configuration]
    if dep_commit is not None and "dep_commit" in frame:
        frame = frame[frame["dep_commit"] == dep_commit]
    if config_name is not None and "config" in frame:
        frame = frame[frame["config"] == config_name]
    return frame


def _config_order(config: dict) -> dict[str, list[str]]:
    """Return the per-algorithm config order (config.json), for the figures.

    An algorithm without an explicit config entry gets a single implicit
    ``default`` (matching ``config.py effective_configs``).

    Args:
        config: the parsed configuration.

    Returns:
        dict: algorithm -> ordered config names (``{algo: [names]}``).

    """
    entries = config.get("configs", {})
    entries = entries if isinstance(entries, dict) else {}
    order: dict[str, list[str]] = {}
    for algorithm in config.get("algorithms", []):
        configs = entries.get(algorithm)
        order[str(algorithm)] = list(map(str, configs)) if isinstance(configs, dict) and configs else ["default"]
    return order


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
    parser.add_argument(
        "--dep-commit",
        default=None,
        help="compute the statistics and the fits only from runs of that commit (default: all of them)",
    )
    parser.add_argument(
        "--config",
        dest="config_name",
        default=None,
        help="compute the statistics and the fits only from runs of that allocator config (default: all of them)",
    )
    args = parser.parse_args()
    main(args.output, args.configuration, args.dep_commit, args.config_name)
