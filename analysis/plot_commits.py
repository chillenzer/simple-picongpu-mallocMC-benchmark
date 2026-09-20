"""Per-machine dependency-commit comparison figures from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws, for
every machine that has a (machine, config) combination with at least two
dependency commits, one figure (`figures/commits-<machine>.pdf`): one row per
(scenario, algorithm) and two shared columns. The left column is the baseline
comparison: a marker (with a p75 error bar) at each commit's ordinal position
for the zero-delay p50 runtime (s), with the commit's PIConGPU short hash (the
`picongpu` column, 8 chars) annotating the marker. The right column is the
delay sweep: the p50 runtime (s) against the log malloc-delay (ns) as one line
per commit, with each commit's fitted model curve and bootstrap sleeve where a
fit exists. A row whose commit is baseline-only (no sweep runs) draws the
baseline marker but the sweep column says "no sweep data". The figure is
`commits-<machine>.pdf`, or `commits-<machine>-<config>.pdf` when `--config`
is `all` (all configs at once). The default config is `default`. The figure
skips (a note, no file) when the (machine, config) data has only one distinct
commit. For a baseline-only run series this is one of the primary comparison
outputs; the sweep column is simply then empty per row.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import performance_model
from results_io import (
    RESULTS,
    algorithm_order,
    commit_order,
    grid_label,
    load_results,
    machine_titles,
    read_fit_covs,
    read_table,
    scenario_key,
    scenario_name,
)
from run_logs import FREE_DELAY, MALLOC_DELAY

mpl.use("pdf")

FIGURES = Path("figures")
# One distinct marker per commit, in commit order (a forest-plot style).
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
# Padding for the snug x (log) limit of the sweep columns, as a fraction of the
# log span (see plot_sweeps).
_X_LOG_PAD_FRAC = 0.05


def _commit_hex(runs: pd.DataFrame) -> dict[str, str]:
    """Map each commit to its PIConGPU short hash (8 chars) from the runs.

    Args:
        runs: the machine/config-filtered runs table.

    Returns:
        dict[str, str]: the commit name to its short (8-char) PIConGPU hex,
        from the `picongpu` column (the first non-empty value per commit).

    """
    if "picongpu" not in runs:
        return {}
    mapping: dict[str, str] = {}
    for commit, group in runs.groupby("dep_commit", dropna=False):
        values = [str(h) for h in group["picongpu"] if isinstance(h, str) and h]
        if values:
            mapping[str(commit)] = values[0][:8]
    return mapping


def _commits_present(frame: pd.DataFrame, commits: list[str]) -> list[str]:
    """Return the commits present in a frame, in the file's commit order.

    Args:
        frame: the table to inspect (any table with a `dep_commit` column).
        commits: the file's commit order.

    Returns:
        list[str]: the present commits, ordered.

    """
    present = set(frame["dep_commit"])
    ordered = [c for c in commits if c in present]
    ordered += [c for c in sorted(present) if c not in commits]
    return ordered


def _snug_logx(ax: plt.Axes) -> None:
    """Limit a log axis snugly to its drawn points (see plot_sweeps).

    Args:
        ax: the axis to limit (x direction, in data coordinates).

    """
    lows: list[float] = []
    highs: list[float] = []
    for child in ax.get_children():
        if hasattr(child, "get_xdata"):
            values = np.asarray(child.get_xdata(), dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
        elif hasattr(child, "get_segments"):
            segs = [seg for seg in child.get_segments() if len(seg)]
            values = np.concatenate([seg[:, 0] for seg in segs]) if segs else np.array([])
            values = values[np.isfinite(values) & (values > 0)]
        else:
            continue
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    if not lows or max(highs) <= min(lows):
        return
    lo0, hi0 = min(lows), max(highs)
    pad = _X_LOG_PAD_FRAC * math.log10(hi0 / lo0)
    ax.set_xlim(10 ** (math.log10(lo0) - pad), 10 ** (math.log10(hi0) + pad))


def _baseline_marker(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    ax: plt.Axes, baselines: pd.DataFrame, algo: str, scen: tuple, commits: list[str], commit_hex: dict
) -> None:
    """Draw one (scenario, algorithm) baseline column: a marker per commit.

    Args:
        ax: the column's axis (x categorical commit ordinal, y log runtime).
        baselines: the machine/config-filtered baselines rows.
        algo: the row's algorithm.
        scen: the row's scenario key.
        commits: the commit order (drives the ordinal x positions).
        commit_hex: the commit to PIConGPU short hash map (annotations).

    """
    by_commit: dict[str, tuple[float, float | None]] = {}
    for _, row in baselines[baselines["algorithm"] == algo].iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != scen:
            continue
        if pd.notna(row["p50"]):
            p75 = float(row["p75"]) if pd.notna(row["p75"]) and row["p75"] >= row["p50"] else None
            by_commit[str(row["dep_commit"])] = (float(row["p50"]), p75)
    for i, commit in enumerate(commits):
        found = by_commit.get(commit)
        if found is None:
            continue
        p50, p75 = found
        ax.errorbar(
            [p50],
            [i],
            xerr=[max(p50 * 0.02, 0.02 * p75) if p75 else 0.0],
            fmt="none",
            ecolor="0.35",
            elinewidth=1,
            capsize=3,
            zorder=3,
        )
        ax.scatter(
            [p50],
            [i],
            marker=MARKERS[(commits.index(commit)) % len(MARKERS)],
            s=60,
            color="0.25",
            zorder=4,
        )
        hex8 = commit_hex.get(commit, "")
        if hex8:
            ax.annotate(hex8, (p50, i), textcoords="offset points", xytext=(6, -8), fontsize=6, va="top")


def _sweep_points(group_stats: pd.DataFrame, commit: str, algo: str, scen: tuple) -> list[tuple[float, float]]:
    """Return the malloc-delay sweep points of one (commit, algo, scenario).

    Args:
        group_stats: the machine/group statistics to draw from.
        commit: the commit name.
        algo: the algorithm.
        scen: the scenario key.

    Returns:
        list[tuple[float, float]]: the (malloc-delay ns, p50 runtime s) points.

    """
    match = group_stats[
        (group_stats["dep_commit"] == commit) & (group_stats["algorithm"] == algo) & (group_stats[FREE_DELAY] == 0)
    ]
    points: list[tuple[float, float]] = []
    for _, row in match.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != scen:
            continue
        if pd.notna(row["p50"]) and pd.notna(row[MALLOC_DELAY]) and row[MALLOC_DELAY] > 0:
            points.append((float(row[MALLOC_DELAY]), float(row["p50"])))
    return points


def _fit_row(fits: pd.DataFrame, commit: str, algo: str, scen: tuple) -> pd.Series | None:
    """Return the usable 2-D fit row of one (commit, algo, scenario), or None.

    Args:
        fits: the fits table.
        commit: the commit name.
        algo: the algorithm.
        scen: the scenario key.

    Returns:
        pd.Series | None: the 2-D fit row, or None when absent/incomplete.

    """
    for _, row in fits[fits["dep_commit"] == commit].iterrows():
        if row["algorithm"] != algo or pd.isna(row["model"]) or row["model"] != "2d":
            continue
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != scen:
            continue
        if all(pd.notna(row[k]) for k in ("W", "N_malloc", "N_free", "A_malloc", "A_free", "m0_ns", "f0_ns")):
            return row
    return None


def _sweep_fit(ax: plt.Axes, fit_row: pd.Series, cov: tuple | None, xs: list[float], color: str) -> None:
    """Overlay the fitted model curve and its bootstrap sleeve on a sweep line.

    Args:
        ax: the sweep column's axis.
        fit_row: the 2-D fit row (parameters in nanoseconds where needed).
        cov: the (fit_params, pcov) pair, or None.
        xs: the sweep point delays in ns (the curve's x range).
        color: the curve's colour.

    """
    p7 = (
        float(fit_row["W"]),
        float(fit_row["N_malloc"]),
        float(fit_row["N_free"]),
        float(fit_row["A_malloc"]),
        float(fit_row["A_free"]),
        float(fit_row["m0_ns"]) * 1e-9,
        float(fit_row["f0_ns"]) * 1e-9,
    )
    x_ns = np.geomspace(float(xs[0]), float(xs[-1]), 128)
    x_s = x_ns * 1e-9
    held = np.zeros_like(x_s)
    ax.plot(x_ns, performance_model.model_2d(x_s, held, p7), color=color, linewidth=2.2, alpha=0.9, zorder=4)
    if cov is None or len(cov[0]) != 7:
        return

    def fn(p: np.ndarray) -> np.ndarray:
        return performance_model.model_2d(x_s, held, p)

    lower = (0.0, 0.0, 0.0, 0.0, 0.0, performance_model.EPS_S, performance_model.EPS_S)
    band = performance_model.bootstrap_band(x_s, fn, cov[0], cov[1], lower=lower)
    if band is not None:
        ax.fill_between(x_ns, band[0], band[1], color=color, alpha=0.12, zorder=2)


def _sweep_line(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    ax: plt.Axes,
    group_stats: pd.DataFrame,
    fits: pd.DataFrame,
    covs: dict,
    commit: str,
    algo: str,
    scen: tuple,
    configs: list[str],
    color: str,
) -> bool:
    """Draw one commit's malloc-delay sweep line (and fit) in the sweep column.

    Args:
        ax: the sweep column's axis (x log delay ns, y log runtime s).
        group_stats: the machine/group statistics.
        fits: the fits rows.
        covs: the machine's fit covariance entries (per commit/config/... key).
        commit: the commit being drawn.
        algo: the row's algorithm.
        scen: the row's scenario key.
        configs: the config names being drawn (for the fit key lookups).
        color: the commit's line colour.

    Returns:
        bool: True when at least one sweep point was drawn.

    """
    points = sorted(_sweep_points(group_stats, commit, algo, scen))
    if not points:
        return False
    xs = [p[0] for p in points]
    ax.plot(xs, [p[1] for p in points], color=color, marker=".", markersize=4, linewidth=1.2, zorder=3)
    fit_row = _fit_row(fits, commit, algo, scen)
    if fit_row is None:
        return True
    cov = None
    for key, value in covs.items():
        if (
            key[1] == commit
            and key[2] in configs
            and key[3] == fit_row["setup"]
            and key[4] == algo
            and key[5] == grid_label(fit_row["x"], fit_row["y"], fit_row["z"])
        ):
            cov = value
            break
    _sweep_fit(ax, fit_row, cov, xs, color)
    return True


def _rows(baselines: pd.DataFrame, group_stats: pd.DataFrame, algorithms: list[str]) -> list[tuple]:
    """Return the (scenario, algorithm) row order, in first-appearance order.

    Args:
        baselines: the machine/config-filtered baselines rows.
        group_stats: the machine/config-filtered group statistics.
        algorithms: the file's algorithm order.

    Returns:
        list[tuple]: the (scenario key, algorithm) rows.

    """
    frames = [f for f in (baselines, group_stats) if not f.empty]
    scen_order = list(
        dict.fromkeys(scenario_key(r["setup"], r["x"], r["y"], r["z"]) for f in frames for _, r in f.iterrows())
    )
    rows: list[tuple] = []
    for scen in scen_order:
        algos = set()
        for frame in frames:
            for _, row in frame.iterrows():
                if scenario_key(row["setup"], row["x"], row["y"], row["z"]) == scen:
                    algos.add(row["algorithm"])
        ordered = sorted(algos, key=lambda a: algorithms.index(a) if a in algorithms else len(algorithms))
        rows.extend((scen, algo) for algo in ordered)
    return rows


def plot_machine(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    title: str,
    baselines: pd.DataFrame,
    group_stats: pd.DataFrame,
    fits: pd.DataFrame,
    covs: dict,
    comm: list[str],
    configs: list[str],
    algorithms: list[str],
    config_hex: dict[str, str],
) -> plt.Figure:
    """Build one machine's commit-comparison figure for one config.

    Args:
        title: the figure title (hardware, config, commit count).
        baselines: the machine/config-filtered baselines rows.
        group_stats: the machine/config-filtered group statistics.
        fits: the machine/config-filtered fits rows.
        covs: the machine's fit covariance entries.
        comm: the commit order to compare across.
        configs: the config names being drawn (for fit key lookups).
        algorithms: the file's algorithm order.
        config_hex: the commit to PIConGPU short hash map.

    Returns:
        plt.Figure: the figure.

    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rows = _rows(baselines, group_stats, algorithms)
    n_rows = max(len(rows), 1)
    fig = plt.figure(figsize=(6.5 * 2, 4.0 * n_rows), layout="constrained")
    fig.suptitle(title)
    gridspec = fig.add_gridspec(n_rows, 2)
    y0 = None
    for i, (scen, algo) in enumerate(rows):
        base_ax = fig.add_subplot(gridspec[i, 0], sharey=y0) if y0 is not None else fig.add_subplot(gridspec[i, 0])
        swp_ax = (
            fig.add_subplot(gridspec[i, 1], sharey=y0, sharex=None)
            if y0 is not None
            else fig.add_subplot(gridspec[i, 1])
        )
        if y0 is None:
            y0 = base_ax
        _baseline_marker(base_ax, baselines, algo, scen, comm, config_hex)
        drew = False
        for j, commit in enumerate(comm):
            if _sweep_line(swp_ax, group_stats, fits, covs, commit, algo, scen, configs, cycle[j % len(cycle)]):
                drew = True
        if not drew:
            swp_ax.text(
                0.5, 0.5, "no sweep data", transform=swp_ax.transAxes, ha="center", va="center", fontsize=7, color="0.5"
            )
        base_ax.set_title("", fontsize=10)
        base_ax.set_yticks(range(len(comm)))
        base_ax.set_yticklabels(comm, fontsize=8)
        base_ax.invert_yaxis()
        swp_ax.set_xscale("log")
        base_ax.set_ylabel(f"{algo}\n{scenario_name(*scen)}", fontsize=8, rotation=90, labelpad=10)
    # Column titles once, on the top row.
    fig.axes[0].set_xlabel("zero-delay p50 runtime (s)")
    fig.axes[0].set_title("baseline (zero-delay) runtime per commit")
    fig.axes[1].set_xlabel("malloc delay (ns)")
    fig.axes[1].set_title("delay sweep per commit")
    for ax in fig.axes[2:]:
        if ax.get_title():
            ax.set_title("")
    for ax in fig.axes:
        if ax.get_xscale() == "log":
            _snug_logx(ax)
    return fig


def _eligible(baselines: pd.DataFrame, commits: list[str]) -> set[tuple[str, str]]:
    """Return the (machine, config) pairs with at least two distinct commits.

    Args:
        baselines: the full baselines table.
        commits: the file's commit order.

    Returns:
        set[tuple[str, str]]: the qualifying (machine, config) pairs.

    """
    eligible: set[tuple[str, str]] = set()
    for (machine, config), group in baselines.groupby(["machine", "config"], dropna=False):
        if len(_commits_present(group, commits)) >= 2:
            eligible.add((str(machine), str(config)))
    return eligible


def _save_machine(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    machine: str,
    configs_for_m: list[str],
    results_tables: dict,
    commits: list[str],
    algorithms: list[str],
    titles: dict,
    *,
    multi_config: bool,
) -> int:
    """Draw and save one machine's commit figures over its eligible configs.

    Args:
        machine: the machine label.
        configs_for_m: the machine's eligible configs (after filtering).
        results_tables: the tables read from the results file.
        commits: the file's commit order.
        algorithms: the file's algorithm order.
        titles: the file's machine hardware titles.
        multi_config: whether all configs of the machine are drawn at once
            (so the files carry a ``-<config>`` suffix).

    Returns:
        int: the number of figures written.

    """
    wrote = 0
    t = results_tables
    runs_m = t["runs"][(t["runs"]["machine"] == machine)]
    hex_map = _commit_hex(runs_m)
    covs_m = {key: value for key, value in t["covs"].items() if key[0] == machine}
    for config in configs_for_m:
        base = t["baselines"][(t["baselines"]["machine"] == machine) & (t["baselines"]["config"] == config)]
        show = config if config else "(legacy)"
        fig = plot_machine(
            f"{titles.get(machine, machine)} — config {show}",
            base,
            t["group_stats"][(t["group_stats"]["machine"] == machine) & (t["group_stats"]["config"] == config)],
            t["fits"][(t["fits"]["machine"] == machine) & (t["fits"]["config"] == config)],
            covs_m,
            commits,
            [config] if config else ["legacy"],
            algorithms,
            hex_map,
        )
        suffix = f"-{config}" if multi_config else ""
        path = FIGURES / f"commits-{machine}{suffix}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path} (config {show})")
        wrote += 1
    return wrote


def _save_all(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    baselines: pd.DataFrame,
    runs: pd.DataFrame,
    group_stats: pd.DataFrame,
    fits: pd.DataFrame,
    covs: dict,
    commits: list[str],
    algorithms: list[str],
    titles: dict,
    *,
    machine: str | None,
    config: str,
) -> int:
    """Select the eligible machines and dispatch the per-machine figure runs.

    Args:
        baselines: the full baselines table.
        runs: the full runs table.
        group_stats: the full group statistics table.
        fits: the full fits table.
        covs: the full fit covariance entries.
        commits: the file's commit order.
        algorithms: the file's algorithm order.
        titles: the file's machine hardware titles.
        machine: the machine to draw, or None for every eligible machine.
        config: the config to compare commits across; "all" for every config.

    Returns:
        int: the process exit code (1 for a bad ``--machine`` value).

    """
    eligible = _eligible(baselines, commits)
    if not eligible:
        print("no (machine, config) with at least two commits; nothing to draw", file=sys.stderr)
        return 0
    config_filter = None if config in {None, "all"} else config
    machines = list(dict.fromkeys(m for m, _c in eligible))
    if machine is not None:
        if machine not in machines:
            print(f"machine {machine!r} not eligible (available: {', '.join(machines)})", file=sys.stderr)
            return 1
        machines = [machine]
    FIGURES.mkdir(exist_ok=True)
    multi_config = config_filter is None and len({c for _m, c in eligible}) > 1
    tables = {
        "baselines": baselines,
        "runs": runs,
        "group_stats": group_stats,
        "fits": fits,
        "covs": covs,
    }
    for m in machines:
        configs_for_m = [c for (mm, c) in eligible if mm == m]
        if config_filter is not None:
            configs_for_m = [c for c in configs_for_m if c == config_filter]
        if configs_for_m:
            _save_machine(m, configs_for_m, tables, commits, algorithms, titles, multi_config=multi_config)
    return 0


def main(*, machine: str | None = None, config: str = "default", show: bool = False, results: Path = RESULTS) -> int:
    """Draw the per-machine dependency-commit comparison figures.

    Args:
        machine: the machine to draw, or None for every eligible machine.
        config: the allocator config to compare commits across; "all" for
            every config in the data (one figure per (machine, config)).
        show: display the figures in a window (blocking).
        results: the results file, e.g. `output/results.h5`.

    Returns:
        int: the process exit code.

    """
    try:
        file = load_results(results)
    except OSError as err:
        print(f"{err}\nno results file: run `python3 analysis/compute_results.py` first", file=sys.stderr)
        return 1
    with file:
        baselines = read_table(file, "baselines")
        runs = read_table(file, "runs")
        group_stats = read_table(file, "group_stats")
        fits = read_table(file, "fits")
        covs = read_fit_covs(file)
        commits = commit_order(file)
        algorithms = algorithm_order(file)
        titles = machine_titles(file)
    if baselines.empty or "dep_commit" not in baselines:
        print("no baselines (or no commit column) in the results file; nothing to draw", file=sys.stderr)
        return 0
    code = _save_all(
        baselines,
        runs,
        group_stats,
        fits,
        covs,
        commits,
        algorithms,
        titles,
        machine=machine,
        config=config,
    )
    if show:
        plt.show()
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-machine dependency-commit comparison figures from output/results.h5"
        " (figures/commits-<machine>.pdf)."
    )
    parser.add_argument("--machine", help="draw only this machine's figure (default: every eligible machine).")
    parser.add_argument(
        "--config",
        default="default",
        help="the config to compare commits across; 'all' for every config (default: %(default)s)",
    )
    parser.add_argument(
        "--show", action="store_true", help="display the figures in a window (blocking); by default only saved."
    )
    parser.add_argument("--results", type=Path, default=RESULTS, help="the results file (default: %(default)s).")
    args = parser.parse_args()
    sys.exit(main(machine=args.machine, config=args.config, show=args.show, results=args.results))
