"""Per-machine allocator-config comparison figures from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws,
for every machine that has a (machine, commit) combination with an algorithm
carrying at least two allocator configs, one figure
(`figures/configs-<machine>.pdf`): one row per scenario (setup, grid) and one
column per such algorithm, each cell a bar chart of the zero-delay p50
runtime (s) per config (x tick = config name, in the file's `config_order`),
with a whisker to the p75, the `default` config's bar hatched to read as the
reference. The y-axis is shared per row (so config magnitudes within one
scenario are comparable across the row's algorithms). For a baseline-only
run series (no delay sweep) this is one of the primary comparison outputs;
the fit sections of the sweep figures are unrelated. The default commit is
the first of the file's `commit_order` (typically `default`), or all commits
when `--dep-commit all`. The figure skips (a note, no file) when no
(machine, commit) pair shows an algorithm with at least two configs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from results_io import (
    RESULTS,
    algorithm_order,
    config_order,
    load_results,
    machine_titles,
    read_table,
    scenario_key,
    scenario_name,
)

mpl.use("pdf")

FIGURES = Path("figures")


def _scenarios(baselines: pd.DataFrame) -> list[tuple]:
    """Return the (setup, grid) scenario rows of the data, in first-appearance order.

    Args:
        baselines: the baselines rows to order (already machine/commit filtered).

    Returns:
        list[tuple]: the canonical scenario keys, in first-appearance order.

    """
    scenarios: list[tuple] = []
    for _, row in baselines.iterrows():
        key = scenario_key(row["setup"], row["x"], row["y"], row["z"])
        if key not in scenarios:
            scenarios.append(key)
    return scenarios


def _configs_present(baselines: pd.DataFrame, algo: str, cfg_order: list[str]) -> list[str]:
    """Return the algorithm's config names present in the data, in config order.

    Args:
        baselines: the machine/commit-filtered baselines rows.
        algo: the algorithm.
        cfg_order: the file's config order for the algorithm.

    Returns:
        list[str]: the present configs, ordered.

    """
    present = set(baselines[baselines["algorithm"] == algo]["config"])
    ordered = [c for c in cfg_order if c in present]
    ordered += [c for c in sorted(present) if c not in cfg_order]
    return ordered


def _cell_stats(baselines: pd.DataFrame, algo: str, cfg: str, scen: tuple) -> tuple[float | None, float | None]:
    """Return (p50, p75) zero-delay runtimes of one (algo, config, scenario).

    Args:
        baselines: the machine/commit-filtered baselines rows.
        algo: the algorithm.
        cfg: the config name.
        scen: the scenario key.

    Returns:
        tuple[float | None, float | None]: the (p50, p75) in seconds, or
        (None, None) when the cell is empty.

    """
    match = baselines[(baselines["algorithm"] == algo) & (baselines["config"] == cfg)]
    for _, row in match.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) == scen:
            p50 = float(row["p50"]) if pd.notna(row["p50"]) else None
            p75 = float(row["p75"]) if pd.notna(row["p75"]) else None
            return (p50, p75)
    return (None, None)


def _bar_for(ax: plt.Axes, baselines: pd.DataFrame, algo: str, cfg: str, scen: tuple, i: int) -> bool:  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    """Draw one config's bar in a cell; report whether it was drawn.

    Args:
        ax: the cell's axis.
        baselines: the machine/commit-filtered baselines rows.
        algo: the cell's algorithm.
        cfg: the config name.
        scen: the row's scenario key.
        i: the bar's x position (its index in the config order).

    Returns:
        bool: True when a bar was drawn (the value was available).

    """
    p50, p75 = _cell_stats(baselines, algo, cfg, scen)
    if p50 is None:
        return False
    error = (p75 - p50) if (p75 is not None and p75 >= p50) else 0.0
    default = cfg == "default"
    ax.bar(
        [i],
        [p50],
        width=0.62,
        yerr=[[0.0], [error]],
        capsize=3,
        color="0.62" if default else "0.38",
        edgecolor="black",
        hatch="//" if default else None,
        zorder=3,
    )
    return True


def _draw_cell(  # ruff: ignore[too-many-arguments]
    ax: plt.Axes, baselines: pd.DataFrame, algo: str, configs: list[str], scen: tuple, *, first_row: bool
) -> None:
    """Fill one (scenario, algorithm) cell with its per-config bars.

    Args:
        ax: the cell's axis.
        baselines: the machine/commit-filtered baselines rows.
        algo: the cell's algorithm.
        configs: the bar order (the algorithm's present configs).
        scen: the row's scenario key.
        first_row: whether this is the bottom row (carries the x tick labels).

    """
    positions: list[int] = []
    labels: list[str] = []
    for i, cfg in enumerate(configs):
        if _bar_for(ax, baselines, algo, cfg, scen, i):
            positions.append(i)
            labels.append(cfg)
    if first_row:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    if not positions:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.5")
    ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)


def plot_machine(
    title: str, baselines: pd.DataFrame, dep_commit: str, cfg_order: dict, algorithms: list[str]
) -> plt.Figure:
    """Build one machine's config-comparison figure for one commit.

    Args:
        title: the figure title (the hardware, with the commit).
        baselines: the machine/commit-filtered baselines rows.
        dep_commit: the commit the figure compares configs across.
        cfg_order: the file's per-algorithm config order.
        algorithms: the file's algorithm order.

    Returns:
        plt.Figure: the figure, or None when nothing qualifies (no algorithm
        with at least two configs).

    """
    qualified = [algo for algo in algorithms if len(_configs_present(baselines, algo, cfg_order.get(algo, []))) >= 2]
    if not qualified:
        return None
    scenarios = _scenarios(baselines)
    n_rows, n_cols = len(scenarios), len(qualified)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    fig.suptitle(f"{title} — allocator configs ({dep_commit})")
    for j, algo in enumerate(qualified):
        configs = _configs_present(baselines, algo, cfg_order.get(algo, []))
        for i, scen in enumerate(scenarios):
            ax = axes[i, j]
            _draw_cell(ax, baselines, algo, configs, scen, first_row=(i == n_rows - 1))
            ax.set_title(algo, fontsize=10)
            ax.set_yscale("log")
            ax.tick_params(axis="y", labelsize=7)
    # Each row's y tick labels sit on that row's right-most column: name the
    # row by its scenario and the shared axis by the metric (per-row shared y,
    # so config magnitudes within one scenario compare across the row's
    # algorithms).
    for i, scen in enumerate(scenarios):
        axes[i, n_cols - 1].set_ylabel(f"{scenario_name(*scen)} — zero-delay p50 (s)", rotation=270)
    return fig


def _eligible(baselines: pd.DataFrame, cfg_order: dict, algorithms: list[str]) -> set[tuple[str, str]]:
    """Return the (machine, commit) pairs with an algorithm carrying >= 2 configs.

    Args:
        baselines: the full baselines table.
        cfg_order: the file's per-algorithm config order.
        algorithms: the file's algorithm order.

    Returns:
        set[tuple[str, str]]: the qualifying (machine, dep_commit) pairs.

    """
    eligible: set[tuple[str, str]] = set()
    for (machine, commit), group in baselines.groupby(["machine", "dep_commit"], dropna=False):
        present = sum(1 for algo in algorithms if len(_configs_present(group, algo, cfg_order.get(algo, []))) >= 2) > 0
        if present:
            eligible.add((str(machine), str(commit)))
    return eligible


def _save_machine(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    machine: str,
    commits_for_m: list[str],
    baselines: pd.DataFrame,
    cfg_order: dict,
    algorithms: list[str],
    titles: dict,
    *,
    multi_commit: bool,
) -> int:
    """Draw and save one machine's config figures over its eligible commits.

    Args:
        machine: the machine label.
        commits_for_m: the machine's eligible commits (after filtering).
        baselines: the full baselines table (filtered per commit inside).
        cfg_order: the file's per-algorithm config order.
        algorithms: the file's algorithm order.
        titles: the file's machine hardware titles.
        multi_commit: whether all commits of the machine are being drawn at
            once (so the files are disambiguated by a ``-<commit>`` suffix).

    Returns:
        int: the number of figures written.

    """
    wrote = 0
    for commit in commits_for_m:
        subset = baselines[(baselines["machine"] == machine) & (baselines["dep_commit"] == commit)]
        shown = commit if commit else "(legacy)"
        fig = plot_machine(f"{titles.get(machine, machine)} — {shown}", subset, commit, cfg_order, algorithms)
        if fig is None:
            continue
        suffix = f"-{commit}" if multi_commit else ""
        path = FIGURES / f"configs-{machine}{suffix}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path} (commit {shown})")
        wrote += 1
    return wrote


def main(
    *, machine: str | None = None, dep_commit: str = "default", show: bool = False, results: Path = RESULTS
) -> int:
    """Draw the per-machine allocator-config comparison figures.

    Args:
        machine: the machine to draw, or None for every eligible machine.
        dep_commit: the commit to compare configs across; "all" for every
        commit in the data (one figure per (machine, commit)).
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
        cfg_order = config_order(file)
        algorithms = algorithm_order(file)
        titles = machine_titles(file)
    if baselines.empty or "dep_commit" not in baselines:
        print("no baselines (or no commit column) in the results file; nothing to draw", file=sys.stderr)
        return 0
    eligible = _eligible(baselines, cfg_order, algorithms)
    if not eligible:
        print("no (machine, commit) with an algorithm carrying at least two configs; nothing to draw", file=sys.stderr)
        return 0
    commit_filter = None if dep_commit in {None, "all"} else dep_commit
    machines = [m for m, _c in eligible]
    machines = list(dict.fromkeys(machines))
    if machine is not None:
        if machine not in machines:
            print(f"machine {machine!r} not eligible (available: {', '.join(machines)})", file=sys.stderr)
            return 1
        machines = [machine]
    FIGURES.mkdir(exist_ok=True)
    multi_commit = commit_filter is None and len({c for _m, c in eligible}) > 1
    total = 0
    for m in machines:
        commits_for_m = [c for (mm, c) in eligible if mm == m]
        if commit_filter is not None:
            commits_for_m = [c for c in commits_for_m if c == commit_filter]
        total += _save_machine(m, commits_for_m, baselines, cfg_order, algorithms, titles, multi_commit=multi_commit)
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-machine allocator-config comparison figures from output/results.h5"
        " (figures/configs-<machine>.pdf)."
    )
    parser.add_argument("--machine", help="draw only this machine's figure (default: every eligible machine).")
    parser.add_argument(
        "--dep-commit",
        default="default",
        help="the commit to compare configs across; 'all' for every commit (default: %(default)s)",
    )
    parser.add_argument(
        "--show", action="store_true", help="display the figures in a window (blocking); by default only saved."
    )
    parser.add_argument("--results", type=Path, default=RESULTS, help="the results file (default: %(default)s).")
    args = parser.parse_args()
    sys.exit(main(machine=args.machine, dep_commit=args.dep_commit, show=args.show, results=args.results))
