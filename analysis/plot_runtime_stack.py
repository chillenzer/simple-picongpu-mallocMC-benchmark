"""Per-scenario runtime-budget figures from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws,
for every (setup, grid) scenario with at least one usable fit, one figure
(`figures/runtime-stack-<setup>-<grid>.pdf`): the x-axis is the hardware
(short name) of every sweep machine with data, and under each hardware one
triplet of three segment bars per allocator (in the file's
`algorithm_order`): the fitted W (grey), A_malloc (blue) and A_free
(orange) side by side, so the individual costs compare directly across the
allocators, with the fitted total annotated above each triplet and each bar
labelled with its percentage of the triplet's total runtime. The segments
come from the scenario's combined (shared-parameter) fit -- one shared W
and the per-allocator A_malloc/A_free of the fit made across all of the
scenario's algorithms -- where it exists, and from the allocator's
individual fit otherwise. By default every scenario gets its figure,
`--name` restricts the run to one, e.g. `--name FoilLCT-256x1280`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from results_io import (
    RESULTS,
    algorithm_order,
    load_results,
    read_table,
    scenario_filename,
    scenario_key,
    scenario_name,
    sweep_machine_labels,
)

mpl.use("pdf")

FIGURES = Path("figures")
# the segments of the runtime-budget figure, as (label, fill colour) pairs,
# in bar order within each algorithm's triplet; the algorithm is carried by
# the triplet's position and its label, not by colour.
RUNTIME_SEGMENTS = (("W", "0.55"), ("A_malloc", "tab:blue"), ("A_free", "tab:orange"))


def _floor0(value: float) -> float:
    """Clamp a fitted segment to a non-negative display height.

    Args:
        value: the fitted segment value in seconds.

    Returns:
        float: the value, floored at zero.

    """
    return max(0.0, float(value))


def _scenario_sort_key(scenario: tuple) -> tuple:
    """Return the sort key of a (setup, x, y, z) scenario.

    Args:
        scenario: the scenario key (a missing 2-D z sorts first).

    Returns:
        tuple: the sort key.

    """
    return (scenario[0], scenario[1] or 0, scenario[2] or 0, scenario[3] if scenario[3] is not None else -1)


def _scenario_keys(table: pd.DataFrame) -> list[tuple]:
    """Collect the ordered unique scenario keys of one fits-like table.

    Rows without a complete fit (no model or a missing W) are ignored.

    Args:
        table: a fits table (the individual `fits` or the combined
        `shared_fits`).

    Returns:
        list[tuple]: the ordered scenario keys.

    """
    seen: set[tuple] = set()
    scenarios: list[tuple] = []
    for _, row in table.iterrows():
        if not row["model"] or pd.isna(row["W"]):
            continue
        key = scenario_key(row["setup"], row["x"], row["y"], row["z"])
        if key not in seen:
            seen.add(key)
            scenarios.append(key)
    return sorted(scenarios, key=_scenario_sort_key)


def list_scenarios(fits: pd.DataFrame, shared: pd.DataFrame) -> list[tuple]:
    """Collect the unique (setup, x, y, z) scenarios with usable fit data.

    A scenario qualifies when its individual fits or its combined fit has
    at least one usable row, so a scenario whose machines only have
    no-delay runs gets no figure.

    Args:
        fits: the individual fits table of the results file.
        shared: the combined (shared-parameter) fits table.

    Returns:
        list[tuple]: the ordered scenario keys.

    """
    scenarios = _scenario_keys(fits)
    seen = set(scenarios)
    for key in _scenario_keys(shared):
        if key not in seen:
            seen.add(key)
            scenarios.append(key)
    return sorted(scenarios, key=_scenario_sort_key)


def _row_budget(row: pd.Series) -> tuple[float, float, float]:
    """Return the (W, A_malloc, A_free) budget of one fits row, floored at 0.

    A missing allocation cost (a 1-D fit on the other operation) is
    treated as 0.

    Args:
        row: one row of a fits table.

    Returns:
        tuple[float, float, float]: the floored budget.

    """
    return (
        _floor0(row["W"]),
        _floor0(row["A_malloc"]) if pd.notna(row["A_malloc"]) else 0.0,
        _floor0(row["A_free"]) if pd.notna(row["A_free"]) else 0.0,
    )


def scenario_budgets(
    fits: pd.DataFrame,
    shared: pd.DataFrame,
    target: tuple,
) -> tuple[dict[str, dict[str, tuple[float, float, float]]], bool]:
    """Per-machine runtime budget of one (setup, grid) scenario.

    Built from the scenario's combined fit where it exists -- one shared W
    plus the per-allocator A_malloc/A_free of the fit made across all of
    the scenario's algorithms -- and from the valid individual fits of
    each machine otherwise.

    Args:
        fits: the individual fits table of the results file.
        shared: the combined (shared-parameter) fits table.
        target: the (setup, x, y, z) scenario key.

    Returns:
        tuple: (per machine, the {algorithm: (W, A_malloc, A_free)}
        budget; True when the budget came from the combined fit).

    """
    budget: dict[str, dict[str, tuple[float, float, float]]] = {}
    for _, row in shared.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != target or not row["model"] or pd.isna(row["W"]):
            continue
        budget.setdefault(row["machine"], {})[row["algorithm"]] = _row_budget(row)
    if budget:
        return budget, True
    budget = {}
    for _, row in fits.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != target or not row["model"] or pd.isna(row["W"]):
            continue
        budget.setdefault(row["machine"], {})[row["algorithm"]] = _row_budget(row)
    return budget, False


class TripletCtx(NamedTuple):
    """Drawing context for one algorithm's runtime-budget triplet."""

    ax: plt.Axes
    x: float
    bar_w: float
    segments: tuple[float, float, float]
    algorithm: str


def draw_runtime_triplet(ctx: TripletCtx) -> float:
    """Draw one algorithm's three segment bars, each with a percentage label.

    The bars of the triplet are coloured per segment (W grey, A_malloc
    blue, A_free orange); the algorithm itself is named by its slot and
    the label under the x axis, the fitted total is annotated above, and
    each bar is labelled with its share of the triplet's total runtime,
    W/(W+A_malloc+A_free) for the W bar and so on.

    Args:
        ctx: the triplet drawing context.

    Returns:
        float: the triplet's total height.

    """
    ax, x, bar_w = ctx.ax, ctx.x, ctx.bar_w
    total = sum(ctx.segments)
    for k, ((_seg, color), value) in enumerate(zip(RUNTIME_SEGMENTS, ctx.segments, strict=True)):
        ax.bar(x + (k - 1) * bar_w, value, width=bar_w, color=color, edgecolor="k", linewidth=0.5)
        if value > 0 and total > 0:
            ax.text(
                x + (k - 1) * bar_w,
                value / 2,
                f"{100 * value / total:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
            )
    ax.text(x, total, f"{total:.3g}", ha="center", va="bottom", fontsize=8)
    ax.text(
        x,
        -0.06,
        ctx.algorithm,
        transform=ax.get_xaxis_transform(),
        rotation=-90,
        ha="left",
        va="center",
        fontsize=8,
        color="0.25",
    )
    return total


def runtime_stack_figure(
    scenario: tuple,
    entries: list[tuple[str, dict]],
    algorithms: list[str],
    *,
    combined: bool = False,
) -> plt.Figure:
    """Draw one figure: the runtime budget per allocator, per hardware.

    The x-axis is the hardware; under each hardware one triplet of three
    segment-coloured bars per allocator (the triplet per allocator, in the
    file's `algorithm_order`): the fitted W (grey), A_malloc (blue) and
    A_free (orange) side by side. The allocator is named by the label
    under its triplet, the fitted total is annotated above it, and each
    bar carries its percentage of the triplet's total.

    Args:
        scenario: the (setup, x, y, z) scenario key.
        entries: per hardware, (short name, {algorithm: (W, A_malloc,
        A_free)}).
        algorithms: the allocators' display order.
        combined: whether the budgets come from the scenario's combined
        (shared-parameter) fit rather than the individual fits; named in
        the title.

    Returns:
        plt.Figure: the figure.

    """
    setup, x, y, z = scenario
    hw_names = [hw for hw, _budget in entries]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if not hw_names:
        return fig
    present = [algorithm for algorithm in algorithms if any(algorithm in budget for _hw, budget in entries)]
    n_algorithms = len(present)
    triplet_w = 0.8 / n_algorithms
    bar_w = 0.8 * triplet_w / 3
    max_total = 0.0
    for i, _hw in enumerate(hw_names):
        for j, algorithm in enumerate(present):
            if algorithm not in entries[i][1]:
                continue
            pos = i + (j - (n_algorithms - 1) / 2) * triplet_w
            total = draw_runtime_triplet(TripletCtx(ax, pos, bar_w, entries[i][1][algorithm], algorithm))
            max_total = max(max_total, total)
    if max_total <= 0:
        return fig
    ax.set_xticks(range(len(hw_names)))
    ax.set_xticklabels(hw_names, fontsize=9)
    ax.set_xlim(-0.45, len(hw_names) - 1 + 0.45)
    ax.set_ylim(0, max_total * 1.15)
    ax.set_ylabel("runtime (s)")
    ax.set_xlabel("hardware", labelpad=25)
    handles = [
        Patch(facecolor=color, edgecolor="k", linewidth=0.5, label=segment) for segment, color in RUNTIME_SEGMENTS
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    if combined:
        fig.suptitle(f"Runtime budget: {scenario_name(setup, x, y, z)} (combined fit)")
    else:
        fig.suptitle(f"Runtime budget: {scenario_name(setup, x, y, z)}")
    return fig


def main(*, name: str | None = None, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the per-scenario runtime-budget figures from one results file.

    Args:
        name: the scenario to draw, as in the figure filename without the
        `runtime-stack-` prefix and `.pdf` suffix, e.g. `FoilLCT-256x1280`;
        None for every scenario with at least one usable fit.
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
        fits = read_table(file, "fits")
        shared = read_table(file, "shared_fits") if "shared_fits" in file else pd.DataFrame()
        runs = read_table(file, "runs")
        algorithms = algorithm_order(file)
        sweep = sweep_machine_labels(file)
    if fits.empty:
        print("no fits in the results file", file=sys.stderr)
        return 1
    machine_hardware = runs.drop_duplicates("machine").set_index("machine")["hardware"].to_dict()
    # The x-axis lists every sweep machine with data, in the file's order;
    # a machine without a fit in the scenario keeps its (empty) slot.
    machines = [m for m in sweep if m in machine_hardware]
    scenarios = list_scenarios(fits, shared)
    if name is not None:
        scenarios = [scenario for scenario in scenarios if scenario_filename(*scenario) == f"runtime-stack-{name}.pdf"]
        if not scenarios:
            available = ", ".join(scenario_name(*scenario) for scenario in list_scenarios(fits, shared))
            print(f"no scenario {name!r} with fits (available: {available})", file=sys.stderr)
            return 1
    FIGURES.mkdir(exist_ok=True)
    for scenario in scenarios:
        budget, combined = scenario_budgets(fits, shared, scenario)
        entries = [(str(machine_hardware[m]), budget.get(m, {})) for m in machines]
        path = FIGURES / scenario_filename(*scenario)
        runtime_stack_figure(scenario, entries, algorithms, combined=combined).savefig(path)
        print(f"wrote {path}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-scenario runtime-budget figures from output/results.h5 "
        "(figures/runtime-stack-<setup>-<grid>.pdf)."
    )
    parser.add_argument(
        "--name",
        help="draw only that scenario, e.g. FoilLCT-256x1280 (default: every scenario with fits)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the figures in a window (blocking); by default they are only saved",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results file (default: %(default)s)",
    )
    args = parser.parse_args()
    sys.exit(main(name=args.name, show=args.show, results=args.results))
