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
allocators, with the fitted total annotated above and the measured
zero-delay runtime (median, IQR error bar) overlaid at each triplet. By
default every scenario gets its figure, `--name` restricts the run to one,
e.g. `--name FoilLCT-256x1280`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
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


def list_scenarios(fits: pd.DataFrame) -> list[tuple]:
    """Collect the unique (setup, x, y, z) scenarios with at least one usable fit.

    Rows of the fits table without a complete fit (no model or a missing
    W) are ignored, so a scenario whose machines only have no-delay runs
    gets no figure.

    Args:
        fits: the fits table of the results file.

    Returns:
        list[tuple]: the ordered scenario keys.

    """
    seen: set[tuple] = set()
    scenarios = []
    for _, row in fits.iterrows():
        if not row["model"] or pd.isna(row["W"]):
            continue
        key = scenario_key(row["setup"], row["x"], row["y"], row["z"])
        if key not in seen:
            seen.add(key)
            scenarios.append(key)
    return sorted(scenarios, key=lambda k: (k[0], k[1] or 0, k[2] or 0, k[3] if k[3] is not None else -1))


def scenario_budgets(fits: pd.DataFrame, target: tuple) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Per-machine runtime budget of one (setup, grid) scenario.

    Built from the valid fits of each machine; a missing allocation cost
    (a 1-D fit on the other operation) is treated as 0.

    Args:
        fits: the fits table of the results file.
        target: the (setup, x, y, z) scenario key.

    Returns:
        dict: per machine, the {algorithm: (W, A_malloc, A_free)} budget.

    """
    budget: dict[str, dict[str, tuple[float, float, float]]] = {}
    for _, row in fits.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != target:
            continue
        if not row["model"] or pd.isna(row["W"]):
            continue
        budget.setdefault(row["machine"], {})[row["algorithm"]] = (
            _floor0(row["W"]),
            _floor0(row["A_malloc"]) if pd.notna(row["A_malloc"]) else 0.0,
            _floor0(row["A_free"]) if pd.notna(row["A_free"]) else 0.0,
        )
    return budget


def scenario_baselines(baselines: pd.DataFrame, target: tuple) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Per-machine measured zero-delay runtime of one (setup, grid) scenario.

    Args:
        baselines: the baselines table of the results file.
        target: the (setup, x, y, z) scenario key.

    Returns:
        dict: per machine, the {algorithm: (p25, p50, p75)} baseline.

    """
    points: dict[str, dict[str, tuple[float, float, float]]] = {}
    for _, row in baselines.iterrows():
        if scenario_key(row["setup"], row["x"], row["y"], row["z"]) != target:
            continue
        points.setdefault(row["machine"], {})[row["algorithm"]] = (
            float(row["p25"]),
            float(row["p50"]),
            float(row["p75"]),
        )
    return points


class TripletCtx(NamedTuple):
    """Drawing context for one algorithm's runtime-budget triplet."""

    ax: plt.Axes
    x: float
    bar_w: float
    segments: tuple[float, float, float]
    baseline: tuple[float, float, float] | None
    algorithm: str


def draw_runtime_triplet(ctx: TripletCtx) -> float:
    """Draw one algorithm's three segment bars and its baseline point.

    The bars of the triplet are coloured per segment (W grey, A_malloc
    blue, A_free orange); the algorithm itself is named by its slot and
    the label under the x axis, and the fitted total is annotated above.

    Args:
        ctx: the triplet drawing context.

    Returns:
        float: the triplet's total height.

    """
    ax, x, bar_w = ctx.ax, ctx.x, ctx.bar_w
    for k, ((_seg, color), value) in enumerate(zip(RUNTIME_SEGMENTS, ctx.segments, strict=True)):
        ax.bar(x + (k - 1) * bar_w, value, width=bar_w, color=color, edgecolor="k", linewidth=0.5)
    total = sum(ctx.segments)
    top = max(total, ctx.baseline[2] if ctx.baseline is not None else 0.0)
    ax.text(x, top, f"{total:.3g}", ha="center", va="bottom", fontsize=8)
    if ctx.baseline is not None:
        lo, med, hi = ctx.baseline
        ax.errorbar(
            [x],
            [med],
            yerr=[[med - lo], [hi - med]],
            fmt="none",
            ecolor="k",
            elinewidth=1,
            capsize=2.5,
            zorder=5,
        )
        ax.plot([x], [med], marker="o", markersize=5, mfc="k", mec="w", mew=0.5, zorder=6)
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
    entries: list[tuple[str, dict, dict]],
    algorithms: list[str],
) -> plt.Figure:
    """Draw one figure: the runtime budget per allocator, per hardware.

    The x-axis is the hardware; under each hardware one triplet of three
    segment-coloured bars per allocator (the triplet per allocator, in the
    file's `algorithm_order`): the fitted W (grey), A_malloc (blue) and
    A_free (orange) side by side. The allocator is named by the label
    under its triplet; the fitted total is annotated above it, and the
    measured zero-delay point (median, IQR) is overlaid.

    Args:
        scenario: the (setup, x, y, z) scenario key.
        entries: per hardware, (short name, {algorithm: (W, A_malloc,
        A_free)}, {algorithm: (p25, p50, p75)}).
        algorithms: the allocators' display order.

    Returns:
        plt.Figure: the figure.

    """
    setup, x, y, z = scenario
    hw_names = [hw for hw, _budget, _baseline in entries]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if not hw_names:
        return fig
    present = [algorithm for algorithm in algorithms if any(algorithm in budget for _hw, budget, _base in entries)]
    n_algorithms = len(present)
    triplet_w = 0.8 / n_algorithms
    bar_w = 0.8 * triplet_w / 3
    max_total = 0.0
    for i, _hw in enumerate(hw_names):
        for j, algorithm in enumerate(present):
            if algorithm not in entries[i][1]:
                continue
            pos = i + (j - (n_algorithms - 1) / 2) * triplet_w
            total = draw_runtime_triplet(
                TripletCtx(ax, pos, bar_w, entries[i][1][algorithm], entries[i][2].get(algorithm), algorithm)
            )
            if algorithm in entries[i][2]:
                max_total = max(max_total, entries[i][2][algorithm][2])
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
    handles.append(Line2D([], [], marker="o", color="k", mfc="k", mec="k", label="measured (0 delay)"))
    ax.legend(handles=handles, loc="upper left", frameon=True)
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
        baselines = read_table(file, "baselines")
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
    scenarios = list_scenarios(fits)
    if name is not None:
        scenarios = [scenario for scenario in scenarios if scenario_filename(*scenario) == f"runtime-stack-{name}.pdf"]
        if not scenarios:
            available = ", ".join(scenario_name(*scenario) for scenario in list_scenarios(fits))
            print(f"no scenario {name!r} with fits (available: {available})", file=sys.stderr)
            return 1
    FIGURES.mkdir(exist_ok=True)
    for scenario in scenarios:
        budget = scenario_budgets(fits, scenario)
        baseline = scenario_baselines(baselines, scenario)
        entries = [(str(machine_hardware[m]), budget.get(m, {}), baseline.get(m, {})) for m in machines]
        path = FIGURES / scenario_filename(*scenario)
        runtime_stack_figure(scenario, entries, algorithms).savefig(path)
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
