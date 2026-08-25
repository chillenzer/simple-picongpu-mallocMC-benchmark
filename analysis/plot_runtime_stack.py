"""Per-scenario runtime-budget figures from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws,
for every (setup, grid) scenario with fits, one figure
(`figures/runtime-stack-<setup>-<grid>.pdf`): the x-axis is the hardware,
and under each hardware a bar per allocator (in the file's
`algorithm_order`) stacks the fitted W, A_malloc and A_free up to the
total runtime, overlaid with the measured zero-delay runtime (median,
IQR error bar). By default every scenario gets its figure, `--name`
restricts the run to one, e.g. `--name FoilLCT-256x1280`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from results_io import (
    RESULTS,
    algorithm_order,
    load_results,
    read_table,
    scenario_filename,
    scenario_key,
    scenario_name,
)

mpl.use("pdf")

FIGURES = Path("figures")
# the stacked segments of the runtime-budget figure, in stacking order
# (bottom to top), as (legend label, hatch) pairs; the fill colour is
# taken from the default property cycle.
RUNTIME_SEGMENTS = (("W", "//"), ("A_malloc", "\\\\"), ("A_free", "xx"))


def _floor0(value: float) -> float:
    """Clamp a fitted segment to a non-negative display height.

    Args:
        value: the fitted segment value in seconds.

    Returns:
        float: the value, floored at zero.

    """
    return max(0.0, float(value))


def list_scenarios(fits: pd.DataFrame) -> list[tuple]:
    """Collect the unique (setup, x, y, z) scenarios across all machines.

    Args:
        fits: the fits table of the results file.

    Returns:
        list[tuple]: the ordered scenario keys.

    """
    seen: set[tuple] = set()
    scenarios = []
    for _, row in fits.iterrows():
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


class BarCtx(NamedTuple):
    """Drawing context for one runtime-budget bar."""

    ax: plt.Axes
    pos: float
    bar_w: float
    segments: tuple[float, float, float]
    algorithm: str
    baseline: tuple[float, float, float] | None
    show_seg_labels: bool
    show_point_label: bool


def draw_runtime_bar(ctx: BarCtx) -> float:
    """Draw one stacked runtime bar with its algorithm sub-label and optional baseline point.

    Args:
        ctx: the bar drawing context.

    Returns:
        float: the bar's total height.

    """
    ax, pos, bar_w = ctx.ax, ctx.pos, ctx.bar_w
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bottom = 0.0
    for i, ((seg_label, hatch), value) in enumerate(zip(RUNTIME_SEGMENTS, ctx.segments, strict=True)):
        ax.bar(
            pos,
            value,
            width=bar_w,
            bottom=bottom,
            color=cycle[i],
            hatch=hatch,
            edgecolor="k",
            linewidth=0.5,
            label=seg_label if ctx.show_seg_labels else None,
        )
        bottom += value
    ax.text(pos, bottom, f"{bottom:.3g}", ha="center", va="bottom", fontsize=8)
    ax.text(
        pos,
        -0.14,
        ctx.algorithm,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="0.45",
    )
    if ctx.baseline is not None:
        lo, med, hi = ctx.baseline
        ax.errorbar(
            [pos],
            [med],
            yerr=[[med - lo], [hi - med]],
            fmt="none",
            ecolor="k",
            elinewidth=1,
            capsize=0,
            zorder=5,
        )
        ax.plot(
            [pos],
            [med],
            marker="o",
            markersize=5,
            mfc="k",
            mec="w",
            mew=0.5,
            zorder=6,
            label="measured (0 delay)" if ctx.show_point_label else None,
        )
    return bottom


def runtime_stack_figure(
    scenario: tuple,
    entries: list[tuple[str, dict, dict]],
    algorithms: list[str],
) -> plt.Figure:
    """Draw one figure: the runtime budget stacked per allocator, per hardware.

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
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if not hw_names:
        return fig
    bar_w = 0.8 / len(algorithms)
    max_total = 0.0
    first_bar = True
    first_point = True
    for i, _hw in enumerate(hw_names):
        for j, algorithm in enumerate(algorithms):
            if algorithm not in entries[i][1]:
                continue
            pos = i + (j - (len(algorithms) - 1) / 2) * bar_w
            total = draw_runtime_bar(
                BarCtx(
                    ax,
                    pos,
                    bar_w,
                    entries[i][1][algorithm],
                    algorithm,
                    entries[i][2].get(algorithm),
                    first_bar,
                    first_point,
                )
            )
            first_bar = False
            if algorithm in entries[i][2]:
                first_point = False
                max_total = max(max_total, entries[i][2][algorithm][2])
            max_total = max(max_total, total)
    if max_total <= 0:
        return fig
    ax.set_xticks(range(len(hw_names)))
    ax.set_xticklabels(hw_names, fontsize=9)
    ax.set_xlim(-0.6, len(hw_names) - 0.4)
    ax.set_ylim(0, max_total * 1.12)
    ax.set_ylabel("runtime (s)")
    ax.set_xlabel("hardware")
    ax.legend(loc="upper left")
    fig.suptitle(f"Runtime budget: {scenario_name(setup, x, y, z)}")
    fig.subplots_adjust(bottom=0.18)
    return fig


def _machine_names(fits: pd.DataFrame, baselines: pd.DataFrame) -> list:
    """Return the machine order of a scenario's figure: first appearance in fits, then baselines.

    Args:
        fits: the fits table of the results file.
        baselines: the baselines table of the results file.

    Returns:
        list: the machine labels, in display order.

    """
    return list(pd.concat([fits["machine"], baselines["machine"]]).drop_duplicates())


def main(*, name: str | None = None, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the per-scenario runtime-budget figures from one results file.

    Args:
        name: the scenario to draw, as in the figure filename without the
        `runtime-stack-` prefix and `.pdf` suffix, e.g. `FoilLCT-256x1280`;
        None for every scenario with fits.
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
    if fits.empty:
        print("no fits in the results file", file=sys.stderr)
        return 1
    machine_hardware = runs.drop_duplicates("machine").set_index("machine")["hardware"].to_dict()
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
        entries = [
            (str(machine_hardware[m]).split()[-1], budget.get(m, {}), baseline.get(m, {}))
            for m in _machine_names(fits, baselines)
            if m in budget or m in baseline
        ]
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
