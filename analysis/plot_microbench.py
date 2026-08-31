"""The microbenchmark's allocation-cost figures, from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads the alloc_cost tables of `output/results.h5` (frozen from the
memmansurvey suite's perf CSVs by `make microbench-results`) and draws the
suite's three allocation-cost figures, in the style of the data release's
`plot_synthetic_allocation.py`:

- `figures/microbench-allocation.pdf`: the time per operation at each
  allocation size,
- `figures/microbench-allocation-mixed.pdf`: the time per operation over each
  allocation size range,
- `figures/microbench-allocation-scaling.pdf`: the time per operation at each
  thread count (a fixed allocation size),

plus `figures/microbench-legend.pdf`, the (run, allocator) -> colour/marker
key. Each (run, allocator) series is the mean per operation with the standard
deviation as the error bar; the important allocators are drawn at full
opacity, the rest collapsed into a min-max sleeve. A run whose table is empty
is skipped.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_io import RESULTS, load_results, read_table

mpl.use("pdf")

FIGURES = Path("figures")
# The figure size, in the suite's 90%-scaled paper width.
FIGSIZE = (0.9 * 5.05, 0.9 * 2.63)
# One marker per (run, allocator) series, cycling over the markers.
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
# The allocators the suite measured, in the shared colour order; the subset
# the figures highlight.
ALLOCATORS = [
    "CUDA",
    "Gallatin",
    "Ouroboros-C-S",
    "Ouroboros-C-VA",
    "Ouroboros-C-VL",
    "Ouroboros-P-S",
    "Ouroboros-P-VA",
    "Ouroboros-P-VL",
    "RegEff-AW",
    "RegEff-C",
    "RegEff-CF",
    "RegEff-CFM",
    "RegEff-CM",
    "ScatterAlloc",
    "mallocMC",
    "XMalloc",
]
IMPORTANT_ALLOCATORS = ("Gallatin", "mallocMC", "RegEff-AW", "CUDA")
ALPHA_IMPORTANT = 1.0
ALPHA_UNIMPORTANT = 0.3
# The alloc_cost table names of the results file, in figure order.
TABLE_NAMES = ("alloc_cost", "alloc_cost_mixed", "alloc_cost_scaling")
# Each figure: the (table name, x column, categorical x, x-axis label).
FIGURE_SPECS = {
    "microbench-allocation": ("alloc_cost", "size_bytes", False, "Allocation size [bytes]"),
    "microbench-allocation-mixed": ("alloc_cost_mixed", "range", True, "Allocation range [bytes]"),
    "microbench-allocation-scaling": ("alloc_cost_scaling", "num_threads", False, "Thread count"),
}


def display_name(allocator: str) -> str:
    """Return the figure label of an allocator (mallocMC is FlatterScatter).

    Args:
        allocator: the allocator name.

    Returns:
        str: the display label.

    """
    return "FlatterScatter" if allocator == "mallocMC" else allocator


def range_key(text: str) -> tuple[int, int]:
    """Return the (lower, upper) bound of an allocation-range label.

    Args:
        text: the range label, "lo-hi".

    Returns:
        tuple[int, int]: the lower and upper bound.

    """
    low, high = text.split("-")
    return (int(low), int(high))


def style_map(jobids: list[int], allocators: list[str]) -> dict[tuple[int, str], dict[str, object]]:
    """Assign each (run, allocator) series its colour, marker, and opacity.

    The colour and marker cycles run over the runs (outermost) then the
    allocators, so each (run, allocator) pair has its own colour and marker.
    The important allocators are drawn at full opacity, the rest faded.

    Args:
        jobids: the run jobids, in figure order.
        allocators: the allocator names, in colour order.

    Returns:
        dict[tuple[int, str], dict[str, object]]: the (run, allocator) ->
        {"color", "marker", "alpha"} map.

    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {}
    for index, jobid in enumerate(jobids):
        for offset, allocator in enumerate(allocators):
            key = index * len(allocators) + offset
            styles[jobid, allocator] = {
                "color": colors[key % len(colors)],
                "marker": MARKERS[key % len(MARKERS)],
                "alpha": ALPHA_IMPORTANT if allocator in IMPORTANT_ALLOCATORS else ALPHA_UNIMPORTANT,
            }
    return styles


def draw_series(ax: plt.Axes, x: np.ndarray, stats: pd.DataFrame, label: str, style: dict[str, object]) -> None:
    """Draw one (run, allocator) series as an error bar (the mean, std).

    Args:
        ax: the axis to draw into.
        x: the series' x values.
        stats: the series' per-x statistics (the mean and std columns).
        label: the legend label.
        style: the series' {"color", "marker", "alpha"}.

    """
    ax.errorbar(
        x,
        stats["mean_ms"].to_numpy(),
        yerr=stats["std_ms"].to_numpy(),
        linestyle="none",
        marker=style["marker"],
        label=label,
        color=style["color"],
        alpha=float(style["alpha"]),
    )


def draw_sleeve(ax: plt.Axes, series: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Fill the min-max envelope of the non-important series' means.

    Args:
        ax: the axis to draw into.
        series: the (x, mean) series of the non-important allocators.

    """
    if not series:
        return
    points = pd.DataFrame({"x": np.concatenate([x for x, _ in series]), "y": np.concatenate([y for _, y in series])})
    bounds = points.groupby("x")["y"].agg(["min", "max"]).sort_index()
    ax.fill_between(bounds.index, bounds["min"], bounds["max"], color="0.6", alpha=ALPHA_UNIMPORTANT, label="others")


def allocation_figure(data: pd.DataFrame, x_column: str, *, categorical: bool, xlabel: str) -> plt.Figure:
    """Draw one allocation-cost figure: error bars and the others' sleeve.

    Args:
        data: one alloc_cost table.
        x_column: the table's x column (the size, the range, or the threads).
        categorical: True when the x values are drawn as categories (the
        allocation ranges) instead of on a logarithmic axis.
        xlabel: the x-axis label.

    Returns:
        plt.Figure: the chart.

    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    jobids = sorted(int(jobid) for jobid in data["jobid"].drop_duplicates())
    styles = style_map(jobids, ALLOCATORS)
    positions = (
        {value: index for index, value in enumerate(sorted(data[x_column].drop_duplicates(), key=range_key))}
        if categorical
        else None
    )
    others: list[tuple[np.ndarray, np.ndarray]] = []
    for (jobid, allocator), style in styles.items():
        sub = data[(data["jobid"] == jobid) & (data["allocator"] == allocator)]
        if sub.empty:
            continue
        x = sub[x_column].map(positions).to_numpy() if categorical else sub[x_column].to_numpy()
        if allocator in IMPORTANT_ALLOCATORS:
            draw_series(ax, x, sub, display_name(allocator), style)
        else:
            others.append((x, sub["mean_ms"].to_numpy()))
    draw_sleeve(ax, others)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time per op. [ms]")
    ax.set_yscale("log", base=2)
    if categorical:
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(list(positions))
    else:
        ax.set_xscale("log", base=2)
    return fig


def legend_figure(styles: dict[tuple[int, str], dict[str, object]]) -> plt.Figure:
    """Draw the (run, allocator) -> colour/marker legend figure.

    Args:
        styles: the (run, allocator) -> {"color", "marker", "alpha"} map.

    Returns:
        plt.Figure: the legend.

    """
    entries = list(styles.items())
    ncol = min(4, max(1, len(entries)))
    nrows = max(1, -(-len(entries) // ncol))
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1] * nrows / 4.0))
    for (jobid, allocator), style in entries:
        ax.plot(
            [0],
            [0],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            label=f"{jobid} / {display_name(allocator)}",
        )
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.axis("off")
    ax.legend(loc="center", ncol=ncol)
    return fig


def save_figure(fig: plt.Figure, name: str) -> None:
    """Tight-layout, save one figure under `figures/`, and report it.

    A figure whose decorations do not fit (a dense legend) skips the tight
    layout with a warning; that one warning is suppressed.

    Args:
        fig: the figure to save.
        name: the file name, relative to `figures/`.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout")
        fig.tight_layout()
    fig.savefig(FIGURES / name)
    print(f"wrote {FIGURES / name}")


def build_figures(tables: dict[str, pd.DataFrame]) -> list[tuple[plt.Figure, str]]:
    """Build each non-empty allocation figure and the legend.

    Args:
        tables: the alloc_cost tables, by table name.

    Returns:
        list[tuple[plt.Figure, str]]: the (figure, file name) pairs, in order.

    """
    data = pd.concat([table for table in tables.values() if not table.empty], ignore_index=True)
    jobids = sorted(int(jobid) for jobid in data["jobid"].drop_duplicates())
    styles = style_map(jobids, ALLOCATORS)
    figures = [
        (allocation_figure(tables[table], x_column, categorical=categorical, xlabel=xlabel), f"{figure}.pdf")
        for figure, (table, x_column, categorical, xlabel) in FIGURE_SPECS.items()
        if not tables[table].empty
    ]
    figures.append((legend_figure(styles), "microbench-legend.pdf"))
    return figures


def main(*, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the microbenchmark's allocation-cost figures from one results file.

    Args:
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
        tables = {name: (read_table(file, name) if name in file else pd.DataFrame()) for name in TABLE_NAMES}
    if all(table.empty for table in tables.values()):
        print("no microbenchmark alloc_cost tables in the results file; nothing to draw", file=sys.stderr)
        return 0
    FIGURES.mkdir(exist_ok=True)
    for fig, name in build_figures(tables):
        save_figure(fig, name)
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The microbenchmark's allocation-cost figures from output/results.h5 "
        "(figures/microbench-allocation*.pdf, microbench-legend.pdf)."
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
    sys.exit(main(show=args.show, results=args.results))
