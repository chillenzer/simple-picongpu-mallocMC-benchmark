"""The native-cost comparison: the fitted absorbed slack vs the microbenchmark's c_a.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws the
comparison between the fitted per-call absorbed slack (A/N, the Tier-2
quantity) and the independently-measured native per-call cost c_a of the
microbenchmark, one panel per operation (malloc left, free right). Each point
is a fitted group (machine, setup, algorithm, grid), coloured by machine
(hardware); the dotted diagonal is the "absorbed slack = native cost"
hypothesis the review argues is not supported. Only groups with a matching
microbenchmark cost are shown; without one, the figure is empty and a note is
printed. Saved to `figures/native-cost.pdf`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_io import HARDWARE_ORDER, RESULTS, load_results, machine_titles, read_table

mpl.use("pdf")

FIGURES = Path("figures")
# One (operation, absorbed-slack column, native-cost column) panel.
PANELS = (
    ("malloc", "A_malloc", "N_malloc", "c_a_malloc_ns"),
    ("free", "A_free", "N_free", "c_a_free_ns"),
)


def _group_label(row: pd.Series) -> str:
    """Format one fit group as a short label (setup, grid, algorithm).

    Args:
        row: a row of the fits table.

    Returns:
        str: the label, e.g. `KHI 128x128x128 FlatterScatter`.

    """
    dims = [int(row[k]) for k in ("x", "y", "z") if pd.notna(row[k])]
    return f"{row['setup']} {'x'.join(map(str, dims))} {row['algorithm']}"


def _panel_points(
    fits: pd.DataFrame, a_col: str, n_col: str, c_col: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """One panel's (native cost, absorbed slack) points, in microseconds.

    Args:
        fits: the fits table.
        a_col: the fitted absorbed-delay column (s).
        n_col: the fitted call-count column.
        c_col: the native per-call cost column (ns).

    Returns:
        tuple: (c_a in us, absorbed slack in us, the labels, the machines).

    """
    sel = fits
    mask = (
        sel[a_col].notna()
        & sel[n_col].notna()
        & sel[c_col].notna()
        & (sel[n_col] > 0)
        & (sel[c_col] > 0)
        & np.isfinite(sel[a_col])
    )
    sel = sel[mask]
    c_us = sel[c_col].to_numpy() / 1e3
    slack_us = (sel[a_col] / sel[n_col]).to_numpy() * 1e6
    labels = [_group_label(row) for _, row in sel.iterrows()]
    machines = [str(m) for m in sel["machine"].astype(str)]
    return c_us, slack_us, labels, machines


def make_figure(fits: pd.DataFrame, titles: dict) -> plt.Figure:
    """Draw the absorbed-slack-vs-native-cost comparison (one panel per operation).

    Args:
        fits: the fits table of the results file.
        titles: the machine -> hardware title map (for the machine colours' legend).

    Returns:
        plt.Figure: the chart.

    """
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5.5), sharey=True)
    machines = [m for m in HARDWARE_ORDER if str(m) in set(fits["machine"].astype(str))]
    if not machines:
        machines = list(fits["machine"].astype(str).drop_duplicates())
    for ax, (op, a_col, n_col, c_col) in zip(axes, PANELS, strict=True):
        c_us, slack_us, labels, machines_pts = _panel_points(fits, a_col, n_col, c_col)
        ax.set_title(f"{op}: absorbed slack vs native cost")
        ax.set_xlabel(f"native cost c_a ({op}), us per call")
        ax.set_xscale("log")
        if op == "malloc":
            ax.set_ylabel("fitted absorbed slack A/N, us per call")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("")
        if not c_us.size:
            ax.text(
                0.5,
                0.5,
                "no microbenchmark cost\nfor any fitted group",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        # The hypothesis line: absorbed slack = native cost (the review's rejected reading).
        span = np.logspace(np.log10(c_us.min() * 0.5), np.log10(max(c_us.max(), slack_us.max()) * 2), 50)
        ax.plot(span, span, ":", color="0.5", linewidth=1)
        for c, s, label, machine in zip(c_us, slack_us, labels, machines_pts, strict=True):
            color = prop_colors[machines.index(machine) % 10]
            ax.plot(c, s, "o", color=color, markersize=6, alpha=0.85)
            ax.annotate(label, (c, s), textcoords="offset points", xytext=(4, 4), fontsize=7)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=titles.get(m, m),
                markerfacecolor=prop_colors[i % 10],
            )
            for i, m in enumerate(machines)
        ]
        handles.append(plt.Line2D([0], [0], color="0.5", linestyle=":", label="A/N = c_a"))
        ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def main(*, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the native-cost comparison figure from one results file.

    Args:
        show: display the figure in a window (blocking).
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
        titles = machine_titles(file)
    if "c_a_malloc_ns" not in fits.columns or "c_a_free_ns" not in fits.columns:
        print("the fits table has no native-cost columns; rerun `python3 analysis/compute_results.py`", file=sys.stderr)
        return 1
    if fits["c_a_malloc_ns"].isna().all() and fits["c_a_free_ns"].isna().all():
        print("no microbenchmark cost matches any fitted group (no native-cost figure)")
        return 0
    FIGURES.mkdir(exist_ok=True)
    fig = make_figure(fits, titles)
    fig.savefig(FIGURES / "native-cost.pdf")
    print(f"wrote {FIGURES / 'native-cost.pdf'}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The absorbed-slack-vs-native-cost comparison from output/results.h5 (figures/native-cost.pdf)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the figure in a window (blocking); by default it is only saved",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results file (default: %(default)s)",
    )
    args = parser.parse_args()
    sys.exit(main(show=args.show, results=args.results))
