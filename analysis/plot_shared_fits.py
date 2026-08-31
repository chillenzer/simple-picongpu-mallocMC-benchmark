"""Per-machine forest figures of the shared-parameter (combined) performance-model fits.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws,
for every sweep machine with a combined fit, a forest figure
(`figures/sweeps-shared-<machine>.pdf`), one row per (scenario, algorithm)
and one column per parameter. The W, N_malloc, and N_free columns show each
algorithm's individual fit (dot, error bar) against the value fitted once
across the algorithms (dashed vertical with a plus marker); the A_malloc
and A_free columns show, per algorithm, the individual value (open marker)
and the value of the combined fit (filled marker), joined by a segment.
Every A_* value is annotated with its slack ratio f -- the absorbed
delay as a share of the zero-delay runtime T0 = W + A_malloc + A_free
(T0 is summed over the terms the row carries; f is a convention-dependent
ratio, not a runtime budget). The
x-axes are logarithmic. By default every machine gets its figure,
`--machine` restricts the run to one.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_io import (
    RESULTS,
    algorithm_order,
    load_results,
    machine_titles,
    read_table,
    scenario_key,
    scenario_name,
    sweep_machine_labels,
)

mpl.use("pdf")

FIGURES = Path("figures")

# One figure column per parameter: the field name (present in both the
# individual `fits` and the `shared_fits` tables), the panel kind, and the
# column title. "shared" panels compare every algorithm's individual value
# against the value fitted once across the algorithms; "compare" panels
# show, per algorithm, the individual value against the combined fit's.
PANELS = (
    ("W", "shared", "W: shared runtime (s)"),
    ("N_malloc", "shared", "N_malloc: calls per run"),
    ("N_free", "shared", "N_free: calls per run"),
    ("A_malloc", "compare", "A_malloc: absorbed delay (s)"),
    ("A_free", "compare", "A_free: absorbed delay (s)"),
)
# One distinct marker per algorithm, in the file's algorithm order.
MARKERS = ("o", "s", "^", "D")


class ForestData(NamedTuple):
    """The drawing data of one machine's shared-fit forest figure.

    `rows` is the (scenario key, algorithm) row order; the `colors` and
    `markers` maps assign, per scenario and per algorithm, the drawing
    colour and marker; `scen_ref` is the first `shared_fits` row per
    scenario (carrying the shared values), `individuals` the individual
    fits and `shared_rows` the `shared_fits` rows, keyed by
    (scenario, algorithm).
    """

    rows: list
    colors: dict
    markers: dict
    scen_ref: dict
    individuals: dict
    shared_rows: dict


def _scen(row: pd.Series) -> tuple:
    """Return the (setup, x, y, z) scenario key of one fits row.

    Args:
        row: one row of the `fits` or `shared_fits` table.

    Returns:
        tuple: the scenario key.

    """
    return scenario_key(row["setup"], row["x"], row["y"], row["z"])


def _build_rows(shared: pd.DataFrame, algorithms: list[str]) -> tuple[list, list, dict]:
    """Build the (scenario, algorithm) row order and the per-scenario data.

    Args:
        shared: the `shared_fits` rows of one machine.
        algorithms: the file's algorithm order.

    Returns:
        tuple: (the row list of (scenario key, algorithm), the scenario
        order, the first `shared_fits` row per scenario).

    """
    scen_order: list[tuple] = []
    for _, row in shared.iterrows():
        key = _scen(row)
        if key not in scen_order:
            scen_order.append(key)
    scen_ref = {key: shared[shared.apply(lambda r: _scen(r) == key, axis=1)].iloc[0] for key in scen_order}
    rows: list[tuple] = []
    for key in scen_order:
        present = list(dict.fromkeys(shared[shared.apply(lambda r: _scen(r) == key, axis=1)]["algorithm"]))
        present.sort(key=lambda a: algorithms.index(a) if a in algorithms else len(algorithms))
        rows.extend((key, algo) for algo in present)
    return rows, scen_order, scen_ref


def _hbar(ax: plt.Axes, value: float, error: float, y: float, color: str) -> None:
    """Draw a horizontal 1-sigma error bar where it stays positive.

    Args:
        ax: the axis.
        value: the bar's centre (must be positive for the log axis).
        error: the 1-sigma error.
        y: the bar's row.
        color: the bar's colour.

    """
    if np.isfinite(error) and value > 0 and value - error > 0:
        ax.errorbar([value], [y], xerr=error, fmt="none", color=color, elinewidth=1, capsize=3, zorder=3)


def _draw_shared_panel(ax: plt.Axes, data: ForestData, field: str, *, first: bool) -> None:
    """Draw one shared-parameter column: the individual dots and the shared value.

    Each algorithm's individual value is drawn as a dot (its error bar
    included); the value fitted once across the algorithms is drawn as a
    dashed vertical spanning the scenario's rows, with a plus marker and
    its own error bar.

    Args:
        ax: the axis to fill.
        data: the figure's drawing data.
        field: the parameter field name.
        first: whether this panel carries the figure legend.

    """
    err_field = f"{field}_err"
    span: dict = {}
    for i, (scen, _algo) in enumerate(data.rows):
        if scen not in span:
            span[scen] = (i, i)
        else:
            span[scen] = (min(span[scen][0], i), max(span[scen][1], i))
    shared_labeled = False
    for scen, (lo, hi) in span.items():
        ref = data.scen_ref[scen]
        value, error = float(ref[field]), float(ref[err_field])
        if value <= 0:
            continue
        center = (lo + hi) / 2
        label = "shared fit" if first and not shared_labeled else None
        shared_labeled = True
        ax.plot(
            [value, value],
            [lo - 0.45, hi + 0.45],
            color=data.colors[scen],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
            label=label,
        )
        ax.scatter(
            [value],
            [center],
            marker="P",
            s=110,
            facecolors="none",
            edgecolors=data.colors[scen],
            linewidth=1.5,
            zorder=5,
        )
        _hbar(ax, value, error, center, data.colors[scen])
    ind_labeled = False
    for i, (scen, algo) in enumerate(data.rows):
        ind = data.individuals.get((scen, algo))
        if ind is None or pd.isna(ind[field]) or ind[field] <= 0:
            continue
        value, error = float(ind[field]), float(ind[err_field])
        label = "individual fit" if first and not ind_labeled else None
        ind_labeled = True
        ax.errorbar(
            [value],
            [i],
            xerr=None,
            yerr=0.18,
            linestyle="none",
            marker=data.markers[algo],
            markersize=6,
            color=data.colors[scen],
            label=label,
            zorder=4,
        )
        _hbar(ax, value, error, i, data.colors[scen])


def _op_fraction(row: pd.Series, field: str) -> float | None:
    """Return the slack ratio f = A/T0 of one fits row's A_* value.

    f is the absorbed delay as a share of T0, the zero-delay runtime, the
    sum of the W, A_malloc and A_free terms the row carries.

    Args:
        row: one row of the `fits` or `shared_fits` table.
        field: the A_* field to express as a fraction.

    Returns:
        float | None: the fraction (0 when T0 is degenerate), or None when
        the value is unavailable.

    """
    if pd.isna(row[field]):
        return None
    total = sum(float(row[t]) for t in ("W", "A_malloc", "A_free") if pd.notna(row[t]))
    return float(row[field]) / total if total > 0 else 0.0


def _fraction_text(fraction: float) -> str:
    """Return one slack ratio's display label, e.g. `f=12.9%`.

    Args:
        fraction: the slack ratio (absorbed delay over zero-delay runtime), in [0, 1].

    Returns:
        str: the label.

    """
    return f"f={100 * fraction:.3g}%"


def _draw_compare_panel(ax: plt.Axes, data: ForestData, field: str, *, first: bool) -> None:
    """Draw one A-column: per algorithm, the individual and combined values.

    Each algorithm's individual value (open marker) is joined by a segment
    to the value of the combined fit (filled marker), each with its
    error bar and annotated above (individual) or below (combined) with
    its slack ratio f.

    Args:
        ax: the axis to fill.
        data: the figure's drawing data.
        field: the parameter field name.
        first: whether this panel carries the figure legend.

    """
    err_field = f"{field}_err"
    ind_labeled = False
    shared_labeled = False
    for i, (scen, algo) in enumerate(data.rows):
        ind = data.individuals.get((scen, algo))
        row = data.shared_rows.get((scen, algo))
        value_i = None
        if ind is not None and pd.notna(ind[field]) and ind[field] > 0:
            value_i = (float(ind[field]), float(ind.get(err_field, np.nan)))
        value_j = None
        if row is not None and pd.notna(row[field]) and row[field] > 0:
            value_j = (float(row[field]), float(row.get(err_field, np.nan)))
        if value_i is None and value_j is None:
            continue
        if value_i is not None and value_j is not None:
            ax.plot([value_i[0], value_j[0]], [i, i], color=data.colors[scen], linewidth=1.2, zorder=2)
        if value_i is not None:
            label = "individual fit" if first and not ind_labeled else None
            ind_labeled = True
            ax.scatter(
                [value_i[0]],
                [i],
                marker=data.markers[algo],
                s=55,
                facecolors="none",
                edgecolors=data.colors[scen],
                linewidth=1.4,
                zorder=4,
                label=label,
            )
            _hbar(ax, value_i[0], value_i[1], i, data.colors[scen])
            fraction = _op_fraction(ind, field)
            if fraction is not None:
                ax.text(
                    value_i[0],
                    i - 0.22,
                    _fraction_text(fraction),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=data.colors[scen],
                    zorder=6,
                )
        if value_j is not None:
            label = "shared fit" if first and not shared_labeled else None
            shared_labeled = True
            ax.scatter(
                [value_j[0]], [i], marker=data.markers[algo], s=45, color=data.colors[scen], zorder=5, label=label
            )
            _hbar(ax, value_j[0], value_j[1], i, data.colors[scen])
            fraction = _op_fraction(row, field)
            if fraction is not None:
                ax.text(
                    value_j[0],
                    i + 0.22,
                    _fraction_text(fraction),
                    ha="center",
                    va="top",
                    fontsize=6,
                    color=data.colors[scen],
                    zorder=6,
                )


def _snug_xlim(ax: plt.Axes) -> None:
    """Limit a logarithmic axis snugly to its drawn content.

    Args:
        ax: the axis to limit (x direction, in data coordinates).

    """
    parts = []
    for child in ax.get_children():
        if hasattr(child, "get_xdata"):
            parts.append(np.asarray(child.get_xdata(), dtype=float))
        elif hasattr(child, "get_segments"):
            segs = child.get_segments()
            if segs:
                parts.append(np.concatenate([seg[:, 0] for seg in segs]))
    values = np.concatenate(parts) if parts else np.array([])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return
    lo, hi = float(values.min()), float(values.max())
    pad = 0.03 * max(math.log10(hi / lo), 1e-9)
    ax.set_xlim(10 ** (math.log10(lo) - pad), 10 ** (math.log10(hi) + pad))


def plot_machine(title: str, shared: pd.DataFrame, fits: pd.DataFrame, algorithms: list[str]) -> plt.Figure:
    """Build one machine's shared-fit forest figure.

    Args:
        title: the figure title (the hardware the runs were made on).
        shared: the machine's `shared_fits` rows.
        fits: the machine's individual fits rows.
        algorithms: the file's algorithm order.

    Returns:
        plt.Figure: the figure.

    """
    rows, scen_order, scen_ref = _build_rows(shared, algorithms)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    marker_order = list(dict.fromkeys(algo for _scen, algo in rows))
    data = ForestData(
        rows,
        {scen: cycle[i % len(cycle)] for i, scen in enumerate(scen_order)},
        {algo: MARKERS[i % len(MARKERS)] for i, algo in enumerate(marker_order)},
        scen_ref,
        {(_scen(row), row["algorithm"]): row for _, row in fits.iterrows() if pd.notna(row["model"])},
        {(_scen(row), row["algorithm"]): row for _, row in shared.iterrows()},
    )
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.1 * len(PANELS), 4.6), sharey=True, layout="constrained")
    fig.suptitle(title)
    for i, (ax, (field, kind, label)) in enumerate(zip(axes, PANELS, strict=True)):
        if kind == "shared":
            _draw_shared_panel(ax, data, field, first=i == 0)
        else:
            _draw_compare_panel(ax, data, field, first=i == 3)
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xlabel("fitted value")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("")
    axes[0].set_yticks(list(range(len(rows))))
    axes[0].set_yticklabels([f"{scenario_name(*scen)}\n{algo}" for scen, algo in rows], fontsize=7)
    axes[0].invert_yaxis()
    for ax in axes:
        _snug_xlim(ax)
        ax.grid(which="both", color="0.9", linewidth=0.5)
        ax.margins(y=0.02)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
    return fig


def main(*, machine: str | None = None, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the per-machine shared-fit forest figures from one results file.

    Args:
        machine: the machine to draw, or None for every machine with a
        combined fit.
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
        shared_fits = read_table(file, "shared_fits") if "shared_fits" in file else pd.DataFrame()
        fits = read_table(file, "fits")
        algorithms = algorithm_order(file)
        sweep = sweep_machine_labels(file)
        titles = machine_titles(file)
    if shared_fits.empty:
        print("no shared (combined) fits in the results file; nothing to draw", file=sys.stderr)
        return 0
    machines = [m for m in sweep if m in set(shared_fits["machine"])]
    if machine is not None and machine not in machines:
        available = ", ".join(machines) or "none"
        print(f"machine {machine!r} has no combined fit (available: {available})", file=sys.stderr)
        return 1
    if machine is not None:
        machines = [machine]
    FIGURES.mkdir(exist_ok=True)
    for m in machines:
        shared = shared_fits[shared_fits["machine"] == m]
        fig = plot_machine(titles.get(m, m), shared, fits[fits["machine"] == m], algorithms)
        fig.savefig(FIGURES / f"sweeps-shared-{m}.pdf")
        print(f"wrote {FIGURES / f'sweeps-shared-{m}.pdf'}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-machine shared-fit forest figures from output/results.h5"
        " (figures/sweeps-shared-<machine>.pdf)."
    )
    parser.add_argument(
        "--machine",
        help="draw only this machine's figure (default: every machine with a combined fit)",
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
    sys.exit(main(machine=args.machine, show=args.show, results=args.results))
