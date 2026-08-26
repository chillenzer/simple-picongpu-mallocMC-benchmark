"""Per-machine delay-sweep figures from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws,
for every sweep machine with delay runs, one figure titled by the
machine's hardware title (from the file's `machine_titles`): one row per
algorithm present in the data (in the
`algorithm_order` of the file), the malloc delay sweep (free delay 0) on
the left and the free delay sweep (malloc delay 0) on the right, each axis
titled after the swept delay, all axes sharing the x- and y-axes and
drawing each scenario (setup, grid) with the same colour and marker, on
every axis that carries it. Each swept curve shows the median with IQR
error bars and overlays the fitted model
(solid), the extrapolation to A_malloc = A_free = 0 (dashed) and, for
two-operation fits, the intermediate extrapolation that keeps only the held
operation's native cost (dotted); the gaps at the smallest delay are marked
by a short line. Each fit line carries a transparent sleeve, the
25/75-percentile envelope of the model evaluated at 512 parameter draws
from the fitted parameters and their covariance (a bootstrap, since the
parameters enter the model non-linearly). The figure is saved to
`figures/sweeps-<machine>.pdf`; by default every machine gets its figure,
`--machine` restricts the run to one.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import amdahl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from results_io import (
    RESULTS,
    algorithm_order,
    grid_label,
    load_results,
    machine_titles,
    read_fit_covs,
    read_table,
    sweep_machine_labels,
)
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY

mpl.use("pdf")

FIGURES = Path("figures")

# one distinct marker per (setup, grid) series, shared by all axes; the
# series' colour is likewise assigned once per series (from the default
# property cycle), in the same order as its marker.
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
# A gap marker is only drawn when the model difference exceeds this (s).
VISIBLE_GAP_S = 1e-9
# Padding, in decades, left between the drawn content and the frame by
# `snug_ylims` and `snug_xlims`; both axes are logarithmic, so the margin is
# applied in log space, where a small linear percentage is only a vanishing
# fraction of a decade.
_LOG_PAD_DEC = 0.05


class Fit1d(NamedTuple):
    """1-D Amdahl fit on a single delay; `direction` names that delay.

    `cov` is the `(fit_params, pcov)` pair used for the bootstrap sleeves,
    or None when the covariance is unavailable.
    """

    direction: str
    W: float
    N: float
    A: float
    s0: float
    cov: tuple | None


class Fit2d(NamedTuple):
    """2-D Amdahl fit (both delays vary); the delays are in seconds.

    `cov` is the `(fit_params, pcov)` pair used for the bootstrap sleeves,
    or None when the covariance is unavailable.
    """

    W: float
    n_malloc: float
    n_free: float
    a_malloc: float
    a_free: float
    m0: float
    f0: float
    cov: tuple | None


class Curve(NamedTuple):
    """The per-curve drawing context for a fitted model line.

    `x_ns` is the x-grid in nanoseconds (the axis), `x_s` the same grid in
    seconds (the model), `x0` the smallest delay of the curve, and `held_s`
    the held (secondary) delay in seconds.
    """

    ax: plt.Axes
    x_ns: np.ndarray
    x_s: np.ndarray
    x0: float
    held_s: float
    params: Fit1d | Fit2d
    short: str
    color: str


class Cluster(NamedTuple):
    """The per-cluster drawing context shared by all of its curves."""

    ax: plt.Axes
    x_delay: str
    short: str
    secondary_short: str
    series_markers: dict
    series_colors: dict
    fits_by_key: dict


class MachineData(NamedTuple):
    """One machine's drawing data, as read from the results file.

    `covs` is this machine's (setup, algorithm, grid label) to
    (fit_params, pcov) map (the `fits/cov/` entries).
    """

    machine: str
    stats: pd.DataFrame
    fits: pd.DataFrame
    covs: dict


_ModelFn = Callable[[np.ndarray], np.ndarray]


def _group_key(name: tuple) -> tuple:
    # NaN group values (missing z in 2D runs) do not compare equal, so
    # canonicalize them; the key is only used for dictionary lookups.
    return tuple(None if isinstance(k, float) and np.isnan(k) else k for k in name)


def series_label(info: tuple, secondary: str = "free") -> str:
    """Format a (setup, algorithm, grid, held-delay) group key as a legend label.

    Args:
        info: the group key (setup, algorithm, x, y, z, <secondary delay>).
        secondary: the name of the held (secondary) delay.

    Returns:
        str: the formatted legend label.

    """
    # The algorithm is not part of the label: in multi-algorithm figures each
    # row is one algorithm and is named by its row header.
    grid_string = "x".join(map(str, map(int, np.asarray(info[2:5])[~np.isnan(info[2:5])])))
    suffix = "" if info[5] == 0 else f", {secondary}: {int(info[5])} ns"
    return f"{info[0]} {grid_string}{suffix}"


def legend_first(axes: Iterable[plt.Axes]) -> None:
    """Show the legend, at matplotlib's auto "best" location, on the left-most axis with a curve.

    Args:
        axes: the axes, left to right.

    """
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
            break


def _snug_shared_axis(fig: plt.Figure, axis: str) -> None:
    """Fit one shared logarithmic axis snugly to everything drawn on it.

    The default autoscale spans every artist and pads that range with a 5%
    log-space margin, leaving a generous band around the content. Re-limit the
    shared axis to the union, over all of the figure's axes, of the drawn data
    points (medians and their IQR bars), the fitted and extrapolation lines
    (and their A/f gap markers), and the bootstrap error sleeves, with
    ``_LOG_PAD_DEC`` of a decade left between the content and the frame so
    nothing is clipped.

    Args:
        fig: the figure whose shared axis is re-limited.
        axis: "x" or "y".

    """
    getter = "get_xdata" if axis == "x" else "get_ydata"
    index = 0 if axis == "x" else 1
    lows: list[float] = []
    highs: list[float] = []
    for ax in fig.axes:
        for child in ax.get_children():
            if hasattr(child, getter):
                values = np.asarray(getattr(child, getter)(), dtype=float)
            elif hasattr(child, "get_segments"):
                segments = child.get_segments()
                values = np.concatenate([seg[:, index] for seg in segments]) if len(segments) else np.array([])
            elif hasattr(child, "get_paths"):
                paths = [path for path in child.get_paths() if len(path.vertices)]
                values = np.concatenate([path.vertices[:, index] for path in paths]) if paths else np.array([])
            else:
                continue
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    if not lows:
        return
    # Pad in log space, so the margin is a visible, symmetric fraction of a
    # decade on the log-scale axis.
    lo = 10 ** (math.log10(min(lows)) - _LOG_PAD_DEC)
    hi = 10 ** (math.log10(max(highs)) + _LOG_PAD_DEC)
    setter = "set_xlim" if axis == "x" else "set_ylim"
    for ax in fig.axes:
        if ax.containers:
            getattr(ax, setter)(lo, hi)


def snug_ylims(fig: plt.Figure) -> None:
    """Fit the shared y-axis snugly to everything drawn on it (see `_snug_shared_axis`).

    Args:
        fig: the figure whose shared y-axis is re-limited.

    """
    _snug_shared_axis(fig, "y")


def snug_xlims(fig: plt.Figure) -> None:
    """Fit the shared x-axis snugly to everything drawn on it (see `_snug_shared_axis`).

    Args:
        fig: the figure whose shared x-axis is re-limited.

    """
    _snug_shared_axis(fig, "x")


def collect_fits(fits: pd.DataFrame, covs: dict) -> dict[tuple, Fit1d | Fit2d]:
    """Collect the usable fits of one machine, keyed by (setup, algorithm, grid) group key.

    The per-fit `(fit_params, pcov)` pair comes from the `fits/cov/` groups
    of the results file. Rows without a complete fit (missing or NaN
    parameters) are omitted.

    Args:
        fits: the fits table, restricted to one machine.
        covs: this machine's `fits/cov/` entries, keyed by (setup,
        algorithm, grid label).

    Returns:
        dict[tuple, Fit1d | Fit2d]: the usable fits, keyed by group key.

    """
    fits_by_key = {}
    for _, row in fits.iterrows():
        key = _group_key(tuple(row[k] for k in GROUP_KEYS))
        cov = covs.get((row["setup"], row["algorithm"], grid_label(row["x"], row["y"], row["z"])))
        if row["model"] == "2d" and all(
            pd.notna(row[k])
            for k in (
                "W",
                "N_malloc",
                "N_free",
                "A_malloc",
                "A_free",
                "m0_ns",
                "f0_ns",
            )
        ):
            fits_by_key[key] = Fit2d(
                W=float(row["W"]),
                n_malloc=float(row["N_malloc"]),
                n_free=float(row["N_free"]),
                a_malloc=float(row["A_malloc"]),
                a_free=float(row["A_free"]),
                m0=float(row["m0_ns"]) * 1e-9,
                f0=float(row["f0_ns"]) * 1e-9,
                cov=cov,
            )
        elif row["model"] == "1d-malloc" and all(pd.notna(row[k]) for k in ("W", "N_malloc", "A_malloc", "m0_ns")):
            fits_by_key[key] = Fit1d(
                direction="malloc",
                W=float(row["W"]),
                N=float(row["N_malloc"]),
                A=float(row["A_malloc"]),
                s0=float(row["m0_ns"]) * 1e-9,
                cov=cov,
            )
        elif row["model"] == "1d-free" and all(pd.notna(row[k]) for k in ("W", "N_free", "A_free", "f0_ns")):
            fits_by_key[key] = Fit1d(
                direction="free",
                W=float(row["W"]),
                N=float(row["N_free"]),
                A=float(row["A_free"]),
                s0=float(row["f0_ns"]) * 1e-9,
                cov=cov,
            )
    return fits_by_key


def draw_fit_2d(
    curve: Curve,
    sleeve: Callable[[_ModelFn, tuple], None],
) -> None:
    """Draw the 2-D fit of one curve and its two A/f gap markers.

    The solid line is the full model, the dashed line the linear
    extrapolation (both Amdahl terms at 0) and the dotted line the fit
    with only the plotted operation's Amdahl term at 0. The gap
    markers split at the dotted line: the upper part is the native
    cost of the plotted operation, the lower part the one of the held
    operation.

    Args:
        curve: the per-curve drawing context.
        sleeve: draws the bootstrap sleeve of a model line.

    """
    x_s, held_s = curve.x_s, curve.held_s
    params = curve.params
    held_arr = np.full_like(x_s, held_s)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, amdahl.EPS_S, amdahl.EPS_S)
    if curve.short == "malloc":
        model = amdahl.model_2d(
            x_s,
            held_arr,
            (params.W, params.n_malloc, params.n_free, params.a_malloc, params.a_free, params.m0, params.f0),
        )
        linear = params.W + params.n_malloc * x_s + params.n_free * held_s
        dotted = linear + params.a_free * params.f0 / (held_s + params.f0)

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return amdahl.model_2d(x_s, held_arr, p)

        def fn_dash(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s

        def fn_dot(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s + p[4] * p[6] / (held_s + p[6])
    else:
        model = amdahl.model_2d(
            held_arr,
            x_s,
            (params.W, params.n_malloc, params.n_free, params.a_malloc, params.a_free, params.m0, params.f0),
        )
        linear = params.W + params.n_free * x_s + params.n_malloc * held_s
        dotted = linear + params.a_malloc * params.m0 / (held_s + params.m0)

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return amdahl.model_2d(held_arr, x_s, p)

        def fn_dash(p: np.ndarray) -> np.ndarray:
            return p[0] + p[2] * x_s + p[1] * held_s

        def fn_dot(p: np.ndarray) -> np.ndarray:
            return p[0] + p[2] * x_s + p[1] * held_s + p[3] * p[5] / (held_s + p[5])

    if float(model[0]) - float(dotted[0]) > VISIBLE_GAP_S:
        y_lo, y_hi = float(dotted[0]), float(model[0])
        curve.ax.plot((curve.x0, curve.x0), (y_lo, y_hi), color=curve.color, linewidth=1, alpha=0.8)
    if float(dotted[0]) - float(linear[0]) > VISIBLE_GAP_S:
        y_lo, y_hi = float(linear[0]), float(dotted[0])
        curve.ax.plot((curve.x0, curve.x0), (y_lo, y_hi), color=curve.color, linewidth=1, alpha=0.8)
    # The model takes second-based delays; the axis is in ns.
    sleeve(fn_dot, lower)
    curve.ax.plot(curve.x_ns, dotted, color=curve.color, linestyle=":", alpha=0.8)
    sleeve(fn_curve, lower)
    curve.ax.plot(curve.x_ns, model, color=curve.color, linestyle="-", alpha=0.8)
    sleeve(fn_dash, lower)
    curve.ax.plot(curve.x_ns, linear, color=curve.color, linestyle="--", alpha=0.8)


def draw_fit_1d(
    curve: Curve,
    sleeve: Callable[[_ModelFn, tuple], None],
) -> None:
    """Draw the 1-D fit of one curve and its A/f gap marker.

    The solid line is the full model, the dashed line the linear
    extrapolation to A = 0; the marker spans the gap between them at
    the smallest delay.

    Args:
        curve: the per-curve drawing context.
        sleeve: draws the bootstrap sleeve of a model line.

    """
    x_s, x_ns = curve.x_s, curve.x_ns
    x0, params, color = curve.x0, curve.params, curve.color
    lower = (0.0, 0.0, 0.0, amdahl.EPS_S)
    model = amdahl.model_1d(x_s, params.W, params.N, params.A, params.s0)
    linear = params.W + params.N * x_s

    def fn_curve(p: np.ndarray) -> np.ndarray:
        return amdahl.model_1d(x_s, p[0], p[1], p[2], p[3])

    def fn_dash(p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x_s

    if float(model[0]) > float(linear[0]):
        y_lo, y_hi = float(linear[0]), float(model[0])
        curve.ax.plot((x0, x0), (y_lo, y_hi), color=color, linewidth=1, alpha=0.8)
    # The model takes second-based delays; the axis is in ns.
    sleeve(fn_curve, lower)
    curve.ax.plot(x_ns, model, color=color, linestyle="-", alpha=0.8)
    sleeve(fn_dash, lower)
    curve.ax.plot(x_ns, linear, color=color, linestyle="--", alpha=0.8)


def draw_fit_curves(curve: Curve) -> bool:
    """Draw the fitted model lines of one curve and its A/f gap markers.

    Dispatches on the fit type.

    Args:
        curve: the per-curve drawing context.

    Returns:
        bool: True when a 2-D fit was drawn.

    """

    def sleeve(fn: _ModelFn, lower: tuple) -> None:
        # Bootstrap sleeve of a fit line: draw the fitted
        # parameters from their covariance (the model is
        # non-linear in them) and fill the 25/75-percentile
        # envelope of the model values, the IQR convention of
        # the data's error bars. No covariance -> no sleeve.
        if curve.params.cov is None:
            return
        band = amdahl.bootstrap_band(curve.x_s, fn, curve.params.cov[0], curve.params.cov[1], lower=lower)
        if band is not None:
            curve.ax.fill_between(curve.x_ns, band[0], band[1], color=curve.color, alpha=0.3)

    if isinstance(curve.params, Fit2d):
        draw_fit_2d(curve, sleeve)
        return True
    draw_fit_1d(curve, sleeve)
    return False


def draw_model_lines(cluster: Cluster, x: np.ndarray, color: str, params: Fit1d | Fit2d, held_s: float) -> bool:
    """Draw the fitted model lines of one curve over its x range.

    Args:
        cluster: the per-cluster drawing context.
        x: the curve's x values (delays in nanoseconds).
        color: the curve's color.
        params: the fitted model parameters.
        held_s: the held (secondary) delay in seconds.

    Returns:
        bool: True when a 2-D fit was drawn.

    """
    # Draw the fitted model over the x-delay range this curve covers.
    x_data = x[x > 0]
    if not (len(x_data) > 1 and float(x_data[-1]) > float(x_data[0])):
        return False
    x_ns = np.geomspace(float(x_data[0]), float(x_data[-1]), 100)
    x_s = x_ns * 1e-9
    x0 = float(x_ns[0])
    curve = Curve(cluster.ax, x_ns, x_s, x0, held_s, params, cluster.short, color)
    return draw_fit_curves(curve)


def draw_series(cluster: Cluster, result: pd.DataFrame, name: tuple) -> bool:
    """Draw one curve: the errorbar points and, when a fit exists, the model lines.

    Args:
        cluster: the per-cluster drawing context.
        result: the statistics rows of the curve's group.
        name: the curve's group key (setup, algorithm, grid, held delay).

    Returns:
        bool: True when a 2-D fit was drawn.

    """
    frame = result.reset_index(drop=False)
    # One permutation sorts the points by delay; sorting each column
    # separately would misalign the IQR bounds against the delays.
    order = frame[cluster.x_delay].to_numpy().argsort()
    x, ye_min, y, ye_max = frame.iloc[order][[cluster.x_delay, "p25", "p50", "p75"]].to_numpy().T
    # Only the pure sweep is shown: runs where the other (held) delay is
    # 0. Runs with both delays > 0 belong to neither figure.
    if name[5] != 0:
        return False
    # A sweep needs at least two distinct x values above 0.
    x_pos = x[x > 0]
    if len(np.unique(x_pos)) < 2:
        return False
    # The x-axis is logarithmic, so the zero-delay point is not representable
    # there (it is invisible on the plot anyway); drop it so it cannot corrupt
    # the autoscaled x limits.
    pos = x > 0
    x, ye_min, y, ye_max = x[pos], ye_min[pos], y[pos], ye_max[pos]
    key = _group_key((name[0], name[2], name[3], name[4]))
    cluster.ax.errorbar(
        x,
        y,
        yerr=(y - ye_min, ye_max - y),
        linestyle="none",
        marker=cluster.series_markers[key],
        label=series_label(name, cluster.secondary_short),
        color=cluster.series_colors[key],
    )
    # The fit is keyed by (setup, algorithm, grid): unlike the colour and
    # marker, it depends on the algorithm of this row.
    params = cluster.fits_by_key.get(_group_key(name[:5]))
    # A 1-D fit only applies when it was made on the plotted delay; the
    # 2-D fit applies to either direction.
    if params is not None and isinstance(params, Fit1d) and params.direction != cluster.short:
        params = None
    if params is None:
        return False
    return draw_model_lines(cluster, x, cluster.series_colors[key], params, float(name[5]) * 1e-9)


def plot_delay_axis(
    ax: plt.Axes,
    data: MachineData,
    series_markers: dict,
    series_colors: dict,
    x_delay: str = MALLOC_DELAY,
) -> plt.Axes:
    """Fill one sweep axis: the data curves, the fits, and the axis decoration.

    The x-axis shows `x_delay`; the other delay is held per curve.

    Args:
        ax: the axis to fill.
        data: the machine's drawing data.
        series_markers: the per-series marker assignment.
        series_colors: the per-series colour assignment.
        x_delay: the swept delay column, MALLOC_DELAY or FREE_DELAY.

    Returns:
        plt.Axes: the filled axis.

    """
    secondary = FREE_DELAY if x_delay == MALLOC_DELAY else MALLOC_DELAY
    short = "malloc" if x_delay == MALLOC_DELAY else "free"
    secondary_short = "free" if secondary == FREE_DELAY else "malloc"
    fits_by_key = collect_fits(data.fits, data.covs)
    cluster = Cluster(ax, x_delay, short, secondary_short, series_markers, series_colors, fits_by_key)
    # One curve per (setup, algorithm, grid, held delay): the x-axis is
    # `x_delay`.
    has_2d = False
    for name, result in data.stats.groupby([*GROUP_KEYS, secondary], dropna=False):
        has_2d = draw_series(cluster, result, name) or has_2d
    ax.set_title(f"{short} time scan")
    lines = ["solid: full fit"]
    if has_2d:
        lines += [f"dotted: A_{short} = 0", "dashed: A_malloc = A_free = 0"]
    else:
        lines.append("dashed: extrapolation to A = 0")
    ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_xlabel(f"{short} sleep_time (ns)")
    ax.set_ylabel("runtime (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def single_algorithm_figure(title: str, data: MachineData, series_markers: dict, series_colors: dict) -> plt.Figure:
    """Build the single-row figure: the two sweeps side by side, both axes shared.

    Args:
        title: the figure title (the hardware the runs were made on).
        data: the machine's drawing data.
        series_markers: the per-series marker assignment.
        series_colors: the per-series colour assignment.

    Returns:
        plt.Figure: the figure.

    """
    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5.5), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(title)
    plot_delay_axis(axes[0], data, series_markers, series_colors, x_delay=MALLOC_DELAY)
    plot_delay_axis(axes[1], data, series_markers, series_colors, x_delay=FREE_DELAY)
    # The y-axis is shared: subplots already hides the right axis'
    # tick labels, drop its label too. The shared x-axis keeps its tick
    # labels under both side-by-side columns, as in any subplots figure.
    axes[1].set_ylabel("")
    # Legend on the left-most axis that actually has a curve (one
    # cluster's delay sweep may not have arrived yet).
    legend_first(axes)
    return fig


def multi_algorithm_figure(
    title: str,
    data: MachineData,
    series_markers: dict,
    series_colors: dict,
    algorithms: list[str],
) -> plt.Figure:
    """Build the multi-row figure: one row per algorithm, the left column names it.

    All data axes share the x- and y-axes; the shared x tick labels and
    column titles are shown on the last row that has data.

    Args:
        title: the figure title (the hardware the runs were made on).
        data: the machine's drawing data.
        series_markers: the per-series marker assignment.
        series_colors: the per-series colour assignment.
        algorithms: the algorithms present in the data, in row order.

    Returns:
        plt.Figure: the figure.

    """
    stats = data.stats
    fits = data.fits
    sub_data = {
        algorithm: MachineData(
            machine=data.machine,
            stats=stats[stats["algorithm"] == algorithm],
            fits=fits[fits["algorithm"] == algorithm],
            covs=data.covs,
        )
        for algorithm in algorithms
    }
    fig = plt.figure(figsize=(6.5 * 2 + 1.2, 5.5 * len(algorithms)), layout="constrained")
    fig.suptitle(title)
    gridspec = fig.add_gridspec(len(algorithms), 3, width_ratios=[0.16, 1, 1])
    row_axes = []
    for i, algorithm in enumerate(algorithms):
        if i == 0:
            malloc_ax = fig.add_subplot(gridspec[i, 1])
            free_ax = fig.add_subplot(gridspec[i, 2], sharey=malloc_ax, sharex=malloc_ax)
        else:
            # All axes share both axes (the sweeps cover the same delay
            # range); the shared x tick labels are shown on the last row
            # that has data (see below).
            malloc_ax = fig.add_subplot(gridspec[i, 1], sharey=row_axes[0][0], sharex=row_axes[0][0])
            free_ax = fig.add_subplot(gridspec[i, 2], sharey=malloc_ax, sharex=malloc_ax)
        # The left column names the row's algorithm.
        label_ax = fig.add_subplot(gridspec[i, 0])
        label_ax.axis("off")
        label_ax.text(0.9, 0.5, algorithm, rotation=90, ha="right", va="center", fontsize=12)
        plot_delay_axis(malloc_ax, sub_data[algorithm], series_markers, series_colors, x_delay=MALLOC_DELAY)
        plot_delay_axis(free_ax, sub_data[algorithm], series_markers, series_colors, x_delay=FREE_DELAY)
        free_ax.set_ylabel("")
        row_axes.append((malloc_ax, free_ax))
    # The x-axis is shared, but add_subplot(sharex=...) does not hide the
    # redundant tick labels (subplots does): keep them, together with the
    # repeated column titles, on the last row that actually has data.
    shown = max(
        (i for i, (malloc_ax, free_ax) in enumerate(row_axes) if malloc_ax.containers or free_ax.containers),
        default=None,
    )
    for i, (malloc_ax, free_ax) in enumerate(row_axes):
        if i != shown:
            malloc_ax.tick_params(axis="x", labelbottom=False)
            free_ax.tick_params(axis="x", labelbottom=False)
            malloc_ax.set_xlabel("")
            free_ax.set_xlabel("")
    # Legend on the left-most axis of the first row that has a curve.
    legend_first(ax for malloc_ax, free_ax in row_axes for ax in (malloc_ax, free_ax))
    return fig


def machine_figure(title: str, data: MachineData, algorithms: list[str]) -> plt.Figure:
    """Build one machine's figure: one row per algorithm, the two sweeps side by side.

    Args:
        title: the figure title (the hardware the runs were made on).
        data: the machine's drawing data.
        algorithms: the figure's algorithm row order (the file's
        `algorithm_order`).

    Returns:
        plt.Figure: the figure.

    """
    stats = data.stats
    # Assign each scenario (setup, grid) its marker and its colour once, in
    # plot order, so the same scenario is drawn with the same marker and
    # colour on every axis and in every algorithm row, even where only some
    # axes carry it.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_markers = {}
    series_colors = {}
    for name, _ in stats.groupby(["setup", "x", "y", "z"], dropna=False):
        key = _group_key(name)
        if key not in series_markers:
            series_markers[key] = MARKERS[len(series_markers) % len(MARKERS)]
            series_colors[key] = cycle[len(series_colors) % len(cycle)]
    if len(series_markers) > len(MARKERS) or len(series_colors) > len(cycle):
        warnings.warn(
            f"more than {min(len(MARKERS), len(cycle))} (setup, grid) scenarios; markers and colours repeat",
            stacklevel=2,
        )
    present = set(stats["algorithm"])
    present_algorithms = [algorithm for algorithm in algorithms if algorithm in present]
    if len(present_algorithms) <= 1:
        # Single-row layout: one algorithm only, no row header.
        fig = single_algorithm_figure(title, data, series_markers, series_colors)
    else:
        fig = multi_algorithm_figure(title, data, series_markers, series_colors, present_algorithms)
    # Re-limit the shared axes to snugly fit the drawn content (data
    # points, fit lines and sleeves; see `snug_xlims` and `snug_ylims`)
    # instead of the padded autoscaled range.
    snug_ylims(fig)
    snug_xlims(fig)
    return fig


def _machines_with_delay_runs(runs: pd.DataFrame) -> list[str]:
    """Return the machines with at least one delay-injected run, in first-appearance order.

    Args:
        runs: the runs table of the results file.

    Returns:
        list[str]: the machine labels eligible for a sweep figure.

    """
    eligible = runs[runs[MALLOC_DELAY].notna() | runs[FREE_DELAY].notna()]
    return list(eligible["machine"].drop_duplicates())


def main(*, machine: str | None = None, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the per-machine sweep figures from one results file.

    Args:
        machine: the machine to draw, or None for every machine with delay runs.
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
        runs = read_table(file, "runs")
        group_stats = read_table(file, "group_stats")
        fits = read_table(file, "fits")
        covs = read_fit_covs(file)
        algorithms = algorithm_order(file)
        sweep = sweep_machine_labels(file)
        titles = machine_titles(file)
    machines = [m for m in sweep if m in _machines_with_delay_runs(runs)]
    if machine is not None:
        if machine not in machines:
            print(
                f"machine {machine!r} has no delay runs (available: {', '.join(machines) or 'none'})", file=sys.stderr
            )
            return 1
        machines = [machine]
    FIGURES.mkdir(exist_ok=True)
    for m in machines:
        data = MachineData(
            machine=m,
            stats=group_stats[group_stats["machine"] == m].drop(columns=["machine"]),
            fits=fits[fits["machine"] == m].drop(columns=["machine"]),
            covs={key[1:]: value for key, value in covs.items() if key[0] == m},
        )
        fig = machine_figure(titles.get(m, m), data, algorithms)
        fig.savefig(FIGURES / f"sweeps-{m}.pdf")
        print(f"wrote {FIGURES / f'sweeps-{m}.pdf'}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-machine delay-sweep figures from output/results.h5 (figures/sweeps-<machine>.pdf)."
    )
    parser.add_argument(
        "--machine",
        help="draw only this machine's figure (default: every machine with delay runs)",
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
