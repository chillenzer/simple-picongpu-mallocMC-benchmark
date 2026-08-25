"""Analyze PIConGPU/mallocMC allocation-latency benchmark logs.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads the raw `run_all.sh` logs directly (no pre-filtering; both the old
per-variant layout and the current one) and fits each (example, grid) sweep
to the constrained Amdahl allocation model. It then plots the runtime
against the imposed delay (median with IQR error bars) in one figure per
cluster (see `CLUSTERS`), saved to `figures/<cluster>.pdf`, the figure
titled by the hardware the runs were made on, with the malloc sleep_time
sweep (free sleep_time = 0) on the
left and the free sleep_time sweep (malloc sleep_time = 0) on the right,
each axis titled after the swept delay ("malloc time scan", "free time
scan"), the two axes sharing the y-axis. Each swept
curve overlays the fitted model (solid), the extrapolation to
A_malloc = A_free = 0 (dashed) and, for two-operation fits, the
intermediate extrapolation that keeps only the held operation's native
cost (dotted); the gaps at the smallest delay are marked by a short
    line. Each fit line carries a transparent sleeve, the
    25/75-percentile envelope of the model
    evaluated at 512 parameter draws from the fitted parameters and their
    covariance (a bootstrap, since the parameters enter the model
    non-linearly). It also produces one cross-cluster figure,
    `figures/runtime-stack.pdf`: for each scenario (setup, algorithm, grid),
     a group of neighbouring bars, one per cluster's hardware, each stacking
     the fitted W, A_malloc and A_free up to the total runtime, overlaid with
     the measured zero-delay runtime and its IQR error bar.

Model
-----
Each of the N allocation calls on the (serial, host-side) critical path takes
its native cost c_a plus the imposed delay s, so

    T(s) = W + A + N*s,   A = N*c_a (native allocation time)

with W the runtime without any allocation cost. The native part is not needed
for large sleeptimes (negligible against the imposed delay) but acts as a
correction at small sleeptimes, so the sweep is fitted with

    T(s) = W + N*s + A*s0/(s+s0)             (Amdahl model)

Alternative formulations:

    T(s) = W + A (s/c_a + 1 / (1 + d * s/c_a)), d = c_a / s0

    T(s) = W + A (1 + s/c_a + s^2/(s0c_a) )/(1+s/s0)

which reduces to the Amdahl line W + N*s for s >> s0 and to W + A for
s -> 0. Parameters are (W, N, A, s0): baseline runtime, allocation calls per
run, native allocation time, and the sleeptime scale over which the native
cost fades. The Amdahl fraction of the runtime spent in allocations is

    f = A / (W + A)      (native allocation time / total time at zero delay)

The fit is `scipy.optimize.curve_fit` with bounds W>=0, N>=0, A>=0 and
s0 in [0.05*s_min, 0.5*s_range], so the reported f is always in [0, 1). The
s0 lower limit keeps the correction from being confined to s=0 only; the s0
upper limit keeps it (and A) identifiable instead of a constant offset
degenerate with W. A robust linear solution over a log-s0 grid provides the
initial guess; if `curve_fit` fails to converge that linear solution is
reported instead. The parameter covariance from `curve_fit` is propagated to
f (and to W, N, A, s0) as standard errors.

When a sweep imposes a delay on both operations (the (malloc_delay,
free_delay) combination runs of run_all.sh), each operation gets its own
Amdahl term, i.e. the native cost of an operation only fades while its own
imposed delay grows. Such a sweep is then fitted with the two-operation
(separable, no cross-term) model

    T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0)

and the Amdahl fractions of the runtime spent in allocations and frees are
reported separately: f_malloc = A_m/T0, f_free = A_f/T0, T0 = W + A_m + A_f.
Groups whose runs vary only one of the two delays fall back to the 1-D model
above, fitted on that delay.

`sleeptimes` are in nanoseconds, `runtimes` in seconds. If the native
per-allocation cost c_a (ns) is known, pass it: A = N*c_a is then used and
only (W, N, s0) are fitted.
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY, parse_logs
from scipy.optimize import OptimizeWarning, curve_fit

# The creation policies the benchmark can run, in the row order of the
# figures; a figure only shows the algorithms present in its data.
ALGORITHM_ORDER = ("FlatterScatter", "ScatterAlloc", "Gallatin")
# cluster name -> (run-log directory, hardware name for the figure title)
CLUSTERS = {
    "hal": (Path("output") / "hal-sleeptimes", "NVIDIA A30"),
    "rosi": (Path("output") / "rosi-sleeptimes", "NVIDIA V100"),
}
# the one figure per cluster is written to `figures/<cluster>.pdf`
FIGURES = Path("figures")
# The statistics, plot and fit are computed only for runs with this
# configuration: "run-time" (delays injected via MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY) or "compile-time" (per-variant builds); None uses all
# of them. The parsed results are always complete.
CONFIGURATION = None

# one distinct marker per (setup, grid) series, shared by all axes
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
# the stacked segments of the runtime-budget figure, in stacking order
# (bottom to top), as (legend label, colour) pairs.
RUNTIME_SEGMENTS = (("W", "#8c8c8c"), ("A_malloc", "#1f77b4"), ("A_free", "#ff7f0e"))
_INF = float("inf")
# Near-zero floor for second-based runtimes: the Amdahl fraction is
# reported as 0 when the total runtime drops below it, and the fade
# scales (s0, m0, f0) are bounded by it from below.
EPS_S = 1e-12
# A gap marker is only drawn when the model difference exceeds this (s).
VISIBLE_GAP_S = 1e-9


def simple_statistics(full_results: pd.DataFrame) -> pd.DataFrame:
    """Group the parsed runs and describe the runtime of each group.

    Returns:
        pd.DataFrame: the runtime description of each (setup, algorithm, grid, delay) group.

    """
    # Group by the frame's column order (not a set): a set's iteration order
    # is hash-randomized per process, which would shuffle the printed index.
    group_cols = [c for c in full_results.columns if c not in {"runtime in s", "name"}]
    return full_results.groupby(group_cols, dropna=False).apply(
        lambda df: df["runtime in s"].describe(), include_groups=False
    )


def label(info: tuple, secondary: str = "free") -> str:
    """Format a (setup, algorithm, grid, held-delay) group key as a legend label.

    Returns:
        str: the formatted legend label.

    """
    # Plot group key (setup, algorithm, x, y, z, <secondary delay>); the
    # suffix names the held (secondary) delay when it is non-zero. The
    # algorithm is not part of the label: in multi-algorithm figures each
    # row is one algorithm and is named by its row header.
    grid_string = "x".join(map(str, map(int, np.asarray(info[2:5])[~np.isnan(info[2:5])])))
    suffix = "" if info[5] == 0 else f", {secondary}: {int(info[5])} ns"
    return f"{info[0]} {grid_string}{suffix}"


def _group_key(name: tuple) -> tuple:
    # NaN group values (missing z in 2D runs) do not compare equal, so
    # canonicalize them; the key is only used for dictionary lookups.
    return tuple(None if isinstance(k, float) and np.isnan(k) else k for k in name)


def _legend_first(axes: Iterable[plt.Axes]) -> None:
    """Show the legend, pinned to the top, on the left-most axis with a curve.

    The location is fixed (rather than matplotlib's auto "best") so the
    legend stays at the top even as the data or the A/f labels move.
    """
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
            break


def _plot_single_algorithm_fig(
    title: str, simple_results: pd.DataFrame, fits: pd.DataFrame | None, series_markers: dict
) -> plt.Figure:
    """Build the single-row figure: the two sweeps side by side, y-axis shared.

    Returns:
        plt.Figure: the figure.

    """
    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5.5), sharey=True, layout="constrained")
    fig.suptitle(title)
    _plot_cluster(axes[0], simple_results, fits, series_markers, x_delay=MALLOC_DELAY)
    _plot_cluster(axes[1], simple_results, fits, series_markers, x_delay=FREE_DELAY)
    # The y-axis is shared: subplots already hides the right axis'
    # tick labels, drop its label too.
    axes[1].set_ylabel("")
    # Legend on the left-most axis that actually has a curve (one
    # cluster's delay sweep may not have arrived yet).
    _legend_first(axes)
    return fig


def _plot_multi_algorithm_fig(
    title: str,
    simple_results: pd.DataFrame,
    fits: pd.DataFrame | None,
    series_markers: dict,
    algorithms: list,
) -> plt.Figure:
    """Build the multi-row figure: one row per algorithm, the left column names it.

    Returns:
        plt.Figure: the figure.

    """
    fig = plt.figure(figsize=(6.5 * 2 + 1.2, 5.5 * len(algorithms)), layout="constrained")
    fig.suptitle(title)
    gridspec = fig.add_gridspec(len(algorithms), 3, width_ratios=[0.16, 1, 1])
    row_axes = []
    for i, algorithm in enumerate(algorithms):
        # Keep the full MultiIndex (drop nothing): `_plot_cluster`
        # re-groups on the level names.
        sub = simple_results[simple_results.index.get_level_values("algorithm") == algorithm]
        fits_sub = None if fits is None else fits[fits["algorithm"] == algorithm]
        malloc_ax = fig.add_subplot(gridspec[i, 1], sharey=None if i == 0 else row_axes[0][0])
        free_ax = fig.add_subplot(gridspec[i, 2], sharey=malloc_ax)
        # The left column names the row's algorithm.
        label_ax = fig.add_subplot(gridspec[i, 0])
        label_ax.axis("off")
        label_ax.text(0.9, 0.5, algorithm, rotation=90, ha="right", va="center", fontsize=12)
        _plot_cluster(malloc_ax, sub, fits_sub, series_markers, x_delay=MALLOC_DELAY)
        _plot_cluster(free_ax, sub, fits_sub, series_markers, x_delay=FREE_DELAY)
        free_ax.set_ylabel("")
        row_axes.append((malloc_ax, free_ax))
    # Legend on the left-most axis of the first row that has a curve.
    _legend_first(ax for malloc_ax, free_ax in row_axes for ax in (malloc_ax, free_ax))
    return fig


def simple_plot(cluster_results: list[tuple[str, pd.DataFrame, pd.DataFrame | None]]) -> list[plt.Figure]:
    """One figure per cluster, one row per algorithm, the two sweeps side by side.

    `cluster_results` is a list of `(title, simple_results, fits)` entries,
    one per cluster; `title` is the hardware the runs were made on. Each
    figure is titled by the hardware; each row is one creation policy
    (only the algorithms present in the data, in `ALGORITHM_ORDER`) with
    two axes, titled "malloc time scan" and "free time scan", sharing the
    y-axis with all other axes of the figure (only the left axis shows its
    tick labels and label): the malloc delay sweep (free delay 0) on the
    left and the free delay sweep (malloc delay 0) on the right. A figure
    with a single algorithm has exactly the one-row layout; with several,
    the left column names the row's algorithm. Each (setup, grid) series
    gets a distinct marker, consistently on all axes and figures; the
    legend is shown on each figure's left-most axis that has a curve.

    Returns:
        list[plt.Figure]: one figure per cluster.

    """
    # Assign each (setup, algorithm, grid) series its marker once, in plot
    # order, so the same series is drawn with the same marker on every axis
    # of every figure.
    series_markers = {}
    for _, simple_results, _ in cluster_results:
        for name, _ in simple_results.groupby(list(GROUP_KEYS), dropna=False):
            key = _group_key(name)
            if key not in series_markers:
                series_markers[key] = MARKERS[len(series_markers) % len(MARKERS)]
    if len(series_markers) > len(MARKERS):
        warnings.warn(f"more than {len(MARKERS)} (setup, algorithm, grid) series; the markers repeat", stacklevel=2)
    figs = []
    for title, simple_results, fits in cluster_results:
        # `simple_results` is the groupby output of `simple_statistics`: the
        # group keys (including the algorithm) are index levels, not columns.
        present = set(simple_results.index.get_level_values("algorithm"))
        algorithms = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in present]
        if len(algorithms) <= 1:
            # Backwards-compatible layout: the single-row figure, exactly as
            # before multi-algorithm sweeps existed.
            fig = _plot_single_algorithm_fig(title, simple_results, fits, series_markers)
        else:
            fig = _plot_multi_algorithm_fig(title, simple_results, fits, series_markers, algorithms)
        figs.append(fig)
    return figs


def _scenario_key(setup: str, algorithm: str, x: float, y: float, z: float) -> tuple:
    """Canonical key for a (setup, algorithm, grid) scenario.

    Grid dimensions are normalized so the same scenario matches across
    clusters even when a dimension is an int in one and a whole-valued
    float in the other; a missing (2-D) ``z`` maps to ``None``.

    Returns:
        tuple: (setup, algorithm, x, y, z) with normalized grid dimensions.

    """

    def norm(v: float | None) -> float | None:
        if v is None:
            return None
        v = float(v)
        if not np.isfinite(v):
            return None
        return int(v) if v.is_integer() else v

    return (setup, algorithm, norm(x), norm(y), norm(z))


def _scenario_label(setup: str, x: float, y: float, z: float) -> str:
    """Human-readable scenario name, e.g. ``FoilLCT 256x1280``.

    A missing (2-D) ``z`` is dropped from the grid.

    Returns:
        str: the scenario label.

    """

    def dim(v: float | None) -> str | None:
        if v is None:
            return None
        v = float(v)
        if not np.isfinite(v):
            return None
        return str(int(v)) if v.is_integer() else str(v)

    grid = "x".join(d for d in (dim(x), dim(y), dim(z)) if d is not None)
    return f"{setup} {grid}"


def _runtime_budget(
    per_cluster: list[tuple[str, pd.DataFrame, pd.DataFrame | None]],
) -> list[tuple[str, dict[tuple, tuple[float, float, float]]]]:
    """Per-cluster runtime budget: the scenario key to (W, A_malloc, A_free).

    Built from the valid fits of each cluster; a missing allocation cost
    (a 1-D fit on the other operation) is treated as 0.

    Returns:
        list[tuple[str, dict]]: per cluster, (hardware short name,
        {scenario key: (W, A_malloc, A_free)}).

    """
    budget = []
    for title, _simple, fits in per_cluster:
        by_key: dict[tuple, tuple[float, float, float]] = {}
        if fits is not None:
            for _, row in fits.iterrows():
                if row["model"] is None or pd.isna(row["W"]):
                    continue
                key = _scenario_key(row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
                a_malloc = float(row["A_malloc"]) if pd.notna(row["A_malloc"]) else 0.0
                a_free = float(row["A_free"]) if pd.notna(row["A_free"]) else 0.0
                by_key[key] = (float(row["W"]), a_malloc, a_free)
        budget.append((title.split()[-1], by_key))
    return budget


def _baseline_points(
    per_cluster: list[tuple[str, pd.DataFrame, pd.DataFrame | None]],
) -> list[tuple[str, dict[tuple, tuple[float, float, float]]]]:
    """Per-cluster measured runtime at zero delay, per scenario.

    The (malloc, free) = (0, 0) group of `simple_results` is the baseline
    run without any injected delay; its IQR is the error bar.

    Returns:
        list[tuple[str, dict]]: per cluster, (hardware short name,
        {scenario key: (25%, 50%, 75%)}).

    """
    points = []
    for title, simple, _fits in per_cluster:
        by_key: dict[tuple, tuple[float, float, float]] = {}
        if simple is not None:
            base = simple.reset_index()
            base = base[(base[MALLOC_DELAY] == 0) & (base[FREE_DELAY] == 0)]
            for _, row in base.iterrows():
                key = _scenario_key(row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
                by_key[key] = (float(row["25%"]), float(row["50%"]), float(row["75%"]))
        points.append((title.split()[-1], by_key))
    return points


def _ordered_scenarios(budget: list[tuple[str, dict]]) -> list[tuple]:
    """Order the unique scenario keys by setup, algorithm, then grid.

    Returns:
        list[tuple]: the ordered scenario keys.

    """
    keys = set()
    for _hw, by_key in budget:
        keys.update(by_key)
    return sorted(keys, key=lambda k: (k[0], k[1], k[2], k[3], k[4] if k[4] is not None else -1))


def _scenario_labels(keys: list[tuple]) -> dict[tuple, str]:
    """Build the display label per scenario key.

    A label is disambiguated with the algorithm when a setup+grid is shared
    by several algorithms.

    Returns:
        dict[tuple, str]: the display label per scenario key.

    """
    base = {key: _scenario_label(key[0], key[2], key[3], key[4]) for key in keys}
    counts = {}
    for lab in base.values():
        counts[lab] = counts.get(lab, 0) + 1
    return {key: (lab if counts[lab] == 1 else f"{lab} ({key[1]})") for key, lab in base.items()}


def _draw_runtime_bars(ax: plt.Axes, keys: list[tuple], budget: list[tuple[str, dict]], bar_w: float) -> float:
    """Draw the stacked runtime bars.

    Returns:
        float: the tallest bar's total runtime.

    """
    n_hw = len(budget)
    labels_set = False
    max_total = 0.0
    for i, key in enumerate(keys):
        for h, (_hw, by_key) in enumerate(budget):
            if key not in by_key:
                continue
            x = i + (h - (n_hw - 1) / 2) * bar_w
            bottom = 0.0
            for (label, color), value in zip(RUNTIME_SEGMENTS, by_key[key], strict=True):
                ax.bar(x, value, width=bar_w, bottom=bottom, color=color, label=None if labels_set else label)
                bottom += value
            labels_set = True
            max_total = max(max_total, bottom)
            ax.text(x, bottom, f"{bottom:.3g}", ha="center", va="bottom", fontsize=8)
    return max_total


def _draw_baseline_points(ax: plt.Axes, keys: list[tuple], baseline: list[tuple[str, dict]], bar_w: float) -> float:
    """Overlay the measured zero-delay point (median, IQR error bar) on each bar.

    The error bar is the true IQR of the (0, 0) runs, drawn without caps so it
    reads as a plain vertical mark.

    Returns:
        float: the highest error-bar top (0 when there are no points).

    """
    n_hw = len(baseline)
    labels_set = False
    max_hi = 0.0
    for i, key in enumerate(keys):
        for h, (_hw, by_key) in enumerate(baseline):
            if key not in by_key:
                continue
            lo, med, hi = by_key[key]
            x = i + (h - (n_hw - 1) / 2) * bar_w
            ax.errorbar(
                [x],
                [med],
                yerr=[[med - lo], [hi - med]],
                fmt="none",
                ecolor="k",
                elinewidth=1,
                capsize=0,
                zorder=5,
            )
            ax.plot(
                [x],
                [med],
                marker="o",
                markersize=5,
                mfc="k",
                mec="w",
                mew=0.5,
                zorder=6,
                label=None if labels_set else "measured (0 delay)",
            )
            labels_set = True
            max_hi = max(max_hi, hi)
    return max_hi


def _set_runtime_xaxis(
    ax: plt.Axes, keys: list[tuple], labels: dict[tuple, str], budget: list[tuple[str, dict]], bar_w: float
) -> None:
    """Set the scenario x-axis ticks and the hardware sub-label under each bar.

    The scenario is the major tick, centred on its group (two lines: setup,
    then grid). The hardware is a lighter sub-label a row below, under each
    bar. The caller reserves the bottom margin for the extra row.
    """
    n_hw = len(budget)
    major = []
    for key in keys:
        lab = labels[key]
        first, sep, rest = lab.partition(" ")
        major.append(f"{first}\n{rest}" if sep else lab)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(major, fontsize=9)
    for i, _key in enumerate(keys):
        for h, (hw, _by_key) in enumerate(budget):
            x = i + (h - (n_hw - 1) / 2) * bar_w
            ax.text(
                x,
                -0.16,
                hw,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="0.45",
            )


def stacked_runtime_fig(per_cluster: list[tuple[str, pd.DataFrame, pd.DataFrame | None]]) -> plt.Figure:
    """One cross-cluster figure: the runtime budget stacked per scenario.

    Each scenario (setup, algorithm, grid) is a group of neighbouring bars,
    one per cluster; a bar stacks the runtime components W, A_malloc and
    A_free up to the total runtime. The segment colour names the component,
    the position within the group the hardware.

    Args:
        per_cluster: (title, simple_results, fits) per cluster, as in `main`.

    Returns:
        plt.Figure: the figure.

    """
    budget = _runtime_budget(per_cluster)
    baseline = _baseline_points(per_cluster)
    keys = _ordered_scenarios(budget)
    # Manual layout: the bottom margin carries the two-line scenario labels
    # plus the hardware sub-label row that `_set_runtime_xaxis` adds.
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.20)
    if not keys:
        return fig
    bar_w = 0.8 / len(budget)
    max_total = _draw_runtime_bars(ax, keys, budget, bar_w)
    # the measured points can sit a touch above the fitted total; keep them
    # inside the axis.
    max_total = max(max_total, _draw_baseline_points(ax, keys, baseline, bar_w))
    _set_runtime_xaxis(ax, keys, _scenario_labels(keys), budget, bar_w)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, max_total * 1.12)
    ax.set_ylabel("runtime (s)")
    ax.legend(loc="upper left")
    fig.suptitle("Runtime budget per scenario: W + A_malloc + A_free")
    return fig


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


class Model1d(NamedTuple):
    """Parameters of the 1-operation Amdahl model T(s) = W + N*s + A*s0/(s+s0).

    The delay s is in seconds.
    """

    W: float
    N: float
    A: float
    s0: float


class Model2d(NamedTuple):
    """Parameters of the 2-operation Amdahl model, the delays in seconds."""

    W: float
    N_m: float
    N_f: float
    A_m: float
    A_f: float
    m0: float
    f0: float


class FitSetup(NamedTuple):
    """Inputs precomputed for the constrained 1-D fit of one sweep.

    `bounds` is the (lo_b, hi_b) search range of s0, `guess` the robust
    grid solution (W, N, A, s0), and `eps` the floor for the initial guess.
    """

    bounds: tuple[float, float]
    guess: tuple[float, float, float, float]
    eps: float


class ConstrainedFit(NamedTuple):
    """A successful curve_fit attempt of one sweep.

    `notes` carries the corner-solution diagnostics (A floored at 0, fade
    scale at the search cap); the caller appends them to its warnings.
    """

    model: Model1d | Model2d
    pcov: np.ndarray | None
    fit_params: list[float] | None
    notes: list[str]


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
    fits_by_key: dict


_ModelFn = Callable[[np.ndarray], np.ndarray]


def _collect_fits(fits: pd.DataFrame | None) -> dict[tuple, Fit1d | Fit2d]:
    """Collect the usable fits, keyed by (setup, algorithm, grid) group key.

    Rows without a complete fit (missing or NaN parameters) are
    omitted; `fits` of None yields an empty dict.

    Returns:
        dict[tuple, Fit1d | Fit2d]: the usable fits, keyed by group key.

    """
    fits_by_key = {}
    if fits is not None:
        for _, row in fits.iterrows():
            key = _group_key(tuple(row[k] for k in GROUP_KEYS))
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
                    cov=row["cov"],
                )
            elif row["model"] == "1d-malloc" and all(pd.notna(row[k]) for k in ("W", "N_malloc", "A_malloc", "m0_ns")):
                fits_by_key[key] = Fit1d(
                    direction="malloc",
                    W=float(row["W"]),
                    N=float(row["N_malloc"]),
                    A=float(row["A_malloc"]),
                    s0=float(row["m0_ns"]) * 1e-9,
                    cov=row["cov"],
                )
            elif row["model"] == "1d-free" and all(pd.notna(row[k]) for k in ("W", "N_free", "A_free", "f0_ns")):
                fits_by_key[key] = Fit1d(
                    direction="free",
                    W=float(row["W"]),
                    N=float(row["N_free"]),
                    A=float(row["A_free"]),
                    s0=float(row["f0_ns"]) * 1e-9,
                    cov=row["cov"],
                )
    return fits_by_key


def _draw_fit_2d(
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
    """
    x_s, held_s = curve.x_s, curve.held_s
    params = curve.params
    held_arr = np.full_like(x_s, held_s)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S)
    if curve.short == "malloc":
        model = _model_2d(
            x_s,
            held_arr,
            (params.W, params.n_malloc, params.n_free, params.a_malloc, params.a_free, params.m0, params.f0),
        )
        linear = params.W + params.n_malloc * x_s + params.n_free * held_s
        dotted = linear + params.a_free * params.f0 / (held_s + params.f0)

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return _model_2d(x_s, held_arr, p)

        def fn_dash(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s

        def fn_dot(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s + p[4] * p[6] / (held_s + p[6])
    else:
        model = _model_2d(
            held_arr,
            x_s,
            (params.W, params.n_malloc, params.n_free, params.a_malloc, params.a_free, params.m0, params.f0),
        )
        linear = params.W + params.n_free * x_s + params.n_malloc * held_s
        dotted = linear + params.a_malloc * params.m0 / (held_s + params.m0)

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return _model_2d(held_arr, x_s, p)

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


def _draw_fit_1d(
    curve: Curve,
    sleeve: Callable[[_ModelFn, tuple], None],
) -> None:
    """Draw the 1-D fit of one curve and its A/f gap marker.

    The solid line is the full model, the dashed line the linear
    extrapolation to A = 0; the marker spans the gap between them at
    the smallest delay.
    """
    x_s, x_ns = curve.x_s, curve.x_ns
    x0, params, color = curve.x0, curve.params, curve.color
    lower = (0.0, 0.0, 0.0, EPS_S)
    model = _model(x_s, params.W, params.N, params.A, params.s0)
    linear = params.W + params.N * x_s

    def fn_curve(p: np.ndarray) -> np.ndarray:
        return _model(x_s, p[0], p[1], p[2], p[3])

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


def _draw_fit_curves(curve: Curve) -> bool:
    """Draw the fitted model lines of one curve and its A/f gap markers.

    Dispatches on the fit type.

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
        band = _bootstrap_band(curve.x_s, fn, curve.params.cov[0], curve.params.cov[1], lower=lower)
        if band is not None:
            curve.ax.fill_between(curve.x_ns, band[0], band[1], color=curve.color, alpha=0.3)

    if isinstance(curve.params, Fit2d):
        _draw_fit_2d(curve, sleeve)
        return True
    _draw_fit_1d(curve, sleeve)
    return False


def _draw_curve_fit(cluster: Cluster, x: np.ndarray, color: str, params: Fit1d | Fit2d, held_s: float) -> bool:
    """Draw the fitted model lines of one curve over its x range.

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
    return _draw_fit_curves(curve)


def _draw_series(cluster: Cluster, result: pd.DataFrame, name: tuple) -> bool:
    """Draw one curve: the errorbar points and, when a fit exists, the model lines.

    Returns:
        bool: True when a 2-D fit was drawn.

    """
    x, ye_min, y, ye_max = np.sort(result.reset_index(drop=False)[[cluster.x_delay, "25%", "50%", "75%"]].to_numpy().T)
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
    eb = cluster.ax.errorbar(
        x,
        y,
        yerr=(y - ye_min, ye_max - y),
        linestyle="none",
        marker=cluster.series_markers[_group_key(name[:5])],
        label=label(name, cluster.secondary_short),
    )
    color = eb.lines[0].get_color()
    params = cluster.fits_by_key.get(_group_key(name[:5]))
    # A 1-D fit only applies when it was made on the plotted delay; the
    # 2-D fit applies to either direction.
    if params is not None and isinstance(params, Fit1d) and params.direction != cluster.short:
        params = None
    if params is None:
        return False
    return _draw_curve_fit(cluster, x, color, params, float(name[5]) * 1e-9)


def _plot_cluster(
    ax: plt.Axes,
    simple_results: pd.DataFrame,
    fits: pd.DataFrame | None,
    series_markers: dict,
    x_delay: str = MALLOC_DELAY,
) -> plt.Axes:
    # The x-axis shows `x_delay`; the other delay is held per curve.
    secondary = FREE_DELAY if x_delay == MALLOC_DELAY else MALLOC_DELAY
    short = "malloc" if x_delay == MALLOC_DELAY else "free"
    secondary_short = "free" if secondary == FREE_DELAY else "malloc"
    fits_by_key = _collect_fits(fits)
    cluster = Cluster(ax, x_delay, short, secondary_short, series_markers, fits_by_key)
    # One curve per (setup, algorithm, grid, held delay): the x-axis is
    # `x_delay`.
    results = simple_results.groupby([*GROUP_KEYS, secondary], dropna=False)
    has_2d = False
    for name, result in results:
        has_2d = _draw_series(cluster, result, name) or has_2d
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


def _model(s: np.ndarray, W: float, N: float, A: float, s0: float) -> np.ndarray:
    return W + N * s + A * s0 / (s + s0)


def _model_2d(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    W, N_m, N_f, A_m, A_f, m0, f0 = p
    return W + N_m * m + N_f * f + A_m * m0 / (m + m0) + A_f * f0 / (f + f0)


def _model_c_a(s: np.ndarray, W: float, N: float, s0: float, c: float) -> np.ndarray:
    A = N * c
    return W + N * s + A * s0 / (s + s0)


_BOOTSTRAP_N = 512
_BOOTSTRAP_PERCENTILES = (25.0, 75.0)
_BOOTSTRAP_SEED = 0


def _bootstrap_band(
    x_s: np.ndarray,
    fn: _ModelFn,
    params: Sequence[float],
    pcov: np.ndarray,
    lower: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Percentile envelope of `fn(x_s, *p)` over p ~ N(params, pcov).

    The parameters enter the model non-linearly (the Amdahl terms), so the
    sleeve is a resampling bootstrap rather than an analytic error
    propagation: draw `_BOOTSTRAP_N` parameter vectors from the multivariate
    normal of the fitted parameters and their covariance, evaluate the
    line's model function `fn` on the x-grid `x_s` (in seconds) for each,
    and take the `_BOOTSTRAP_PERCENTILES` of the resampled values.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: the pointwise (lo, hi)
        percentiles of the resampled model values, or None when `pcov` is
        unavailable or non-finite (no sleeve).

    """
    cov = np.asarray(pcov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return None
    # curve_fit's covariance is symmetric in exact arithmetic (an inverse of
    # the symmetric J^T J), but only to rounding error in floating point.
    # Symmetrize so the eigendecomposition below is exact and well-defined.
    cov = 0.5 * (cov + cov.T)
    # The parameters are nearly degenerate, so this covariance is
    # near-singular (condition number ~1e13-1e18). A Cholesky factor of such
    # a matrix is numerically unreliable in floating point (error ~
    # cond * eps) and injects spurious variance -- it sampled W with ~200x
    # its true width, inflating the sleeve ~172x. An eigendecomposition is
    # stable for near-singular PSD matrices: the tiny negative eigenvalues
    # are just rounding, clip them to zero, and sample along the
    # eigenvectors.
    eig, evec = np.linalg.eigh(cov)
    eig = np.clip(eig, 0.0, None)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = (
        np.asarray(params, dtype=float) + (rng.standard_normal((_BOOTSTRAP_N, len(params))) * np.sqrt(eig)) @ evec.T
    )
    if lower is not None:
        # The multivariate tails can cross the fit's bounds; clip them so the
        # A*s0/(x+s0)-type terms cannot blow up on the wrong side.
        samples = np.clip(samples, np.asarray(lower, dtype=float), None)
    values = np.empty((_BOOTSTRAP_N, x_s.size))
    for i, p in enumerate(samples):
        values[i] = np.asarray(fn(p), dtype=float)
    lo, hi = np.percentile(values, _BOOTSTRAP_PERCENTILES, axis=0)
    return lo, hi


def _fit_lsq(h: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares for t = W + N*s + A*h.

    Returns:
        tuple[float, float, float, float]: W, N, A, and the residual sum of squares.

    """
    sol, *_ = np.linalg.lstsq(np.vstack([np.ones_like(s), s, h]).T, t, rcond=None)
    res = t - (sol[0] + sol[1] * s + sol[2] * h)
    return (*[float(v) for v in sol], float(np.sum(res**2)))


def _grid_guess(s: np.ndarray, t: np.ndarray, lo: float, hi: float) -> tuple[float, float, float, float]:
    """Robust unconstrained solution: linear in (W, N, A) for each s0 on a log grid.

    Returns:
        tuple[float, float, float, float]: W, N, A, and the best s0 from the log grid.

    """
    hi_g = max(float(hi), lo * 1.5)
    best = None
    for s0 in np.logspace(np.log10(lo), np.log10(hi_g), 60):
        W, N, A, ss_res = _fit_lsq(s0 / (s + s0), s, t)
        if best is None or ss_res < best[0]:
            best = (ss_res, s0, W, N, A)
    _, s0, W, N, A = best
    return W, N, A, float(s0)


def _uncertainties(p: np.ndarray, pcov: np.ndarray | None, f_of_p: Callable[[np.ndarray], float]) -> list[float]:
    """Compute the standard errors for the fitted parameters p and for f = f_of_p(p).

    Derived from the parameter covariance; NaNs if the covariance is unavailable.

    Returns:
        list[float]: the standard errors of the fitted parameters, then that of f.

    """
    if pcov is None or not np.all(np.isfinite(np.asarray(pcov))):
        return [float("nan")] * (len(p) + 1)
    pcov = np.asarray(pcov, dtype=float)
    errs = [float(e) for e in np.sqrt(np.clip(np.diag(pcov), 0.0, None))]
    g = np.zeros(len(p))
    for i in range(len(p)):
        pp = np.asarray(p, dtype=float).copy()
        pm = np.asarray(p, dtype=float).copy()
        step = max(abs(p[i]) * 1e-6, 1e-12)
        pp[i] += step
        pm[i] -= step
        g[i] = (f_of_p(pp) - f_of_p(pm)) / (2 * step)
    var_f = float(g @ pcov @ g)
    errs.append(float(np.sqrt(max(var_f, 0.0))))
    return errs


def _f_of_p_free(p: np.ndarray) -> float:
    """Compute the Amdahl fraction f = A/(W + A) of the 1-D parameters p.

    Returns:
        float: the Amdahl fraction of runtime spent in allocations.

    """
    return p[2] / (p[0] + p[2]) if p[0] + p[2] > EPS_S else 0.0


def _f_of_p_ca(p: np.ndarray, c: float) -> float:
    """Compute the Amdahl fraction of the A = N*c constrained 1-D parameters p.

    Returns:
        float: the Amdahl fraction of runtime spent in allocations.

    """
    return p[1] * c / (p[0] + p[1] * c) if p[0] + p[1] * c > EPS_S else 0.0


def _fit_setup_1d(s: np.ndarray, t: np.ndarray) -> FitSetup:
    """Precompute the s0 bounds, robust initial guess, and guess floor of a 1-D fit.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.

    Returns:
        FitSetup: the (lo_b, hi_b) bounds, the grid guess, and eps.

    """
    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = max(0.05 * s_min_pos, EPS_S)
    hi = 0.5 * (s.max() - s.min())
    guess = _grid_guess(s, t, lo, hi)  # robust, unconstrained (linear in W, N, A per s0)
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    lo_b = max(lo, EPS_S)
    hi_b = max(hi, lo_b * 1.5)
    return FitSetup(bounds=(lo_b, hi_b), guess=guess, eps=eps)


def _p0_1d(guess: tuple[float, float, float, float], eps: float, bounds: tuple[float, float]) -> list[float]:
    """Compute the initial curve_fit parameters of the unconstrained 1-D fit.

    Returns:
        list[float]: the grid guess floored at `eps`, s0 clipped to the bounds.

    """
    gW, gN, gA, gs0 = guess
    return [max(gW, eps), max(gN, eps), max(gA, eps), float(np.clip(gs0, bounds[0], bounds[1]))]


def _p0_ca(guess: tuple[float, float, float, float], eps: float, bounds: tuple[float, float]) -> list[float]:
    """Compute the initial curve_fit parameters of the A = N*c constrained 1-D fit.

    Returns:
        list[float]: the grid guess floored at `eps`, s0 clipped to the bounds.

    """
    gW, gN, _gA, gs0 = guess
    return [max(gW, eps), max(gN, eps), float(np.clip(gs0, bounds[0], bounds[1]))]


def _fit_notes_1d(model: Model1d, guess: tuple[float, float, float, float], bounds: tuple[float, float]) -> list[str]:
    """Report the corner solutions of the constrained 1-D fit.

    Returns:
        list[str]: the diagnostics (A floored at 0, s0 at the search cap).

    """
    gW, gA = guess[0], guess[2]
    W, A = model.W, model.A
    notes = []
    # ruff's SIM300 "fix" would move the constant expression to the left of
    # the comparison, i.e. create a genuine Yoda condition.
    if gA < -1e-6 * max(abs(gW), 1e-9) and A <= 1e-6 * max(abs(W), 1e-9):  # ruff: ignore[yoda-conditions]
        notes.append(
            "unconstrained fit wanted A<0 (smallest-sleeptime runtime below the Amdahl "
            "line); A constrained to 0 so f is floored at 0"
        )
    if model.s0 >= bounds[1] * 0.999:
        notes.append(
            "s0 reached the search cap: the native correction does not clearly "
            "fade within the sweep, so A and W (hence f) are weakly constrained"
        )
    return notes


def _floored_guess_1d(guess: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Floor the A term of the 1-D grid guess at zero.

    Returns:
        tuple[float, float, float, float]: the floored (W, N, A, s0) guess.

    """
    W, N, A, s0 = guess
    return W, N, max(0.0, A), s0


def _fit_3_points(s: np.ndarray, t: np.ndarray, finish: Callable[..., dict]) -> dict:
    """Fit a 3-point sweep: Amdahl line through the two largest sleeptimes.

    Too few points to determine the correction shape, so A is the (floored)
    residual at the smallest sleeptime.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.
        finish: the result assembler of the calling sweep.

    Returns:
        dict: the fit result with s0, its errors, and the covariance NaN.

    """
    (N, W), *_ = np.linalg.lstsq(np.vstack([s[-2:], np.ones(2)]).T, t[-2:], rcond=None)
    A = max(0.0, float(t[0] - (W + N * s[0])))
    return finish(
        Model1d(W, N, A, float("nan")),
        None,
        None,
        note="only 3 points: correction shape not identifiable; N and W from the two "
        "largest, A the (floored) residual at the smallest sleeptime",
    )


def _fit_constrained_1d(
    s: np.ndarray,
    t: np.ndarray,
    c_a: float | None,
    setup: FitSetup,
) -> ConstrainedFit:
    """Run curve_fit for a 1-D sweep, unconstrained or A = N*c constrained.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.
        c_a: the A = N*c constraint in nanoseconds, or None for the full model.
        setup: the precomputed bounds, guess, and floor.

    Returns:
        ConstrainedFit: the fitted model, its covariance, and corner notes.

    """
    if c_a is None:
        p0 = _p0_1d(setup.guess, setup.eps, setup.bounds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                _model,
                s,
                t,
                p0=p0,
                bounds=([0.0, 0.0, 0.0, setup.bounds[0]], [_INF, _INF, _INF, setup.bounds[1]]),
                maxfev=20000,
            )
        model = Model1d(*(float(v) for v in popt))
        return ConstrainedFit(model, pcov, list(popt), _fit_notes_1d(model, setup.guess, setup.bounds))
    c = c_a * 1e-9
    p0 = _p0_ca(setup.guess, setup.eps, setup.bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            lambda qq, W, N, s0: _model_c_a(qq, W, N, s0, c),
            s,
            t,
            p0=p0,
            bounds=([0.0, 0.0, setup.bounds[0]], [_INF, _INF, setup.bounds[1]]),
            maxfev=20000,
        )
    W, N, s0 = (float(v) for v in popt)
    model = Model1d(W, N, N * c, s0)
    return ConstrainedFit(model, pcov, list(popt), _fit_notes_1d(model, setup.guess, setup.bounds))


def fit_allocation_fraction(sleeptimes: pd.Series, runtimes: pd.Series, c_a: float | None = None) -> dict:
    """Fit one sleeptime sweep to the constrained Amdahl model.

    Returns:
        dict: W, N, A, s0, T0, f, r2 plus their standard errors (W_err, N_err,
        A_err, s0_err, f_err, NaN when unavailable), warnings, and the data
        with per-point residuals.

    Raises:
        ValueError: if the arrays have unequal shapes or fewer than 3 points.

    """
    s = np.asarray(sleeptimes, dtype=float) * 1e-9  # ns -> s
    t = np.asarray(runtimes, dtype=float)
    if s.shape != t.shape:
        msg = "sleeptimes and runtimes must have the same shape"
        raise ValueError(msg)
    if s.size < 3:
        msg = "need at least 3 data points"
        raise ValueError(msg)
    order = np.argsort(s)
    s, t = s[order], t[order]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    notes = []

    def finish(
        model: Model1d, pcov: np.ndarray | None, fit_params: list[float] | None, note: str | None = None
    ) -> dict:
        W, N, A, s0 = model.W, model.N, model.A, model.s0
        pred = _model(s, W, N, A, s0)
        r2 = 1.0 - float(np.sum((t - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        if N <= 0:
            notes.append("non-positive slope: no allocation cost visible in this sweep")
        if note:
            notes.append(note)
        W, N, A = float(W), float(N), float(A)
        s0 = float("nan") if math.isnan(s0) else float(s0)
        T0 = W + A
        f = A / T0 if T0 > EPS_S else 0.0
        if fit_params is None:
            W_e = N_e = A_e = s0_e = f_e = float("nan")
        elif c_a is None:
            W_e, N_e, A_e, s0_e, f_e = _uncertainties(fit_params, pcov, _f_of_p_free)
        else:
            c = c_a * 1e-9
            W_e, N_e, s0_e, f_e = _uncertainties(fit_params, pcov, lambda p: _f_of_p_ca(p, c))
            A_e = float("nan") if math.isnan(N_e) else N_e * c
        return {
            "W": W,  # runtime without allocation cost (s)
            "N": N,  # allocation calls per run
            "A": A,  # native allocation time (s)
            "s0": s0,  # fade scale of the native correction (s)
            "T0": T0,  # total runtime at zero delay (s)
            "f": f,  # Amdahl fraction of runtime spent in allocations
            "W_err": W_e,
            "N_err": N_e,
            "A_err": A_e,
            "s0_err": s0_e,
            "f_err": f_e,
            "r2": r2,
            "warnings": notes,
            "sleeptimes": s,
            "runtimes": t,
            "residuals": t - pred,
            "fit_params": fit_params,  # fitted parameter vector (or None)
            "pcov": pcov,  # parameter covariance matrix (or None)
        }

    if s.size == 3:
        return _fit_3_points(s, t, finish)
    setup = _fit_setup_1d(s, t)
    try:
        cf = _fit_constrained_1d(s, t, c_a, setup)
        notes.extend(cf.notes)
        return finish(cf.model, cf.pcov, cf.fit_params)
    except (RuntimeError, ValueError) as err:
        # curve_fit failed to converge; report the robust linear solution.
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); using the robust linear solution")
        return finish(
            Model1d(*_floored_guess_1d(setup.guess)),
            None,
            None,
            note="constrained fit unavailable; linear grid solution reported (A floored at 0)",
        )


def _f_of_m(p: np.ndarray) -> float:
    """Compute the Amdahl fraction of the malloc operation of the 2-D parameters p.

    Returns:
        float: the Amdahl fraction of runtime spent in allocations.

    """
    return p[3] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0


def _f_of_f(p: np.ndarray) -> float:
    """Compute the Amdahl fraction of the free operation of the 2-D parameters p.

    Returns:
        float: the Amdahl fraction of runtime spent in frees.

    """
    return p[4] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0


def _fit_bounds_2d(m: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    """Compute the search ranges of the 2-D fade scales.

    Returns:
        tuple[float, float, float, float]: the (lo_m, hi_m, lo_f, hi_f) bounds.

    """
    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)
    return lo_m, hi_m, lo_f, hi_f


def _grid_row_2d(
    m: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    m0: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, ...]:
    """Best f0 of the 2-D grid search at a fixed m0.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        m0: the fixed malloc fade scale in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.

    Returns:
        tuple[float, ...]: (ss_res, W, N_m, N_f, A_m, A_f, m0, f0) of the best row.

    """
    best = None
    for f0 in np.logspace(np.log10(bounds[2]), np.log10(bounds[3]), 25):
        X = np.vstack([np.ones_like(m), m, f, m0 / (m + m0), f0 / (f + f0)]).T
        sol, *_ = np.linalg.lstsq(X, t, rcond=None)
        ss_res = float(np.sum((t - X @ sol) ** 2))
        if best is None or ss_res < best[0]:
            best = (ss_res, float(f0), *[float(v) for v in sol])
    ss_res, f0, W, N_m, N_f, A_m, A_f = best
    return (ss_res, W, N_m, N_f, A_m, A_f, m0, f0)


def _grid_guess_2d(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[float, ...]:
    """Robust unconstrained 2-D solution, linear in (W, N_m, N_f, A_m, A_f) per (m0, f0).

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.

    Returns:
        tuple[float, ...]: W, N_m, N_f, A_m, A_f, m0, f0 minimizing the residual.

    """
    m, f = sweep
    best = None
    for m0 in np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), 25):
        row = _grid_row_2d(m, f, t, float(m0), bounds)
        if best is None or row[0] < best[0]:
            best = row
    _ss_res, W, N_m, N_f, A_m, A_f, m0, f0 = best
    return W, N_m, N_f, A_m, A_f, m0, f0


def _floored_guess_2d(guess: tuple[float, ...]) -> tuple[float, ...]:
    """Floor A_malloc and A_free of the 2-D grid guess at zero.

    Returns:
        tuple[float, ...]: the floored (W, N_m, N_f, A_m, A_f, m0, f0) guess.

    """
    W, N_m, N_f, A_m, A_f, m0, f0 = guess
    return W, N_m, N_f, max(0.0, A_m), max(0.0, A_f), m0, f0


def _p0_2d(guess: tuple[float, ...], eps: float) -> list[float]:
    """Compute the initial curve_fit parameters of the 2-D fit.

    Returns:
        list[float]: the grid guess floored at `eps`.

    """
    gW, gNm, gNf, gAm, gAf, gm0, gf0 = guess
    return [max(gW, eps), max(gNm, eps), max(gNf, eps), max(gAm, eps), max(gAf, eps), gm0, gf0]


def _fit_notes_2d(model: Model2d, guess: tuple[float, ...], bounds: tuple[float, float, float, float]) -> list[str]:
    """Report the corner solutions of the constrained 2-D fit.

    Returns:
        list[str]: the diagnostics (A floored at 0, fade scale at the cap).

    """
    gAm, gAf = guess[3], guess[4]
    hi_m, hi_f = bounds[1], bounds[3]
    notes = []
    if gAm < 0 and model.A_m <= 1e-6 * max(abs(model.W), 1e-9):
        notes.append("unconstrained fit wanted A_malloc<0; A_malloc constrained to 0 so f_malloc is floored at 0")
    if gAf < 0 and model.A_f <= 1e-6 * max(abs(model.W), 1e-9):
        notes.append("unconstrained fit wanted A_free<0; A_free constrained to 0 so f_free is floored at 0")
    if model.m0 >= hi_m * 0.999:
        notes.append(
            "m0 reached the search cap: the native malloc cost does not clearly fade, so "
            "A_malloc and W (hence f_malloc) are weakly constrained"
        )
    if model.f0 >= hi_f * 0.999:
        notes.append(
            "f0 reached the search cap: the native free cost does not clearly fade, so "
            "A_free and W (hence f_free) are weakly constrained"
        )
    return notes


def _fit_constrained_2d(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
    guess: tuple[float, ...],
) -> ConstrainedFit:
    """Run curve_fit for a 2-D sweep.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.
        guess: the robust grid solution (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        ConstrainedFit: the fitted model, its covariance, and corner notes.

    """
    m, f = sweep
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    p0 = _p0_2d(guess, eps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            lambda pf, W, N_m, N_f, A_m, A_f, m0, f0: _model_2d(pf[0], pf[1], (W, N_m, N_f, A_m, A_f, m0, f0)),
            (m, f),
            t,
            p0=p0,
            bounds=(
                [0.0, 0.0, 0.0, 0.0, 0.0, bounds[0], bounds[2]],
                [_INF, _INF, _INF, _INF, _INF, bounds[1], bounds[3]],
            ),
            maxfev=40000,
        )
    model = Model2d(*(float(v) for v in popt))
    return ConstrainedFit(model, pcov, list(popt), _fit_notes_2d(model, guess, bounds))


def fit_allocation_fraction_2d(m_delays: pd.Series, f_delays: pd.Series, runtimes: pd.Series) -> dict:
    """Fit one (malloc, free) delay combination sweep to the constrained two-operation Amdahl model.

    The model is T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0).

    Returns:
        dict: W, N_m, N_f, A_m, A_f, m0, f0, T0, f_malloc, f_free, r2 plus
        their standard errors (NaN when unavailable), warnings, and the data
        with per-point residuals.

    Raises:
        ValueError: if the arrays have unequal shapes or fewer than 3 points.

    """
    m = np.asarray(m_delays, dtype=float) * 1e-9  # ns -> s
    f = np.asarray(f_delays, dtype=float) * 1e-9
    t = np.asarray(runtimes, dtype=float)
    if m.shape != f.shape or m.shape != t.shape:
        msg = "delays and runtimes must all have the same shape"
        raise ValueError(msg)
    if m.size < 3:
        msg = "need at least 3 data points"
        raise ValueError(msg)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    notes = []

    def finish(
        model: Model2d, pcov: np.ndarray | None, fit_params: list[float] | None, note: str | None = None
    ) -> dict:
        pred = _model_2d(m, f, (model.W, model.N_m, model.N_f, model.A_m, model.A_f, model.m0, model.f0))
        r2 = 1.0 - float(np.sum((t - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        if model.N_m <= 0:
            notes.append("non-positive malloc slope: no allocation cost visible for the malloc delays")
        if model.N_f <= 0:
            notes.append("non-positive free slope: no free cost visible for the free delays")
        if note:
            notes.append(note)
        T0 = model.W + model.A_m + model.A_f
        f_malloc = model.A_m / T0 if T0 > EPS_S else 0.0
        f_free = model.A_f / T0 if T0 > EPS_S else 0.0
        if fit_params is None:
            W_e = Nm_e = Nf_e = Am_e = Af_e = m0_e = f0_e = f_malloc_e = f_free_e = float("nan")
        else:
            W_e, Nm_e, Nf_e, Am_e, Af_e, m0_e, f0_e, f_malloc_e = _uncertainties(fit_params, pcov, _f_of_m)
            f_free_e = _uncertainties(fit_params, pcov, _f_of_f)[-1]
        return {
            "W": float(model.W),  # runtime without any allocation or free cost (s)
            "N_m": float(model.N_m),  # allocation calls per run
            "N_f": float(model.N_f),  # free calls per run
            "A_m": float(model.A_m),  # native allocation time (s)
            "A_f": float(model.A_f),  # native free time (s)
            "m0": float("nan") if math.isnan(model.m0) else float(model.m0),  # malloc fade scale (s)
            "f0": float("nan") if math.isnan(model.f0) else float(model.f0),  # free fade scale (s)
            "T0": T0,  # total runtime at zero delay (s)
            "f_malloc": f_malloc,  # Amdahl fraction of runtime spent in allocations
            "f_free": f_free,  # Amdahl fraction of runtime spent in frees
            "W_err": W_e,
            "N_m_err": Nm_e,
            "N_f_err": Nf_e,
            "A_m_err": Am_e,
            "A_f_err": Af_e,
            "m0_err": m0_e,
            "f0_err": f0_e,
            "f_malloc_err": f_malloc_e,
            "f_free_err": f_free_e,
            "r2": r2,
            "warnings": notes,
            "m_delays": m,
            "f_delays": f,
            "residuals": t - pred,
            "fit_params": fit_params,  # fitted parameter vector (or None)
            "pcov": pcov,  # parameter covariance matrix (or None)
        }

    sweep = (m, f)
    bounds = _fit_bounds_2d(m, f)
    guess = _grid_guess_2d(sweep, t, bounds)
    try:
        cf = _fit_constrained_2d(sweep, t, bounds, guess)
        notes.extend(cf.notes)
        return finish(cf.model, cf.pcov, cf.fit_params)
    except (RuntimeError, ValueError) as err:
        # curve_fit failed to converge; report the robust linear solution.
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); using the robust linear solution")
        return finish(
            Model2d(*_floored_guess_2d(guess)),
            None,
            None,
            note="constrained fit unavailable; linear grid solution reported (A_malloc/A_free floored at 0)",
        )


def _to_ns(value: float) -> float:
    value = float(value)
    return float("nan") if math.isnan(value) else value * 1e9


def _fit_cov(res: dict) -> tuple | None:
    """Extract the fitted parameter vector and its covariance for the bootstrap sleeves.

    Returns:
        tuple | None: (fit_params, pcov) for the bootstrap sleeves, or None if unavailable.

    """
    return (res["fit_params"], res["pcov"]) if res["fit_params"] is not None else None


def _fit_one_group(m: pd.Series, f: pd.Series, runtimes: pd.Series, c_a: float | None) -> dict:
    """Fit one (setup, algorithm, grid) group and map the result onto a row.

    Groups spanning both delays get the two-operation model; groups spanning
    one delay fall back to the 1-D model on that delay.

    Args:
        m: the group's malloc sleeptimes in nanoseconds.
        f: the group's free sleeptimes in nanoseconds.
        runtimes: the group's runtimes in seconds.
        c_a: the A = N*c constraint in nanoseconds, or None.

    Returns:
        dict: the fitted row fields (model, parameters, note, cov).

    """
    if m.nunique() >= 2 and f.nunique() >= 2:
        res = fit_allocation_fraction_2d(m, f, runtimes)
        return {
            "model": "2d",
            "W": res["W"],
            "T0": res["T0"],
            "r2": res["r2"],
            "N_malloc": res["N_m"],
            "A_malloc": res["A_m"],
            "m0_ns": _to_ns(res["m0"]),
            "f_malloc": res["f_malloc"],
            "f_malloc_err": res["f_malloc_err"],
            "N_free": res["N_f"],
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
        res = fit_allocation_fraction(delays, runtimes, c_a=c_a)
        row = {
            "model": f"1d-{varying}",
            "W": res["W"],
            "T0": res["T0"],
            "r2": res["r2"],
        }
        if varying == "malloc":
            row.update(
                {
                    "N_malloc": res["N"],
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


def fit_sweep(df: pd.DataFrame, c_a: float | None = None, configuration: str | None = None) -> pd.DataFrame:
    """Fit every (setup, algorithm, x, y, z) group of a parsed sweep DataFrame.

    Groups whose runs span both the malloc and the free delay are fitted with
    the two-operation model of `fit_allocation_fraction_2d`; groups spanning
    only one delay fall back to the 1-D model of `fit_allocation_fraction`
    on that delay. `df` is the output of `parse_logs`; `configuration`, if
    given, restricts the fit to that configuration.

    Returns:
        pd.DataFrame: one row per group with the fitted parameters; the
        `cov` column carries the fitted parameter vector and its covariance
        (or None) so the plot can draw bootstrap sleeves around the fit lines.

    """
    if configuration is not None:
        df = df[df["configuration"] == configuration]
    no_fit = {
        "model": None,
        "W": np.nan,
        "T0": np.nan,
        "r2": np.nan,
        "N_malloc": np.nan,
        "A_malloc": np.nan,
        "m0_ns": np.nan,
        "f_malloc": np.nan,
        "f_malloc_err": np.nan,
        "N_free": np.nan,
        "A_free": np.nan,
        "f0_ns": np.nan,
        "f_free": np.nan,
        "f_free_err": np.nan,
        "note": None,
        "cov": None,
    }
    rows = []
    for key, frame in df.groupby(list(GROUP_KEYS), dropna=False):
        grp = frame.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime in s"])
        row = {**dict(zip(GROUP_KEYS, key, strict=True)), "n_runs": len(grp), **no_fit}
        m, f = grp["malloc_sleeptime"], grp["free_sleeptime"]
        try:
            row.update(_fit_one_group(m, f, grp["runtime in s"], c_a))
        except ValueError as err:
            row.update({**no_fit, "note": str(err)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(GROUP_KEYS), na_position="last")


def _print_fraction_summary(fits: pd.DataFrame) -> None:
    # Name the algorithm only when more than one is present: single-policy
    # output stays exactly as before multi-algorithm sweeps.
    show_algorithm = fits["algorithm"].nunique() > 1
    ok = fits[fits["f_malloc"].between(0, 1, inclusive="neither") | fits["f_free"].between(0, 1, inclusive="neither")]
    if len(ok):
        print("\nAmdahl fraction of runtime spent in the operation (f = A/T0):")
        for _, r in ok.iterrows():
            fractions = []
            for name in ("f_malloc", "f_free"):
                if r[name] == r[name]:
                    err = "" if r[f"{name}_err"] != r[f"{name}_err"] else f" +/- {100 * r[f'{name}_err']:.1f}"
                    fractions.append(f"{name} = {100 * r[name]:.1f}{err}%")
            extra = [f"A_{tag} = {r[f'A_{tag}']:.2f} s" for tag in ("malloc", "free") if r[f"A_{tag}"] == r[f"A_{tag}"]]
            algorithm = f"{r.algorithm:<14s} " if show_algorithm else ""
            print(
                f"  {r.setup:<16s} {algorithm}grid {int(r.x)}x{int(r.y)}"
                + (f"x{int(r.z)}" if pd.notna(r.z) else "")
                + f" [{r.model}] : "
                + ", ".join(fractions)
                + f"   (W = {r.W:.2f} s"
                + (", " + ", ".join(extra) if extra else "")
                + ")"
            )
            if r["note"]:
                print(f"      note: {r.note}")


def main(clusters: dict | None = None, *, show: bool = False) -> None:
    """Parse, fit and plot every cluster's delay sweeps.

    With `show`, the figures are displayed in a window (blocking); by default
    they are only written to `figures/`.

    """
    per_cluster = []
    cluster_names = []
    for name, (log_dir, title) in (clusters or CLUSTERS).items():
        log_paths = sorted(Path(log_dir).glob("run_*"))
        if not log_paths:
            continue
        print(f"=== {title}: {log_dir} ({len(log_paths)} log files) ===")
        full_results = parse_logs(log_paths)
        print(full_results)
        if CONFIGURATION is not None:
            # The parsed data is complete; only the plotted subset is filtered.
            full_results = full_results[full_results["configuration"] == CONFIGURATION]
        simple_results = simple_statistics(full_results)
        print(simple_results)
        fits = fit_sweep(full_results)
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(fits.drop(columns=["cov"]).to_string(index=False, float_format=lambda v: f"{v:10.3g}"))
        _print_fraction_summary(fits)
        per_cluster.append((title, simple_results, fits))
        cluster_names.append(name)
    if per_cluster:
        # One figure per cluster: the malloc sweep (left) and the free sweep
        # (right) next to each other.
        figs = simple_plot(per_cluster)
        FIGURES.mkdir(exist_ok=True)
        for fig, name in zip(figs, cluster_names, strict=True):
            fig.savefig(FIGURES / f"{name}.pdf")
        # One cross-cluster figure: the runtime budget stacked per scenario.
        stacked_runtime_fig(per_cluster).savefig(FIGURES / "runtime-stack.pdf")
        if show:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PIConGPU/mallocMC allocation-latency benchmark logs.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the figures in a window (blocking); by default they are only saved",
    )
    args = parser.parse_args()
    main(show=args.show)
