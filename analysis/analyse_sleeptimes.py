"""Analyze PIConGPU/mallocMC allocation-latency benchmark logs.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads the raw `run_all.sh` logs directly (no pre-filtering; both the old
per-variant layout and the current one) and fits each (example, grid) sweep
to the constrained Amdahl allocation model. It then plots the runtime
against the imposed delay (median with IQR error bars) in one figure per
cluster (see `CLUSTERS`), the figure titled by the hardware the runs were
made on, with the malloc sleep_time sweep (free sleep_time = 0) on the
left and the free sleep_time sweep (malloc sleep_time = 0) on the right,
each axis titled after the swept delay ("malloc time scan", "free time
scan"), the two axes sharing the y-axis. Each swept
curve overlays the fitted model (solid), the extrapolation to
A_malloc = A_free = 0 (dashed) and, for two-operation fits, the
intermediate extrapolation that keeps only the held operation's native
cost (dotted); the gaps at the smallest delay are annotated with the
native cost A and Amdahl fraction f of each operation. Each fit line
carries a transparent sleeve, the 25/75-percentile envelope of the model
evaluated at 512 parameter draws from the fitted parameters and their
covariance (a bootstrap, since the parameters enter the model
non-linearly):

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

import math
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

ALGORITHM = "FlatterScatter"
# The creation policies the benchmark can run, in the row order of the
# figures; a figure only shows the algorithms present in its data.
ALGORITHM_ORDER = ("FlatterScatter", "ScatterAlloc", "Gallatin")
CD_CMD = "+ cd "
RUN_CMD = "bin/picongpu "
MALLOC_DELAY_CMD = "MALLOCMC_MALLOC_DELAY="
FREE_DELAY_CMD = "MALLOCMC_FREE_DELAY="
# Pre-rename logs: the malloc delay was then injected via MALLOCMC_SLEEP_TIME.
LEGACY_MALLOC_DELAY_CMD = "MALLOCMC_SLEEP_TIME="
# `cd` trace lines that carry run context: the oldest per-variant layout
# (`cd build/<Example>/<Algorithm>-sleep<N>`, delays compiled in), the
# current one-build-per-(example, algorithm) layout
# (`cd .../build/<Example>/<Algorithm>`, delays still run-time), and the
# interim one-build-per-example layout (`cd .../build/<Example>`, which
# always ran FlatterScatter); every other `cd` (for example the `cd $WD`
# return in run_folder.sh) is ignored. The three patterns are mutually
# exclusive: `-sleep` breaks the trailing `\w+$` of the other two, and the
# per-algorithm path has one word component too many for the per-example
# one.
VARIANT_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)-sleep(\d+)$")
BUILD_ALGO_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)$")
BUILD_CD_RE = re.compile(r"(?:^|/)build/(\w+)$")
# cluster name -> (run-log directory, hardware name for the figure title)
CLUSTERS = {
    "hal": (Path("output") / "hal-sleeptimes", "NVIDIA A30"),
    "rosi": (Path("output") / "rosi-sleeptimes", "NVIDIA V100"),
}
# The statistics, plot and fit are computed only for runs with this
# configuration: "run-time" (delays injected via MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY) or "compile-time" (per-variant builds); None uses all
# of them. The parsed results are always complete.
CONFIGURATION = None

GROUP_KEYS = ("setup", "algorithm", "x", "y", "z")
MALLOC_DELAY = "malloc_sleeptime"
FREE_DELAY = "free_sleeptime"
DELAY_COLUMNS = (MALLOC_DELAY, FREE_DELAY)
# one distinct marker per (setup, grid) series, shared by all axes
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
_INF = float("inf")
# Near-zero floor for second-based runtimes: the Amdahl fraction is
# reported as 0 when the total runtime drops below it, and the fade
# scales (s0, m0, f0) are bounded by it from below.
EPS_S = 1e-12
# A gap marker is only drawn when the model difference exceeds this (s).
VISIBLE_GAP_S = 1e-9
# Label placement in axes fraction: boxes within this of touching count as
# overlapping, and at least this much clear space is kept between them.
LABEL_OVERLAP_EPS = 0.005
LABEL_MIN_SPACE = 0.02


def parse_setup(line: str) -> dict | None:
    """Parse the run context of a `cd` trace line; None if the line is unrelated.

    Returns:
        dict | None: the parsed setup context, or None if the line is unrelated.

    """
    path = line.rsplit(maxsplit=1)[-1]
    m = VARIANT_CD_RE.search(path)
    if m:
        # One build per (example, algorithm, sleeptime): the delay was
        # compiled into the binary (a malloc delay, no free delay).
        return {
            "setup": m[1],
            "algorithm": m[2],
            "malloc_sleeptime": int(m[3]),
            "free_sleeptime": 0,
            "configuration": "compile-time",
        }
    m = BUILD_ALGO_CD_RE.search(path)
    if m:
        # One build per (example, algorithm): the creation policy is
        # compiled into the binary; the delays are still injected at run
        # time via the MALLOCMC_*_DELAY environment variables.
        return {"setup": m[1], "algorithm": m[2]}
    m = BUILD_CD_RE.search(path)
    if m:
        return {"setup": m[1], "algorithm": ALGORITHM}
    return None


def parse_grid(line: str) -> dict[str, int]:
    """Parse the `-g` grid dimensions out of a picongpu command line.

    Returns:
        dict[str, int]: the grid dimensions, keyed by x, y, z.

    """
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            line.split(RUN_CMD, 1)[1].split("-g", 1)[1].split("-", maxsplit=1)[0].strip().split(" "),
            # 2-D grids have only two values; the zip truncates the keys to
            # the dimensions present (the missing one becomes NaN downstream).
            strict=False,
        )
    }


def parse_simulation_time(line: str) -> dict[str, float]:
    """Parse a `calculation  simulation time` line into a runtime dict.

    Returns:
        dict[str, float]: the simulation runtime in seconds.

    """
    return {"runtime in s": float(line.split("=")[1][: -len("sec")])}


def parse_log(log_path: Path) -> Iterator[dict]:
    """Yield one record per picongpu run of a single run log.

    Yields:
        dict: one record per picongpu run of the log.

    """
    with log_path.open("r", encoding="utf-8") as file:
        context = {}
        pending = None
        malloc_delay = None
        free_delay = None
        for line in map(str.strip, file):
            if line.startswith(CD_CMD):
                # A new run context invalidates the remembered delay values.
                setup = parse_setup(line)
                if setup is not None:
                    context = setup
                    malloc_delay = None
                    free_delay = None
            elif line.startswith("+ "):
                # With `set -x`, the delay env prefixes are traced on their
                # own line(s) before the picongpu line; the new layout puts
                # both on the same line. Pre-rename logs only carry the malloc
                # prefix, under the legacy name.
                if MALLOC_DELAY_CMD in line:
                    malloc_delay = int(line.split(MALLOC_DELAY_CMD, 1)[1].split()[0])
                elif LEGACY_MALLOC_DELAY_CMD in line:
                    malloc_delay = int(line.split(LEGACY_MALLOC_DELAY_CMD, 1)[1].split()[0])
                if FREE_DELAY_CMD in line:
                    free_delay = int(line.split(FREE_DELAY_CMD, 1)[1].split()[0])
                if RUN_CMD in line and "setup" in context:
                    # In the run-time layout the env vars override the variant
                    # sleeptime; in the per-variant layout they are absent.
                    pending = dict(context) | parse_grid(line)
                    if malloc_delay is not None or free_delay is not None:
                        pending |= {
                            "malloc_sleeptime": (malloc_delay if malloc_delay is not None else 0),
                            "free_sleeptime": (free_delay if free_delay is not None else 0),
                            "configuration": "run-time",
                        }
            elif line.startswith("calculation") and "simulation time" in line and pending is not None:
                yield {**pending, **parse_simulation_time(line)}
                pending = None


def run_to_df(run: dict) -> pd.DataFrame:
    """Build a DataFrame from one run's records, tagged with its name.

    Returns:
        pd.DataFrame: the run's records, tagged with the run's name.

    """
    return pd.DataFrame(run["runs"]).assign(name=run["name"])


def runs_to_df(runs: Iterable[dict]) -> pd.DataFrame:
    """Concatenate the per-run DataFrames, filling a missing z with NaN.

    Returns:
        pd.DataFrame: the concatenated per-run frames, a missing z filled with NaN.

    """
    tmp = pd.concat(map(run_to_df, runs))
    return tmp.assign(z=tmp.get("z", np.nan))


def parse_logs(log_paths: Iterable[Path]) -> pd.DataFrame:
    """Parse every run log into a single DataFrame.

    Returns:
        pd.DataFrame: every run log parsed into one frame.

    """
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)


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
            fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5.5), sharey=True, layout="constrained")
            fig.suptitle(title)
            _plot_cluster(axes[0], simple_results, fits, series_markers, x_delay=MALLOC_DELAY)
            _plot_cluster(axes[1], simple_results, fits, series_markers, x_delay=FREE_DELAY)
            # The y-axis is shared: subplots already hides the right axis'
            # tick labels, drop its label too.
            axes[1].set_ylabel("")
            # Legend on the left-most axis that actually has a curve (one
            # cluster's delay sweep may not have arrived yet).
            for ax in axes:
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
                    break
        else:
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
            for malloc_ax, free_ax in row_axes:
                for ax in (malloc_ax, free_ax):
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend()
                        break
                else:
                    continue
                break
        figs.append(fig)
    return figs


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
    ax: plt.Axes,
    x_ns: np.ndarray,
    x_s: np.ndarray,
    x0: float,
    held_s: float,
    params: Fit2d,
    short: str,
    color: str,
    gaps: list[tuple[float, float, str, str]],
    sleeve: Callable[[_ModelFn, tuple], None],
) -> None:
    """Draw the 2-D fit of one curve and its two A/f gap markers.

    The solid line is the full model, the dashed line the linear
    extrapolation (both Amdahl terms at 0) and the dotted line the fit
    with only the plotted operation's Amdahl term at 0. The gap
    markers (appended to `gaps`) split at the dotted line: the upper
    part is the native cost of the plotted operation, the lower part
    the one of the held operation.
    """
    held_arr = np.full_like(x_s, held_s)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S)
    if short == "malloc":
        curve = _model_2d(
            x_s,
            held_arr,
            params.W,
            params.n_malloc,
            params.n_free,
            params.a_malloc,
            params.a_free,
            params.m0,
            params.f0,
        )
        linear = params.W + params.n_malloc * x_s + params.n_free * held_s
        dotted = linear + params.a_free * params.f0 / (held_s + params.f0)
        a_up, a_dn = params.a_malloc, params.a_free

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return _model_2d(x_s, held_arr, *p)

        def fn_dash(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s

        def fn_dot(p: np.ndarray) -> np.ndarray:
            return p[0] + p[1] * x_s + p[2] * held_s + p[4] * p[6] / (held_s + p[6])
    else:
        curve = _model_2d(
            held_arr,
            x_s,
            params.W,
            params.n_malloc,
            params.n_free,
            params.a_malloc,
            params.a_free,
            params.m0,
            params.f0,
        )
        linear = params.W + params.n_free * x_s + params.n_malloc * held_s
        dotted = linear + params.a_malloc * params.m0 / (held_s + params.m0)
        a_up, a_dn = params.a_free, params.a_malloc

        def fn_curve(p: np.ndarray) -> np.ndarray:
            return _model_2d(held_arr, x_s, *p)

        def fn_dash(p: np.ndarray) -> np.ndarray:
            return p[0] + p[2] * x_s + p[1] * held_s

        def fn_dot(p: np.ndarray) -> np.ndarray:
            return p[0] + p[2] * x_s + p[1] * held_s + p[3] * p[5] / (held_s + p[5])

    t0 = params.W + params.a_malloc + params.a_free
    held_name = "free" if short == "malloc" else "malloc"
    if float(curve[0]) - float(dotted[0]) > VISIBLE_GAP_S:
        y_lo, y_hi = float(dotted[0]), float(curve[0])
        ax.plot((x0, x0), (y_lo, y_hi), color=color, linewidth=1, alpha=0.8)
        f_val = a_up / t0 if t0 > EPS_S else 0.0
        gaps.append((np.sqrt(y_lo * y_hi), x0, color, f"A_{short} = {a_up:.2f} s, f = {100 * f_val:.1f}%"))
    if float(dotted[0]) - float(linear[0]) > VISIBLE_GAP_S:
        y_lo, y_hi = float(linear[0]), float(dotted[0])
        ax.plot((x0, x0), (y_lo, y_hi), color=color, linewidth=1, alpha=0.8)
        f_val = a_dn / t0 if t0 > EPS_S else 0.0
        gaps.append((np.sqrt(y_lo * y_hi), x0, color, f"A_{held_name} = {a_dn:.2f} s, f = {100 * f_val:.1f}%"))
    # The model takes second-based delays; the axis is in ns.
    sleeve(fn_dot, lower)
    ax.plot(x_ns, dotted, color=color, linestyle=":", alpha=0.8)
    sleeve(fn_curve, lower)
    ax.plot(x_ns, curve, color=color, linestyle="-", alpha=0.8)
    sleeve(fn_dash, lower)
    ax.plot(x_ns, linear, color=color, linestyle="--", alpha=0.8)


def _draw_fit_1d(
    ax: plt.Axes,
    x_ns: np.ndarray,
    x_s: np.ndarray,
    x0: float,
    params: Fit1d,
    short: str,
    color: str,
    gaps: list[tuple[float, float, str, str]],
    sleeve: Callable[[_ModelFn, tuple], None],
) -> None:
    """Draw the 1-D fit of one curve and its A/f gap marker.

    The solid line is the full model, the dashed line the linear
    extrapolation to A = 0; the marker (appended to `gaps`) spans the
    gap between them at the smallest delay.
    """
    lower = (0.0, 0.0, 0.0, EPS_S)
    curve = _model(x_s, params.W, params.N, params.A, params.s0)
    linear = params.W + params.N * x_s

    def fn_curve(p: np.ndarray) -> np.ndarray:
        return _model(x_s, p[0], p[1], p[2], p[3])

    def fn_dash(p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x_s

    if float(curve[0]) > float(linear[0]):
        y_lo, y_hi = float(linear[0]), float(curve[0])
        ax.plot((x0, x0), (y_lo, y_hi), color=color, linewidth=1, alpha=0.8)
        f_val = params.A / (params.W + params.A) if params.W + params.A > EPS_S else 0.0
        gaps.append((np.sqrt(y_lo * y_hi), x0, color, f"A_{short} = {params.A:.2f} s, f = {100 * f_val:.1f}%"))
    # The model takes second-based delays; the axis is in ns.
    sleeve(fn_curve, lower)
    ax.plot(x_ns, curve, color=color, linestyle="-", alpha=0.8)
    sleeve(fn_dash, lower)
    ax.plot(x_ns, linear, color=color, linestyle="--", alpha=0.8)


def _draw_fit_curves(
    ax: plt.Axes,
    x_ns: np.ndarray,
    x_s: np.ndarray,
    x0: float,
    held_s: float,
    params: Fit1d | Fit2d,
    short: str,
    color: str,
    gaps: list[tuple[float, float, str, str]],
) -> bool:
    """Draw the fitted model lines of one curve and its A/f gap markers.

    Dispatches on the fit type and appends the gap markers to `gaps`.

    Returns:
        bool: True when a 2-D fit was drawn.

    """

    def sleeve(fn: _ModelFn, lower: tuple) -> None:
        # Bootstrap sleeve of a fit line: draw the fitted
        # parameters from their covariance (the model is
        # non-linear in them) and fill the 25/75-percentile
        # envelope of the model values, the IQR convention of
        # the data's error bars. No covariance -> no sleeve.
        if params.cov is None:
            return
        band = _bootstrap_band(x_s, fn, params.cov[0], params.cov[1], lower=lower)
        if band is not None:
            ax.fill_between(x_ns, band[0], band[1], color=color, alpha=0.3)

    if isinstance(params, Fit2d):
        _draw_fit_2d(ax, x_ns, x_s, x0, held_s, params, short, color, gaps, sleeve)
        return True
    _draw_fit_1d(ax, x_ns, x_s, x0, params, short, color, gaps, sleeve)
    return False


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
    # One curve per (setup, algorithm, grid, held delay): the x-axis is
    # `x_delay`.
    plot_keys = (*GROUP_KEYS, secondary)
    results = simple_results.groupby(list(plot_keys), dropna=False)
    gaps = []
    delays = set()
    has_2d = False
    for name, result in results:
        x, ye_min, y, ye_max = np.sort(result.reset_index(drop=False)[[x_delay, "25%", "50%", "75%"]].to_numpy().T)
        # Only the pure sweep is shown: runs where the other (held) delay is
        # 0. Runs with both delays > 0 belong to neither figure.
        if name[5] != 0:
            continue
        # A sweep needs at least two distinct x values above 0.
        if len(np.unique(x[x > 0])) < 2:
            continue
        delays.update(x[x > 0])
        # The x-axis is logarithmic, so the zero-delay point is not
        # representable there (it is invisible on the plot anyway); drop it
        # so it cannot corrupt the autoscaled x limits.
        pos = x > 0
        x, ye_min, y, ye_max = x[pos], ye_min[pos], y[pos], ye_max[pos]
        eb = ax.errorbar(
            x,
            y,
            yerr=(y - ye_min, ye_max - y),
            linestyle="none",
            marker=series_markers[_group_key(name[:5])],
            label=label(name, secondary_short),
        )
        color = eb.lines[0].get_color()
        params = fits_by_key.get(_group_key(name[:5]))
        # A 1-D fit only applies when it was made on the plotted delay; the
        # 2-D fit applies to either direction.
        if params is not None and isinstance(params, Fit1d) and params.direction != short:
            params = None
        if params is not None:
            held_s = float(name[5]) * 1e-9
            # Draw the fitted model over the x-delay range this curve covers.
            x_data = x[x > 0]
            if len(x_data) > 1 and float(x_data[-1]) > float(x_data[0]):
                x_ns = np.geomspace(float(x_data[0]), float(x_data[-1]), 100)
                x_s = x_ns * 1e-9
                x0 = float(x_ns[0])
                has_2d = _draw_fit_curves(ax, x_ns, x_s, x0, held_s, params, short, color, gaps) or has_2d
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
    _place_gap_labels(ax, gaps, delays)
    return ax


def _place_gap_labels(ax: plt.Axes, gaps: list[tuple[float, float, str, str]], delays: set[float]) -> None:
    """Place the A/f gap labels and their leader lines on the axis.

    Each label sits in the clear band between the first and second
    delays, at their geometric mean; a thin leader line joins it to
    its gap marker. Labels that touch or overlap are separated
    vertically until they no longer do.
    """
    delay_xs = sorted(delays)
    if not (gaps and len(delay_xs) >= 2):
        return
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    gc = float(10 ** (0.5 * (np.log10(delay_xs[0]) + np.log10(delay_xs[1]))))
    gc_frac = (np.log10(gc) - np.log10(x_lo)) / (np.log10(x_hi) - np.log10(x_lo))
    placed = []
    for y_mid, x0, color, text in gaps:
        mid_frac = (np.log10(y_mid) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        lfy = min(mid_frac + LABEL_MIN_SPACE, 0.96)
        # The leader line is its own artist (a bbox-anchored arrow breaks
        # matplotlib's layout path clipping). Its tail is anchored at
        # the label centre in axes fraction -- the same coordinates as the
        # text -- so it tracks the box through the layout. The box,
        # drawn on top, hides the part of the line under it, so the visible
        # segment runs from the box edge to the gap.
        arrow = ax.annotate(
            "",
            xy=(x0, y_mid),
            xytext=(gc_frac, lfy),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8, "alpha": 0.8},
            zorder=4,
        )
        lab = ax.text(
            gc_frac,
            lfy,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.95,
                "linewidth": 0.5,
            },
            zorder=5,
        )
        placed.append([arrow, lab, lfy])
    if len(placed) > 1:
        # A 2-D group contributes two labels (one per operation); separate
        # any that touch or overlap, vertically, in axes fraction. The
        # arrows follow their box centres.
        inv = ax.transAxes.inverted()
        for _ in range(64):
            ax.figure.canvas.draw()
            boxes = []
            for _, lab, _lfy in placed:
                c = inv.transform(lab.get_window_extent(ax.figure.canvas.get_renderer()).corners())
                boxes.append((c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()))
            moved = False
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    xi0, xi1, yi0, yi1 = boxes[i]
                    xj0, xj1, yj0, yj1 = boxes[j]
                    if min(xi1, xj1) - max(xi0, xj0) <= 0:
                        continue
                    y_overlap = min(yi1, yj1) - max(yi0, yj0)
                    if y_overlap > -LABEL_OVERLAP_EPS:
                        shift = 0.5 * (y_overlap + LABEL_MIN_SPACE)
                        if 0.5 * (yi0 + yi1) >= 0.5 * (yj0 + yj1):
                            placed[i], placed[j] = (
                                (placed[i][0], placed[i][1], min(max(placed[i][2] + shift, 0.0), 1.0)),
                                (placed[j][0], placed[j][1], min(max(placed[j][2] - shift, 0.0), 1.0)),
                            )
                        else:
                            placed[i], placed[j] = (
                                (placed[i][0], placed[i][1], min(max(placed[i][2] - shift, 0.0), 1.0)),
                                (placed[j][0], placed[j][1], min(max(placed[j][2] + shift, 0.0), 1.0)),
                            )
                        placed[i][1].set_position((gc_frac, placed[i][2]))
                        placed[j][1].set_position((gc_frac, placed[j][2]))
                        moved = True
            if not moved:
                break
        for arrow, lab, lfy in placed:
            arrow.xyann = (gc_frac, lfy)
            lab.set_position((gc_frac, lfy))


def _model(s: np.ndarray, W: float, N: float, A: float, s0: float) -> np.ndarray:
    return W + N * s + A * s0 / (s + s0)


def _model_2d(
    m: np.ndarray,
    f: np.ndarray,
    W: float,
    N_m: float,
    N_f: float,
    A_m: float,
    A_f: float,
    m0: float,
    f0: float,
) -> np.ndarray:
    return W + N_m * m + N_f * f + A_m * m0 / (m + m0) + A_f * f0 / (f + f0)


def _model_c_a(s: np.ndarray, W: float, N: float, s0: float, c: float) -> np.ndarray:
    A = N * c
    return W + N * s + A * s0 / (s + s0)


def _bootstrap_band(
    x_s: np.ndarray,
    fn: _ModelFn,
    params: Sequence[float],
    pcov: np.ndarray,
    n: int = 512,
    percentiles: tuple[float, float] = (25.0, 75.0),
    seed: int = 0,
    lower: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Percentile envelope of `fn(x_s, *p)` over p ~ N(params, pcov).

    The parameters enter the model non-linearly (the Amdahl terms), so the
    sleeve is a resampling bootstrap rather than an analytic error
    propagation: draw `n` parameter vectors from the multivariate normal of
    the fitted parameters and their covariance, evaluate the line's model
    function `fn` on the x-grid `x_s` (in seconds) for each.

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
    rng = np.random.default_rng(seed)
    samples = np.asarray(params, dtype=float) + (rng.standard_normal((n, len(params))) * np.sqrt(eig)) @ evec.T
    if lower is not None:
        # The multivariate tails can cross the fit's bounds; clip them so the
        # A*s0/(x+s0)-type terms cannot blow up on the wrong side.
        samples = np.clip(samples, np.asarray(lower, dtype=float), None)
    values = np.empty((n, x_s.size))
    for i, p in enumerate(samples):
        values[i] = np.asarray(fn(p), dtype=float)
    lo, hi = np.percentile(values, percentiles, axis=0)
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
        W: float,
        N: float,
        A: float,
        s0: float,
        pcov: np.ndarray | None,
        fit_params: list[float] | None,
        f_of_p: Callable[[np.ndarray], float],
        note: str | None = None,
    ) -> dict:
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
            W_e, N_e, A_e, s0_e, f_e = _uncertainties(fit_params, pcov, f_of_p)
        else:
            c = c_a * 1e-9
            W_e, N_e, s0_e, f_e = _uncertainties(fit_params, pcov, f_of_p)
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

    def f_free(p: np.ndarray) -> float:
        return p[2] / (p[0] + p[2]) if p[0] + p[2] > EPS_S else 0.0

    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = max(0.05 * s_min_pos, EPS_S)
    hi = 0.5 * (s.max() - s.min())

    # Robust, unconstrained initial guess (linear in W, N, A per s0).
    gW, gN, gA, gs0 = _grid_guess(s, t, lo, hi)
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))

    if s.size == 3:
        # Too few points to determine the correction shape: Amdahl line through
        # the two largest sleeptimes, A the (floored) residual at the smallest.
        (N, W), *_ = np.linalg.lstsq(np.vstack([s[-2:], np.ones(2)]).T, t[-2:], rcond=None)
        A = max(0.0, float(t[0] - (W + N * s[0])))
        return finish(
            W,
            N,
            A,
            float("nan"),
            None,
            None,
            f_free,
            note="only 3 points: correction shape not identifiable; N and W from the two "
            "largest, A the (floored) residual at the smallest sleeptime",
        )

    lo_b = max(lo, EPS_S)
    hi_b = max(hi, lo_b * 1.5)
    try:
        if c_a is None:
            p0 = [
                max(gW, eps),
                max(gN, eps),
                max(gA, eps),
                float(np.clip(gs0, lo_b, hi_b)),
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    _model,
                    s,
                    t,
                    p0=p0,
                    bounds=([0.0, 0.0, 0.0, lo_b], [_INF, _INF, _INF, hi_b]),
                    maxfev=20000,
                )
            W, N, A, s0 = (float(v) for v in popt)
            # ruff's SIM300 "fix" would move the constant expression to the
            # left of the comparison, i.e. create a genuine Yoda condition.
            if gA < -1e-6 * max(abs(gW), 1e-9) and A <= 1e-6 * max(abs(W), 1e-9):  # ruff: ignore[SIM300]
                notes.append(
                    "unconstrained fit wanted A<0 (smallest-sleeptime runtime below the Amdahl "
                    "line); A constrained to 0 so f is floored at 0"
                )
            if s0 >= hi_b * 0.999:
                notes.append(
                    "s0 reached the search cap: the native correction does not clearly "
                    "fade within the sweep, so A and W (hence f) are weakly constrained"
                )
            return finish(W, N, A, s0, pcov, list(popt), f_free)
        c = c_a * 1e-9

        def f_ca(p: np.ndarray) -> float:
            return p[1] * c / (p[0] + p[1] * c) if p[0] + p[1] * c > EPS_S else 0.0

        p0 = [max(gW, eps), max(gN, eps), float(np.clip(gs0, lo_b, hi_b))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                lambda qq, W, N, s0: _model_c_a(qq, W, N, s0, c),
                s,
                t,
                p0=p0,
                bounds=([0.0, 0.0, lo_b], [_INF, _INF, hi_b]),
                maxfev=20000,
            )
        W, N, s0 = (float(v) for v in popt)
        A = N * c
        if s0 >= hi_b * 0.999:
            notes.append(
                "s0 reached the search cap: the native correction does not clearly "
                "fade within the sweep, so A and W (hence f) are weakly constrained"
            )
        return finish(W, N, A, s0, pcov, list(popt), f_ca)
    except (RuntimeError, ValueError) as err:
        # curve_fit failed to converge; report the robust linear solution.
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); using the robust linear solution")
        return finish(
            gW,
            gN,
            max(0.0, gA),
            gs0,
            None,
            None,
            f_free,
            note="constrained fit unavailable; linear grid solution reported (A floored at 0)",
        )


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
        W: float,
        N_m: float,
        N_f: float,
        A_m: float,
        A_f: float,
        m0: float,
        f0: float,
        pcov: np.ndarray | None,
        fit_params: list[float] | None,
        note: str | None = None,
    ) -> dict:
        pred = _model_2d(m, f, W, N_m, N_f, A_m, A_f, m0, f0)
        r2 = 1.0 - float(np.sum((t - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        if N_m <= 0:
            notes.append("non-positive malloc slope: no allocation cost visible for the malloc delays")
        if N_f <= 0:
            notes.append("non-positive free slope: no free cost visible for the free delays")
        if note:
            notes.append(note)
        T0 = W + A_m + A_f
        f_malloc = A_m / T0 if T0 > EPS_S else 0.0
        f_free = A_f / T0 if T0 > EPS_S else 0.0
        if fit_params is None:
            W_e = Nm_e = Nf_e = Am_e = Af_e = m0_e = f0_e = f_malloc_e = f_free_e = float("nan")
        else:

            def f_of_m(p: np.ndarray) -> float:
                return p[3] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0

            def f_of_f(p: np.ndarray) -> float:
                return p[4] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0

            W_e, Nm_e, Nf_e, Am_e, Af_e, m0_e, f0_e, f_malloc_e = _uncertainties(fit_params, pcov, f_of_m)
            f_free_e = _uncertainties(fit_params, pcov, f_of_f)[-1]
        return {
            "W": float(W),  # runtime without any allocation or free cost (s)
            "N_m": float(N_m),  # allocation calls per run
            "N_f": float(N_f),  # free calls per run
            "A_m": float(A_m),  # native allocation time (s)
            "A_f": float(A_f),  # native free time (s)
            "m0": float("nan") if math.isnan(m0) else float(m0),  # malloc fade scale (s)
            "f0": float("nan") if math.isnan(f0) else float(f0),  # free fade scale (s)
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

    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)

    def grid_guess() -> tuple[float, float, float, float, float, float, float]:
        # Robust unconstrained solution: linear in (W, N_m, N_f, A_m, A_f)
        # for each (m0, f0) on a log grid.
        best = None
        for m0 in np.logspace(np.log10(lo_m), np.log10(hi_m), 25):
            h_m = m0 / (m + m0)
            for f0 in np.logspace(np.log10(lo_f), np.log10(hi_f), 25):
                h_f = f0 / (f + f0)
                X = np.vstack([np.ones_like(m), m, f, h_m, h_f]).T
                sol, *_ = np.linalg.lstsq(X, t, rcond=None)
                ss_res = float(np.sum((t - X @ sol) ** 2))
                if best is None or ss_res < best[0]:
                    best = (ss_res, float(m0), float(f0), *[float(v) for v in sol])
        _, m0, f0, W, N_m, N_f, A_m, A_f = best
        return W, N_m, N_f, A_m, A_f, m0, f0

    gW, gNm, gNf, gAm, gAf, gm0, gf0 = grid_guess()
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    try:
        p0 = [
            max(gW, eps),
            max(gNm, eps),
            max(gNf, eps),
            max(gAm, eps),
            max(gAf, eps),
            gm0,
            gf0,
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                lambda pf, W, N_m, N_f, A_m, A_f, m0, f0: _model_2d(pf[0], pf[1], W, N_m, N_f, A_m, A_f, m0, f0),
                (m, f),
                t,
                p0=p0,
                bounds=(
                    [0.0, 0.0, 0.0, 0.0, 0.0, lo_m, lo_f],
                    [_INF, _INF, _INF, _INF, _INF, hi_m, hi_f],
                ),
                maxfev=40000,
            )
        W, N_m, N_f, A_m, A_f, m0, f0 = (float(v) for v in popt)
        if gAm < 0 and A_m <= 1e-6 * max(abs(W), 1e-9):
            notes.append("unconstrained fit wanted A_malloc<0; A_malloc constrained to 0 so f_malloc is floored at 0")
        if gAf < 0 and A_f <= 1e-6 * max(abs(W), 1e-9):
            notes.append("unconstrained fit wanted A_free<0; A_free constrained to 0 so f_free is floored at 0")
        if m0 >= hi_m * 0.999:
            notes.append(
                "m0 reached the search cap: the native malloc cost does not clearly fade, so "
                "A_malloc and W (hence f_malloc) are weakly constrained"
            )
        if f0 >= hi_f * 0.999:
            notes.append(
                "f0 reached the search cap: the native free cost does not clearly fade, so "
                "A_free and W (hence f_free) are weakly constrained"
            )
        return finish(W, N_m, N_f, A_m, A_f, m0, f0, pcov, list(popt))
    except (RuntimeError, ValueError) as err:
        # curve_fit failed to converge; report the robust linear solution.
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); using the robust linear solution")
        return finish(
            gW,
            gNm,
            gNf,
            max(0.0, gAm),
            max(0.0, gAf),
            gm0,
            gf0,
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
            if m.nunique() >= 2 and f.nunique() >= 2:
                res = fit_allocation_fraction_2d(m, f, grp["runtime in s"])
                row.update(
                    {
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
                    }
                )
                row["note"] = "; ".join(res["warnings"])
                row["cov"] = _fit_cov(res)
            elif m.nunique() >= 2 or f.nunique() >= 2:
                varying = "malloc" if m.nunique() >= 2 else "free"
                delays = m if varying == "malloc" else f
                res = fit_allocation_fraction(delays, grp["runtime in s"], c_a=c_a)
                row.update(
                    {
                        "model": f"1d-{varying}",
                        "W": res["W"],
                        "T0": res["T0"],
                        "r2": res["r2"],
                    }
                )
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
                base_notes = list(res["warnings"])
                base_notes.append(f"only the {varying} delay varies; fitted the 1-D model on it")
                row["note"] = "; ".join(base_notes)
                row["cov"] = _fit_cov(res)
            else:
                row["note"] = "fewer than 2 distinct delays in each operation"
        except ValueError as err:
            row.update(
                {
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
                    "note": str(err),
                }
            )
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


def main(clusters: dict | None = None) -> None:
    """Parse, fit and plot every cluster's delay sweeps."""
    per_cluster = []
    for log_dir, title in (clusters or CLUSTERS).values():
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
    if per_cluster:
        # One figure per cluster: the malloc sweep (left) and the free sweep
        # (right) next to each other.
        _ = simple_plot(per_cluster)
        plt.show()


if __name__ == "__main__":
    main()
