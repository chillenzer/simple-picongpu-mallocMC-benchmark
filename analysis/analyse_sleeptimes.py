"""Analyze PIConGPU/mallocMC allocation-latency benchmark logs.

Reads the raw `run_all.sh` logs directly (no pre-filtering; both the old
per-variant layout and the current one), plots the runtime against the
malloc delay per example, grid and free delay (median with IQR error bars)
in one figure with one axis per cluster (see `CLUSTERS`), titled by the
hardware the runs were made on, and fits each (example, grid) sweep to the
constrained Amdahl allocation model:

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

import warnings
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.optimize import OptimizeWarning, curve_fit

ALGORITHM = "FlatterScatter"
CD_CMD = "+ cd "
RUN_CMD = "bin/picongpu "
MALLOC_DELAY_CMD = "MALLOCMC_MALLOC_DELAY="
FREE_DELAY_CMD = "MALLOCMC_FREE_DELAY="
# Pre-rename logs: the malloc delay was then injected via MALLOCMC_SLEEP_TIME.
LEGACY_MALLOC_DELAY_CMD = "MALLOCMC_SLEEP_TIME="
# `cd` trace lines that carry run context: the old per-variant layout
# (`cd build/<Example>/<Algorithm>-sleep<N>`) and the current one-build-per
# example layout (`cd .../build/<Example>`); every other `cd` (for example
# the `cd $WD` return in run_folder.sh) is ignored.
VARIANT_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)-sleep(\d+)$")
BUILD_CD_RE = re.compile(r"(?:^|/)build/(\w+)$")
# cluster name -> (run-log directory, hardware title for the plot)
CLUSTERS = {
    "hal": (Path("output") / "hal-sleeptimes", "HAL (NVIDIA A30)"),
    "rosi": (Path("output") / "rosi-sleeptimes", "RoSI (NVIDIA V100)"),
}
# The statistics, plot and fit are computed only for runs with this
# configuration: "run-time" (delays injected via MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY) or "compile-time" (per-variant builds); None uses all
# of them. The parsed results are always complete.
CONFIGURATION = None

GROUP_KEYS = ("setup", "x", "y", "z")
# one distinct marker per (setup, grid) series, shared by all axes
MARKERS = ("o", "s", "^", "D", "v", "P", "h", "X", "8")
_INF = float("inf")


def parse_setup(line: str):
    # Run context of a `cd` trace line, or None if the line is unrelated.
    path = line.split()[-1]
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
    m = BUILD_CD_RE.search(path)
    if m:
        return {"setup": m[1], "algorithm": ALGORITHM}
    return None


def parse_grid(line: str):
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            line.split(RUN_CMD, 1)[1].split("-g", 1)[1].split("-")[0].strip().split(" "),
        )
    }


def parse_simulation_time(line: str):
    return {"runtime in s": float(line.split("=")[1][: -len("sec")])}


def parse_log(log_path: Path):
    with log_path.open("r") as file:
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


def run_to_df(run: dict):
    return pd.DataFrame(run["runs"]).assign(name=run["name"])


def runs_to_df(runs: Iterable[dict]):
    tmp = pd.concat(map(run_to_df, runs))
    return tmp.assign(z=tmp.get("z", np.nan))


def parse_logs(log_paths: Iterable[Path]):
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)


def simple_statistics(full_results: pd.DataFrame):
    return full_results.groupby(list(set(full_results.columns) - {"runtime in s", "name"}), dropna=False).apply(
        lambda df: df["runtime in s"].describe(), include_groups=False
    )


def label(info):
    # Plot group key (setup, x, y, z, free_sleeptime).
    grid_string = "x".join(map(str, map(int, np.asarray(info[1:4])[~np.isnan(info[1:4])])))
    suffix = "" if info[4] == 0 else f", free: {int(info[4])} ns"
    return f"{info[0]} {grid_string}{suffix}"


def _group_key(name):
    # NaN group values (missing z in 2D runs) do not compare equal, so
    # canonicalize them; the key is only used for dictionary lookups.
    return tuple(None if isinstance(k, float) and np.isnan(k) else k for k in name)


def simple_plot(cluster_results):
    """One axis per cluster, next to each other in a single figure.

    `cluster_results` is a list of `(title, simple_results, fits)` entries,
    one per cluster; `title` is the hardware the runs were made on. Each
    (setup, grid) series gets a distinct marker, consistently on all axes;
    the legend is only shown on the right-most axis.
    """
    fig, axes = plt.subplots(1, len(cluster_results), figsize=(6.5 * len(cluster_results), 5.5))
    if len(cluster_results) == 1:
        axes = [axes]
    # Assign each (setup, grid) series its marker once, in plot order, so the
    # same series is drawn with the same marker on every axis.
    series_markers = {}
    for _, simple_results, _ in cluster_results:
        for name, _ in simple_results.groupby(list(GROUP_KEYS + ("free_sleeptime",)), dropna=False):
            key = _group_key(name[:4])
            if key not in series_markers:
                series_markers[key] = MARKERS[len(series_markers) % len(MARKERS)]
    if len(series_markers) > len(MARKERS):
        warnings.warn(f"more than {len(MARKERS)} (setup, grid) series; the markers repeat")
    for i, (ax, (title, simple_results, fits)) in enumerate(zip(axes, cluster_results)):
        _plot_cluster(ax, simple_results, fits, title, series_markers, show_legend=(i == len(axes) - 1))
    fig.tight_layout()
    return fig


def _plot_cluster(
    ax,
    simple_results: pd.DataFrame,
    fits: pd.DataFrame | None,
    title: str,
    series_markers: dict,
    show_legend: bool,
):
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
                fits_by_key[key] = (
                    "2d",
                    float(row["W"]),
                    float(row["N_malloc"]),
                    float(row["N_free"]),
                    float(row["A_malloc"]),
                    float(row["A_free"]),
                    float(row["m0_ns"]) * 1e-9,
                    float(row["f0_ns"]) * 1e-9,
                )
            elif row["model"] == "1d-malloc" and all(pd.notna(row[k]) for k in ("W", "N_malloc", "A_malloc", "m0_ns")):
                fits_by_key[key] = (
                    "1d",
                    float(row["W"]),
                    float(row["N_malloc"]),
                    float(row["A_malloc"]),
                    float(row["m0_ns"]) * 1e-9,
                )
    # One curve per (grid, free delay): the x-axis is the malloc delay.
    plot_keys = GROUP_KEYS + ("free_sleeptime",)
    results = simple_results.groupby(list(plot_keys), dropna=False)
    gaps = []
    for name, result in results:
        x, ye_min, y, ye_max = np.sort(
            result.reset_index(drop=False)[["malloc_sleeptime", "25%", "50%", "75%"]].to_numpy().T
        )
        eb = ax.errorbar(
            x,
            y,
            yerr=(y - ye_min, ye_max - y),
            linestyle="none",
            marker=series_markers[_group_key(name[:4])],
            label=label(name),
        )
        color = eb.lines[0].get_color()
        params = fits_by_key.get(_group_key(name[:4]))
        if params is not None:
            free_s = float(name[4]) * 1e-9
            # Draw the fitted model over the malloc delays this curve covers.
            m_data = x[x > 0]
            if len(m_data) > 1 and float(m_data[-1]) > float(m_data[0]):
                m_ns = np.geomspace(float(m_data[0]), float(m_data[-1]), 100)
                m_s = m_ns * 1e-9
                if params[0] == "2d":
                    _, W, Nm, Nf, Am, Af, m0, f0 = params
                    curve = _model_2d(m_s, np.full_like(m_s, free_s), W, Nm, Nf, Am, Af, m0, f0)
                    linear = W + Nm * m_s + Nf * free_s
                    a_val = Am + Af
                else:
                    _, W, N, A, s0 = params
                    curve = _model(m_s, W, N, A, s0)
                    linear = W + N * m_s
                    a_val = A
                # The model takes second-based delays; the axis is in ns.
                ax.plot(m_ns, curve, color=color, linestyle="-", alpha=0.8)
                ax.plot(m_ns, linear, color=color, linestyle="--", alpha=0.8)
                # The solid/dashed gap is the native cost A that the dashed line
                # (the extrapolation to A = 0) drops; mark it at the smallest
                # malloc delay, in the series color.
                if float(curve[0]) > float(linear[0]):
                    x0 = float(m_ns[0])
                    y_lo, y_hi = float(linear[0]), float(curve[0])
                    ax.plot((x0, x0), (y_lo, y_hi), color=color, linewidth=1, alpha=0.8)
                    f_val = a_val / (W + a_val) if W + a_val > 1e-12 else 0.0
                    gaps.append((0.5 * (y_lo + y_hi), x0, color, f"A = {a_val:.2f} s, f = {100 * f_val:.1f}%"))
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        "solid: full fit, dashed: extrapolation to A = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )
    ax.set_xlabel("malloc sleep_time (ns)")
    ax.set_ylabel("runtime (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Keep the A/f labels out of the data: each sits in the clear, to the right
    # of the leftmost delay, and a thin leader line joins it to its gap.
    if gaps:
        lo, hi = ax.get_ylim()
        for y_mid, x0, color, text in gaps:
            mid_frac = (np.log10(y_mid) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            label_frac = (0.30, min(mid_frac + 0.02, 0.96))
            # The leader line is its own artist: a bbox-anchored arrow breaks
            # matplotlib's tight-layout path clipping.
            arrow_start = ax.transData.inverted().transform(
                ax.transAxes.transform((label_frac[0] - 0.005, label_frac[1]))
            )
            ax.annotate(
                "",
                xy=(x0, y_mid),
                xytext=arrow_start,
                arrowprops=dict(arrowstyle="->", color=color, linewidth=0.8, alpha=0.8),
            )
            ax.annotate(
                text,
                xy=label_frac,
                xytext=label_frac,
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.85, linewidth=0.5),
            )
    if show_legend:
        ax.legend()
    return ax


def _model(s, W, N, A, s0):
    return W + N * s + A * s0 / (s + s0)


def _model_2d(m, f, W, N_m, N_f, A_m, A_f, m0, f0):
    return W + N_m * m + N_f * f + A_m * m0 / (m + m0) + A_f * f0 / (f + f0)


def _model_c_a(s, W, N, s0, c):
    A = N * c
    return W + N * s + A * s0 / (s + s0)


def _fit_lsq(h, s, t):
    """Least squares for t = W + N*s + A*h; returns (W, N, A, ss_res)."""
    sol, *_ = np.linalg.lstsq(np.vstack([np.ones_like(s), s, h]).T, t, rcond=None)
    res = t - (sol[0] + sol[1] * s + sol[2] * h)
    return (*[float(v) for v in sol], float(np.sum(res**2)))


def _grid_guess(s, t, lo, hi):
    """Robust unconstrained solution: linear in (W, N, A) for each s0 on a
    log grid. Returns (W, N, A, s0)."""
    hi_g = max(float(hi), lo * 1.5)
    best = None
    for s0 in np.logspace(np.log10(lo), np.log10(hi_g), 60):
        W, N, A, ss_res = _fit_lsq(s0 / (s + s0), s, t)
        if best is None or ss_res < best[0]:
            best = (ss_res, s0, W, N, A)
    _, s0, W, N, A = best
    return W, N, A, float(s0)


def _uncertainties(p, pcov, f_of_p):
    """Standard errors for the fitted parameters p and for f = f_of_p(p),
    from the parameter covariance. NaNs if the covariance is unavailable."""
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


def fit_allocation_fraction(sleeptimes, runtimes, c_a: float | None = None) -> dict:
    """Fit one sleeptime sweep to the constrained Amdahl model.

    Returns a dict with W, N, A, s0, T0, f, r2 plus their standard errors
    (W_err, N_err, A_err, s0_err, f_err, NaN when unavailable), warnings, and
    the data with per-point residuals.
    """
    s = np.asarray(sleeptimes, dtype=float) * 1e-9  # ns -> s
    t = np.asarray(runtimes, dtype=float)
    if s.shape != t.shape:
        raise ValueError("sleeptimes and runtimes must have the same shape")
    if s.size < 3:
        raise ValueError("need at least 3 data points")
    order = np.argsort(s)
    s, t = s[order], t[order]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    notes = []

    def finish(W, N, A, s0, pcov, fit_params, f_of_p, note=None):
        pred = _model(s, W, N, A, s0)
        r2 = 1.0 - float(np.sum((t - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        if N <= 0:
            notes.append("non-positive slope: no allocation cost visible in this sweep")
        if note:
            notes.append(note)
        W, N, A = float(W), float(N), float(A)
        s0 = float("nan") if s0 != s0 else float(s0)
        T0 = W + A
        f = A / T0 if T0 > 1e-12 else 0.0
        if fit_params is None:
            W_e = N_e = A_e = s0_e = f_e = float("nan")
        elif c_a is None:
            W_e, N_e, A_e, s0_e, f_e = _uncertainties(fit_params, pcov, f_of_p)
        else:
            c = c_a * 1e-9
            W_e, N_e, s0_e, f_e = _uncertainties(fit_params, pcov, f_of_p)
            A_e = float("nan") if N_e != N_e else N_e * c
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
        }

    f_free = lambda p: p[2] / (p[0] + p[2]) if p[0] + p[2] > 1e-12 else 0.0
    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = max(0.05 * s_min_pos, 1e-12)
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

    lo_b = max(lo, 1e-12)
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
            if gA < -1e-6 * max(abs(gW), 1e-9) and A <= 1e-6 * max(abs(W), 1e-9):
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
        f_ca = lambda p: (p[1] * c / (p[0] + p[1] * c) if p[0] + p[1] * c > 1e-12 else 0.0)
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
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); " "using the robust linear solution")
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


def fit_allocation_fraction_2d(m_delays, f_delays, runtimes) -> dict:
    """Fit one (malloc, free) delay combination sweep to the constrained
    two-operation Amdahl model
        T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0).

    Returns a dict with W, N_m, N_f, A_m, A_f, m0, f0, T0, f_malloc, f_free,
    r2 plus their standard errors (NaN when unavailable), warnings, and the
    data with per-point residuals.
    """
    m = np.asarray(m_delays, dtype=float) * 1e-9  # ns -> s
    f = np.asarray(f_delays, dtype=float) * 1e-9
    t = np.asarray(runtimes, dtype=float)
    if m.shape != f.shape or m.shape != t.shape:
        raise ValueError("delays and runtimes must all have the same shape")
    if m.size < 3:
        raise ValueError("need at least 3 data points")
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    notes = []

    def finish(W, N_m, N_f, A_m, A_f, m0, f0, pcov, fit_params, note=None):
        pred = _model_2d(m, f, W, N_m, N_f, A_m, A_f, m0, f0)
        r2 = 1.0 - float(np.sum((t - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        if N_m <= 0:
            notes.append("non-positive malloc slope: no allocation cost visible for the malloc delays")
        if N_f <= 0:
            notes.append("non-positive free slope: no free cost visible for the free delays")
        if note:
            notes.append(note)
        T0 = W + A_m + A_f
        f_malloc = A_m / T0 if T0 > 1e-12 else 0.0
        f_free = A_f / T0 if T0 > 1e-12 else 0.0
        if fit_params is None:
            W_e = Nm_e = Nf_e = Am_e = Af_e = m0_e = f0_e = f_malloc_e = f_free_e = float("nan")
        else:
            f_of_m = lambda p: (p[3] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > 1e-12 else 0.0)
            f_of_f = lambda p: (p[4] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > 1e-12 else 0.0)
            W_e, Nm_e, Nf_e, Am_e, Af_e, m0_e, f0_e, f_malloc_e = _uncertainties(fit_params, pcov, f_of_m)
            f_free_e = _uncertainties(fit_params, pcov, f_of_f)[-1]
        return {
            "W": float(W),  # runtime without any allocation or free cost (s)
            "N_m": float(N_m),  # allocation calls per run
            "N_f": float(N_f),  # free calls per run
            "A_m": float(A_m),  # native allocation time (s)
            "A_f": float(A_f),  # native free time (s)
            "m0": float("nan") if m0 != m0 else float(m0),  # malloc fade scale (s)
            "f0": float("nan") if f0 != f0 else float(f0),  # free fade scale (s)
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
        }

    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, 1e-12)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, 1e-12)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)

    def grid_guess():
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
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); " "using the robust linear solution")
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
            note="constrained fit unavailable; linear grid solution reported " "(A_malloc/A_free floored at 0)",
        )


def _to_ns(value: float):
    value = float(value)
    return float("nan") if value != value else value * 1e9


def fit_sweep(df: pd.DataFrame, c_a: float | None = None, configuration: str | None = None) -> pd.DataFrame:
    """Fit every (setup, x, y, z) group of a parsed sweep DataFrame.

    Groups whose runs span both the malloc and the free delay are fitted with
    the two-operation model of `fit_allocation_fraction_2d`; groups spanning
    only one delay fall back to the 1-D model of `fit_allocation_fraction`
    on that delay. `df` is the output of `parse_logs`; `configuration`, if
    given, restricts the fit to that configuration.
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
    }
    rows = []
    for key, grp in df.groupby(list(GROUP_KEYS), dropna=False):
        grp = grp.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime in s"])
        row = {**dict(zip(GROUP_KEYS, key)), "n_runs": len(grp), **no_fit}
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


def _print_fraction_summary(fits: pd.DataFrame):
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
            print(
                f"  {r.setup:<16s} grid {int(r.x)}x{int(r.y)}"
                + (f"x{int(r.z)}" if pd.notna(r.z) else "")
                + f" [{r.model}] : "
                + ", ".join(fractions)
                + f"   (W = {r.W:.2f} s"
                + (", " + ", ".join(extra) if extra else "")
                + ")"
            )
            if r["note"]:
                print(f"      note: {r.note}")


def main(clusters: dict | None = None):
    per_cluster = []
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
            print(fits.to_string(index=False, float_format=lambda v: f"{v:10.3g}"))
        _print_fraction_summary(fits)
        per_cluster.append((title, simple_results, fits))
    if per_cluster:
        _ = simple_plot(per_cluster)
        plt.show()


if __name__ == "__main__":
    main()
