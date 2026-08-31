"""Re-analysis of the allocation-model fits in output/results.h5.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Read-only: reads the results file (and, implicitly, the run data stored in
its `runs` table) and re-fits every (machine, setup, algorithm, grid) sweep
with nested models to test whether the saturation terms
A_m*m0/(m+m0) + A_f*f0/(f+f0) are supported by the data.

Prints six blocks:
  1. per group: reproduction of the stored 7-parameter fit (same pipeline),
     F-tests / BIC of the full 7-param model vs the 3-param linear null and
     vs the 5-param models with one saturation term removed, the
     A_m/A_f flat-direction profiles (delta-SSR with A fixed), the fitted
     T0 vs the measured (0,0) baseline, and pcov condition numbers;
  2. multi-modality of the flagged group: SSR/r2 of the stored fit, of a
     fresh refit from the same documented grid initial guess, and of the
     linear null, on the identical data;
  3. sensitivity of the headline fractions to the fade-scale seed;
  4. the baseline-anchored view: E(s) = T(s) - T(0) - N*s per arm, the
     plateau deficit d = T(0) - (large-delay line intercept) = the total
     delay absorbed by parallel work, and d/N = the absorbable slack per
     call. This is the test of the "the delay is hidden because allocation
     is not the bottleneck" hypothesis;
  5. the model-consistent decomposition: on the malloc arm
     T(m,0) -> W + A_free + N_malloc*m as m -> inf, so the data pin
     A_malloc = d_m, A_free = d_f, W = T(0) - d_m - d_f directly; this
     block compares the stored fit to those data-pinned values and prints
     the stored model's max residual against the raw data;
  6. the gauge symmetry of the model: the exact per-arm delay-origin shift
     (W -> W + N*m0*d, A -> A/(1+d), m0 -> m0*(1+d), m -> m - m0*d), whose
     invariants are N, A*m0 and W - N*m0. The reported fractions A/T0 are
     not invariants, so this block quantifies how far f_malloc / f_free move
     along the gauge orbit over the fade-scale search window, and how much of
     each gauge direction is a flat direction of the fit (its share in the
     two flattest pcov directions).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import f as fdist

RESULTS = "output/results.h5"
EPS_S = 1e-12

pd.set_option("display.width", 260)


def read_table(f: h5py.File, grp: str) -> pd.DataFrame:
    """Read one table group of the results file into a DataFrame.

    Args:
        f: the open results file.
        grp: the table group name.

    Returns:
        pd.DataFrame: the table in its stored column order.

    """
    g = f[grp]
    keys = [k for k in g if isinstance(g[k], h5py.Dataset)]
    order = g.attrs.get("column_order")
    if isinstance(order, bytes):
        order = order.decode()
    if isinstance(order, str):
        order = order.split(",")
    if order:
        keys = [k for k in order if k in g] + [k for k in keys if k not in order]
    return pd.DataFrame({k: (g[k][()].tolist() if g[k].dtype.kind in "OUS" else g[k][()]) for k in keys})


def model_full(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Evaluate the full two-operation allocation model (7 parameters).

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_malloc, N_free, A_malloc, A_free, m0, f0).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, Nm, Nf, Am, Af, m0, f0 = p
    return W + Nm * m + Nf * f + Am * m0 / (m + m0) + Af * f0 / (f + f0)


def model_nom(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Evaluate the full model without the malloc saturation term.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_malloc, N_free, A_free, f0).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, Nm, Nf, Af, f0 = p
    return W + Nm * m + Nf * f + Af * f0 / (f + f0)


def model_nofree(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Evaluate the full model without the free saturation term.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_malloc, N_free, A_malloc, m0).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, Nm, Nf, Am, m0 = p
    return W + Nm * m + Nf * f + Am * m0 / (m + m0)


def ssr_of(
    fn: Callable[[np.ndarray, np.ndarray, Sequence[float]], np.ndarray],
    m: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    p: Sequence[float],
) -> float:
    """Return the sum of squared residuals of one packed-parameter model.

    Args:
        fn: a model of (m, f, p), p the packed parameter vector.
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        p: the packed parameter vector.

    Returns:
        float: the sum of squared residuals.

    """
    return float(np.sum((t - fn(m, f, p)) ** 2))


def _fit(
    fn: Callable[[tuple, ...], np.ndarray],
    xdata: tuple,
    t: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Run one bounded curve_fit with the OptimizeWarnings suppressed.

    Args:
        fn: the model (xdata plus the individual parameters).
        xdata: the model's data (the delay arrays).
        t: the runtimes in seconds.
        p0: the initial parameter vector.
        bounds: the (lower, upper) parameter bounds.

    Returns:
        tuple: (the fitted parameters, the parameter covariance).

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(fn, xdata, t, p0=p0, bounds=bounds, maxfev=40000)
    return popt, pcov


def fit_full(
    m: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    hi: tuple[float, float],
    p0: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the full 7-parameter model with the documented bounds.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        hi: the (malloc, free) fade-scale upper bounds (s).
        p0: the initial parameters, or None for the all-ones default.

    Returns:
        tuple: (the fitted parameters, the parameter covariance).

    """
    lo = [0.0, 0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S]
    hi_b = [np.inf] * 5 + [hi[0], hi[1]]
    return _fit(
        lambda x, W, Nm, Nf, Am, Af, m0, f0: model_full(x[0], x[1], (W, Nm, Nf, Am, Af, m0, f0)),
        (m, f),
        t,
        p0 if p0 is not None else list(np.ones(7)),
        (lo, hi_b),
    )


def fit_linear(m: np.ndarray, f: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the 3-parameter linear null T = W + N_malloc*m + N_free*f.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.

    Returns:
        tuple: (the least-squares parameters, the sum of squared residuals).

    """
    X = np.vstack([np.ones_like(m), m, f]).T
    sol, *_ = np.linalg.lstsq(X, t, rcond=None)
    return sol, float(np.sum((t - X @ sol) ** 2))


def _grid_cell(m: np.ndarray, f: np.ndarray, t: np.ndarray, m0: float, f0: float) -> tuple[float, np.ndarray]:
    """One (m0, f0) cell of the robust grid guess: its SSR and least-squares fit.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        m0: the malloc fade scale of the cell (s).
        f0: the free fade scale of the cell (s).

    Returns:
        tuple: (the cell's sum of squared residuals, its least-squares
        parameters).

    """
    X = np.vstack([np.ones_like(m), m, f, m0 / (m + m0), f0 / (f + f0)]).T
    sol, *_ = np.linalg.lstsq(X, t, rcond=None)
    return float(np.sum((t - X @ sol) ** 2)), sol


def grid_guess(m: np.ndarray, f: np.ndarray, t: np.ndarray, bounds: tuple[float, float, float, float]) -> list[float]:
    """Return the stored pipeline's robust initial guess (allocation_model._grid_guess_2d).

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        bounds: (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).

    Returns:
        list[float]: the 0-floored initial parameters (W, N_malloc,
        N_free, A_malloc, A_free, m0, f0).

    Raises:
        RuntimeError: if the grid search yields no cell.

    """
    lo_m, hi_m, lo_f, hi_f = bounds
    best: tuple[float, float, float, list[float]] | None = None
    for m0 in np.logspace(np.log10(lo_m), np.log10(hi_m), 25):
        for f0 in np.logspace(np.log10(lo_f), np.log10(hi_f), 25):
            ssr, sol = _grid_cell(m, f, t, float(m0), float(f0))
            if best is None or ssr < best[0]:
                best = (ssr, float(m0), float(f0), [float(v) for v in sol])
    if best is None:
        msg = "the grid search yielded no cell"
        raise RuntimeError(msg)
    return [max(v, EPS_S) for v in best[3]] + [best[1], best[2]]


def bic(n: int, k: int, ssr: float) -> float:
    """Return the BIC of one model (up to an additive constant in k).

    Args:
        n: the number of data points.
        k: the number of fitted parameters.
        ssr: the sum of squared residuals.

    Returns:
        float: the BIC.

    """
    return float(n * np.log(ssr / n) + k * np.log(n))


def bounds_of(m: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    """Return the documented fade-scale search window of one sweep.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.

    Returns:
        tuple: (lo_m, hi_m, lo_f, hi_f) in seconds.

    """
    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * float(m_min_pos), EPS_S)
    hi_m = max(0.5 * (float(m.max()) - float(m.min())), lo_m * 1.5)
    lo_f = max(0.05 * float(f_min_pos), EPS_S)
    hi_f = max(0.5 * (float(f.max()) - float(f.min())), lo_f * 1.5)
    return lo_m, hi_m, lo_f, hi_f


def profile_A(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    pin: tuple[str, float],
    hi: tuple[float, float],
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the delta-SSR profile with one A fixed at grid values.

    A_malloc (pin[0] = 'm') or A_free (pin[0] = 'f') is fixed at each grid
    value while the remaining six parameters are re-optimized in-bounds.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        pin: (which, a_hi): "m" (fix A_malloc) or "f" (fix A_free), and the
        upper end of the A grid in seconds.
        hi: the (malloc, free) fade-scale upper bounds (s).
        base: the packed full-model parameters at the base point.

    Returns:
        tuple: (the A grid, the delta-SSR profile, NaN where the fit
        failed to converge).

    """
    which, a_hi = pin
    hi_m, hi_f = hi
    m, f = sweep
    ssr_base = ssr_of(model_full, m, f, t, base)
    grid = np.linspace(0.0, a_hi, 41)
    lo = [0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S]
    hi_b = [np.inf, np.inf, np.inf, np.inf, hi_m, hi_f]
    out: list[float] = []
    for a in grid:
        if which == "m":
            p0 = [base[0], base[1], base[2], base[4], base[5], base[6]]
            try:
                popt, _ = _fit(
                    lambda x, W, Nm, Nf, Af, m0, f0: model_full(x[0], x[1], (W, Nm, Nf, a, Af, m0, f0)),
                    (m, f),
                    t,
                    p0,
                    (lo, hi_b),
                )
            except RuntimeError, ValueError:
                out.append(float("nan"))
                continue
            packed = (popt[0], popt[1], popt[2], a, popt[3], popt[4], popt[5])
        else:
            p0 = [base[0], base[1], base[2], base[3], base[5], base[6]]
            try:
                popt, _ = _fit(
                    lambda x, W, Nm, Nf, Am, m0, f0: model_full(x[0], x[1], (W, Nm, Nf, Am, a, m0, f0)),
                    (m, f),
                    t,
                    p0,
                    (lo, hi_b),
                )
            except RuntimeError, ValueError:
                out.append(float("nan"))
                continue
            packed = (popt[0], popt[1], popt[2], popt[3], a, popt[4], popt[5])
        out.append(ssr_of(model_full, m, f, t, packed) - ssr_base)
    return grid, np.asarray(out)


def seed_sensitivity(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> list[tuple[str, tuple[float, float, float, float, float, float] | None]]:
    """Full 2-D fit with the fade-scale seeds varied over the search window.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).

    Returns:
        list: per seed ("grid", "bottom", "mid"), (A_malloc, A_free,
        f_malloc, f_free, m0, f0) of the converged fit, or None when the
        fit did not converge.

    """
    lo_m, hi_m, lo_f, hi_f = bounds
    out: list[tuple[str, tuple[float, float, float, float, float, float] | None]] = []
    guess = grid_guess(sweep[0], sweep[1], t, bounds)
    seeds = {
        "grid": (guess[5], guess[6]),
        "bottom": (lo_m, lo_f),
        "mid": (float(np.sqrt(lo_m * hi_m)), float(np.sqrt(lo_f * hi_f))),
    }
    for name, (m0, f0) in seeds.items():
        p0 = [
            max(guess[0], EPS_S),
            max(guess[1], EPS_S),
            max(guess[2], EPS_S),
            max(guess[3], EPS_S),
            max(guess[4], EPS_S),
            m0,
            f0,
        ]
        try:
            popt, _ = _fit(
                lambda x, W, Nm, Nf, Am, Af, m0_, f0_: model_full(x[0], x[1], (W, Nm, Nf, Am, Af, m0_, f0_)),
                sweep,
                t,
                p0,
                ([0.0] * 5 + [EPS_S, EPS_S], [np.inf] * 5 + [hi_m, hi_f]),
            )
        except RuntimeError, ValueError:
            out.append((name, None))
            continue
        W = float(popt[0])
        Am = float(popt[3])
        Af = float(popt[4])
        T0 = W + Am + Af
        vals = (Am, Af, Am / T0 if T0 > EPS_S else 0.0, Af / T0 if T0 > EPS_S else 0.0, float(popt[5]), float(popt[6]))
        out.append((name, vals))
    return out


def _id_res(sweep: tuple[np.ndarray, np.ndarray], p: Sequence[float]) -> float:
    """Max identity residual of the gauge shifts at |d|, |e| up to 0.5 (s).

    Args:
        sweep: the (m, f) delay arrays in seconds.
        p: the packed full-model parameters.

    Returns:
        float: the max absolute deviation from the exact gauge identity.

    """
    m, fv = sweep
    W, Nm, Nf, Am, Af, m0, f0 = p
    base = model_full(m, fv, p)
    worst = 0.0
    for d in (-0.5, -0.2, 0.05, 0.2, 0.5):
        r = model_full(m - m0 * d, fv, (W + Nm * m0 * d, Nm, Nf, Am / (1 + d), Af, m0 * (1 + d), f0))
        worst = max(worst, float(np.max(np.abs(r - base))))
    for e in (-0.5, -0.2, 0.05, 0.2, 0.5):
        r = model_full(m, fv - f0 * e, (W + Nf * f0 * e, Nm, Nf, Am, Af / (1 + e), m0, f0 * (1 + e)))
        worst = max(worst, float(np.max(np.abs(r - base))))
    return worst


def _orbit_range(p: Sequence[float], which: str, bounds: tuple[float, float]) -> tuple[float, float]:
    """Range of A/T0 along one arm's gauge orbit over a fade-scale window.

    Args:
        p: the packed full-model parameters.
        which: "m" (the malloc arm) or "f" (the free arm).
        bounds: the (lo, hi) fade-scale window of the arm (s).

    Returns:
        tuple: (the min, the max) of the arm's A/T0 fraction on the orbit
        (includes the point d = 0).

    """
    W, Nm, Nf, Am, Af, m0, f0 = p
    N, A, s0 = (Nm, Am, m0) if which == "m" else (Nf, Af, f0)
    a_other = Af if which == "m" else Am
    lo, hi = bounds

    def ratio(d: float) -> float:
        a = A / (1 + d)
        t0 = W + N * s0 * d + a + a_other
        return a / t0 if t0 > EPS_S else 0.0

    g = np.unique(np.concatenate([np.linspace(lo / s0 - 1.0, hi / s0 - 1.0, 121), [0.0]]))
    vals = np.array([ratio(d) for d in g])
    return float(vals.min()), float(vals.max())


def _flat_share(pcov: np.ndarray | None, g_m: np.ndarray, g_f: np.ndarray) -> tuple[float, float]:
    """Share of each gauge direction in the two flattest pcov directions.

    Args:
        pcov: the parameter covariance of the stored fit (or None).
        g_m: the malloc gauge direction vector.
        g_f: the free gauge direction vector.

    Returns:
        tuple: (the malloc, the free) share (1 = a flat direction of the
        fit, i.e. the data do not pin the gauge; NaN when pcov is
        unavailable).

    """
    if pcov is None or not np.all(np.isfinite(pcov)):
        return float("nan"), float("nan")
    cov = np.asarray(pcov, dtype=float)
    _eig, evec = np.linalg.eigh(0.5 * (cov + cov.T))
    v1, v2 = evec[:, 0], evec[:, 1]

    def frac(g: np.ndarray) -> float:
        gn = g / np.linalg.norm(g)
        return float(np.hypot(np.dot(v1, gn), np.dot(v2, gn)))

    return frac(g_m), frac(g_f)


def gauge_block(
    m: np.ndarray,
    fv: np.ndarray,
    stored: np.ndarray,
    bounds: tuple[float, float, float, float],
    pcov: np.ndarray | None,
) -> dict[str, float | tuple[float, float]]:
    """Gauge-symmetry diagnostic for one stored 7-parameter fit.

    The model is exactly invariant under the per-arm delay-origin shifts
        malloc: W -> W + N_m*m0*d, A_m -> A_m/(1+d), m0 -> m0*(1+d), m -> m - m0*d
        free:   W -> W + N_f*f0*e, A_f -> A_f/(1+e), f0 -> f0*(1+e), f -> f - f0*e
    with invariants (N_m, N_f, A_m*m0, A_f*f0, W - N_m*m0 - N_f*f0). The
    reported fractions f_m = A_m/T0 and f_f = A_f/T0 are not invariants, so
    this quantifies how far they move along the gauge orbit over the fade-scale
    search window, plus the share of each gauge direction in the two flattest
    directions of `pcov` (1 = a flat direction of the fit, i.e. the data do
    not pin the gauge).

    Args:
        m: the malloc delays in seconds.
        fv: the free delays in seconds.
        stored: the stored 7-parameter fit.
        bounds: the (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).
        pcov: the stored fit's parameter covariance (or None).

    Returns:
        dict: id_res (max identity residual, s), fm0, ff0 (the stored
        fractions), fm_rng, ff_rng (fraction ranges on the gauge orbit),
        and flat (the gauge-direction shares in the two flattest pcov
        directions).

    """
    W = float(stored[0])
    Am = float(stored[3])
    Af = float(stored[4])
    p = tuple(float(v) for v in stored)
    worst = _id_res((m, fv), p)
    T0 = W + Am + Af
    fm0 = Am / T0 if T0 > EPS_S else 0.0
    ff0 = Af / T0 if T0 > EPS_S else 0.0
    fm_rng = _orbit_range(p, "m", (bounds[0], bounds[1]))
    ff_rng = _orbit_range(p, "f", (bounds[2], bounds[3]))
    g_m = np.array([p[1] * p[5], 0.0, 0.0, -Am, 0.0, p[5], 0.0])
    g_f = np.array([p[2] * p[6], 0.0, 0.0, 0.0, -Af, 0.0, p[6]])
    flat = _flat_share(pcov, g_m, g_f)
    return {"id_res": worst, "fm0": fm0, "ff0": ff0, "fm_rng": fm_rng, "ff_rng": ff_rng, "flat": flat}


def arm_offset(m: np.ndarray, f: np.ndarray, t: np.ndarray) -> dict[str, tuple[float, float] | None]:
    """Return the per-arm offsets above the two-largest-delay line.

    Per arm (per-delay medians): the offset above the line through the two
    largest-delay points, at the smallest delay and at mid-range (s).

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.

    Returns:
        dict: per arm ("malloc", "free"), the (smallest-delay, mid-range)
        offsets in seconds, or None when the arm has fewer than 4 distinct
        delays.

    """
    out = {}
    for tag, arm, xcol in (
        ("malloc", (m > 0) & (f == 0), m),
        ("free", (m == 0) & (f > 0), f),
    ):
        x = xcol[arm]
        y = t[arm]
        if len(np.unique(x)) < 4:
            out[tag] = None
            continue
        ux = np.unique(x)
        ymed = np.array([np.median(y[x == v]) for v in ux])
        N = (ymed[-1] - ymed[-2]) / (ux[-1] - ux[-2])
        b0 = ymed[-1] - N * ux[-1]
        i_mid = len(ux) // 2
        out[tag] = (
            float(ymed[0] - (b0 + N * ux[0])),
            float(ymed[i_mid] - (b0 + N * ux[i_mid])),
        )
    return out


def arm_anchored(m: np.ndarray, f: np.ndarray, t: np.ndarray, t00: float) -> dict[str, dict[str, float] | None]:
    """Return the per-arm diagnostics anchored at the zero-delay runtime.

    Anchored at the measured zero-delay runtime t00: the excess E(s) =
    T(s) - t00 - N*s over the nominal line through the baseline, the
    plateau deficit d = t00 - (line intercept), and the local slope between
    the two smallest delays relative to N (the large-delay slope). Hiding of
    the small delay shows up as E < 0 and slope_small < N.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        t00: the measured zero-delay runtime median (s).

    Returns:
        dict: per arm ("malloc", "free"), d, E0, Emid, Emax, s_sm_over_N,
        N, d_per_call_us, or None when the arm has fewer than 4 distinct
        delays.

    """
    out = {}
    for tag, arm, xcol in (
        ("malloc", (m > 0) & (f == 0), m),
        ("free", (m == 0) & (f > 0), f),
    ):
        x = xcol[arm]
        y = t[arm]
        if len(np.unique(x)) < 4:
            out[tag] = None
            continue
        ux = np.unique(x)
        ymed = np.array([np.median(y[x == v]) for v in ux])
        N = (ymed[-1] - ymed[-2]) / (ux[-1] - ux[-2])
        b0 = ymed[-1] - N * ux[-1]
        d = t00 - b0  # baseline minus large-delay line intercept (s)
        E = ymed - t00 - N * ux  # excess above the nominal line from the baseline
        slope_sm = (ymed[1] - ymed[0]) / (ux[1] - ux[0])
        out[tag] = {
            "d": float(d),
            "E0": float(E[0]),
            "Emid": float(E[len(ux) // 2]),
            "Emax": float(E[-1]),
            "s_sm_over_N": float(slope_sm / N),
            "N": float(N),
            "d_per_call_us": float(d / N * 1e6) if N > 0 else float("nan"),
        }
    return out


def arm_profile(m: np.ndarray, f: np.ndarray, t: np.ndarray) -> dict[str, tuple[np.ndarray | None, np.ndarray | None]]:
    """Return the full per-arm offset profiles above the large-delay line.

    Per arm (per-delay medians): the offset profile above the line through
    the two largest-delay points.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.

    Returns:
        dict: per arm ("malloc", "free"), (the offset profile, the distinct
        delays), or (None, None) when the arm has fewer than 4 distinct
        delays.

    """
    out = {}
    for tag, arm, xcol in (
        ("malloc", (m > 0) & (f == 0), m),
        ("free", (m == 0) & (f > 0), f),
    ):
        x = xcol[arm]
        y = t[arm]
        if len(np.unique(x)) < 4:
            out[tag] = (None, None)
            continue
        ux = np.unique(x)
        ymed = np.array([np.median(y[x == v]) for v in ux])
        N = (ymed[-1] - ymed[-2]) / (ux[-1] - ux[-2])
        b0 = ymed[-1] - N * ux[-1]
        out[tag] = (ymed - (b0 + N * ux), ux)
    return out


def run_spread(sub: pd.DataFrame) -> float:
    """Within-cell runtime spread of one group's runs (s).

    Args:
        sub: the group's runs.

    Returns:
        float: the standard deviation of the replicate runtimes (cells with
        at least 2 runs) around their per-cell median, NaN when fewer than 3
        replicates exist.

    """
    g = sub.groupby(["malloc_sleeptime", "free_sleeptime"])["runtime_s"]
    med = g.transform("median")
    cnt = g.transform("count")
    dev = (sub["runtime_s"] - med)[cnt >= 2]
    return float(dev.std()) if dev.size >= 3 else float("nan")


def group_of(runs: pd.DataFrame, key: tuple[str, str, str, int, int, float]) -> pd.DataFrame:
    """Return the runs of one (machine, setup, algorithm, x, y, z) group.

    Args:
        runs: all runs.
        key: the group's (machine, setup, algorithm, x, y, z); a NaN z
        selects the z-less groups.

    Returns:
        pd.DataFrame: the group's runs with complete delay/runtime fields.

    """
    sub = runs[
        (runs["machine"] == key[0])
        & (runs["setup"] == key[1])
        & (runs["algorithm"] == key[2])
        & (runs["x"] == key[3])
        & (runs["y"] == key[4])
    ]
    sub = sub[sub["z"].isna()] if np.isnan(key[5]) else sub[(sub["z"] == key[5])]
    return sub.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime_s"])


class _GroupData(NamedTuple):
    """The per-run data of one group, plus its stored fit.

    `m`/`fv` are the per-run malloc/free sleeptimes (s), `t` the runtimes
    (s), `n` the run count, `n_00` the (0,0) run count, `sub` the group's
    raw runs, and `stored` the stored 7-parameter fit.
    """

    m: np.ndarray
    fv: np.ndarray
    t: np.ndarray
    n: int
    n_00: int
    sub: pd.DataFrame
    stored: np.ndarray


def _group_arrays(runs: pd.DataFrame, row: pd.Series) -> _GroupData:
    """Return the per-run data of one group.

    Args:
        runs: all runs.
        row: the group's fits row.

    Returns:
        _GroupData: the group's delays, runtimes, counts, raw runs, and
        stored fit.

    """
    key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
    sub = group_of(runs, key)
    m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
    fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
    t = sub["runtime_s"].to_numpy(float)
    stored = np.asarray(
        [
            row["W"],
            row["N_malloc"],
            row["N_free"],
            row["A_malloc"],
            row["A_free"],
            row["m0_ns"] * 1e-9,
            row["f0_ns"] * 1e-9,
        ],
        dtype=float,
    )
    return _GroupData(m, fv, t, int(t.size), int(((m == 0) & (fv == 0)).sum()), sub, stored)


def _f_test(ssr_red: float, ssr_full: float, n: int, df: int) -> tuple[float, float]:
    """F-test of the full 7-parameter model against one reduced model.

    Args:
        ssr_red: the reduced model's sum of squared residuals.
        ssr_full: the full model's sum of squared residuals.
        n: the number of data points.
        df: the number of parameters the reduced model lacks.

    Returns:
        tuple: (the F statistic, its p value).

    """
    f_stat = ((ssr_red - ssr_full) / df) / (ssr_full / (n - 7))
    return f_stat, float(fdist.sf(f_stat, df, n - 7))


def _nested_ssr(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    hi: tuple[float, float],
    p0: list[float],
) -> tuple[float, float]:
    """SSRs of the 5-parameter models with one saturation term removed.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        hi: the (malloc, free) fade-scale upper bounds (s).
        p0: the full model's initial parameters (for the seeds).

    Returns:
        tuple: (the SSR without the malloc term, the SSR without the free
        term).

    """
    m, f = sweep
    p_nom, _ = _fit(
        lambda x, W, Nm, Nf, Af, f0: model_nom(x[0], x[1], (W, Nm, Nf, Af, f0)),
        sweep,
        t,
        [p0[0], p0[1], p0[2], p0[4], p0[6]],
        ([0.0, 0.0, 0.0, 0.0, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi[1]]),
    )
    ssr_nom = ssr_of(model_nom, m, f, t, p_nom)
    p_nof, _ = _fit(
        lambda x, W, Nm, Nf, Am, m0: model_nofree(x[0], x[1], (W, Nm, Nf, Am, m0)),
        sweep,
        t,
        [p0[0], p0[1], p0[2], p0[3], p0[5]],
        ([0.0, 0.0, 0.0, 0.0, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi[0]]),
    )
    ssr_nof = ssr_of(model_nofree, m, f, t, p_nof)
    return ssr_nom, ssr_nof


def _model_tests(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    stored: np.ndarray,
    bounds: tuple[float, float, float, float],
    p0: list[float],
) -> dict:
    """Refit one group and test the stored fit against nested models.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        stored: the stored 7-parameter fit.
        bounds: the (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).
        p0: the documented grid-guess initial parameters.

    Returns:
        dict: the fresh fit (p_full, pcov), the SSRs of the full, stored,
        linear, malloc-less, and free-less models, the total sum of
        squares, the F-test / BIC / condition-number diagnostics, and the
        max reproduction distance dmax.

    """
    m, fv = sweep
    hi = (bounds[1], bounds[3])
    p_full, pcov = fit_full(m, fv, t, hi, p0)
    n = int(t.size)
    ssr_full = ssr_of(model_full, m, fv, t, p_full)
    ssr_stored = ssr_of(model_full, m, fv, t, stored)
    _, ssr_lin = fit_linear(m, fv, t)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    ssr_nom, ssr_nof = _nested_ssr(sweep, t, hi, p0)
    res: dict = {
        "n": n,
        "p_full": p_full,
        "pcov": pcov,
        "ssr_full": ssr_full,
        "ssr_stored": ssr_stored,
        "ssr_lin": ssr_lin,
        "ss_tot": ss_tot,
    }
    res["F_stored"], res["p_stored"] = _f_test(ssr_lin, ssr_stored, n, 4)
    res["F_refit"], res["p_refit"] = _f_test(ssr_lin, ssr_full, n, 4)
    res["F_nom"], res["p_nom"] = _f_test(ssr_nom, ssr_full, n, 2)
    res["F_nof"], res["p_nof"] = _f_test(ssr_nof, ssr_full, n, 2)
    res["dBIC"] = bic(n, 7, ssr_full) - bic(n, 3, ssr_lin)
    res["cond"] = float(np.linalg.cond(pcov)) if np.all(np.isfinite(pcov)) else float("inf")
    res["dmax"] = float(np.max(np.abs(np.asarray(p_full) - stored)))
    return res


def _base00(sweep: tuple[np.ndarray, np.ndarray], t: np.ndarray) -> float:
    """Return the measured (0,0) baseline runtime median.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.

    Returns:
        float: the (0,0) runtime median in seconds (NaN when absent).

    """
    m, fv = sweep
    x = t[(m == 0) & (fv == 0)]
    return float(np.median(x)) if x.size else float("nan")


def _flat_profiles(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
    p_full: np.ndarray,
    spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the A values whose fixing costs no more than one spread^2 in SSR.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).
        p_full: the packed fresh-fit parameters.
        spread: the per-run spread (s).

    Returns:
        tuple: (the acceptable A_malloc values, the acceptable A_free
        values) in seconds.

    """
    hi = (bounds[1], bounds[3])
    am_hi = max(4 * max(float(p_full[3]), 1e-2), 1e-2)
    af_hi = max(4 * max(float(p_full[4]), 1e-2), 1e-2)
    gm, dm = profile_A(sweep, t, ("m", am_hi), hi, p_full)
    gf, dfp = profile_A(sweep, t, ("f", af_hi), hi, p_full)
    thr = spread**2
    am_ok = gm[dm <= thr] if np.any(np.isfinite(dm)) else np.array([])
    af_ok = gf[dfp <= thr] if np.any(np.isfinite(dfp)) else np.array([])
    return am_ok, af_ok


def _group_label(row: pd.Series) -> str:
    """Return the short display label of one group.

    Args:
        row: the group's fits row.

    Returns:
        str: the machine, the setup (3 chars for KelvinHelmholtz), the
        grid (the z size omitted for z-less groups), and the algorithm
        prefix.

    """
    setup = row["setup"][:3] if row["setup"] == "KelvinHelmholtz" else row["setup"]
    z = f"x{int(row['z'])}" if not np.isnan(row["z"]) else ""
    return f"{row['machine']} {setup} {int(row['x'])}x{int(row['y'])}{z} {row['algorithm'][:4]}"


def _pin_block(
    out_row: dict,
    fits_row: pd.Series,
    anchored: dict,
    base00: float,
    data: tuple,
) -> None:
    """Fill the data-pinned decomposition columns of one row dict.

    The model's own asymptotes pin the decomposition: on the malloc arm
    T(m,0) -> W + A_free + N_malloc*m as m -> inf, so the data fix
    (W, A_malloc, A_free) = (T0 - d_m - d_f, d_m, d_f).

    Args:
        out_row: the row dict being built (updated in place).
        fits_row: the stored fits row.
        anchored: the arm_anchored result.
        base00: the measured (0,0) baseline runtime (s).
        data: (the malloc delays, the free delays, the runtimes, the
        stored fit), the first three in seconds.

    """
    m, fv, t, stored = data
    dm = anchored["malloc"]["d"] if anchored["malloc"] else float("nan")
    df_ = anchored["free"]["d"] if anchored["free"] else float("nan")
    w_pin = base00 - dm - df_
    resid_stored = np.abs(model_full(m, fv, stored) - t)
    out_row["Am_stored"] = float(fits_row["A_malloc"])
    out_row["Af_stored"] = float(fits_row["A_free"])
    out_row["d_m"] = dm
    out_row["d_f"] = df_
    out_row["W_pin"] = w_pin
    out_row["dW"] = float(fits_row["W"]) - w_pin
    out_row["dAm"] = float(fits_row["A_malloc"]) - dm
    out_row["dAf"] = float(fits_row["A_free"]) - df_
    out_row["f_m_pin%"] = 100 * dm / base00 if base00 > EPS_S else float("nan")
    out_row["f_f_pin%"] = 100 * df_ / base00 if base00 > EPS_S else float("nan")
    out_row["maxres_stored"] = float(resid_stored.max())


def _multimod_block(group: str, res: dict, stored: np.ndarray) -> dict:
    """Return the multi-modality numbers of the flagged group.

    Args:
        group: the group's label.
        res: the _model_tests result.
        stored: the stored 7-parameter fit.

    Returns:
        dict: the SSR / r2 / F-test numbers of the stored, grid-start,
        and linear fits on the identical data, plus the fitted
        parameters.

    """
    return {
        "group": group,
        "n": res["n"],
        "ssr_stored": res["ssr_stored"],
        "ssr_refit": res["ssr_full"],
        "ssr_lin": res["ssr_lin"],
        "r2_stored": 1 - res["ssr_stored"] / res["ss_tot"],
        "r2_refit": 1 - res["ssr_full"] / res["ss_tot"],
        "r2_lin": 1 - res["ssr_lin"] / res["ss_tot"],
        "F_stored": res["F_stored"],
        "p_stored": res["p_stored"],
        "F_refit": res["F_refit"],
        "p_refit": res["p_refit"],
        "dBIC_refit": res["dBIC"],
        "stored": stored,
        "refit": res["p_full"],
        "dmax": res["dmax"],
    }


def _group_row(runs: pd.DataFrame, row: pd.Series) -> tuple[dict, list, dict, dict, dict | None]:
    """Re-fit one group and build its per-group result row.

    Args:
        runs: all runs.
        row: the group's fits row.

    Returns:
        tuple: (the result row dict, the seed-sensitivity entries, the
        gauge diagnostic, the anchored-arm diagnostic, and the
        multi-modality dict or None).

    """
    gd = _group_arrays(runs, row)
    bounds = bounds_of(gd.m, gd.fv)
    p0 = grid_guess(gd.m, gd.fv, gd.t, bounds)
    res = _model_tests((gd.m, gd.fv), gd.t, gd.stored, bounds, p0)
    spread = run_spread(gd.sub)
    if not np.isfinite(spread):
        spread = float(np.sqrt(res["ssr_full"] / max(gd.n - 7, 1)))
    base00 = _base00((gd.m, gd.fv), gd.t)
    am_ok, af_ok = _flat_profiles((gd.m, gd.fv), gd.t, bounds, res["p_full"], spread)
    arm = arm_offset(gd.m, gd.fv, gd.t)
    arms = arm_profile(gd.m, gd.fv, gd.t)
    seeds = seed_sensitivity((gd.m, gd.fv), gd.t, bounds)
    gauge = gauge_block(gd.m, gd.fv, gd.stored, bounds, res["pcov"])
    out_row: dict = {
        "group": _group_label(row),
        "n": gd.n,
        "n00": gd.n_00,
        "dmax": res["dmax"],
        "F_stored": res["F_stored"],
        "p_stored": res["p_stored"],
        "F_refit": res["F_refit"],
        "p_refit": res["p_refit"],
        "F_Am": res["F_nom"],
        "p_Am": res["p_nom"],
        "F_Af": res["F_nof"],
        "p_Af": res["p_nof"],
        "dBIC": res["dBIC"],
        "cond": res["cond"],
        "f_m%": 100 * (row["A_malloc"] / row["T0"] if row["T0"] > EPS_S else 0),
        "f_f%": 100 * (row["A_free"] / row["T0"] if row["T0"] > EPS_S else 0),
        "cam_us": (row["A_malloc"] / row["N_malloc"]) * 1e6 if row["N_malloc"] == row["N_malloc"] else float("nan"),
        "caf_us": (row["A_free"] / row["N_free"]) * 1e6 if row["N_free"] == row["N_free"] else float("nan"),
        "T0": row["T0"],
        "base00": base00,
        "W": row["W"],
        "spread": spread,
        "fm_err": float(row["f_malloc_err"]) if np.isfinite(row["f_malloc_err"]) else float("nan"),
        "ff_err": float(row["f_free_err"]) if np.isfinite(row["f_free_err"]) else float("nan"),
        "am_off_s": arm["malloc"][0] if arm["malloc"] else float("nan"),
        "am_off_m": arm["malloc"][1] if arm["malloc"] else float("nan"),
        "af_off_s": arm["free"][0] if arm["free"] else float("nan"),
        "af_off_m": arm["free"][1] if arm["free"] else float("nan"),
        "am_range": (float(am_ok.min()), float(am_ok.max())) if am_ok.size else (float("nan"), float("nan")),
        "af_range": (float(af_ok.min()), float(af_ok.max())) if af_ok.size else (float("nan"), float("nan")),
        "T0_fit": float(res["p_full"][0]) + float(res["p_full"][3]) + float(res["p_full"][4]),
        "dSSR_pr": (res["ssr_lin"] - res["ssr_full"]) / gd.n,
        "dSSR_pr_sig2": (res["ssr_lin"] - res["ssr_full"]) / gd.n / spread**2 if spread > 0 else float("nan"),
        "am_offmin": float(arms["malloc"][0].min()) if arms["malloc"][0] is not None else float("nan"),
        "am_offmax": float(arms["malloc"][0].max()) if arms["malloc"][0] is not None else float("nan"),
        "af_offmin": float(arms["free"][0].min()) if arms["free"][0] is not None else float("nan"),
        "af_offmax": float(arms["free"][0].max()) if arms["free"][0] is not None else float("nan"),
    }
    anchored = arm_anchored(gd.m, gd.fv, gd.t, base00)
    _pin_block(out_row, row, anchored, base00, (gd.m, gd.fv, gd.t, gd.stored))
    multimod: dict | None = None
    if row["note"] and isinstance(row["note"], bytes):
        multimod = _multimod_block(out_row["group"], res, gd.stored)
    return out_row, seeds, gauge, anchored, multimod


def main() -> None:
    """Re-fit every sweep, test the stored fits, and print the critique blocks.

    Reads output/results.h5 (read-only) and prints the per-group
    reproduction / nested-model / identifiability table, the flagged
    group's multi-modality, the seed sensitivity, the baseline-anchored
    view, the data-pinned decomposition, and the gauge symmetry
    diagnostic.

    """
    f = h5py.File(RESULTS, "r")
    runs = read_table(f, "runs")
    fits = read_table(f, "fits")
    for col in ("machine", "setup", "algorithm"):
        runs[col] = [v.decode() if isinstance(v, bytes) else v for v in runs[col]]
        fits[col] = [v.decode() if isinstance(v, bytes) else v for v in fits[col]]
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")

    rows: list[dict] = []
    multimod: dict | None = None
    seed_results: dict[str, list] = {}
    anchored_results: dict[str, dict] = {}
    gauge_results: dict[str, dict] = {}
    for _, row in fits.iterrows():
        out_row, seeds, gauge, anchored, multi = _group_row(runs, row)
        rows.append(out_row)
        seed_results[out_row["group"]] = seeds
        gauge_results[out_row["group"]] = gauge
        anchored_results[out_row["group"]] = anchored
        if multi is not None:
            multimod = multi
    f.close()
    df = pd.DataFrame(rows)

    _print_blocks(df, multimod, seed_results, anchored_results, gauge_results)


def _print_blocks(
    df: pd.DataFrame,
    multimod: dict | None,
    seed_results: dict[str, list],
    anchored_results: dict[str, dict],
    gauge_results: dict[str, dict],
) -> None:
    """Print the six critique blocks.

    Args:
        df: the per-group results.
        multimod: the flagged group's multi-modality numbers (or None).
        seed_results: per group, the seed-sensitivity entries.
        anchored_results: per group, the anchored-arm diagnostics.
        gauge_results: per group, the gauge diagnostics.

    """
    _print_per_group(df)
    _print_multimod(multimod)
    _print_seeds(seed_results)
    _print_anchored(anchored_results)
    _print_pin(df)
    _print_gauge(df, gauge_results)


def _print_per_group(df: pd.DataFrame) -> None:
    """Print block 1: the per-group reproduction / nested-model table.

    Args:
        df: the per-group results.

    """

    def frange(r: tuple[float, float], T0: float) -> str:
        if not (np.isfinite(r[0]) and np.isfinite(r[1])):
            return "-"
        return f"[{100 * r[0] / T0:.1f}%,{100 * r[1] / T0:.1f}%]"

    df["am_frange"] = [frange(r, T) for r, T in zip(df["am_range"], df["T0_fit"], strict=True)]
    df["af_frange"] = [frange(r, T) for r, T in zip(df["af_range"], df["T0_fit"], strict=True)]

    out = df[
        [
            "group",
            "n",
            "n00",
            "dmax",
            "F_stored",
            "p_stored",
            "F_refit",
            "p_refit",
            "F_Am",
            "p_Am",
            "F_Af",
            "p_Af",
            "dBIC",
            "cond",
            "f_m%",
            "f_f%",
            "cam_us",
            "caf_us",
            "T0",
            "base00",
            "W",
            "spread",
            "dSSR_pr",
            "dSSR_pr_sig2",
            "am_offmin",
            "am_offmax",
            "af_offmin",
            "af_offmax",
            "am_range",
            "am_frange",
            "af_range",
            "af_frange",
        ]
    ].copy()
    for c in (
        "dmax",
        "F_stored",
        "F_refit",
        "F_Am",
        "F_Af",
        "dBIC",
        "f_m%",
        "f_f%",
        "cam_us",
        "caf_us",
        "T0",
        "base00",
        "W",
        "spread",
        "dSSR_pr",
        "am_offmin",
        "am_offmax",
        "af_offmin",
        "af_offmax",
    ):
        out[c] = df[c].map(lambda v: f"{v:.3g}")
    out["p_stored"] = df["p_stored"].map(lambda v: f"{v:.1e}")
    out["p_refit"] = df["p_refit"].map(lambda v: f"{v:.1e}")
    out["p_Am"] = df["p_Am"].map(lambda v: f"{v:.1e}")
    out["p_Af"] = df["p_Af"].map(lambda v: f"{v:.1e}")
    out["cond"] = df["cond"].map(lambda v: f"{v:.1e}")
    out["dSSR_pr"] = df["dSSR_pr"].map(lambda v: f"{v:.3g}")
    out["dSSR_pr_sig2"] = df["dSSR_pr_sig2"].map(lambda v: f"{v:.2f}")
    out["am_range"] = df["am_range"].map(lambda r: f"[{r[0]:.2g},{r[1]:.2g}]")
    out["af_range"] = df["af_range"].map(lambda r: f"[{r[0]:.2g},{r[1]:.2g}]")
    print("=== 1. per group (reproduction, nested-model tests, identifiability, T0) ===")
    print(out.to_string(index=False))
    print()
    print("F_stored/p_stored: stored 7-param fit vs 3-param linear null (df=4).")
    print("F_refit/p_refit: same, but with a fresh refit from the documented grid initial guess.")
    print("F_Am/p_Am (F_Af/p_Af): full vs 5-param model with the malloc (free) term removed (df=2).")
    print("dBIC: BIC(full 7-param) - BIC(linear null); negative favours the full model.")
    print("dmax: max |fresh refit - stored| (s) -- reproduction of the stored pipeline.")
    print(
        "am/af_range: A values (s) whose fix costs <= spread^2 extra SSR (wide = f unconstrained); "
        "*_frange: the same, as f=A/T0."
    )
    print("cam/caf: implied native cost per call c_a = A/N (us); T0/base00: fitted vs measured (0,0) runtime.")
    print(
        "am_offmin/max, af_offmin/max: per-delay-median offset (s) above the two-largest-delay line, "
        "min and max over the arm."
    )
    print("  (the model's A*s0/(s+s0) requires a monotonically decaying positive offset; negative = below the line).")
    print(
        "dSSR_pr: per-run SSR the 4 saturation params buy over the line (s^2/run); "
        "dSSR_pr_sig2: same, in units of the per-run spread^2."
    )


def _print_multimod(multimod: dict | None) -> None:
    """Print block 2: the flagged group's multi-modality (or nothing).

    Args:
        multimod: the flagged group's multi-modality numbers (or None).

    """
    if multimod is None:
        return
    mm = multimod
    print()
    print("=== 2. multi-modality, flagged group ===")
    print(f"group: {mm['group']}  (n={mm['n']})")
    print(
        f"stored fit:    SSR={mm['ssr_stored']:9.1f}  r2={mm['r2_stored']:.6f}  F vs line={mm['F_stored']:7.2f}  "
        f"p={mm['p_stored']:.2e}"
    )
    print(
        f"grid-start:    SSR={mm['ssr_refit']:9.1f}  r2={mm['r2_refit']:.6f}  F vs line={mm['F_refit']:7.2f}  "
        f"p={mm['p_refit']:.2e}  dBIC={mm['dBIC_refit']:+.1f}"
    )
    print(f"linear null:   SSR={mm['ssr_lin']:9.1f}  r2={mm['r2_lin']:.6f}")
    print(f"stored params:  {np.array2string(mm['stored'], precision=5)}")
    print(f"grid-start p:   {np.array2string(np.asarray(mm['refit']), precision=5)}")
    print(f"max |stored - grid-start| = {mm['dmax']:.1f} s")
    print("note: identical data, identical bounds, identical documented initial guess;")
    print("      the fresh run lands in a different (worse) local minimum than the stored fit,")
    print("      and is even worse than the 3-parameter line.")


def _print_seeds(seed_results: dict[str, list]) -> None:
    """Print block 3: the headline-fraction sensitivity to the fade-scale seed.

    Args:
        seed_results: per group, the seed-sensitivity entries.

    """
    print()
    print("=== 3. headline-fraction sensitivity to the fade-scale seed ===")
    print("same data, same bounds; only the (m0, f0) starting value changes")
    print("(grid = the documented pipeline seed; bottom = the window bottom, a496cda's seed; mid = geometric mid).")
    show = [g for g in seed_results if ("128x128x128" in g) or ("256x128x128" in g)]
    for g in show:
        print(f"\n{g}")
        for name, vals in seed_results[g]:
            if vals is None:
                print(f"  {name:7s}  did not converge")
                continue
            Am, Af, fm, ff, m0s, f0s = vals
            print(
                f"  {name:7s}  A_malloc={Am:8.3f} s  A_free={Af:8.4f} s  f_malloc={100 * fm:5.2f}%  "
                f"f_free={100 * ff:5.2f}%  (m0={m0s * 1e9:.3g} ns, f0={f0s * 1e9:.3g} ns)"
            )


def _print_anchored(anchored_results: dict[str, dict]) -> None:
    """Print block 4: the baseline-anchored view per group.

    Args:
        anchored_results: per group, the anchored-arm diagnostics.

    """
    print()
    print("=== 4. baseline-anchored view: is the small delay hidden by parallel work? ===")
    print("E(s) = T(s) - T(0) - N*s, anchored at the measured zero-delay median; N = large-delay slope.")
    print("d = T(0) - intercept of the large-delay line = plateau of the deficit; d/N = hideable slack per call.")
    print("s_sm/N = slope between the two smallest delays divided by N: <1 => the delay increment is")
    print("partially hidden (the runtime does not rise by N*s); >1 => the delay is amplified.")
    for g, anchored in anchored_results.items():
        for tag, a_ in (("malloc", anchored["malloc"]), ("free", anchored["free"])):
            if a_ is None:
                continue
            print(f"\n{g} {tag}: N={a_['N']:.4g}  d={a_['d']:+8.2f} s ({a_['d_per_call_us']:+.3f} us per call)")
            print(
                f"    E(small)={a_['E0']:+8.2f} s   E(mid)={a_['Emid']:+8.2f} s   E(large)={a_['Emax']:+8.2f} s   "
                f"s_sm/N={a_['s_sm_over_N']:+.3f}"
            )


def _print_pin(df: pd.DataFrame) -> None:
    """Print block 5: the stored fit vs the data-pinned decomposition.

    Args:
        df: the per-group results.

    """
    print()
    print("=== 5. stored fit vs the data-pinned (model-consistent) decomposition ===")
    print("The model's own asymptotes make A_malloc = d_m and A_free = d_f (the measured")
    print("plateaus of block 4), W = T0 - d_m - d_f: three quantities the data fix directly.")
    print("dW/dAm/dAf: stored value minus the data-pinned value (s). f_*_pin: d/T0.")
    pin = df[
        [
            "group",
            "W",
            "dW",
            "Am_stored",
            "dAm",
            "Af_stored",
            "dAf",
            "f_m%",
            "f_m_pin%",
            "f_f%",
            "f_f_pin%",
            "maxres_stored",
        ]
    ]
    for c in ("W", "dW", "Am_stored", "dAm", "Af_stored", "dAf"):
        pin[c] = df[c].map(lambda v: f"{v:+.2f}" if c.startswith("d") else f"{v:.2f}")
    for c in ("f_m%", "f_m_pin%", "f_f%", "f_f_pin%", "maxres_stored"):
        pin[c] = df[c].map(lambda v: f"{v:.2f}")
    print(pin.to_string(index=False))


def _print_gauge(df: pd.DataFrame, gauge_results: dict[str, dict]) -> None:
    """Print block 6: the gauge symmetry of the reported fractions.

    Args:
        df: the per-group results.
        gauge_results: per group, the gauge diagnostics.

    """
    print()
    print("=== 6. gauge symmetry: the reported fractions are not model invariants ===")
    print("The model is exactly invariant under the per-arm delay-origin shifts")
    print("  malloc: W -> W + N_m*m0*d   A_m -> A_m/(1+d)   m0 -> m0*(1+d)   m -> m - m0*d")
    print("  free:   W -> W + N_f*f0*e   A_f -> A_f/(1+e)   f0 -> f0*(1+e)   f -> f - f0*e")
    print("invariants: N_m, N_f, A_m*m0, A_f*f0, W - N_m*m0 - N_f*f0; id = max identity residual (s).")
    print("f_m%/f_f%: stored fractions. *_win: fraction range along the gauge orbit over the fade-scale")
    print("search window (f is a convention, not a model invariant). w/err: that range width in units of")
    print("the stored 1-sigma fraction error. flat_gm/gf: share of the malloc/free gauge direction in the")
    print("two flattest pcov directions (1 = a flat direction, the data do not pin the gauge; 0 = pinned).")

    def rngstr(r: tuple[float, float]) -> str:
        return f"[{100 * r[0]:.1f},{100 * r[1]:.1f}]" if np.isfinite(r[0]) and np.isfinite(r[1]) else "-"

    gtab: list[dict] = []
    for grp, g in gauge_results.items():
        fm_rng, ff_rng = g["fm_rng"], g["ff_rng"]
        fm_err = df.loc[df["group"] == grp, "fm_err"].iloc[0]
        ff_err = df.loc[df["group"] == grp, "ff_err"].iloc[0]
        fm_w = fm_rng[1] - fm_rng[0] if np.isfinite(fm_rng[0]) and np.isfinite(fm_rng[1]) else float("nan")
        ff_w = ff_rng[1] - ff_rng[0] if np.isfinite(ff_rng[0]) and np.isfinite(ff_rng[1]) else float("nan")
        gtab.append(
            {
                "group": grp,
                "id": f"{g['id_res']:.1e}",
                "f_m%": f"{100 * g['fm0']:.1f}",
                "f_m_win": rngstr(fm_rng),
                "w/err_m": f"{fm_w / fm_err:5.1f}" if np.isfinite(fm_w) and fm_err > EPS_S else "-",
                "f_f%": f"{100 * g['ff0']:.1f}",
                "f_f_win": rngstr(ff_rng),
                "w/err_f": f"{ff_w / ff_err:5.1f}" if np.isfinite(ff_w) and ff_err > EPS_S else "-",
                "flat_gm": f"{g['flat'][0]:.2f}" if np.isfinite(g["flat"][0]) else "-",
                "flat_gf": f"{g['flat'][1]:.2f}" if np.isfinite(g["flat"][1]) else "-",
            }
        )
    print(pd.DataFrame(gtab).to_string(index=False))


if __name__ == "__main__":
    main()
