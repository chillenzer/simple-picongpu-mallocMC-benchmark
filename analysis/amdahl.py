"""The Amdahl allocation model and its constrained fits.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Each of the N allocation calls on the (serial, host-side) critical path takes
its native cost c_a plus the imposed delay s, so

    T(s) = W + A + N*s,   A = N*c_a (native allocation time)

with W the runtime without any allocation cost. The native part is not needed
for large sleeptimes (negligible against the imposed delay) but acts as a
correction at small sleeptimes, so a sweep is fitted with

    T(s) = W + N*s + A*s0/(s+s0)             (Amdahl model)

which reduces to the Amdahl line W + N*s for s >> s0 and to W + A for
s -> 0. The parameters are (W, N, A, s0): baseline runtime, allocation calls
per run, native allocation time, and the sleeptime scale over which the
native cost fades. The Amdahl fraction of the runtime spent in allocations is

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
import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

_INF = float("inf")
# Near-zero floor for second-based runtimes: the Amdahl fraction is
# reported as 0 when the total runtime drops below it, and the fade
# scales (s0, m0, f0) are bounded by it from below.
EPS_S = 1e-12


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


def model_1d(s: np.ndarray, W: float, N: float, A: float, s0: float) -> np.ndarray:
    """Evaluate the 1-operation Amdahl model T(s) = W + N*s + A*s0/(s+s0).

    Args:
        s: the imposed delay in seconds.
        W: the baseline runtime (s).
        N: the allocation calls per run.
        A: the native allocation time (s).
        s0: the fade scale of the native correction (s).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    return W + N * s + A * s0 / (s + s0)


def model_2d(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Evaluate the 2-operation Amdahl model, the delays m (malloc) and f (free) in seconds.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, N_m, N_f, A_m, A_f, m0, f0 = p
    return W + N_m * m + N_f * f + A_m * m0 / (m + m0) + A_f * f0 / (f + f0)


def model_c_a(s: np.ndarray, W: float, N: float, s0: float, c: float) -> np.ndarray:
    """Evaluate the 1-D model with the A = N*c constraint, c in seconds.

    Args:
        s: the imposed delay in seconds.
        W: the baseline runtime (s).
        N: the allocation calls per run.
        s0: the fade scale of the native correction (s).
        c: the native per-call cost (s), so that A = N*c.

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    A = N * c
    return W + N * s + A * s0 / (s + s0)


_BOOTSTRAP_N = 512
_BOOTSTRAP_PERCENTILES = (25.0, 75.0)
_BOOTSTRAP_SEED = 0


def bootstrap_band(
    x_s: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
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

    Args:
        x_s: the model x-grid in seconds.
        fn: the model function of the fitted parameter vector.
        params: the fitted parameter vector.
        pcov: the parameter covariance matrix.
        lower: per-parameter lower bounds the sampled parameters are clipped
        to, or None.

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

    Args:
        h: the Amdahl correction term h = s0/(s+s0), one value per point.
        s: the sleeptimes in seconds.
        t: the runtimes in seconds.

    Returns:
        tuple[float, float, float, float]: W, N, A, and the residual sum of squares.

    """
    sol, *_ = np.linalg.lstsq(np.vstack([np.ones_like(s), s, h]).T, t, rcond=None)
    res = t - (sol[0] + sol[1] * s + sol[2] * h)
    return (*[float(v) for v in sol], float(np.sum(res**2)))


def _grid_guess(s: np.ndarray, t: np.ndarray, lo: float, hi: float) -> tuple[float, float, float, float]:
    """Robust unconstrained solution: linear in (W, N, A) for each s0 on a log grid.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.
        lo: the search-range lower bound of s0 (s).
        hi: the search-range upper bound of s0 (s).

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

    Args:
        p: the fitted parameter vector.
        pcov: the parameter covariance matrix, or None.
        f_of_p: the Amdahl fraction as a function of the parameter vector.

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

    Args:
        p: the 1-D parameter vector (W, N, A, s0).

    Returns:
        float: the Amdahl fraction of runtime spent in allocations.

    """
    return p[2] / (p[0] + p[2]) if p[0] + p[2] > EPS_S else 0.0


def _f_of_p_ca(p: np.ndarray, c: float) -> float:
    """Compute the Amdahl fraction of the A = N*c constrained 1-D parameters p.

    Args:
        p: the constrained 1-D parameter vector (W, N, s0).
        c: the native per-call cost (s).

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

    Args:
        guess: the robust grid solution (W, N, A, s0).
        eps: the floor for the initial guess.
        bounds: the (lo_b, hi_b) search range of s0.

    Returns:
        list[float]: the grid guess floored at `eps`, s0 clipped to the bounds.

    """
    gW, gN, gA, gs0 = guess
    return [max(gW, eps), max(gN, eps), max(gA, eps), float(np.clip(gs0, bounds[0], bounds[1]))]


def _p0_ca(guess: tuple[float, float, float, float], eps: float, bounds: tuple[float, float]) -> list[float]:
    """Compute the initial curve_fit parameters of the A = N*c constrained 1-D fit.

    Args:
        guess: the robust grid solution (W, N, A, s0); its A is unused.
        eps: the floor for the initial guess.
        bounds: the (lo_b, hi_b) search range of s0.

    Returns:
        list[float]: the grid guess floored at `eps`, s0 clipped to the bounds.

    """
    gW, gN, _gA, gs0 = guess
    return [max(gW, eps), max(gN, eps), float(np.clip(gs0, bounds[0], bounds[1]))]


def _fit_notes_1d(model: Model1d, guess: tuple[float, float, float, float], bounds: tuple[float, float]) -> list[str]:
    """Report the corner solutions of the constrained 1-D fit.

    Args:
        model: the fitted 1-D model.
        guess: the robust grid solution (W, N, A, s0).
        bounds: the (lo_b, hi_b) search range of s0.

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

    Args:
        guess: the robust grid solution (W, N, A, s0).

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
                model_1d,
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
            lambda qq, W, N, s0: model_c_a(qq, W, N, s0, c),
            s,
            t,
            p0=p0,
            bounds=([0.0, 0.0, setup.bounds[0]], [_INF, _INF, setup.bounds[1]]),
            maxfev=20000,
        )
    W, N, s0 = (float(v) for v in popt)
    model = Model1d(W, N, N * c, s0)
    return ConstrainedFit(model, pcov, list(popt), _fit_notes_1d(model, setup.guess, setup.bounds))


def fit_1d(sleeptimes: pd.Series, runtimes: pd.Series, c_a: float | None = None) -> dict:
    """Fit one sleeptime sweep to the constrained Amdahl model.

    Args:
        sleeptimes: the imposed delays in nanoseconds.
        runtimes: the runtimes in seconds.
        c_a: the A = N*c constraint in nanoseconds, or None for the full model.

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
        pred = model_1d(s, W, N, A, s0)
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

    Args:
        p: the 2-D parameter vector (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        float: the Amdahl fraction of runtime spent in allocations.

    """
    return p[3] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0


def _f_of_f(p: np.ndarray) -> float:
    """Compute the Amdahl fraction of the free operation of the 2-D parameters p.

    Args:
        p: the 2-D parameter vector (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        float: the Amdahl fraction of runtime spent in frees.

    """
    return p[4] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0


def _fit_bounds_2d(m: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    """Compute the search ranges of the 2-D fade scales.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.

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

    Args:
        guess: the robust grid solution (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        tuple[float, ...]: the floored (W, N_m, N_f, A_m, A_f, m0, f0) guess.

    """
    W, N_m, N_f, A_m, A_f, m0, f0 = guess
    return W, N_m, N_f, max(0.0, A_m), max(0.0, A_f), m0, f0


def _p0_2d(guess: tuple[float, ...], eps: float) -> list[float]:
    """Compute the initial curve_fit parameters of the 2-D fit.

    Args:
        guess: the robust grid solution (W, N_m, N_f, A_m, A_f, m0, f0).
        eps: the floor for the initial guess.

    Returns:
        list[float]: the grid guess floored at `eps`.

    """
    gW, gNm, gNf, gAm, gAf, gm0, gf0 = guess
    return [max(gW, eps), max(gNm, eps), max(gNf, eps), max(gAm, eps), max(gAf, eps), gm0, gf0]


def _fit_notes_2d(model: Model2d, guess: tuple[float, ...], bounds: tuple[float, float, float, float]) -> list[str]:
    """Report the corner solutions of the constrained 2-D fit.

    Args:
        model: the fitted 2-D model.
        guess: the robust grid solution (W, N_m, N_f, A_m, A_f, m0, f0).
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.

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
            lambda pf, W, N_m, N_f, A_m, A_f, m0, f0: model_2d(pf[0], pf[1], (W, N_m, N_f, A_m, A_f, m0, f0)),
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


def fit_2d(m_delays: pd.Series, f_delays: pd.Series, runtimes: pd.Series) -> dict:
    """Fit one (malloc, free) delay combination sweep to the two-operation Amdahl model.

    The model is T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0).

    Args:
        m_delays: the malloc delays in nanoseconds.
        f_delays: the free delays in nanoseconds.
        runtimes: the runtimes in seconds.

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
        pred = model_2d(m, f, (model.W, model.N_m, model.N_f, model.A_m, model.A_f, model.m0, model.f0))
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
