"""The allocation model and its constrained fits.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

With no imposed delay the measured runtime is the full calculation's real
runtime with each allocator (the (0, 0) baseline); the delay arms study how the
allocator interacts with the pipeline. Each of the N allocation calls on the
critical path takes the imposed delay s, so at large delay

    T(s) = W + N*s,

with W the runtime at zero delay. The delay does not simply add: part of it is
hidden by the parallel work that runs while the allocator spins, so the
absorbed part fades over the sleeptime scale s0 and a sweep is fitted with

    T(s) = W + N*s + A * g(s/s0)             (allocation model)

which equals the baseline T0 = W + A for s -> 0 and approaches the asymptote
W + N*s for s >> s0, where every call's delay is exposed. The parameters are
(W, N, A, s0): baseline runtime, allocation calls per run, the total delay
absorbed per run (A), and the sleeptime scale over which the absorption fades.
The per-call absorbed slack is c = A/N. The ratio

    f = A / (W + A)      (absorbed delay / total time at zero delay)

is a convention-dependent slack ratio, not a runtime budget and not the native
allocation cost (see analysis-review.md: the "A = native allocation time"
reading is not supported by the data).

The fade shape g is a normalized function of the reduced delay u = s/s0 with
g(0) = 1 and g(inf) = 0, so that A is always the absorbed delay at zero delay.
The candidate shapes are collected in `FADE_SHAPES` (hyperbola, exponential,
lorentzian, truncated, quadratic) and the model is fit with any of them via
the `fade` argument of `model_1d`, `model_2d`, `fit_1d`, `fit_2d` and
`fit_combined`. The default and selected best shape is the exponential
`g(u) = exp(-u)` (see qa-fade-term.md, Q3): among the two-parameter candidates
it has the best mean fit, is smooth, has a finite total absorbed cost
(integral over u is A*s0, vs the hyperbola's divergent A*s0/u tail), and pins
its scale s0 as a gauge invariant. The hyperbola `g(u) = 1/(1+u)` -- the
original working model -- is kept as a candidate for comparison.

The fit is `scipy.optimize.curve_fit` with bounds W>=0, N>=0, A>=0 and
s0 in [0.05*s_min, 0.5*s_range], so the reported f is always in [0, 1). The
s0 lower limit keeps the correction from being confined to s=0 only; the s0
upper limit keeps it (and A) identifiable instead of a constant offset
degenerate with W. A robust linear solution over a log-s0 grid provides the
initial guess; if `curve_fit` fails to converge that linear solution is
reported instead. The parameter covariance from `curve_fit` is propagated to
f (and to W, N, A, s0) as standard errors.

When a sweep imposes a delay on both operations (the (malloc_delay,
free_delay) combination runs of the delay matrix), each operation gets its own
saturation term, i.e. the absorbed delay of an operation only fades while its
own imposed delay grows. Such a sweep is then fitted with the two-operation
(separable, no cross-term) model

    T(m, f) = W + N_m*m + N_f*f + A_m*g(m/m0) + A_f*g(f/f0)

(with the selected best shape, the exponential, g(u) = exp(-u)) and the
slack ratios f_malloc = A_m/T0, f_free = A_f/T0 (absorbed delay over
zero-delay runtime, T0 = W + A_m + A_f) are reported separately.
Groups whose runs vary only one of the two delays fall back to the 1-D model
above, fitted on that delay.

Several algorithms' sweeps of the same (setup, grid) scenario can be fit
jointly with `fit_combined`: the baseline runtime W and the call counts
N_malloc / N_free are algorithm-invariant and are fit once on the pooled
data, while each algorithm keeps its own absorbed delays (A_malloc, A_free)
and fade scales (m0, f0), so every algorithm's curve is shared exactly
where theory says it may differ between algorithms.

`sleeptimes` are in nanoseconds, `runtimes` in seconds. If an independently
measured native per-call cost c_a (ns) is available, pass it: A = N*c_a is
then used and only (W, N, s0) are fitted — the path to a native-cost
reading, unused by default (no such microbenchmark exists yet).
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
# Near-zero floor for second-based runtimes: the fraction is
# reported as 0 when the total runtime drops below it, and the fade
# scales (s0, m0, f0) are bounded by it from below.
EPS_S = 1e-12


def _fade_hyperbola(u: np.ndarray) -> np.ndarray:
    """Hyperbola fade shape g(u) = 1/(1+u) (the original working model).

    Args:
        u: the reduced delay s/s0.

    Returns:
        np.ndarray: the fade shape, 1 at u = 0 and 0 for u -> inf.

    """
    return 1.0 / (1.0 + u)


def _fade_exponential(u: np.ndarray) -> np.ndarray:
    """Exponential fade shape g(u) = exp(-u) (the selected best model).

    Args:
        u: the reduced delay s/s0.

    Returns:
        np.ndarray: the fade shape, 1 at u = 0 and 0 for u -> inf.

    """
    return np.exp(-u)


def _fade_lorentzian(u: np.ndarray) -> np.ndarray:
    """Lorentzian fade shape g(u) = 1/(1+u^2).

    Args:
        u: the reduced delay s/s0.

    Returns:
        np.ndarray: the fade shape, 1 at u = 0 and 0 for u -> inf.

    """
    return 1.0 / (1.0 + u * u)


def _fade_truncated(u: np.ndarray) -> np.ndarray:
    """Truncated-linear fade shape g(u) = max(1-u, 0) (the overlap null model).

    Args:
        u: the reduced delay s/s0.

    Returns:
        np.ndarray: the fade shape, 1 at u = 0 and 0 for u >= 1.

    """
    return np.maximum(1.0 - u, 0.0)


def _fade_quadratic(u: np.ndarray) -> np.ndarray:
    """Quadratic-overlap fade shape g(u) = max(1-u, 0)^2.

    Args:
        u: the reduced delay s/s0.

    Returns:
        np.ndarray: the fade shape, 1 at u = 0 and 0 for u >= 1.

    """
    return np.maximum(1.0 - u, 0.0) ** 2


#: The candidate fade shapes g(u) of the allocation model, keyed by name.
#: Each satisfies g(0) = 1 and g(u -> inf) = 0, so the model term A*g(s/s0)
#: is the absorbed delay fading from A at zero delay to 0 at large delay.
#: The power-law candidate (three parameters, A, s0, k) is not a member of
#: this two-parameter family; it is fit only by the comparison tooling.
FADE_SHAPES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hyperbola": _fade_hyperbola,
    "exponential": _fade_exponential,
    "lorentzian": _fade_lorentzian,
    "truncated": _fade_truncated,
    "quadratic": _fade_quadratic,
}

#: The selected best fade shape, and the default of every fit and model call.
BEST_FADE = "exponential"


def _fade(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the normalized fade shape function of one candidate by name.

    Args:
        name: the fade key, a member of `FADE_SHAPES`.

    Returns:
        Callable: the fade shape g(u).

    Raises:
        ValueError: if the name is not a known fade shape.

    """
    try:
        return FADE_SHAPES[name]
    except KeyError:
        msg = f"unknown fade {name!r} (available: {', '.join(FADE_SHAPES)})"
        raise ValueError(msg) from None


class Model1d(NamedTuple):
    """Parameters of the 1-operation allocation model T(s) = W + N*s + A*g(s/s0).

    The delay s is in seconds and g is the candidate fade shape (default the
    exponential, g(u) = exp(-u); see the module docstring and `FADE_SHAPES`).
    """

    W: float
    N: float
    A: float
    s0: float


class Model2d(NamedTuple):
    """Parameters of the 2-operation allocation model, the delays in seconds."""

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
    grid solution (W, N, A, s0), `eps` the floor for the initial guess, and
    `fade` the fade shape name the fit uses.
    """

    bounds: tuple[float, float]
    guess: tuple[float, float, float, float]
    eps: float
    fade: str = BEST_FADE


class ConstrainedFit(NamedTuple):
    """A successful curve_fit attempt of one sweep.

    `notes` carries the corner-solution diagnostics (A floored at 0, fade
    scale at the search cap); the caller appends them to its warnings.
    """

    model: Model1d | Model2d
    pcov: np.ndarray | None
    fit_params: list[float] | None
    notes: list[str]


def model_1d(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    s: np.ndarray, W: float, N: float, A: float, s0: float, fade: str = BEST_FADE
) -> np.ndarray:
    """Evaluate the 1-operation allocation model T(s) = W + N*s + A*g(s/s0).

    Args:
        s: the imposed delay in seconds.
        W: the baseline runtime (s).
        N: the allocation calls per run.
        A: the total imposed delay absorbed by parallel work per run (s).
        s0: the sleeptime scale over which the absorption fades (s).
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    return W + N * s + A * _fade(fade)(s / s0)


def model_2d(m: np.ndarray, f: np.ndarray, p: Sequence[float], fade: str = BEST_FADE) -> np.ndarray:
    """Evaluate the 2-operation allocation model, the delays m (malloc) and f (free) in seconds.

    The model is T(m, f) = W + N_m*m + N_f*f + A_m*g(m/m0) + A_f*g(f/f0).

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_m, N_f, A_m, A_f, m0, f0).
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, N_m, N_f, A_m, A_f, m0, f0 = p
    shape = _fade(fade)
    return W + N_m * m + N_f * f + A_m * shape(m / m0) + A_f * shape(f / f0)


def model_c_a(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    s: np.ndarray, W: float, N: float, s0: float, c: float, fade: str = BEST_FADE
) -> np.ndarray:
    """Evaluate the 1-D model with the A = N*c constraint, c in seconds.

    Args:
        s: the imposed delay in seconds.
        W: the baseline runtime (s).
        N: the allocation calls per run.
        s0: the fade scale of the native correction (s).
        c: the native per-call cost (s), so that A = N*c.
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    A = N * c
    return W + N * s + A * _fade(fade)(s / s0)


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

    The parameters enter the model non-linearly (the saturation terms), so the
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
        h: the fade shape column h = g(s/s0), one value per point.
        s: the sleeptimes in seconds.
        t: the runtimes in seconds.

    Returns:
        tuple[float, float, float, float]: W, N, A, and the residual sum of squares.

    """
    sol, *_ = np.linalg.lstsq(np.vstack([np.ones_like(s), s, h]).T, t, rcond=None)
    res = t - (sol[0] + sol[1] * s + sol[2] * h)
    return (*[float(v) for v in sol], float(np.sum(res**2)))


def _grid_guess(
    s: np.ndarray, t: np.ndarray, lo: float, hi: float, fade: str = BEST_FADE
) -> tuple[float, float, float, float]:
    """Robust unconstrained solution: linear in (W, N, A) for each s0 on a log grid.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.
        lo: the search-range lower bound of s0 (s).
        hi: the search-range upper bound of s0 (s).
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        tuple[float, float, float, float]: W, N, A, and the best s0 from the log grid.

    """
    shape = _fade(fade)
    hi_g = max(float(hi), lo * 1.5)
    best = None
    for s0 in np.logspace(np.log10(lo), np.log10(hi_g), 60):
        W, N, A, ss_res = _fit_lsq(shape(s / s0), s, t)
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
        f_of_p: the fraction as a function of the parameter vector.

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
    """Compute the slack ratio f = A/(W + A) (absorbed delay over the zero-delay runtime), from the 1-D parameters p.

    Args:
        p: the 1-D parameter vector (W, N, A, s0).

    Returns:
        float: the slack ratio (absorbed delay / zero-delay runtime).

    """
    return p[2] / (p[0] + p[2]) if p[0] + p[2] > EPS_S else 0.0


def _f_of_p_ca(p: np.ndarray, c: float) -> float:
    """Compute the slack ratio f = A/(W + A) from the A = N*c constrained 1-D parameters p.

    With the c_a constraint, A is the native cost and f the true native
    cost fraction; unconstrained, f is the convention-dependent slack
    ratio (absorbed delay / zero-delay runtime).

    Args:
        p: the constrained 1-D parameter vector (W, N, s0).
        c: the native per-call cost (s).

    Returns:
        float: the slack ratio (A / zero-delay runtime).

    """
    return p[1] * c / (p[0] + p[1] * c) if p[0] + p[1] * c > EPS_S else 0.0


def _fit_setup_1d(s: np.ndarray, t: np.ndarray, fade: str = BEST_FADE) -> FitSetup:
    """Precompute the s0 bounds, robust initial guess, and guess floor of a 1-D fit.

    Args:
        s: sorted sleeptimes in seconds.
        t: the runtimes in seconds.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        FitSetup: the (lo_b, hi_b) bounds, the grid guess, eps, and the fade.

    """
    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = max(0.05 * s_min_pos, EPS_S)
    hi = 0.5 * (s.max() - s.min())
    guess = _grid_guess(s, t, lo, hi, fade)  # robust, unconstrained (linear in W, N, A per s0)
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    lo_b = max(lo, EPS_S)
    hi_b = max(hi, lo_b * 1.5)
    return FitSetup(bounds=(lo_b, hi_b), guess=guess, eps=eps, fade=fade)


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
            "unconstrained fit wanted A<0 (smallest-sleeptime runtime below the "
            "large-delay asymptote); A constrained to 0 so f is floored at 0"
        )
    if model.s0 >= bounds[1] * 0.999:
        notes.append(
            "s0 reached the search cap: the absorption does not clearly "
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
    """Fit a 3-point sweep: the large-delay line through the two largest sleeptimes.

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
                lambda qq, W, N, A, s0: model_1d(qq, W, N, A, s0, fade=setup.fade),
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
            lambda qq, W, N, s0: model_c_a(qq, W, N, s0, c, fade=setup.fade),
            s,
            t,
            p0=p0,
            bounds=([0.0, 0.0, setup.bounds[0]], [_INF, _INF, setup.bounds[1]]),
            maxfev=20000,
        )
    W, N, s0 = (float(v) for v in popt)
    model = Model1d(W, N, N * c, s0)
    return ConstrainedFit(model, pcov, list(popt), _fit_notes_1d(model, setup.guess, setup.bounds))


def fit_1d(sleeptimes: pd.Series, runtimes: pd.Series, c_a: float | None = None, fade: str = BEST_FADE) -> dict:
    """Fit one sleeptime sweep to the constrained allocation model.

    Args:
        sleeptimes: the imposed delays in nanoseconds.
        runtimes: the runtimes in seconds.
        c_a: the A = N*c constraint in nanoseconds, or None for the full model.
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

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
        pred = model_1d(s, W, N, A, s0, fade=fade)
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
            "W": W,  # no-delay runtime (s)
            "N": N,  # allocation calls per run
            "A": A,  # total imposed delay absorbed per run (s)
            "s0": s0,  # fade scale of the absorption (s)
            "T0": T0,  # total runtime at zero delay (s)
            "f": f,  # slack ratio: absorbed delay / zero-delay runtime
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
    setup = _fit_setup_1d(s, t, fade)
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
    """Compute the slack ratio f_malloc = A_m/(W + A_m + A_f), from the 2-D parameters p.

    The malloc absorbed delay over the zero-delay runtime T0.

    Args:
        p: the 2-D parameter vector (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        float: the slack ratio (absorbed delay / zero-delay runtime).

    """
    return p[3] / (p[0] + p[3] + p[4]) if p[0] + p[3] + p[4] > EPS_S else 0.0


def _f_of_f(p: np.ndarray) -> float:
    """Compute the slack ratio f_free = A_f/(W + A_m + A_f), from the 2-D parameters p.

    The free absorbed delay over the zero-delay runtime T0.

    Args:
        p: the 2-D parameter vector (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        float: the slack ratio (absorbed delay / zero-delay runtime).

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


def _grid_row_2d(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    m: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    m0: float,
    bounds: tuple[float, float, float, float],
    fade: str = BEST_FADE,
) -> tuple[float, ...]:
    """Best f0 of the 2-D grid search at a fixed m0.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        m0: the fixed malloc fade scale in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        tuple[float, ...]: (ss_res, W, N_m, N_f, A_m, A_f, m0, f0) of the best row.

    """
    shape = _fade(fade)
    best = None
    for f0 in np.logspace(np.log10(bounds[2]), np.log10(bounds[3]), 25):
        X = np.vstack([np.ones_like(m), m, f, shape(m / m0), shape(f / f0)]).T
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
    fade: str = BEST_FADE,
) -> tuple[float, ...]:
    """Robust unconstrained 2-D solution, linear in (W, N_m, N_f, A_m, A_f) per (m0, f0).

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        tuple[float, ...]: W, N_m, N_f, A_m, A_f, m0, f0 minimizing the residual.

    """
    m, f = sweep
    best = None
    for m0 in np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), 25):
        row = _grid_row_2d(m, f, t, float(m0), bounds, fade)
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
            "m0 reached the search cap: the absorbed malloc delay does not clearly fade, so "
            "A_malloc and W (hence f_malloc) are weakly constrained"
        )
    if model.f0 >= hi_f * 0.999:
        notes.append(
            "f0 reached the search cap: the absorbed free delay does not clearly fade, so "
            "A_free and W (hence f_free) are weakly constrained"
        )
    return notes


def _fit_constrained_2d(
    sweep: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
    guess: tuple[float, ...],
    fade: str = BEST_FADE,
) -> ConstrainedFit:
    """Run curve_fit for a 2-D sweep.

    Args:
        sweep: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        bounds: the (lo_m, hi_m, lo_f, hi_f) search ranges.
        guess: the robust grid solution (W, N_m, N_f, A_m, A_f, m0, f0).
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        ConstrainedFit: the fitted model, its covariance, and corner notes.

    """
    m, f = sweep
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    p0 = _p0_2d(guess, eps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            lambda pf, W, N_m, N_f, A_m, A_f, m0, f0: model_2d(
                pf[0], pf[1], (W, N_m, N_f, A_m, A_f, m0, f0), fade=fade
            ),
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


def fit_2d(m_delays: pd.Series, f_delays: pd.Series, runtimes: pd.Series, fade: str = BEST_FADE) -> dict:
    """Fit one (malloc, free) delay combination sweep to the two-operation allocation model.

    The model is T(m, f) = W + N_m*m + N_f*f + A_m*g(m/m0) + A_f*g(f/f0).

    Args:
        m_delays: the malloc delays in nanoseconds.
        f_delays: the free delays in nanoseconds.
        runtimes: the runtimes in seconds.
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

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
        pred = model_2d(m, f, (model.W, model.N_m, model.N_f, model.A_m, model.A_f, model.m0, model.f0), fade=fade)
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
            "W": float(model.W),  # no-delay runtime (s)
            "N_m": float(model.N_m),  # allocation calls per run
            "N_f": float(model.N_f),  # free calls per run
            "A_m": float(model.A_m),  # absorbed malloc delay per run (s)
            "A_f": float(model.A_f),  # absorbed free delay per run (s)
            "m0": float("nan") if math.isnan(model.m0) else float(model.m0),  # malloc fade scale (s)
            "f0": float("nan") if math.isnan(model.f0) else float(model.f0),  # free fade scale (s)
            "T0": T0,  # total runtime at zero delay (s)
            "f_malloc": f_malloc,  # slack ratio: absorbed malloc delay / zero-delay runtime
            "f_free": f_free,  # slack ratio: absorbed free delay / zero-delay runtime
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
    guess = _grid_guess_2d(sweep, t, bounds, fade)
    try:
        cf = _fit_constrained_2d(sweep, t, bounds, guess, fade)
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


def _combined_kind(m: np.ndarray, f: np.ndarray) -> str:
    """Classify the pooled delay data of a combined fit, as the individual fits do.

    Args:
        m: the pooled malloc delays in seconds.
        f: the pooled free delays in seconds.

    Returns:
        str: "2d" when both delays vary, "1d-malloc" / "1d-free" when only
        one does, or "" when neither varies (no fit possible).

    """
    if len(np.unique(m)) >= 2 and len(np.unique(f)) >= 2:
        return "2d"
    if len(np.unique(m)) >= 2:
        return "1d-malloc"
    if len(np.unique(f)) >= 2:
        return "1d-free"
    return ""


def _combined_bounds(kind: str, n: int, m: np.ndarray, f: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute the parameter bounds of a combined fit.

    The shared W and the shared call count(s) and the per-algorithm A terms
    are floored at 0; each algorithm's fade scale(s) share the pooled
    per-operation search range of the individual fits.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        n: the number of pooled algorithms.
        m: the pooled malloc delays in seconds.
        f: the pooled free delays in seconds.

    Returns:
        tuple: the (lower, upper) bound vectors, one entry per parameter.

    """
    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (float(m.max()) - float(m.min())), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (float(f.max()) - float(f.min())), lo_f * 1.5)
    if kind == "2d":
        lo_b = [0.0, 0.0, 0.0]
        hi_b = [_INF, _INF, _INF]
        for _ in range(n):
            lo_b += [0.0, 0.0, lo_m, lo_f]
            hi_b += [_INF, _INF, hi_m, hi_f]
        return lo_b, hi_b
    lo0, hi0 = (lo_m, hi_m) if kind == "1d-malloc" else (lo_f, hi_f)
    lo_b = [0.0, 0.0]
    hi_b = [_INF, _INF]
    for _ in range(n):
        lo_b += [0.0, lo0]
        hi_b += [_INF, hi0]
    return lo_b, hi_b


def _combined_predict(
    kind: str, x: tuple | np.ndarray, p: np.ndarray, indicators: list[np.ndarray], fade: str = BEST_FADE
) -> np.ndarray:
    """Evaluate the combined model at the pooled points.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        x: the delay data -- (m, f) for "2d", one array otherwise -- in seconds.
        p: the full parameter vector: the shared parameters first, then one
        (A_malloc, A_free, m0, f0) block per algorithm ("2d") or one
        (A, s0) block ("1d-*"), in the pooled algorithms' order.
        indicators: per-algorithm 0/1 indicator arrays of the pooled points.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        np.ndarray: the model runtimes in seconds.

    """
    shape = _fade(fade)
    if kind == "2d":
        m, f = x
        t = p[0] + p[1] * m + p[2] * f
        for k, ind in enumerate(indicators):
            A_m, A_f, m0, f0 = p[3 + 4 * k : 7 + 4 * k]
            t += ind * (A_m * shape(m / m0) + A_f * shape(f / f0))
        return t
    s = x
    t = p[0] + p[1] * s
    for k, ind in enumerate(indicators):
        A, s0 = p[2 + 2 * k : 4 + 2 * k]
        t += ind * (A * shape(s / s0))
    return t


def _combined_layout(kind: str) -> tuple[int, int]:
    """Shared-parameter layout of the combined model.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".

    Returns:
        tuple[int, int]: (the number of shared parameters, the
        per-algorithm block size).

    """
    return (3, 4) if kind == "2d" else (2, 2)


def _combined_order(order: Sequence[str] | None, first_seen: list[str]) -> list[str]:
    """Normalize the pooled algorithms' order against the data.

    Args:
        order: the requested order, or None for first appearance.
        first_seen: the algorithms in first-appearance order.

    Returns:
        list[str]: the requested algorithms that are present, first, then
        the remaining ones in first-appearance order.

    """
    if order is None:
        return list(first_seen)
    wanted = [str(a) for a in order if str(a) in set(first_seen)]
    return wanted + [a for a in first_seen if a not in wanted]


def _combined_design(
    kind: str,
    x: tuple | np.ndarray,
    indicators: list[np.ndarray],
    fades: list[tuple],
    fade: str = BEST_FADE,
) -> np.ndarray:
    """Design matrix of the combined model with the fade scales held fixed.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        x: the delay data (as for `_combined_predict`), in seconds.
        indicators: per-algorithm 0/1 indicator arrays.
        fades: per algorithm, the fixed fade scale(s) in seconds --
        (m0, f0) for "2d", (s0,) for "1d-*".
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        np.ndarray: the design matrix, one column per fitted coefficient.

    """
    shape = _fade(fade)
    if kind == "2d":
        m, f = x
        columns: list[np.ndarray] = [np.ones_like(m), m, f]
    else:
        s = x
        columns = [np.ones_like(s), s]
    for ind, fl in zip(indicators, fades, strict=True):
        if kind == "2d":
            m0, f0 = fl
            columns.append(shape(m / m0) * ind)
            columns.append(shape(f / f0) * ind)
        else:
            (s0,) = fl
            columns.append(shape(s / s0) * ind)
    return np.vstack(columns).T


def _robust_combined_vector(kind: str, sol: np.ndarray, fades: list[tuple]) -> list[float]:
    """Insert the fixed fade scales into a robust combined solution.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        sol: the least-squares coefficients of `_combined_design`.
        fades: per algorithm, the fixed fade scale(s) in seconds.

    Returns:
        list[float]: the full parameter vector with the fades inserted.

    """
    head, stride = _combined_layout(kind)
    n_a = stride - (2 if kind == "2d" else 1)
    p = [0.0] * (head + stride * len(fades))
    p[:head] = [float(v) for v in sol[:head]]
    for k, fl in enumerate(fades):
        base = head + k * stride
        p[base : base + n_a] = [float(v) for v in sol[base : base + n_a]]
        p[base + n_a : base + stride] = [float(v) for v in fl]
    return p


def _combined_robust(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    kind: str,
    x: tuple | np.ndarray,
    t: np.ndarray,
    indicators: list[np.ndarray],
    fades: list[tuple],
    fade: str = BEST_FADE,
) -> tuple[float, list[float]]:
    """Least squares of the combined model with the fade scales held fixed.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        x: the delay data (as for `_combined_predict`), in seconds.
        t: the runtimes in seconds.
        indicators: per-algorithm 0/1 indicator arrays.
        fades: per algorithm, the fixed fade scale(s) in seconds.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        tuple: (the residual sum of squares, the full parameter vector
        with the fixed fades inserted).

    """
    matrix = _combined_design(kind, x, indicators, fades, fade)
    sol, *_ = np.linalg.lstsq(matrix, t, rcond=None)
    ss_res = float(np.sum((t - matrix @ sol) ** 2))
    return ss_res, _robust_combined_vector(kind, sol, fades)


def _own_guess(kind: str, m_a: np.ndarray, f_a: np.ndarray, t_a: np.ndarray, fade: str = BEST_FADE) -> dict[str, float]:
    """Robust grid solution of one algorithm's own sweep.

    Args:
        kind: "2d", "1d-malloc", or "1d-free" (the pooled kind).
        m_a: the algorithm's malloc delays in seconds.
        f_a: the algorithm's free delays in seconds.
        t_a: the algorithm's runtimes in seconds.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        dict: W, N_m, N_f, A_m, A_f, m0, f0 in seconds; the inapplicable
        operation's parameters are NaN, as are all of them with fewer
        than 3 points.

    """
    nan = float("nan")
    out: dict[str, float] = {"W": nan, "N_m": nan, "N_f": nan, "A_m": nan, "A_f": nan, "m0": nan, "f0": nan}
    if t_a.size < 3:
        return out
    if kind == "2d":
        values = _grid_guess_2d((m_a, f_a), t_a, _fit_bounds_2d(m_a, f_a), fade)
        out.update(dict(zip(("W", "N_m", "N_f", "A_m", "A_f", "m0", "f0"), values, strict=True)))
        return out
    if kind == "1d-malloc":
        lo_m, hi_m, _lo_f, _hi_f = _fit_bounds_2d(m_a, f_a)
        W, N, A, s0 = _grid_guess(m_a, t_a, max(lo_m, EPS_S), hi_m, fade)
        out.update({"W": W, "N_m": N, "A_m": A, "m0": s0})
        return out
    _lo_m, _hi_m, lo_f, hi_f = _fit_bounds_2d(m_a, f_a)
    W, N, A, s0 = _grid_guess(f_a, t_a, lo_f, hi_f, fade)
    out.update({"W": W, "N_f": N, "A_f": A, "f0": s0})
    return out


def _pick_fade(own: dict[str, float], sub: dict[str, float], name: str, lo: float, hi: float) -> float:
    """One fade-scale initial value, clipped to the search range.

    The individual fit's value where finite, else the robust grid value,
    else the geometric mid of the range.

    Args:
        own: the individual fit's parameters (seconds).
        sub: the robust grid solution of the algorithm's own sweep.
        name: the fade's key ("m0" or "f0").
        lo: the fade's search lower bound (s).
        hi: the fade's search upper bound (s).

    Returns:
        float: the initial fade scale in seconds.

    """
    for source in (own, sub):
        try:
            value = float(source.get(name))
        except TypeError:
            value = float("nan")
        except ValueError:
            value = float("nan")
        if not math.isnan(value):
            return float(np.clip(value, lo, hi))
    return float(np.sqrt(lo * hi))


def _combined_initial_fades(
    prep: _CombinedPrep,
    p0_map: dict[str, dict[str, float]],
) -> list[tuple[float, ...]]:
    """Return the per-algorithm fade-scale initial values for the pooled solution.

    Args:
        prep: the prepared inputs of the combined fit.
        p0_map: per algorithm, the individual fit's parameters (seconds).

    Returns:
        list: per algorithm, (m0, f0) for "2d" and (s0,) for "1d-*", in
        seconds, clipped to their search ranges.

    """
    fades: list[tuple[float, ...]] = []
    head = _combined_layout(prep.kind)[0]
    for a in prep.order:
        own = p0_map.get(a, {})
        sub = _own_guess(prep.kind, prep.m[prep.alg == a], prep.f[prep.alg == a], prep.t[prep.alg == a], prep.fade)
        if prep.kind == "2d":
            fades.append(
                (
                    _pick_fade(own, sub, "m0", prep.lo_b[head + 2], prep.hi_b[head + 2]),
                    _pick_fade(own, sub, "f0", prep.lo_b[head + 3], prep.hi_b[head + 3]),
                )
            )
        else:
            name = "m0" if prep.kind == "1d-malloc" else "f0"
            fades.append((_pick_fade(own, sub, name, prep.lo_b[head + 1], prep.hi_b[head + 1]),))
    return fades


def _term_seed(own: dict[str, float], name: str, fallback: float) -> float:
    """One absorbed-delay initial value: the individual fit's, 0-floored.

    Args:
        own: the individual fit's parameters (seconds).
        name: the delay's key ("A_m" or "A_f").
        fallback: the pooled least-squares value on absence or NaN.

    Returns:
        float: the 0-floored initial value.

    """
    try:
        value = float(own.get(name))
    except TypeError:
        value = fallback
    except ValueError:
        value = fallback
    if math.isnan(value):
        value = fallback
    return max(0.0, value)


def _fit_residual_terms(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    kind: str,
    data: tuple | np.ndarray,
    r: np.ndarray,
    p0: tuple[float, ...],
    bounds: tuple[float, ...],
    fade: str = BEST_FADE,
) -> tuple[list[float] | None, str | None]:
    """Fit one algorithm's saturation terms on the shared-parameter residuals.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        data: the algorithm's delay data, (m, f) for "2d" and the varying
        delay otherwise, in seconds.
        r: the residuals after the shared parameters are subtracted (s).
        p0: the initial (A..., s0...) values, in model order.
        bounds: the fade-scale search ranges.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        tuple: (the fitted values, None), or (None, the error message) on
        non-convergence.

    """
    shape = _fade(fade)
    if kind == "2d":

        def model_2d(x: tuple | np.ndarray, A_m: float, m0: float, A_f: float, f0: float) -> np.ndarray:
            mm, ff = x
            return A_m * shape(mm / m0) + A_f * shape(ff / f0)

        model = model_2d
        lo = [0.0, bounds[0], 0.0, bounds[2]]
        hi = [_INF, bounds[1], _INF, bounds[3]]
    else:

        def model_1d(sx: np.ndarray, A: float, s0: float) -> np.ndarray:
            return A * shape(sx / s0)

        model = model_1d
        lo = [0.0, bounds[0]]
        hi = [_INF, bounds[1]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _pcov = curve_fit(model, data, r, p0=list(p0), bounds=(lo, hi), maxfev=20000)
        return [float(v) for v in popt], None
    except (RuntimeError, ValueError) as err:
        return None, str(err).splitlines()[0]


def _stage1_costs(kind: str, p_stage1: list[float]) -> list[tuple[float, ...]]:
    """Return the pooled least-squares absorbed delays, per algorithm block.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        p_stage1: the stage-1 parameter vector.

    Returns:
        list: per block, the (unfloored) absorbed delay value(s).

    """
    head, stride = _combined_layout(kind)
    n_a = stride - (2 if kind == "2d" else 1)
    n = (len(p_stage1) - head) // stride
    return [tuple(p_stage1[head + k * stride : head + k * stride + n_a]) for k in range(n)]


def _combined_stage2(
    prep: _CombinedPrep,
    p_stage1: list[float],
    p0_map: dict[str, dict[str, float]],
    fades: list[tuple[float, ...]],
) -> tuple[list[float], list[str]]:
    """Refine the per-algorithm terms on the stage-1 pooled residuals.

    With the shared parameters subtracted, each algorithm's (A, s0)
    saturation terms are fit on its own points, starting from the individual
    fit's costs and the pooled solution's fades.

    Args:
        prep: the prepared inputs of the combined fit.
        p_stage1: the pooled least-squares parameter vector (stage 1).
        p0_map: per algorithm, the individual fit's parameters (seconds).
        fades: the per-algorithm fade-scale initial values.

    Returns:
        tuple: (the refined parameter vector -- the shared parameters
        plus the fitted blocks -- the per-algorithm warnings).

    """
    kind = prep.kind
    head, stride = _combined_layout(kind)
    W = p_stage1[0]
    p = list(p_stage1)
    notes: list[str] = []
    for k, a in enumerate(prep.order):
        idx = prep.indicators[k] == 1
        base = head + k * stride
        own = p0_map.get(a, {})
        if kind == "2d":
            p0 = (
                _term_seed(own, "A_m", p[base]),
                fades[k][0],
                _term_seed(own, "A_f", p[base + 1]),
                fades[k][1],
            )
            fitted, err = _fit_residual_terms(
                kind,
                (prep.m[idx], prep.f[idx]),
                prep.t[idx] - W - p_stage1[1] * prep.m[idx] - p_stage1[2] * prep.f[idx],
                p0,
                prep.fade_bounds,
                prep.fade,
            )
        else:
            s = prep.m[idx] if kind == "1d-malloc" else prep.f[idx]
            p0 = (_term_seed(own, "A_m" if kind == "1d-malloc" else "A_f", p[base]), fades[k][0])
            fitted, err = _fit_residual_terms(
                kind, s, prep.t[idx] - W - p_stage1[1] * s, p0, prep.fade_bounds, prep.fade
            )
        if fitted is None:
            p[base] = max(0.0, p[base])
            if kind == "2d":
                p[base + 1] = max(0.0, p[base + 1])
            notes.append(
                f"{a}: the saturation terms did not converge on the pooled "
                f"residuals ({err}); the pooled values are kept"
            )
            continue
        if kind == "2d":
            p[base : base + stride] = [fitted[0], fitted[2], fitted[1], fitted[3]]
        else:
            p[base : base + stride] = fitted
    return p, notes


def _clip_shared(kind: str, p: list[float], floor: float) -> list[float]:
    """Copy a parameter vector with its shared part clipped to its bound.

    The pooled least-squares solution is unbounded, while curve_fit
    requires its initial guess strictly inside the bounds.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        p: the parameter vector.
        floor: the lower bound for the shared parameters.

    Returns:
        list[float]: the vector with the shared part clipped at `floor`.

    """
    head = _combined_layout(kind)[0]
    p = list(p)
    p[:head] = [max(v, floor) for v in p[:head]]
    return p


def _combined_optimize(
    prep: _CombinedPrep,
    p0_vector: list[float],
    fallback: list[float],
) -> tuple[list[float], np.ndarray | None, list[str]]:
    """Run the final joint curve_fit of a combined fit.

    Args:
        prep: the prepared inputs of the combined fit.
        p0_vector: the initial parameter vector (the stage-2 solution).
        fallback: the robust parameter vector to report when curve_fit
        does not converge.

    Returns:
        tuple: (the fitted parameter vector, the parameter covariance or
        None, the warnings).

    """
    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                prep.model, prep.xdata, prep.t, p0=p0_vector, bounds=(prep.lo_b, prep.hi_b), maxfev=40000
            )
        return [float(v) for v in popt], pcov, notes
    except (RuntimeError, ValueError) as err:
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]})")
        notes.append("constrained fit unavailable; the staged pooled least-squares solution is reported")
        return list(fallback), None, notes


class CombinedSweep(NamedTuple):
    """The pooled delay-sweep data of one scenario, every algorithm.

    The per-point arrays must be in the same order as `algorithms`.
    """

    m_delays: pd.Series  # the malloc delays in nanoseconds
    f_delays: pd.Series  # the free delays in nanoseconds
    runtimes: pd.Series  # the runtimes in seconds
    algorithms: pd.Series  # the per-point algorithm labels


class _CombinedPrep(NamedTuple):
    """The validated, prepared inputs of one combined fit."""

    kind: str  # "2d", "1d-malloc", or "1d-free"
    m: np.ndarray  # pooled malloc delays (s)
    f: np.ndarray  # pooled free delays (s)
    t: np.ndarray  # pooled runtimes (s)
    alg: np.ndarray  # per-point algorithm labels
    order: list[str]  # the pooled algorithms' order
    indicators: list[np.ndarray]  # per-algorithm 0/1 indicator arrays
    lo_b: list[float]  # lower bounds, one per parameter
    hi_b: list[float]  # upper bounds, one per parameter
    eps: float  # the floor for the initial guess
    ss_tot: float  # the pooled total sum of squares
    xdata: tuple | np.ndarray  # the delay data for curve_fit
    fade_bounds: tuple[float, ...]  # the (lo, hi) range of each fade scale
    model: Callable[..., np.ndarray]  # the curve_fit model function
    fade: str  # the fade shape name (a member of `FADE_SHAPES`)


def _combined_fun(kind: str, indicators: list[np.ndarray], fade: str = BEST_FADE) -> Callable[..., np.ndarray]:
    """Return the curve_fit model function of one combined fit.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        indicators: per-algorithm 0/1 indicator arrays.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        Callable[..., np.ndarray]: the model runtime of the pooled points
        for a parameter vector.

    """

    def model(x: tuple | np.ndarray, *params: float) -> np.ndarray:
        return _combined_predict(kind, x, np.asarray(params, dtype=float), indicators, fade)

    return model


def _combined_prepare(sweep: CombinedSweep, order: Sequence[str] | None, fade: str = BEST_FADE) -> _CombinedPrep:
    """Validate and prepare the pooled data of one combined fit.

    Args:
        sweep: the pooled sweep data (every algorithm of one scenario).
        order: the pooled algorithms' order, or None for first
        appearance.
        fade: the fade shape name (a member of `FADE_SHAPES`).

    Returns:
        _CombinedPrep: the seconds-converted arrays, the normalized
        algorithm order, the parameter bounds, and the initial-guess
        constants.

    Raises:
        ValueError: if the arrays have unequal shapes, fewer than 3
        pooled points, or neither delay varies.

    """
    m = np.asarray(sweep.m_delays, dtype=float) * 1e-9  # ns -> s
    f = np.asarray(sweep.f_delays, dtype=float) * 1e-9
    t = np.asarray(sweep.runtimes, dtype=float)
    alg = np.array([str(a) for a in sweep.algorithms], dtype=object)
    if m.shape != f.shape or m.shape != t.shape or m.shape != alg.shape:
        msg = "delays, runtimes and algorithms must all have the same shape"
        raise ValueError(msg)
    if m.size < 3:
        msg = "need at least 3 data points"
        raise ValueError(msg)
    kind = _combined_kind(m, f)
    if not kind:
        msg = "fewer than 2 distinct delays in the pooled data"
        raise ValueError(msg)
    first_seen = list(dict.fromkeys(alg))
    order = _combined_order(order, first_seen)
    indicators = [(alg == a).astype(float) for a in order]
    lo_b, hi_b = _combined_bounds(kind, len(order), m, f)
    head = _combined_layout(kind)[0]
    return _CombinedPrep(
        kind,
        m,
        f,
        t,
        alg,
        order,
        indicators,
        lo_b,
        hi_b,
        1e-9 * max(1.0, float(np.max(np.abs(t)))),
        float(np.sum((t - t.mean()) ** 2)),
        (m, f) if kind == "2d" else (m if kind == "1d-malloc" else f),
        (lo_b[head + 2], hi_b[head + 2], lo_b[head + 3], hi_b[head + 3])
        if kind == "2d"
        else (lo_b[head + 1], hi_b[head + 1]),
        _combined_fun(kind, indicators, fade),
        fade,
    )


def _combined_corner_notes(
    kind: str,
    p: np.ndarray,
    order: list[str],
    guess_A: list[tuple[float, ...]],
    hi_b: list[float],
) -> list[str]:
    """Corner diagnostics of a combined fit, as in the individual fits.

    An absorbed delay the data wanted negative is floored at 0; a fade
    scale at the search cap leaves its A (and W) weakly constrained.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        p: the fitted parameter vector.
        order: the pooled algorithms' order.
        guess_A: per algorithm, the (unfloored) absorbed-delay initial values.
        hi_b: the upper bounds, one entry per parameter.

    Returns:
        list[str]: the diagnostics.

    """
    notes: list[str] = []
    head, stride = _combined_layout(kind)
    small = 1e-6 * max(abs(float(p[0])), 1e-9)
    for k, a in enumerate(order):
        base = head + k * stride
        if kind == "2d":
            if guess_A[k][0] < 0 and p[base] <= small:
                notes.append(f"unconstrained fit wanted A_malloc<0 ({a}); A_malloc (and f_malloc) constrained to 0")
            if guess_A[k][1] < 0 and p[base + 1] <= small:
                notes.append(f"unconstrained fit wanted A_free<0 ({a}); A_free (and f_free) constrained to 0")
            if p[base + 2] >= hi_b[base + 2] * 0.999:
                notes.append(f"m0 ({a}) reached the search cap: A_malloc and W are weakly constrained")
            if p[base + 3] >= hi_b[base + 3] * 0.999:
                notes.append(f"f0 ({a}) reached the search cap: A_free and W are weakly constrained")
        else:
            tag = "malloc" if kind == "1d-malloc" else "free"
            if guess_A[k][0] < 0 and p[base] <= small:
                notes.append(f"unconstrained fit wanted A_{tag}<0 ({a}); A_{tag} constrained to 0")
            if p[base + 1] >= hi_b[base + 1] * 0.999:
                notes.append(f"{tag[0]}0 ({a}) reached the search cap: A_{tag} and W are weakly constrained")
    return notes


def _combined_slope_notes(kind: str, p: np.ndarray) -> list[str]:
    """Compute the shared-slope diagnostics of a combined fit.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        p: the fitted parameter vector.

    Returns:
        list[str]: one note per non-positive shared parameter.

    """
    notes: list[str] = []
    if float(p[0]) <= 0:
        notes.append("non-positive baseline runtime: no runtime budget visible in this sweep")
    if kind in {"2d", "1d-malloc"} and float(p[1]) <= 0:
        notes.append("non-positive malloc slope: no allocation cost visible for the shared malloc delay")
    n_f_idx = 2 if kind == "2d" else 1
    if kind in {"2d", "1d-free"} and float(p[n_f_idx]) <= 0:
        notes.append("non-positive free slope: no free cost visible for the shared free delay")
    return notes


class _CombinedErrors(NamedTuple):
    """The standard-error closures of one combined fit."""

    param: Callable[[int], float]  # per-parameter standard error
    function: Callable[[Callable[[np.ndarray], float]], float]  # error of a function of the parameters


def _combined_errors(pcov: np.ndarray | None, p: np.ndarray) -> _CombinedErrors:
    """Compute the standard errors of a combined fit from its covariance.

    Args:
        pcov: the parameter covariance matrix, or None.
        p: the fitted parameter vector.

    Returns:
        _CombinedErrors: the per-parameter standard error and the
        chain-rule error of a function of the parameters, or constant
        NaN closures when the covariance is unavailable.

    """
    if pcov is None or not np.all(np.isfinite(np.asarray(pcov, dtype=float))):

        def err_of(_i: int) -> float:
            return float("nan")

        def f_err(_f_of_p: Callable[[np.ndarray], float]) -> float:
            return float("nan")

        return _CombinedErrors(err_of, f_err)
    cov = np.asarray(pcov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    sqrt_diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    def err_of(i: int) -> float:
        return float(sqrt_diag[i])

    def f_err(f_of_p: Callable[[np.ndarray], float]) -> float:
        g = np.zeros(p.size)
        for i in range(p.size):
            pp = p.copy()
            pm = p.copy()
            step = max(abs(p[i]) * 1e-6, 1e-12)
            pp[i] += step
            pm[i] -= step
            g[i] = (f_of_p(pp) - f_of_p(pm)) / (2 * step)
        return float(np.sqrt(max(float(g @ cov @ g), 0.0)))

    return _CombinedErrors(err_of, f_err)


def _combined_per_algorithm(
    kind: str,
    p: np.ndarray,
    W: float,
    order: list[str],
    errors: _CombinedErrors,
) -> dict[str, dict[str, float]]:
    """Every pooled algorithm's absorbed delays, fade scales, and slack ratios.

    Args:
        kind: "2d", "1d-malloc", or "1d-free".
        p: the fitted parameter vector.
        W: the shared baseline runtime in seconds.
        order: the pooled algorithms' order.
        errors: the standard-error closures of the parameter vector.

    Returns:
        dict: per algorithm, A_malloc/A_free (s), m0/f0 (s), the slack ratios
        (absorbed delay / zero-delay runtime), and their standard errors (NaN where inapplicable).

    """
    head, stride = _combined_layout(kind)
    err_of = errors.param
    f_err = errors.function
    per_algorithm: dict[str, dict[str, float]] = {}
    for k, a in enumerate(order):
        base = head + k * stride
        if kind == "2d":
            A_m, A_f = float(p[base]), float(p[base + 1])
            T0 = W + A_m + A_f

            def f_malloc_of(q: np.ndarray, b: int = base) -> float:
                total = q[0] + q[b] + q[b + 1]
                return q[b] / total if total > EPS_S else 0.0

            def f_free_of(q: np.ndarray, b: int = base) -> float:
                total = q[0] + q[b] + q[b + 1]
                return q[b + 1] / total if total > EPS_S else 0.0

            per_algorithm[a] = {
                "A_malloc": A_m,
                "A_malloc_err": err_of(base),
                "A_free": A_f,
                "A_free_err": err_of(base + 1),
                "m0": float(p[base + 2]),
                "m0_err": err_of(base + 2),
                "f0": float(p[base + 3]),
                "f0_err": err_of(base + 3),
                "f_malloc": A_m / T0 if T0 > EPS_S else 0.0,
                "f_malloc_err": f_err(f_malloc_of),
                "f_free": A_f / T0 if T0 > EPS_S else 0.0,
                "f_free_err": f_err(f_free_of),
            }
            continue
        if kind == "1d-malloc":
            A, s0 = float(p[base]), float(p[base + 1])
            T0 = W + A

            def f_malloc_of_1d(q: np.ndarray, b: int = base) -> float:
                total = q[0] + q[b]
                return q[b] / total if total > EPS_S else 0.0

            per_algorithm[a] = {
                "A_malloc": A,
                "A_malloc_err": err_of(base),
                "A_free": float("nan"),
                "A_free_err": float("nan"),
                "m0": s0,
                "m0_err": err_of(base + 1),
                "f0": float("nan"),
                "f0_err": float("nan"),
                "f_malloc": A / T0 if T0 > EPS_S else 0.0,
                "f_malloc_err": f_err(f_malloc_of_1d),
                "f_free": float("nan"),
                "f_free_err": float("nan"),
            }
            continue
        A, s0 = float(p[base]), float(p[base + 1])
        T0 = W + A

        def f_free_of_1d(q: np.ndarray, b: int = base) -> float:
            total = q[0] + q[b]
            return q[b] / total if total > EPS_S else 0.0

        per_algorithm[a] = {
            "A_malloc": float("nan"),
            "A_malloc_err": float("nan"),
            "A_free": A,
            "A_free_err": err_of(base),
            "m0": float("nan"),
            "m0_err": float("nan"),
            "f0": s0,
            "f0_err": err_of(base + 1),
            "f_malloc": float("nan"),
            "f_malloc_err": float("nan"),
            "f_free": A / T0 if T0 > EPS_S else 0.0,
            "f_free_err": f_err(f_free_of_1d),
        }
    return per_algorithm


def fit_combined(
    sweep: CombinedSweep,
    order: Sequence[str] | None = None,
    p0: dict[str, dict[str, float]] | None = None,
    fade: str = BEST_FADE,
) -> dict:
    """Fit one (machine, setup, grid) scenario over all of its algorithms.

    The shared parameters -- W (the baseline runtime) and the malloc/free
    call counts N_malloc and N_free -- are fit once on the pooled data of
    every pooled algorithm, while each algorithm keeps its own absorbed
    delays (A_malloc, A_free) and fade scales (m0, f0): the two-operation
    allocation model of `model_2d` with a per-algorithm (A, s0) pair, or the
    1-D reduction when only one delay varies in the pooled data. The fit
    is staged -- a pooled least-squares solution for the shared
    parameters, a per-algorithm fit of the saturation terms on the pooled
    residuals, and a final joint curve_fit started from those values --
    so the result does not depend on the initial guess.

    Args:
        sweep: the pooled sweep data (every algorithm of the scenario).
        order: the pooled algorithms' order; defaults to first appearance.
        p0: per algorithm, the individual fit's parameters (keys W, N_m,
        N_f, A_m, A_f, m0, f0) in seconds as the initial guess; missing
        values fall back to a robust grid guess on the algorithm's own
        data.
        fade: the fade shape name (a member of `FADE_SHAPES`); defaults to
        the selected best shape, the exponential.

    Returns:
        dict: the model kind, the pooled algorithms' order, the shared
        parameters (W, N_malloc, N_free) with their standard errors, the
        pooled r2, the per-algorithm absorbed delays, fade scales, and
        slack ratios (absorbed delay / zero-delay runtime) with their standard errors, the warnings, and
        the joint (fit_params, pcov) pair.

    """
    prep = _combined_prepare(sweep, order, fade)
    fades = _combined_initial_fades(prep, p0 or {})
    _ss_res, p_stage1 = _combined_robust(prep.kind, prep.xdata, prep.t, prep.indicators, fades, prep.fade)
    p_stage2, stage_notes = _combined_stage2(prep, p_stage1, p0 or {}, fades)
    fit_params, pcov, opt_notes = _combined_optimize(
        prep, _clip_shared(prep.kind, p_stage2, prep.eps), _clip_shared(prep.kind, p_stage2, 0.0)
    )
    notes = stage_notes + opt_notes
    p = np.asarray(fit_params, dtype=float)
    r2 = (
        1.0
        - float(np.sum((prep.t - _combined_predict(prep.kind, prep.xdata, p, prep.indicators, prep.fade)) ** 2))
        / prep.ss_tot
        if prep.ss_tot > 0
        else float("nan")
    )
    W = float(p[0])
    n_f_idx = 2 if prep.kind == "2d" else 1
    notes.extend(_combined_slope_notes(prep.kind, p))
    notes.extend(_combined_corner_notes(prep.kind, p, prep.order, _stage1_costs(prep.kind, p_stage1), prep.hi_b))
    errors = _combined_errors(pcov, p)
    return {
        "model": prep.kind,
        "order": list(prep.order),
        "W": W,
        "W_err": errors.param(0),
        "N_malloc": float(p[1]) if prep.kind in {"2d", "1d-malloc"} else float("nan"),
        "N_malloc_err": errors.param(1) if prep.kind in {"2d", "1d-malloc"} else float("nan"),
        "N_free": float(p[n_f_idx]) if prep.kind in {"2d", "1d-free"} else float("nan"),
        "N_free_err": errors.param(n_f_idx) if prep.kind in {"2d", "1d-free"} else float("nan"),
        "r2": r2,
        "per_algorithm": _combined_per_algorithm(prep.kind, p, W, prep.order, errors),
        "warnings": notes,
        "fit_params": fit_params,
        "pcov": pcov,
    }
