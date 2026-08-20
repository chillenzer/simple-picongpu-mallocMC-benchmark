"""Fit sleeptime-sweep runtimes to the constrained Amdahl allocation model.

Model
-----
Each of the N allocation calls on the (serial, host-side) critical path takes
its native cost c_a plus the imposed delay s, so

    T(s) = W + A + N*s,   A = N*c_a (native allocation time)

with W the runtime without any allocation cost. The native part is not needed
for large sleeptimes (negligible against the imposed delay) but acts as a
correction at small sleeptimes, so the sweep is fitted with

    T(s) = W + N*s + A*s0/(s+s0)             (Amdahl model)

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

Usage
-----
    import sys
    sys.path.insert(0, "analysis")
    from fit_allocations import fit_allocation_fraction, fit_sweep

    res   = fit_allocation_fraction(sleeptimes_ns, runtimes_s)
    table = fit_sweep(df)   # df from analyse_sleeptimes.parse_logs

or run `python3 analysis/fit_allocations.py` from the repository root to fit
every (example, grid) group in `output/hal-sleeptimes/run_*`.

`sleeptimes` are in nanoseconds, `runtimes` in seconds. If the native
per-allocation cost c_a (ns) is known, pass it: A = N*c_a is then used and
only (W, N, s0) are fitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

__all__ = ["fit_allocation_fraction", "fit_sweep", "main"]

GROUP_KEYS = ("setup", "x", "y", "z")
_INF = float("inf")


def _model(s, W, N, A, s0):
    return W + N * s + A * s0 / (s + s0)


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
            "W_err": W_e, "N_err": N_e, "A_err": A_e, "s0_err": s0_e, "f_err": f_e,
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
        return finish(W, N, A, float("nan"), None, None, f_free,
                      note="only 3 points: correction shape not identifiable; N and W from the two "
                           "largest, A the (floored) residual at the smallest sleeptime")

    lo_b = max(lo, 1e-12)
    hi_b = max(hi, lo_b * 1.5)
    try:
        if c_a is None:
            p0 = [max(gW, eps), max(gN, eps), max(gA, eps),
                  float(np.clip(gs0, lo_b, hi_b))]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(_model, s, t, p0=p0,
                                       bounds=([0.0, 0.0, 0.0, lo_b], [_INF, _INF, _INF, hi_b]),
                                       maxfev=20000)
            W, N, A, s0 = (float(v) for v in popt)
            if gA < -1e-6 * max(abs(gW), 1e-9) and A <= 1e-6 * max(abs(W), 1e-9):
                notes.append(
                    "unconstrained fit wanted A<0 (smallest-sleeptime runtime below the Amdahl "
                    "line); A constrained to 0 so f is floored at 0"
                )
            if s0 >= hi_b * 0.999:
                notes.append("s0 reached the search cap: the native correction does not clearly "
                             "fade within the sweep, so A and W (hence f) are weakly constrained")
            return finish(W, N, A, s0, pcov, list(popt), f_free)
        c = c_a * 1e-9
        f_ca = lambda p: p[1] * c / (p[0] + p[1] * c) if p[0] + p[1] * c > 1e-12 else 0.0
        p0 = [max(gW, eps), max(gN, eps), float(np.clip(gs0, lo_b, hi_b))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(lambda qq, W, N, s0: _model_c_a(qq, W, N, s0, c), s, t,
                                   p0=p0, bounds=([0.0, 0.0, lo_b], [_INF, _INF, hi_b]), maxfev=20000)
        W, N, s0 = (float(v) for v in popt)
        A = N * c
        if s0 >= hi_b * 0.999:
            notes.append("s0 reached the search cap: the native correction does not clearly "
                         "fade within the sweep, so A and W (hence f) are weakly constrained")
        return finish(W, N, A, s0, pcov, list(popt), f_ca)
    except (RuntimeError, ValueError) as err:
        # curve_fit failed to converge; report the robust linear solution.
        notes.append(f"curve_fit did not converge ({str(err).splitlines()[0]}); "
                     "using the robust linear solution")
        return finish(gW, gN, max(0.0, gA), gs0, None, None, f_free,
                      note="constrained fit unavailable; linear grid solution reported (A floored at 0)")


def fit_sweep(df: pd.DataFrame, c_a: float | None = None, configuration: str | None = None) -> pd.DataFrame:
    """Fit every (setup, x, y, z) group of a parsed sweep DataFrame.

    `df` is the output of `analyse_sleeptimes.parse_logs`; `configuration`,
    if given, restricts the fit to that configuration ("run-time" or
    "compile-time").
    """
    if configuration is not None:
        df = df[df["configuration"] == configuration]
    rows = []
    for key, grp in df.groupby(list(GROUP_KEYS), dropna=False):
        grp = grp.dropna(subset=["sleeptime", "runtime in s"])
        row = {**dict(zip(GROUP_KEYS, key)), "n_runs": len(grp)}
        if grp["sleeptime"].nunique() < 2:
            row.update({"N": np.nan, "W": np.nan, "A": np.nan, "f": np.nan, "f_err": np.nan,
                        "T0": np.nan, "r2": np.nan, "s0_ns": np.nan,
                        "note": "fewer than 2 distinct sleeptimes"})
            rows.append(row)
            continue
        try:
            res = fit_allocation_fraction(grp["sleeptime"], grp["runtime in s"], c_a=c_a)
        except ValueError as err:
            row.update({"N": np.nan, "W": np.nan, "A": np.nan, "f": np.nan, "f_err": np.nan,
                        "T0": np.nan, "r2": np.nan, "s0_ns": np.nan,
                        "note": str(err)})
            rows.append(row)
            continue
        row.update({k: res[k] for k in ("W", "N", "A", "T0", "f", "f_err", "r2")})
        row["s0_ns"] = np.nan if res["s0"] != res["s0"] else res["s0"] * 1e9
        row["note"] = "; ".join(res["warnings"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(GROUP_KEYS), na_position="last")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyse_sleeptimes as a

    full = a.parse_logs(sorted(a.LOG_PATHS))
    print(f"{len(full)} runs parsed\n")
    table = fit_sweep(full)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False, float_format=lambda v: f"{v:10.3g}"))
    ok = table[table["f"].between(0, 1, inclusive="neither")]
    if len(ok):
        print("\nAmdahl fraction of runtime spent in allocations (f = A/T0):")
        for _, r in ok.iterrows():
            err = "" if r["f_err"] != r["f_err"] else f" +/- {100 * r['f_err']:.1f}"
            print(
                f"  {r.setup:<16s} grid {int(r.x)}x{int(r.y)}"
                + (f"x{int(r.z)}" if pd.notna(r.z) else "")
                + f" : f = {100 * r.f:.1f}{err}%   (N = {r.N:.3e} allocations, W = {r.W:.2f} s, A = {r.A:.2f} s)"
            )
            if r["note"]:
                print(f"      note: {r.note}")


if __name__ == "__main__":
    main()
