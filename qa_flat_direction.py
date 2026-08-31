"""Q1: the flat direction of the performance-model fit.

Characterization, the one-fewer-parameter question, and the options.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads output/results.h5 (runs, fits, fits/cov, shared_fit_cov) and re-fits as
needed. Prints:
  P1. per stored fit: cond(pcov), 4 smallest eigenpairs of pcov with
      parameter attribution, Jacobian null-residual of each (flatness),
      gauge-direction contrast (Jacobian residual + share in flat directions);
  P2. function-space dimension: local rank of the model map at the measured
      delays, gauge-orbit image at fixed delays (not a redundancy), and a
      3-parameter submodel lack-of-fit demonstration;
  P3. reparameterizations (T0-anchored, K= A*s0 products, eigen-coordinates):
      reparameterized pcovs and condition numbers;
  P4. options: fix the fade scales (linear fit, cond, f sensitivity),
      A = N*c_a constraint (5-param fit, cond, f), combined-fit pcov cond,
      profile-likelihood along the fade-scale direction.

Run: /tmp/opencode/venv/bin/python qa_flat_direction.py
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

RESULTS = "/workspace/output/results.h5"
EPS_S = 1e-12
NAMES = ["W", "N_m", "N_f", "A_m", "A_f", "m0", "f0"]


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
    """Evaluate the two-operation performance model at the packed parameters.

    T = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0).

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the parameters (W, N_m, N_f, A_m, A_f, m0, f0).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    W, Nm, Nf, Am, Af, m0, f0 = p
    return W + Nm * m + Nf * f + Am * m0 / (m + m0) + Af * f0 / (f + f0)


def jac_full(m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Build the 7-column Jacobian of model_full at the packed parameters p.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: the packed 7-parameter vector.

    Returns:
        np.ndarray: the Jacobian, one row per data point.

    """
    Am, Af = float(p[3]), float(p[4])
    m0, f0 = float(p[5]), float(p[6])
    J = np.empty((m.size, 7))
    J[:, 0] = 1.0
    J[:, 1] = m
    J[:, 2] = f
    J[:, 3] = m0 / (m + m0)
    J[:, 4] = f0 / (f + f0)
    J[:, 5] = Am * m / (m + m0) ** 2
    J[:, 6] = Af * f / (f + f0) ** 2
    return J


def gauge_tangents(p: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Build the gauge tangent vectors, d/d(d->0) of the exact per-arm gauge shifts.

    See critique_fit.gauge_block for the transformation.

    Args:
        p: the packed 7-parameter vector.

    Returns:
        tuple: (the malloc gauge tangent, the free gauge tangent).

    """
    Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p[1:])
    gm = np.array([Nm * m0, 0.0, 0.0, -Am, 0.0, m0, 0.0])
    gf = np.array([Nf * f0, 0.0, 0.0, 0.0, -Af, 0.0, f0])
    return gm, gf


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
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)
    return lo_m, hi_m, lo_f, hi_f


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
    """Return the robust grid-guess initial parameters.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        bounds: (lo_m, hi_m, lo_f, hi_f) fade-scale search window (s).

    Returns:
        list[float]: the 0-floored initial parameters (W, N_m, N_f, A_m,
        A_f, m0, f0).

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


def fmt_vec(v: Sequence[float], tol: float = 0.08) -> str:
    """Format the eigenvector as 'a*x + b*y' of its significant components.

    Args:
        v: the eigenvector.
        tol: the magnitude floor for a component to be printed.

    Returns:
        str: the signed component string, or "(all < tol)".

    """
    terms = []
    for name, c in zip(NAMES, v, strict=True):
        if abs(c) >= tol:
            terms.append(f"{c:+.2f}*{name}")
    return " ".join(terms) if terms else "(all < tol)"


def _decode_str_cols(df: pd.DataFrame) -> None:
    """Decode the bytes string columns (machine, setup, algorithm) in place.

    Args:
        df: the table to decode.

    """
    for col in ("machine", "setup", "algorithm"):
        df[col] = [v.decode() if isinstance(v, bytes) else v for v in df[col]]


def _group_entry(runs: pd.DataFrame, row: pd.Series, f: h5py.File) -> dict:
    """Build one group's entry dict from its fits row.

    Args:
        runs: all runs.
        row: the group's fits row.
        f: the open results file (for the stored pcov and fit params).

    Returns:
        dict: name, key, m, f, t, pcov, p, row.

    """
    key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
    sub = runs[
        (runs["machine"] == key[0])
        & (runs["setup"] == key[1])
        & (runs["algorithm"] == key[2])
        & (runs["x"] == key[3])
        & (runs["y"] == key[4])
    ]
    sub = sub[sub["z"].isna()] if np.isnan(key[5]) else sub[sub["z"] == key[5]]
    sub = sub.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime_s"])
    m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
    fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
    t = sub["runtime_s"].to_numpy(float)
    grid = f"{int(key[3])}x{int(key[4])}" + (f"x{int(key[5])}" if not np.isnan(key[5]) else "")
    path = f"fits/cov/{key[0]}/{key[1]}/{key[2]}/{grid}"
    pcov = f[path + "/pcov"][:]
    fit_params = f[path + "/fit_params"][:]
    zs = int(key[5]) if not np.isnan(key[5]) else 0
    return {
        "name": f"{key[0]} {key[1][:4]} {key[2][:4]} {key[3]:.0f}x{key[4]:.0f}x{zs}",
        "key": key,
        "m": m,
        "f": fv,
        "t": t,
        "pcov": pcov,
        "p": fit_params,
        "row": row,
    }


def load_groups() -> tuple[list[dict], list[tuple[str, np.ndarray]]]:
    """Load the per-group data and the shared combined pcovs.

    Returns:
        tuple: (the per-group dicts, the shared (name, pcov) pairs).

    """
    f = h5py.File(RESULTS, "r")
    runs = read_table(f, "runs")
    fits = read_table(f, "fits")
    for df in (runs, fits):
        _decode_str_cols(df)
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")
    groups = [_group_entry(runs, row, f) for _, row in fits.iterrows()]
    shared = [
        (f"{k}/{k2}/{k3}", f["shared_fit_cov"][k][k2][k3]["pcov"][:])
        for k in f["shared_fit_cov"]
        for k2 in f["shared_fit_cov"][k]
        for k3 in f["shared_fit_cov"][k][k2]
    ]
    f.close()
    return groups, shared


def _p1_eigpairs(
    eig: np.ndarray,
    evec: np.ndarray,
    order: np.ndarray,
    info_mat: np.ndarray,
    J: np.ndarray,
) -> None:
    """Print the pcov eigenpairs, flattest first.

    Args:
        eig: the pcov eigenvalues.
        evec: the pcov eigenvectors (columns).
        order: the flattest-first eigenvalue order.
        info_mat: the information matrix J'J/sig2.
        J: the Jacobian.

    """
    for rank, k in enumerate(order):
        v = evec[:, k]
        info = float(v @ info_mat @ v)
        jv = J @ v
        print(
            f"{'':44s}   [{rank}] eig={eig[k]:10.3e} s^2 (1-sigma={np.sqrt(max(eig[k], 0)):8.3g} s) "
            f"info={info:9.3e}  max|Jv|={np.abs(jv).max():8.3g} s   {fmt_vec(v)}"
        )


def _p1_gauge(J: np.ndarray, p: np.ndarray, evec: np.ndarray, order: np.ndarray) -> None:
    """Print the gauge-direction contrast against the two flattest directions.

    Args:
        J: the Jacobian.
        p: the stored 7-parameter fit.
        evec: the pcov eigenvectors (columns).
        order: the flattest-first eigenvalue order.

    """
    gm, gf = gauge_tangents(p)
    v1, v2 = evec[:, order[0]], evec[:, order[1]]

    def flatfrac(gv: np.ndarray) -> float:
        gn = gv / np.linalg.norm(gv)
        return float(np.hypot(np.dot(v1, gn), np.dot(v2, gn)))

    jgm = J @ gm
    jgf = J @ gf
    print(
        f"{'':44s} gauge: max|J g_m|={np.abs(jgm).max():.3g} s (N_m*m0={p[1] * p[5]:.3g} s), "
        f"max|J g_f|={np.abs(jgf).max():.3g} s; share in 2 flattest: "
        f"{flatfrac(gm):.3f}/{flatfrac(gf):.3f} (0 = pinned, 1 = flat)"
    )


def _p1_group(g: dict) -> None:
    """Print one group's P1 line, eigenpairs, and gauge contrast.

    Args:
        g: the loaded group.

    """
    m, fv, t, p, pcov = g["m"], g["f"], g["t"], g["p"], g["pcov"]
    J = jac_full(m, fv, p)
    sig2 = float(np.sum((t - model_full(m, fv, p)) ** 2)) / max(t.size - 7, 1)
    info_mat = J.T @ J / sig2
    cov = 0.5 * (pcov + pcov.T)
    cond_p = float(np.linalg.cond(cov))
    eig, evec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]  # flattest first
    d1s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    diagstr = "  ".join(f"{nm}={v:.1g}" for nm, v in zip(NAMES, d1s, strict=True))
    print(f"{g['name']:44s} {t.size:4d} {cond_p:9.1e} {np.sqrt(sig2):7.2f} | diag 1-sigma (s): {diagstr}")
    _p1_eigpairs(eig, evec, order, info_mat, J)
    _p1_gauge(J, p, evec, order)


def part1(groups: list[dict]) -> None:
    """Print block P1: per stored fit, the pcov flat directions.

    Args:
        groups: the loaded groups.

    """
    print("=" * 100)
    print("P1. per stored fit: cond(pcov), flat directions (largest pcov eigenvalues)")
    print("=" * 100)
    print("info(v) = v' J'J v / sig2 = information in sigma^2 units along unit direction v.")
    print("info ~ 100 = 10 sigma; info ~ 1 = 1 sigma; info << 1 = the data do not see v (flat).")
    print("pcov eigenvectors are printed with the LARGEST pcov eigenvalue first (flattest).")
    print()
    print(f"{'group':44s} {'n':>4s} {'cond(pcov)':>9s} {'sig(s)':>7s} | all 7 pcov eigenpairs (flattest first)")
    for g in groups:
        _p1_group(g)
    print()
    print("The gauge tangents g_m/g_f are the d->0 derivatives of the exact gauge shifts;")
    print("their max|Jg| is O(N*s0) (huge) -> the gauge MOVES the curve at the fixed measured")
    print("delays; its share in the two flattest pcov directions is the critique block-6 'flat'")
    print("fraction (0.00 in all groups: the gauge is not the fit's flat direction).")


def _p2_rank(groups: list[dict]) -> None:
    """Print the local rank of the 2-D model's Jacobian per group.

    Args:
        groups: the loaded groups.

    """
    print("(a) 2D model: local rank of the 7-column Jacobian at the stored params (per group):")
    for g in groups:
        J = jac_full(g["m"], g["f"], g["p"])
        r = np.linalg.matrix_rank(J, tol=1e-8 * max(J.shape))
        print(f"    {g['name']:44s} rank = {r}")


def _p2_crux(g: dict) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Compute the 1-D crux on one group's malloc arm: rank, invariants, orbit.

    Args:
        g: the group (the first one).

    Returns:
        tuple: (the arm delays s, the arm runtimes y, W, N, A, s0, the arm
        mask).

    """
    arm = (g["m"] > 0) & (g["f"] == 0)
    s, y = g["m"][arm], g["t"][arm]
    W = float(g["p"][0])
    Nm = float(g["p"][1])
    Am = float(g["p"][3])
    m0 = float(g["p"][5])
    J1 = np.vstack([np.ones_like(s), s, m0 / (s + m0), Am * s / (s + m0) ** 2]).T
    r1 = np.linalg.matrix_rank(J1, tol=1e-8 * max(J1.shape))
    # gauge invariants under the exact transformation, d = 0.3:
    d = 0.3
    inv = {
        "N": (Nm, Nm),
        "K=A*s0": (Am * m0, (Am / (1 + d)) * (m0 * (1 + d))),
        "B=W-N*s0": (W - Nm * m0, (W + Nm * m0 * d) - Nm * (m0 * (1 + d))),
    }
    ref = model_1d_at(s, W, Nm, Am, m0)
    orw = 0.0
    for d_ in np.linspace(-0.5, 0.5, 21):
        ga = model_1d_at(s, W + Nm * m0 * d_, Nm, Am / (1 + d_), m0 * (1 + d_))
        orw = max(orw, float(np.max(np.abs(ga - ref))))
    print(
        f"\n(b) 1D crux, {g['name']} malloc arm ({arm.sum()} points, "
        f"s in [{s.min() * 1e9:.0f}, {s.max() * 1e9:.0f}] ns, stored m0={m0 * 1e9:.3g} ns):"
    )
    print(f"    rank of the 4-column Jacobian [1, s, s0/(s+s0), A s/(s+s0)^2] = {r1}  -> the 4-parameter")
    print("        family has local dimension 4 at these delays: (W, N, A, s0) are all needed.")
    print(
        "    gauge invariants N, K=A*s0, B=W-N*s0 under d=0.3: "
        + ", ".join(f"{k} {v[0]:.6e} -> {v[1]:.6e}" for k, v in inv.items())
    )
    print(f"    gauge orbit at FIXED delays: max |T - T_gauge| = {orw:.4g} s over d in [-0.5, 0.5]")
    print("        (= ~ N*m0*|d| : the gauge shifts the curve horizontally by s0*d, i.e. vertically")
    print("          by N*s0*d at fixed s). So the gauge is a redundancy of the model FORM (it")
    print("          re-describes the same curve from a different delay origin), NOT of the fit: with")
    print("          the measured delays fixed, each gauge value is a DIFFERENT function.")
    return s, y, W, Nm, Am, m0, arm


def _p2_fit4(
    s: np.ndarray,
    y: np.ndarray,
    init: tuple[float, float, float, float],
    arm: np.ndarray,
) -> tuple[float | None, float | None]:
    """Fit the 4-parameter 1-D arm model and print the result.

    Args:
        s: the arm delays in seconds.
        y: the arm runtimes in seconds.
        init: the (W, N, A, s0) seed for the fit.
        arm: the arm mask (for the run count).

    Returns:
        tuple: (the 4-parameter SSR, its 1-sigma), both None when the fit
        did not converge.

    """
    W, Nm, Am, m0 = init
    lo = max(0.05 * s[s > 0].min(), EPS_S)
    hi = 0.5 * (s.max() - s.min())
    try:
        p4, _ = _fit(
            lambda sx, W_, N_, A_, s0_: W_ + N_ * sx + A_ * s0_ / (sx + s0_),
            s,
            y,
            [max(W, EPS_S), max(Nm, EPS_S), max(Am, EPS_S), m0],
            ([0.0, 0.0, 0.0, lo], [np.inf, np.inf, np.inf, hi]),
        )
        ssr4 = float(np.sum((y - model_1d_at(s, *p4)) ** 2))
        sig4 = np.sqrt(ssr4 / max(arm.sum() - 4, 1))
        W4, N4, A4, s04 = (float(v) for v in p4)
        print(f"    4-param arm fit: W'={W4:.3g} N={N4:.6g} A={A4:.3g} s0={s04 * 1e9:.3g} ns (sig={sig4:.2f} s)")
    except RuntimeError, ValueError:
        ssr4, sig4 = None, None
    return ssr4, sig4


def _p2_invariants(s: np.ndarray, W: float, Nm: float, Am: float, m0: float) -> None:
    """Show that the 3 gauge invariants do not span the 1-D family.

    Holding (N, K=A*s0, B=W-N*s0) fixed and varying s0 moves the curve
    (s0 is the 4th function degree of freedom).

    Args:
        s: the arm delays in seconds.
        W: the stored W (s).
        Nm: the stored N_malloc.
        Am: the stored A_malloc (s).
        m0: the stored malloc fade scale (s).

    """
    K, B = Am * m0, W - Nm * m0
    spread = 0.0
    for fac in (0.1, 0.3, 1.0, 3.0, 10.0):
        s0x = m0 * fac
        A_x = K / s0x
        W_x = B + Nm * s0x
        ga = W_x + Nm * s + A_x * s0x / (s + s0x)
        ref = W + Nm * s + Am * m0 / (s + m0)
        spread = max(spread, float(np.max(np.abs(ga - ref))))
    print(f"    holding the 3 gauge invariants (N, K=A*s0={K:.3g}, B=W-N*s0={B:.3g}) fixed and")
    print(f"    varying s0 in [0.1x, 10x] stored: max |T - T_ref| on the arm = {spread:.3g} s")
    print("    -> the 3 invariants span a 3-D subfamily; s0 is a 4th function degree of freedom,")
    print("       so no 3-parameter (gauge-invariant-only) model spans the 4-D family.")


def _p2_fits3(s: np.ndarray, y: np.ndarray, m0: float, ssr4: float | None, sig4: float | None) -> None:
    """Print the dSSR of the 3-parameter fits with s0 fixed externally.

    Args:
        s: the arm delays in seconds.
        y: the arm runtimes in seconds.
        m0: the stored malloc fade scale (s).
        ssr4: the 4-parameter arm SSR (or None).
        sig4: the 4-parameter arm 1-sigma (or None).

    """
    print("    dropping s0 (fix it externally): 3-param linear fits, dSSR vs the 4-param arm optimum:")
    if ssr4 is not None:
        for fac in (0.1, 1.0, 10.0):
            s0x = m0 * fac
            X = np.vstack([np.ones_like(s), s, s0x / (s + s0x)]).T
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            dssr = float(np.sum((y - X @ sol) ** 2)) - ssr4
            print(
                f"        s0 x{fac:<4g} dSSR = {dssr:9.3g} s^2 = {dssr / sig4**2:8.2f} sigma^2   "
                f"(A = {sol[2]:8.3g} s, f = A/(W'+A) = {100 * sol[2] / (sol[0] + sol[2]):6.2f}%)"
            )


def part2(groups: list[dict]) -> None:
    """Print block P2: the function-space dimension at the fixed delays.

    Args:
        groups: the loaded groups.

    """
    print()
    print("=" * 100)
    print("P2. function-space dimension at the FIXED experimental delays")
    print("=" * 100)
    _p2_rank(groups)
    s, y, W, Nm, Am, m0, arm = _p2_crux(groups[0])
    ssr4, sig4 = _p2_fit4(s, y, (W, Nm, Am, m0), arm)
    _p2_invariants(s, W, Nm, Am, m0)
    _p2_fits3(s, y, m0, ssr4, sig4)
    print()
    print("interpretation: at fixed delays the model map is 7-D (2D) / 4-D (1D) and the gauge is not in")
    print("its kernel (it changes the function by O(N*s0)). No algebraic reparameterization with one")
    print("fewer parameter spans the same functions: a 6-parameter (3-parameter in 1D) family is a")
    print("proper subset. The dSSR above is the DATA's information about the dropped parameter: it is")
    print("weak (that is the near-degeneracy, a data limitation) but nonzero (the parameter is not a")
    print("gauge, so it cannot be reparameterized away).")


def model_1d_at(s: np.ndarray, W: float, N: float, A: float, s0: float) -> np.ndarray:
    """Evaluate the 1-D performance model T(s) = W + N*s + A*s0/(s+s0).

    Args:
        s: the delays in seconds.
        W: the baseline term (s).
        N: the slope.
        A: the saturation cost (s).
        s0: the fade scale (s).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    return W + N * s + A * s0 / (s + s0)


def _d2_matrix(p: np.ndarray) -> np.ndarray:
    """Build the R_K reparameterization Jacobian dq2/dp.

    q2 = (T0, N_m, N_f, A_m, K_m, A_f, K_f) with K = A*s0.

    Args:
        p: the stored 7-parameter fit.

    Returns:
        np.ndarray: the 7x7 chain-rule matrix.

    """
    D2 = np.zeros((7, 7))
    D2[0] = [1.0, 0, 0, 1.0, 1.0, 0, 0]  # dT0
    D2[1] = [0, 1, 0, 0, 0, 0, 0]  # dN_m
    D2[2] = [0, 0, 1, 0, 0, 0, 0]  # dN_f
    D2[3] = [0, 0, 0, 1.0, 0, 0, 0]  # dA_m
    D2[4] = [0, 0, 0, p[5], 0, p[3], 0]  # dK_m = m0 dA_m + A_m dm0
    D2[5] = [0, 0, 0, 0, 1.0, 0, 0]  # dA_f
    D2[6] = [0, 0, 0, 0, p[6], 0, p[4]]  # dK_f
    return D2


def _p3_conds(cov: np.ndarray, info_mat: np.ndarray, D2: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute the five condition numbers of one P3 row.

    Args:
        cov: the symmetrized stored pcov.
        info_mat: the information matrix J'J/sig2.
        D2: the R_K chain-rule matrix.

    Returns:
        tuple: (cond orig, cond R_T0, cond R_K, cond R_eig, cond(J'J)).

    """
    D1 = np.eye(7)
    D1[0] = [1.0, 0, 0, 1.0, 1.0, 0, 0]  # T0 = W + A_m + A_f
    _, evecs = np.linalg.eigh(info_mat)
    De = evecs.T
    cond_o = float(np.linalg.cond(cov))
    cond1 = np.linalg.cond(0.5 * (D1 @ cov @ D1.T + (D1 @ cov @ D1.T).T))
    cond2 = float(np.linalg.cond(D2 @ cov @ D2.T)) if np.linalg.matrix_rank(D2) == 7 else float("inf")
    conde = float(np.linalg.cond(De @ cov @ De.T))
    condI = float(np.linalg.eigvalsh(info_mat).max() / np.linalg.eigvalsh(info_mat).min())
    return cond_o, cond1, cond2, conde, condI


def _p3_row(g: dict) -> None:
    """Print one group's P3 reparameterization condition numbers.

    Args:
        g: the loaded group.

    """
    p, pcov = g["p"], g["pcov"]
    J = jac_full(g["m"], g["f"], p)
    sig2 = float(np.sum((g["t"] - model_full(g["m"], g["f"], p)) ** 2)) / max(g["t"].size - 7, 1)
    info_mat = J.T @ J / sig2
    cov = 0.5 * (pcov + pcov.T)
    D2 = _d2_matrix(p)
    cond_o, cond1, cond2, conde, condI = _p3_conds(cov, info_mat, D2)
    print(f"{g['name']:44s} {cond_o:9.1e} {cond1:10.1e} {cond2:10.1e} {conde:9.1e} {condI:10.1e}")


def _p3_flat_dirs(g: dict, n2: list[str]) -> None:
    """Print the flattest 3 R_K directions of one group (if D2 is full rank).

    Args:
        g: the loaded group.
        n2: the R_K parameter names.

    """
    p, pcov = g["p"], g["pcov"]
    D2 = _d2_matrix(p)
    if np.linalg.matrix_rank(D2) != 7:
        return
    cov2 = D2 @ (0.5 * (pcov + pcov.T)) @ D2.T
    e2, v2 = np.linalg.eigh(cov2)
    print(f"\nR_K flat directions, {g['name']} (flattest 3 in R_K coordinates):")
    for k in np.argsort(e2)[::-1][:3]:
        vecstr = " ".join(f"{c:+.2f}*{n}" for n, c in zip(n2, v2[:, k], strict=True) if abs(c) >= 0.08)
        print(f"  eig={e2[k]:.2e} s^2 (1-sigma={np.sqrt(max(e2[k], 0)):.3g} s)  {vecstr}")


def part3(groups: list[dict]) -> None:
    """Print block P3: reparameterizations and the conditioning.

    Args:
        groups: the loaded groups.

    """
    print()
    print("=" * 100)
    print("P3. reparameterizations: does a better parameterization fix the conditioning?")
    print("=" * 100)
    print("orig:  p  = (W, N_m, N_f, A_m, A_f, m0, f0)")
    print("R_T0:  q1 = (T0, N_m, N_f, A_m, A_f, m0, f0),  T0 = W + A_m + A_f")
    print("R_K:   q2 = (T0, N_m, N_f, A_m, K_m, A_f, K_f), K_m = A_m*m0, K_f = A_f*f0")
    print("R_eig: q3 = V^T p with V the eigenvectors of J^T J (the information principal axes)")
    print()
    print(f"{'group':44s} {'cond orig':>9s} {'cond R_T0':>10s} {'cond R_K':>10s} {'cond eig':>9s} {'cond(JtJ)':>10s}")
    for g in groups:
        _p3_row(g)
    print()
    print("R_eig diagonalizes the information (pcov_R_eig = diag(sigma^2/lambda_i)); cond(orig) = cond(R_eig)")
    print("always (orthogonal rotation). Non-orthogonal R_K can lower the numeric condition number")
    print("further by aligning parameter axes with the data-sensitive combinations (K = A*s0 is what")
    print("the large-delay data actually see) -- but NO reparameterization changes the information")
    print("lambda_i (in sigma^2 units): the sub-sigma^2 flat directions are a data fact, not a")
    print("coordinate artifact. (R_K singular at the A=0 corner -> inf.)")
    n2 = ["T0", "N_m", "N_f", "A_m", "K_m", "A_f", "K_f"]
    for g in (groups[0], groups[2], groups[10]):
        _p3_flat_dirs(g, n2)


def _p4_fades_row(
    m: np.ndarray,
    fv: np.ndarray,
    t: np.ndarray,
    mm0: float,
    ff0: float,
) -> tuple[float, np.ndarray, float, float]:
    """One fixed-fade linear 5-parameter fit of one group.

    Args:
        m: the malloc delays in seconds.
        fv: the free delays in seconds.
        t: the runtimes in seconds.
        mm0: the fixed malloc fade scale (s).
        ff0: the fixed free fade scale (s).

    Returns:
        tuple: (cond(XtX), the 1-sigma diagonals, f_m, f_f).

    """
    X = np.vstack([np.ones_like(m), m, fv, mm0 / (m + mm0), ff0 / (fv + ff0)]).T
    sol, *_ = np.linalg.lstsq(X, t, rcond=None)
    ssr = float(np.sum((t - X @ sol) ** 2))
    Wl = float(sol[0])
    Aml = float(sol[3])
    Afl = float(sol[4])
    T0 = Wl + Aml + Afl
    fm = 100 * Aml / T0 if T0 > EPS_S else 0.0
    ff = 100 * Afl / T0 if T0 > EPS_S else 0.0
    XtX = X.T @ X
    sig5 = np.sqrt(max(ssr, 1e-30) / max(t.size - 5, 1))
    c5 = sig5 * np.linalg.inv(XtX)
    return float(np.linalg.cond(XtX)), sig5 * np.sqrt(np.clip(np.diag(c5), 0, None)), fm, ff


def _p4_fades(groups: list[dict]) -> None:
    """Section A: fix the fade scales externally (linear 5-parameter fit).

    Args:
        groups: the loaded groups.

    """
    print()
    print("--- A. fix the fade scales externally (linear 5-param fit) ---")
    print(
        "T = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0) with m0,f0 FIXED -> exact lstsq in (W,N_m,N_f,A_m,A_f)"
    )
    print(
        f"{'group':44s} {'cond(XtX)':>9s} {'1sig(W)':>8s} {'1sig(A_m)':>9s} {'1sig(A_f)':>9s} "
        f"{'f_m %':>7s} {'f_f %':>7s} | fades x0.1: f_m%  x10: f_m%   (cost of the external s0 choice)"
    )
    for g in groups:
        m, fv, t = g["m"], g["f"], g["t"]
        m0, f0 = float(g["p"][5]), float(g["p"][6])
        res = {}
        for label, (mm0, ff0) in {"stored": (m0, f0), "x0.1": (0.1 * m0, 0.1 * f0), "x10": (10 * m0, 10 * f0)}.items():
            res[label] = _p4_fades_row(m, fv, t, mm0, ff0)
        c0, s0arr, fm0, ff0 = res["stored"]
        print(
            f"{g['name']:44s} {c0:9.1e} {s0arr[0]:8.2f} {s0arr[3]:9.3g} {s0arr[4]:9.3g} {fm0:7.2f} {ff0:7.2f} | "
            f"      {res['x0.1'][2]:7.2f}   {res['x10'][2]:7.2f}"
        )


def _native_model(x: tuple, cm: float, cf: float, p: Sequence[float]) -> float:
    """Evaluate the constrained model T = W + N_m*m + N_f*f + N_m*c_m*m0/(m+m0) + N_f*c_f*f0/(f+f0).

    Args:
        x: the (m, f) delays in seconds.
        cm: the malloc native cost per call (s).
        cf: the free native cost per call (s).
        p: the parameters (W, N_m, N_f, m0, f0).

    Returns:
        float: the model runtime (s).

    """
    W_, Nmm, Nff, mm0, ff0 = p
    mm, ff = x
    return W_ + Nmm * mm + Nff * ff + Nmm * cm * mm0 / (mm + mm0) + Nff * cf * ff0 / (ff + ff0)


def _p4_flattest(
    x: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    popt: np.ndarray,
    cm: float,
    cf: float,
) -> float:
    """Compute the flattest information direction of the constrained model.

    Args:
        x: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        popt: the constrained fit's parameters (W, N_m, N_f, m0, f0).
        cm: the malloc native cost per call (s).
        cf: the free native cost per call (s).

    Returns:
        float: the smallest eigenvalue of J5'J5/sig2 (1 = 1 sigma).

    """
    m, fv = x
    N5m, N5f, m05, f05 = (float(v) for v in popt[1:])
    J5 = np.empty((t.size, 5))
    J5[:, 0] = 1.0
    J5[:, 1] = m + cm * m05 / (m + m05)
    J5[:, 2] = fv + cf * f05 / (fv + f05)
    J5[:, 3] = N5m * cm * m / (m + m05) ** 2
    J5[:, 4] = N5f * cf * fv / (fv + f05) ** 2
    sig5 = float(np.sum((t - _native_model((m, fv), cm, cf, popt)) ** 2)) / max(t.size - 5, 1)
    I5 = J5.T @ J5 / sig5
    return float(np.linalg.eigvalsh(I5).min())


def _p4_native_fit(
    x: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    p: list[float],
    cc: tuple[float, float],
    hi: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """Fit the constrained 5-parameter model of section B.

    Args:
        x: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        p: the stored 7-parameter fit (floats).
        cc: the (cm, cf) native costs per call (s).
        hi: the (malloc, free) fade-scale upper bounds (s).

    Returns:
        tuple: (cond(pcov5), dSSR vs the stored fit, f_m, f_f, flattest
        info).

    """
    m, fv = x
    cm, cf = cc
    popt, pcov5 = _fit(
        lambda x_, W_, Nmm, Nff, mm0, ff0: _native_model(x_, cm, cf, (W_, Nmm, Nff, mm0, ff0)),
        (m, fv),
        t,
        [p[0], p[1], p[2], p[5], p[6]],
        ([0.0, 0.0, 0.0, EPS_S, EPS_S], [np.inf, np.inf, np.inf, hi[0], hi[1]]),
    )
    c5 = float(np.linalg.cond(0.5 * (pcov5 + pcov5.T)))
    T05 = float(popt[0]) + float(popt[1]) * cm + float(popt[2]) * cf
    fm5 = 100 * float(popt[1]) * cm / T05 if T05 > EPS_S else 0.0
    ff5 = 100 * float(popt[2]) * cf / T05 if T05 > EPS_S else 0.0
    dSSR = float(np.sum((t - _native_model((m, fv), cm, cf, popt)) ** 2)) - float(
        np.sum((t - model_full(m, fv, p)) ** 2)
    )
    flat5 = _p4_flattest(x, t, popt, cm, cf)
    return c5, dSSR, fm5, ff5, flat5


def _p4_native_row(g: dict) -> None:
    """Print one group's section-B constrained-fit line.

    Args:
        g: the loaded group.

    """
    m, fv, t = g["m"], g["f"], g["t"]
    p = [float(v) for v in g["p"]]
    cm, cf = p[3] / max(p[1], 1e-9), p[4] / max(p[2], 1e-9)
    _, hi_m, _, hi_f = bounds_of(m, fv)
    try:
        c5, dSSR, fm5, ff5, flat5 = _p4_native_fit((m, fv), t, p, (cm, cf), (hi_m, hi_f))
    except (RuntimeError, ValueError) as err:
        print(f"{g['name']:44s} did not converge: {str(err).splitlines()[0]}")
        return
    print(
        f"{g['name']:44s} {c5:11.1e} {flat5:13.3e} {dSSR:9.3g} {fm5:7.2f} {ff5:7.2f} "
        f"{100 * p[3] / (p[0] + p[3] + p[4]):12.2f}"
    )


def _p4_native(groups: list[dict]) -> None:
    """Section B: the A = N*c_a constraint (native cost from a microbenchmark).

    Args:
        groups: the loaded groups.

    """
    print()
    print("--- B. A = N*c_a constraint (native cost from an independent microbenchmark) ---")
    print("T = W + N_m*m + N_f*f + N_m*c_m*m0/(m+m0) + N_f*c_f*f0/(f+f0): 5 free params (W,N_m,N_f,m0,f0)")
    print("c_m,c_f stand-ins here = stored implied A/N (a real run needs the independent microbenchmark)")
    print("flattest-info = smallest eigenvalue of J5'J5/sig2 of the constrained model (1 = 1 sigma)")
    print(
        f"{'group':44s} {'cond(pcov5)':>11s} {'flattest-info':>13s} {'dSSR':>9s} {'f_m %':>7s} "
        f"{'f_f %':>7s} {'stored f_m %':>12s}"
    )
    for g in groups:
        _p4_native_row(g)


def _p4_combined(shared: list[tuple[str, np.ndarray]]) -> None:
    """Section C: the combined fit across algorithms (stored pcovs).

    Args:
        shared: the shared (name, pcov) pairs.

    """
    print()
    print("--- C. combined fit across algorithms (shared W, N): stored 11x11 pcovs ---")
    sh_names = ["W", "N_m", "N_f"]
    for k in range(2):  # per-algorithm block order: (A_m, A_f, m0, f0)
        sh_names += [f"A_m{k}", f"A_f{k}", f"m0{k}", f"f0{k}"]
    for name, pc in shared:
        c = 0.5 * (pc + pc.T)
        e, v = np.linalg.eigh(c)
        npar = pc.shape[0]
        names = sh_names[:npar] if npar == 11 else sh_names[:3] + [f"p{i}" for i in range(npar - 3)]
        vec = " ".join(f"{cc:+.2f}*{nm}" for nm, cc in zip(names, v[:, 0], strict=True) if abs(cc) >= 0.08)
        print(f"{name:40s} cond={np.linalg.cond(c):10.1e}  smallest eig={e[0]:.2e}  eigenvector: {vec}")
    print("(smallest eigen ~ 0 in the 11x11 = a flat direction survives pooling across algorithms)")


def _p4_profile_fixed(
    which: str,
    grid: np.ndarray,
    x: tuple[np.ndarray, np.ndarray],
    ctx: tuple[np.ndarray, np.ndarray, float],
    hi: tuple[float, float],
) -> np.ndarray:
    """Fix one parameter on a grid, refit the other 6: the dSSR profile.

    Args:
        which: "m0" (fix the malloc fade scale) or "A_m" (fix A_malloc).
        grid: the values to fix the parameter at.
        x: the (m, f) delay arrays in seconds.
        ctx: (the runtimes, the stored fit, the stored SSR).
        hi: the (malloc, free) fade-scale upper bounds (s).

    Returns:
        np.ndarray: the (fixed value, dSSR vs the stored fit) profile.

    """
    m, fv = x
    t, p, ssr7 = ctx
    hi_m, hi_f = hi
    out: list[tuple[float, float]] = []
    for val in grid:
        if which == "m0":
            p0 = [p[0], p[1], p[2], max(p[3], 1e-6), p[4], p[6]]
            lo = [0.0, 0.0, 0.0, 0.0, 0.0, EPS_S]
            hi_b = [np.inf, np.inf, np.inf, np.inf, np.inf, hi_f]
            popt, _ = _fit(
                lambda x, W_, Nmm, Nff, Aml, Afl, ff0: model_full(
                    x[0], x[1], (W_, Nmm, Nff, Aml, Afl, float(val), ff0)
                ),
                (m, fv),
                t,
                p0,
                (lo, hi_b),
            )
            full_p = (popt[0], popt[1], popt[2], popt[3], popt[4], float(val), popt[5])
        else:  # fix A_m only, refit (W, N_m, N_f, A_f, m0, f0) -> the (W, A_m) trade-off
            p0 = [p[0], p[1], p[2], p[4], max(p[5], EPS_S), p[6]]
            lo = [0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S]
            hi_b = [np.inf, np.inf, np.inf, np.inf, hi_m, hi_f]
            popt, _ = _fit(
                lambda x, W_, Nmm, Nff, Afl, m0l, ff0: model_full(
                    x[0], x[1], (W_, Nmm, Nff, float(val), Afl, m0l, ff0)
                ),
                (m, fv),
                t,
                p0,
                (lo, hi_b),
            )
            full_p = (popt[0], popt[1], popt[2], float(val), popt[3], popt[4], popt[5])
        d = float(np.sum((t - model_full(m, fv, full_p)) ** 2)) - ssr7
        out.append((float(val), d))
    return np.array(out)


def _p4_m0_window(pm: np.ndarray, p: np.ndarray, sig2: float) -> None:
    """Print the m0 profile's 1-sigma and 4-sigma windows.

    Args:
        pm: the m0 dSSR profile (grid values, dSSR).
        p: the stored 7-parameter fit.
        sig2: the per-residual variance of the stored fit.

    """
    ok = np.isfinite(pm[:, 1])
    in1 = ok & (pm[:, 1] <= sig2)
    in4 = ok & (pm[:, 1] <= 4 * sig2)
    lo1 = pm[in1, 0].min() if in1.any() else float("nan")
    hi1 = pm[in1, 0].max() if in1.any() else float("nan")
    lo4 = pm[in4, 0].min() if in4.any() else float("nan")
    hi4 = pm[in4, 0].max() if in4.any() else float("nan")
    pmin = pm[ok, 0][np.argmin(pm[ok, 1])] * 1e9
    print(
        f"  fix m0:  1-sigma window [{lo1:.3g}, {hi1:.3g}] ns;  4-sigma window "
        f"[{lo4:.3g}, {hi4:.3g}] ns; "
        f"stored m0 = {p[5] * 1e9:.3g} ns; profile minimum at m0 = {pmin:.3g} ns (dSSR={pm[ok, 1].min():.3g} s^2)"
    )


def _p4_fm_at(p: np.ndarray, a: float) -> float:
    """f_malloc at A_m = a, with the stored T0 = W + A_m + A_f.

    The refit at that A_m would shift W and A_f, but T0 is pinned, so the
    stored T0 is the approximation used here.

    Args:
        p: the stored 7-parameter fit.
        a: the A_malloc value (s).

    Returns:
        float: 100 * a / T0.

    """
    return 100 * a / (float(p[0]) + float(p[3]) + float(p[4]))


def _p4_am_window(pa: np.ndarray, p: np.ndarray, sig2: float) -> None:
    """Print the A_m profile's 1-sigma and 4-sigma windows.

    Args:
        pa: the A_m dSSR profile (grid values, dSSR).
        p: the stored 7-parameter fit.
        sig2: the per-residual variance of the stored fit.

    """
    oka = np.isfinite(pa[:, 1])
    ina1 = oka & (pa[:, 1] <= sig2)
    ina4 = oka & (pa[:, 1] <= 4 * sig2)
    lo1 = pa[ina1, 0].min() if ina1.any() else float("nan")
    hi1 = pa[ina1, 0].max() if ina1.any() else float("nan")
    lo4 = pa[ina4, 0].min() if ina4.any() else float("nan")
    hi4 = pa[ina4, 0].max() if ina4.any() else float("nan")
    print(
        f"  fix A_m: 1-sigma window [{lo1:.3g}, {hi1:.3g}] s;  4-sigma window "
        f"[{lo4:.3g}, {hi4:.3g}] s; stored A_m = {p[3]:.3g} s"
    )
    lo_w = _p4_fm_at(p, float(pa[ina1, 0].min())) if ina1.any() else float("nan")
    hi_w = _p4_fm_at(p, float(pa[ina1, 0].max())) if ina1.any() else float("nan")
    print(
        f"  => f_malloc in the 1-sigma A_m window: [{lo_w:.2f}%, "
        f"{hi_w:.2f}%]  (stored f_malloc = {_p4_fm_at(p, float(p[3])):.2f}%)"
    )


def _p4_profile(groups: list[dict]) -> None:
    """Section D: profile-likelihood along the flat directions.

    Args:
        groups: the loaded groups.

    """
    print()
    print("--- D. profile-likelihood along the flat directions (quantify, not eliminate) ---")
    g = next(x for x in groups if x["name"].startswith("hal Kelv Flat 256x128x128"))
    m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
    lo_m, hi_m, _, hi_f = bounds_of(m, fv)
    ssr7 = float(np.sum((t - model_full(m, fv, p)) ** 2))
    sig2 = ssr7 / (t.size - 7)
    print(f"group: {g['name']} (n={t.size}); fix one parameter on a grid, refit the other 6, dSSR:")
    grid_m = np.logspace(np.log10(lo_m), np.log10(hi_m), 21)
    pm = _p4_profile_fixed("m0", grid_m, (m, fv), (t, p, ssr7), (hi_m, hi_f))
    _p4_m0_window(pm, p, sig2)
    Am_hi = max(10 * p[3], 1.0)
    pa = _p4_profile_fixed("A_m", np.linspace(0.0, Am_hi, 31), (m, fv), (t, p, ssr7), (hi_m, hi_f))
    _p4_am_window(pa, p, sig2)
    print("  (this is the delta-SSR profile the review's critique_fit.profile_A computes; it QUANTIFIES the")
    print("   flat direction: the window is set by the data, and is orders of magnitude wider than the")
    print("   Gaussian pcov error when the direction is genuinely under-resolved)")


def _p4_seed_fit(
    x: tuple[np.ndarray, np.ndarray],
    t: np.ndarray,
    p0: list[float],
    hi: tuple[float, float],
) -> tuple[float, float, float, float, float, float, float] | None:
    """One seeded full 7-parameter fit.

    Args:
        x: the (m, f) delay arrays in seconds.
        t: the runtimes in seconds.
        p0: the initial parameters.
        hi: the (malloc, free) fade-scale upper bounds (s).

    Returns:
        tuple | None: (ssr, A_m, A_f, m0, f0, f_m, f_f) of the converged
        fit, or None when it did not converge.

    """
    m, fv = x
    try:
        popt, _ = fit_full(m, fv, t, hi, p0)
    except RuntimeError, ValueError:
        return None
    Aml, Afl = float(popt[3]), float(popt[4])
    T0 = float(popt[0]) + Aml + Afl
    return (
        float(np.sum((t - model_full(m, fv, popt)) ** 2)),
        Aml,
        Afl,
        float(popt[5]),
        float(popt[6]),
        100 * Aml / T0,
        100 * Afl / T0,
    )


def _p4_seed_print(label: str, rows: list[tuple], ssr_best: float) -> None:
    """Print the fade-scale seed rows of one group.

    Args:
        label: the group's name.
        rows: the (seed, ssr, A_m, A_f, m0, f0, f_m, f_f) rows.
        ssr_best: the best (smallest) SSR of the seeds.

    """
    print(f"\n{label}")
    for name, ssr, Aml, Afl, m0_, f0_, fm_, ff_ in rows:
        d = ssr - ssr_best if np.isfinite(ssr) else float("nan")
        print(
            f"  seed {name:7s} SSR={ssr:9.1f} (d={d:7.1f})  A_m={Aml:8.2f} s A_f={Afl:8.2f} s  "
            f"m0={m0_ * 1e9:8.3g} ns f0={f0_ * 1e9:8.3g} ns  f_m={fm_:6.2f}% f_f={ff_:6.2f}%"
        )


def _p4_seed_rows(g: dict) -> None:
    """Print the fade-scale seed sensitivity of one group.

    Args:
        g: the loaded group.

    """
    m, fv, t = g["m"], g["f"], g["t"]
    lo_m, hi_m, lo_f, hi_f = bounds_of(m, fv)
    guess = grid_guess(m, fv, t, (lo_m, hi_m, lo_f, hi_f))
    ssr_best = np.inf
    rows: list[tuple] = []
    for name, (sm0, sf0) in {
        "grid": (guess[5], guess[6]),
        "bottom": (lo_m, lo_f),
        "mid": (float(np.sqrt(lo_m * hi_m)), float(np.sqrt(lo_f * hi_f))),
    }.items():
        p0 = [
            max(guess[0], EPS_S),
            max(guess[1], EPS_S),
            max(guess[2], EPS_S),
            max(guess[3], EPS_S),
            max(guess[4], EPS_S),
            sm0,
            sf0,
        ]
        vals = _p4_seed_fit((m, fv), t, p0, (hi_m, hi_f))
        if vals is None:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            rows.append((name, *vals))
            ssr_best = min(ssr_best, vals[0])
    _p4_seed_print(g["name"], rows, ssr_best)


def _p4_seeds(groups: list[dict]) -> None:
    """Section E: the global flat direction (multi-modality in the seeds).

    Args:
        groups: the loaded groups.

    """
    print()
    print("--- E. global flat direction: multi-modality in the fade-scale seeds ---")
    print("full 7-param fit from different (m0,f0) seeds, same data/bounds (review block 3): each seed")
    print("lands in a local minimum; comparable SSR with very different f = the GLOBAL flat valley")
    for g in (
        x
        for x in groups
        if x["name"].startswith(
            ("hal Kelv Flat 128x128x128", "rosi Kelv Flat 256x128x128", "hal Kelv Flat 256x128x128")
        )
    ):
        _p4_seed_rows(g)


def _p4_grad(p: np.ndarray, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Compute the numerical central-difference gradient of fn at p (7 components).

    Args:
        p: the parameter point.
        fn: the scalar function of the parameters.

    Returns:
        np.ndarray: the 7-component gradient.

    """
    gg = np.zeros(7)
    for i in range(7):
        pp, pm = p.copy(), p.copy()
        step = max(abs(p[i]) * 1e-7, 1e-12)
        pp[i] += step
        pm[i] -= step
        gg[i] = (fn(pp) - fn(pm)) / (2 * step)
    return gg


def _p4_chain_row(g: dict) -> None:
    """Print one group's chain-rule errors line.

    Args:
        g: the loaded group.

    """
    p, pcov = g["p"], g["pcov"]
    cov = 0.5 * (pcov + pcov.T)
    T0 = float(p[0]) + float(p[3]) + float(p[4])
    gT = _p4_grad(p, lambda q: q[0] + q[3] + q[4])
    gm = _p4_grad(p, lambda q: q[3] / (q[0] + q[3] + q[4]))
    gf = _p4_grad(p, lambda q: q[4] / (q[0] + q[3] + q[4]))
    eT = float(np.sqrt(max(gT @ cov @ gT, 0.0)))
    em = float(np.sqrt(max(gm @ cov @ gm, 0.0)))
    ef = float(np.sqrt(max(gf @ cov @ gf, 0.0)))
    row = g["row"]
    ems = float(row["f_malloc_err"]) if np.isfinite(row["f_malloc_err"]) else float("nan")
    efs = float(row["f_free_err"]) if np.isfinite(row["f_free_err"]) else float("nan")
    print(f"{g['name']:44s} {eT / T0:12.2%} {em:10.2%} {ef:10.2%} {ems:14.2%} {efs:14.2%}")


def _p4_chain(groups: list[dict]) -> None:
    """Section F: the chain-rule errors of the identifiable combinations.

    Args:
        groups: the loaded groups.

    """
    print()
    print("--- F. what the data DO pin: chain-rule errors of the identifiable combinations ---")
    print("g' pcov g for T0 = W + A_m + A_f (the measured baseline) and f_m = A_m/T0, f_f = A_f/T0:")
    print(
        f"{'group':44s} {'1sig(T0)/T0':>12s} {'1sig(f_m)':>10s} {'1sig(f_f)':>10s} "
        f"{'stored f_m_err':>14s} {'stored f_f_err':>14s}"
    )
    for g in groups:
        _p4_chain_row(g)
    print("(T0 is pinned to a few % by the (0,0) baseline; the chain-rule f errors are the stored ones,")
    print(" i.e. the headline fraction inherits the (W,A) flat-direction uncertainty.)")


def part4(groups: list[dict], shared: list[tuple[str, np.ndarray]]) -> None:
    """Print block P4: the options for the flat direction.

    Args:
        groups: the loaded groups.
        shared: the shared (combined) pcovs.

    """
    print()
    print("=" * 100)
    print("P4. options")
    print("=" * 100)
    _p4_fades(groups)
    _p4_native(groups)
    _p4_combined(shared)
    _p4_profile(groups)
    _p4_seeds(groups)
    _p4_chain(groups)


def main() -> None:
    """Run all QA parts against output/results.h5 and print the results."""
    groups, shared = load_groups()
    part1(groups)
    part2(groups)
    part3(groups)
    part4(groups, shared)


if __name__ == "__main__":
    main()
