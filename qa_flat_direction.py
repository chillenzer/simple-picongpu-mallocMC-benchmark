"""Q1: the flat direction of the allocation-model fit -- characterization,
the one-fewer-parameter question, and the options.

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

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

RESULTS = "/workspace/output/results.h5"
EPS_S = 1e-12
NAMES = ["W", "N_m", "N_f", "A_m", "A_f", "m0", "f0"]


def read_table(f, grp):
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


def model_full(m, f, W, Nm, Nf, Am, Af, m0, f0):
    return W + Nm * m + Nf * f + Am * m0 / (m + m0) + Af * f0 / (f + f0)


def jac_full(m, f, p):
    """7-column Jacobian of model_full at parameters p."""
    W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
    J = np.empty((m.size, 7))
    J[:, 0] = 1.0
    J[:, 1] = m
    J[:, 2] = f
    J[:, 3] = m0 / (m + m0)
    J[:, 4] = f0 / (f + f0)
    J[:, 5] = Am * m / (m + m0) ** 2
    J[:, 6] = Af * f / (f + f0) ** 2
    return J


def gauge_tangents(p):
    """d/d(d->0) of the exact per-arm gauge shifts (see critique_fit.gauge_block)."""
    W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
    gm = np.array([Nm * m0, 0.0, 0.0, -Am, 0.0, m0, 0.0])
    gf = np.array([Nf * f0, 0.0, 0.0, 0.0, -Af, 0.0, f0])
    return gm, gf


def bounds_of(m, f):
    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)
    return lo_m, hi_m, lo_f, hi_f


def grid_guess(m, f, t, bounds):
    lo_m, hi_m, lo_f, hi_f = bounds
    best = None
    for m0 in np.logspace(np.log10(lo_m), np.log10(hi_m), 25):
        for f0 in np.logspace(np.log10(lo_f), np.log10(hi_f), 25):
            X = np.vstack([np.ones_like(m), m, f, m0 / (m + m0), f0 / (f + f0)]).T
            sol, *_ = np.linalg.lstsq(X, t, rcond=None)
            ssr = float(np.sum((t - X @ sol) ** 2))
            if best is None or ssr < best[0]:
                best = (ssr, float(m0), float(f0), *[float(v) for v in sol])
    _ssr, m0, f0, W, Nm, Nf, Am, Af = best
    return [max(W, EPS_S), max(Nm, EPS_S), max(Nf, EPS_S), max(Am, EPS_S), max(Af, EPS_S), m0, f0]


def _fit(fn, xdata, t, p0, lo, hi):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(fn, xdata, t, p0=p0, bounds=(lo, hi), maxfev=40000)
    return popt, pcov


def fit_full(m, f, t, hi_m, hi_f, p0=None):
    def fn(x, W, Nm, Nf, Am, Af, m0, f0):
        return model_full(x[0], x[1], W, Nm, Nf, Am, Af, m0, f0)

    lo = [0.0, 0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S]
    hi = [np.inf] * 5 + [hi_m, hi_f]
    return _fit(fn, (m, f), t, p0 if p0 is not None else list(np.ones(7)), lo, hi)


def fmt_vec(v, tol=0.08):
    """Eigenvector as 'a*x + b*y' of its significant components."""
    terms = []
    for name, c in zip(NAMES, v):
        if abs(c) >= tol:
            terms.append(f"{c:+.2f}*{name}")
    return " ".join(terms) if terms else "(all < tol)"


def load_groups():
    f = h5py.File(RESULTS, "r")
    runs = read_table(f, "runs")
    fits = read_table(f, "fits")
    for df in (runs, fits):
        for col in ("machine", "setup", "algorithm"):
            df[col] = [v.decode() if isinstance(v, bytes) else v for v in df[col]]
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")
    groups = []
    for i, row in fits.iterrows():
        key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
        sub = runs[
            (runs["machine"] == key[0])
            & (runs["setup"] == key[1])
            & (runs["algorithm"] == key[2])
            & (runs["x"] == key[3])
            & (runs["y"] == key[4])
        ]
        if np.isnan(key[5]):
            sub = sub[sub["z"].isna()]
        else:
            sub = sub[sub["z"] == key[5]]
        sub = sub.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime_s"])
        m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
        fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
        t = sub["runtime_s"].to_numpy(float)
        grid = (
            f"{int(key[3])}x{int(key[4])}"
            + (f"x{int(key[5])}" if not np.isnan(key[5]) else "")
        )
        path = f"fits/cov/{key[0]}/{key[1]}/{key[2]}/{grid}"
        pcov = f[path + "/pcov"][:]
        fit_params = f[path + "/fit_params"][:]
        zs = int(key[5]) if not np.isnan(key[5]) else 0
        groups.append(
            {
                "name": f"{key[0]} {key[1][:4]} {key[2][:4]} {key[3]:.0f}x{key[4]:.0f}x{zs}",
                "key": key,
                "m": m,
                "f": fv,
                "t": t,
                "pcov": pcov,
                "p": fit_params,
                "row": row,
            }
        )
    shared = []
    for k in f["shared_fit_cov"]:
        for k2 in f["shared_fit_cov"][k]:
            for k3 in f["shared_fit_cov"][k][k2]:
                shared.append((f"{k}/{k2}/{k3}", f["shared_fit_cov"][k][k2][k3]["pcov"][:]))
    f.close()
    return groups, shared


def part1(groups):
    print("=" * 100)
    print("P1. per stored fit: cond(pcov), flat directions (largest pcov eigenvalues)")
    print("=" * 100)
    print("info(v) = v' J'J v / sig2 = information in sigma^2 units along unit direction v.")
    print("info ~ 100 = 10 sigma; info ~ 1 = 1 sigma; info << 1 = the data do not see v (flat).")
    print("pcov eigenvectors are printed with the LARGEST pcov eigenvalue first (flattest).")
    print()
    print(f"{'group':44s} {'n':>4s} {'cond(pcov)':>9s} {'sig(s)':>7s} | all 7 pcov eigenpairs (flattest first)")
    for g in groups:
        m, fv, t, p, pcov = g["m"], g["f"], g["t"], g["p"], g["pcov"]
        n = t.size
        J = jac_full(m, fv, p)
        pred = model_full(m, fv, *p)
        ssr = float(np.sum((t - pred) ** 2))
        sig2 = ssr / max(n - 7, 1)
        I = J.T @ J / sig2
        cov = 0.5 * (pcov + pcov.T)
        cond_p = float(np.linalg.cond(cov))
        eig, evec = np.linalg.eigh(cov)
        order = np.argsort(eig)[::-1]  # flattest first
        d1s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        diagstr = "  ".join(f"{nm}={v:.1g}" for nm, v in zip(NAMES, d1s))
        print(f"{g['name']:44s} {n:4d} {cond_p:9.1e} {np.sqrt(sig2):7.2f} | diag 1-sigma (s): {diagstr}")
        for rank, k in enumerate(order):
            v = evec[:, k]
            info = float(v @ I @ v)
            jv = J @ v
            print(
                f"{'':44s}   [{rank}] eig={eig[k]:10.3e} s^2 (1-sigma={np.sqrt(max(eig[k],0)):8.3g} s) "
                f"info={info:9.3e}  max|Jv|={np.abs(jv).max():8.3g} s   {fmt_vec(v)}"
            )
        # gauge contrast against the TRUE flat directions (2 largest pcov eigenvalues)
        gm, gf = gauge_tangents(p)
        v1, v2 = evec[:, order[0]], evec[:, order[1]]

        def flatfrac(gv):
            gn = gv / np.linalg.norm(gv)
            return float(np.hypot(np.dot(v1, gn), np.dot(v2, gn)))

        jgm = J @ gm
        jgf = J @ gf
        print(
            f"{'':44s} gauge: max|J g_m|={np.abs(jgm).max():.3g} s (N_m*m0={p[1]*p[5]:.3g} s), "
            f"max|J g_f|={np.abs(jgf).max():.3g} s; share in 2 flattest: "
            f"{flatfrac(gm):.3f}/{flatfrac(gf):.3f} (0 = pinned, 1 = flat)"
        )
    print()
    print("The gauge tangents g_m/g_f are the d->0 derivatives of the exact gauge shifts;")
    print("their max|Jg| is O(N*s0) (huge) -> the gauge MOVES the curve at the fixed measured")
    print("delays; its share in the two flattest pcov directions is the critique block-6 'flat'")
    print("fraction (0.00 in all groups: the gauge is not the fit's flat direction).")


def part2(groups):
    print()
    print("=" * 100)
    print("P2. function-space dimension at the FIXED experimental delays")
    print("=" * 100)
    # (a) 2D: local rank of the 7-column Jacobian in every group
    print("(a) 2D model: local rank of the 7-column Jacobian at the stored params (per group):")
    for g in groups:
        m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
        J = jac_full(m, fv, p)
        r = np.linalg.matrix_rank(J, tol=1e-8 * max(J.shape))
        print(f"    {g['name']:44s} rank = {r}")
    # (b) 1D crux on one group's malloc arm: T(s) = W + N s + A s0/(s+s0)
    g = groups[0]
    m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
    arm = (m > 0) & (fv == 0)
    s, y = m[arm], t[arm]
    W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
    J1 = np.vstack([np.ones_like(s), s, m0 / (s + m0), Am * s / (s + m0) ** 2]).T
    r1 = np.linalg.matrix_rank(J1, tol=1e-8 * max(J1.shape))
    g1 = np.array([Nm * m0, 0.0, -Am, m0])  # gauge tangent (dW, dN, dA, ds0)
    Jg1 = J1 @ g1
    # gauge invariants under the exact transformation, d = 0.3:
    d = 0.3
    Wp, Ap, m0p = W + Nm * m0 * d, Am / (1 + d), m0 * (1 + d)
    inv = {
        "N": (Nm, Nm),
        "K=A*s0": (Am * m0, Ap * m0p),
        "B=W-N*s0": (W - Nm * m0, Wp - Nm * m0p),
    }
    base1 = model_1d_at(s, W, Nm, Am, m0)
    orw = 0.0
    for d_ in np.linspace(-0.5, 0.5, 21):
        orw = max(orw, float(np.max(np.abs(model_1d_at(s, W + Nm * m0 * d_, Nm, Am / (1 + d_), m0 * (1 + d_)) - base1))))
    print(f"\n(b) 1D crux, {g['name']} malloc arm ({arm.sum()} points, s in [{s.min()*1e9:.0f}, {s.max()*1e9:.0f}] ns, stored m0={m0*1e9:.3g} ns):")
    print(f"    rank of the 4-column Jacobian [1, s, s0/(s+s0), A s/(s+s0)^2] = {r1}  -> the 4-parameter")
    print(f"        family has local dimension 4 at these delays: (W, N, A, s0) are all needed.")
    print(f"    gauge invariants N, K=A*s0, B=W-N*s0 under d=0.3: " +
          ", ".join(f"{k} {v[0]:.6e} -> {v[1]:.6e}" for k, v in inv.items()))
    print(f"    gauge orbit at FIXED delays: max |T - T_gauge| = {orw:.4g} s over d in [-0.5, 0.5]")
    print(f"        (= ~ N*m0*|d| : the gauge shifts the curve horizontally by s0*d, i.e. vertically")
    print(f"          by N*s0*d at fixed s). So the gauge is a redundancy of the model FORM (it")
    print(f"          re-describes the same curve from a different delay origin), NOT of the fit: with")
    print(f"          the measured delays fixed, each gauge value is a DIFFERENT function.")
    # (c) cost of dropping one parameter: proper nested comparison on the arm.
    # 4-param fit (W, N, A, s0) first, then 3-param fits with s0 fixed at 0.1x/1x/10x stored.
    lo = max(0.05 * s[s > 0].min(), EPS_S)
    hi = 0.5 * (s.max() - s.min())

    def fn4(sx, W_, N_, A_, s0_):
        return W_ + N_ * sx + A_ * s0_ / (sx + s0_)

    try:
        p4, _ = _fit(fn4, s, y, [max(W, EPS_S), max(Nm, EPS_S), max(Am, EPS_S), m0],
                     [0, 0, 0, lo], [np.inf, np.inf, np.inf, hi])
        ssr4 = float(np.sum((y - fn4(s, *p4)) ** 2))
        sig4 = np.sqrt(ssr4 / max(arm.sum() - 4, 1))
        W4, N4, A4, s04 = (float(v) for v in p4)
        print(f"    4-param arm fit: W'={W4:.3g} N={N4:.6g} A={A4:.3g} s0={s04*1e9:.3g} ns (sig={sig4:.2f} s)")
    except (RuntimeError, ValueError):
        ssr4, sig4 = None, None
    # (b2) the 3 gauge invariants (N, K=A*s0, B=W-N*s0) do NOT determine the function at fixed
    # delays: holding (N, K, B) fixed and varying s0 moves the curve (s0 is the 4th dof).
    K, B = Am * m0, W - Nm * m0
    spread = 0.0
    for fac in (0.1, 0.3, 1.0, 3.0, 10.0):
        s0x = m0 * fac
        A_x = K / s0x
        W_x = B + Nm * s0x
        spread = max(spread, float(np.max(np.abs((W_x + Nm * s + A_x * s0x / (s + s0x)) - (W + Nm * s + Am * m0 / (s + m0))))))
    print(f"    holding the 3 gauge invariants (N, K=A*s0={K:.3g}, B=W-N*s0={B:.3g}) fixed and")
    print(f"    varying s0 in [0.1x, 10x] stored: max |T - T_ref| on the arm = {spread:.3g} s")
    print(f"    -> the 3 invariants span a 3-D subfamily; s0 is a 4th function degree of freedom,")
    print(f"       so no 3-parameter (gauge-invariant-only) model spans the 4-D family.")
    print(f"    dropping s0 (fix it externally): 3-param linear fits, dSSR vs the 4-param arm optimum:")
    if ssr4 is not None:
        for fac in (0.1, 1.0, 10.0):
            s0x = m0 * fac
            X = np.vstack([np.ones_like(s), s, s0x / (s + s0x)]).T
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            dssr = float(np.sum((y - X @ sol) ** 2)) - ssr4
            print(f"        s0 x{fac:<4g} dSSR = {dssr:9.3g} s^2 = {dssr/sig4**2:8.2f} sigma^2   "
                  f"(A = {sol[2]:8.3g} s, f = A/(W'+A) = {100*sol[2]/(sol[0]+sol[2]):6.2f}%)")
    print()
    print("interpretation: at fixed delays the model map is 7-D (2D) / 4-D (1D) and the gauge is not in")
    print("its kernel (it changes the function by O(N*s0)). No algebraic reparameterization with one")
    print("fewer parameter spans the same functions: a 6-parameter (3-parameter in 1D) family is a")
    print("proper subset. The dSSR above is the DATA's information about the dropped parameter: it is")
    print("weak (that is the near-degeneracy, a data limitation) but nonzero (the parameter is not a")
    print("gauge, so it cannot be reparameterized away).")


def model_1d_at(s, W, N, A, s0):
    return W + N * s + A * s0 / (s + s0)


def part3(groups):
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
        m, fv, t, p, pcov = g["m"], g["f"], g["t"], g["p"], g["pcov"]
        J = jac_full(m, fv, p)
        pred = model_full(m, fv, *p)
        sig2 = float(np.sum((t - pred) ** 2)) / max(t.size - 7, 1)
        I = J.T @ J / sig2
        cov = 0.5 * (pcov + pcov.T)
        W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)

        def reparam(q):
            return np.linalg.cond(0.5 * (q @ cov @ q.T + (q @ cov @ q.T).T))

        D1 = np.eye(7)
        D1[0] = [1.0, 0, 0, 1.0, 1.0, 0, 0]  # T0 = W + A_m + A_f
        # R_K: q2 = (T0, N_m, N_f, A_m, K_m, A_f, K_f); dq2/dp by chain rule
        D2 = np.zeros((7, 7))
        D2[0] = [1.0, 0, 0, 1.0, 1.0, 0, 0]  # dT0
        D2[1] = [0, 1, 0, 0, 0, 0, 0]  # dN_m
        D2[2] = [0, 0, 1, 0, 0, 0, 0]  # dN_f
        D2[3] = [0, 0, 0, 1.0, 0, 0, 0]  # dA_m
        D2[4] = [0, 0, 0, m0, 0, Am, 0]  # dK_m = m0 dA_m + A_m dm0
        D2[5] = [0, 0, 0, 0, 1.0, 0, 0]  # dA_f
        D2[6] = [0, 0, 0, 0, f0, 0, Af]  # dK_f
        # eigen-coordinates
        evals, evecs = np.linalg.eigh(I)
        De = evecs.T
        cond_o = float(np.linalg.cond(cov))
        cond1 = reparam(D1)
        cond2 = float(np.linalg.cond(D2 @ cov @ D2.T)) if np.linalg.matrix_rank(D2) == 7 else float("inf")
        conde = float(np.linalg.cond(De @ cov @ De.T))
        condI = float(np.linalg.eigvalsh(I).max() / np.linalg.eigvalsh(I).min())
        print(f"{g['name']:44s} {cond_o:9.1e} {cond1:10.1e} {cond2:10.1e} {conde:9.1e} {condI:10.1e}")
    print()
    print("R_eig diagonalizes the information (pcov_R_eig = diag(sigma^2/lambda_i)); cond(orig) = cond(R_eig)")
    print("always (orthogonal rotation). Non-orthogonal R_K can lower the numeric condition number")
    print("further by aligning parameter axes with the data-sensitive combinations (K = A*s0 is what")
    print("the large-delay data actually see) -- but NO reparameterization changes the information")
    print("lambda_i (in sigma^2 units): the sub-sigma^2 flat directions are a data fact, not a")
    print("coordinate artifact. (R_K singular at the A=0 corner -> inf.)")
    n2 = ["T0", "N_m", "N_f", "A_m", "K_m", "A_f", "K_f"]
    for g in (groups[0], groups[2], groups[10]):
        m, fv, t, p, pcov = g["m"], g["f"], g["t"], g["p"], g["pcov"]
        W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
        D2 = np.zeros((7, 7))
        D2[0] = [1.0, 0, 0, 1.0, 1.0, 0, 0]
        D2[1] = [0, 1, 0, 0, 0, 0, 0]
        D2[2] = [0, 0, 1, 0, 0, 0, 0]
        D2[3] = [0, 0, 0, 1.0, 0, 0, 0]
        D2[4] = [0, 0, 0, m0, 0, Am, 0]
        D2[5] = [0, 0, 0, 0, 1.0, 0, 0]
        D2[6] = [0, 0, 0, 0, f0, 0, Af]
        if np.linalg.matrix_rank(D2) == 7:
            cov2 = D2 @ (0.5 * (pcov + pcov.T)) @ D2.T
            e2, v2 = np.linalg.eigh(cov2)
            print(f"\nR_K flat directions, {g['name']} (flattest 3 in R_K coordinates):")
            for k in np.argsort(e2)[::-1][:3]:
                vecstr = " ".join(f"{c:+.2f}*{n}" for n, c in zip(n2, v2[:, k]) if abs(c) >= 0.08)
                print(f"  eig={e2[k]:.2e} s^2 (1-sigma={np.sqrt(max(e2[k],0)):.3g} s)  {vecstr}")


def part4(groups, shared):
    print()
    print("=" * 100)
    print("P4. options")
    print("=" * 100)
    print()
    print("--- A. fix the fade scales externally (linear 5-param fit) ---")
    print("T = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0) with m0,f0 FIXED -> exact lstsq in (W,N_m,N_f,A_m,A_f)")
    print(f"{'group':44s} {'cond(XtX)':>9s} {'1sig(W)':>8s} {'1sig(A_m)':>9s} {'1sig(A_f)':>9s} {'f_m %':>7s} {'f_f %':>7s} | fades x0.1: f_m%  x10: f_m%   (cost of the external s0 choice)")
    for g in groups:
        m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
        W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
        ssr7 = float(np.sum((t - model_full(m, fv, *p)) ** 2))
        res = {}
        for label, (mm0, ff0) in {"stored": (m0, f0), "x0.1": (0.1 * m0, 0.1 * f0), "x10": (10 * m0, 10 * f0)}.items():
            X = np.vstack([np.ones_like(m), m, fv, mm0 / (m + mm0), ff0 / (fv + ff0)]).T
            sol, *_ = np.linalg.lstsq(X, t, rcond=None)
            ssr = float(np.sum((t - X @ sol) ** 2))
            Wl, Nml, Nfl, Aml, Afl = (float(v) for v in sol)
            T0 = Wl + Aml + Afl
            fm = 100 * Aml / T0 if T0 > EPS_S else 0.0
            ff = 100 * Afl / T0 if T0 > EPS_S else 0.0
            XtX = X.T @ X
            sig5 = np.sqrt(max(ssr, 1e-30) / max(t.size - 5, 1))
            c5 = sig5 * np.linalg.inv(X.T @ X)
            res[label] = (float(np.linalg.cond(XtX)), (sig5 * np.sqrt(np.clip(np.diag(c5), 0, None))), fm, ff)
        c0, s0arr, fm0, ff0 = res["stored"]
        print(
            f"{g['name']:44s} {c0:9.1e} {s0arr[0]:8.2f} {s0arr[3]:9.3g} {s0arr[4]:9.3g} {fm0:7.2f} {ff0:7.2f} | "
            f"      {res['x0.1'][2]:7.2f}   {res['x10'][2]:7.2f}"
        )
    print()
    print("--- B. A = N*c_a constraint (native cost from an independent microbenchmark) ---")
    print("T = W + N_m*m + N_f*f + N_m*c_m*m0/(m+m0) + N_f*c_f*f0/(f+f0): 5 free params (W,N_m,N_f,m0,f0)")
    print("c_m,c_f stand-ins here = stored implied A/N (a real run needs the independent microbenchmark)")
    print("flattest-info = smallest eigenvalue of J5'J5/sig2 of the constrained model (1 = 1 sigma)")
    print(f"{'group':44s} {'cond(pcov5)':>11s} {'flattest-info':>13s} {'dSSR':>9s} {'f_m %':>7s} {'f_f %':>7s} {'stored f_m %':>12s}")
    for g in groups:
        m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
        W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p)
        cm, cf = Am / max(Nm, 1e-9), Af / max(Nf, 1e-9)
        ssr7 = float(np.sum((t - model_full(m, fv, *p)) ** 2))

        def fn(x, W_, Nmm, Nff, mm0, ff0):
            mm, ff = x
            return W_ + Nmm * mm + Nff * ff + Nmm * cm * mm0 / (mm + mm0) + Nff * cf * ff0 / (ff + ff0)

        lo_m, hi_m, lo_f, hi_f = bounds_of(m, fv)
        try:
            popt, pcov5 = _fit(fn, (m, fv), t, [W, Nm, Nf, m0, f0], [0, 0, 0, EPS_S, EPS_S],
                               [np.inf, np.inf, np.inf, hi_m, hi_f])
            c5 = float(np.linalg.cond(0.5 * (pcov5 + pcov5.T)))
            W5, N5m, N5f, m05, f05 = (float(v) for v in popt)
            A5m, A5f = N5m * cm, N5f * cf
            T05 = W5 + A5m + A5f
            fm5 = 100 * A5m / T05 if T05 > EPS_S else 0.0
            ff5 = 100 * A5f / T05 if T05 > EPS_S else 0.0
            dSSR = float(np.sum((t - fn((m, fv), *popt)) ** 2)) - ssr7
            # flattest information direction of the constrained model
            J5 = np.empty((t.size, 5))
            J5[:, 0] = 1.0
            J5[:, 1] = m + cm * m05 / (m + m05)
            J5[:, 2] = fv + cf * f05 / (fv + f05)
            J5[:, 3] = N5m * cm * m / (m + m05) ** 2
            J5[:, 4] = N5f * cf * fv / (fv + f05) ** 2
            sig5 = float(np.sum((t - fn((m, fv), *popt)) ** 2)) / max(t.size - 5, 1)
            I5 = J5.T @ J5 / sig5
            flat5 = float(np.linalg.eigvalsh(I5).min())
            fm_store = 100 * float(p[3]) / (float(p[0]) + float(p[3]) + float(p[4]))
            print(f"{g['name']:44s} {c5:11.1e} {flat5:13.3e} {dSSR:9.3g} {fm5:7.2f} {ff5:7.2f} {fm_store:12.2f}")
        except (RuntimeError, ValueError) as err:
            print(f"{g['name']:44s} did not converge: {str(err).splitlines()[0]}")
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
        vec = " ".join(f"{cc:+.2f}*{nm}" for nm, cc in zip(names, v[:, 0]) if abs(cc) >= 0.08)
        print(f"{name:40s} cond={np.linalg.cond(c):10.1e}  smallest eig={e[0]:.2e}  eigenvector: {vec}")
    print("(smallest eigen ~ 0 in the 11x11 = a flat direction survives pooling across algorithms)")
    print()
    print("--- D. profile-likelihood along the flat directions (quantify, not eliminate) ---")
    g = [x for x in groups if x["name"].startswith("hal Kelv Flat 256x128x128")][0]
    m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
    lo_m, hi_m, lo_f, hi_f = bounds_of(m, fv)
    ssr7 = float(np.sum((t - model_full(m, fv, *p)) ** 2))
    sig2 = ssr7 / (t.size - 7)
    print(f"group: {g['name']} (n={t.size}); fix one parameter on a grid, refit the other 6, dSSR:")

    def profile_fixed(which, grid):
        out = []
        for val in grid:
            if which == "m0":

                def fn(x, W_, Nmm, Nff, Aml, Afl, ff0):
                    mm, ff = x
                    return model_full(mm, ff, W_, Nmm, Nff, Aml, Afl, val, ff0)

                p0 = [p[0], p[1], p[2], max(p[3], 1e-6), p[4], p[6]]
                lo, hi = [0, 0, 0, 0, 0, EPS_S], [np.inf, np.inf, np.inf, np.inf, np.inf, hi_f]
            else:  # fix A_m only, refit (W, N_m, N_f, A_f, m0, f0) -> the (W, A_m) trade-off

                def fn(x, W_, Nmm, Nff, Afl, m0l, ff0):
                    mm, ff = x
                    return model_full(mm, ff, W_, Nmm, Nff, val, Afl, m0l, ff0)

                p0 = [p[0], p[1], p[2], p[4], max(p[5], EPS_S), p[6]]
                lo, hi = [0, 0, 0, 0, EPS_S, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi_m, hi_f]
            popt, _ = _fit(fn, (m, fv), t, p0, lo, hi)
            d = float(np.sum((t - fn((m, fv), *popt)) ** 2)) - ssr7
            out.append((float(val), d))
        return np.array(out)

    pm = profile_fixed("m0", np.logspace(np.log10(lo_m), np.log10(hi_m), 21))
    ok = np.isfinite(pm[:, 1])
    in1 = ok & (pm[:, 1] <= sig2)
    in4 = ok & (pm[:, 1] <= 4 * sig2)
    print(f"  fix m0:  1-sigma window [{pm[in1, 0].min() if in1.any() else float('nan')*1e9:.3g}, "
          f"{pm[in1, 0].max() if in1.any() else float('nan')*1e9:.3g}] ns;  4-sigma window "
          f"[{pm[in4, 0].min() if in4.any() else float('nan')*1e9:.3g}, {pm[in4, 0].max() if in4.any() else float('nan')*1e9:.3g}] ns; "
          f"stored m0 = {p[5]*1e9:.3g} ns; profile minimum at m0 = {pm[ok, 0][np.argmin(pm[ok, 1])]*1e9:.3g} ns (dSSR={pm[ok, 1].min():.3g} s^2)")
    Am_hi = max(10 * p[3], 1.0)
    pa = profile_fixed("A_m", np.linspace(0.0, Am_hi, 31))
    oka = np.isfinite(pa[:, 1])
    ina1 = oka & (pa[:, 1] <= sig2)
    ina4 = oka & (pa[:, 1] <= 4 * sig2)

    def fm_at(a):
        # T0 from the refit at that A_m is needed; approximate with stored T0 (W + A_f also shifts, but T0 is pinned)
        return 100 * a / (float(p[0]) + float(p[3]) + float(p[4]))

    print(f"  fix A_m: 1-sigma window [{pa[ina1, 0].min() if ina1.any() else float('nan'):.3g}, "
          f"{pa[ina1, 0].max() if ina1.any() else float('nan'):.3g}] s;  4-sigma window "
          f"[{pa[ina4, 0].min() if ina4.any() else float('nan'):.3g}, {pa[ina4, 0].max() if ina4.any() else float('nan'):.3g}] s; "
          f"stored A_m = {p[3]:.3g} s")
    print(f"  => f_malloc in the 1-sigma A_m window: [{fm_at(pa[ina1, 0].min()) if ina1.any() else float('nan'):.2f}%, "
          f"{fm_at(pa[ina1, 0].max()) if ina1.any() else float('nan'):.2f}%]  (stored f_malloc = {fm_at(p[3]):.2f}%)")
    print("  (this is the delta-SSR profile the review's critique_fit.profile_A computes; it QUANTIFIES the")
    print("   flat direction: the window is set by the data, and is orders of magnitude wider than the")
    print("   Gaussian pcov error when the direction is genuinely under-resolved)")
    print()
    print("--- E. global flat direction: multi-modality in the fade-scale seeds ---")
    print("full 7-param fit from different (m0,f0) seeds, same data/bounds (review block 3): each seed")
    print("lands in a local minimum; comparable SSR with very different f = the GLOBAL flat valley")
    for g in [x for x in groups if x["name"].startswith(("hal Kelv Flat 128x128x128", "rosi Kelv Flat 256x128x128", "hal Kelv Flat 256x128x128"))]:
        m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
        lo_m, hi_m, lo_f, hi_f = bounds_of(m, fv)
        guess = grid_guess(m, fv, t, (lo_m, hi_m, lo_f, hi_f))
        ssr_best = np.inf
        rows = []
        for name, (sm0, sf0) in {"grid": (guess[5], guess[6]),
                                 "bottom": (lo_m, lo_f),
                                 "mid": (float(np.sqrt(lo_m * hi_m)), float(np.sqrt(lo_f * hi_f)))}.items():
            p0 = [max(guess[0], EPS_S), max(guess[1], EPS_S), max(guess[2], EPS_S),
                  max(guess[3], EPS_S), max(guess[4], EPS_S), sm0, sf0]
            try:
                popt, _ = fit_full(m, fv, t, hi_m, hi_f, p0=p0)
                ssr = float(np.sum((t - model_full(m, fv, *popt)) ** 2))
                W_, Nmm, Nff, Aml, Afl, m0_, f0_ = (float(v) for v in popt)
                T0_ = W_ + Aml + Afl
                rows.append((name, ssr, Aml, Afl, m0_, f0_, 100 * Aml / T0_, 100 * Afl / T0_))
                ssr_best = min(ssr_best, ssr)
            except (RuntimeError, ValueError):
                rows.append((name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        print(f"\n{g['name']}")
        for name, ssr, Aml, Afl, m0_, f0_, fm_, ff_ in rows:
            d = ssr - ssr_best if np.isfinite(ssr) else float("nan")
            print(f"  seed {name:7s} SSR={ssr:9.1f} (d={d:7.1f})  A_m={Aml:8.2f} s A_f={Afl:8.2f} s  "
                  f"m0={m0_*1e9:8.3g} ns f0={f0_*1e9:8.3g} ns  f_m={fm_:6.2f}% f_f={ff_:6.2f}%")
    print()
    print("--- F. what the data DO pin: chain-rule errors of the identifiable combinations ---")
    print("g' pcov g for T0 = W + A_m + A_f (the measured baseline) and f_m = A_m/T0, f_f = A_f/T0:")
    print(f"{'group':44s} {'1sig(T0)/T0':>12s} {'1sig(f_m)':>10s} {'1sig(f_f)':>10s} {'stored f_m_err':>14s} {'stored f_f_err':>14s}")
    for g in groups:
        p, pcov = g["p"], g["pcov"]
        cov = 0.5 * (pcov + pcov.T)
        W, _Nm, _Nf, Am, Af = (float(v) for v in p[:5])
        T0 = W + Am + Af

        def grad(fn):
            gg = np.zeros(7)
            for i in range(7):
                pp, pm = p.copy(), p.copy()
                step = max(abs(p[i]) * 1e-7, 1e-12)
                pp[i] += step
                pm[i] -= step
                gg[i] = (fn(pp) - fn(pm)) / (2 * step)
            return gg

        gT = grad(lambda q: q[0] + q[3] + q[4])
        gm = grad(lambda q: q[3] / (q[0] + q[3] + q[4]))
        gf = grad(lambda q: q[4] / (q[0] + q[3] + q[4]))
        eT = float(np.sqrt(max(gT @ cov @ gT, 0.0)))
        em = float(np.sqrt(max(gm @ cov @ gm, 0.0)))
        ef = float(np.sqrt(max(gf @ cov @ gf, 0.0)))
        row = g["row"]
        ems = float(row["f_malloc_err"]) if np.isfinite(row["f_malloc_err"]) else float("nan")
        efs = float(row["f_free_err"]) if np.isfinite(row["f_free_err"]) else float("nan")
        print(f"{g['name']:44s} {eT/T0:12.2%} {em:10.2%} {ef:10.2%} {ems:14.2%} {efs:14.2%}")
    print("(T0 is pinned to a few % by the (0,0) baseline; the chain-rule f errors are the stored ones,")
    print(" i.e. the headline fraction inherits the (W,A) flat-direction uncertainty.)")


def main():
    groups, shared = load_groups()
    part1(groups)
    part2(groups)
    part3(groups)
    part4(groups, shared)


if __name__ == "__main__":
    main()
