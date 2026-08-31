"""Re-analysis of the allocation-model fits in output/results.h5.

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

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import f as fdist

RESULTS = "output/results.h5"
EPS_S = 1e-12

pd.set_option("display.width", 260)


def read_table(f: h5py.File, grp: str) -> pd.DataFrame:
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


def model_nom(m, f, W, Nm, Nf, Af, f0):
    """Full model without the malloc saturation term."""
    return W + Nm * m + Nf * f + Af * f0 / (f + f0)


def model_nofree(m, f, W, Nm, Nf, Am, m0):
    """Full model without the free saturation term."""
    return W + Nm * m + Nf * f + Am * m0 / (m + m0)


def ssr_of(fn, m, f, t, *p):
    return float(np.sum((t - fn(m, f, *p)) ** 2))


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


def fit_linear(m, f, t):
    X = np.vstack([np.ones_like(m), m, f]).T
    sol, *_ = np.linalg.lstsq(X, t, rcond=None)
    return sol, float(np.sum((t - X @ sol) ** 2))


def grid_guess(m, f, t, bounds):
    """The stored pipeline's robust initial guess (allocation_model._grid_guess_2d)."""
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


def bic(n, k, ssr):
    return n * np.log(ssr / n) + k * np.log(n)


def bounds_of(m, f):
    m_min_pos = m[m > 0].min() if np.any(m > 0) else m.max()
    f_min_pos = f[f > 0].min() if np.any(f > 0) else f.max()
    lo_m = max(0.05 * m_min_pos, EPS_S)
    hi_m = max(0.5 * (m.max() - m.min()), lo_m * 1.5)
    lo_f = max(0.05 * f_min_pos, EPS_S)
    hi_f = max(0.5 * (f.max() - f.min()), lo_f * 1.5)
    return lo_m, hi_m, lo_f, hi_f


def profile_A(m, f, t, hi_m, hi_f, which, a_hi, base, ngrid=41):
    """Delta-SSR profile with A_malloc (which='m') or A_free (which='f') fixed
    at grid values, the remaining six parameters re-optimized in-bounds."""
    ssr_base = ssr_of(model_full, m, f, t, *base)
    grid = np.linspace(0.0, a_hi, ngrid)
    out = []
    for a in grid:
        if which == "m":
            def fn(x, W, Nm, Nf, Af, m0, f0):
                mm, ff = x
                return model_full(mm, ff, W, Nm, Nf, a, Af, m0, f0)

            p0 = [base[0], base[1], base[2], base[4], base[5], base[6]]
        else:
            def fn(x, W, Nm, Nf, Am, m0, f0):
                mm, ff = x
                return model_full(mm, ff, W, Nm, Nf, Am, a, m0, f0)

            p0 = [base[0], base[1], base[2], base[3], base[5], base[6]]
        lo = [0.0, 0.0, 0.0, 0.0, EPS_S, EPS_S]
        hi = [np.inf, np.inf, np.inf, np.inf, hi_m, hi_f]
        try:
            popt, _ = _fit(fn, (m, f), t, p0, lo, hi)
            d = float(np.sum((t - fn((m, f), *popt)) ** 2)) - ssr_base
        except (RuntimeError, ValueError):
            d = np.nan
        out.append(d)
    return grid, np.asarray(out)


def seed_sensitivity(m, f, t, lo_m, hi_m, lo_f, hi_f, ngrid=25):
    """Full 2-D fit with the fade-scale seeds varied over the search window.

    Seeds: the documented grid guess, the window bottom (a496cda's seeding),
    and the geometric mid. Reports A_malloc, A_free, f_malloc, f_free for each.
    """
    out = []
    guess = grid_guess(m, f, t, (lo_m, hi_m, lo_f, hi_f))
    seeds = {
        "grid": (guess[5], guess[6]),
        "bottom": (lo_m, lo_f),
        "mid": (float(np.sqrt(lo_m * hi_m)), float(np.sqrt(lo_f * hi_f))),
    }
    for name, (m0, f0) in seeds.items():
        p0 = [max(guess[0], EPS_S), max(guess[1], EPS_S), max(guess[2], EPS_S),
              max(guess[3], EPS_S), max(guess[4], EPS_S), m0, f0]
        try:
            popt, _ = _fit(lambda x, W, Nm, Nf, Am, Af, mm, ff: model_full(x[0], x[1], W, Nm, Nf, Am, Af, mm, ff),
                           (m, f), t, p0, [0.0] * 5 + [EPS_S, EPS_S], [np.inf] * 5 + [hi_m, hi_f])
        except (RuntimeError, ValueError):
            out.append((name, None))
            continue
        W, Nm, Nf, Am, Af = (float(v) for v in popt[:5])
        T0 = W + Am + Af
        out.append((name, (Am, Af, Am / T0 if T0 > EPS_S else 0.0, Af / T0 if T0 > EPS_S else 0.0, float(popt[5]), float(popt[6]))))
    return out


def gauge_block(m, fv, stored, lo_m, hi_m, lo_f, hi_f, pcov):
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
    """
    W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in stored)
    base = model_full(m, fv, W, Nm, Nf, Am, Af, m0, f0)

    worst = 0.0
    for d in (-0.5, -0.2, 0.05, 0.2, 0.5):
        r = model_full(m - m0 * d, fv, W + Nm * m0 * d, Nm, Nf, Am / (1 + d), Af, m0 * (1 + d), f0)
        worst = max(worst, float(np.max(np.abs(r - base))))
    for e in (-0.5, -0.2, 0.05, 0.2, 0.5):
        r = model_full(m, fv - f0 * e, W + Nf * f0 * e, Nm, Nf, Am, Af / (1 + e), m0, f0 * (1 + e))
        worst = max(worst, float(np.max(np.abs(r - base))))

    T0 = W + Am + Af
    fm0 = Am / T0 if T0 > EPS_S else 0.0
    ff0 = Af / T0 if T0 > EPS_S else 0.0

    def fm_of_d(d):
        Ad = Am / (1 + d)
        T0d = W + Nm * m0 * d + Ad + Af
        return Ad / T0d if T0d > EPS_S else 0.0

    def ff_of_e(e):
        Ae = Af / (1 + e)
        T0e = W + Nf * f0 * e + Am + Ae
        return Ae / T0e if T0e > EPS_S else 0.0

    # gauge orbit over the whole fade-scale search window (includes d = 0)
    dg = np.unique(np.concatenate([np.linspace(lo_m / m0 - 1.0, hi_m / m0 - 1.0, 121), [0.0]]))
    eg = np.unique(np.concatenate([np.linspace(lo_f / f0 - 1.0, hi_f / f0 - 1.0, 121), [0.0]]))
    # how far f moves along the gauge orbit over that window
    fmg = np.array([fm_of_d(d) for d in dg])
    ffg = np.array([ff_of_e(e) for e in eg])
    fm_rng = (float(fmg.min()), float(fmg.max()))
    ff_rng = (float(ffg.min()), float(ffg.max()))

    g_m = np.array([Nm * m0, 0.0, 0.0, -Am, 0.0, m0, 0.0])
    g_f = np.array([Nf * f0, 0.0, 0.0, 0.0, -Af, 0.0, f0])
    flat = (float("nan"), float("nan"))
    if pcov is not None and np.all(np.isfinite(pcov)):
        cov = np.asarray(pcov, dtype=float)
        _eig, evec = np.linalg.eigh(0.5 * (cov + cov.T))
        v1, v2 = evec[:, 0], evec[:, 1]

        def frac(g):
            gn = g / np.linalg.norm(g)
            return float(np.hypot(np.dot(v1, gn), np.dot(v2, gn)))

        flat = (frac(g_m), frac(g_f))

    return {"id_res": worst, "fm0": fm0, "ff0": ff0, "fm_rng": fm_rng, "ff_rng": ff_rng, "flat": flat}


def arm_offset(m, f, t):
    """Per arm (per-delay medians): offset above the line through the two
    largest-delay points, at the smallest delay and at mid-range (s)."""
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


def arm_anchored(m, f, t, t00):
    """Per arm, anchored at the measured zero-delay runtime t00:
    E(s) = T(s) - t00 - N*s (the excess over the nominal line through the
    baseline), the plateau deficit d = t00 - (line intercept), and the local
    slope between the two smallest delays relative to N (the large-delay slope).
    Hiding of the small delay shows up as E < 0 and slope_small < N."""
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


def arm_profile(m, f, t):
    """Per arm: full per-delay-median offset profile above the two-largest-delay
    line. Returns (offsets, delays) per arm."""
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
    g = sub.groupby(["malloc_sleeptime", "free_sleeptime"])["runtime_s"]
    med = g.transform("median")
    cnt = g.transform("count")
    dev = (sub["runtime_s"] - med)[cnt >= 2]
    return float(dev.std()) if dev.size >= 3 else float("nan")


def group_of(runs: pd.DataFrame, key) -> pd.DataFrame:
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
        sub = sub[(sub["z"] == key[5])]
    return sub.dropna(subset=["malloc_sleeptime", "free_sleeptime", "runtime_s"])


def main():
    f = h5py.File(RESULTS, "r")
    runs = read_table(f, "runs")
    fits = read_table(f, "fits")
    for col in ("machine", "setup", "algorithm"):
        runs[col] = [v.decode() if isinstance(v, bytes) else v for v in runs[col]]
        fits[col] = [v.decode() if isinstance(v, bytes) else v for v in fits[col]]
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")

    rows = []
    multimod = None
    seed_results: dict[str, list] = {}
    anchored_results: dict[str, tuple] = {}
    gauge_results: dict[str, dict] = {}
    for i, row in fits.iterrows():
        key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
        sub = group_of(runs, key)
        m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
        fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
        t = sub["runtime_s"].to_numpy(float)
        n = t.size
        n_00 = int(((m == 0) & (fv == 0)).sum())

        lo_m, hi_m, lo_f, hi_f = bounds_of(m, fv)
        stored = np.asarray(
            [row["W"], row["N_malloc"], row["N_free"], row["A_malloc"], row["A_free"], row["m0_ns"] * 1e-9, row["f0_ns"] * 1e-9],
            dtype=float,
        )
        p0 = grid_guess(m, fv, t, (lo_m, hi_m, lo_f, hi_f))
        p_full, pcov = fit_full(m, fv, t, hi_m, hi_f, p0=p0)
        ssr_full = ssr_of(model_full, m, fv, t, *p_full)
        ssr_stored = ssr_of(model_full, m, fv, t, *stored)
        sol_lin, ssr_lin = fit_linear(m, fv, t)
        ss_tot = float(np.sum((t - t.mean()) ** 2))

        def fn_nom(x, W, Nm, Nf, Af, f0):
            return model_nom(x[0], x[1], W, Nm, Nf, Af, f0)

        def fn_nof(x, W, Nm, Nf, Am, m0):
            return model_nofree(x[0], x[1], W, Nm, Nf, Am, m0)

        p_nom, _ = _fit(fn_nom, (m, fv), t, [p0[0], p0[1], p0[2], p0[4], p0[6]],
                        [0, 0, 0, 0, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi_f])
        ssr_nom = ssr_of(model_nom, m, fv, t, *p_nom)
        p_nof, _ = _fit(fn_nof, (m, fv), t, [p0[0], p0[1], p0[2], p0[3], p0[5]],
                        [0, 0, 0, 0, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi_m])
        ssr_nof = ssr_of(model_nofree, m, fv, t, *p_nof)

        F_stored = ((ssr_lin - ssr_stored) / 4) / (ssr_stored / (n - 7))
        p_stored = float(fdist.sf(F_stored, 4, n - 7))
        F_refit = ((ssr_lin - ssr_full) / 4) / (ssr_full / (n - 7))
        p_refit = float(fdist.sf(F_refit, 4, n - 7))
        F_nom = ((ssr_nom - ssr_full) / 2) / (ssr_full / (n - 7))
        p_nom = float(fdist.sf(F_nom, 2, n - 7))
        F_nof = ((ssr_nof - ssr_full) / 2) / (ssr_full / (n - 7))
        p_nof = float(fdist.sf(F_nof, 2, n - 7))
        dbic = bic(n, 7, ssr_full) - bic(n, 3, ssr_lin)
        cond = float(np.linalg.cond(pcov)) if np.all(np.isfinite(pcov)) else float("inf")

        W, Nm, Nf, Am, Af, m0, f0 = (float(v) for v in p_full)
        T0 = W + Am + Af
        spread = run_spread(sub)
        if not np.isfinite(spread):
            spread = float(np.sqrt(ssr_full / max(n - 7, 1)))
        base00 = t[(m == 0) & (fv == 0)]
        base00 = float(np.median(base00)) if base00.size else float("nan")

        dmax = float(np.max(np.abs(np.asarray(p_full) - stored)))
        gm, dm = profile_A(m, fv, t, hi_m, hi_f, "m", max(4 * max(Am, 1e-2), 1e-2), p_full)
        gf, dfp = profile_A(m, fv, t, hi_m, hi_f, "f", max(4 * max(Af, 1e-2), 1e-2), p_full)
        thr = spread**2
        am_ok = gm[dm <= thr] if np.any(np.isfinite(dm)) else np.array([])
        af_ok = gf[dfp <= thr] if np.any(np.isfinite(dfp)) else np.array([])
        arm = arm_offset(m, fv, t)
        arms = arm_profile(m, fv, t)
        seeds = seed_sensitivity(m, fv, t, lo_m, hi_m, lo_f, hi_f)
        gauge = gauge_block(m, fv, stored, lo_m, hi_m, lo_f, hi_f, pcov)

        rows.append(
            {
                "group": f"{key[0]} {key[1][:3] if key[1] == 'KelvinHelmholtz' else key[1]} "
                + f"{int(key[3])}x{int(key[4])}"
                + (f"x{int(key[5])}" if not np.isnan(key[5]) else "")
                + f" {key[2][:4]}",
                "n": n,
                "n00": n_00,
                "dmax": dmax,
                "F_stored": F_stored,
                "p_stored": p_stored,
                "F_refit": F_refit,
                "p_refit": p_refit,
                "F_Am": F_nom,
                "p_Am": p_nom,
                "F_Af": F_nof,
                "p_Af": p_nof,
                "dBIC": dbic,
                "cond": cond,
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
                "T0_fit": T0,
                "dSSR_pr": (ssr_lin - ssr_full) / n,
                "dSSR_pr_sig2": (ssr_lin - ssr_full) / n / spread**2 if spread > 0 else float("nan"),
                "am_offmin": float(arms["malloc"][0].min()) if arms["malloc"][0] is not None else float("nan"),
                "am_offmax": float(arms["malloc"][0].max()) if arms["malloc"][0] is not None else float("nan"),
                "af_offmin": float(arms["free"][0].min()) if arms["free"][0] is not None else float("nan"),
                "af_offmax": float(arms["free"][0].max()) if arms["free"][0] is not None else float("nan"),
            }
        )
        seed_results[rows[-1]["group"]] = seeds
        gauge_results[rows[-1]["group"]] = gauge
        anchored = arm_anchored(m, fv, t, base00)
        anchored_results[rows[-1]["group"]] = anchored
        # model-consistent decomposition pinned by the data:
        # T(m,0) -> W + A_free + N_malloc*m  =>  A_malloc = d_m (the plateau)
        # T(0,f) -> W + A_malloc + N_free*f  =>  A_free   = d_f
        # so the data fix (W, A_malloc, A_free) = (T0 - d_m - d_f, d_m, d_f).
        dm = anchored["malloc"]["d"] if anchored["malloc"] else float("nan")
        df_ = anchored["free"]["d"] if anchored["free"] else float("nan")
        W_pin = base00 - dm - df_
        resid_stored = np.abs(model_full(m, fv, *stored) - t)
        rows[-1]["Am_stored"] = float(row["A_malloc"])
        rows[-1]["Af_stored"] = float(row["A_free"])
        rows[-1]["d_m"] = dm
        rows[-1]["d_f"] = df_
        rows[-1]["W_pin"] = W_pin
        rows[-1]["dW"] = float(row["W"]) - W_pin
        rows[-1]["dAm"] = float(row["A_malloc"]) - dm
        rows[-1]["dAf"] = float(row["A_free"]) - df_
        rows[-1]["f_m_pin%"] = 100 * dm / base00 if base00 > EPS_S else float("nan")
        rows[-1]["f_f_pin%"] = 100 * df_ / base00 if base00 > EPS_S else float("nan")
        rows[-1]["maxres_stored"] = float(resid_stored.max())
        # keep the flagged group's multi-modality numbers
        if row["note"] and isinstance(row["note"], bytes) and row["note"]:
            r2_stored = 1 - ssr_stored / ss_tot
            r2_refit = 1 - ssr_full / ss_tot
            r2_lin = 1 - ssr_lin / ss_tot
            multimod = {
                "group": rows[-1]["group"],
                "n": n,
                "ssr_stored": ssr_stored,
                "ssr_refit": ssr_full,
                "ssr_lin": ssr_lin,
                "r2_stored": r2_stored,
                "r2_refit": r2_refit,
                "r2_lin": r2_lin,
                "F_stored": F_stored,
                "p_stored": p_stored,
                "F_refit": F_refit,
                "p_refit": p_refit,
                "dBIC_refit": dbic,
                "stored": stored,
                "refit": p_full,
                "dmax": dmax,
            }
    f.close()
    df = pd.DataFrame(rows)

    def frange(r, T0):
        if not (np.isfinite(r[0]) and np.isfinite(r[1])):
            return "-"
        return f"[{100*r[0]/T0:.1f}%,{100*r[1]/T0:.1f}%]"

    df["am_frange"] = [frange(r, T) for r, T in zip(df["am_range"], df["T0_fit"])]
    df["af_frange"] = [frange(r, T) for r, T in zip(df["af_range"], df["T0_fit"])]

    out = df[
        [
            "group", "n", "n00", "dmax", "F_stored", "p_stored", "F_refit", "p_refit",
            "F_Am", "p_Am", "F_Af", "p_Af", "dBIC", "cond",
            "f_m%", "f_f%", "cam_us", "caf_us", "T0", "base00", "W", "spread", "dSSR_pr", "dSSR_pr_sig2",
            "am_offmin", "am_offmax", "af_offmin", "af_offmax",
            "am_range", "am_frange", "af_range", "af_frange",
        ]
    ].copy()
    for c in ("dmax", "F_stored", "F_refit", "F_Am", "F_Af", "dBIC", "f_m%", "f_f%", "cam_us", "caf_us",
              "T0", "base00", "W", "spread", "dSSR_pr", "am_offmin", "am_offmax", "af_offmin", "af_offmax"):
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
    print("am/af_range: A values (s) whose fix costs <= spread^2 extra SSR (wide = f unconstrained); *_frange: the same, as f=A/T0.")
    print("cam/caf: implied native cost per call c_a = A/N (us); T0/base00: fitted vs measured (0,0) runtime.")
    print("am_offmin/max, af_offmin/max: per-delay-median offset (s) above the two-largest-delay line, min and max over the arm.")
    print("  (the model's A*s0/(s+s0) requires a monotonically decaying positive offset; negative = below the line).")
    print("dSSR_pr: per-run SSR the 4 saturation params buy over the line (s^2/run); dSSR_pr_sig2: same, in units of the per-run spread^2.")
    if multimod:
        mm = multimod
        print()
        print("=== 2. multi-modality, flagged group ===")
        print(f"group: {mm['group']}  (n={mm['n']})")
        print(f"stored fit:    SSR={mm['ssr_stored']:9.1f}  r2={mm['r2_stored']:.6f}  F vs line={mm['F_stored']:7.2f}  p={mm['p_stored']:.2e}")
        print(f"grid-start:    SSR={mm['ssr_refit']:9.1f}  r2={mm['r2_refit']:.6f}  F vs line={mm['F_refit']:7.2f}  p={mm['p_refit']:.2e}  dBIC={mm['dBIC_refit']:+.1f}")
        print(f"linear null:   SSR={mm['ssr_lin']:9.1f}  r2={mm['r2_lin']:.6f}")
        print(f"stored params:  {np.array2string(mm['stored'], precision=5)}")
        print(f"grid-start p:   {np.array2string(np.asarray(mm['refit']), precision=5)}")
        print(f"max |stored - grid-start| = {mm['dmax']:.1f} s")
        print("note: identical data, identical bounds, identical documented initial guess;")
        print("      the fresh run lands in a different (worse) local minimum than the stored fit,")
        print("      and is even worse than the 3-parameter line.")
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
            print(f"  {name:7s}  A_malloc={Am:8.3f} s  A_free={Af:8.4f} s  f_malloc={100*fm:5.2f}%  f_free={100*ff:5.2f}%  (m0={m0s*1e9:.3g} ns, f0={f0s*1e9:.3g} ns)")
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
            print(f"    E(small)={a_['E0']:+8.2f} s   E(mid)={a_['Emid']:+8.2f} s   E(large)={a_['Emax']:+8.2f} s   s_sm/N={a_['s_sm_over_N']:+.3f}")
    print()
    print("=== 5. stored fit vs the data-pinned (model-consistent) decomposition ===")
    print("The model's own asymptotes make A_malloc = d_m and A_free = d_f (the measured")
    print("plateaus of block 4), W = T0 - d_m - d_f: three quantities the data fix directly.")
    print("dW/dAm/dAf: stored value minus the data-pinned value (s). f_*_pin: d/T0.")
    pin = df[["group", "W", "dW", "Am_stored", "dAm", "Af_stored", "dAf", "f_m%", "f_m_pin%", "f_f%", "f_f_pin%", "maxres_stored"]]
    for c in ("W", "dW", "Am_stored", "dAm", "Af_stored", "dAf"):
        pin[c] = df[c].map(lambda v: f"{v:+.2f}" if c.startswith("d") else f"{v:.2f}")
    for c in ("f_m%", "f_m_pin%", "f_f%", "f_f_pin%", "maxres_stored"):
        pin[c] = df[c].map(lambda v: f"{v:.2f}")
    print(pin.to_string(index=False))
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
    gtab = []
    for grp, g in gauge_results.items():
        fm_rng, ff_rng = g["fm_rng"], g["ff_rng"]

        def rngstr(r):
            return f"[{100*r[0]:.1f},{100*r[1]:.1f}]" if np.isfinite(r[0]) and np.isfinite(r[1]) else "-"

        fm_err = df.loc[df["group"] == grp, "fm_err"].iloc[0]
        ff_err = df.loc[df["group"] == grp, "ff_err"].iloc[0]
        fm_w = fm_rng[1] - fm_rng[0] if np.isfinite(fm_rng[0]) and np.isfinite(fm_rng[1]) else float("nan")
        ff_w = ff_rng[1] - ff_rng[0] if np.isfinite(ff_rng[0]) and np.isfinite(ff_rng[1]) else float("nan")
        gtab.append(
            {
                "group": grp,
                "id": f"{g['id_res']:.1e}",
                "f_m%": f"{100*g['fm0']:.1f}",
                "f_m_win": rngstr(fm_rng),
                "w/err_m": f"{fm_w/fm_err:5.1f}" if np.isfinite(fm_w) and fm_err > EPS_S else "-",
                "f_f%": f"{100*g['ff0']:.1f}",
                "f_f_win": rngstr(ff_rng),
                "w/err_f": f"{ff_w/ff_err:5.1f}" if np.isfinite(ff_w) and ff_err > EPS_S else "-",
                "flat_gm": f"{g['flat'][0]:.2f}" if np.isfinite(g["flat"][0]) else "-",
                "flat_gf": f"{g['flat'][1]:.2f}" if np.isfinite(g["flat"][1]) else "-",
            }
        )
    print(pd.DataFrame(gtab).to_string(index=False))


if __name__ == "__main__":
    main()
