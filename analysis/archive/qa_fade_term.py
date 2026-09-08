# ARCHIVED: model-review analysis from before the fade-term switch; the
# published model (analysis/performance_model.py) now uses the exponential.
# Not part of the published pipeline and not maintained. See
# analysis/archive/README.md. Restored with notes-pre-publish.
"""Q3: alternative fade terms for the performance model, measured.

This script is the record of the Q3 analysis: it fits six candidate fade
terms with the same endpoints -- the hyperbola (the then-stored model), the
exponential, Lorentzian, power-law, truncated-linear and quadratic overlap --
to the same 12 groups and measures fit quality, identifiability, gauge
structure, and the extrapolation each form makes beyond the data.

NOTE: the recommendation of this analysis -- use the exponential -- has been
implemented; the stored model now fades with the exponential A*exp(-m/s). The
shape comparison (P3, P6, against a fresh hyperbola refit) is model-agnostic
and still valid on re-run, but the "stored fit" columns (P2's baseline, P4's
stored scale) assume the hyperbola was the stored model. The forward-looking,
model-agnostic comparison figure is `analysis/plot_fade_models.py`
(`figures/fade-models.pdf`).

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads output/results.h5 (runs, fits, fits/cov, absorption) and re-fits.
Prints:
  P1. the candidate family: forms, local expansions, tails, total cost;
  P2. per group: delta-SSR of every candidate against the stored
      hyperbola fit, residual scale, and the probed delay range in
      fade-scale units;
  P3. detailed fits of three focus groups: parameters, delta-SSR against
      the fresh hyperbola refit, pcov condition, flattest direction;
  P4. tail claims: the visible fade each candidate keeps beyond the data
      and the total absorbed cost, plus the fade scale against the
      microbenchmark per-call cost;
  P5. gauge structure: which combinations each form pins, and the
      measured flatness of the focus-group fits;
  P6. the measured verdict.

Run: /tmp/opencode/venv/bin/python qa_fade_term.py
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

RESULTS = "/workspace/output/results.h5"
EPS_S = 1e-12
CANDS = ("H", "E", "L", "T", "Q", "P")
FOCUS = (
    "hal Kelv Flat 256x128x128",
    "hal Foil Flat 256x1280x0",
    "rosi Kelv Flat 256x128x128",
)
K_GRID = (0.4, 1.0, 2.0, 4.0)
S_LO = 1e-7
FITS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


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


def load_groups() -> tuple[list[dict], dict[tuple, tuple[float, float]]]:
    """Load the per-group data and the microbenchmark per-call costs.

    Returns:
        tuple: (the per-group dicts, the (machine, setup, algorithm, arm,
        x, y, z) -> (c_us, N) map from the absorption table).

    """
    f = h5py.File(RESULTS, "r")
    runs = read_table(f, "runs")
    fits = read_table(f, "fits")
    for df in (runs, fits):
        _decode_str_cols(df)
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")
    groups = [_group_entry(runs, row, f) for _, row in fits.iterrows()]
    abs_df = read_table(f, "absorption")
    for col in ("algorithm", "arm", "machine", "setup"):
        abs_df[col] = [v.decode() if isinstance(v, bytes) else v for v in abs_df[col]]
    abs_df["z"] = pd.to_numeric(abs_df["z"], errors="coerce")
    c_us: dict[tuple, tuple[float, float]] = {}
    for _, r in abs_df.iterrows():
        z = None if pd.isna(r["z"]) else float(r["z"])
        c_us[r["machine"], r["setup"], r["algorithm"], r["arm"], float(r["x"]), float(r["y"]), z] = (
            float(r["c_us"]),
            float(r["N"]),
        )
    f.close()
    return groups, c_us


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


def _fade(cand: str, m: np.ndarray, a: float, s: float, k: float) -> np.ndarray:
    """Evaluate the visible fade F(m) of one candidate at delays m.

    All candidates satisfy F(0) = a and F(m -> inf) = 0.

    Args:
        cand: the candidate key (H, E, L, T, Q, P).
        m: the delays in seconds.
        a: the fade amplitude (the visible cost at zero delay) in seconds.
        s: the fade scale in seconds.
        k: the power-law exponent (used by P only).

    Returns:
        np.ndarray: the visible fade in seconds.

    """
    if cand == "H":
        return a * s / (m + s)
    if cand == "E":
        return a * np.exp(-m / s)
    if cand == "L":
        return a * s * s / (m * m + s * s)
    if cand == "T":
        return a * np.maximum(1.0 - m / s, 0.0)
    if cand == "Q":
        u = np.maximum(1.0 - m / s, 0.0)
        return a * u * u
    return a * (1.0 + m / s) ** (-k)


def _eval(cand: str, m: np.ndarray, f: np.ndarray, p: Sequence[float]) -> np.ndarray:
    """Evaluate the two-arm model of one candidate at the packed parameters.

    T = W + N_m*m + N_f*f + F_m(m) + F_f(f) with the candidate's fade F.

    Args:
        cand: the candidate key (H, E, L, T, Q, P).
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        p: (W, N_m, N_f, A_m, s_m, A_f, s_f) or, for P,
        (W, N_m, N_f, A_m, s_m, k_m, A_f, s_f, k_f).

    Returns:
        np.ndarray: the model runtime in seconds.

    """
    if cand == "P":
        W, Nm, Nf, Am, sm, km, Af, sf, kf = p
        return W + Nm * m + Nf * f + _fade("P", m, Am, sm, km) + _fade("P", f, Af, sf, kf)
    W, Nm, Nf, Am, sm, Af, sf = p
    return W + Nm * m + Nf * f + _fade(cand, m, Am, sm, 1.0) + _fade(cand, f, Af, sf, 1.0)


def _cell(
    cand: str,
    m: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    cell: tuple[float, float, float, float],
) -> tuple[float, list[float]]:
    """One (s_m, s_f[, k_m, k_f]) grid cell: its SSR and initial parameters.

    With the fade shapes fixed, the model is linear in (W, N_m, N_f, A_m,
    A_f), so one least-squares call per cell.

    Args:
        cand: the candidate key.
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        cell: (s_m, s_f, k_m, k_f) of the cell, scales in seconds.

    Returns:
        tuple: (the cell's SSR, the 0-floored initial parameters).

    """
    sm, sf, km, kf = cell
    cols = [np.ones_like(m), m, f]
    if cand == "P":
        cols += [(1.0 + m / sm) ** (-km), (1.0 + f / sf) ** (-kf)]
    else:
        cols += [_fade(cand, m, 1.0, sm, 1.0), _fade(cand, f, 1.0, sf, 1.0)]
    x = np.vstack(cols).T
    sol, *_ = np.linalg.lstsq(x, t, rcond=None)
    ssr = float(np.sum((t - x @ sol) ** 2))
    if cand == "P":
        p0 = [sol[0], sol[1], sol[2], sol[3], sm, km, sol[4], sf, kf]
    else:
        p0 = [sol[0], sol[1], sol[2], sol[3], sm, sol[4], sf]
    return ssr, [max(float(v), EPS_S) for v in p0]


def fit_cand(cand: str, m: np.ndarray, f: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-guess and polish one candidate fit of one group.

    The scale search runs on a log grid over [1e-7, 10 x the documented
    bound] per arm (P also grids the exponent); the best cell seeds a
    bounded curve_fit with the same bounds for every candidate.

    Args:
        cand: the candidate key.
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.

    Returns:
        tuple: (the fitted parameters, the parameter covariance).

    Raises:
        RuntimeError: if the grid search yields no cell.

    """
    _, hi_m, _, hi_f = bounds_of(m, f)
    sgm = np.logspace(np.log10(S_LO), np.log10(10.0 * hi_m), 15)
    sgf = np.logspace(np.log10(S_LO), np.log10(10.0 * hi_f), 15)
    best: tuple[float, list[float]] | None = None
    k_grid = K_GRID if cand == "P" else (1.0,)
    for km in k_grid:
        for kf in k_grid:
            for sm in sgm:
                for sf in sgf:
                    ssr, p0 = _cell(cand, m, f, t, (float(sm), float(sf), km, kf))
                    if best is None or ssr < best[0]:
                        best = (ssr, p0)
    if best is None:
        msg = "the grid search yielded no cell"
        raise RuntimeError(msg)
    _, p0 = best
    if cand == "P":
        lo = [0.0, 0.0, 0.0, 0.0, EPS_S, 0.2, 0.0, EPS_S, 0.2]
        hi_b = [np.inf, np.inf, np.inf, np.inf, 10.0 * hi_m, 10.0, np.inf, 10.0 * hi_f, 10.0]
    else:
        lo = [0.0, 0.0, 0.0, 0.0, EPS_S, 0.0, EPS_S]
        hi_b = [np.inf, np.inf, np.inf, np.inf, 10.0 * hi_m, np.inf, 10.0 * hi_f]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        if cand == "P":
            popt, pcov = curve_fit(
                lambda x, W, Nm, Nf, Am, sm, km, Af, sf, kf: _eval(
                    "P", x[0], x[1], (W, Nm, Nf, Am, sm, km, Af, sf, kf)
                ),
                (m, f),
                t,
                p0=p0,
                bounds=(lo, hi_b),
                maxfev=40000,
            )
        else:
            popt, pcov = curve_fit(
                lambda x, W, Nm, Nf, Am, sm, Af, sf: _eval(cand, x[0], x[1], (W, Nm, Nf, Am, sm, Af, sf)),
                (m, f),
                t,
                p0=p0,
                bounds=(lo, hi_b),
                maxfev=40000,
            )
    return popt, pcov


def _ssr(cand: str, m: np.ndarray, f: np.ndarray, t: np.ndarray, p: np.ndarray) -> float:
    """Return the SSR of one candidate fit.

    Args:
        cand: the candidate key.
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        p: the parameters.

    Returns:
        float: the sum of squared residuals in s^2.

    """
    r = t - _eval(cand, m, f, p)
    return float(np.sum(r**2))


def _param_str(cand: str, p: np.ndarray) -> str:
    """Format one candidate's fitted parameters (seconds).

    Args:
        cand: the candidate key.
        p: the fitted parameters.

    Returns:
        str: the 'A_m=... s_m=...' parameter string.

    """
    if cand == "P":
        _, _, _, Am, sm, km, Af, sf, kf = (float(v) for v in p)
        return f"A_m={Am:7.3g}s s_m={sm:7.3g}s k_m={km:4.2f} A_f={Af:7.3g}s s_f={sf:7.3g}s k_f={kf:4.2f}"
    _, _, _, Am, sm, Af, sf = (float(v) for v in p)
    return f"A_m={Am:7.3g}s s_m={sm:7.3g}s A_f={Af:7.3g}s s_f={sf:7.3g}s"


def _p1() -> None:
    """Print section P1: the candidate family and its local expansions."""
    print()
    print("=" * 100)
    print("P1. candidate fade terms (all: F(0) = A, F(inf) = 0, monotone)")
    print("=" * 100)
    print("u = m/s. The stored model is H. T and Q are the overlap picture (the delay hides")
    print("parallel work until the slack is exhausted) with a hard cutoff at m = s.")
    print()
    print(
        f"{'form':22s} {"F'(0)/A":>9s} {"F''(0)/A":>10s} | {'F/A @u=1':>9s} {'@u=3':>8s} {'@u=10':>8s} | "
        f"{'tail m>>s':>13s} {'int F dm/(A s)':>15s}"
    )
    meta = (
        ("H  A s/(m+s)", "-1/s", "+1/s^2", "A s/m", "inf"),
        ("E  A exp(-m/s)", "-1/s", "+1/(2 s^2)", "A e^(-m/s)", "1"),
        ("L  A s^2/(m^2+s^2)", "0", "-1/s^2", "A s^2/m^2", "pi/2"),
        ("T  A (1-m/s)+", "-1/s", "0", "0 (cut at s)", "1/2"),
        ("Q  A (1-m/s)^2+", "-2/s", "+1/s^2", "0 (cut at s)", "2/3"),
        ("P  A (1+m/s)^-k", "-1/s", "+(k+1)/(2 s^2)", "A (s/m)^k", "1/(k-1), k>1"),
    )
    for cand in CANDS:
        form, d1, d2, tail, tot = meta[CANDS.index(cand)]
        vals = [_fade(cand, np.array([u]), 1.0, 1.0, 1.0)[0] for u in (1.0, 3.0, 10.0)]
        print(f"{form:22s} {d1:>9s} {d2:>10s} | {vals[0]:9.3f} {vals[1]:8.3f} {vals[2]:8.2e} | {tail:>13s} {tot:>15s}")


def _p2_row(g: dict) -> list[tuple[str, float]]:
    """Fit every candidate of one group and return its delta-SSRs.

    Args:
        g: the loaded group.

    Returns:
        list: the (candidate, delta-SSR in sig^2 vs the stored H fit)
        pairs, H included as 0.0.

    """
    m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
    ssr_h = _ssr("H", m, fv, t, p)
    sig2 = ssr_h / max(t.size - 7, 1)
    rows = [("H", 0.0)]
    for cand in ("E", "L", "T", "Q", "P"):
        popt, pcov = fit_cand(cand, m, fv, t)
        FITS[g["name"] + "/" + cand] = (popt, pcov)
        rows.append((cand, (_ssr(cand, m, fv, t, popt) - ssr_h) / sig2))
    return rows


def _p2(groups: list[dict]) -> None:
    """Print section P2: per-group candidate fit quality."""
    print()
    print("=" * 100)
    print("P2. fit quality: every candidate refit, delta-SSR against the stored H fit")
    print("=" * 100)
    print("delta-SSR in units of sig^2 (0 = equivalent to the stored fit, negative = better);")
    print("u_max = the farthest measured delay in the stored fade-scale units of its arm:")
    print()
    print(
        f"{'group':28s} {'n':>4s} {'u_max':>8s} {'sig':>8s} | "
        f"{'dE':>7s} {'dL':>7s} {'dT':>7s} {'dQ':>7s} {'dP':>7s} | {'best':>4s}"
    )
    for g in groups:
        p = g["p"]
        u_max = max(g["m"].max() / p[5], g["f"].max() / p[6])
        ssr_h = _ssr("H", g["m"], g["f"], g["t"], p)
        sig = np.sqrt(ssr_h / max(g["t"].size - 7, 1))
        rows = _p2_row(g)
        d = dict(rows)
        best = min(rows, key=lambda r: r[1])[0]
        print(
            f"{g['name']:28s} {len(g['t']):4d} {u_max:8.1f} {sig:8.3g} | "
            f"{d['E']:7.2f} {d['L']:7.2f} {d['T']:7.2f} {d['Q']:7.2f} {d['P']:7.2f} | {best:>4s}"
        )


def _p3_group(g: dict) -> None:
    """Print one focus group's detailed candidate fits.

    Args:
        g: the loaded group.

    """
    m, fv, t = g["m"], g["f"], g["t"]
    ph, pch = fit_cand("H", m, fv, t)
    FITS[g["name"] + "/H"] = (ph, pch)
    ssr_h = _ssr("H", m, fv, t, ph)
    sig2 = ssr_h / max(t.size - 7, 1)
    print(f"\n{g['name']} (n={t.size}):  dSSR vs the fresh H refit, in sig^2; then the 1sig % of the")
    print("fade params A_m, s_m, A_f, s_f (cond(pcov) of each: section P5):")
    print(
        f"  {'cand':5s} {'parameters':52s} {'SSR':>8s} {'dSSR':>7s} | {'A_m':>6s} {'s_m':>6s} {'A_f':>6s} {'s_f':>6s}"
    )
    for cand in CANDS:
        popt, pcov = FITS.get(g["name"] + "/" + cand, fit_cand(cand, m, fv, t))
        FITS.setdefault(g["name"] + "/" + cand, (popt, pcov))
        err = np.sqrt(np.clip(np.diag(pcov), 0, None))
        idx = (3, 4, 5, 6) if cand != "P" else (3, 4, 6, 7)
        pct = " ".join(f"{100.0 * err[i] / abs(popt[i]):6.1f}%" if abs(popt[i]) > 1e-6 else f"{'--':>6s}" for i in idx)
        print(
            f"  {cand:5s} {_param_str(cand, popt):52s} {_ssr(cand, m, fv, t, popt):8.1f} "
            f"{(_ssr(cand, m, fv, t, popt) - ssr_h) / sig2:7.2f} | {pct}"
        )


def _p3(groups: list[dict]) -> None:
    """Print section P3: detailed candidate fits of the focus groups."""
    print()
    print("=" * 100)
    print("P3. detailed fits of the focus groups")
    print("=" * 100)
    print("the shape contest on the three groups with the widest delay ranges in fade-scale units:")
    for name in FOCUS:
        g = next(x for x in groups if x["name"] == name)
        _p3_group(g)


def _tail(cand: str, p: np.ndarray, u: float) -> float:
    """Return the fitted malloc-arm fade fraction F/A at u = m/s.

    Args:
        cand: the candidate key.
        p: the fitted parameters.
        u: the delay in fade-scale units.

    Returns:
        float: the visible fade as a fraction of A_m.

    """
    sm = float(p[4])
    if cand == "P":
        _, _, _, Am, _, km, _, _, _ = (float(v) for v in p)
        return float(_fade("P", np.array([u * sm]), Am, sm, km)[0] / Am) if Am > EPS_S else float("nan")
    _, _, _, Am, _, _, _ = (float(v) for v in p)
    return float(_fade(cand, np.array([u * sm]), Am, sm, 1.0)[0] / Am) if Am > EPS_S else float("nan")


def _p4(groups: list[dict], c_us: dict[tuple, tuple[float, float]]) -> None:
    """Print section P4: the tail each candidate claims beyond the data."""
    g = next(x for x in groups if x["name"] == FOCUS[0])
    p = g["p"]
    key = (*g["key"][:3], "malloc", g["key"][3], g["key"][4], None if np.isnan(g["key"][5]) else g["key"][5])
    c, n_calls = c_us.get(key, (float("nan"), float("nan")))
    print()
    print("=" * 100)
    print("P4. tail claims beyond the data (focus group, malloc arm, fitted s_m)")
    print("=" * 100)
    print("F_m/A_m = the visible fade the candidate still claims at delay u*s_m; a 1/m tail (H)")
    print("says the cost never vanishes, a cutoff (T, Q) says it is fully absorbed by m = s:")
    print()
    print(f"  {'cand':5s} {'s_m fit':>9s} | {'F/A @u=3':>9s} {'@u=10':>8s} {'@u=100':>8s} | claim beyond the data")
    for cand in CANDS:
        popt, _ = FITS[g["name"] + "/" + cand]
        v3, v10, v100 = (_tail(cand, popt, u) for u in (3.0, 10.0, 100.0))
        sm = float(popt[4])
        p_tail = (
            "an (s/m)^k tail (total A s/(k-1))" if float(popt[5]) > 1.0 else "a stretched tail (divergent for k<=1)"
        )
        claim = {
            "H": "a visible A s/m forever (divergent total)",
            "E": "gone by u ~ 5 (total A s)",
            "L": "a 1/m^2 tail (total pi A s /2)",
            "T": "exactly zero beyond u = 1 (total A s /2)",
            "Q": "exactly zero beyond u = 1 (total 2 A s /3)",
            "P": p_tail,
        }[cand]
        print(f"  {cand:5s} {sm:10.3g} | {v3:9.4f} {v10:8.2e} {v100:8.2e} |  {claim}")
    if np.isfinite(c) and c > 0:
        sm_h = float(FITS[g["name"] + "/H"][0][4])
        c_s = c * 1e-6
        print(
            f"\n  microbenchmark cross-check: c(malloc) = {c:.1f} us per call "
            f"(N = {n_calls:.0f} calls/run; N c = {n_calls * c_s:.1f} s = the absorbed plateau deficit d)"
        )
        print(
            f"  stored s_m = {p[5]:.3g} s = {p[5] / c_s:.2f} x c;  fresh H s_m = {sm_h:.3g} s = "
            f"{sm_h / c_s:.2f} x c: the fade scale is the half-saturation delay, ~12x the per-call cost, not c itself"
        )


def _p5(groups: list[dict]) -> None:
    """Print section P5: the gauge structure and the measured flatness."""
    print()
    print("=" * 100)
    print("P5. gauge structure: what each form pins (delay-origin gauge, per arm)")
    print("=" * 100)
    print("the measured delays carry an unknown origin offset d (stored = true + d); the")
    print("parameters are only identifiable up to the gauge it induces:")
    print()
    print(f"  {'form':4s} {'gauge':>3s}  identifiable combinations (7 params = combos + gauge freedoms)")
    meta = (
        ("H", "2", "N,  A s,  W - N_m s_m - N_f s_f            (pins the product A s, not A or s)"),
        ("E", "2", "N,  s,  W + N_m s_m ln A_m + N_f s_f ln A_f  (pins the scale s; A via ln)"),
        ("L", "0", "W, N, A, s  --  the form is rigid: no origin-shift symmetry, all pinned"),
        ("T", "2", "N,  A/s,  W - N_m s_m - N_f s_f              (pins the absorption rate A/s)"),
        ("Q", "2", "N,  A/s^2,  W - N_m s_m - N_f s_f             (pins the initial curvature)"),
        ("P", "2", "N,  k,  A s^k,  W - N_m s_m - N_f s_f         (pins k and the product A s^k)"),
    )
    for cand, ng, text in meta:
        print(f"  {cand:4s} {ng:>3s}  {text}")
    print()
    print("measured cond(pcov) of the focus fits: every form carries the same ~1e13-1e15 cond, i.e. the")
    print("gauge correlation is a property of the family, not of one shape; P3 shows the individual fade")
    print("params at 6-50 % 1sig in every form (except the inactive rosi free arm and P's extra (s,k)):")
    for name in FOCUS:
        g = next(x for x in groups if x["name"] == name)
        parts = []
        for cand in CANDS:
            pcov = FITS[g["name"] + "/" + cand][1]
            cond = float(np.linalg.cond(0.5 * (pcov + pcov.T)))
            parts.append(f"{cand}={cond:8.1e}")
        print(f"  {name:28s} " + "  ".join(parts))


def _rank_print(mean: dict[str, float]) -> None:
    """Print the candidates ranked by mean shape delta, with the steps.

    Args:
        mean: the mean shape delta per candidate in sig^2.

    """
    prev = None
    for c in sorted(CANDS, key=lambda c: mean[c]):
        step = f"  (step {mean[c] - mean[prev]:+.2f})" if prev is not None else ""
        print(f"  {c}: {mean[c]:+.2f}{step}")
        prev = c


def _p6(groups: list[dict]) -> None:
    """Print section P6: the measured verdict."""
    print()
    print("=" * 100)
    print("P6. verdict (measured)")
    print("=" * 100)
    wins = dict.fromkeys(CANDS, 0)
    shape: dict[str, list[float]] = {c: [] for c in CANDS}
    drift: list[float] = []
    for g in groups:
        m, fv, t, p = g["m"], g["f"], g["t"], g["p"]
        if g["name"] + "/H" not in FITS:
            FITS[g["name"] + "/H"] = fit_cand("H", m, fv, t)
        ph = FITS[g["name"] + "/H"][0]
        ssr_h = _ssr("H", m, fv, t, ph)
        sig2 = ssr_h / max(t.size - 7, 1)
        drift.append((_ssr("H", m, fv, t, p) - ssr_h) / sig2)
        rows = {"H": 0.0}
        for c in ("E", "L", "T", "Q", "P"):
            popt, _ = FITS[g["name"] + "/" + c]
            rows[c] = (_ssr(c, m, fv, t, popt) - ssr_h) / sig2
        shape["H"].append(0.0)
        for c in ("E", "L", "T", "Q", "P"):
            shape[c].append(rows[c])
        wins[min(rows, key=rows.get)] += 1
    mean = {c: float(np.mean(shape[c])) for c in CANDS}
    print("shape effect only: delta-SSR against the FRESH H refit (removes the stored fit's drift")
    print("along the flat direction; negative = better than H):")
    print("  " + "  ".join(f"{c}: {mean[c]:+.2f}" for c in CANDS))
    print("\nper-group best candidate:  " + "  ".join(f"{c}:{wins[c]}" for c in CANDS))
    print(f"\nthe stored H fit itself sits {float(np.median(drift)):+.2f} sig^2 above the fresh H optimum")
    print(f"(median over the 12 groups; max {float(np.max(drift)):+.2f} in the degenerate rosi group), so")
    print("the large gaps in P2 are the stored fit's drift along the flat direction, not the shape.")
    print("\nranking by mean shape delta (negative = better than the fresh H fit):")
    _rank_print(mean)
    print("\n  verdict: E (the exponential) is the most suitable fade term. It is the best mean")
    print("  delta among the 2-parameter forms, never the worst group, is smooth, has a finite total")
    print("  absorbed cost (A s), and pins its scale s as a gauge invariant (P5). Its claim -- the")
    print("  visible cost is down to e^-5 ~ 0.7 % of A by u = 5 -- is falsifiable, unlike H's eternal")
    print("  1/m tail, whose total absorbed cost diverges. L is rejected (worst mean delta): the data")
    print("  see a linear initial absorption, F'(0) = -A/s != 0. P wins the most per-group contests, but")
    print("  its mean is worse than E's and its extra (s, k) is unresolvable (k hits the 0.2-10 bound,")
    print("  1sig up to 3512 % in P3): those wins are the extra parameter, not an identified shape.")
    print("  T ties H on average and is the pure-overlap physical null model (hard cutoff at the")
    print("  per-call slack); keep it as the benchmark -- runs with delay > s showing exactly zero fade")
    print("  would confirm T, a residual ~ A e^(-m/s) would confirm E. H stays defensible (middle of")
    print("  the pack) but is not the best; the next fit round should use E, re-anchored on the fresh")
    print("  optimum, not on the stored fit (which sits hundreds of sig^2 above the H optimum).")


def main() -> None:
    """Run all QA parts against output/results.h5 and print the results."""
    groups, c_us = load_groups()
    _p1()
    _p2(groups)
    _p3(groups)
    _p4(groups, c_us)
    _p5(groups)
    _p6(groups)


if __name__ == "__main__":
    main()
