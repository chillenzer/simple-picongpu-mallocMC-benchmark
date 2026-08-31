"""Refined per-arm table for the report: parametric-bootstrap errors that stay
valid for single-replicate delay cells (a per-cell noise floor from the
measured within-cell spread), plus the shape fits and the scenario comparison."""
from __future__ import annotations

import sys

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, "/workspace")
from critique_fit import group_of, read_table, run_spread  # noqa: E402

pd.set_option("display.width", 260)
RNG = np.random.default_rng(1)
N_ITER = 600


def arm_stats(x, y, t00, t_base, t_arm, s_arm, spread_floor, sigma00):
    N = (y[-1] - y[-2]) / (x[-1] - x[-2])
    b0 = y[-1] - N * x[-1]
    d = t00 - b0
    c = d / N
    E = y - t00 - N * x
    O = y - b0 - N * x
    slope_sm = (y[1] - y[0]) / (x[1] - x[0])
    cells = np.unique(s_arm)
    idx = {v: np.flatnonzero(s_arm == v) for v in cells}

    def sigma(v):
        sel = t_arm[idx[v]]
        s = float(np.std(sel, ddof=1)) if sel.size >= 3 else float("nan")
        return s if np.isfinite(s) and s > 0 else spread_floor

    # parametric bootstrap: each cell ~ N(median, sigma_floor)
    ds = []
    cs = []
    for _ in range(N_ITER):
        s_base = sigma00 if np.isfinite(sigma00) and sigma00 > 0 else spread_floor
        tb = float(np.median(RNG.normal(t00, s_base, len(t_base))))
        yb = np.array([float(np.median(RNG.normal(yv, sigma(v), len(idx[v])))) for v, yv in zip(cells, y)])
        Nb = (yb[-1] - yb[-2]) / (x[-1] - x[-2])
        b0b = yb[-1] - Nb * x[-1]
        ds.append(tb - b0b)
        cs.append((tb - b0b) / Nb if Nb > 0 else np.nan)
    ds = np.asarray(ds)
    cs = np.asarray(cs)

    def hyp(s, a, s0):
        return a * s0 / (s + s0)

    out = {"N": N, "b0": b0, "d": d, "c": c, "E0": E[0], "E1": E[1], "Emax": E[-1],
           "O3rd": O[-3] if len(O) >= 3 else np.nan, "slope_sm_over_N": slope_sm / N,
           "d_err": ds.std(ddof=1), "c_err": np.nanstd(cs, ddof=1)}
    try:
        popt, pcov = curve_fit(hyp, x, O, p0=[max(d, 1e-3), 1e-4],
                               bounds=([0, 1e-9], [np.inf, np.inf]), maxfev=40000)
        a, s0 = (float(v) for v in popt)
        out["hyp_a"] = a
        out["hyp_s0"] = s0
        out["hyp_s0_err"] = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.nan
        out["hyp_ssr"] = float(np.sum((O - hyp(x, a, s0)) ** 2))
    except (RuntimeError, ValueError):
        out.update({"hyp_a": np.nan, "hyp_s0": np.nan, "hyp_s0_err": np.nan, "hyp_ssr": np.nan})
    out["ssr_0pfam"] = float(np.sum((E + d * x / (x + d / N)) ** 2)) if d > 0 else np.nan
    best = None
    for ss in np.logspace(np.log10(x[0] * 0.5), np.log10(x[-1] * 0.99), 300):
        ssr = float(np.sum((O - np.where(x < ss, d, 0.0)) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, float(ss))
    out["step_ssr"] = best[0]
    out["step_s"] = best[1]
    out["ssr_line"] = float(np.sum(O ** 2))
    if d > 0:
        hit = np.flatnonzero(E <= -d / 2)
        out["knee"] = float(x[hit[0]]) if hit.size else np.nan
    else:
        out["knee"] = np.nan
    out["Odec"] = bool(np.all(np.diff(O) < 0.5))
    out["Tmon"] = bool(np.all(np.diff(y) > 0))
    return out


def main():
    f = h5py.File("output/results.h5", "r")
    fits = read_table(f, "fits")
    runs = read_table(f, "runs")
    for col in ("machine", "setup", "algorithm"):
        runs[col] = [v.decode() if isinstance(v, bytes) else v for v in runs[col]]
        fits[col] = [v.decode() if isinstance(v, bytes) else v for v in fits[col]]
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")

    rows = []
    for i, row in fits.iterrows():
        key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
        sub = group_of(runs, key)
        m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
        fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
        t = sub["runtime_s"].to_numpy(float)
        t_base = t[(m == 0) & (fv == 0)]
        t00 = float(np.median(t_base))
        sigma00 = float(np.std(t_base, ddof=1)) if t_base.size >= 3 else np.nan
        spread = run_spread(sub)
        if not np.isfinite(spread):
            spread = 1.0
        name = f"{row['machine']} {row['setup'][:3]} {int(row['x'])}x{int(row['y'])}"
        name += f"x{int(row['z'])}" if not np.isnan(row["z"]) else ""
        alg = row["algorithm"]
        for tag, mask, xcol, fade in (
            ("malloc", (m > 0) & (fv == 0), m, row["m0_ns"] * 1e-9),
            ("free", (m == 0) & (fv > 0), fv, row["f0_ns"] * 1e-9),
        ):
            x = np.unique(xcol[mask])
            y = np.array([float(np.median(t[mask & (xcol == v)])) for v in x])
            if len(x) < 4:
                continue
            r = arm_stats(x, y, t00, t_base, t[mask], xcol[mask], spread, sigma00)
            rows.append({
                "machine": row["machine"], "setup": row["setup"][:3],
                "grid": f"{int(row['x'])}x{int(row['y'])}" + (f"x{int(row['z'])}" if not np.isnan(row['z']) else ""),
                "alg": alg, "arm": tag, "n": int(mask.sum()), "n00": int(t_base.size),
                "T0": t00, "spread": spread,
                "N": r["N"], "d": r["d"], "d_err": r["d_err"],
                "c": r["c"] * 1e6, "c_err": r["c_err"] * 1e6,
                "E100ns": r["E0"], "E10us": r["E1"], "Emax": r["Emax"], "O3rd": r["O3rd"],
                "slsmN": r["slope_sm_over_N"],
                "s0": r["hyp_s0"] * 1e6, "s0_err": r["hyp_s0_err"] * 1e6 if np.isfinite(r["hyp_s0_err"]) else np.nan,
                "a": r["hyp_a"], "ssr_hyp": r["hyp_ssr"], "ssr_0pfam": r["ssr_0pfam"],
                "ssr_step": r["step_ssr"], "ssr_line": r["ssr_line"],
                "s0_stored": fade * 1e6, "knee": r["knee"] * 1e6,
                "Odec": r["Odec"], "Tmon": r["Tmon"],
            })
    df = pd.DataFrame(rows)
    df.to_csv("/tmp/opencode/qa2_arms.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    show = df[["machine", "setup", "grid", "alg"[:4], "arm", "n", "T0", "N", "d", "d_err",
               "c", "c_err", "E100ns", "E10us", "Emax", "O3rd", "slsmN",
               "s0", "a", "ssr_hyp", "ssr_0pfam", "ssr_step", "s0_stored", "knee", "Odec", "Tmon"]]
    print("=== per arm (parametric-bootstrap errors) ===")
    print(show.to_string(index=False))
    print("\nc = d/N in us; s0, s0_stored, knee in us; a in s; E*/O3rd in s; ssr in s^2 (per-delay medians).")

    # scenario comparison
    df["scen"] = df["machine"] + " " + df["setup"] + " " + df["grid"]
    print("\n=== scenario comparison (same machine/setup/grid) ===")
    for name, sub_ in df.groupby("scen"):
        if sub_["alg"].nunique() < 2:
            continue
        print(f"\n{name}")
        for arm in ("malloc", "free"):
            s2 = sub_[sub_["arm"] == arm]
            if s2.shape[0] < 2:
                continue
            Nm = s2["N"].to_numpy()
            print("  " + arm.ljust(6) + "N:   " + "  ".join(f"{a[:4]}={v:.3e}" for a, v in zip(s2["alg"], Nm)))
            print("       d/N:  " + "  ".join(f"{a[:4]}={v:8.1f}+/-{e:6.1f} us" for a, v, e in zip(s2["alg"], s2["c"], s2["c_err"])))
            print("       d:    " + "  ".join(f"{a[:4]}={v:8.2f}+/-{e:6.2f} s" for a, v, e in zip(s2["alg"], s2["d"], s2["d_err"])))
    f.close()


if __name__ == "__main__":
    main()
