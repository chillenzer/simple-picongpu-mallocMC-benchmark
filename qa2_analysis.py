"""Q2 analysis of whether the delay is 'hidden by parallel work'.

What do the plateau deficit d and the fade scale measure, and can we read off
allocation costs?

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Per arm (malloc arm = m>0, f==0; free arm = m==0, f>0) we compute, from the
raw run data only (no fitted parameters except where noted):
  N      = large-delay slope (two largest delays) = calls per run
  b0     = large-delay line intercept
  d      = t00 - b0 = plateau deficit (pure data quantity, gauge-invariant)
  c      = d/N = per-call quantity (gauge-invariant)
  E(s)   = T(s) - t00 - N*s (excess above the nominal line through baseline)
  O(s)   = T(s) - b0 - N*s (offset above the large-delay line; O(0)=d)
and then test the SHAPE of O(s):
  (II) hyperbola  O(s) = a*s0/(s+s0)          (the fitted model's form)
  (L)  sharp knee O(s) = d (s < s*), 0 (s>s*) (literal hiding max(c+s, H))
  (0)  no offset O(s) = 0                    (pure line through the baseline)
and a Monte-Carlo uncertainty on d, c from the run-level replicates.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, "/workspace")
from critique_fit import group_of, read_table

pd.set_option("display.width", 260)
RNG = np.random.default_rng(0)


class ArmSweep(NamedTuple):
    """The per-run data of one arm's Monte-Carlo resampling and shape fits.

    `x`/`y` are the per-delay (delay, runtime median) points, `t00` the
    zero-delay runtime median, `t_base` the zero-delay runtimes, `t_arm`
    the arm's runtimes, and `s_arm` the arm's sleeptimes (s).
    """

    x: np.ndarray
    y: np.ndarray
    t00: float
    t_base: np.ndarray
    t_arm: np.ndarray
    s_arm: np.ndarray


class GroupData(NamedTuple):
    """The per-run data of one (group, algorithm) sweep, for both arms.

    `m`/`fv` are the per-run malloc/free sleeptimes (s), `t` the runtimes
    (s), `t00` the zero-delay runtime median, `t_base` the zero-delay
    runtimes, `name` the group label, and `row` the group's `fits` table
    row.
    """

    m: np.ndarray
    fv: np.ndarray
    t: np.ndarray
    t00: float
    t_base: np.ndarray
    name: str
    row: pd.Series


def _resample(sweep: ArmSweep, x: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample the per-cell medians to get the Monte-Carlo errors of d and c.

    Args:
        sweep: the arm's per-run data.
        x: the per-delay values in seconds.
        n_iter: the number of resamples.

    Returns:
        tuple: the (d, c) arrays of the resamples.

    """
    cells = np.unique(sweep.s_arm)
    idx: dict[np.float64, np.ndarray] = {v: np.flatnonzero(sweep.s_arm == v) for v in cells}
    ds = []
    cs = []
    for _ in range(n_iter):
        tb = float(np.median(sweep.t_base[RNG.choice(len(sweep.t_base), len(sweep.t_base), replace=True)]))
        yb = np.array([float(np.median(sweep.t_arm[RNG.choice(idx[v], len(idx[v]), replace=True)])) for v in cells])
        Nb = (yb[-1] - yb[-2]) / (x[-1] - x[-2])
        b0b = yb[-1] - Nb * x[-1]
        db = tb - b0b
        ds.append(db)
        cs.append(db / Nb if Nb > 0 else np.nan)
    return np.asarray(ds), np.asarray(cs)


def _hyp_fit(x: np.ndarray, off: np.ndarray, d: float) -> tuple[float, float, float] | None:
    """Fit the hyperbola shape O(s) = a*s0/(s+s0) to the offset curve.

    Args:
        x: the per-delay values (s).
        off: the offset above the large-delay line (s).
        d: the plateau deficit (s), the p0 floor for a.

    Returns:
        tuple: (a, s0, the shape SSR), or None when the fit does not
        converge.

    """

    def hyp(s: np.ndarray, a: float, s0: float) -> np.ndarray:
        return a * s0 / (s + s0)

    try:
        popt, _ = curve_fit(hyp, x, off, p0=[max(d, 1e-3), 1e-4], bounds=([0, 1e-9], [np.inf, np.inf]), maxfev=40000)
    except RuntimeError, ValueError:
        return None
    a, s0 = (float(v) for v in popt)
    return (a, s0, float(np.sum((off - hyp(x, a, s0)) ** 2)))


def _hyp_dpin_fit(x: np.ndarray, e: np.ndarray, d: float) -> tuple[float, float] | None:
    """Fit the fixed-deficit hyperbola O(s) = -d*s/(s+s0) to the excess curve.

    Args:
        x: the per-delay values (s).
        e: the excess over the nominal line (s).
        d: the plateau deficit (s), pinned in the shape.

    Returns:
        tuple: (s0, the shape SSR), or None when the fit does not converge.

    """
    try:
        popt, _ = curve_fit(lambda s, s0: -d * s / (s + s0), x, e, p0=[1e-4], bounds=([1e-9], [np.inf]), maxfev=40000)
    except RuntimeError, ValueError:
        return None
    return (float(popt[0]), float(np.sum((e + d * x / (x + popt[0])) ** 2)))


def _step_scan(x: np.ndarray, off: np.ndarray, d: float) -> tuple[float, float] | None:
    """Scan the sharp-knee delay s* by minimizing the sharp-knee shape SSR.

    The shape is O(s) = d for s < s*, 0 for s >= s*.

    Args:
        x: the per-delay values (s).
        off: the offset above the large-delay line (s).
        d: the plateau deficit (s).

    Returns:
        tuple: (the best SSR, the knee delay s*).

    """
    best: tuple[float, float] | None = None
    for ss in np.logspace(np.log10(x[0] * 0.5), np.log10(x[-1] * 0.99), 300):
        ssr = float(np.sum((off - np.where(x < ss, d, 0.0)) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, float(ss))
    return best


def per_arm(sweep: ArmSweep, n_iter: int = 400) -> dict:
    """Compute one arm's plateau deficit, per-call slack, and shape fits.

    Args:
        sweep: the arm's per-run data.
        n_iter: the number of Monte-Carlo resamples.

    Returns:
        dict: N, b0, d, c, the E offsets at small/mid/max delay, the
        O_last3 offset, slope_sm_over_N, the Monte-Carlo d_err/c_err, the
        shape fits (hyp, hyp_dpin), the sharp-knee scan (step), ssr0, the
        data knee, and the O_decreasing/monotone_T flags.

    """
    x = np.asarray(sweep.x, float)
    y = np.asarray(sweep.y, float)
    N = (y[-1] - y[-2]) / (x[-1] - x[-2])
    b0 = y[-1] - N * x[-1]
    d = sweep.t00 - b0
    c = d / N
    E = y - sweep.t00 - N * x
    off = y - b0 - N * x
    slope_sm = (y[1] - y[0]) / (x[1] - x[0])
    ds, cs = _resample(sweep, x, n_iter)
    res = {
        "N": N,
        "b0": b0,
        "d": d,
        "c": c,
        "E0": E[0],
        "E1": E[1],
        "Emax": E[-1],
        "O_last3": off[-3] if len(off) >= 3 else np.nan,
        "slope_sm_over_N": slope_sm / N,
        "d_err": ds.std(),
        "c_err": cs.std(),
    }
    res["hyp"] = _hyp_fit(x, off, d)
    res["hyp_dpin"] = _hyp_dpin_fit(x, E, d)
    res["step"] = _step_scan(x, off, d)
    res["ssr0"] = float(np.sum(off**2))
    if d > 0:
        hit = np.flatnonzero(-d / 2 >= E)
        res["knee_data"] = float(x[hit[0]]) if hit.size else float("nan")
    else:
        res["knee_data"] = float("nan")
    res["O_decreasing"] = bool(np.all(np.diff(off) < 0.5))
    res["monotone_T"] = bool(np.all(np.diff(y) > 0))
    return res


def _group_data(runs: pd.DataFrame, row: pd.Series) -> GroupData:
    """Extract one fits row's group of runs into per-arm sweep data.

    Args:
        runs: the parsed runs table.
        row: one row of the `fits` table.

    Returns:
        GroupData: the group's per-run data and the fits row.

    """
    key = (row["machine"], row["setup"], row["algorithm"], row["x"], row["y"], row["z"])
    sub = group_of(runs, key)
    m = sub["malloc_sleeptime"].to_numpy(float) * 1e-9
    fv = sub["free_sleeptime"].to_numpy(float) * 1e-9
    t = sub["runtime_s"].to_numpy(float)
    t_base = t[(m == 0) & (fv == 0)]
    t00 = float(np.median(t_base))
    name = f"{row['machine']} {row['setup'][:3]} {int(row['x'])}x{int(row['y'])}"
    name += f"x{int(row['z'])}" if not np.isnan(row["z"]) else ""
    name += f" {row['algorithm']}"
    return GroupData(m, fv, t, t00, t_base, name, row)


def _arm_rows(g: GroupData) -> list[dict]:
    """Build the per-arm table rows of one group.

    Args:
        g: the group's per-run data and fits row.

    Returns:
        list: one row per arm with at least 4 distinct delays.

    """
    rows: list[dict] = []
    for tag, mask, xcol, fade in (
        ("malloc", (g.m > 0) & (g.fv == 0), g.m, g.row["m0_ns"] * 1e-9),
        ("free", (g.m == 0) & (g.fv > 0), g.fv, g.row["f0_ns"] * 1e-9),
    ):
        x = np.unique(xcol[mask])
        y = np.array([float(np.median(g.t[mask & (xcol == v)])) for v in x])
        if len(x) < 4:
            continue
        r = per_arm(ArmSweep(x, y, g.t00, g.t_base, g.t[mask], xcol[mask]))
        smin = float(x[x > 0].min())
        rows.append(
            {
                "group": g.name,
                "arm": tag,
                "n": int(mask.sum()),
                "smin_ns": smin * 1e9,
                "smax_ms": x[-1] * 1e3,
                "N": r["N"],
                "t00": g.t00,
                "d": r["d"],
                "d_err": r["d_err"],
                "c_us": r["c"] * 1e6,
                "c_err_us": r["c_err"] * 1e6,
                "E100ns": r["E0"],
                "E10us": r["E1"],
                "Emax": r["Emax"],
                "O3rd": r["O_last3"],
                "slsmN": r["slope_sm_over_N"],
                "hyp_a": r["hyp"][0] if r["hyp"] else np.nan,
                "hyp_s0_us": r["hyp"][1] * 1e6 if r["hyp"] else np.nan,
                "hyp_ssr": r["hyp"][2] if r["hyp"] else np.nan,
                "step_s_us": r["step"][1] * 1e6 if r["step"] else np.nan,
                "step_ssr": r["step"][0] if r["step"] else np.nan,
                "ssr0": r["ssr0"],
                "stored_s0_us": fade * 1e6,
                "knee_us": r["knee_data"] * 1e6,
                "Odec": r["O_decreasing"],
                "Tmon": r["monotone_T"],
            }
        )
    return rows


def _print_per_arm(df: pd.DataFrame) -> None:
    """Print the per-arm (data-anchored) table.

    Args:
        df: the per-arm rows.

    """

    def frow(v: float, fmt: str = "{:.4g}") -> str:
        return fmt.format(v) if np.isfinite(v) else "-"

    df["ratio_step_hyp"] = df["step_ssr"] / df["hyp_ssr"]
    df["s0_vs_c"] = df["hyp_s0_us"] / df["c_us"]
    out = df.copy()
    for c in (
        "N",
        "d",
        "d_err",
        "E100ns",
        "E10us",
        "Emax",
        "O3rd",
        "slsmN",
        "hyp_a",
        "hyp_s0_us",
        "hyp_ssr",
        "step_s_us",
        "step_ssr",
        "ssr0",
        "stored_s0_us",
        "knee_us",
        "ratio_step_hyp",
        "s0_vs_c",
    ):
        out[c] = out[c].map(frow)
    out["c"] = (out["c_us"]).map(frow)
    out["c_err"] = out["c_err_us"].map(frow)
    print("=== A. per arm (data-anchored) ===")
    print(
        out[
            [
                "group",
                "arm",
                "n",
                "smin_ns",
                "smax_ms",
                "N",
                "d",
                "d_err",
                "c",
                "c_err",
                "E100ns",
                "E10us",
                "Emax",
                "O3rd",
                "slsmN",
                "hyp_a",
                "hyp_s0_us",
                "hyp_ssr",
                "step_s_us",
                "step_ssr",
                "ratio_step_hyp",
                "stored_s0_us",
                "s0_vs_c",
                "knee_us",
                "Odec",
                "Tmon",
            ]
        ].to_string(index=False)
    )
    print("\nE100ns/E10us/Emax/O3rd in s; hyp_a in s; hyp_s0/step_s/knee/s0 stored in us;")
    print("c in us (d/N); slsmN = slope(100ns->10us)/N; ratio_step_hyp = SSR(step)/SSR(hyp);")
    print("s0_vs_c = fitted fade scale / (d/N); Odec = offset decreasing; Tmon = T monotone.")


def _print_sign_test(df: pd.DataFrame) -> None:
    """Print the malloc-vs-free sign test of the two readings, per group.

    Args:
        df: the per-arm rows.

    """
    print("\n=== B. malloc vs free within a group: sign test of the two readings ===")
    print("  (II) native-cost reading: c = d/N, expect c_free < c_malloc  (d/N)_f < (d/N)_m")
    print("  (H)  hiding reading:     d/N = H - c, expect (d/N)_f > (d/N)_m")
    g = df.groupby("group")
    for name, sub_ in g:
        if sub_["arm"].nunique() < 2:
            continue
        cm = sub_.loc[sub_["arm"] == "malloc", "c_us"].iloc[0]
        cf = sub_.loc[sub_["arm"] == "free", "c_us"].iloc[0]
        dm = sub_.loc[sub_["arm"] == "malloc", "d"].iloc[0]
        df_ = sub_.loc[sub_["arm"] == "free", "d"].iloc[0]
        print(
            f"  {name:42s} (d/N)_m={cm:8.1f} us  (d/N)_f={cf:8.1f} us  diff(f-m)={cf - cm:+8.1f} us"
            f"   d_m={dm:6.1f}s d_f={df_:6.1f}s"
        )


def _print_n_consistency(df: pd.DataFrame) -> None:
    """Print the N consistency and d/N spread across algorithms, per scenario.

    Args:
        df: the per-arm rows.

    """
    print("\n=== C. same (machine,setup,grid): N consistency and d/N spread across algorithms ===")
    df2 = df.copy()
    df2["scen"] = df2["group"].str.replace("FlatterScatter|ScatterAlloc", "", regex=True).str.strip()
    for name, sub_ in df2.groupby("scen"):
        if sub_["group"].nunique() < 2:
            continue
        for arm in ("malloc", "free"):
            s2 = sub_[sub_["arm"] == arm]
            if s2.shape[0] < 2:
                continue
            Nm = s2["N"].to_numpy()
            cm = s2["c_us"].to_numpy()
            nm = (Nm.max() - Nm.min()) / Nm.mean() * 100
            print(
                f"  {name:28s} {arm:6s} N spread={nm:5.2f}%   d/N: "
                + "  ".join(f"{a[-12:]}={v:.1f}us" for a, v in zip(s2["group"], cm, strict=True))
            )


def main() -> None:
    """Print the Q2 per-arm analysis, the sign test, and the N consistency."""
    f = h5py.File("output/results.h5", "r")
    fits = read_table(f, "fits")
    runs = read_table(f, "runs")
    for col in ("machine", "setup", "algorithm"):
        runs[col] = [v.decode() if isinstance(v, bytes) else v for v in runs[col]]
        fits[col] = [v.decode() if isinstance(v, bytes) else v for v in fits[col]]
    runs["z"] = pd.to_numeric(runs["z"], errors="coerce")
    rows: list[dict] = []
    for _, row in fits.iterrows():
        rows.extend(_arm_rows(_group_data(runs, row)))
    df = pd.DataFrame(rows)
    f.close()
    _print_per_arm(df)
    _print_sign_test(df)
    _print_n_consistency(df)


if __name__ == "__main__":
    main()
