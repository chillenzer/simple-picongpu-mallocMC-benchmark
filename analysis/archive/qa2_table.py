# ARCHIVED: hyperbola-era model-review analysis, superseded by
# analysis/performance_model.py; not part of the published pipeline and not
# maintained. See analysis/archive/README.md. Restored with notes-pre-publish.
"""Refined per-arm table for the report.

Parametric-bootstrap errors that stay valid for single-replicate delay cells
(a per-cell noise floor from the measured within-cell spread), plus the shape
fits and the scenario comparison.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, "/workspace")
from critique_fit import group_of, read_table, run_spread

pd.set_option("display.width", 260)
RNG = np.random.default_rng(1)
N_ITER = 600
ARMS_CSV = Path("/tmp/opencode/qa2_arms.csv")  # ruff: ignore[hardcoded-temp-file]


class GroupData(NamedTuple):
    """The per-run data of one (group, algorithm) sweep, for both arms.

    `m`/`fv` are the per-run malloc/free sleeptimes (s), `t` the runtimes
    (s), `t00` the zero-delay runtime median, `t_base` the zero-delay
    runtimes, `spread` the per-run noise floor, `sigma00` the zero-delay
    runtime std, and `row` the group's `fits` table row.
    """

    m: np.ndarray
    fv: np.ndarray
    t: np.ndarray
    t00: float
    t_base: np.ndarray
    spread: float
    sigma00: float
    row: pd.Series


class ArmData(NamedTuple):
    """The per-arm data of one arm_stats call.

    `x`/`y` are the per-delay (delay, runtime median) points, `t00` the
    zero-delay runtime median, `t_base` the zero-delay runtimes, `t_arm`
    the arm's runtimes, `s_arm` the arm's sleeptimes (s), `spread_floor`
    the noise floor, and `sigma00` the zero-delay runtime std.
    """

    x: np.ndarray
    y: np.ndarray
    t00: float
    t_base: np.ndarray
    t_arm: np.ndarray
    s_arm: np.ndarray
    spread_floor: float
    sigma00: float


def _bootstrap_errors(data: ArmData) -> tuple[np.ndarray, np.ndarray]:
    """Resample the plateau deficit d and the slack c = d/N parametrically.

    Each cell ~ N(median, sigma_floor).

    Args:
        data: the arm's per-delay and per-run data.

    Returns:
        tuple: the (d, c) arrays of the N_ITER resamples.

    """
    cells = np.unique(data.s_arm)
    idx: dict[np.float64, np.ndarray] = {v: np.flatnonzero(data.s_arm == v) for v in cells}

    def sigma(v: np.float64) -> float:
        sel = data.t_arm[idx[v]]
        s = float(np.std(sel, ddof=1)) if sel.size >= 3 else float("nan")
        return s if np.isfinite(s) and s > 0 else data.spread_floor

    ds = []
    cs = []
    for _ in range(N_ITER):
        s_base = data.sigma00 if np.isfinite(data.sigma00) and data.sigma00 > 0 else data.spread_floor
        tb = float(np.median(RNG.normal(data.t00, s_base, len(data.t_base))))
        yb = np.array(
            [float(np.median(RNG.normal(yv, sigma(v), len(idx[v])))) for v, yv in zip(cells, data.y, strict=True)]
        )
        Nb = (yb[-1] - yb[-2]) / (data.x[-1] - data.x[-2])
        b0b = yb[-1] - Nb * data.x[-1]
        ds.append(tb - b0b)
        cs.append((tb - b0b) / Nb if Nb > 0 else np.nan)
    return np.asarray(ds), np.asarray(cs)


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


def _hyp_fit(x: np.ndarray, off: np.ndarray, d: float) -> dict[str, float]:
    """Fit the hyperbola shape O(s) = a*s0/(s+s0) to the offset curve.

    Args:
        x: the per-delay values (s).
        off: the offset above the large-delay line (s).
        d: the plateau deficit (s), the p0 floor for a.

    Returns:
        dict: hyp_a, hyp_s0, hyp_s0_err, hyp_ssr (all NaN when the fit
        does not converge).

    """

    def hyp(s: np.ndarray, a: float, s0: float) -> np.ndarray:
        return a * s0 / (s + s0)

    try:
        popt, pcov = curve_fit(hyp, x, off, p0=[max(d, 1e-3), 1e-4], bounds=([0, 1e-9], [np.inf, np.inf]), maxfev=40000)
    except RuntimeError, ValueError:
        return {"hyp_a": np.nan, "hyp_s0": np.nan, "hyp_s0_err": np.nan, "hyp_ssr": np.nan}
    a, s0 = (float(v) for v in popt)
    return {
        "hyp_a": a,
        "hyp_s0": s0,
        "hyp_s0_err": float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.nan,
        "hyp_ssr": float(np.sum((off - hyp(x, a, s0)) ** 2)),
    }


def arm_stats(data: ArmData) -> dict:
    """Compute one arm's plateau deficit, per-call slack, and shape fits.

    Args:
        data: the arm's per-delay and per-run data.

    Returns:
        dict: N, b0, d, c, the E/O offsets at small/mid/max delay, the
        bootstrap d_err/c_err, the hyperbola fit (hyp_*), the SSRs of the
        shape models (ssr_*), the knee delay, and the Odec/Tmon
        monotonicity flags.

    """
    x, y = data.x, data.y
    N = (y[-1] - y[-2]) / (x[-1] - x[-2])
    b0 = y[-1] - N * x[-1]
    d = data.t00 - b0
    c = d / N
    E = y - data.t00 - N * x
    off = y - b0 - N * x
    slope_sm = (y[1] - y[0]) / (x[1] - x[0])
    ds, cs = _bootstrap_errors(data)
    out = {
        "N": N,
        "b0": b0,
        "d": d,
        "c": c,
        "E0": E[0],
        "E1": E[1],
        "Emax": E[-1],
        "O3rd": off[-3] if len(off) >= 3 else np.nan,
        "slope_sm_over_N": slope_sm / N,
        "d_err": ds.std(ddof=1),
        "c_err": np.nanstd(cs, ddof=1),
    }
    out.update(_hyp_fit(x, off, d))
    out["ssr_0pfam"] = float(np.sum((E + d * x / (x + d / N)) ** 2)) if d > 0 else np.nan
    step = _step_scan(x, off, d)
    out["step_ssr"] = step[0]
    out["step_s"] = step[1]
    out["ssr_line"] = float(np.sum(off**2))
    if d > 0:
        hit = np.flatnonzero(-d / 2 >= E)
        out["knee"] = float(x[hit[0]]) if hit.size else np.nan
    else:
        out["knee"] = np.nan
    out["Odec"] = bool(np.all(np.diff(off) < 0.5))
    out["Tmon"] = bool(np.all(np.diff(y) > 0))
    return out


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
    sigma00 = float(np.std(t_base, ddof=1)) if t_base.size >= 3 else np.nan
    spread = run_spread(sub)
    if not np.isfinite(spread):
        spread = 1.0
    name = f"{row['machine']} {row['setup'][:3]} {int(row['x'])}x{int(row['y'])}"
    name += f"x{int(row['z'])}" if not np.isnan(row["z"]) else ""
    return GroupData(m, fv, t, t00, t_base, spread, sigma00, row)


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
        r = arm_stats(ArmData(x, y, g.t00, g.t_base, g.t[mask], xcol[mask], g.spread, g.sigma00))
        rows.append(
            {
                "machine": g.row["machine"],
                "setup": g.row["setup"][:3],
                "grid": f"{int(g.row['x'])}x{int(g.row['y'])}"
                + (f"x{int(g.row['z'])}" if not np.isnan(g.row["z"]) else ""),
                "alg": g.row["algorithm"],
                "arm": tag,
                "n": int(mask.sum()),
                "n00": int(g.t_base.size),
                "T0": g.t00,
                "spread": g.spread,
                "N": r["N"],
                "d": r["d"],
                "d_err": r["d_err"],
                "c": r["c"] * 1e6,
                "c_err": r["c_err"] * 1e6,
                "E100ns": r["E0"],
                "E10us": r["E1"],
                "Emax": r["Emax"],
                "O3rd": r["O3rd"],
                "slsmN": r["slope_sm_over_N"],
                "s0": r["hyp_s0"] * 1e6,
                "s0_err": r["hyp_s0_err"] * 1e6 if np.isfinite(r["hyp_s0_err"]) else np.nan,
                "a": r["hyp_a"],
                "ssr_hyp": r["hyp_ssr"],
                "ssr_0pfam": r["ssr_0pfam"],
                "ssr_step": r["step_ssr"],
                "ssr_line": r["ssr_line"],
                "s0_stored": fade * 1e6,
                "knee": r["knee"] * 1e6,
                "Odec": r["Odec"],
                "Tmon": r["Tmon"],
            }
        )
    return rows


def _print_scenario_comparison(df: pd.DataFrame) -> None:
    """Print the scenario comparison of the arms across algorithms.

    Args:
        df: the per-arm table.

    """
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
            print(
                "  "
                + arm.ljust(6)
                + "N:   "
                + "  ".join(f"{a[:4]}={v:.3e}" for a, v in zip(s2["alg"], Nm, strict=True))
            )
            print(
                "       d/N:  "
                + "  ".join(
                    f"{a[:4]}={v:8.1f}+/-{e:6.1f} us" for a, v, e in zip(s2["alg"], s2["c"], s2["c_err"], strict=True)
                )
            )
            print(
                "       d:    "
                + "  ".join(
                    f"{a[:4]}={v:8.2f}+/-{e:6.2f} s" for a, v, e in zip(s2["alg"], s2["d"], s2["d_err"], strict=True)
                )
            )


def main() -> None:
    """Build the per-arm table (writing qa2_arms.csv) and print it."""
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
    df.to_csv(ARMS_CSV)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    show = df[
        [
            "machine",
            "setup",
            "grid",
            "alg"[:4],
            "arm",
            "n",
            "T0",
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
            "s0",
            "a",
            "ssr_hyp",
            "ssr_0pfam",
            "ssr_step",
            "s0_stored",
            "knee",
            "Odec",
            "Tmon",
        ]
    ]
    print("=== per arm (parametric-bootstrap errors) ===")
    print(show.to_string(index=False))
    print("\nc = d/N in us; s0, s0_stored, knee in us; a in s; E*/O3rd in s; ssr in s^2 (per-delay medians).")
    _print_scenario_comparison(df)
    f.close()


if __name__ == "__main__":
    main()
