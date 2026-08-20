"""Fit sleeptime-sweep runtimes to the Amdahl allocation model.

Model
-----
Each of the N allocation calls on the (serial, host-side) critical path takes
its native cost c_a plus the imposed delay s, so

    T(s) = W + A + N*s,   A = N*c_a (native allocation time)

with W the runtime without any allocation cost. The native part is not needed
for large sleeptimes (it is negligible against the imposed delay), but it acts
as a correction at small sleeptimes, so the sweep is fitted with

    T(s) = W + N*s + A*s0/(s+s0)            (Amdahl model)

which reduces to the Amdahl line W + N*s for s >> s0 and to W + A for
s -> 0. The parameters are (W, N, A, s0): the baseline runtime, the number
of allocation calls per run, the native allocation time, and the sleeptime
scale over which the native cost fades. The Amdahl fraction of the runtime
spent in allocations is

    f = A / (W + A)      (native allocation time / total time at zero delay)

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
per-allocation cost c_a (ns) is known, pass it: A = N*c_a is used instead of
the fitted correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["fit_allocation_fraction", "fit_sweep", "main"]

GROUP_KEYS = ("setup", "x", "y", "z")


def _fit_lsq(h, s, t):
    """Least squares for t = W + N*s + A*h; returns (W, N, A, ss_res)."""
    sol, *_ = np.linalg.lstsq(np.vstack([np.ones_like(s), s, h]).T, t, rcond=None)
    res = t - (sol[0] + sol[1] * s + sol[2] * h)
    return (*[float(v) for v in sol], float(np.sum(res**2)))


def fit_allocation_fraction(sleeptimes, runtimes, c_a: float | None = None) -> dict:
    """Fit one sleeptime sweep to the Amdahl model.

    Returns a dict with W, N, A, s0, T0, f, r2, warnings, and the data plus
    the per-point residuals.
    """
    s = np.asarray(sleeptimes, dtype=float) * 1e-9  # ns -> s
    t = np.asarray(runtimes, dtype=float)
    if s.shape != t.shape:
        raise ValueError("sleeptimes and runtimes must have the same shape")
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    order = np.argsort(s)
    s, t = s[order], t[order]
    warnings = []
    if s.size < 3:
        raise ValueError("need at least 3 data points")

    def finish(W, N, A, s0, res, note=None):
        r2 = 1.0 - float(np.sum(res**2)) / ss_tot if ss_tot > 0 else float("nan")
        if N <= 0:
            warnings.append("non-positive slope: no allocation cost visible in this sweep")
        if A < -1e-6 * max(abs(W), 1e-9):
            warnings.append(
                "negative native allocation time: the smallest-sleeptime runtime lies below "
                "the Amdahl line extrapolated from the larger sleeptimes, so the data do not "
                "follow T = W + A + N*s (e.g. the allocation count itself changes with the delay)"
            )
        if note:
            warnings.append(note)
        if s.size < 6:
            warnings.append(f"only {s.size} points: the parameters are poorly constrained")
        W, N, A, s0 = float(W), float(N), float(A), (float(s0) if s0 == s0 else float("nan"))
        return {
            "W": W,  # runtime without allocation cost (s)
            "N": N,  # allocation calls per run
            "A": A,  # native allocation time (s)
            "s0": s0,  # fade scale of the native correction (s)
            "T0": W + A,  # total runtime at zero delay (s)
            "f": (A / (W + A)) if (W + A) > 0 else float("nan"),  # Amdahl fraction
            "r2": r2,
            "warnings": warnings,
            "sleeptimes": s,
            "runtimes": t,
            "residuals": res,
        }

    if s.size == 3:
        # Not enough points to determine the correction shape: fit the Amdahl
        # line to the two largest (native ~ 0) points; A is the residual at the
        # smallest sleeptime.
        (N, W), *_ = np.linalg.lstsq(np.vstack([s[-2:], np.ones(2)]).T, t[-2:], rcond=None)
        A = float(t[0] - (W + N * s[0]))
        return finish(W, N, A, float("nan"), t - (W + N * s),
                      note="only 3 points: correction shape not identifiable; "
                           "N and W from the two largest, A the residual at the smallest sleeptime")

    # Grid search over log s0; (W, N, A) are linear given s0. s0 is capped to
    # half the sweep range so the correction stays within the measured span and
    # A is identifiable (a larger s0 would make it a constant offset degenerate
    # with W).
    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = max(0.05 * s_min_pos, 1e-12)
    hi = 0.5 * (s.max() - s.min())
    best = None
    for s0 in np.logspace(np.log10(min(lo, hi)), np.log10(max(lo, hi)), 80):
        W, N, A, ss_res = _fit_lsq(s0 / (s + s0), s, t)
        if best is None or ss_res < best[0]:
            best = (ss_res, float(s0), W, N, A)
    _, s0, W, N, A = best
    if c_a is not None:
        A = N * c_a * 1e-9  # ns -> s
    res = t - (W + N * s + A * s0 / (s + s0))
    if s0 >= max(lo, hi) * 0.999:
        warnings.append(
            "s0 reached the search cap: the native correction does not clearly fade within "
            "the sweep, so A and W (hence f) are weakly constrained"
        )
    return finish(W, N, A, s0, res)


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
            row.update({"N": np.nan, "W": np.nan, "A": np.nan, "f": np.nan,
                        "T0": np.nan, "r2": np.nan, "s0_ns": np.nan,
                        "note": "fewer than 2 distinct sleeptimes"})
            rows.append(row)
            continue
        try:
            res = fit_allocation_fraction(grp["sleeptime"], grp["runtime in s"], c_a=c_a)
        except ValueError as err:
            row.update({"N": np.nan, "W": np.nan, "A": np.nan, "f": np.nan,
                        "T0": np.nan, "r2": np.nan, "s0_ns": np.nan,
                        "note": str(err)})
            rows.append(row)
            continue
        row.update({k: res[k] for k in ("W", "N", "A", "T0", "f", "r2")})
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
        print(table.to_string(index=False))
    ok = table[table["f"].between(0, 1, inclusive="neither")]
    if len(ok):
        print("\nAmdahl fraction of runtime spent in allocations (f = A/T0):")
        for _, r in ok.iterrows():
            print(
                f"  {r.setup:<16s} grid {int(r.x)}x{int(r.y)}"
                + (f"x{int(r.z)}" if pd.notna(r.z) else "")
                + f" : f = {100 * r.f:.1f}%   (N = {r.N:.3e} allocations, W = {r.W:.2f} s, A = {r.A:.2f} s)"
            )
            if r["note"]:
                print(f"      note: {r.note}")


if __name__ == "__main__":
    main()
