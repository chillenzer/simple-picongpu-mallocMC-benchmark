"""Diagnose the KelvinHelmholtz small-delay amplification humps from the raw runs.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and
characterizes the two anomalies of the delay-sweep arms, model-free from
the raw run runtimes (the companion document is `khi-humps.md`):

- the **hump** arms, whose runtime at the smallest imposed delay rises
  *above* the (0, 0) baseline plus the nominal N*s stall sum (amplification:
  the run slows more than the sum of the injected stalls), and
- the **over-exposed** arms, whose large-delay line extrapolates to a
  zero-delay intercept *above* the baseline (d < 0).

Everything is computed without fitted parameters: the per-delay excess
E(s) = median(s) - baseline - N*s uses only the per-delay runtime medians,
the (0, 0) baseline median, and the large-delay slope N (the line through
the two largest delays). The diagnosis is therefore immune to the
(W, A, s0) flat direction of the allocation-model fits.

Prints the sections of the diagnosis (A: the per-arm overview with flags,
B: the hump arms' per-delay excess, C: every run's excess at the hump
peak, D: every run's excess at the smallest delay, E: the scaling of the
hump peak, F: the malloc-vs-free comparison of the flagged arms); it
writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from results_io import RESULTS, RUN_TIME, grid_label, load_results, read_table
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY

BANNER = "+" * 35
# The plateau deficit below which an arm counts as over-exposed (s): the
# large-delay line's intercept sits that far above the baseline.
OVER_EXPOSED_S = 0.0


class ArmProfile(NamedTuple):
    """The model-free excess characterization of one (group, arm).

    All times in seconds; `delays` are the arm's distinct imposed delays
    ascending, `medians`/`excess`/`sigma` indexed the same way, and `flags`
    the arm's anomaly class (Hump / OverExp / NonMono, or "insufficient").
    """

    machine: str
    setup: str
    algorithm: str
    grid: str
    arm: str
    n: int
    calls: float
    baseline: float
    deficit: float
    delays: np.ndarray
    medians: np.ndarray
    excess: np.ndarray
    sigma: np.ndarray
    flags: tuple[str, ...]


def _noise_scale(values: np.ndarray) -> float:
    """Return a robust noise scale (MAD-based sigma) of one delay cell.

    Args:
        values: the runtimes (s) of one delay cell.

    Returns:
        float: the cell's noise scale, NaN when the cell has fewer than
        three runs (a MAD of a two-run cell is not a scale).

    """
    if values.size < 3:
        return float("nan")
    mad = float(np.median(np.abs(values - np.median(values))))
    return 1.4826 * mad if mad > 0 else float(np.std(values))


def _flags(excess: np.ndarray, sigma: np.ndarray, deficit: float) -> tuple[str, ...]:
    """Classify an arm's excess curve.

    Args:
        excess: the per-delay excess E(s) (s).
        sigma: the per-delay noise scales (s).
        deficit: the plateau deficit d = baseline - intercept (s).

    Returns:
        tuple: the flags: Hump (E at the smallest delay is positive),
        OverExp (d < 0), NonMono (E rises again beyond its noise), or
        ("insufficient",) when the curve is unusable.

    """
    if not np.all(np.isfinite(excess)):
        return ("insufficient",)
    flags: list[str] = []
    if excess[0] > 0:
        flags.append("Hump")
    if deficit < OVER_EXPOSED_S:
        flags.append("OverExp")
    tolerance = 2.0 * np.maximum(np.abs(sigma[:-1]), np.abs(sigma[1:]))
    if np.any(np.diff(excess) > tolerance):
        flags.append("NonMono")
    return tuple(flags) or ("clean",)


def _profile_arm(
    key: tuple[str, str, str, str],
    arm: str,
    delays_ns: np.ndarray,
    runtimes: np.ndarray,
    baseline: float,
) -> ArmProfile:
    """Build the excess profile of one arm from its raw runs.

    Args:
        key: the arm's group identity (machine, setup, algorithm, grid).
        arm: "malloc" (free delay 0) or "free" (malloc delay 0).
        delays_ns: the arm's imposed delays in nanoseconds.
        runtimes: the arm's runtimes in seconds.
        baseline: the (0, 0) baseline runtime median in seconds.

    Returns:
        ArmProfile: the arm's excess characterization; the slope,
        intercept and excess are NaN when the arm has fewer than four
        distinct delays.

    """
    x = np.asarray(delays_ns, dtype=float) * 1e-9  # ns -> s
    y = np.asarray(runtimes, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    medians = np.array([np.median(y[x == v]) for v in ux])
    sigma = np.array([_noise_scale(y[x == v]) for v in ux])
    if ux.size < 4:
        return ArmProfile(
            *key,
            arm,
            int(x.size),
            float("nan"),
            baseline,
            float("nan"),
            ux,
            medians,
            np.full(ux.shape, float("nan")),
            sigma,
            ("insufficient",),
        )
    calls = (medians[-1] - medians[-2]) / (ux[-1] - ux[-2])
    intercept = medians[-1] - calls * ux[-1]
    deficit = baseline - intercept
    excess = medians - baseline - calls * ux
    return ArmProfile(
        *key,
        arm,
        int(x.size),
        float(calls),
        baseline,
        float(deficit),
        ux,
        medians,
        excess,
        sigma,
        _flags(excess, sigma, deficit),
    )


def _banner(name: str) -> None:
    """Print a section banner.

    Args:
        name: the section's title line.

    """
    print(BANNER)
    print(name)
    print(BANNER)


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    """Print a column-aligned text table.

    Args:
        header: the column titles.
        rows: the rows, one list of strings per row.

    """
    widths = [len(text) for text in header]
    for row in rows:
        for index, text in enumerate(row):
            widths[index] = max(widths[index], len(text))
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))


def _fmt(value: float, spec: str = ".4g") -> str:
    """Format one number, rendering non-finite values as `--`.

    Args:
        value: the value.
        spec: the format spec.

    Returns:
        str: the formatted value.

    """
    return f"{value:{spec}}" if np.isfinite(value) else "--"


def _ns(value_s: float) -> str:
    """Format a seconds value in nanoseconds for the delay axis.

    Args:
        value_s: the value in seconds.

    Returns:
        str: the value in nanoseconds (e.g. `100 ns`).

    """
    return f"{value_s * 1e9:.3g} ns"


def _hump_peak(profile: ArmProfile) -> tuple[int, float, float]:
    """Return the hump peak (index, delay in s, excess in s) of one arm.

    Args:
        profile: a hump arm's profile.

    Returns:
        tuple: (the peak's position among the distinct delays, the peak's
        delay in seconds, the peak's excess in seconds).

    """
    index = int(np.argmax(profile.excess))
    return index, float(profile.delays[index]), float(profile.excess[index])


def collect_profiles(runs: pd.DataFrame) -> list[ArmProfile]:
    """Build the excess profile of every (group, arm) of the sweep runs.

    Args:
        runs: the parsed runs table of the results file.

    Returns:
        list[ArmProfile]: one profile per (machine, setup, algorithm,
        grid, arm) that has both a (0, 0) baseline and arm runs.

    """
    frame = runs.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
    profiles: list[ArmProfile] = []
    for key, grp in frame.groupby(["machine", *GROUP_KEYS], dropna=False):
        machine, setup, algorithm, x, y, z = key
        grid = grid_label(x, y, z)
        m = grp[MALLOC_DELAY].to_numpy(dtype=float)
        f = grp[FREE_DELAY].to_numpy(dtype=float)
        t = grp[RUN_TIME].to_numpy(dtype=float)
        baseline_mask = (m == 0) & (f == 0)
        if not baseline_mask.any():
            continue
        baseline = float(np.median(t[baseline_mask]))
        for arm, mask, delay in (
            ("malloc", (m > 0) & (f == 0), m),
            ("free", (m == 0) & (f > 0), f),
        ):
            if not mask.any():
                continue
            profiles.append(_profile_arm((machine, setup, algorithm, grid), arm, delay[mask], t[mask], baseline))
    return profiles


def _arm_frame(frame: pd.DataFrame, profile: ArmProfile, at_delay_s: float | None = None) -> pd.DataFrame:
    """Select one arm's rows (one profile's group) from a runs table.

    Args:
        frame: the runs table (with the delay columns present).
        profile: the arm's profile (its group identity).
        at_delay_s: restrict to the arm runs at this imposed delay (s),
            or None for every delay.

    Returns:
        pd.DataFrame: the arm's rows.

    """
    grid_parts = profile.grid.split("x")
    x_val = float(grid_parts[0])
    y_val = float(grid_parts[1])
    z_val = float("nan") if len(grid_parts) < 3 else float(grid_parts[2])
    z_mask = frame["z"].isna() if np.isnan(z_val) else (frame["z"] == z_val)
    sub = frame[
        (frame["machine"] == profile.machine)
        & (frame["setup"] == profile.setup)
        & (frame["algorithm"] == profile.algorithm)
        & (frame["x"] == x_val)
        & (frame["y"] == y_val)
        & z_mask
    ]
    m = sub[MALLOC_DELAY].to_numpy(dtype=float)
    f = sub[FREE_DELAY].to_numpy(dtype=float)
    mask = (m > 0) & (f == 0) if profile.arm == "malloc" else (m == 0) & (f > 0)
    if at_delay_s is not None:
        mask &= np.isclose(m + f, at_delay_s * 1e9)
    return sub[mask]


def _baseline_of(frame: pd.DataFrame, profile: ArmProfile) -> float:
    """Return the (0, 0) baseline median of one profile's group.

    Args:
        frame: the runs table (with the delay columns present).
        profile: the arm's profile (its group identity).

    Returns:
        float: the baseline runtime median in seconds.

    """
    grid_parts = profile.grid.split("x")
    x_val = float(grid_parts[0])
    y_val = float(grid_parts[1])
    z_val = float("nan") if len(grid_parts) < 3 else float(grid_parts[2])
    z_mask = frame["z"].isna() if np.isnan(z_val) else (frame["z"] == z_val)
    sub = frame[
        (frame["machine"] == profile.machine)
        & (frame["setup"] == profile.setup)
        & (frame["algorithm"] == profile.algorithm)
        & (frame["x"] == x_val)
        & (frame["y"] == y_val)
        & z_mask
    ]
    m = sub[MALLOC_DELAY].to_numpy(dtype=float)
    f = sub[FREE_DELAY].to_numpy(dtype=float)
    return float(np.median(sub.loc[(m == 0) & (f == 0), RUN_TIME]))


def section_a(profiles: list[ArmProfile]) -> None:
    """Print the per-arm overview with the anomaly flags.

    Args:
        profiles: every arm's profile.

    """
    _banner("A. Arm overview: the model-free excess characterization (E(100 ns) = excess at the smallest delay)")
    rows: list[list[str]] = []
    for profile in profiles:
        if "insufficient" in profile.flags:
            label = f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}"
            rows.append([label] + ["--"] * 4 + ["insufficient"])
            continue
        _, s_peak, e_peak = _hump_peak(profile)
        rows.append(
            [
                f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}",
                _fmt(profile.calls, ".4g"),
                _fmt(profile.deficit, "+.4g"),
                _fmt(profile.excess[0], "+.4g"),
                _fmt(e_peak, "+.4g"),
                f"{', '.join(profile.flags)} (peak {_ns(s_peak)})",
            ]
        )
    _print_table(["arm", "N", "d (s)", "E(100 ns) (s)", "E_peak (s)", "flags"], rows)


def section_b(profiles: list[ArmProfile]) -> list[ArmProfile]:
    """Print the hump arms' per-delay excess; return the hump arms.

    Args:
        profiles: every arm's profile.

    Returns:
        list[ArmProfile]: the hump arms (for the per-run sections).

    """
    humps = [profile for profile in profiles if "Hump" in profile.flags]
    _banner("B. Hump arms: the per-delay excess E(s) = median - baseline - N*s (s)")
    rows: list[list[str]] = []
    for profile in humps:
        index, _, _ = _hump_peak(profile)
        marker = " *" if index in {0, 1, 2} else ""
        per_delay = "  ".join(
            f"{_ns(delay)}:{_fmt(excess, '+.3g')}" for delay, excess in zip(profile.delays, profile.excess, strict=True)
        )
        rows.append(
            [
                f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}{marker}",
                _fmt(profile.calls, ".4g"),
                _fmt(profile.deficit, "+.4g"),
                per_delay,
            ]
        )
    _print_table(["arm (*: peak at the smallest delays)", "N", "d", "E(s) per delay"], rows)
    return humps


def _per_run_excess(frame: pd.DataFrame, profile: ArmProfile, at_delay_s: float) -> np.ndarray:
    """Return the per-run excess at one delay of one arm.

    Args:
        frame: the runs table (with the delay columns present).
        profile: the arm's profile.
        at_delay_s: the imposed delay in seconds.

    Returns:
        np.ndarray: per run, E = t - baseline - N*at_delay_s (s).

    """
    sub = _arm_frame(frame, profile, at_delay_s=at_delay_s)
    if sub.empty:
        return np.array([])
    baseline = _baseline_of(frame, profile)
    return (sub[RUN_TIME] - baseline - profile.calls * at_delay_s).to_numpy(dtype=float)


def section_c(humps: list[ArmProfile], runs: pd.DataFrame) -> None:
    """Print every run's excess at the peak delay of each hump arm.

    The data carry one to three runs per delay cell, so the per-run values
    (not just a summary) are what the reproducibility question rests on: a
    hump present in every run of the cell is not a single bad run.

    Args:
        humps: the hump arms' profiles.
        runs: the parsed runs table of the results file.

    """
    _banner("C. Hump arms: every run's excess at the peak delay (s; per-run E = t - baseline - N*s_peak)")
    frame = runs.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
    rows: list[list[str]] = []
    for profile in humps:
        _, s_peak, _ = _hump_peak(profile)
        values = _per_run_excess(frame, profile, s_peak)
        rows.append(
            [
                f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}",
                _ns(s_peak),
                str(int(values.size)),
                "  ".join(_fmt(value, "+.3g") for value in values),
            ]
        )
    _print_table(["arm", "s_peak", "n", "E of every run"], rows)


def section_d(humps: list[ArmProfile], runs: pd.DataFrame) -> None:
    """Print every run's excess at the smallest delay of each hump arm.

    The hump flag is defined at the smallest delay (100 ns); listing every
    run's excess there shows whether the flag is a single outlier or the
    whole cell.

    Args:
        humps: the hump arms' profiles.
        runs: the parsed runs table of the results file.

    """
    _banner("D. Hump arms: every run's excess at the smallest delay (s; the hump flag's definition point)")
    frame = runs.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
    rows: list[list[str]] = []
    for profile in humps:
        values = _per_run_excess(frame, profile, float(profile.delays[0]))
        rows.append(
            [
                f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}",
                _ns(float(profile.delays[0])),
                str(int(values.size)),
                "  ".join(_fmt(value, "+.3g") for value in values),
            ]
        )
    _print_table(["arm", "s_min", "n", "E of every run"], rows)


def section_e(humps: list[ArmProfile]) -> None:
    """Print the scaling of every hump arm's peak.

    Args:
        humps: the hump arms's profiles.

    """
    _banner("E. Hump arms: the peak's scaling (amplification = E_peak / (N*s_peak), the excess per injected stall sum)")
    rows: list[list[str]] = []
    for profile in humps:
        _, s_peak, e_peak = _hump_peak(profile)
        nominal = profile.calls * s_peak
        cells = float(np.prod(np.array(profile.grid.split("x"), dtype=float)))
        rows.append(
            [
                f"{profile.machine} {profile.setup} {profile.grid} {profile.algorithm} {profile.arm}",
                _ns(s_peak),
                _fmt(e_peak, "+.3g"),
                _fmt(nominal, ".3g"),
                _fmt(e_peak / nominal if nominal > 0 else float("nan"), ".3g"),
                _fmt(e_peak / profile.calls * 1e6 if profile.calls > 0 else float("nan"), ".3g"),
                f"{cells:.3g}",
                _fmt(profile.baseline, ".4g"),
            ]
        )
    _print_table(
        ["arm", "s_peak", "E_peak (s)", "N*s_peak (s)", "E/(N*s)", "E/N (us/call)", "cells", "baseline (s)"],
        rows,
    )


def section_f(profiles: list[ArmProfile]) -> None:
    """Print the malloc-vs-free comparison of the flagged arms.

    Args:
        profiles: every arm's profile.

    """
    _banner("F. Flagged arms: malloc vs free within each (machine, setup, grid, algorithm)")
    rows: list[list[str]] = []
    grouped: dict[tuple[str, str, str, str], dict[str, ArmProfile]] = {}
    for profile in profiles:
        if "clean" in profile.flags or "insufficient" in profile.flags:
            continue
        grouped.setdefault((profile.machine, profile.setup, profile.grid, profile.algorithm), {})[profile.arm] = profile
    for (machine, setup, grid, algorithm), arms in grouped.items():
        for arm in ("malloc", "free"):
            profile = arms.get(arm)
            if profile is None:
                continue
            index, _, e_peak = _hump_peak(profile)
            rows.append(
                [
                    f"{machine} {setup} {grid} {algorithm} {arm}",
                    ", ".join(profile.flags),
                    _fmt(profile.deficit, "+.4g"),
                    _fmt(profile.excess[0], "+.4g"),
                    _fmt(e_peak, "+.4g"),
                    str(index),
                ]
            )
    _print_table(["arm", "flags", "d (s)", "E(100 ns) (s)", "E_peak (s)", "peak idx"], rows)


def main(*, results: Path = RESULTS) -> int:
    """Run the hump diagnosis on one results file and print the sections.

    Args:
        results: the results file, e.g. `output/results.h5`.

    Returns:
        int: the process exit code.

    """
    try:
        file = load_results(results)
    except OSError as err:
        print(f"{err}\nno results file: run `python3 analysis/compute_results.py` first", file=sys.stderr)
        return 1
    with file:
        runs = read_table(file, "runs")
    profiles = collect_profiles(runs)
    if not profiles:
        print("no delay arms found in the results file")
        return 0
    section_a(profiles)
    humps = section_b(profiles)
    if humps:
        section_c(humps, runs)
        section_d(humps, runs)
        section_e(humps)
    section_f(profiles)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose the KelvinHelmholtz small-delay amplification humps from output/results.h5 (prints only)."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results file to diagnose (default: %(default)s)",
    )
    args = parser.parse_args()
    sys.exit(main(results=args.results))
