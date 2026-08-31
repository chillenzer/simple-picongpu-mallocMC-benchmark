"""Print summary tables of the computed benchmark results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and
prints the total run count with its superseded share
(`Runs: N (M superseded)`) and, when any, the excluded archived runs;
then the summary tables: per machine, the parsed runs (with `--raw`) and
the group runtime statistics; the performance-model fit tables, a
comparison of the shared-parameter (combined) fits against the individual
fits, the slack-ratio summary (f = A/T0, the absorbed delay over the
zero-delay runtime — a convention-dependent ratio, not a runtime budget),
and the per-arm absorbed-delay slack (the plateau deficit d and the
per-call c = d/N, the gauge-invariant Tier-2 quantities); the No-delay
runtimes; the FoilLCT / KelvinHelmholtz figure statistics; and, from the
microbenchmark suite (the `alloc_cost` table, empty when the frozen
microbench table is absent), the native per-call allocation costs of
every allocator as the mean milliseconds per operation at each
allocation size, one table per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from results_io import RESULTS, load_results, no_delay_mask, read_table, scenario_key, scenario_name

BANNER = "+" * 35


def print_table(name: str, text: str) -> None:
    """Print a results frame under a banner.

    Args:
        name: the table's banner line.
        text: the table text to print.

    """
    print(BANNER)
    print(name)
    print(BANNER)
    print(text)
    print()


def print_fraction_summary(fits: pd.DataFrame) -> None:
    """Print the slack ratio f = A/T0 of every fit with a ratio in (0, 1).

    f is the absorbed delay as a share of the zero-delay runtime T0 — a
    convention-dependent ratio, not a runtime budget and not the native
    allocation cost (analysis-review.md).

    Args:
        fits: the fits table of the results file.

    """
    fit = fits[fits["f_malloc"].between(0, 1, inclusive="neither") | fits["f_free"].between(0, 1, inclusive="neither")]
    if len(fit):
        # Name the algorithm only when more than one is present: single-policy
        # output stays exactly as before multi-algorithm sweeps.
        show_algorithm = fits["algorithm"].nunique() > 1
        print("\nSlack ratio (f = A/T0, the absorbed delay over the zero-delay runtime):")
        for _, r in fit.iterrows():
            fractions = []
            for name in ("f_malloc", "f_free"):
                if r[name] == r[name]:
                    err = "" if r[f"{name}_err"] != r[f"{name}_err"] else f" +/- {100 * r[f'{name}_err']:.1f}"
                    fractions.append(f"{name} = {100 * r[name]:.1f}{err}%")
            extra = [f"A_{tag} = {r[f'A_{tag}']:.2f} s" for tag in ("malloc", "free") if r[f"A_{tag}"] == r[f"A_{tag}"]]
            algorithm = f"{r.algorithm:<14s} " if show_algorithm else ""
            print(
                f"  {r.machine:<14s} {r.setup:<16s} {algorithm}grid {int(r.x)}x{int(r.y)}"
                + (f"x{int(r.z)}" if pd.notna(r.z) else "")
                + f" [{r.model}] : "
                + ", ".join(fractions)
                + f"   (W = {r.W:.2f} s"
                + (", " + ", ".join(extra) if extra else "")
                + ")"
            )
            if r["note"]:
                print(f"      note: {r.note}")


def _fmt_param(value: float, err: float) -> str:
    """Format one fitted value with its standard error, or `?` if unavailable.

    Args:
        value: the fitted value.
        err: the standard error.

    Returns:
        str: `value +/- err`, the bare value, or `?` when unavailable.

    """
    if pd.isna(value):
        return "?"
    if pd.isna(err):
        return f"{value:.3g}"
    return f"{value:.3g} +/- {err:.2g}"


def _compare_params(srow: pd.Series, irow: pd.Series, name: str) -> str:
    """One algorithm's fitted value against the shared value, with agreement.

    Args:
        srow: a row carrying the shared values (W, N_malloc, N_free).
        irow: a row carrying the algorithm's individual values.
        name: the parameter name ("W", "N_malloc", or "N_free").

    Returns:
        str: `value +/- err (d +/-x.x%, overlap|separate)`.

    """
    svalue, serr = srow[name], srow[f"{name}_err"]
    ivalue, ierr = irow[name], irow[f"{name}_err"]
    if pd.notna(ivalue) and svalue and abs(svalue) >= 0.01 * abs(ivalue):
        diff = f"d {100 * (ivalue - svalue) / abs(svalue):+.1f}%"
    else:
        diff = "d ?"
    ci = "? CI" if pd.isna(ierr) or pd.isna(serr) else "overlap" if abs(ivalue - svalue) <= ierr + serr else "separate"
    return f"{_fmt_param(ivalue, ierr)} ({diff}, {ci})"


def print_shared_fit_summary(shared_fits: pd.DataFrame, fits: pd.DataFrame) -> None:
    """Print the shared-parameter fit of every multi-algorithm scenario.

    For each (machine, setup, grid) scenario with more than one algorithm,
    the shared values of W, N_malloc, and N_free are compared against each
    algorithm's individual fit: the fit values with their standard errors,
    the relative difference from the shared value, and whether the two
    1-sigma intervals overlap.

    Args:
        shared_fits: the `shared_fits` table of the results file.
        fits: the individual fits table of the results file.

    """
    if shared_fits.empty:
        return

    def key(r: pd.Series) -> tuple:
        return (r["machine"], scenario_key(r["setup"], r["x"], r["y"], r["z"]), r["algorithm"])

    shared: dict[tuple, list[pd.Series]] = {}
    for _, r in shared_fits.iterrows():
        shared.setdefault(key(r)[:2], []).append(r)
    individuals = {key(r): r for _, r in fits.iterrows() if pd.notna(r["model"])}
    print("\nShared-parameter fit (W, N_malloc, N_free fit across the algorithms):")
    for (machine, scen), rows in shared.items():
        ref = rows[0]
        print(
            f"  {machine:<14s} {scenario_name(ref['setup'], ref['x'], ref['y'], ref['z']):<24s} "
            f"[{ref['model']}, {ref['n_runs']} runs, r2 = {ref['r2']:.3f}]"
        )
        for name in ("W", "N_malloc", "N_free"):
            parts = [f"shared = {_fmt_param(ref[name], ref[f'{name}_err'])}"]
            for a in rows:
                ind = individuals.get((machine, scen, a["algorithm"]))
                if ind is None or pd.isna(ind[name]):
                    parts.append(f"{a['algorithm']} = --")
                else:
                    parts.append(f"{a['algorithm']} = {_compare_params(ref, ind, name)}")
            print(f"    {name}: " + " | ".join(parts))
        if ref["note"]:
            print(f"    note: {ref['note']}")


def _print_alloc_cost(alloc_cost: pd.DataFrame) -> None:
    """Print the microbenchmark's native per-call allocation costs, one table per run.

    The mean milliseconds per operation at each allocation size, pivoted
    with the allocators (and their operation) as rows and the allocation
    sizes as columns.

    Args:
        alloc_cost: the `alloc_cost` table of the results file.

    """
    if alloc_cost.empty:
        return
    for jobid, group in alloc_cost.groupby("jobid", sort=True):
        hardware = str(group["hardware"].iloc[0])
        pivot = group.pivot_table(index=["allocator", "operation"], columns="size_bytes", values="mean_ms")
        pivot = pivot.reindex(columns=sorted(pivot.columns))
        print_table(
            f"Native allocation cost: run {int(jobid)} ({hardware}, mean ms per operation by size)",
            pivot.to_string(float_format=lambda v: f"{v:10.4g}"),
        )


def _print_fits_ca(fits_ca: pd.DataFrame) -> None:
    """Print the secondary A = N*c_a constrained (native-cost) fit, if any.

    Args:
        fits_ca: the `fits_ca` table of the results file.

    """
    if fits_ca.empty:
        return
    print_table(
        "A = N*c_a constrained fits (native-cost reading; c_a from the microbenchmark)",
        fits_ca.to_string(index=False, float_format=lambda v: f"{v:10.3g}"),
    )


def main(*, results: Path = RESULTS, raw: bool = False) -> int:
    """Print the summary tables of one results file.

    Args:
        results: the results file, e.g. `output/results.h5`.
        raw: also print the raw parsed runs per machine.

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
        group_stats = read_table(file, "group_stats")
        fits = read_table(file, "fits")
        fits_ca = read_table(file, "fits_ca") if "fits_ca" in file else pd.DataFrame()
        shared_fits = read_table(file, "shared_fits") if "shared_fits" in file else pd.DataFrame()
        absorption = read_table(file, "absorption") if "absorption" in file else pd.DataFrame()
        alloc_cost = read_table(file, "alloc_cost") if "alloc_cost" in file else pd.DataFrame()
        foil = read_table(file, "foil")
        foil_pvalue = read_table(file, "foil_pvalue")
        khi = read_table(file, "khi")
        excluded_sources = json.loads(file.attrs.get("excluded_sources", "{}"))
        excluded_runs = file.attrs.get("excluded_runs")
    if runs.empty:
        print("no runs found in the results file")
        # The microbenchmark table is independent of the PIConGPU runs, so
        # it is printed even when there are none.
        _print_alloc_cost(alloc_cost)
        return 0
    if "superseded" in runs:
        print(f"Runs: {len(runs)} ({int(runs['superseded'].sum())} superseded)")
    if excluded_sources:
        print(f"Archived, excluded runs: {excluded_runs}")
        for name, reason in sorted(excluded_sources.items()):
            print(f"  {name}: {reason}")
        print()
    for machine in runs["machine"].drop_duplicates():
        # The legacy paper-world runs carry no sweep machine (empty label);
        # they have no group statistics of their own.
        label = machine if machine else "legacy"
        if raw:
            print_table(f"Parsed runs: {label}", runs[runs["machine"] == machine].to_string(index=False))
        stats = group_stats[group_stats["machine"] == machine]
        if len(stats):
            print_table(f"Group statistics: {label}", stats.to_string(index=False))
    print_table("Fits", fits.to_string(index=False, float_format=lambda v: f"{v:10.3g}"))
    _print_fits_ca(fits_ca)
    print_shared_fit_summary(shared_fits, fits)
    print_fraction_summary(fits)
    if len(absorption):
        print_table(
            "Absorbed delay per arm (d = plateau deficit in s, c = d/N in us per call)",
            absorption.to_string(index=False, float_format=lambda v: f"{v:10.3g}"),
        )
    print_table("No-delay runtimes", runs[no_delay_mask(runs)].to_string(index=False))
    print_table(
        "Foil metadata",
        foil.merge(foil_pvalue, on="hardware", how="left").to_string(index=False),
    )
    print_table("Khi metadata", khi.to_string(index=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print summary tables of the computed benchmark results (output/results.h5)."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results file to summarize (default: %(default)s)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="also print the raw parsed runs per machine",
    )
    args = parser.parse_args()
    sys.exit(main(results=args.results, raw=args.raw))
