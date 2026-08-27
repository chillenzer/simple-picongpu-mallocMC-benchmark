<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Coordination note: results recording redesign (work in progress)

Temporary coordination file for a concurrent agent working on related
benchmarks/analysis changes in this repository. It records the reasoning
and intentions behind the work described here, plus a closing status
report. **Delete this file when the work is merged** (it is meant to
survive exactly the merge window).

Updated: 2026-08-27, branch `picongpu-allocation-time`, base `4c43718`.

## What is being implemented

Two related changes to how benchmark results are recorded and consumed:

1. **Self-describing run logs.** Each benchmark run log gains a single
   machine-parseable metadata line, `# metadata: <compact JSON>`
   (schema 1), emitted at run time by a new `logmeta.py` helper. The
   metadata merges the run's compile-time facts (compiler/CUDA from the
   binary's own `CMakeCache.txt`, paired with the binary's sha256), its
   run-time context (machine, hostname, commit, dependency pins, slurm
   job, example/algorithm, delays, repetitions), and the host hardware
   (GPU, CPU, OS). The human-readable one-liner header (`# run: ...`)
   stays; everything else moves into the JSON.

2. **Legacy cut.** The pre-redesign benchmark logs (268 files, ~800 MB,
   3512 analyzable runs in 8 directories under `output/`, plus an
   archived 190-run nanosleep directory) are frozen once into a local,
   git-ignored `legacy/legacy_results.h5` by `legacy/make_legacy_results.py`.
   The main analysis (`analysis/compute_results.py`) reads that h5 plus
   the new-format logs of the sweep machines from the config machines
   table; it no longer contains the historical multi-layout log parsers
   or the `LEGACY_HARDWARE` directory map.

## Why

- Run logs today are not self-contained: the header carries only short
  (8-char) pin hashes, a short commit, and none of the hardware snapshot
  the mid-era logs used to record (`nvidia-smi` was silently dropped);
  the compiler and CUDA versions are nowhere. Analysis therefore cannot
  tell what any log was built with without external forensics, and two
  logs claiming the same short hash cannot be told apart.
- The analysis hard-codes the legacy world: `LEGACY_HARDWARE` in
  `analysis/compute_results.py` maps legacy output directory names to
  hardware names, and `analysis/run_logs.py` carries three
  historical `cd`-layout regexes. Every new machine or log generation
  touches the analysis.
- Output directories of the sweep machines (`output/hal-sleeptimes`,
  `output/rosi-sleeptimes`) currently hold legacy-era logs indistinguishable
  (by format) from future new ones.
- There is no frozen, reproducible record of the pre-cut data: the
  numbers always re-derive from the raw logs, whose parsing rules are
  spread across the analysis.
- One log per (combination, repetition) also lumps every grid of the
  example's flags file into one file; the redesign gives **one log per
  grid run** (one flags-file line), named with a checksum of that line.

## Binding decisions (agreed with the user)

- **No data in git.** Commit nothing data-related: no `legacy_results.h5`,
  no logs, no manifests. `legacy/` commits only scripts and docs;
  `legacy/logs/` and `legacy/legacy_results.h5` are git-ignored and stay
  local/on the clusters.
- **Nanosleep runs are archived but excluded.** The
  `hal-sleeptimes-nanosleep` runs (190) are parsed into the legacy h5 but
  carry `machine=""` and `hardware=""` and a reason in the h5 attribute
  `excluded_sources`; the analysis drops exactly the rows with empty
  `hardware`, in one single choke point in `compute_results.py`, before
  any table (incl. `rep` numbering) is computed. The exclusion is made
  visible: `summarize_results.py` prints the archived-and-excluded count.
- **Metadata is one compact JSON line** (`# metadata: {...}`, schema 1)
  so parsing is "find the line, `json.load()`". A run log is new format
  **iff** it contains such a line with `"schema": 1` (the cut rule).
- **Compile-time metadata is joined at run time**, not at analysis time:
  `run_stamp.sh` reads `build/<Ex>/<Algo>/CMakeCache.txt` (the same tree
  as the binary it runs; `make clean` removes both together) and merges
  it into the run log, next to the binary's sha256, so the pair
  (binary hash, compiler/CUDA/flags) is unambiguous per log. The
  setup/build logs (`log_setup_<machine>.sh`) also emit a metadata JSON,
  but it is human provenance only — the analysis never reads setup logs.
- **One log per grid run** (per flags-file line), not per
  (combination, repetition). On a re-run of a repetition, the old logs
  of that (machine, example, algorithm, combo, rep) are removed first so
  an interrupted-and-resumed series never leaves duplicate records.
- **Dependency pinning is already final** (commits `1808372`..`e677d96`):
  `requirements.txt` floors + `conda-lock.yml` + committed explicit
  `conda-linux-64.lock`, `make env` order mamba > micromamba > conda.
  Do not re-touch that part of the env workflow.

## The plan (three commits, each left green on its own)

0. This file: reasoning + intentions (committed first).
1. **Legacy isolation.** `legacy/` (scripts + README committed; data
   git-ignored), `make legacy-results` and `make legacy-verify`,
   `compute_results.py` reads the h5 when present but still falls back
   to the legacy directory parsing (deprecation warning) so this commit
   changes no numbers. Gate: the `runs` table and every results table
   identical before/after (a baseline results.h5 was generated on
   `4c43718` before any change).
2. **Cut over.** Remove `LEGACY_HARDWARE` and the historical layout
   fallbacks from the analysis; the strict parser accepts only
   `schema: 1` logs (anything else in a sweep machine directory is an
   error pointing at `make legacy-results`); a missing
   `legacy/legacy_results.h5` is a clear error. `sweep_machines` /
   `machine_titles` attributes must keep naming the machines that have
   run data (hal, rosi today), because `plot_sweeps.py` and
   `plot_shared_fits.py` iterate over them.
3. **Metadata v1 + per-grid logs.** `logmeta.py`; `run_folder.sh` gains
   an optional 6th argument (one flags-file line); `run_stamp.sh` writes
   one log per flags line with the two-line header
   (`# run:` + `# metadata:`) and the stale-log cleanup; the three
   `log_setup_<machine>.sh` gain the setup-side metadata line. Validated
   with a fixture build tree (stub `picongpu`, stub `CMakeCache.txt`).

## Invariants for the merge (do not break these)

- **Numbers.** `output/results.h5` tables (`runs`, `group_stats`, `fits`,
  `shared_fits`, `baselines`, `foil`, `foil_pvalue`, `khi`) must stay
  identical to the `4c43718` baseline for the legacy data (3512 runs,
  file order preserved; `rep` numbering depends on it — machine dirs
  come first, then legacy dirs in `LEGACY_HARDWARE` order, the excluded
  dir last) until the cut-over commit, after which the same tables
  re-derive from `legacy_results.h5` and must still be identical.
- **Run-stamp/resume semantics.** A finished stamp still means "do not
  re-run"; only editing `flags/<Ex>.flags` invalidates a stamped example;
  stamps are keyed only on that (no build dependencies).
- **`make` stays green**: `all` = figures + summary from
  `output/results.h5`; every figure still builds by name.
- **Hardware names** (A30, V100, A100, GH200, `MI250X (1 GCD)`) are the
  paper-figure grouping keys; do not rename them.
- The Amdahl model and `analysis/amdahl.py` are not part of this work.
- No force-push; commits land incrementally on
  `picongpu-allocation-time` (fetch before push).

## Files this work touches (merge-conflict hot spots)

`run_stamp.sh`, `run_folder.sh`, `log_setup_{hal,rosi,rosi_a100}.sh`,
`Makefile` (run/analysis sections only), `analysis/compute_results.py`,
`analysis/run_logs.py`, `analysis/summarize_results.py`, `.gitignore`,
`REUSE.toml`, `README.md`, new: `logmeta.py`, `legacy/`.

## Status

- [x] Baseline generated on `4c43718` (`/tmp/baseline_results.h5`,
  3512 runs; summary captured).
- [ ] Commit 1: legacy isolation.
- [ ] Commit 2: cut over.
- [ ] Commit 3: metadata v1 + per-grid logs.
- [ ] Closing status report (this file, updated + committed at the end).
