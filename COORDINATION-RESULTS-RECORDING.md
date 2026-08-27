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
- [x] Commit 1: legacy isolation (`3eec6ae`).
- [x] Commit 2: cut over (`e82fc09`).
- [x] Commit 3: metadata v1 + per-grid logs (`762391a`).
- [x] Closing status report (this file, updated + committed at the end).

## Closing status report (2026-08-27)

All three commits are on `picongpu-allocation-time` and pushed. Every
commit left the full pre-commit suite green and `make` green; no data
files were committed.

### Validation evidence

- **Commit 1** (`3eec6ae`): both input modes (transitional directory
  parsing, frozen h5) produce all results tables identical to the
  `4c43718` baseline (values and dtypes); `make legacy-verify` OK
  (3702 runs frozen from 279 files, excluded: 1 directory); `make`
  output differs only by the new "Archived, excluded runs: 190"
  summary lines.
- **Commit 2** (`e82fc09`): after the cut, all 8 data tables identical
  to the baseline; a missing `legacy/legacy_results.h5` is a clear
  error (exit 1, points at `make legacy-results`); a pre-redesign log
  left in a sweep machine's output directory raises
  `LegacyLogError` with the move-to-legacy hint; `sweep_machines` /
  `machine_titles` still name `hal, rosi`.
- **Commit 3** (`762391a`): fixture sandbox (stub `picongpu` emitting
  a `calculation  simulation time:` line, stub `CMakeCache.txt`,
  stub compiler/nvcc version binaries): `make runs` wrote one log per
  flags-file line (36 of them), each with the two-line header and a
  parseable schema-1 metadata line (40-hex commit + dirty, full pins,
  binary sha256, compiler/CUDA/flags/build type from the binary's own
  cache, GPU/CPU/OS snapshot, flags line + its 8-hex sha + command);
  stale-log cleanup removed a planted old-naming log before the
  re-run and only the unstamped (combination, repetition) re-ran; the
  stamp carries one log path per line. Strict parser on that
  directory: exactly 36 records; a `kind: "setup"` session log
  contributes 0 records; a legacy-format file raises
  `LegacyLogError`. End-to-end `compute_results.py` +
  `summarize_results.py` on the fixture: runs table = 3548 rows
  (3512 frozen legacy + 36 new), `sweep_machines` lists the fixture
  machine with its title, excluded runs 190. In this checkout the
  sweep output directories are empty, so `make legacy-verify` and the
  full `make` are unchanged: every table identical to the baseline,
  `make` output differing only by commit 1's excluded-runs lines.

### Deviations from the plan above

1. **`log_run_<machine>.sh` also gained the metadata line.** The plan
   listed only the `log_setup_*` launchers, but the run launchers'
   session logs also land in the machine's output directory, which
   `compute_results.py` globs file-by-file; without a
   `kind: "setup"` line the strict parser would reject them. (The
   parser's docstring already anticipated "build or full-series
   launch" session logs.)
2. **`REUSE.toml` got an annotations block** (`legacy/logs/**`,
   `legacy/legacy_results.h5`, MIT) instead of relying on
   `.gitignore`: the reuse-tool 6.2.0 hook discovers ignored files via
   `git ls-files ... --ignored --others --directory`, and git omits a
   *nested* fully-ignored directory when its parent has untracked
   non-ignored files, so the hook walked the frozen logs and failed.
3. **Sweep labels** are kept when the log directory exists **or** the
   frozen table has rows for the machine (so `hal`/`rosi` survive with
   empty directories and the plotting drivers keep working).
4. **Frozen parser quirks are intentional**: pandas 3.0 upcasts
   int64→float64 when a 0-record file frame carries the per-file
   `name` column, so `legacy/make_legacy_results.py` keeps the
   historical per-file frame construction; and its `--check` mode
   normalizes the text columns before comparing (an h5 round-trip
   turns missing text into `""` while a fresh parse yields NaN).
5. **Log naming**: the line hash is the first 8 hex of the sha256 of
   the exact flags line (no trailing newline), and the stale-log
   cleanup pattern matches the pre-redesign naming as well (it has no
   line-hash component).

### Machine-side onboarding (for whoever runs the next benchmarks)

On a fresh checkout on a machine: `bash legacy/move_legacy_logs.sh`
(if the relocation was not done there yet), then
`make legacy-results`; `make legacy-verify` reports whether any
frozen input file changed. New runs since the cut write one log per
grid line as described above — old-format logs must not be dropped
into a sweep machine's output directory (the analysis rejects them
on purpose).
