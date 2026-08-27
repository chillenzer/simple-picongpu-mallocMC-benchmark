<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Merge notes: append-only run vintages (work in progress)

Temporary coordination file for concurrent work in this repository. It
records the reasoning and intentions behind the change described here,
plus a closing status report. **Delete this file when the work is merged**
(it is meant to survive exactly the merge window).

Updated: 2026-08-27, branch `picongpu-allocation-time`, base `d12a58a`.

## What is being implemented

Up to now, re-running a finished (machine, example, algorithm, delay
combination, repetition) **removed** the earlier logs of that identity
before writing the new ones (`rm -f` in `run_stamp.sh`). The requirement
is: no run data may ever be lost or overridden — superseded runs must
stay available and stay in the analysis, filterable by their metadata.

1. **Append-only runs.** `run_stamp.sh` no longer removes anything. A
   re-run of an identity writes the new vintage of the logs next to the
   older ones in the machine's output directory; a same-second re-run
   appends a counter to the file name instead of clobbering. All vintages
   are parsed, so all of them end up in `output/results.h5`; nothing
   downstream filters them out — they are extra data.
2. **`superseded` marker.** The stamp (`run-stamps/<m>/<Ex>/<Algo>/<m>_<f>/rep-<i>.stamp`,
   content = the run's log paths) already tracks the *current* vintage of
   an identity. The analysis now reads it: a runs row whose log is listed
   in the identity's stamp gets `superseded = 0`; a row whose identity has
   a stamp that does not list its log gets `1`; an identity without a
   stamp (fresh data, or after `make clean-runs`) is all `0`. Frozen
   legacy (h5) rows are `0` by construction.
3. **Remaining full-metadata columns.** On top of the provenance columns
   already landed (`started_utc`, `commit`, `binary_sha256`, `picongpu`,
   `mallocmc`, `gpu`, `gpu_driver`, `cuda_version`, `cpu`, `compiler`,
   `host`, `slurm_job`), the `runs` table gains the log-derived columns
   that make vintages filterable end to end: `log` (the source file
   name), `nominal_rep` (the repetition declared by the run), `flag_sha`
   (the flags-file line's 8-hex sha — the one filter that separates a
   flags-edit re-run when binary and commit are identical), `hw_os`,
   `cxx_flags`, `cuda_flags`, `build_type` — plus the stamp-derived
   `superseded`. All of these are already in every schema-1 metadata
   line, so `logmeta.py` is unchanged.
4. **Session logs -> `sessions/`.** The launchers
   (`log_{setup,run}_*.sh`) write their session logs into
   `<output dir>/sessions/` instead of the output directory proper. They
   are a free-text backup of the launch environment, never re-read by the
   analysis (which parses the top-level run logs only), and never
   superseded — they are archived from the moment of creation.

## Why

- A re-run today destroys the previous measurement of the same identity:
  the numbers it produced can no longer be recovered or compared.
- The vintages of an identity differ in exactly the dimensions the
  metadata records (the binary's sha256 after a rebuild, the commit after
  a repo move, the flags line after an edit, the time). With the columns
  above, any vintage can be selected out of the `runs` table without
  re-reading the logs.
- The superseded marker is derived, not stored: the stamp is already the
  single source of truth for "what is the current vintage" (make's
  skip-check reads it), so the analysis just reads the same fact.

## Invariants for the merge

- No log file is ever removed or overwritten by make. The only `rm` left
  in `run_stamp.sh` is gone; `make clean` / `distclean` still touch only
  derived artifacts (build/, figures/, results.h5, src/), never logs.
- The analysis input set is still exactly the top-level files of each
  sweep machine's output directory (`sessions/` is a subdirectory and is
  not parsed; the `kind: "setup"` guard still applies if a session log is
  ever parsed).
- Stamp/resume semantics are unchanged: a finished stamp still means "do
  not re-run"; only editing `flags/<Ex>.flags` invalidates runs; a new
  consumer of the stamp (the analysis) never writes it.
- The pre-existing columns of every results table (historical columns
  plus the landed `RUN_METRIC_COLUMNS`/`RUN_SOURCE_COLUMNS`) keep their
  values and dtypes for existing data; this change only appends columns
  to the `runs` table and adds one summary line.
- `legacy/` (the frozen table and its verify) is untouched by this work.
- No force-push; commits land incrementally on `picongpu-allocation-time`
  (fetch before push, rebase + re-gate on new upstream).

## Files this work touches

`run_stamp.sh`, `log_{setup,run}_{hal,rosi,rosi_a100}.sh`, `Makefile`
(run section only), `README.md`, `analysis/results_io.py` (column
constants), `analysis/run_logs.py` (records), `analysis/compute_results.py`
(`RUNS_COLUMNS`, the superseded derivation), `analysis/summarize_results.py`
(one line), new: `LOG-ARCHIVE-MERGE-NOTES.md` (this file).

## Status

- [ ] Commit 1: append-only harness + `sessions/`.
- [ ] Commit 2: `superseded` flag + the seven metadata columns.
- [ ] Commit 3: closing status report (this file, updated at the end).
