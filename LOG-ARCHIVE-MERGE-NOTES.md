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

### Done

- [x] Append-only harness (`run_stamp.sh` no longer removes the earlier
  vintages, the same-second name collision takes a `.1`, `.2`, ...
  suffix) + session logs to `<output dir>/sessions/` (six launchers).
- [x] `superseded` flag (stamp-derived, `RUN_VINTAGE_COLUMNS`) + the
  seven metadata columns (`log`, `nominal_rep` via
  `RUN_NOMINAL_COLUMNS`, `flag_sha`, `hw_os`, `cxx_flags`,
  `cuda_flags`, `build_type` in `RUN_SOURCE_COLUMNS`).
- [x] The frozen legacy rows default to the log-derived columns
  (empty strings / no nominal repetition) and `superseded = 0`,
  applied to the legacy frame before the runs-table concat.
- [x] The summary line `Runs: N (M superseded)` in
  `summarize_results.py`; README (analysis and `make runs` sections)
  documents the vintage model, the columns and the
  `runs[runs["superseded"] == 0]` filter.

### Closing report

The work is complete on `picongpu-allocation-time` on top of
`d12a58a`. Gates: with the sweep machines' output directories empty,
every pre-existing column of every results table is identical (values
and dtypes) to the `d12a58a` baseline after a re-freeze of the local
legacy table with `make legacy-results`; only the appended `runs`
columns, one summary line and the re-padded no-delay table are new. A
disposable fixture (stub `picongpu` binary emitting the real
`initialization time:` / `calculation ... simulation time:` /
`full simulation time:` lines, a git repository, a flags file with
`-s`) was driven through two plain re-runs and three vintages of one
identity: the older vintages are all parsed and stay in the `runs`
table with `superseded = 1` (their `rep` numbers them in file order
behind the current vintage, which is the stamp's only entry,
`superseded = 0`); a pre-seeded same-second log name is not clobbered
(the new log takes the `.1` suffix); a session log in `sessions/` is
not parsed; a flags edit changes `flag_sha` on the re-run's rows while
`binary_sha256` / `commit` stay stable, and a stub-binary change
changes `binary_sha256`. Nothing downstream filters superseded rows;
the numbers of the current vintages are unchanged by the work.

Remaining for the merge window: none of this work's decisions blocks
other changes to `analysis/` or the launchers; the one behavioral
contract to respect when touching the run logs is that make never
removes or overwrites a run log (the analysis keys the vintage state
on the file names and the stamps). **Delete this file when the work
is merged.**
