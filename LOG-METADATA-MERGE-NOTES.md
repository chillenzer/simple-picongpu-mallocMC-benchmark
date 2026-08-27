<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# TEMPORARY — merge notes: run-log provenance metadata in `results.h5`

> **This file is temporary merge scaffolding.** It exists so a separate
> agent performing a related re-organisation can merge the log-metadata
> change without losing intent. **Delete this file once the merge is done**
> (it is not part of the feature and not referenced by any script).

## What this change does

Every run stored in `output/results.h5` now carries the **provenance of the
run log it came from**: when and where it ran, and what was measured. The
values are parsed out of each historical log layout by a new module
(`analysis/log_meta.py`) and copied onto every run row of that log. In
addition, each completed run now records two metrics the old parser dropped
(the full and the initialisation simulation times) and the number of
simulation steps (`-s`) it ran.

This is a **log-side + analysis-side** change:

- *analysis side* (reads existing logs): new `log_meta.py` parser; the
  `runs` table gains provenance columns and two timing metrics.
- *log side* (writes future logs): `run_stamp.sh` now records a richer
  metadata header (start datetime, host, slurm nodes, GPU name+driver,
  loaded modules, compiler) and the build writes a `.source-stamp` marking
  the exact source hashes the binary was built from.

## Files touched

| File | Change |
| --- | --- |
| `analysis/log_meta.py` | **NEW.** Generation-aware parser: one provenance dict per log file. |
| `analysis/run_logs.py` | Records now also capture `full_runtime_s`, `init_time_s`, `sim_steps`; yield point moved to the `full simulation time:` line. |
| `analysis/results_io.py` | New column constants `RUN_METRIC_COLUMNS`, `RUN_SOURCE_COLUMNS`. |
| `analysis/compute_results.py` | `_parse_dir` merges the per-log provenance onto each run row; new columns wired into `RUNS_COLUMNS`. |
| `run_stamp.sh` | Richer metadata header; sources the machine profile once to record compiler + modules. |
| `Makefile` | Binary build rule also writes `build/<Ex>/<Algo>/.source-stamp` (the checked-out source hashes). |
| `README.md` | Documents the new columns and the log-side header. |

## The new `runs` columns

Added by `compute_results.RUNS_COLUMNS` (order in the stored table):

- **after `runtime_s`, before `rep`** (`RUN_METRIC_COLUMNS`):
  `full_runtime_s`, `init_time_s`, `sim_steps`.
- **at the end, after `rep`** (`RUN_SOURCE_COLUMNS`):
  `started_utc`, `commit`, `binary_sha256`, `picongpu`, `mallocmc`,
  `gpu`, `gpu_driver`, `cuda_version`, `cpu`, `compiler`, `host`,
  `slurm_job`.

All existing columns and their order are **unchanged**; the new columns are
additive. Text columns are `""` when a log generation does not carry the
value; numeric ones are `NaN`. The column *names* are the contract — the
single source of truth is `results_io.RUN_METRIC_COLUMNS` /
`RUN_SOURCE_COLUMNS`, which `compute_results` imports.

## Design decisions and trade-offs (read before touching these)

1. **Provenance is stored per-run-row, not in a separate table.** A run
   belongs to exactly one log file, and a file's identity is its path, so
   the per-log dict is broadcast onto all of that file's runs. Trade-off:
   the values repeat within a file (redundant) but the `runs` table stays
   self-describing — consumers can filter/group/join by provenance without
   a second table. If the re-organisation introduces a dedicated
   `log_metadata` table instead, the broadcast in `_parse_dir` (the
   `frame.merge(pd.DataFrame(rows).set_index("name"), on="name")` line) is
   the place to swap out.

2. **The set of analysed runs is deliberately unchanged.** This is the most
   important invariant. The old `run_logs.parse_log` yielded one record per
   `calculation  simulation time:` line. The rewrite yields at the
   `full simulation time:` line (so the init and full times, which bracket
   the calculation time, attach to the right record). Runs that *aborted
   before* printing a calculation time would now produce a record with no
   `runtime_s`; to keep the analysed set identical, `_flush()` only yields a
   pending record **if it has `runtime in s`**. Net effect: same 3557 runs,
   same `runtime_s` values; only the new columns differ. **Do not remove the
   `_flush()` runtime guard** — dropping it would add the aborted runs and
   change every fit/statistic.

3. **Concurrent (multi-GPU) rosi logs.** rosi runs several combinations at
   once, so the `initialisation`/`calculation`/`full` time lines of
   different runs interleave in one log. `_time_line()` takes the **first**
   value of each line type for a record, which is exactly the value the old
   parser recorded. Changing this to "last wins" or "match by run id" would
   silently change the fitted runtimes.

4. **Provenance value precedence.** For the PIConGPU / mallocMC version:
   `# source:` (the actually checked-out hashes) **supersedes** the
   `# picongpu:` / `# mallocmc:` pinned 8-char hashes when present; otherwise
   the config.json pin (8-char) is the fallback. `log_meta.parse_log_metadata`
   applies the pin *only if* the log carried nothing. Compiler: the header
   `# cxx:` (the real compiler) wins over the traced `spack load gcc@..` /
   `module load ..` line, which in turn wins over the machine's `config.json`
   `modules` list (the last-resort hint). This ladder is intentional
   (measured > checked-out > pinned > configured).

5. **Generation-aware parser, absent fields are `""`.** `log_meta._detect`
   classifies a log as `run-stamp` (leading `# key: value` header),
   `logging-env` (legacy `git log`+`nvidia-smi`+`hwinfo` dump), `slurm`
   (`slurm-<id>.out`), or `unknown` (plain `set -x`, e.g.
   `output/hal/log_*.txt`). Whatever a generation does not carry comes out
   empty. The `unavailable` sentinel a probe wrote is mapped to `""`. A
   log's `generation` tag is returned by the parser but is **not** stored as
   a column (it is per-log diagnostics).

6. **`run_stamp.sh` now sources the machine profile.** Previously only
   `run_folder.sh` sourced it; now `run_stamp.sh` sources it once up front so
   the header can record `${LOADEDMODULES}` and the compiler
   (`${CXX:-cc} --version`). Consequences / trade-offs:
   - the profile is sourced **twice** per run (once per script) — idempotent
     but not free; acceptable because a run is expensive.
   - under `set -e`, a failing profile (missing spack/module) aborts the
     stamp before the header is written. That matches the old behaviour
     (a failed profile also fails the run, which then writes no stamp), so
     the resume logic is unchanged.
   - `nvidia-smi` (for `# gpu:`) and the compiler probe are both guarded so
     a machine without them yields `unavailable` (→ `""`) rather than a
     non-zero exit.

7. **`.source-stamp` and the `$$$$` escaping.** The stamp is written by the
   *binary build* rule, not the run rule. That rule body lives inside a
   `define`/`$(eval $(call ...))`, so the recipe text is expanded by make
   **twice**; a shell `$(git ...)` command substitution therefore needs
   `$$$$` in the source (make collapses it once at eval, once at recipe
   time, leaving `$(` for the shell). I verified this exact construct in a
   minimal `.ONESHELL` define/eval Makefile. If the re-organisation moves or
   inlines this rule, re-check the dollar escaping.

## Invariants the merge must preserve

- The `runs` table keeps **exactly 3557 rows** and every pre-existing column
  is **row-identical** to before this change. (Verified by diffing the old
  and new `results.h5` — see below. If the merge adds new logs/directories
  the row count will legitimately change, but the *parsing* of the existing
  logs must not change.)
- All 134 pre-existing HDF5 datasets (fits, group_stats, baselines, foil,
  khi, shared_fits, …) are **value-identical**.
- New columns are additive and name-addressed; no consumer relies on their
  positions. Keep `RUN_METRIC_COLUMNS`/`RUN_SOURCE_COLUMNS` as the name
  source of truth and in sync with `compute_results.RUNS_COLUMNS`.
- The `_flush()` "only yield a record that has `runtime in s`" guard and the
  `_time_line()` first-value-wins behaviour (invariants 2 and 3 above).

## What I verified

- `make` from a clean `output/results.h5`: builds the file plus all 8
  figures, exit 0.
- Old-vs-new `results.h5` diff: **134 shared datasets identical, 0
  differences; the 15 new provenance/metric datasets only in the new file.**
  `runs` table: 3557 rows, all shared columns row-identical.
- Spot-checked parsed provenance per machine against the logs:
  `hal`→A30 / gcc 12.3.0 / EPYC 7452; `rosi`→V100 / nvidia-compilers
  25.1-CUDA-12.6.0; rosi-sleeptimes→CUDA 12.8; slurm logs→correct job id +
  gcc 12.2.0; MI250X (lumi) has no compiler line (none logged) → `""`.
- `log_meta` unit-level: a synthetic `run_stamp.sh` header parses to the
  right commit/gpu/driver/cuda/host/slurm_job; `unavailable`→`""`;
  multi-GPU `# gpu:` takes the first GPU.
- Lint: `ruff check analysis/` clean (select ALL + preview), `shellcheck`
  clean, `shfmt -i 2 -ci` clean, `reuse lint` 41/41 (new file carries the
  SPDX header).
- The `.source-stamp` escaping verified in an actual minimal make run (see
  decision 7).

## Open risks / not verified here

- **No real picongpu build / no GPU machine in this container.** The
  analysis-side parsing is verified against the historical logs and a
  synthetic header; the *log-side* (`run_stamp.sh` header + `.source-stamp`)
  is verified only structurally (escaping, `set -e` guards, path under
  `.ONESHELL`). A real `make build` + `make runs` on hal/rosi is the
  outstanding end-to-end check: confirm the header is written, `.source-stamp`
  is created and read into `# source:`, and the new log parses with
  generation `run-stamp`.
- The profile is now sourced in `run_stamp.sh` (decision 6). Watch for any
  machine whose profile is not idempotent or has side effects under a second
  source; none were observed in the hal/rosi/hemera profiles.
- `# modules:` is written to the log for human reference but is **not** a
  `runs` column (it is per-log and not needed for the analysis); the
  compiler is derived from it only as the last-resort config hint.

## Status report (final)

- **State: implementation complete and verified to the extent possible in
  this container.** No plan changes were needed during implementation; the
  only corrections made while wiring it up were internal to the parsing
  (the aborted-run guard, the concurrent first-value-wins rule, and a
  generator-style refactor to satisfy the repo's ruff `ALL` ruleset without
  `noqa`).
- No unexpected findings that change the feature. The historical log
  layouts all parse as intended; the one generation with a *new* format
  (`run-stamp`) has no local examples yet, so it is covered by the synthetic
  header test rather than a real log.
- The change is committed as its own feature commit on this branch (see
  git log); this notes file is committed separately and is **not** pushed,
  per the session convention. The outstanding end-to-end check (a real
  `make build` + `make runs` on a GPU machine) remains for whoever has
  machine access.
- **For the merge agent:** the highest-value things to keep are
  `log_meta.py`, the `_flush()`/`_time_line()` behaviour, the
  `RUN_*_COLUMNS` constants, and the `$$$$` escaping in the Makefile
  binary-build rule. Everything else (README text, the `run_stamp.sh`
  header) is descriptive and can flex with your re-organisation.
