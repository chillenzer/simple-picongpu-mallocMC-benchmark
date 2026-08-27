<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Legacy benchmark data (frozen)

This directory freezes the pre-redesign benchmark logs so the main
analysis reads the legacy runs from one stable table instead of
re-parsing historical log layouts.

- `move_legacy_logs.sh` — one-time relocation: the historical
  per-cluster output directories under `output/` are moved whole into
  `logs/`, and the sweep machines' output directories are carved out
  (legacy-era files only; the directories stay in place for the new
  runs). New-format logs (those with a `# metadata:` JSON line) are
  never touched.
- `make_legacy_results.py` — parses the moved logs with a frozen copy of
  the historical parser and writes `legacy_results.h5`: the runs table
  (with machine/hardware attribution, in the historical file order) and
  a per-file SHA-256 source manifest. Beyond the historical columns, the
  table records the per-run metrics (the full and the initialisation
  runtimes, the number of simulation steps) and, from `log_meta.py`, the
  provenance of the log file each run came from (start datetime, commit,
  the PIConGPU / mallocMC versions, the GPU and its driver, the CUDA
  version, the CPU, the compiler, the host, the slurm job; the empty
  string where a log generation carries nothing).
- `log_meta.py` — the generation-aware provenance parser of the
  historical log layouts (the "Logging environment" dumps, the slurm
  build+run logs, the redesign-era header); used only by
  `make_legacy_results.py`.
- `make legacy-results` — builds `legacy_results.h5` from `logs/`.
- `make legacy-verify` — reparses `logs/` and checks the file against it.

Nothing in this directory that is data is committed: `logs/` and
`legacy_results.h5` are git-ignored and live on the machines that hold
the logs. On a fresh checkout, copy `legacy/logs/` from a machine that
already has the historical `output/` tree (or run the move there), then
`make legacy-results`.

The archived-but-excluded runs (`hal-sleeptimes-nanosleep`, an
experiment outside the benchmark matrix) are parsed into the file but
attributed to the empty hardware name; the main analysis drops exactly
those rows and records the exclusion in the results file's attributes.

The frozen parser must not change: `legacy_results.h5` is the
reproducible record of the numbers computed before the redesign, and
`make legacy-verify` will report a change to any input file. The
constraint applies to the eleven historical columns (their order and
values); the metrics and provenance columns appended after them are
recorded for completeness and never feed the fits. A machine that froze
its table before the metrics/provenance columns existed must re-run
`make legacy-results` after a pull to pick up the extended schema. When
the main analysis reads the frozen table, it defaults, at read time, the
log-derived columns the table does not record to the empty value and
marks the frozen rows `superseded = 0`.
