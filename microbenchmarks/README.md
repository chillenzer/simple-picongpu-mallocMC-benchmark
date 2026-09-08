<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Microbenchmark of the device allocators (memmansurvey)

This directory integrates the `memmansurvey` microbenchmark (pinned as the
`microbenchmarks/memmansurvey` submodule) as an independent measurement of the
native per-call allocation cost `c_a` of each device allocator. The main
analysis uses that cost as the native-cost reading of the performance model
(*The method* in the README; `analysis/performance_model.py`): where the frozen
table supplies a matching cost for a group's hardware, allocator and
operation, the reported `fits` row carries `c_a_{malloc,free}_ns` and
`slack_over_native_*`, and a separate `A = N*c_a` constrained fit (the
`fits_ca` table) gives the native-cost budget.

- `memmansurvey/` — the pinned microbenchmark (a git submodule; initialize
  with `git submodule update --init microbenchmarks/memmansurvey`).
  The pin is kept in sync by the `make microbench-src` target and the
  `microbench` section of `config.json`.
- `make microbench-results` — freezes the raw allocation-test CSVs (the data
  directory below) once into `microbench_results.h5` (git-ignored, not
  committed), via `analysis/make_microbench.py`; the runs are declared in the
  `microbench.runs` table of `config.json` (jobid, hardware).
- `make microbench-verify` — reparses the raw CSVs and checks the frozen
  table against them.
- `make microbench-audit` — audits the raw CSVs for missing x-values and
  their causes.

The raw CSVs (`microbenchmarks/data/`) and the frozen table
(`microbenchmarks/microbench_results.h5`) are machine-generated benchmark
data and are not committed; they live on the machine that ran the
microbenchmark and travel with the released data archive (*Where the data
lives* in the README). On a fresh checkout without them the analysis runs
gracefully without the native-cost reading (the native columns stay empty
and the `fits_ca` table is empty).
