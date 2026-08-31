<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Plan: supplement the performance models with the microbenchmark data

**Status:** implemented (Phases 1–5). Supersedes nothing; it is the follow-up
to `analysis-review.md` §6–§7 and the microbenchmark integration
(`analysis/make_microbench.py`, commit "Integrate the microbenchmark allocation
costs into the analysis pipeline"). The open questions below are resolved;
each records the decision that was made.

## 1. What the microbenchmark gives us

The `microbenchmarks/memmansurvey` suite measures, with no imposed delay, the
**native per-call cost** of each GPU device allocator: `alloc_cost` has one row
per (run, hardware, allocator, operation, size) with `mean_ms` (the host-side
per-operation time, kernel launches included). Frozen once by
`analysis/make_microbench.py` into `microbenchmarks/microbench_results.h5`
(`make microbench-results`, read by `compute_results.py` as the optional
`alloc_cost` table).

So the microbenchmark is an **independent measurement of the native per-call
cost `c_a`** (in ns, = `mean_ms × 1e3`) for each (hardware, allocator,
operation). This is exactly the quantity the review says must be measured
independently to give `f` a genuine allocation-cost meaning.

## 2. The physics (why this matters)

Per `analysis-review.md`: the performance model's fitted `A` (the saturating
term `A·g(s/s0)`) measures **delay hidden by parallel slack**, not the native
allocation cost. The native cost `c_a` is *constant* (a 10 s busy-wait does not
make malloc cheaper); the review's true straight line is
`T(s) = W + N·c_a + N·s` with `A = N·c_a` a constant. The model's *fading* `A`
is a fitting stand-in that conflates (i) the constant native cost and (ii) the
absorbed (hidden) delay. Two distinct, useful quantities fall out:

- **Native-cost budget:** `A_alloc = N·c_a` (constant) → `f_native = A_alloc / T0`.
  Using the measured `c_a` (not the fitted `A`) is the only non-circular route
  to a runtime budget that books the allocation cost against the allocator.
- **Absorbed slack vs native cost:** the fitted `A/N` (per-call absorbed slack,
  3–291 µs) compared against the measured `c_a` (per-call native cost) gives a
  new, meaningful ratio — *how much of the native per-call cost the pipeline
  hides*.

## 3. What is wired now

| piece | state |
|---|---|
| microbench suite + freeze pipeline | present (`make_microbench.py`, `make microbench-results`); the frozen table is optional (empty when absent) and **not built in this env** (data lives on the benchmark machine, git-ignored) |
| 1-D and 2-D `c_a` constraint | present (`model_c_a` / `model_2d_c_a`; `fit_1d(c_a=...)` and `fit_2d(c_malloc=..., c_free=...)`) |
| `c_a` supplied by the pipeline | present — `fit_sweep(runs, alloc_cost)` resolves `c_malloc` / `c_free` per group via `_resolve_c_a` |
| reporting of native cost / absorbed-vs-native | present — primary `fits` row carries `c_a_{malloc,free}_ns`, `slack_over_native_{malloc,free}`, `native_note`; the secondary `A = N·c_a` fit is the `fits_ca` table; the comparison figure is `figures/native-cost.pdf` (`analysis/plot_native_cost.py`) |

## 4. The implementation

### Phase 1 — `c_a` lookup (`compute_results.py::_resolve_c_a`)
Resolves the native per-call cost for one (hardware, allocator, operation):
`_resolve_c_a(alloc_cost, hardware, algorithm, operation, setup) -> (ns|None, note)`.
It matches
- **hardware**: the group's short hardware name, normalized by `_gpu_model`
  (vendor stripped, so `NVIDIA A30` == `A30`), to the microbench run's hardware;
- **allocator → algorithm**: the microbench `allocator` to the benchmark
  `algorithm`, via `ALLOCATOR_ALIASES` (e.g. `ScatterAlloc` → `Scatter`,
  `Gallatin` → `GallatinCuda`) plus the exact name;
- **operation**: `MICROBENCH_OPERATION` maps the model arm to the microbench
  operation (`malloc` → `alloc`, `free` → `free`);
- **size**: the nearest measured `size_bytes` to the representative allocation
  size `KHI_FRAME_BYTES = 7696` B (see the Size decision below).
It returns `mean_ms × 1e3` ns, or `(None, note)` with an explanatory note when
there is no microbenchmark data, no matching hardware, or no matching allocator —
so the fit simply stays unconstrained.

### Phase 2 — 1-D and 2-D `c_a` constraint (`performance_model.py`)
- `model_2d_c_a(m, f, p, c_malloc, c_free, fade)`:
  `T = W + N_m·m + N_f·f + (N_m·c_malloc)·g(m/m0) + (N_f·c_free)·g(f/f0)`.
- `fit_2d(..., c_malloc=None, c_free=None)`: with **both** set, fit
  `(W, N_m, N_f, m0, f0)` (5 free params) with `A_m = N_m·c_malloc`,
  `A_f = N_f·c_free`; otherwise the unconstrained 7-parameter model. Standard
  errors are propagated through the constraint (`_fit_2d_errors`,
  `_f_of_{m,f}_ca`). `fit_1d(c_a=...)` was already present.
- The combined (shared-parameter) fit is intentionally left unconstrained (see
  the Combined-fit decision).

### Phase 3 — pipeline wiring (`compute_results.py::fit_sweep`)
`fit_sweep(runs, alloc_cost)` resolves `c_malloc` / `c_free` per group and:
- keeps the **unconstrained fit as the primary** row in `fits` (it measures
  absorbed slack); that row also reports `c_a_{malloc,free}_ns`,
  `slack_over_native_{malloc,free}` (`(A/N) / c_a`), and `native_note`;
- where the microbenchmark supplies **both** costs for the group's hardware,
  records a secondary `A = N·c_a` constrained fit in the `fits_ca` table
  (`model = 2d-ca` / `1d-*-ca`).
Without matching microbenchmark data a group has `NaN` native columns and no
`fits_ca` row — the graceful fallback that makes the analysis general across
hardware.

### Phase 4 — reporting & figure
- `analysis/plot_native_cost.py` → `figures/native-cost.pdf` (Makefile target
  `figures/native-cost.pdf`): one panel per operation, each fitted group plotted
  as (native cost `c_a`, absorbed slack `A/N`) in µs per call, coloured by
  machine, with the dotted `A/N = c_a` hypothesis line (the review's rejected
  reading). Data-optional: with no matching microbenchmark cost it prints a note
  and writes no file.
- The native-cost columns are carried through `output/results.h5` (`fits`,
  `fits_ca`).

### Phase 5 — prose
- README "The method": documents the microbench supplement — `c_a` measured
  independently, the unconstrained fit primary, the `A = N·c_a` constrained fit
  (`fits_ca`) as the native-cost budget.
- `analysis-review.md` §6/§7: updated to record that `c_a` is now measured and
  wired (1-D and 2-D).

## 5. Decisions (the former open questions)

1. **Hardware scope — resolved.** Apply `c_a` **only on matching hardware**
   (normalized by `_gpu_model`). The code is written to be **general**: it fails
   gracefully to the unconstrained fit wherever there is no microbenchmark data,
   and shows the full constrained + comparison picture wherever there is. This
   keeps it correct as more runs land on various systems (A100 today; A30/V100
   and others later). No cross-hardware proxy is implied.
2. **Size matching — resolved.** The representative size is the **dominant
   KHI frame, 7696 B** (the paper's number; consistent with the parameter-derived
   ~8 KiB: 2 MiB page ÷ 256 frames). The microbench cost is read at the
   **nearest measured size** to 7696 B. For the **non-KHI setups** (FoilLCT) the
   same 7696 B is used as an **order-of-magnitude estimate only** — their
   allocation patterns differ (ScatterAlloc 2 MiB pages vs FlatterScatter
   128 KiB pages) — and the `native_note` is tagged
   "(order-of-magnitude estimate)".
3. **Hard vs soft — resolved.** The **soft comparison is primary** (the
   unconstrained fit + `slack_over_native` ratio), and the **hard constraint**
   (`A = N·c_a`) is a **separately labelled secondary fit** (`fits_ca`).
4. **Combined fit — resolved.** The shared-parameter combined fit is **not**
   constrained; only the per-group fits carry the native-cost reading (simpler,
   and the combined fit already pools `W`/`N`, so per-algorithm `c` constraints
   are not its purpose).
5. **Data prerequisite — resolved (graceful).** The frozen table is not built in
   this env, so the pipeline and the figure run end-to-end on the empty
   `alloc_cost` table (verified): all groups stay unconstrained, the native
   columns are `NaN`, `fits_ca` is empty, and the figure prints a note. On the
   benchmark machine, `make microbench-results` freezes the A100 runs
   (jobids 9032687, 9057444) and the constrained path lights up for the
   matching hardware/allocator.

## 6. Risks / caveats

- The microbench is a bare allocation loop on A100 with the host-side timer;
  the benchmark's per-call native cost is in a full simulation on possibly
  different hardware. `c_a` is an independent but not identical measurement —
  report it as such, not as a ground-truth equal to the fitted `A/N`.
- The degenerate group (rosi KHI 256×128×128, `A_f → 0` boundary) and the
  `hal` KHI humps remain unexplained; the `c_a` constraint does not address
  them.
- For the non-KHI setups the native-cost figure is order-of-magnitude (see the
  Size decision); the KHI groups are the precise reading.
