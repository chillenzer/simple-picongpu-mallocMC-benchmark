<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Plan: supplement the performance models with the microbenchmark data

**Status:** proposal (not implemented). Supersedes nothing; it is the follow-up
to `analysis-review.md` §6–§7 and the microbenchmark integration
(`analysis/make_microbench.py`, commit "Integrate the microbenchmark allocation
costs into the analysis pipeline").

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

## 3. Current state (what is wired, what is missing)

| piece | state |
|---|---|
| microbench suite + freeze pipeline | present (`make_microbench.py`, `make microbench-results`) |
| frozen `alloc_cost` table read by the analysis | present, but **optional** (empty when the table is absent); **not built in this env** (data lives on the benchmark machine, git-ignored) |
| `c_a` constraint in the model | present for **1-D only** (`model_c_a`, `fit_1d(c_a=...)`) |
| `c_a` constraint in the 2-D model | **missing** (`fit_2d` has no `c_a`) — but all 12 fit groups are 2-D |
| `c_a` supplied by the pipeline | **missing** — `fit_sweep(analyzed)` is called with `c_a=None`; the 2-D branch ignores `c_a` even if given |
| reporting of native cost / absorbed-vs-native ratio | **missing** |

## 4. The plan

### Phase 1 — `c_a` lookup (compute_results.py)
Add a helper that resolves the native per-call cost for a fit group:
`c_a(machine, algorithm, operation, alloc_cost, machines_cfg) -> float | None` (ns).
It matches
- **hardware**: the machine's `hardware` (config `machines.<m>.hardware`) to the
  microbench run's hardware (A100 ↔ `rosi-a100`);
- **allocator → algorithm**: the microbench `allocator` to the benchmark
  `algorithm` (FlatterScatter / ScatterAlloc / Gallatin — same names);
- **operation**: `alloc` → malloc arm, `free` → free arm;
- **size**: the microbench `size_bytes` to the benchmark's allocation size
  (see Open question 2 — this is the non-trivial part).
Returns `mean_ms × 1e3` ns, or `None` when no match (hardware or allocator).

### Phase 2 — 2-D `c_a` constraint (performance_model.py)
- Add `model_2d_c_a(m, f, W, N_m, N_f, c_malloc, c_free, m0, f0, fade)`:
  `T = W + N_m·m + N_f·f + (N_m·c_malloc)·g(m/m0) + (N_f·c_free)·g(f/f0)`.
- Extend `fit_2d(..., c_malloc=None, c_free=None)`:
  - both given → fit `(W, N_m, N_f, m0, f0)` with `A_m = N_m·c_malloc`,
    `A_f = N_f·c_free` (5 free params instead of 7);
  - one given → constrain that arm only;
  - none → current 7-parameter behavior (unchanged default).
- Propagate standard errors through the constraint (mirror the existing
  `_fit_constrained_1d` / `_f_of_p_ca` pattern).
- (Later, optional) extend `fit_combined` for the shared `W`/`N` case with
  per-algorithm `c` constraints.

### Phase 3 — pipeline wiring (compute_results.py)
- In `fit_sweep` / `fit_one_group`, resolve `c_malloc`, `c_free` per group via
  Phase 1 and pass them to `fit_2d` (and `fit_1d` for 1-D fallback groups).
- **Keep the unconstrained fit as the primary** (it measures absorbed slack);
  add columns for the native-cost reading: `A_malloc_native = N_malloc·c_malloc`,
  `A_free_native`, `f_malloc_native`, `f_free_native`, plus the raw
  `c_malloc_ns`, `c_free_ns` used, and a `note` recording the hardware/size the
  `c_a` came from. (Alternatively, run the constrained fit as a separate,
  clearly-labelled table — see Open question 3.)

### Phase 4 — reporting & figures
- `summarize_results.py`: per arm, report the fitted absorbed slack `A/N`, the
  microbench native cost `c_a`, and their ratio (absorbed / native). Report the
  native-cost budget `A = N·c_a` and `f_native`.
- Figures: add a panel (to `sweeps-<machine>.pdf` or a new figure) plotting,
  per arm, absorbed slack vs native cost, and/or the native-cost budget split.
  The headline new quantity: the fraction of the native per-call cost that the
  pipeline hides.

### Phase 5 — prose
- README "The method": document the microbench supplement — `c_a` measured
  independently, `A = N·c_a` gives the native-cost budget, and the fitted `A`
  remains the absorbed-s hiding model.
- `analysis-review.md`: update the §6/§7 status — `c_a` is now measured; the
  native-cost reading is available (with the hardware/size caveats).

## 5. Open questions (decisions needed)

1. **Hardware scope.** The microbench ran only on **A100**, matching
   `rosi-a100`. `hal` (A30) and `rosi` (V100) have no matching `c_a`. Apply the
   constraint only where hardware matches (rosi-a100), and for A30/V100 either
   (a) leave them unconstrained and report the A100 `c_a` as a cross-hardware
   reference, or (b) treat A100 `c_a` as a proxy with an explicit caveat?
2. **Size matching.** The microbench `c_a` is per allocation **size**, but the
   benchmark's actual allocation sizes are implicit (set by the simulation's
   memory layout) and not directly reported. How to pick the size: a
   representative/median size, an estimate from the simulation, mallocMC's own
   counters (size-weighted `c_a`), or report `c_a` over a size band? This is
   the least-well-defined part.
3. **Hard constraint vs soft comparison.** (a) Hard-constrain the fit
   (`A = N·c_a`, fewer free params) — the review's route to a genuine budget,
   but it removes the data's own `A`; or (b) keep the fit free and only report
   the comparison (absorbed slack vs native cost)? Proposal: do (b) for the
   primary fit and (a) as a labelled secondary reading.
4. **Combined fit.** Constrain the shared-parameter combined fit too, or only
   the per-group fits (simpler; the combined fit already pools `W`/`N`)?
5. **Data prerequisite.** The frozen table is not built here. Confirm the
   microbench runs (jobids 9032687, 9057444, A100) are frozen on the benchmark
   machine before Phases 3–4 can be exercised end-to-end.

## 6. Risks / caveats

- The microbench is a bare allocation loop on A100 with the host-side timer;
  the benchmark's per-call native cost is in a full simulation on possibly
  different hardware. `c_a` is an independent but not identical measurement —
  report it as such, not as a ground-truth equal to the fitted `A/N`.
- The degenerate group (rosi KHI 256×128×128, `A_f → 0` boundary) and the
  `hal` KHI humps remain unexplained; the `c_a` constraint does not address
  them.
- Until the size question (Open 2) is settled, the native-cost budget is
  approximate; label it clearly rather than presenting it as exact.
