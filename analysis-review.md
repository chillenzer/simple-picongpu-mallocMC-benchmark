# The allocation model's headline number f is not a data quantity

**Subject:** `analysis/allocation_model.py` — the model
`T(m,f) = W + N_m·m + N_f·f + A_m·m0/(m+m0) + A_f·f0/(f+f0)` that produces the
runtime-budget split (W / A_malloc / A_free) and the fraction
`f = A/T0` shown in the sweep figures.
**Basis:** re-fit of all 12 sweeps in `output/results.h5`
(`python critique_fit.py`; read-only, reproduces the stored fits to ≤ 0.1 s).

## Verdict

The saturation terms `A·s0/(s+s0)` have no physical mechanism — they are, by
the project's own admission, a "stand in for the constant native allocation
cost A" (commit `c52b7c1`). The data do contain a real small-delay excess
runtime, but its shape violates the model's in 10 of 24 delay arms, and the
quantity the model exists to produce — `f`, the fraction of runtime spent in
allocations — is not determined by the data: the fit covariance is
near-singular (condition number 1e11–1e14, one infinite), and `f` moves by a
factor of 2–10 with the optimizer's seed, the s0 bound, and the solver's
version. The data instead support the call counts N, the zero-delay runtime
T0, and the existence of a small-delay excess of unknown mechanism.

## 1. The term has no mechanism; the README derives the line first

The delay is injected per call — "nanosecond delays imposed at run time on
every allocation and free request of the run" (README:11, 43–46: a busy-wait
at the top of `DeviceAllocator::malloc` / `::free`). Each of the N critical
path calls pays its native cost `c_a` plus the delay, so the runtime is
exactly

    T(s) = W + N·c_a + N·s = W + A + N·s,   A = N·c_a, a constant.

The README (README:53–63) derives this straight line, and only then says the
sweep is "fitted with the allocation model" — the hyperbolic variant — as a
fitting convenience. The same structure is in the module docstring
(`allocation_model.py:9–15`). Nothing in the experiment makes the native
per-call cost depend on s: a 10 s busy-wait does not make malloc 20 %
cheaper. `A·s0/(s+s0) → 0` as `s → ∞` encodes exactly that impossible
"fading" of the native cost; the only legitimate sense in which the native
cost is "negligible at large delays" is relative (A/N·s → 0), which the
linear model captures exactly. Commit `c52b7c1` states it plainly: the
saturating term "stands in for the constant native allocation cost A" and
bears "no relation beyond the shared hyperbolic shape" to anything physical.
So the model is a two-parameter (A, s0) wiggle shape with a free scale,
dressed as a decomposition.

## 2. What the data actually show: a real excess, but not this shape

Per delay arm (per-delay medians minus the line through the two largest
delays; `critique_fit.py` block 1):

- The excess is real and far above the per-run noise in 10 of 12 groups
  (per-run SSR improvement of 1.8–188 σ²; BIC prefers the 7-parameter model
  in 10 of 12, ΔBIC −16 … −300; classical F-test p < 1e-3 in 11 of 12).
- But the model requires a **positive, monotonically decaying** offset
  `A·s0/(s+s0)`. The malloc arms show a *hump* instead: e.g. hal KHI 128³
  FlatterScatter: +2.0 s at 100 ns, +9.3 s at 10–13 µs, −2.8 s by 3.2 ms —
  it rises and then crosses below the line. Ten of 24 arms go below the line
  at some delay (malloc arms: 3 of 12, by 2.8–13.9 s; free arms: 7 of 12,
  by 2.0–29.6 s). In the flagged group (rosi KHI 256×128×128) the
  unconstrained fit wants A_free < 0 — the code floors it at 0 (stored note
  in `results.h5`), which is how the reported f_free = 0 comes about.
- In 2 of 12 groups (rosi KHI 256×128×128, 256×128×256) the improvement over
  the line is at the noise level (0.26 and −0.39 σ²/run) — the fit there is
  absorbing nothing.

Whatever physical process produces the hump (a host/device serialization
effect, a batched-allocation effect, or a delay-dependent call pattern — the
per-call busy-wait story of README:43–46 does not produce it), it is not
"native allocation time fading over a scale s0", and a 4-parameter hyperbola
is not a model of it.

## 3. f is not identified: the data fix T0, not the split

- The fit's T0 = W + A_malloc + A_free matches the measured (0,0) baseline in
  all 12 groups within ~5 % (84.8 vs 86.6 s; 161 vs 154; 310 vs 300; 603 vs
  603; 88 vs 89.3; 114 vs 111; 208 vs 206; 400 vs 401). So the zero-delay
  runtime — the total bar length of the runtime-budget figures — is a data
  quantity. The W/A_malloc/A_free *split* of it is not.
- The stored parameter covariance has condition number 1e11–1e14 in 11 of 12
  groups and is infinite in the 12th; the module's own bootstrap code admits
  "the parameters are nearly degenerate" (`allocation_model.py:218`).
- Profile likelihood with A fixed: in rosi KHI 256×128×128, every
  A_malloc = 0 … 14 s costs ≤ 1σ² extra SSR, i.e. f_malloc = 0 … 6.5 % is
  equally good (stored: 4.75 %); A_free = 0 … 0.04 s → f_free ≈ 0 (stored:
  3.8e-09, floored). Similar flat directions in the other groups.
- Implied native cost per call A/N: 50–235 µs for malloc but 10⁻⁷–363 µs for
  free across the 12 groups — a 10-order spread for what should be a fixed
  property of mallocMC's free path.

The W/A split is a free direction of the design; f = A/T0 is the ratio along
it. The runtime-budget percentages (e.g. FoilLCT A30: 78.9/14.0/7.1; V100:
50.6/28.5/20.9) and the f annotations in the forest plot
(`figures/sweeps-shared-hal.pdf`: f_malloc 7.77–15.4 % individual vs
5.6–14.0 % shared) are not readings of the data.

## 4. f moves with the seed, the bound, and the solver

- Documented drift, same KHI 128³ data: f_free = 21.7 ± 30.6 % with the grid
  seed vs 2.0 ± 0.6 % seeding f0 at the window bottom (commit `a496cda`,
  Aug 28), then 5.8 ± 1.8 % after two new free-arm points (`1d8b306`).
- Reproduced on the current data (block 3): hal KHI 128³ FlatterScatter,
  f_malloc 9.72 % (grid/mid seed) → 0.93 % (window-bottom seed), f_free
  5.50 % → 2.47 %; ScatterAlloc 7.77 % → 1.80 %. rosi KHI 256×128×128:
  three seeds give 6.52 % / 2.39 % / 4.24 %.
- Local minima: the stored rosi KHI 256×128×128 fit claims the saturation
  terms beat the 3-parameter line with F = 18.0 (p = 2e-12). Re-running the
  *same documented procedure* (same grid initial guess, same bounds) now
  lands in a different minimum that does not beat the line at all
  (F = −6.9, p = 1.0; block 2). The "significance" of the stored fit is
  which minimum the solver fell into.
- Bounds: s0 ∈ [0.05·s_min, 0.5·s_range] "keep the Amdahl fraction f in
  [0,1), land s0 at the unconstrained optimum" (commit `da3063f`;
  `allocation_model.py:24–28`: the cap exists "to keep it (and A)
  identifiable"). When the data want A < 0, the code silently floors f at 0.

## 5. The validation was circular

"Verified on synthetic sweeps (recovers W, N, A, s0 within ~2%)" (commit
`c94766d`); "a 52-point synthetic combination sweep recovers all seven model
parameters" (`c4eea5a`). Fitting the model to data generated by the same
model tests the fitter, not the model. No independent check — a directly
measured per-call cost, or a sweep dense enough in the bend to resolve s0 —
exists in the repository.

## What the data do support, and what a defensible version needs

- **N (calls per run)** from the large-delay slope: well determined and
  consistent across algorithms (e.g. N_malloc = 2.34e5 FoilLCT A30,
  1.33e5 KHI 128³ A30; individual vs shared fits agree in the forest plot).
- **T0** from the (0,0) baseline, matched by the fits in all 12 groups.
- **A real small-delay excess**: at the smallest delays, each arm's runtime
  sits 5–72 s above the large-delay line extrapolated to zero delay (whose
  intercept lands within ~5 % of the measured (0,0) baseline) — up to ~33 %
  of T0 (rosi KHI 128³: 37 s on 114 s). The per-call busy-wait story of
  README:43–46 does not produce such an excess, and it is currently
  unexplained. That is the open question the data actually pose.

To report a runtime budget split, the native per-call cost c_a must be
measured independently (a zero-delay microbenchmark of
`DeviceAllocator::malloc`/`free`, or mallocMC's own counters) and imposed as
A = N·c_a, leaving only (W, N) to fit; the s0 degree of freedom and the
bounds that exist to make it identifiable should go. Until then, f is a
fitting artifact and the runtime-budget figures should not be read as
decompositions of the measured runtime.

## References and reproduction

- `analysis/allocation_model.py:1–31` (model, bounds), `:218` (degeneracy),
  `:394–421` (A<0 / s0-at-cap diagnostics)
- README:11–12, 37–46 (per-call delay injection), 53–76 (line derived first)
- Commits: `c94766d` (synthetic validation), `da3063f` (bounds, f-floor),
  `a496cda` + `1d8b306` (documented f drift), `c52b7c1` ("stands in for"),
  `4c43718` (f annotations moved into the forest plot)
- Data: `output/results.h5` (`runs`, `fits`, `shared_fits`, `baselines`,
  `fits/cov`); reproduction: `python critique_fit.py` (blocks: 1 nested-model
  tests, 2 multi-modality, 3 seed sensitivity)
