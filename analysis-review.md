# f is not an allocation fraction: the saturation terms measure hidden delay

**Subject:** `analysis/allocation_model.py` — the model
`T(m,f) = W + N_m·m + N_f·f + A_m·m0/(m+m0) + A_f·f0/(f+f0)` that produces the
runtime-budget split (W / A_malloc / A_free) and the fraction
`f = A/T0` shown in the sweep figures.
**Basis:** re-analysis of all 12 sweeps in `output/results.h5`
(`python critique_fit.py`; read-only, reproduces the stored fits to ≤ 0.1 s,
checks the stored model against the raw data, and verifies the model's gauge
symmetry).

## Verdict

The small-delay structure in the sweeps is real, and its mechanism is the
obvious one the project never named: PIConGPU is a pipelined CPU+GPU code, and
the busy-wait injected at the top of `DeviceAllocator::malloc`/`free` is not
on the bottleneck at small-to-moderate delays — parallel CPU/GPU work in
flight absorbs part of it. Anchored at the measured zero-delay runtime, 20 of 24 arms rise by *less*
than N·s, and the absorbed part saturates: the deficit plateaus at d = 1–70
s (an absorbable slack of d/N = 3–291 µs per call); the twelve hal KHI arms
instead start above the nominal line (below). The
project's hyperbola, measured against its own zero-delay value, is exactly
this saturating-absorption shape, so the functional form is defensible — what
is not defensible is the label. A is not "the constant native allocation
cost" (which cannot fade with the imposed delay); it is the total delay
absorbed by parallel slack, and f = A/T0 is "absorbed delay per run /
zero-delay runtime" — a measure of the pipeline's slack, not of the
allocator: a run with more in-flight work has a bigger f even if malloc/free
cost nothing. Section 6 adds the algebraic point: the hyperbola carries a
delay-origin gauge under which f moves freely, so f is not even an invariant
of the model form. On top of that, the *stored* f is not well determined: the data
pin the model-consistent A's directly (the plateaus d_m, d_f, 2–11σ), but the
stored fit drifts 16–64 s in W (and up to 44 s in the A's) from them along
the flat W/N/A direction
(covariance condition number 1e11–1e14; f_malloc 9.72 % → 0.93 % under the
fade-scale seed), and the only recorded validation is circular. What the
model does not capture: all twelve hal KHI arms, where the small delay starts
*above* the nominal line (+0.4 to +12.3 s at 100 ns; eight of them amplify
to peaks of +8.9 to +12.3 s) — which neither hiding nor the model (E ≤ 0 by
construction) can produce — and three free arms that are over-exposed at the
largest delay.

## 1. The stated mechanism is impossible; the shape has a different one

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
cheaper. Commit `c52b7c1` concedes it: the saturating term "stands in for the
constant native allocation cost A" and bears "no relation beyond the shared
hyperbolic shape" to anything physical.

But the same mathematical shape has a second, plausible reading. Measured
against the zero-delay value, the model says

    T(s) - T(0) = N·s - A·s/(s+s0),

i.e. the imposed delay is fully exposed minus a saturating absorbed part
`A·s/(s+s0)` — exactly the signature of a host-side delay being hidden by
parallel work that is in flight, and exposed in full only once the delay
exceeds the available slack. Section 2 shows the data do show this.

## 2. What the data show (anchored at the measured zero-delay runtime)

Per arm (per-delay medians), define E(s) = T(s) − T0 − N·s, where T0 is the
measured (0,0) median and N the slope of the two largest delay points
(`critique_fit.py` block 4):

- E(s) ≈ 0 at the smallest delay (100 ns) for all twelve hal FoilLCT and
  rosi arms (|E| ≤ 4 s, within the run-to-run spread of 0.3–9 s). All twelve
  hal KHI arms instead start positive (E(100 ns) = +0.4 to +12.3 s); see the
  hump item below.
- E(s) then turns negative — the imposed delay adds *less* than N·s — and the
  deficit grows with s, saturating by ~1–10 ms at the plateau d = T0 −
  (large-delay line intercept) = 1–70 s in 20 of 24 arms (positive in 11 of
  12 malloc arms, 9 of 12 free arms). The implied absorbable slack per call
  d/N is 3–291 µs for the malloc arms (−31 to +291 µs including one negative
   outlier) and −105 to +226 µs for the free arms. It grows with problem size
   (hal KHI malloc: 3 µs at 128³, 58–104 µs at 256×128×128, 165–241 µs at
   256×128×256) and depends on the machine's CPU/GPU balance: at 128³ rosi
   absorbs 291 µs per call where hal absorbs 3 µs; at the largest grid both
   are 160–240 µs.
- This is the "allocation is not the bottleneck" picture: while the host
  spins in the injected busy-wait, the GPU (and the other CPU threads) keep
  working, and only the unabsorbed part of the delay reaches the runtime.

The model is a good realization of that picture: it fits the 1190 runs of the
12 fit groups to within 1–61 s (typically < 15 s; block 5), and the
saturation terms are strongly supported over the bare line (per-run SSR
improvement 1.8–188 σ² in 10 of 12 groups; BIC prefers the full model in 11
of 12, ΔBIC −16 … −300; F-test p < 1e-3 in 11 of 12). The model
also pins its own parameters to the data: on the malloc arm
T(m,0) → W + A_free + N_m·m as m → ∞, so the measured plateau is
A_malloc = d_m (likewise A_free = d_f, and W = T0 − d_m − d_f) — three
quantities the data fix without any fit.

What the stored fit gets wrong (block 5, stored minus data-pinned):

- In 4 of 12 groups the stored split is within ~11 s of the pinned values
  (hal FoilLCT Flat/Scat: ΔW −3.4/+0.0 s; hal K256×256 Scat: −10.9 s; rosi
  KHI 128³ Flat: +10.1 s); in the other 8 it drifts 16–64 s in W and up to
  44 s in the A's. rosi KHI
  256×128×128: d_m = 36.5 s but stored A_malloc = 9.9 s — W sits 35.7 s high
  and N_malloc 0.2 % low over the last two decades, which costs only ~8 s at
  the anchor points; and its free arm measures d_f = +7.5 s while the fit
  sits on the A_free ≥ 0 boundary at A_free = 0 (f_free = 0 %), silently
  discarding the measured absorption.
- The four hal KHI hump groups (128³ and 256×128×128, both algorithms) go
  the other way: stored A_m/A_f 12–41 s against measured plateaus of 0.4–19
  s (inflated 1.8–36×, sign-flipped in the 128³ ScatterAlloc group), W
  17–33 s low — the fit distorts the parameters to imitate a shape the model
  cannot have (below).
- Hence the reported f values deviate from the data-pinned f = d/T0 by
  factors of 0.27–35 (rosi KHI 256×128×128 malloc: stored 4.75 % vs pinned
  17.7 %; hal KHI 128³ Flat: 9.72 % vs 0.28 %); f_free by factors of 0.5–19,
  with the sign flipped in three groups and pinned at the zero bound in a
  fourth (where the measured plateau is +7.5 s).

What the model does not capture, anchored at the baseline:

- All twelve hal KHI arms start above the nominal line (E(100 ns) = +0.4 to
  +12.3 s), while all twelve other arms start at ≤ +0.3 s. Eight of the
  twelve (the 128³ and 256×128×128 grids) peak at +8.9 to +12.3 s within the
  first few points (100 ns – 20 µs) before crossing to the large-delay
  deficit — the small delay is *amplified*; the four 256×128×256 arms show
  only a mild positive start (+0.4 to +2.4 s). Where the run-to-run spread
  is measurable the peaks are ≥ 3σ, and in the single-run ScatterAlloc
  groups they stand as a consistent kink against the smooth large-delay
  structure. Neither hiding (E ≤ 0) nor the model (E ≤ 0 by construction)
  can produce it.
- Three free arms are over-exposed at the largest free delay: d_f = −8.3 s
  (hal KHI 128³ Scat), −8.7 s (rosi FoilLCT Flat), −4.0 s (rosi KHI 128³
  Flat); E = +4 to +8.7 s at 56–100 ms.
- hal FoilLCT Flat's free arm is non-monotonic (E drops to −14 s by 562 µs,
  then recovers to −1 s at 32–56 ms).

## 3. f is not identified: the data fix T0 and the plateaus, not the split

- The fit's T0 = W + A_m + A_f matches the measured (0,0) baseline in all 12
  groups within ~5 %, and the plateaus d_m, d_f — hence the model-consistent
  (W, A_m, A_f) — are fixed by the data to 2–11σ (block 5). The stored
  *split* is not: it sits up to 64 s from the pinned values, and the
  parameter covariance has condition number 1e11–1e14 in 11 of 12 groups
  (infinite in the 12th); the module's own bootstrap admits "the parameters
  are nearly degenerate" (`allocation_model.py:218`).
- Profile likelihood: in rosi KHI 256×128×128 every A_malloc = 0 … 14 s costs
  ≤ 1σ² extra SSR (f_malloc = 0 … 6.5 % equally good; stored 4.75 %, pinned
  17.7 %).
- The implied per-call quantity A/N = d/N: 3–291 µs for the malloc arms is
  now physically interpretable (the parallel slack, which really does scale
  with the problem), but the free arms span −105 to +226 µs with sign flips
  between adjacent groups — not a property of a fixed free path.

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

## 6. The model has a gauge symmetry, and f is not an invariant of it

The hyperbolic term is not uniquely parameterised. The model is exactly
invariant under a per-arm shift of the delay origin — for the malloc arm, and
identically for the free arm,

    W  -> W + N·s0·d      A  -> A/(1+d)      s0 -> s0·(1+d)      s -> s - s0·d,

with N unchanged. The product A·s0 and the combination W - N·s0 are invariant,
so a one-arm model has exactly three invariants — N, A·s0, W - N·s0 — and the
two-operation model has five: N_m, N_f, A_m·m0, A_f·f0, and
W - N_m·m0 - N_f·f0 (verified to ≤ 4e-12 s on all 12 stored fits,
`critique_fit.py` block 6). This is a horizontal translation of the fitted
curve: the same runtime versus the *physical* delay, described from a different
origin. The origin is fixed by the experiment (s = 0 is the measured
no-injection baseline), so it is a gauge of the model's internal origin, not a
redundancy of the fit — with s fixed, all four parameters stay independent, and
the gauge direction is not the fit's flat direction (block 6: it carries 0.00
of the two flattest covariance directions in all 12 groups; the flat direction
is the fade scale s0, orthogonal to it).

The reported fraction is not one of the invariants. With T0 = W + A,

    f = A/T0   ->   A / [(W + N·s0·d)(1+d) + A],

equal to f only for d = 0. So f (and T0) moves along the gauge orbit, and the
model form alone fixes nothing about its value: over each fit's own s0 search
window, f_malloc and f_free span roughly [0, 100 %] — a range 26–189× the
stored 1-σ error (block 6). The data pin the gauge, which is why the reported
f is nonetheless well determined; the model can simply not be the reason f has
the value it does.

This is the algebraic half of section 3. Section 3 showed the *data* do not
identify the W/A split (flat W/N/A direction, covariance condition number
1e11–1e14); this shows the *model form* does not make f an invariant either. f
is a convention-dependent, gauge-dependent ratio that is well determined only
because the data happen to pin the gauge — not a quantity defined by the model
or the measurement on their own.

## What the data do support, and what a defensible version needs

- **N (calls per run)** from the large-delay slope: well determined and
  consistent across algorithms (N_malloc = 2.34e5 FoilLCT A30, 1.33e5 KHI
  128³ A30; individual vs shared fits agree in the forest plot).
- **T0** from the (0,0) baseline, matched by the fits in all 12 groups.
- **Saturating delay absorption**: the imposed delay is hidden by parallel
  work up to a per-call slack d/N = 3–291 µs (malloc arms), growing with
  problem size; equivalently, the large-delay line's intercept sits
  d = 1–70 s below the baseline in 20 of 24 arms. This is a quantitative
  fingerprint of the pipeline's host/device overlap and is independently
  testable (it should track in-flight work, not allocator speed).
- **Unexplained**: the hal KHI amplification humps (+0.4 to +12.3 s at the
  smallest delay, in all twelve hal KHI arms) and the three over-exposed free
  arms. These are the open questions the data actually pose.

To report a runtime budget split, the native per-call cost c_a must be
measured independently (a zero-delay microbenchmark of
`DeviceAllocator::malloc`/`free`, or mallocMC's own counters) and imposed as
A_alloc = N·c_a — the fitted A measures absorbed delay, not allocation cost,
and using it for the budget would book the pipeline's slack against the
allocator. The (A, s0) structure can stay as a hiding model, but labeled as
such, with A_m, A_f taken from the measured plateaus (which also removes the
flat W/N/A direction) and s0 resolved by a sweep dense enough in the bend.
Until then, f is a slack fraction with a fitting artifact mixed in, and the
runtime-budget figures should not be read as decompositions of the measured
runtime.

## 7. Follow-up 1 — the flat direction: can a re-parameterization with one fewer parameter remove it?

Full analysis and per-fit tables: `qa-flat-direction.md`
(reproduce: `python qa_flat_direction.py`).

**No.** At the fixed experimental delays the model map is genuinely
4-dimensional (1-op) / 7-dimensional (2-op): the parameter Jacobian has full
rank (the one exception, rosi KHI 256×128×128, sits on the `A_f→0` boundary
and is rank 6). The per-arm delay-origin gauge of §6 has invariants
`N`, `A·s0`, `W−N·s0`, but those span only a 3-D (5-D) subfamily — the fade
scale `s0` is an *additional, independent* degree of freedom of the function
(holding the invariants fixed and moving `s0` by 0.1×–10× still moves the
curve by up to 140 s). A one-fewer-parameter family is a proper subset of the
function space and cannot reproduce the same curves. And the flat direction is
a **data near-degeneracy** (weak information), not a parameter redundancy: the
information spectrum is invariant under any reparameterization, so no change of
coordinates removes it.

Two "degeneracies" must not be conflated. The **gauge** (a redundancy of the
model *form*, §6) is pinned by the fixed delay origin and is *not* the fit's
flat direction (its share of the two flattest covariance directions is
0.008–0.111). The **near-degeneracy** (cond(pcov) 1e11–1e14) is a data
limitation: *locally* the flattest directions are the **N slope-pivots** and
the **(W, A_m, A_f) split**, and the fade scales are locally the *stiffest*;
*globally*, the fade scale is the weakly-identified quantity — seed-dependent
and multi-modal, at the 50 ms search cap in the slow-fade rosi KHI arms, and
exactly degenerate in the `(A_f, f0)` product at the `A_f→0` corner. (The
"0.00 gauge share" quoted in §6 was measured on `eigh`'s first two
eigenvectors, which are the two *stiffest* — numpy sorts ascending — not the
flattest; on the actual flattest it is 0.008–0.111. The practical conclusion
is unchanged.)

What actually helps — options that add information or change what is reported,
not the coordinate system:

- **Fix the fade scales externally** (a sweep dense in the bend, or a prior) →
  the fit becomes an exact linear 5-parameter problem (condition number
  1e11–1e14 → 1e4–1e6); `f` then depends on the chosen `s0`, so report the
  ×0.1/×10 sensitivity.
- **Impose `A = N·c_a`** with `c_a` from an *independent* zero-delay
  microbenchmark of `DeviceAllocator::malloc`/`free` (the model already
  supports `model_c_a`) → the only route that gives `f` a measured
  allocation-cost meaning (using the fitted `A` is circular, §5).
- **Report the identifiable, gauge-invariant, data-pinned combinations** —
  `N`, `T0`, the plateau deficits `d_m`,`d_f` (and per-call `c = d/N`) —
  instead of the `(W, A, s0)` split; give `f` a profile-likelihood CI or a
  multi-start seed range, not the Gaussian 1σ (an order of magnitude too small
  along the flat direction).
- A combined fit across algorithms does **not** remove the per-algorithm flat
  direction; the reparameterization `(T0, N_m, N_f, A_m, K_m=A_m·m0, A_f,
  K_f=A_f·f0)` drops the condition number 2–3 orders and helps local-minimum
  risk but does not remove the data flat direction.

**Bottom line:** no reparameterization eliminates the flat direction. Report
the data-pinned gauge-invariant quantities, fix or bound `s0` with external
information, and for a genuine allocation-cost budget measure `c_a`
independently and impose `A = N·c_a`.

## 8. Follow-up 2 — can the plateau measure the (de-)allocation cost, and can we compare algorithms?

Full analysis, per-arm tables, and shape fits: `qa-allocation-cost.md`
(reproduce: `python qa2_analysis.py`, `python qa2_table.py`).

**The hiding claim is verified, with a refinement.** 20 of 24 arms show the
saturating absorption (plateau deficit `d = T(0) − b0 > 0`, 0.4–70 s per run
= 3–291 µs per call); in all 12 non-KHI arms the 100 ns delay is (almost)
fully absorbed. The delay is a device-timer busy-wait inside
`DeviceAllocator::malloc`/`free`, so while it spins the rest of the pipeline
keeps working — exactly "hidden by other asynchronous tasks when allocation is
not the bottleneck." The refinement: the hiding **fades smoothly
(hyperbolically)**, not as a sharp bottleneck switch. The literal model
`T = W + N·max(c_a+s, H)` is **rejected**: in all 18 shape-resolvable arms the
sharp-knee (step) fit is 1.5–103× worse (SSR) than the hyperbola. Four arms
over-expose (`d < 0`) and the twelve hal KHI arms show small-delay
*amplification humps* (E(100 ns) = +0.4 to +12.3 s) — unexplained anomalies no
hiding model can produce.

**Vertical vs horizontal.** The robust, gauge-invariant, flat-direction-free
quantity is the **vertical plateau deficit `d`** (use `c = d/N`). The
horizontal "plateau length" (the knee) is the fade scale `s0` — a flat,
seed-dependent fit direction, out of range in the slow-fade rosi KHI arms
(`s0` = 12.5–72.9 ms), and undefined in the anomalous arms; where it works
(clean arms) it just re-measures `d/N` (knee ≈ `s0` ≈ `d/N`). So **"measure the
plateau length" is not an independent measure** of the allocation cost — use
the vertical `c = d/N`. (Correction to the preliminary note: the smallest
non-zero delay is 100 ns, so the bend *is* in the data in most arms; the first
"no visible plateau" impression mixed up the smallest and largest delays.)

**Absolute vs relative.** Under the supported reading, `c = d/N` is the
**unused hiding capacity `H − c_a`**, not the native cost — so an *absolute
native cost cannot be deduced* from these data (3–291 µs/call is implausible
as a device-allocation cost, plausible as 0.3–34 % of the per-call pipeline
budget `T0/N`). A *relative* measure is available: for two algorithms in the
same (machine, setup, grid), `Δc = (d_B/N_B) − (d_A/N_A)` is well-defined,
gauge-invariant, flat-robust, and needs no `H`; **if `H` is invariant across
the algorithms, `Δc` is the relative native-cost difference**. The assumption
decomposes: the structural part holds (`N` is algorithm-invariant at
0.00–1.11 %), but `H` itself is untestable (only `H − c_a = d/N` is measured,
and it varies 5–46 % across algorithms and up to 346 µs across malloc/free) —
so the relative statement is conditional on an unverified `H`-invariance.

**Comparing the algorithms** (`c = d/N`, µs/call; hiding reading, larger =
more unused capacity = faster if `H` constant):

| scenario | FlatterScatter | ScatterAlloc | Δ(Flat−Scat) |
|---|---|---|---|
| FoilLCT 256×1280 malloc | 65.5 ± 3.2 | 68.6 ± 11.7 | −3.1 ± 12.1 (n.s.) |
| KHI 256×128×128 malloc | 104.2 ± 7.2 | 57.9 ± 15.6 | **+46.4 ± 17.1 (2.7σ)** — Flat cheaper |
| KHI 256×128×256 malloc | 165.0 ± 7.9 | 241.3 ± 9.4 | **−76.3 ± 12.3 (6.2σ)** — Scat cheaper |
| KHI 256×128×256 free | 176.6 ± 10.7 | 225.8 ± 12.2 | **−49.2 ± 16.1 (3.1σ)** — Scat cheaper |
| KHI 128³ | — | — | no significant plateau, unrankable |

The ranking **flips between the two large grids** — under `H`-invariance the
native-cost advantage switches with problem size; under the native-cost reading
(§8 interpretation II) the same numbers rank the algorithms the other way. The
data cannot adjudicate the interpretation; a zero-delay `DeviceAllocator`
microbenchmark would (and would also give `H = c_a + d/N`). rosi allows no
comparison (single algorithm).

## References and reproduction

- `analysis/allocation_model.py:1–31` (model, bounds), `:218` (degeneracy),
  `:394–421` (A<0 / s0-at-cap diagnostics)
- README:11–12, 37–46 (per-call delay injection), 53–76 (line derived first)
- Commits: `c94766d` (synthetic validation), `da3063f` (bounds, f-floor),
  `a496cda` + `1d8b306` (documented f drift), `c52b7c1` ("stands in for"),
  `4c43718` (f annotations moved into the forest plot)
- Data: `output/results.h5` (`runs`, `fits`, `shared_fits`, `baselines`,
  `fits/cov`); reproduction: `python critique_fit.py` (blocks: 1 nested-model
  tests, 2 multi-modality, 3 seed sensitivity, 4 baseline-anchored
  absorption, 5 stored vs data-pinned decomposition, 6 gauge symmetry)
- Follow-up 1 (flat direction / reparameterization, §7): `qa-flat-direction.md`
  + `qa_flat_direction.py` (per-fit pcov eigen-decomposition, function-space
  rank, reparameterization condition numbers, options A–F)
- Follow-up 2 (plateau / allocation cost / algorithm comparison, §8):
  `qa-allocation-cost.md` + `qa2_analysis.py`, `qa2_table.py` (per-arm N, d,
  c, E, O; hyperbola/step/line shape fits; parametric-bootstrap errors;
  scenario comparison)
