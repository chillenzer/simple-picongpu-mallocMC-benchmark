# Q2 — Can the delay-sweep plateau measure the (de-)allocation cost?

**Data:** `output/results.h5`, 12 fit groups / 24 arms (3558 runs), 2 algorithms
(`FlatterScatter`, `ScatterAlloc`) x 2 setups (`FoilLCT 256x1280`,
`KelvinHelmholtz 128^3 / 256x128x128 / 256x128x256`) on 2 machines (`hal` = A30,
`rosi` = V100). `rosi` was only run with `FlatterScatter`.
**Scripts:** `/workspace/qa2_analysis.py` (shape fits, first-pass table) and
`/workspace/qa2_table.py` (parametric-bootstrap errors, final tables).
Run with `/tmp/opencode/venv/bin/python`.

---

## 0. Bottom line (answers first)

1. **The hiding claim is verified.** 20 of 24 arms show the saturating
   absorption (d > 0: the excess E(s) = T(s) − T(0) − N·s is negative at all
   large delays and saturates at −d, d = 0.4–70 s per run = 3–291 µs per
   call); all 12 non-KHI arms additionally show the small-delay part (the 100
   ns delay is (almost) fully absorbed: E(100 ns) within the run-to-run
   spread, |E| ≤ 5.1 s, typically ≤ 1 s, and E(10 µs) < 0 in 11 of them),
   with the bend inside the measured range (by 10 µs in the hal arms, by
   100 µs–10 ms in the slow-fade rosi KHI arms). The offset above the
   large-delay line decays **smoothly (hyperbolically)**, not as a sharp knee.
   **Four arms do not show hiding** (over-exposed, d < 0): rosi FoilLCT free
   (−8.7 ± 1.2 s), rosi KHI 128³ free (−4.0 ± 7.8 s), hal KHI 128³ Scat free
   (−8.3 ± 2.8 s) and −malloc (−4.2 ± 2.8 s); **all twelve hal KHI arms start
   above the baseline** (E(100 ns) = +0.4 to +12.3 s, amplification humps) and
   one arm (hal FoilLCT Flat free) is non-monotonic. These are the known
   anomalies; no hiding model (and the fitted model, E ≤ 0 by construction) can
   produce them.
2. **The user's literal model T = W + N·max(c_a + s, H) is ruled out by the
   data.** In all 18 arms whose offset has a resolvable decaying shape, the
   sharp-knee (step) fit is 1.5–103× worse (SSR) than the smooth hyperbola;
   the 6 arms where the step does not lose are exactly the pathological ones
   (over-exposed / non-monotonic / hump), where neither shape is resolved.
   What the data show is a *smoothly fading hiding capacity*: the absorbed
   part of the delay is A·s0/(s+s0), total A = d, fade scale
   s0 = 13 µs–1.23 ms in the clean and hump arms, 12.5–72.9 ms in the
   slow-fade rosi KHI arms (the stored pipeline pins f0 at its 50 ms search
   cap in one of them).
3. **Interpretation (II) (A = N·c_a, "native allocation time") and the smooth
   hiding model are the same mathematical curve.** The data cannot statistically
   distinguish them; they differ only in the label put on the baseline offset
   d = N·(H − c_a). Physical plausibility decides: d/N = 3–291 µs per call is
   **implausible as a native GPU-side allocation cost** (device allocators cost
   ~100 ns–µs, even contended) but **plausible as pipeline slack** (it is
   0.3–34 % of the per-call time budget T0/N = 370–2100 µs). The data therefore
   **support the hiding interpretation (I) and not the native-cost reading
   (II)** — but only by plausibility; an independent zero-delay microbenchmark
   of `DeviceAllocator::malloc`/`free` is what would make it conclusive.
4. **The H-constant assumption:** the structural part is verified — N (calls
   per run) is algorithm-invariant at the 0.00–1.11 % level. The rest is not
   testable: the data fix only the combination H − c_a = d/N, which varies
   5–46 % between the two algorithms of the same scenario (significant,
   opposite sign at the two large KHI grids) and by up to 346 µs between malloc
   and free. A single H per (machine, setup, grid) covering both operations is
   **not** consistent with the data (it would force |c_m − c_f| up to 346 µs).
5. **The horizontal "plateau length" is not a usable standalone measure.**
   In the clean arms the data knee, the fitted fade scale s0 and d/N all
   coincide (63–178 µs), so "measuring the plateau" just re-measures d/N — but
   the knee *as a fit parameter* is the fade scale s0, a flat direction of the
   fit (condition number 1e11–1e14, seed-dependent, 26–189× the quoted error
   along the gauge orbit), and in the slow-fade rosi KHI arms the knee lies
   *outside* the measured range (fitted s0 = 12.5–72.9 ms; the stored
   pipeline caps f0 at 50 ms in one of them), so no plateau length is
   measurable there at all. The **vertical deficit d = T(0) − b0 is the
   robust, gauge-invariant, flat-direction-free quantity**; use c = d/N, not
   the knee.
   *Correction of the preliminary note:* the smallest non-zero delay is **100 ns
   in every group, not 5–56 ms** (5–56 ms is the *largest* delay). The stored
   fade scales (16 µs–2.2 ms, one at the 50 ms cap) are **160–22 000× larger
   than the smallest delay** (500 000× for the one at the cap), so the bend is
   inside the measured range in most
   arms — the plateau *is* visible: E(100 ns) is within the run-to-run spread
   in all 12 non-KHI arms (|E| ≤ 5.1 s, usually ≤ 1 s) and the bend (E leaving
   0) occurs inside the measured range (by 10 µs in the hal arms, by
   100 µs–10 ms in the slow-fade rosi KHI arms). d is not "pure extrapolation"
   in those arms. It *is* pure extrapolation in the
   four slow-fade rosi KHI arms (the offset is still 3–23 s at the third
   largest delay).
6. **Absolute cost: no — not from these data.** c = d/N is well defined,
   gauge-invariant and robust, but under the supported (hiding) reading it is
   H − c_a (unused hiding capacity), not the native cost. The native cost c_a
   is bounded only by 0 ≤ c_a ≤ H, and H is unknown; d/N is a lower bound on H,
   not a measurement of c_a.
7. **Relative cost: yes, conditionally.** For two arms of the same
   (machine, setup, grid), Δc ≡ (d_B/N_B) − (d_A/N_A) is well defined,
   gauge-invariant, flat-direction-robust and needs no knowledge of H. **If H
   is invariant across the algorithms, Δc = c_a,B − c_a,A is the relative
   native-cost difference** (sign as written). The knee-based method
   (s*_A − s*_B with H constant) gives the same number in the clean arms (knee
   ≈ d/N) and less elsewhere — it adds nothing.
8. **Algorithms:** under the hiding reading + H-invariance, the ranking flips
   between the two large grids: at KHI 256×128×128 `FlatterScatter` is
   ~46 µs/call *cheaper* in native malloc cost than `ScatterAlloc`
   (104.2 ± 7.2 vs 57.9 ± 15.6 µs of H − c_a; 2.7σ), while at KHI
   256×128×256 `ScatterAlloc` is ~76 µs/call cheaper for malloc
   (165.0 ± 7.9 vs 241.3 ± 9.4; 6.2σ) and ~49 µs/call cheaper for free
   (176.6 ± 10.7 vs 225.8 ± 12.2; 3.1σ). FoilLCT 256×1280 shows no
   significant difference (65.5 ± 3.2 vs 68.6 ± 11.7 µs). KHI 128³ shows no
   significant plateau at all. Under the (II) reading the *same numbers* rank
   the algorithms the other way (smaller d/N = cheaper); the interpretation
   matters and the data cannot adjudicate it.

---

## 1. Per-arm quantities (definitions, all from raw data)

Per arm (malloc arm = m > 0, f == 0; free arm = m == 0, f > 0), per-delay
medians (x sorted):

- `N` = (y[-1] − y[-2]) / (x[-1] − x[-2]) — large-delay slope = calls/run.
- `b0` = y[-1] − N·x[-1] — large-delay line intercept.
- `d` = t00 − b0 — **plateau deficit** (baseline minus large-delay-line
  intercept; pure data, gauge-invariant).
- `c` = d/N — per-call quantity (gauge-invariant).
- `E(s)` = T(s) − t00 − N·s — excess above the nominal line through the
  baseline (hiding: E < 0 at small s).
- `O(s)` = T(s) − b0 − N·s — offset above the large-delay line;
  O(0) = d, O(∞) = 0.
- `slope_sm/N` — local slope between the two smallest delays (100 ns → 10 µs)
  relative to N.

Errors on d and c: parametric bootstrap (600 iterations) resampling each
delay cell as Normal(median, σ_cell) with σ_cell = the within-cell spread
where n ≥ 3, else the group's per-run spread (0.36–3 s) as a noise floor —
this keeps the errors valid for the single-replicate ScatterAlloc arms.

Shape fits of O(s):
- **(H)** hyperbola O = a·s0/(s+s0) (2 params; the model's form),
- **(0)** zero-parameter family O = d·s0'/(s+s0') **with s0' = d/N fixed**
  (the "smoothed step" whose knee equals d/N),
- **(L)** literal sharp knee O = d (s < s*), 0 (s ≥ s*) (1 param; the
  user's max(c_a + s, H) model),
- **(0l)** no offset O = 0.

---

## 2. Per-arm table

All 24 arms (errors: parametric bootstrap; "hiding" = d > 0, E(10 µs) < 0,
E(100 ns) ≈ 0, O decreasing; flags: **Hump** = E(100 ns) > 0, **OverExp** =
d < 0, **NonMono** = O or T non-monotonic, **SlowFade** = O(3rd-largest)
> 0.3·d):

| machine setup grid | alg | arm | N | d ± σ (s) | c = d/N ± σ (µs) | E(100 ns) (s) | E(10 µs) (s) | E(max) (s) | O(3rd) (s) | slsm/N | s0 fit (µs) | stored m0/f0 (µs) | knee (µs) | s0/d/N | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hal Foi 256x1280 | Flat | malloc | 2.344e5 | +15.36 ± 0.76 | 65.5 ± 3.2 | −0.14 | −2.04 | −15.36 | +1.32 | 0.18 | 63.9 | 66.4 | 63.1 | 0.98 | — (clean) |
| hal Foi 256x1280 | Flat | free | 1.555e5 | +0.95 ± 0.94 | 6.1 ± 6.1 | −0.15 | −1.08 | −0.95 | −6.60 | 0.39 | 1.0 | 17.0 | 10 | 0.17 | NonMono |
| hal Foi 256x1280 | Scat | malloc | 2.345e5 | +16.09 ± 2.76 | 68.6 ± 11.7 | −0.03 | −1.79 | −16.09 | +1.83 | 0.24 | 174.2 | 56.4 | 100 | 2.54 | — (clean) |
| hal Foi 256x1280 | Scat | free | 1.553e5 | +10.68 ± 2.86 | 68.8 ± 18.4 | +0.11 | −1.05 | −10.68 | −1.67 | 0.25 | 44.5 | 97.4 | 100 | 0.65 | — |
| hal Kel 128^3 | Flat | malloc | 1.333e5 | +0.43 ± 2.71 | 3.2 ± 20.3 | +1.60 | +8.22 | −0.43 | −2.64 | 6.02 | 160.7 | 775.9 | — | 50.0 | Hump |
| hal Kel 128^3 | Flat | free | 7.975e4 | +0.45 ± 3.44 | 5.6 ± 43.2 | +8.63 | +9.11 | −0.45 | −1.43 | 1.62 | 374.4 | 1221 | — | 67.0 | Hump |
| hal Kel 128^3 | Scat | malloc | 1.340e5 | −4.20 ± 2.80 | −31.3 ± 20.9 | +9.28 | +8.75 | +4.20 | −8.64 | 0.60 | 60.2 | 544.7 | — | −1.9 | Hump, OverExp |
| hal Kel 128^3 | Scat | free | 7.887e4 | −8.28 ± 2.80 | −105 ± 35.6 | +8.70 | +8.89 | +8.28 | −6.17 | 1.24 | 13.0 | 644 | — | −0.1 | Hump, OverExp |
| hal Kel 256x128x128 | Flat | malloc | 1.864e5 | +19.42 ± 1.35 | 104.2 ± 7.2 | +4.66 | +10.36 | −19.42 | +2.25 | 4.09 | 650.8 | 1258 | — | 6.2 | Hump |
| hal Kel 256x128x128 | Flat | free | 1.363e5 | +17.51 ± 2.79 | 128.5 ± 20.2 | +11.22 | +10.25 | −17.51 | +3.38 | 0.28 | 716 | 1735 | 1778 | 5.6 | Hump |
| hal Kel 256x128x128 | Scat | malloc | 1.866e5 | +10.79 ± 2.91 | 57.9 ± 15.6 | +12.28 | +11.15 | −10.79 | −13.94 | 0.39 | 149.4 | 1244 | 1000 | 2.6 | Hump |
| hal Kel 256x128x128 | Scat | free | 1.353e5 | +18.88 ± 2.94 | 139.6 ± 21.6 | +12.13 | +12.08 | −18.88 | +1.96 | 0.96 | 935.8 | 1641 | 3162 | 6.7 | Hump |
| hal Kel 256x128x256 | Flat | malloc | 2.880e5 | +47.51 ± 2.30 | 165.0 ± 7.9 | +0.39 | +1.77 | −47.51 | +2.44 | 1.48 | 550.8 | 1244 | 562 | 3.3 | Hump (mild) |
| hal Kel 256x128x256 | Flat | free | 2.275e5 | +40.17 ± 2.48 | 176.6 ± 10.7 | +1.37 | −0.22 | −40.17 | +1.07 | 0.30 | 604.2 | 2238 | 1000 | 3.4 | Hump (mild) |
| hal Kel 256x128x256 | Scat | malloc | 2.897e5 | +69.92 ± 2.74 | 241.3 ± 9.4 | +2.40 | +0.70 | −69.92 | +11.09 | 0.41 | 1233 | 985.1 | 1778 | 5.1 | Hump (mild) |
| hal Kel 256x128x256 | Scat | free | 2.275e5 | +51.36 ± 2.79 | 225.8 ± 12.2 | +5.39 | +3.34 | −51.36 | −3.15 | 0.09 | 729.4 | 1539 | 1778 | 3.2 | Hump (mild) |
| rosi Foi 256x1280 | Flat | malloc | 1.621e5 | +25.46 ± 1.03 | 157.1 ± 6.2 | −0.62 | −2.09 | −25.46 | +0.07 | 0.09 | 105.6 | 148.0 | 177.8 | 0.67 | — (clean) |
| rosi Foi 256x1280 | Flat | free | 1.199e5 | −8.70 ± 1.19 | −72.6 ± 9.9 | −0.63 | −1.20 | +8.70 | −15.54 | 0.53 | ~0 | 160.1 | — | — | OverExp |
| rosi Kel 128^3 | Flat | malloc | 1.287e5 | +37.46 ± 5.24 | 291 ± 41 | −0.11 | −0.65 | −37.46 | +19.13 | 0.58 | 23 140 | 98.7 | 56 234 (edge) | 79.5 | SlowFade |
| rosi Kel 128^3 | Flat | free | 7.248e4 | −3.99 ± 7.77 | −55.1 ± 107 | −0.13 | −0.40 | +3.99 | −5.66 | 0.62 | ~0 | 1228 | — | — | OverExp |
| rosi Kel 256x128x128 | Flat | malloc | 1.521e5 | +36.45 ± 11.5 | 239.6 ± 75.4 | +0.33 | +0.25 | −36.45 | +3.45 | 0.95 | 12 454 | 776.2 | 31 623 | 52.0 | T nonmono; s0 ≫ d/N |
| rosi Kel 256x128x128 | Flat | free | 9.936e4 | +7.49 ± 13.1 | 75.4 ± 132 | −0.01 | −0.41 | −7.49 | +8.50 | 0.59 | 72 921 | 50 000 (at cap) | 56 234 (edge) | 968 | SlowFade, O nonmono |
| rosi Kel 256x128x256 | Flat | malloc | 2.236e5 | +35.44 ± 24.5 | 158.5 ± 109 | −3.82 | −5.11 | −35.44 | +23.29 | 0.42 | 13 233 | 224.6 | 10 000 | 83.5 | SlowFade, T nonmono |
| rosi Kel 256x128x256 | Flat | free | 1.738e5 | +11.49 ± 32.2 | 66.1 ± 185 | −1.75 | −3.56 | −11.49 | +5.15 | −0.05 | 98.0 | 16.1 | 100 | 1.5 | T nonmono; O3rd = 0.45·d |

"knee" = first delay where E ≤ −d/2 (the data's horizontal plateau end).
"s0/d/N" = fitted fade scale divided by d/N (≈1 in the clean arms).

---

## 3. Q2.1 — Verdict on the hiding claim

**The claim holds where the data are clean (12 non-KHI arms), the
large-delay saturation holds in 20 of 24 arms (all with d > 0), and it fails
in 4 arms (d < 0) plus the small-delay part fails in the 12 hal KHI hump
arms.** Per-arm, with the three requested tests:

- **(a) slope_sm/N < 1:** yes in 19 of 24 arms (0.09–0.96); the five
  exceptions are hal KHI 128³ Flat malloc (6.02), KHI 128³ Flat free (1.62),
  KHI 256×128×128 Flat malloc (4.09), KHI 128³ Scat free (1.24) and KHI
  256×128×256 Flat malloc (1.48) — the small delay is *amplified* there, the
  opposite of hiding.
- **(b) E(s) < 0 at small delay:** in the 12 non-KHI arms E(100 ns) is within
  the per-run spread (|E| ≤ 5.1 s, typically ≤ 0.7 s), and E(10 µs) < 0 in
  11 of 12 (−5.1 to −0.2 s; the one exception, rosi KHI 256×128×128 malloc,
  is marginal at +0.25 s): the 100 ns and 10 µs delays are (almost) fully
  hidden. In the 12 hal KHI arms E(100 ns) = +0.4 to +12.3 s — above the
  baseline.
- **(c) curve reaches the N·s line at large delay:** yes in 9 of 24 arms
   (0 ≤ O at the third-largest delay ≤ 3.5 s, i.e. the offset has faded from
   above to within 19 % of d), with two more
  marginal (hal KHI 256×128×256 Scat malloc 11.1 s; rosi KHI 256×128×128
  free 8.5 s). Not in the slow-fade rosi KHI arms (O = +3.4 to +23.3 s there,
  i.e. 9–114 % of d still above the line — the fade extends beyond the data
  range) and not in the hump/over-exposed arms (O negative mid-range: the
  curve dips *below* the line and may cross back above it at the top).

**Arms that do NOT show hiding (flagged, as expected):**

| arm | d (s) | pathology |
|---|---|---|
| hal KHI 128³ / 256×128×128 / 256×128×256, both algs (12 arms) | +0.4 … +69.9 | **amplification humps**: E(100 ns) > 0, 8 of 12 peak at +8.2 to +12.3 s at 10–20 µs, then decay to −d. No hiding model (E ≤ 0) can produce E > 0. |
| rosi Foi 256×1280 Flat free | −8.70 ± 1.19 | **over-exposed**: E reaches −20.9 s at 1 ms then recovers to +8.7 s = −d at 56 ms; the largest delays add *more* than N·s. |
| rosi KHI 128³ Flat free | −3.99 ± 7.77 | **over-exposed** (same shape, smaller). |
| hal KHI 128³ Scat free | −8.28 ± 2.80 | hump + over-exposed. |
| hal KHI 128³ Scat malloc | −4.20 ± 2.80 | hump + over-exposed. |
| hal Foi 256×1280 Flat free | +0.95 ± 0.94 | **non-monotonic**: E dips to −14.0 s at 562 µs, recovers to −0.95 s at 32–56 ms (O goes to −13.1 s, below the large-delay line, then back to 0). d not significant. |

Representative profiles (E and O vs delay; '#' = 1 s):

```
hal Foi 256x1280 Flat malloc (clean):            hal Kel 128^3 Flat malloc (hump):
 s=0.1us  E=-0.14  O=+15.23                       s=0.1us  E=+1.60  O=+2.02
 s=10us   E=-2.04  O=+13.32                       s=10us   E=+8.22  O=+8.65
 s=100us  E=-9.37  O=+5.99                        s=12.6us E=+8.89  O=+9.32   <- peak
 s=500us  E=-13.40 O=+1.96                        s=631us  E=+0.03  O=+0.46   <- crossing
 s=5.6ms  E=-15.36 O=+0.00                        s=1ms    E=-3.27  O=-2.84
 s=10ms   E=-15.36 O=+0.00                        s=3.2ms  E=-3.07  O=-2.64
                                                 s=5.6ms  E=-0.43  O=+0.00
rosi Kel 128^3 Flat malloc (slow fade):          rosi Foi 256x1280 Flat free (over-exposed):
 s=0.1us  E=-0.11  O=+37.35                       s=0.1us  E=-0.63  O=-9.34
 s=100us  E=-3.36  O=+34.10                       s=1ms    E=-20.86 O=-29.56
 s=3.2ms  E=-6.21  O=+31.25                       s=17.8ms E=-15.22 O=-23.93
 s=10ms   E=-11.64 O=+25.82                       s=31.6ms E=-6.84  O=-15.54
 s=31.6ms E=-18.33 O=+19.13                       s=56.2ms E=+8.70  O=+0.00  <- crosses above
 s=56.2ms E=-37.46 O=+0.00
 s=100ms  E=-37.46 O=+0.00
```

**Mechanism note.** The delay is a busy-wait on the *device* global timer at
the top of mallocMC's `DeviceAllocator::malloc`/`free` (README:43–50), i.e. a
GPU-side stall of the allocation thread. While it spins, other GPU blocks/
streams and the host keep working; only the part of the stall that extends the
critical path reaches the measured runtime. That is exactly the "hidden by
other asynchronous tasks when allocation is not the bottleneck" picture — with
the refinement that the hiding **fades smoothly** with s (no sharp
bottleneck switch) and **saturates** (the absorbed part is bounded by d per
run, not proportional to s).

---

## 4. Q2.3/interpretation — shape: hyperbola vs sharp knee, (II) vs hiding

**Shape test** (per arm, SSR over the per-delay medians of O(s)):

| arm class | SSR hyperbola | SSR step (max model) | SSR 0-param family (knee = d/N) | SSR line (O=0) |
|---|---|---|---|---|
| hal Foi Flat malloc | 4.0 | 414 (×103) | **4.2** | 1500 |
| hal Foi Scat malloc | 17.5 | 98 (×5.6) | 25.2 | — |
| rosi Foi Flat malloc | 15.4 | 334 (×22) | 41.3 | — |
| hal KHI 256×128×128 Flat free | 26.1 | 414 (×16) | 1377 | — |
| hal KHI 256×128×256 Flat malloc | 109.7 | 3226 (×29) | 3607 | — |
| rosi KHI 128³ Flat malloc | 247.2 | 794 (×3.2) | 5373 | — |
| rosi KHI 256×128×256 malloc | 362.7 | 2436 (×6.7) | 2266 | — |
| **18 of 24 arms: hyperbola better by 1.5–103×; the 6 pathological arms (hal Foi Flat free, hal KHI 128³ Scat m/f, rosi Foi Flat free, rosi KHI 128³ Flat free, rosi KHI 256×128×128 Flat free): tied or step slightly better, but with SSR 38–1108 neither shape is resolved there** | | | | |

- The **sharp-knee (literal max) model is rejected** in every arm where the
  offset has a resolvable decaying shape (18 arms): the data's fade is smooth,
  hyperbolic, O(s) = a·s0/(s+s0), with a ≈ d (a/d = 0.7–2.2 across the
  positive-d arms, ≈1.0 in the clean ones: 1.00, 0.95, 1.05, 1.02, 0.97).
- In the clean arms the **zero-parameter family already fits within the noise**
  (SSR 4.2 vs 4.0 for hal Foi Flat malloc): the bend scale is d/N itself —
  data knee (63.1 µs), fitted s0 (63.9 µs) and d/N (65.5 µs) all agree, and
  the stored m0 (66.4 µs) as well. The data are, in the clean arms, a
  one-parameter family: T(s) = t00 + N·s − d·s/(s + d/N).
- In the **slow-fade rosi KHI arms** the offset is still 9–114 % of d at the
  third-largest delay, the (unbounded) hyperbola fit wants s0 = 12.5–72.9 ms
  (s0/(d/N) = 52–968; the stored pipeline pins f0 at its 50 ms search cap in
  one of these arms): the "plateau" (E ≈ 0 region) ends by ~10 µs but the
  fade out to the N·s line takes 10–100 ms. There d = t00 − b0 is a **pure
  extrapolation** and is likely an overestimate (the large-delay line is
  anchored where the offset has not fully faded).
- In the **12 hump arms** E(100 ns) > 0: no model of the form "delay absorbed
  up to a capacity" (E ≤ 0 by construction) fits the small-delay part; the
  large-delay part does decay to −d. These arms are a separate, unexplained
  phenomenon (amplification: E(100 ns) = +0.4 to +12.3 s against a nominal
  N·s = 0.008–0.029 s, and E(10 µs) up to +12.1 s against a nominal
  N·s = 0.8–2.9 s — in 150–600 s runs the run slows several times more than
  the sum of the injected stalls — consistent with a pipeline-drain/refill
  or lock-contention effect, not with hiding).

**Which interpretation do the data support?** Write the smooth hiding model as
T(s) = W' + N·c_a + N·s + N·(H − c_a)·s0/(s+s0): the *excess* hiding capacity
H − c_a (per call) is what fades. Its zero-delay value is W' + N·H and its
large-delay intercept W' + N·c_a, so **d = N·(H − c_a)** — *independently of
the fade shape*. Interpretation (II) writes the identical curve as
T(s) = W + N·s + A·s0/(s+s0) with A = N·c_a and W = W' + N·c_a. **The two are
the same function with the same fitted (A, s0, W); the readings differ only in
where the baseline offset d is booked** — "native allocation time" (II) or
"critical-path occupancy / slack of the pipeline" (hiding). Consequently:

- The data **cannot statistically distinguish** (II) from the hiding model:
  same curve, same parameters. They *can* reject the user's literal
  max() form (done above, by smoothness).
- **Physical plausibility decides.** d/N = 3–291 µs per call. A native
  GPU-side allocation (device spinlock, pool bookkeeping) costs ~100 ns–µs,
  not tens to hundreds of µs — (II) would require malloc/free to be
  30–300× slower than any device allocator is. As pipeline slack, d/N is
  0.3–34 % of the per-call time budget T0/N = 370–2100 µs (e.g. hal FoilLCT:
  65.5 µs of a 370 µs per-call budget; rosi KHI 128³: 291 µs of 860 µs) — a
  plausible fraction of in-flight GPU/host work that can absorb a stall.
- The user's hypothesis ("short plateau → slow allocation") has the **sign of
  the hiding model** (plateau end s* = H − c: larger c → shorter plateau) and
  not of (II) (where a larger c gives a *larger* vertical deficit d = N·c).
  But since the data give only H − c_a (not H or c_a separately), **the data
  cannot confirm the hypothesis**: a short plateau means "little unused
  capacity", which for a *constant H* would mean "slow allocation" — but H is
  not established to be constant (next section), so the inference does not
  follow from these data alone.

**What would confirm the interpretation:** a zero-delay microbenchmark of
`DeviceAllocator::malloc`/`free` (or mallocMC's own call counters) to measure
c_a directly. If c_a ≪ d/N (expected: ≤ µs vs ≥ tens of µs), the hiding
reading is confirmed, H = c_a + d/N is then known per arm, and the native cost
is *not* d/N. The repo's own review (analysis-review.md) reaches the same
verdict from the fit side; this analysis adds the shape evidence (smooth fade,
knee ≈ d/N in the clean arms) and the error bars.

---

## 5. Q2.2 — Is H (the hiding capacity) constant across algorithms?

The user's assumption decomposes into two parts:

**(i) "Changing the allocation algorithm does not change the program
structure around it."** Verified for the call count: within each
(machine, setup, grid), N agrees between the two algorithms at the
**0.00–1.11 %** level (hal Foi: 0.02/0.11 %, hal KHI 128³: 0.53/1.11 %,
256×128×128: 0.11/0.78 %, 256×128×256: 0.59/0.00 %) — far below any plausible
c_a effect. ✓

**(ii) "The range of allocation times that should not produce overhead is
approximately constant."** That range is (up to c_a) the hiding capacity H.
The data fix only H − c_a = d/N per arm, so the implied H per arm is
H_i = c_a,i + d_i/N_i ≥ d_i/N_i; the spread of the *implied* H across
algorithms of one scenario equals the spread of d/N:

| scenario | arm | Flat d/N (µs) | Scat d/N (µs) | spread | Δ(d/N), Flat−Scat |
|---|---|---|---|---|---|
| hal Foi 256×1280 | malloc | 65.5 ± 3.2 | 68.6 ± 11.7 | 4.8 % | −3.1 ± 12.1 µs (0.3σ) |
| hal Foi 256×1280 | free | 6.1 ± 6.1 (NonMono) | 68.8 ± 18.4 | 11× | −62.7 ± 19.4 µs (3.2σ, Flat arm anomalous) |
| hal KHI 128³ | malloc | 3.2 ± 20.3 | −31.3 ± 20.9 | (d not significant) | — |
| hal KHI 128³ | free | 5.6 ± 43.2 | −105 ± 35.6 | (d not significant) | — |
| hal KHI 256×128×128 | malloc | 104.2 ± 7.2 | 57.9 ± 15.6 | 44 % | **+46.4 ± 17.1 µs (2.7σ)** |
| hal KHI 256×128×128 | free | 128.5 ± 20.2 | 139.6 ± 21.6 | 9 % | −11.1 ± 29.7 µs (0.4σ) |
| hal KHI 256×128×256 | malloc | 165.0 ± 7.9 | 241.3 ± 9.4 | 46 % | **−76.3 ± 12.3 µs (6.2σ)** |
| hal KHI 256×128×256 | free | 176.6 ± 10.7 | 225.8 ± 12.2 | 28 % | **−49.2 ± 16.1 µs (3.1σ)** |

Verdict:

- The assumption is **not verified**. If H were algorithm-invariant, the
  5–46 % spread in d/N would be the relative *native-cost* difference
  between the algorithms; 46–76 µs per call is a large native-cost
  difference for device allocators, so it is equally possible that H itself
  varies with the algorithm (different lock/pool behaviour changes the
  pipeline structure — the very structure the user assumed invariant).
  **The data measure Δ(H − c_a) and cannot split it.**
- **Across operations the assumption fails:** (d/N)_free − (d/N)_malloc spans
  −346 to +82 µs between the 12 groups (e.g. rosi KHI 128³: 291 vs −55 µs;
  rosi Foi: 157 vs −73 µs). A single H covering both malloc and free would
  force |c_malloc − c_free| up to ~346 µs per call — implausible for the
  native paths. More likely H_m ≠ H_f (the two operations stall the pipeline
  differently), which is also consistent with the three over-exposed free
  arms.
- Lower bounds on H (taking c_a ≥ 0): H ≥ max(d/N) of the scenario —
  65–69 µs (hal Foi), 104–140 µs (hal KHI 256×128×128), 165–241 µs
  (hal KHI 256×128×256), 157–291 µs (rosi). If c_a ≤ ~1 µs (plausible native
  device cost), then H ≈ d/N within ~1–3 % and H is **not** constant across
  algorithms at better than the 5–46 % level observed.

---

## 6. Q2.3 — Vertical deficit vs horizontal knee: which is usable?

- **Vertical d = t00 − b0:** computed from raw data only (baseline median,
  line through the two largest delays). Gauge-invariant (the model's
  delay-origin gauge, analysis-review §6, leaves N, b0 and hence d
  untouched — unlike the fitted A, which moves along the gauge orbit and
  drifts up to 64 s in W (and up to 44 s in the A's) from the data-pinned values in 8 of 12 groups. Flat-direction-robust: d does not
  involve the fade scale at all, so it is immune to the A–s0–W degeneracy
  (pcov condition 1e11–1e14) that makes the stored A_malloc/A_free and
  m0/f0 seed-dependent.
- **Horizontal knee:** the data's knee (first E ≤ −d/2) is a real, observable
  feature in the clean arms (63–178 µs, above the 100 ns smallest delay —
  the plateau *is* in the data, correcting the preliminary note). But the
  knee as a *model parameter* is the fade scale s0, which is (i) the flat
  direction of the 7-parameter fit, (ii) seed-dependent (the stored pipeline's
  own documented drift: f_malloc 9.72 % → 0.93 % across seeds), and
   (iii) unmeasurable in the slow-fade arms (fitted s0 = 12.5–72.9 ms; the
   stored pipeline caps f0 at 50 ms; the knee sits at the edge of, or
   beyond, the measured range). In those arms "measure the
  plateau length" is a model-dependent extrapolation contaminated by the
  fade scale, and in the hump/over-exposed arms it is not even defined.
- **Agreement:** where both are defined and clean, knee ≈ s0 ≈ d/N
  (s0/(d/N) = 0.65–2.5; hal Foi Flat: 0.98). So the horizontal method, when
  it works, just re-measures the vertical quantity. **Use c = d/N (vertical);
  the knee is a consistency check, not a measure.** The user's
  "measure the plateau length" method does not work as an independent
  allocation-cost measure with these data: it is (a) gauge/seed-contaminated
  via s0, (b) out of range in the slow-fade arms, (c) not defined in the
  anomalous arms, and (d) — even where clean — it measures H − c_a (pipeline
  slack), not c_a.

---

## 7. Q2.4 — Absolute vs relative cost

**Absolute.** c = d/N = 3–291 µs (malloc arms; free arms −105 … +226 µs with
sign flips between adjacent groups — free arms are the less reliable, three
over-exposed). It is a well-defined, gauge-invariant, flat-direction-robust
data quantity, but under the supported hiding reading it is **H − c_a, the
unused hiding capacity per call**, not the native cost. As an absolute
allocation cost it is implausible (30–300× a device allocator's real cost);
as slack it is 0.3–34 % of the per-call budget — plausible. **An absolute
native cost cannot be deduced from these data**; it requires the independent
zero-delay benchmark (then c_a is known and H = c_a + d/N per arm). What the
data do give absolutely: the total absorbable delay per run (d = 0.4–70 s),
the per-call slack bound (d/N), and the fade scale of the absorption
(s0 = 16 µs–1.23 ms in the clean/hump arms, 12.5–72.9 ms in the slow-fade rosi KHI arms — §4).

**Relative.** For two arms of the same (machine, setup, grid):
Δc ≡ (d_B/N_B) − (d_A/N_A). This is (i) well defined (pure data), (ii)
gauge-invariant, (iii) flat-direction-robust (no fade scale involved),
(iv) independent of H — **if H is invariant across the two arms, then
Δc = c_a,B − c_a,A exactly**, so it is the relative native-cost difference
without knowing H. The significant results (same sign convention
Flat − Scat): hal KHI 256×128×128 malloc +46.4 ± 17.1 µs (2.7σ),
hal KHI 256×128×256 malloc −76.3 ± 12.3 µs (6.2σ), free −49.2 ± 16.1 µs
(3.1σ); all others consistent with zero. The horizontal-knee method
(Δc = s*_B − s*_A assuming H constant) reproduces Δ(d/N) in the clean arms
and is less robust elsewhere — **no advantage over the vertical method**.

**Caveats on the relative measure:** (a) it inherits the H-invariance
assumption (unverified, §5); (b) in the slow-fade rosi KHI arms d is an
extrapolation with large errors (±5–32 s) and is biased high if the fade
extends beyond the largest delay; (c) the ScatterAlloc arms have single
replicates per delay, so their errors are dominated by the assumed per-cell
noise floor; (d) Δc mixes a possible algorithm effect on H with the native
cost difference — the 46–76 µs magnitudes at the large grids are large enough
that an H effect cannot be excluded.

---

## 8. Q2.5 — Comparing the algorithms

Per-operation cost table, c = d/N in µs per call (± parametric-bootstrap σ).
Under the supported reading, c = H − c_a: **larger c = more unused capacity =
(fastest if H constant)**. The "(II)" column shows the same numbers read as
native cost (smaller = faster) — the ranking is interpretation-dependent.

| scenario | algorithm | c_malloc (µs) | c_free (µs) | notes |
|---|---|---|---|---|
| hal Foi 256×1280 | FlatterScatter | 65.5 ± 3.2 | 6.1 ± 6.1 | free arm non-monotonic, d not significant |
| hal Foi 256×1280 | ScatterAlloc | 68.6 ± 11.7 | 68.8 ± 18.4 | Δc_malloc = −3.1 ± 12.1 µs: no difference |
| hal KHI 128³ | FlatterScatter | 3.2 ± 20.3 | 5.6 ± 43.2 | hump arms; d not significant — no cost statement possible |
| hal KHI 128³ | ScatterAlloc | −31.3 ± 20.9 | −105 ± 35.6 | hump + over-exposed (d < 0) — excluded |
| hal KHI 256×128×128 | FlatterScatter | 104.2 ± 7.2 | 128.5 ± 20.2 | hump at small s; d significant |
| hal KHI 256×128×128 | ScatterAlloc | 57.9 ± 15.6 | 139.6 ± 21.6 | Δc_malloc = +46.4 ± 17.1 µs (2.7σ): Flat cheaper (hiding reading) |
| hal KHI 256×128×256 | FlatterScatter | 165.0 ± 7.9 | 176.6 ± 10.7 | mild hump |
| hal KHI 256×128×256 | ScatterAlloc | 241.3 ± 9.4 | 225.8 ± 12.2 | Δc_malloc = −76.3 ± 12.3 (6.2σ), Δc_free = −49.2 ± 16.1 (3.1σ): Scat cheaper (hiding reading) |
| rosi Foi 256×1280 | FlatterScatter | 157.1 ± 6.2 | −72.6 ± 9.9 | free over-exposed (d < 0) |
| rosi KHI 128³ | FlatterScatter | 291 ± 41 | −55.1 ± 107 | malloc slow-fade (extrapolated d); free over-exposed |
| rosi KHI 256×128×128 | FlatterScatter | 239.6 ± 75.4 | 75.4 ± 132 | both slow-fade, large errors |
| rosi KHI 256×128×256 | FlatterScatter | 158.5 ± 109 | 66.1 ± 185 | slow-fade, marginal significance |

**Ranking (hiding reading + H-invariance; fastest = largest c):**
- FoilLCT 256×1280 (malloc): ScatterAlloc ≳ FlatterScatter (not significant).
- KHI 256×128×128 (malloc): **FlatterScatter faster** (ScatterAlloc 46 µs/call
  dearer); (free): no difference.
- KHI 256×128×256 (malloc and free): **ScatterAlloc faster** (FlatterScatter
  76 / 49 µs/call dearer).
- KHI 128³: no significant plateau — unrankable.
- rosi: single algorithm — no comparison possible (cross-machine comparison
  is not meaningful for c, since H is machine/pipeline dependent).

Under the (II) reading the ranking flips at every grid where the difference
is significant (smaller d/N = cheaper native cost). The sign flip between the
two large grids (Flat cheaper at 256×128×128, Scat cheaper at 256×128×256) is
itself informative: if H is invariant it says the *native cost advantage
switches with problem size* (plausible: scatter strategies' overhead scales
with segment/block count, which grows with the grid); if H is not invariant
it is instead an algorithm effect on the pipeline slack. The data cannot tell
which — the same limitation as §5.

**Caveats.** (1) All 12 hal KHI arms carry amplification humps at small s;
their d values are significant but their small-delay shape is not a hiding
shape — the hump contributes an unknown offset to the baseline-relative
picture, though d itself (baseline vs large-delay line) is unaffected by the
hump as long as the large-delay side is clean. (2) The four over-exposed arms
(d < 0) and the non-monotonic hal FoilLCT Flat free arm are excluded from any
cost statement. (3) The four slow-fade rosi KHI arms have d as pure
extrapolation with ±5–32 s errors. (4) ScatterAlloc arms have 1 replicate per
delay (n = 9–11) — errors are noise-floor dominated. (5) Only 4 scenarios
contain both algorithms; only 3 give significant relative differences.

---

## 9. Bottom line

1. **Hiding: yes, verified** — 20/24 arms show the saturating absorption
   (d > 0, 0.4–70 s per run = 3–291 µs per call, smooth hyperbolic fade, merge
   with the N·s line in 9/24 arms); the small-delay part (100 ns delay fully
   absorbed) is verified in all 12 non-KHI arms. The claim as
   stated ("hidden by other asynchronous tasks when allocation time is not the
   bottleneck") is the right mechanism, with the refinement that the hiding
   fades smoothly and saturates. Four arms over-expose (d < 0) and the twelve
   hal KHI arms amplify at small delay — unexplained anomalies, outside the
   hiding model.
2. **The literal max(c_a + s, H) model is rejected** (step fit 1.5–103× worse
   than the hyperbola in all 18 shape-resolvable arms; the 6 pathological arms
   resolve neither shape). A *smoothly fading* hiding capacity fits the data.
3. **(II) vs hiding:** mathematically indistinguishable (same curve); the
   data support the hiding label by physical plausibility (3–291 µs is
   implausible as a native device-allocation cost, plausible as 0.3–34 %
   per-call pipeline slack). An independent zero-delay microbenchmark of
   `DeviceAllocator::malloc`/`free` would make the call and would also
   provide H = c_a + d/N.
4. **H-constancy:** N is algorithm-invariant (≤ 1.1 %) — the structural part
   holds. H itself is not testable from these data (only H − c_a = d/N is
   measured); d/N varies 5–46 % across algorithms of one scenario and up to
   346 µs across operations, so a single H per (machine, setup, grid) is
   consistent with the data only if the algorithms' native costs differ by
   exactly that much — assumed, not verified, and not consistent across
   malloc/free.
5. **Vertical vs horizontal:** d (and c = d/N) is the robust,
   gauge-invariant, flat-direction-free quantity; the horizontal knee is the
   fade scale s0 — a flat, seed-dependent direction, out of range in the
   slow-fade arms, undefined in the anomalous arms. The "measure the plateau
   length" method reduces to d/N where it works and is not an independent
   measure. (And the preliminary "no visible plateau" observation was an
   artefact of mixing up the smallest delay (100 ns) with the largest
   (5–56 ms): the plateau/bend is in the data.)
6. **Absolute cost: no** — d/N is the slack H − c_a, bounded below by 0 in
   c_a, not the native cost. **Relative cost: yes, conditionally** —
   Δ(d/N) between algorithms of the same scenario is a well-defined,
   gauge-invariant, flat-robust relative measure that equals the relative
   native-cost difference if H is invariant; the knee method adds nothing.
   Significant results: KHI 256×128×128 malloc: Flat − Scat = +46 µs/call
   (2.7σ); KHI 256×128×256: Flat − Scat = −76 µs/call (malloc, 6.2σ),
   −49 µs/call (free, 3.1σ); FoilLCT: none.
7. **Algorithms:** under the hiding reading, FlatterScatter wins at KHI
   256×128×128 (malloc) and ScatterAlloc at KHI 256×128×256 (malloc and
   free); FoilLCT shows no difference; KHI 128³ is unrankable (no
   significant plateau, hump-dominated). Under the (II) reading the same
   numbers rank them the other way — the comparison is only as strong as the
   H-invariance assumption plus the interpretation choice, and the data
   cannot adjudicate between the two.

---

## 10. Reproduction

Scripts (in `/workspace`): `qa2_analysis.py` (per-arm N, d, c, E, O, shape
fits hyperbola/step/line, knee, first-pass table; also the B/C comparison
blocks), `qa2_table.py` (parametric-bootstrap errors with a per-cell noise
floor, final tables, scenario comparison). Both import the read-only helpers
`read_table`, `group_of`, `run_spread` from `critique_fit.py` and read
`output/results.h5`.

Core per-arm routine (from `qa2_table.py`):

```python
def arm_stats(x, y, t00, t_base, t_arm, s_arm, spread_floor, sigma00):
    N = (y[-1] - y[-2]) / (x[-1] - x[-2])          # large-delay slope = calls/run
    b0 = y[-1] - N * x[-1]                          # large-delay line intercept
    d = t00 - b0                                     # plateau deficit (data only)
    c = d / N
    E = y - t00 - N * x                              # excess over nominal line
    O = y - b0 - N * x                               # offset over large-delay line
    ...
    def hyp(s, a, s0): return a * s0 / (s + s0)      # (H) hyperbola
    popt, pcov = curve_fit(hyp, x, O, p0=[max(d, 1e-3), 1e-4],
                           bounds=([0, 1e-9], [np.inf, np.inf]), maxfev=40000)
    out["ssr_0pfam"] = np.sum((E + d * x / (x + d / N)) ** 2)   # knee = d/N family
    # (L) literal sharp knee: O = d (s < s*), 0 (s >= s*), s* on a log grid
    # parametric bootstrap for d, c: each cell ~ N(median, sigma_cell) with
    # sigma_cell = within-cell spread (n>=3) else the group's run spread floor
```

Key outputs are pasted in the tables above (per-arm table §2; scenario
comparison §5; shape-SSR comparison §4). `critique_fit.py` (pre-existing)
provides blocks 4/5/6: the same anchored E(s)/d view, the stored-fit-vs-data
pinned comparison, and the gauge-symmetry diagnostics quoted here.
