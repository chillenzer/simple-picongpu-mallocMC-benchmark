<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Q1 — Can a re-parameterization with one fewer parameter remove the flat direction of the fit?

**Data:** `output/results.h5`, 12 fit groups / 24 arms (3558 runs), all 7-parameter
two-operation fits. Model, headline fraction and context: `analysis-review.md`
(esp. §3 "f is not identified", §6 "gauge symmetry"), `README.md` lines 65–73.
**Scripts:** `/workspace/qa_flat_direction.py` (the full analysis, parts P1–P4+F)
and `/workspace/critique_fit.py` (reusable helpers). Output:
`/workspace/qa-flat-direction-output.txt`. Run with
`/tmp/opencode/venv/bin/python /workspace/qa_flat_direction.py`.

---

## 0. Bottom line (answers first)

1. **No — you cannot re-parameterize with one fewer parameter to remove the flat
   direction.** At the fixed experimental delays the model map is genuinely
   **7-dimensional** for the two-operation model and **4-dimensional** for the
   one-operation model (the parameter Jacobian has rank 7 / 4; the one exception,
   rosi KHI 256×128×128, has rank 6 because it sits on the `A_f→0` boundary, see
   §2). The per-arm delay-origin gauge has three (five) invariants — `N`,
   `A·s0`, `W−N·s0` per arm — but those span only a **3-D (5-D) subfamily**: the
   fade scale `s0` is an *additional, independent* degree of freedom of the
   function (holding all invariants fixed and moving `s0` by 0.1×–10× still moves
   the fitted curve by up to 140 s, §3). A one-fewer-parameter family is a proper
   *subset* of the function space, so it cannot reproduce the same curves. And the
   flat direction is a **data near-degeneracy** (weak information along the
   direction), not a parameter redundancy — no change of coordinates removes weak
   information (§4 proves the information spectrum is invariant under any
   reparameterization).

2. **The one true model-form redundancy — the gauge — is not a viable "one fewer
   parameter" fix.** It is (a) already *pinned by the experiment* (the delay
   origin `s=0` is the measured no-injection baseline), (b) **not** the fit's flat
   direction (its share of the two flattest covariance directions is only
   0.008–0.111, §2), and (c) removing it costs **two** parameters (7→5 invariants),
   not one, and does not reduce the number of independent functions. It therefore
   does not address the ill-conditioning you are actually seeing.

3. **What the flat directions actually are** (local, from the stored fit's pcov):
   the **N slope-pivots** (1σ = 58–1170 s) and the **`(W, A_m, A_f)` split**
   (1σ = 1.3–4 s); information ≪ 1 σ². The **fade-scale axes `m0`, `f0` are
   locally the *stiffest*** (smallest pcov eigenvalues, information 10⁶–10¹⁰).
   There are two distinct senses of "flat" and they must not be conflated:
   - *local* flatness = curvature of the χ² at the stored minimum (the pcov);
     here the (W,A,N) split and the N slopes are flat, the fade scales are stiff.
   - *global* identifiability = the seed/valley/boundary behaviour. Here the fade
     scale is the weakly-identified quantity: it is seed-dependent and multi-modal,
     it sits at the 50 ms search cap in the slow-fade rosi KHI arms, and it is
     *exactly* degenerate in the `(A_f, f0)` product at the `A_f→0` boundary
     (rank 7→6). `analysis-review.md` §6's "the flat direction is the fade scale
     s0" is this *global* statement.
   - A note on the evidence: the critique's block-6 "gauge share in the two
     flattest directions" was computed on `eigh`'s first two eigenvectors, which
     are the two **stiffest** (numpy sorts eigenvalues ascending), while its
     docstring/§6 call them "flattest". Recomputed on the actual flattest, the
     gauge share is 0.008–0.111 (small but nonzero) — the practical conclusion
     (the gauge is not the dominant flat direction) is unchanged, but the "0.00"
     in the review was measuring the wrong axes.

4. **What actually helps** (these add external information or change what you
   report — they do not change the coordinate system):
   - **(A) Fix the fade scales externally** (dense sweep in the bend, or a
     pipeline/theory prior) → the fit becomes an *exact linear* 5-parameter
     problem, condition number drops to 3·10⁴–5·10⁶. Cost: `f` then depends on
     the externally-chosen `s0` (e.g. hal KHI 128³ malloc: 4.6% at `s0·0.1` →
     9.7% at stored `s0` → 53% at `s0·10`).
   - **(B) Impose `A = N·c_a`** with `c_a` from an **independent** zero-delay
     microbenchmark (the model already supports `model_c_a`) → removes the
     `(A, s0)` degeneracy and is the *only* route giving `f` a physical
     "allocation cost" meaning. Using the fitted `A` for `c_a` is circular.
   - **(C) Combined fit across algorithms** (shared `W`,`N`) → does **not**
     remove the flat direction (smallest eigenvalue ~ 0 survives pooling, §5-C).
   - **(D) Profile-likelihood** along the flat direction → **quantifies** (not
     eliminates) it; the 1σ window is set by the data and is orders of magnitude
     wider than the Gaussian pcov error when the direction is under-resolved
     (rosi KHI 256×128×128: f_malloc 0–6.65% all within 1σ², §5-D).
   - **(E) Multi-start seeds / global optimization** → removes the *local-minimum
     risk* (the stored rosi KHI 256×128×128 fit sits in a minimum 92 σ² worse than
     the best seed) but the global flat valley remains.
   - **(F) Report the identifiable, gauge-invariant, data-pinned combinations**
     — `N`, `T0`, the plateau deficits `d_m`,`d_f` (and per-call `c = d/N`) —
     instead of the `(W, A, s0)` split; give `f` a profile-likelihood CI (D) or a
     seed-range (E), not the Gaussian 1σ.

5. **If you re-fit for numerical stability:** use
   `(T0, N_m, N_f, A_m, K_m=A_m·m0, A_f, K_f=A_f·f0)` — `K=A·s0` is what the
   large-delay data actually see — condition number drops 2–3 orders
   (7.6·10⁸–2.2·10²⁰), **plus** multi-start seeds. This improves conditioning and
   local-minimum risk but does **not** remove the data flat direction.

6. **Bottom line:** no reparameterization (one fewer parameter or otherwise)
   eliminates the flat direction. The robust response is to (i) report the
   data-pinned, gauge-invariant quantities, (ii) fix or bound the fade scale with
   external information, and (iii) for a genuine allocation-cost budget, measure
   `c_a` independently and impose `A = N·c_a`.

---

## 1. The model, the fraction, and two different "degeneracies"

Two-operation model (per arm `m` = malloc calls, `f` = free calls, per-run times
in seconds):

```
T(m, f) = W + N_m·m + N_f·f + A_m·m0/(m+m0) + A_f·f0/(f+f0)
```

Parameters: `W` (no-allocation baseline), `N_m`,`N_f` (calls per run, from the
large-delay slope), `A_m`,`A_f` (plateau deficit / native cost), `m0`,`f0` (fade
scales). One operation drops the corresponding arm. Headline:
`T0 = W + A_m + A_f` (zero-delay runtime) and `f_m = A_m/T0`, `f_f = A_f/T0`.

There are **two distinct degeneracies** that the literature and this question
conflate, and they have different cures:

- **Gauge (a redundancy of the model *form*).** The hyperbolic term is invariant
  under a per-arm shift of the delay origin: `W→W+N·s0·d`, `A→A/(1+d)`,
  `s0→s0(1+d)`, `s→s−s0·d` (per arm), with invariants `N`, `A·s0`, `W−N·s0`. This
  is a horizontal translation of the curve described from a different origin. It
  is **broken by the experiment**: `s=0` is the measured baseline, so the origin
  is fixed. At fixed delays it is *not* a fit redundancy (it moves the curve by
  `O(N·s0·d)`, §3).

- **Near-degeneracy (a limitation of the *data*).** Some directions of the
  parameter space change the fitted curve by less than the data's noise floor, so
  the data cannot pin them. This is the ill-conditioning (cond(pcov) 10¹¹–10¹⁴)
  and it is what "the flat direction" means here.

**Local vs global flatness.** "Flat" is used two ways. *Locally*, the pcov at the
stored minimum measures χ² curvature: a direction `v` (unit, in parameter space)
has information `info(v) = vᵀ Jᵀ J v / σ²` (in σ² units; `info ~ 1` = 1σ,
`info ≪ 1` = the data do not see the direction, `info ~ 100` = 10σ). The largest
pcov eigenvalues = smallest `info` = the locally flat directions. *Globally*, a
parameter can be locally stiff at one minimum yet weakly identified across the
landscape (multi-modal, or pinned at a bound). Both matter and are both reported.

---

## 2. What the flat directions actually are (per stored fit, local pcov)

`cond(pcov)` and the eigen-decomposition of each stored 7-parameter pcov. The
flattest directions (largest eigenvalue / smallest information) are the **N
slope-pivots** and the **`(W, A_m, A_f)` split**; the stiffest (smallest
eigenvalue) are the **pure `m0`, `f0`** fade scales. Full eigenpairs for all 12
groups are in the output file (P1); the summary:

| group | n | cond(pcov) | flattest (dir, 1σ, info) | 2nd flattest (1σ, info) | (W,A) split (1σ, info) | stiffest `m0`,`f0` (info) | gauge share in 2 flattest |
|---|--:|--:|---|--:|---|--:|--:|
| hal Foil Flat 256×1280×0 | 137 | 3.7e14 | +N_m (250 s, 1.6e-5) | −N_f (71 s, 2.0e-4) | 1.7 s, 0.33 | 5.9e9, 1.8e9 | 0.008 / 0.013 |
| hal Foil Scat 256×1280×0 | 19 | 2.4e13 | +N_f (45 s, 5.0e-4) | −N_m (43 s, 5.4e-4) | 0.65 s, 2.4 | 1.2e10, 4.7e9 | 0.011 / 0.012 |
| hal Kelv Flat 128³ | 80 | 1.8e13 | +N_m (426 s, 5.5e-6) | −N_f (64 s, 2.4e-4) | 2.3 s, 0.20 | 1.0e8, 3.9e6 | 0.030 / 0.032 |
| hal Kelv Flat 256×128×128 | 82 | 1.9e14 | +N_f (832 s, 1.4e-6) | −N_m (388 s, 6.7e-6) | 3.1 s, 0.11 | 2.7e8, 4.1e7 | 0.018 / 0.019 |
| hal Kelv Flat 256×128×256 | 81 | 7.9e14 | +N_f (1170 s, 7.3e-7) | −N_m (444 s, 5.1e-6) | 4.0 s, 0.061 | 5.8e8, 8.5e7 | 0.019 / 0.020 |
| hal Kelv Scat 128³ | 22 | 2.2e11 | +N_f (154 s, 4.2e-5) | −N_m (72 s, 1.9e-4) | 3.1 s, 0.11 | 9.3e6, 3.5e6 | 0.038 / 0.036 |
| hal Kelv Scat 256×128×128 | 22 | 1.0e12 | +N_f (268 s, 1.4e-5) | −N_m (112 s, 8.0e-5) | 4.7 s, 0.045 | 1.4e7, 7.4e6 | 0.047 / 0.045 |
| hal Kelv Scat 256×128×256 | 21 | 4.5e12 | +N_f (157 s, 4.1e-5) | −N_m (136 s, 5.4e-5) | 2.9 s, 0.12 | 1.8e8, 9.8e7 | 0.037 / 0.037 |
| rosi Foil Flat 256×1280×0 | 175 | 1.4e14 | +N_m (150 s, 4.4e-5) | −N_f (27 s, 1.4e-3) | 1.3 s, 0.62 | 6.2e9, 6.0e8 | 0.015 / 0.032 |
| rosi Kelv Flat 128³ | 194 | 2.6e12 | −0.97N_m−0.23N_f (65 s, 2.4e-4) | −0.23N_m+0.97N_f (64 s, 2.4e-4) | 3.1 s, 0.10 | 6.0e8, 3.8e6 | 0.029 / 0.044 |
| rosi Kelv Flat 256×128×128 | 179 | **inf** | −0.11W+0.99N_f+0.11A_f (93 s, 1.2e-4) | −N_m (49 s, 4.1e-4) | 4.6 s, 0.047 | 2.1e7, **0** | 0.111 / 0.111 |
| rosi Kelv Flat 256×128×256 | 178 | 1.9e13 | +N_m (152 s, 4.3e-5) | −N_f (58 s, 3.0e-4) | 2.9 s, 0.12 | 1.2e8, 8.1e8 | 0.016 / 0.019 |

Reading:
- **Flattest** = the N slope-pivots (info 10⁻⁷–10⁻³) and the `(W, A_m, A_f)`
  split (info 0.05–0.6). These are the directions the data barely see.
- **Stiffest** = the pure fade-scale axes `m0`, `f0` (info 10⁶–10¹⁰). Locally the
  data pin the bend location tightly — the fade scales are *not* the locally flat
  direction.
- **Gauge share** in the two flattest is 0.008–0.111 (small), confirming the
  gauge is not the dominant flat direction. The single large value (0.111, rosi
   KHI 256×128×128) is a corner artifact: that fit sits on the `A_f→0` boundary
   (stored `A_free ≈ 8×10⁻⁹ s ≈ 0`, `f0` at the 50 ms cap), so the `f0` gauge
   tangent aligns with the exactly-flat `f0` axis there.

**The one genuinely degenerate case.** rosi KHI 256×128×128 has `cond(pcov) = ∞`
and rank **6** (not 7): the 7th eigenvalue is 0 and its eigenvector is pure `f0`
(info 8.3e-16). The stored fit sits on the `A_f→0` boundary (`A_free ≈ 8×10⁻⁹ s`,
i.e. 0) *and* `f0` at the 50 ms search cap. This is the exact `(A_f, f0)`
**product** degeneracy: the data see only `A_f·f0` (≈ 0 here), so with `A_f`
floored at 0 the product is 0 for *any* `f0` and `f0` is exactly undetermined.
This is the only group where a fade-scale axis is locally (not just globally)
flat.

---

## 3. Can you drop one parameter? — the function-space dimension

This is the crux, and it is decided by how many *independent functions* the model
spans at the fixed delays, not by how the parameters are named.

**(a) Local rank of the parameter Jacobian** (per group, at the stored params):
rank = **7** in all groups except rosi KHI 256×128×128 where it is **6** (the
`A_f→0` corner). For the one-operation model the 4-column Jacobian
`[1, s, s0/(s+s0), A·s/(s+s0)²]` has rank **4** at the measured delays. So the
model map is 7-D (2-op) / 4-D (1-op) locally — every parameter moves the curve.

**(b) The gauge is not in the kernel.** The three per-arm gauge invariants are
`N`, `K = A·s0`, `B = W − N·s0`. Verified: under a gauge shift `d = 0.3` all three
are unchanged (to ≤ 10⁻⁸). But moving along the gauge *at the fixed measured
delays* changes the function: `max |T − T_gauge| = 7.8 s` over `d ∈ [−0.5, 0.5]`
(= `N·s0·|d|` — the gauge shifts the curve horizontally by `s0·d`, i.e.
vertically by `N·s0·d` at fixed `s`). So the gauge is a redundancy of the model
*form* (it re-describes the same curve from a different delay origin), **not** of
the fit: with the delays fixed, each gauge value is a *different* function. It
cannot be reparameterized away.

**(c) The invariants span only a subfamily; `s0` is an extra function dof.**
Holding the three invariants `(N, K=A·s0, B=W−N·s0)` fixed and varying `s0` over
`[0.1×, 10×]` of its stored value still moves the one-arm curve by **up to 140 s**.
So the 3 invariants span a 3-D subfamily and `s0` is a genuine **4th** function
degree of freedom — a 3-parameter (gauge-invariant-only) model cannot span the
4-D family. (The 2-op analogue: 5 invariants span 5-D, and the two fade scales
add 2 more function dof → 7-D.)

**(d) The data do see the dropped parameter (weakly).** Fixing `s0` externally
and fitting the remaining 3 parameters (linear), the loss relative to the 4-param
one-arm optimum is:

```
s0 ×0.1 : dSSR = 639 s² = 2836 σ²   (A = 14.2 s, f = 15.7%)
s0 ×1   : dSSR =   1.8 s² =    8 σ²   (A = 15.2 s, f = 17.5%)
s0 ×10  : dSSR = 531 s² = 2358 σ²   (A = 22.6 s, f = 27.4%)
```

The data *do* prefer `s0 ≈ stored` (the ×1 point is the minimum) — `s0` is not a
gauge, it is a real, weakly-pinned parameter. That is exactly the
near-degeneracy: weak information, but nonzero. **You cannot remove it by
relabeling; you can only remove it by adding information (fix `s0`, or constrain
`A`).**

**Verdict: No.** At fixed delays the model spans 7 functions (2-op) / 4 functions
(1-op); a one-fewer-parameter family spans a proper subset. The flat direction is
weak data information, invariant under any reparameterization (§4), and the one
true form-redundancy (the gauge) is pinned by the fixed origin and is not the fit's
flat direction.

---

## 4. Do reparameterizations fix the conditioning?

Tested parameterizations (same 7 functions, different coordinates):

- `orig`: `(W, N_m, N_f, A_m, A_f, m0, f0)`
- `R_T0`: `(T0, N_m, N_f, A_m, A_f, m0, f0)`, `T0 = W + A_m + A_f`
- `R_K`:  `(T0, N_m, N_f, A_m, K_m, A_f, K_f)`, `K_m = A_m·m0`, `K_f = A_f·f0`
- `R_eig`: `Vᵀ p`, `V` the eigenvectors of `JᵀJ` (the information principal axes)

| group | cond orig | cond R_T0 | cond R_K | cond R_eig | cond(JᵀJ) |
|---|--:|--:|--:|--:|--:|
| hal Foil Flat 256×1280×0 | 3.7e14 | 3.7e14 | 3.2e12 | 3.7e14 | 3.7e14 |
| hal Foil Scat 256×1280×0 | 2.4e13 | 2.4e13 | 1.4e11 | 2.4e13 | 2.4e13 |
| hal Kelv Flat 128³ | 1.8e13 | 1.8e13 | 7.4e10 | 1.8e13 | 1.8e13 |
| hal Kelv Flat 256×128×128 | 1.9e14 | 1.9e14 | 1.1e11 | 1.9e14 | 1.9e14 |
| hal Kelv Flat 256×128×256 | 7.9e14 | 7.9e14 | 1.7e11 | 7.9e14 | 7.9e14 |
| hal Kelv Scat 128³ | 2.2e11 | 2.2e11 | 1.4e09 | 2.2e11 | 2.2e11 |
| hal Kelv Scat 256×128×128 | 1.0e12 | 1.0e12 | 7.6e08 | 1.0e12 | 1.0e12 |
| hal Kelv Scat 256×128×256 | 4.5e12 | 4.5e12 | 1.0e09 | 4.5e12 | 4.5e12 |
| rosi Foil Flat 256×1280×0 | 1.4e14 | 1.4e14 | 2.2e11 | 1.4e14 | 1.4e14 |
| rosi Kelv Flat 128³ | 2.6e12 | 2.6e12 | 2.1e10 | 2.6e12 | 2.6e12 |
| rosi Kelv Flat 256×128×128 | inf | inf | 2.2e20 | 8.1e21 | 3.3e24 |
| rosi Kelv Flat 256×128×256 | 1.9e13 | 1.9e13 | 4.8e11 | 1.9e13 | 1.9e13 |

- `R_T0` is a linear shear of the flat axes → condition number unchanged.
- `R_K` is non-orthogonal and aligns parameters with the data-sensitive
  combination `K = A·s0` (what the large-delay data actually see) → condition
  number drops **2–3 orders**. This is the best *numerical* reparameterization.
- `R_eig` diagonalizes the information (`pcov_R_eig = diag(σ²/λ_i)`); it is an
  orthogonal rotation, so `cond(R_eig) = cond(orig)` **always**.
- **The invariant fact:** the information spectrum `λ_i` (in σ² units) is
  identical in every parameterization — only the axes rotate. `R_K` can lower the
  *numeric* condition number by aligning axes with the data, but it cannot change
  any `λ_i`. The sub-σ² flat directions (the N pivots, the W/A split) survive
  every reparameterization; in `R_eig` they simply *become* the explicit
  coordinate axes. **No reparameterization removes the flat direction.** (The
  `R_K` "inf" for rosi KHI 256×128×128 is the `A=0` corner, not a failure of the
  reparameterization.)

`R_K` flat directions (flattest 3, e.g. hal Foil Flat 256×1280×0): `+N_m`
(250 s), `+N_f` (71 s), `−0.47·T0 − 0.41·A_m − 0.78·A_f` (1.4 s) — i.e. the same
N-pivot + W/A-split flatness, now in cleaner coordinates.

---

## 5. What actually helps (options)

### A. Fix the fade scales externally (→ exact linear 5-parameter fit)
Freeze `m0`,`f0` at externally-chosen values; the model is linear in
`(W, N_m, N_f, A_m, A_f)` → one `lstsq`, no local minima, well-conditioned.

| group | cond(XᵀX) | 1σ(W) s | 1σ(A_m) s | 1σ(A_f) s | f_m % | f_f % | f_m @ s0×0.1 | f_m @ s0×10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| hal Foil Flat 256×1280×0 | 8.9e5 | 3.19 | 2.34 | 2.64 | 13.96 | 7.13 | 9.81 | 21.23 |
| hal Foil Scat 256×1280×0 | 1.8e5 | 0.41 | 0.31 | 0.32 | 15.37 | 15.95 | 14.16 | 17.30 |
| hal Kelv Flat 128³ | 1.9e6 | 3.02 | 2.42 | 2.09 | 9.72 | 5.50 | 4.59 | 53.42 |
| hal Kelv Flat 256×128×128 | 4.1e6 | 5.26 | 2.61 | 4.50 | 13.20 | 12.48 | 5.57 | 95.38 |
| hal Kelv Flat 256×128×256 | 5.0e6 | 7.49 | 3.22 | 6.64 | 11.29 | 13.91 | 4.97 | 81.23 |
| hal Kelv Scat 128³ | 7.6e4 | 4.34 | 2.99 | 3.45 | 7.77 | 5.38 | 6.52 | 12.45 |
| hal Kelv Scat 256×128×128 | 1.0e5 | 8.22 | 5.16 | 6.57 | 12.06 | 10.93 | 9.12 | 26.45 |
| hal Kelv Scat 256×128×256 | 9.9e4 | 3.90 | 2.61 | 2.97 | 11.06 | 11.42 | 8.27 | 28.59 |
| rosi Foil Flat 256×1280×0 | 4.3e5 | 2.14 | 1.74 | 1.63 | 28.45 | 20.92 | 22.00 | 51.22 |
| rosi Kelv Flat 128³ | 2.9e4 | 6.43 | 3.85 | 6.05 | 9.69 | 13.90 | 9.90 | 11.69 |
| rosi Kelv Flat 256×128×128 | 1.6e5 | 21.87 | 2.39 | 21.80 | 4.44 | −19.80 | 3.14 | 9.25 |
| rosi Kelv Flat 256×128×256 | 8.0e4 | 6.99 | 6.23 | 5.62 | 3.50 | 1.56 | 2.97 | 4.48 |

**What it does:** removes the `(A, s0)` nonlinearity and the associated flatness;
`cond` drops from 10¹¹–10¹⁴ to 10⁴–10⁶. **Cost:** `f` now *is* the external
`s0` choice — e.g. hal KHI 128³ malloc `f_m` runs 4.6% → 9.7% → 53% for
`s0` at ×0.1 / stored / ×10. You are trading a fitting artifact for a
modeling assumption, so the assumption must come from data (a sweep dense in the
bend) or a defensible prior. **Feasible, recommended** as the honest way to quote
`f` once `s0` is known; report `f` *conditional on the chosen `s0`* and state the
×0.1/×10 sensitivity.

### B. Impose `A = N·c_a` (native cost from an independent microbenchmark)
`T = W + N_m·m + N_f·f + N_m·c_m·m0/(m+m0) + N_f·c_f·f0/(f+f0)`: 5 free
parameters `(W, N_m, N_f, m0, f0)`. Here `c_m`,`c_f` are the per-call native
costs. **What it does:** removes the `(A, s0)` product degeneracy entirely (the
fitted `A` is replaced by `N·c_a`), which is what makes `f = A/T0` a *measured*
allocation-cost fraction rather than an absorbed-delay slack. **Cost:** needs an
independent zero-delay microbenchmark of `DeviceAllocator::malloc`/`free`
(`c_a` from the fitted `A/N` is circular — it is the review §5 finding). With the
circular stand-in (`c_a` = stored `A/N`) the residual flattest-information is the
N-pivot (10⁻⁵–10⁻⁴, a unit artifact, N known to 0.1–0.6%) and `dSSR ≈ 0`.
**Feasible, and the only route that gives `f` its intended meaning** (the model
already supports `model_c_a`).

### C. Combined fit across algorithms (shared `W`, `N`)
The stored 11×11 pcovs. **Does not help:** the smallest eigenvalue is ~0
(1.5e-10–3e-8) with eigenvector = the per-scenario `m0`, i.e. a flat direction
*survives* pooling across algorithms. Pooling shares `W`,`N` but not the
per-algorithm fade scales, which are exactly the weakly-identified ones.

### D. Profile-likelihood along the flat direction (quantify, don't eliminate)
Fix one parameter on a grid, refit the other 6, read off the ΔSSR profile. This
turns the "flat" direction into an explicit confidence interval set by the data,
and shows the Gaussian pcov error is too small when the direction is
under-resolved. Concrete wide-valley example (rosi KHI 256×128×128 malloc, base =
re-optimized fit, threshold = run-to-run spread): the ΔSSR profile along `A_m` is
flat within 1σ² over `A_m ∈ [0, 13.8]` s, i.e. **`f_m ∈ [0, 6.65]%` all within
1σ²** (stored 4.75%). The stored fit's Gaussian `f_m` error (≈0.7%) is an order of
magnitude too small. (For a stiff group, e.g. hal KHI 256×128×128, the same
profile along `A_m`/`m0` is narrow — the window is data-set in each case.)
**Feasible, recommended** as the error bar to report for `f` when using the fit as-is.

### E. Multi-start seeds / global optimization (remove the local-minimum risk)
The full 7-param fit from different `(m0,f0)` seeds lands in different local
minima with comparable SSR but very different `f` — the global flat valley.
- hal KHI 128³: grid/mid seed `f_m` 9.72% (best SSR) vs bottom seed 0.93%
  (ΔSSR ≈ 107 σ²).
- rosi KHI 256×128×128: **grid seed `f_m` 6.52% is the *worst*** (ΔSSR ≈ 92 σ²
  above the bottom seed 2.39%); mid 4.24%. The *stored* fit is one of these local
  minima — its "significance" over the line is which minimum the solver found.
**Feasible, recommended** as a robustness step (report the seed spread as the
`f` range), but it does not remove the underlying flatness — it shows its scale.

### F. Report the identifiable, gauge-invariant, data-pinned combinations
The data pin, to a few %, the combinations that the model and the measurement
share:

| group | 1σ(T0)/T0 | 1σ(f_m) | 1σ(f_f) | ( = stored f errors ) |
|---|--:|--:|--:|---|
| hal Foil Flat 256×1280×0 | 0.98% | 1.42% | 1.64% | yes |
| hal Foil Scat 256×1280×0 | 0.35% | 0.54% | 0.57% | yes |
| hal Kelv Flat 128³ | 0.29% | 2.42% | 1.37% | yes |
| hal Kelv Flat 256×128×128 | 0.13% | 1.39% | 3.74% | yes |
| hal Kelv Flat 256×128×256 | 0.08% | 0.81% | 2.95% | yes |
| hal Kelv Scat 128³ | 0.72% | 1.64% | 2.25% | yes |
| hal Kelv Scat 256×128×128 | 0.47% | 1.50% | 2.48% | yes |
| hal Kelv Scat 256×128×256 | 0.14% | 0.57% | 0.73% | yes |
| rosi Foil Flat 256×1280×0 | 0.70% | 1.18% | 1.17% | yes |
| rosi Kelv Flat 128³ | 1.06% | 1.22% | 2.82% | yes |
| rosi Kelv Flat 256×128×128 | 0.30% | 0.70% | 5.20% | yes |
| rosi Kelv Flat 256×128×256 | 0.40% | 0.54% | 0.52% | yes |

`T0 = W+A_m+A_f` (the measured (0,0) baseline) is pinned to 0.08–1.06%; the
plateau deficits `d_m`,`d_f` (and per-call `c = d/N`) are gauge-invariant and
flat-direction-free. **Recommended:** report `N`, `T0`, `d_m`,`d_f`,`c` as the
data's actual measurements; treat `f = A/T0` as a convention-dependent slack
fraction (option D's profile CI or E's seed range), not a Gaussian 1σ runtime
budget.

---

## 6. Bottom-line recommendation (priority order)

1. **Stop reporting the `(W, A, s0)` split and `f = A/T0` as a runtime budget.**
   Report the data-pinned, gauge-invariant quantities: `N` (large-delay slope),
   `T0` ((0,0) baseline), the plateau deficits `d_m`,`d_f` and per-call
   `c = d/N`. These are well determined and flat-direction-free.
2. **If `f` must be reported, give it a real error bar:** the profile-likelihood
   CI (D) or the multi-start seed range (E) — not the Gaussian pcov 1σ (which is
   an order of magnitude too small along the flat direction).
3. **Fix or bound the fade scale `s0` with external information** — a sweep dense
   in the bend, or a pipeline/theory prior — then the fit is a well-conditioned
   linear problem (A); report `f` *conditional on the chosen `s0`* with the
   ×0.1/×10 sensitivity.
4. **For an allocation-cost budget specifically, measure `c_a` independently**
   (zero-delay `DeviceAllocator` microbenchmark) and impose `A = N·c_a` (B). This
   is the only route that makes `f` a measured cost fraction rather than an
   absorbed-delay slack. Do not use the fitted `A` for `c_a` (circular).
5. **If you re-fit for numerical stability,** use the `(T0, N_m, N_f, A_m,
   K_m=A_m·m0, A_f, K_f=A_f·f0)` parameterization (2–3 orders better condition
   number) **and** multi-start seeds. This helps conditioning and local-minimum
   risk; it does **not** remove the data flat direction.
6. **Do not claim a reparameterization removes the flat direction.** It cannot:
   the flat directions are a data near-degeneracy (invariant information
   spectrum), and the one true form-redundancy (the gauge) is pinned by the fixed
   delay origin and is not the fit's flat direction.

---

## 7. Reproduction

```
/tmp/opencode/venv/bin/python /workspace/qa_flat_direction.py
```

prints P1 (per-fit pcov eigen-decomposition + gauge share), P2 (function-space
rank, gauge-invariant check, gauge orbit at fixed delays, dSSR of dropping `s0`),
P3 (reparameterization condition numbers + R_K flat directions), P4 (options
A–F). Captured output: `/workspace/qa-flat-direction-output.txt`.
Helpers imported from `/workspace/critique_fit.py` (`read_table`, `model_full`,
`group_of`, `arm_anchored`, `bounds_of`, `fit_full`, `grid_guess`, `gauge_block`,
`profile_A`, `seed_sensitivity`, `run_spread`). Data: `/workspace/output/results.h5`
(`runs` 3558 rows, `fits` 12 rows, per-group pcov under `fits/cov/...`, shared
11×11 pcovs under `shared_fit_cov/...`).
