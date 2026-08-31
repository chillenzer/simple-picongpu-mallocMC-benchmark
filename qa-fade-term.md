<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Q3 — Is the hyperbolic fade the right saturation shape, or do the data prefer another?

**Data:** `output/results.h5`, 12 fit groups / 24 arms (3558 runs). Context:
`analysis-review.md` (the model, the gauge, what the data pin), `qa-flat-direction.md`
(Q1: the flat direction), `qa-allocation-cost.md` (Q2: the plateau and the absorbed
delay). Model: `T(m,f) = W + N_m·m + N_f·f + F_m(m) + F_f(f)` with the stored fade
`F_H = A·s/(m+s)`.
**Script:** `/workspace/qa_fade_term.py` (fits five alternative fades to the same 12
groups, parts P1–P6). Output: `/workspace/qa-fade-term-output.txt`. Run with
`/tmp/opencode/venv/bin/python /workspace/qa_fade_term.py` (deterministic, ~3 s).

---

## 0. Bottom line (answers first)

1. **The hyperbola is not the best fade shape, but the data do not pick one clear
   winner.** Re-fitting every candidate against a *fresh* hyperbola refit (so the
   comparison is shape-only, not the stored fit's drift along the flat direction),
   the mean ΔSSR over the 12 groups is: **E −6.71, P −5.10, H 0.00, T +0.38,
   Q +2.98, L +7.67** (in σ²; negative = better). The exponential **E** is the best
   of the two-parameter forms and **never the worst in any group**; the Lorentzian
   **L** is the worst on average (worst in 5 of 12 groups). The per-group best is
   scattered (P 7, L 2, E 1, T 1, Q 1, H 0), i.e. no shape wins everywhere.
2. **The data reject the zero-initial-slope claim (L) and the hard cutoff (T) as the
   *average* shape, but they do not reject T strongly** (T is within +0.4 σ² of H on
   average). The physical picture the data prefer is "absorption starts linearly and
   saturates without a sharp knee" — the exponential does exactly that with one scale.
3. **The extra power-law exponent (P) is not identified.** P wins the most per-group
   contests (7/12), but its mean is worse than E's and its exponent `k` runs into the
   0.2–10 search bound in most groups, with 1-σ up to 3512 % on `(s, k)`. Those wins
   are the extra parameter, not a shape the data resolve. A 2-parameter form is
   sufficient.
4. **Recommendation: switch the fade to the exponential `F_E = A·exp(−m/s)`** for the
   next fit round. It is the best mean ΔSSR of the parsimonious forms, is smooth, has
   a *finite* total absorbed cost (`∫F dm = A·s`, vs the hyperbola's divergent `A·s/m`
   tail), and — unlike the hyperbola — its scale `s` is a **gauge invariant** (P5), so
   the fitted `s` is a claim about the data, not a coordinate choice. Keep the
   truncated-linear **T** as the physical null model (hard cutoff at the per-call
   slack): a future run that shows the fade go *exactly* to zero by delay `s` would
   promote T; a residual `~A·e^{−m/s}` tail confirms E.
5. **Do not re-fit on the stored parameters.** The stored hyperbola fit sits
   **+626 σ²** (median, over the 12 groups) above the *fresh* hyperbola optimum — the
   stored fit drifted along the flat W/N/A direction (Q1) and is not the best
   hyperbola fit, let alone the best shape. Any shape comparison that uses the stored
   fit as the reference (the raw P2 table, all ΔSSR ≈ −50…−150 σ²) is measuring that
   drift, not the shape.

---

## 1. The candidate family (P1)

All candidates share the stored form's endpoints — `F(0) = A`, `F(∞) = 0`, monotone —
so the contest is purely about *how fast the visible fade vanishes with the delay*
`u = m/s`:

| form | `F'(0)/A` | `F/A` at u=1, 3, 10 | tail `m≫s` | `∫F dm/(A·s)` |
|---|---|---|---|---|
| H `A s/(m+s)` | −1/s | 0.50, 0.25, 0.091 | `A s/m` (never vanishes) | ∞ |
| E `A e^{−m/s}` | −1/s | 0.37, 0.050, 4.5e−5 | `A e^{−m/s}` | 1 |
| L `A s²/(m²+s²)` | **0** | 0.50, 0.10, 0.0099 | `A s²/m²` | π/2 |
| T `A(1−m/s)₊` | −1/s | 0, 0, 0 | 0 (cutoff at `m=s`) | 1/2 |
| Q `A(1−m/s)²₊` | **−2/s** | 0, 0, 0 | 0 (cutoff at `m=s`) | 2/3 |
| P `A(1+m/s)^{−k}` | −1/s | k-dependent | `A (s/m)^k` | `1/(k−1)`, k>1 |

T and Q are the *overlap picture* made literal (analysis-review §1: the injected
delay hides parallel work until the per-call slack is exhausted) — a hard cutoff at
the slack duration. E and H are the smooth saturations in between. P generalizes H
(H is P at k=1).

## 2. What the data can actually see

The sweeps probe `u = m/s` from ~10⁻⁴ up to `u_max` = 3×10³ (P2), and the
shape-discriminating window `u ∈ [0.3, 3]` is *populated*: 19–44 points on the malloc
arm and 4–11 on the free arm in the focus groups (of 82–179 runs). Two consequences:

- The **far tail** (`u > 3`) is probed at up to 3×10³, so a claim like H's "a visible
  `A s/m` forever" is falsifiable in principle — at `u=3000` H still claims 0.03 % of
  A, E claims `e^{−3000} = 0`.
- The **near shape** (`u ∈ [0.3, 3]`), which separates E from T/Q and H from L, has
  tens of points per group — enough for the ΔSSR differences of §3 to be real, but
  not enough to separate shapes that differ by < 1 σ² on average.

## 3. Shape contest (P2–P3, P6)

Reference: a **fresh** hyperbola refit per group (grid guess + bounded `curve_fit`,
identical pipeline for all candidates; this removes the stored fit's flat-direction
drift, §0.5). ΔSSR in σ²; negative = better than H:

| mean over 12 groups | H | E | L | T | Q | P |
|---|---|---|---|---|---|---|
| ΔSSR vs fresh H | 0.00 | **−6.71** | +7.67 | +0.38 | +2.98 | −5.10 |
| groups where best | 0 | 1 | 2 | 1 | 1 | 7 |
| groups where worst | 4 | **0** | 5 | 3 | 0 | 0 |

Reading:

- **L is rejected on average** (+7.67 σ², worst in 5 groups, including +63.75 in the
  256×128×256 KHI group). L's defining property is `F'(0) = 0` — the fade does not
  start moving until second order. The data see a *linear* initial absorption
  (`F'(0) = −A/s ≠ 0`), as H/E/T/Q all have.
- **E is the best parsimonious shape** (−6.71 σ²) and never the worst. It is within
  ~1–7 σ² of the per-group best everywhere.
- **P's 7 per-group wins are not a shape signal.** P is the only 3-parameter
  candidate; its `k` runs to the edge of the 0.2–10 search (10.00 in two focus
  groups, 0.35 near the lower bound in the third) and its `(s,k)` block has 1-σ of
  188–3512 % (P3). With the exponent unidentifiable, P is fitting noise in the extra
  dimension. Mean ΔSSR (−5.10) is also worse than E's (−6.71).
- **T ties H on average** (+0.38) — the physical hard cutoff is *not rejected* as the
  average shape, even though it is not the best either. That is the honest status of
  the overlap picture: consistent, but the data do not require the sharp knee.

The large numbers in the raw P2 table (ΔSSR ≈ −50…−150 σ² for *every* candidate) are
**not** shape effects: the stored hyperbola fit itself is +626 σ² (median) above the
fresh hyperbola optimum (max +54970 σ² in the degenerate rosi 256×128×128 group,
where the stored fit sits on the `A_f → 0` boundary with `s_f` pinned to the bound).
Subtracting the fresh-H reference is what turns −152 σ² into the shape-only
deltas above.

## 4. Identifiability and gauge structure (P3, P5)

Every two-parameter form carries the **same** delay-origin gauge
(stored delay = true delay + unknown offset `d`), but they pin *different*
combinations (7 parameters = 5 identifiable combos + 2 gauge freedoms per form):

| form | gauge freedoms | identifiable combinations |
|---|---|---|
| H | 2 | `N`, `A s`, `W − N_m s_m − N_f s_f` — pins the product `A s`, **not A or s** |
| E | 2 | `N`, `s`, `W + N_m s_m ln A_m + N_f s_f ln A_f` — **pins the scale s** |
| L | 0 | `W, N, A, s` — the form is rigid, all pinned |
| T | 2 | `N`, `A/s`, `W − N_m s_m − N_f s_f` — pins the absorption *rate* `A/s` |
| Q | 2 | `N`, `A/s²`, `W − N_m s_m − N_f s_f` — pins the initial *curvature* |
| P | 2 | `N`, `k`, `A s^k`, `W − N_m s_m − N_f s_f` — pins `k` and `A s^k` |

Measured: `cond(pcov)` is 1e12–1e15 for **every** form in every focus group (∞ in
the degenerate rosi group), so the gauge correlation is a property of the family,
not of one shape — no form escapes it, and none is disqualified by it. The
*individual* fade parameters stay well identified (1-σ 6–50 % in P3, the exceptions
being the inactive rosi free arm and P's unresolvable `(s,k)`). The practical payoff
of the table above: **for the hyperbola the fitted `s` is a gauge artifact** (only
`A s` is pinned, Q1), while **for the exponential the fitted `s` is a
gauge-invariant claim** — a concrete reason to prefer E on grounds of what the
numbers mean, independent of the ΔSSR.

## 5. What each form claims beyond the data (P4)

Focus group (hal KHI 256×128×128), visible fade as a fraction of `A_m` at
`u = m/s_m` beyond the measured range:

| form | F/A @ u=3 | @ u=10 | @ u=100 | total `∫F dm` |
|---|---|---|---|---|
| H | 0.250 | 9.1e−2 | 9.9e−3 | **divergent** (`A s/m` tail) |
| E | 0.050 | 4.5e−5 | ~0 | finite, `A s` |
| L | 0.100 | 9.9e−3 | 1.0e−4 | finite, `π A s/2` |
| T, Q | 0 | 0 | 0 | finite, `A s/2`, `2 A s/3` |

The hyperbola's only genuinely different *claim* is its divergent `1/m` tail: it
says the injected delay is **never** fully absorbed, no matter how long. The
microbenchmark cross-check (P4) anchors the scale: `c(malloc) = 104.2 µs` per call,
`N = 186362` calls/run, `N·c = 19.4 s = d` (the absorbed plateau deficit), and the
fitted `s_m = 1.26 ms ≈ 12 × c` — the fade scale is the *half-saturation delay*, an
arm property, not the per-call cost.

## 6. Recommendation

1. **Adopt `F = A·exp(−m/s)`** as the fade term in the next fit round (per arm, as
   before). Best mean ΔSSR of the parsimonious forms, never worst, smooth, finite
   total, gauge-invariant scale.
2. **Report the gauge-invariant quantities**, per Q1's rule: for E that is `N`, `s`,
   and `W + N_m s_m ln A_m + N_f s_f ln A_f` — in particular `s` itself (the
   half-saturation delay ≈ 12× the per-call cost) becomes a directly citable number.
3. **Keep T as the null model.** If a future sweep pushes delays well past `s` and
   the fade is exactly zero there, T (the literal overlap cutoff) is the better
   model; if a `~A e^{−m/s}` residual remains, E is confirmed. The current data
   cannot separate the two (T is +0.4 σ² from H on average).
4. **Refit, do not reuse, the stored parameters** (+626 σ² median drift, §0.5); carry
   the degenerate rosi 256×128×128 group (`A_f → 0` boundary) separately.

## 7. Reproduction

```
/tmp/opencode/venv/bin/python /workspace/qa_fade_term.py
```

Deterministic (no randomness; grid guess + bounded `curve_fit` from the best of a
log grid of 15×15 scale cells per arm, P adds a 4×4 exponent grid). Two runs produce
byte-identical output; the captured output is `qa-fade-term-output.txt`.
