<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# The KelvinHelmholtz small-delay amplification humps: a data-only diagnosis

**Subject:** the delay-sweep arms whose runtime at small imposed delay rises
*above* the (0, 0) baseline plus the nominal `N*s` stall sum — the "humps"
flagged in `analysis-review.md` (Tier 3) and `qa-allocation-cost.md` — plus
the arms whose large-delay line extrapolates above the baseline
("over-exposed", `d < 0`).
**Data:** `output/results.h5` (`runs` table; in this checkout the frozen
legacy table `legacy/legacy_results.h5`, since the sweep machines' output
directories are empty here).
**Reproduction:** `python3 analysis/diagnose_humps.py --results
output/results.h5` (model-free; prints sections A–F below; writes nothing).

## 1. What is measured

For every (machine, setup, algorithm, grid, arm) the script computes, from
the raw runtimes only (no fitted parameter):

- `N` — the calls per run, the slope through the two largest delays;
- `d` — the plateau deficit, `baseline − intercept` of that line
  (the gauge-invariant absorbed delay, `analysis-review.md` §8);
- `E(s) = median(s) − baseline − N*s` — the **excess** at each imposed
  delay: how far the measured runtime sits from "baseline plus the full
  stall sum". A pure absorption (hiding) model has `E(s) ≤ 0` by
  construction; `E(s) > 0` means the run is slower than the sum of its
  injected stalls — **amplification**.

Flags: `Hump` = `E` at the smallest delay (100 ns) > 0; `OverExp` = `d < 0`;
`NonMono` = `E` rises again beyond its noise as `s` grows.

## 2. What the data show

**Scope of the anomaly.** Of the 28 arms with enough delay points, 14 carry
the `Hump` flag and 4 the `OverExp` flag (two of them both). All twelve
humps of the three hal KHI grids (both algorithms, both operations) are
present; the hal KHI 128³ ScatterAlloc arms are the two `OverExp` arms
(`d = −4.2 s` malloc, `−8.3 s` free). FoilLCT is clean except one mild
exception (hal FoilLCT ScatterAlloc free, `E(100 ns) = +0.11 s` against a
152 s baseline); rosi is clean except rosi KHI 256×128×128 FlatterScatter
malloc (`E(100 ns) = +0.33 s`, and its eight runs at that delay split
4 positive / 4 negative — a noisy tail, not a robust hump) and the two
rosi over-exposed free arms (FoilLCT and KHI 128³).

**Shape.** The hump arms rise from 100 ns to a peak at 10–16 µs (or at 100
ns where the cell has only one run), decay, cross zero at 1–10 ms, and
saturate at `−d` at the largest delays — e.g. hal KHI 128³ FlatterScatter
malloc: `+1.6 s` at 100 ns, `+8.9 s` at 12.6 µs, `0` at ~6 ms, `−0.43 s`
(= `−d`) at 10 s. The large-delay part is the normal absorption; the
small-delay part is the unexplained excess.

**Size.** `E_peak = +0.9 … +12.3 s` against baselines of 152–603 s
(0.2–4.2 % of the whole runtime); per call, `E_peak/N = 0.7–114 µs` — the
same scale as the absorbed slack `c = d/N` (3–291 µs, the `absorption`
table). Against the nominal stall sum at the peak delay the excess is
0.6–823× larger (at a 100 ns peak the nominal sum is only 0.01–0.03 s, so
that ratio is a statement about the definition, not a mechanism).

**Not a single bad run.** The frozen data carry one to three runs per delay
cell. Where a hump cell has more than one run, the sign is consistent:
hal KHI 128³ Flat malloc at 100 ns: `+1.2, +1.6, +8.7 s` (3/3); 256×128×128
Flat malloc at 100 ns: `+4.6, +4.7, +10.3 s` (3/3); at the 12.6 µs peak the
same grid's two runs give `+9.2, +8.6 s`; at the 15.8 µs peak of 256×128×128
the two runs give `+11.6, +10.5 s`. The only mixed multi-run cell is
256×128×256 Flat malloc at 100 ns (`+1.5, +0.4, −0.1 s`, the mildest hump).

**Both operations, both algorithms, one machine.** Within every hal KHI
group the malloc and the free arm hump with similar peaks (ratios
0.9–1.2×), and FlatterScatter and ScatterAlloc hump with similar peaks
(0.9–1.2×). The effect is therefore in what the two operations and the two
allocators share — the KHI workload on hal — not in one allocator's
internal path.

**No grid-size monotonicity.** Peak excess: 128³ `+8.9` (Flat) / `+9.3`
(Scat); 256×128×128 `+11.0` / `+12.3`; 256×128×256 `+1.8` / `+2.4`. A
mechanism that scales simply with problem size or in-flight memory is not
supported by the largest grid being the mildest.

**Independence from the absorption level.** hal KHI 128³ has
`d ≈ 0.4–0.4 s` (no measurable absorption) yet humps by `+9 s`; hal KHI
256×128×256 has `d = 47–70 s` (large absorption) yet humps by only
`+1.8–5.4 s`. The hump and the absorption are separate effects.

## 3. What the data rule out

- **A slope/`N` error.** An under- or over-counted `N` would tilt the
  large-delay line; instead `E(s) → −d` consistently at the largest
  delays, and `N` agrees between the two algorithms of each scenario at
  the 0.00–1.11 % level (`qa-allocation-cost.md` §5). The excess is
  additive at small delay, not a slope error.
- **A fit or model artifact.** `E` is computed from raw runtimes with no
  fitted parameter, so the `(W, A, s0)` flat direction of the fits
  (`analysis-review.md` §7) cannot produce it.
- **A generic delay-injection effect.** The FoilLCT arms and most rosi
  arms sit at `E(100 ns) ≤ 0`; the excess is specific to the hal KHI
  scenario.
- **A plain hiding failure.** Absorption models have `E ≤ 0` by
  construction; `E > 0` requires a mechanism that makes the injected stall
  cost *more* than itself.
- **A one-off transient.** Where the cell has multiple runs, all (or a
  consistent majority) of them hump.

## 4. Candidate mechanisms (not adjudicated by this data)

All three below predict `E > 0` at small-to-mid `s` and a return to the
`N*s` line at large `s`; the current data cannot distinguish them.

1. **Lock-cascade contention.** The delay is a busy-wait at the top of
   `DeviceAllocator::malloc`/`free`; if it is held while the allocator's
   lock is held, a stalled caller queues the other allocation threads
   (streams) that arrive during the burst. A single stalled call then
   delays the burst's other callers, and the excess is the queued work —
   independent of `s` once the queue is full, which fits the flat
   100 ns → 16 µs rise and the grid-size non-monotonicity (it would scale
   with the burst structure, not with the memory).
2. **Pipeline drain/refill.** The stall bubbles the asynchronous
   host/device pipeline: work already in flight must drain (or the host
   lookahead must refill) around the stalled call. The excess is then a
   one-time cost of the in-flight work per stall event; it would scale
   with pipeline depth and the allocation rhythm of the KHI timesteps,
   and would be largest where the rhythm is most bursty.
3. **A KHI allocation-rhythm interaction.** The KHI example's per-timestep
   allocation pattern (burst size, concurrency) may place many callers
   inside one stall window in a way FoilLCT's does not. This is the
   workload-side version of (1)/(2) and would explain the hal specificity
   only if the hal pipeline (A30, driver, stream layout) amplifies it.

A discriminating prediction: mechanisms (1) and (2) make the excess
independent of `s` over a band (queue full / pipeline drained) — which is
what the data show between 100 ns and ~10 µs; mechanism (3) alone would
not. But all three share that prediction, so it is not yet a discriminator.

## 5. Decisive measurements (next data campaign)

1. **Per-timestep allocation counts and timestamps** (mallocMC call
   counters, or a trace on the add-delay branch) for the KHI hump arms —
   tests the burst structure directly (mechanisms 1 and 3).
2. **The delay moved outside the allocator lock's critical section** — if
   the hump disappears, mechanism 1 is confirmed; if it remains, 2/3.
3. **A zero-delay `DeviceAllocator::malloc`/`free` microbenchmark** (the
   `c_a` reading the model's `c_a` constraint path was built for) —
   separates the native cost from the pipeline effect and would let the
   `A = N*c_a` constrained fit price the hump against a known `c_a`.
4. **More repetitions per small-delay cell** (the frozen data have 1–3
   runs per cell) to tighten the per-run statistics of the hump.
5. **A reduced-concurrency variant** (one allocation stream/thread) — a
   contention mechanism (1) should collapse the hump; a pipeline effect
   (2) should not.

## 6. Consequences for the benchmark's claims

- The **Tier-1 (0, 0) baseline comparison is untouched**: no delay is
  injected there, so the hump cannot enter it. If the hump's mechanism
  also acts without injected delay, that is part of what the end-to-end
  baseline comparison legitimately measures.
- **KHI cost comparisons at small delay (≲ 10 ms) are not
  interpretable** as allocation cost: the hump (up to ~4 % of the runtime)
  exceeds the nominal stall sum by orders of magnitude. Delay-sweep claims
  on KHI must be made at the large-delay asymptote, where `E → −d` and the
  gauge-invariant absorption quantities apply.
- **FoilLCT and the rosi arms are clean**; the absorption characterization
  (`absorption` table, the `d`/`c` quantities) stands there in full.
- The anomaly is therefore flagged, quantified, and bounded — but it is
  **open**, and the measurements of §5 are the way to close it.
