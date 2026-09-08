<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# Archived analysis (working notes of the model review)

This directory holds the exploratory analysis scripts written during the
2025 review of the performance model. They are **not** part of the published
pipeline and **not maintained**: the current analysis lives in
`analysis/compute_results.py` and the model in `analysis/performance_model.py`,
which supersede everything here. They are archived for provenance and
reproducibility of the review process that chose the published quantities and
model; the written records of that review are the corresponding documents in
`notes/working/` (restored from the `notes-pre-publish` tag / history).

A note on the model: these scripts were written while the model's fade term
was the hyperbola `A·s0/(s+s0)`. The published model now uses the exponential
`A·exp(-s/s0)` (`analysis/performance_model.py`). Everything in this archive
that references "the stored model" refers to the hyperbola-era fit and must
not be read against the current stored fits.

| script | content (at archive time) |
|---|---|
| `critique_fit.py` | re-analysis of the model fits: nested-model tests, seed sensitivity, baseline-anchored absorption, stored-vs-data-pinned decomposition, gauge symmetry |
| `qa_flat_direction.py` (Q1) | the flat direction of the fit and whether a re-parameterization removes it |
| `qa2_analysis.py`, `qa2_table.py` (Q2) | whether the delay-sweep plateau measures (de-)allocation cost |
| `qa_fade_term.py` (Q3) | candidate fade shapes for the model (drove the switch to the exponential) |
| `diagnose_humps.py` | model-free diagnosis of the KelvinHelmholtz small-delay amplification humps |

Inputs/outputs: these scripts read `output/results.h5` (the results file built
by `make results`), which is not in the repository; the raw benchmark data
lives on the benchmark machines and in the released archive (`*Where the data
lives*` in the README). Their `*-output.txt` captures are in `notes/working/`,
not here.
