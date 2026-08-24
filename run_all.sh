#!/bin/bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("FoilLCT" "KelvinHelmholtz")
# (malloc_delay, free_delay) combinations in nanoseconds, injected at run
# time by mallocMC through the MALLOCMC_MALLOC_DELAY / MALLOCMC_FREE_DELAY
# environment variables (run_folder.sh sets them), so one build per example
# serves the whole sweep.
#
# Follow-up sweep for the KelvinHelmholtz 128^3 group: in the two-operation
# Amdahl fit the free fade scale f0 sat at its search cap (the native free
# cost does not fade within the measured free sweep), leaving the (W,
# A_free) pair weakly constrained and the fit's error sleeve wide. These
# four points extend the free-delay arm (malloc delay 0) two decades beyond
# the previous maximum of 1e7 ns, on the same 1/4-decade log grid, so the
# free term A_free*f0/(f+f0) decays into its 1/f tail and the asymptote
# W + A_malloc + N_free*f gets anchored at large f.
COMBINATIONS=("0 17782794" "0 31622777" "0 56234133" "0 100000000")

echo "All combinations:"
echo "${COMBINATIONS[@]}"
for example in "${EXAMPLES[@]}"; do
  # Quoted: the combinations contain spaces ("100 0"), and unquoted
  # ${COMBINATIONS[@]} word-splits each of them into bare numbers, so the
  # free delay is lost and defaults to 0.
  for combination in "${COMBINATIONS[@]}"; do
    read -r malloc_delay free_delay <<<"$combination"
    echo "=============================="
    echo "Running example: $example"
    echo "Using allocator: FlatterScatter, malloc delay: $malloc_delay ns, free delay: $free_delay ns"
    echo "=============================="
    bash run_folder.sh "build/$example" "$FLAGSFOLDER/${example}.flags" "$PROFILE" "$malloc_delay" "$free_delay"
    echo ""
  done
done
