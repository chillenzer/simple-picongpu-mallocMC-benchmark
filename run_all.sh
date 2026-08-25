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
# Full sweep for the two-operation Amdahl fit: a (0, 0) baseline, the
# malloc-delay arm (free delay 0) and the free-delay arm (malloc delay 0)
# on a log grid from 100 ns to 1e8 ns with 1/4-decade steps (skipping the
# intermediate steps between 1e5 and 1e6 ns), and a 3x3 joint grid
# coupling both delays so the fit is constrained off the arms. The large
# delays let each Amdahl term A*s0/(d+s0) decay into its 1/d tail so the
# asymptote W + A + N*d gets anchored; this matters most for free, whose
# native cost fades slowly.
DELAY_SWEEP=(100 10000 100000 1000000 1778279 3162278 5623413 10000000 17782794 31622777 56234133 100000000)
JOINT=(10000 1000000 10000000)

COMBINATIONS=("0 0")
for delay in "${DELAY_SWEEP[@]}"; do
  COMBINATIONS+=("$delay 0" "0 $delay")
done
for malloc_delay in "${JOINT[@]}"; do
  for free_delay in "${JOINT[@]}"; do
    COMBINATIONS+=("$malloc_delay $free_delay")
  done
done

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
