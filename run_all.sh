#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("FoilLCT" "KelvinHelmholtz")
# (malloc_delay, free_delay) combinations in nanoseconds, injected at run
# time by mallocMC through the MALLOCMC_MALLOC_DELAY / MALLOCMC_FREE_DELAY
# environment variables (run_folder.sh sets them), so one build per example
# serves the whole sweep.
#
# The default sweep below is the one the two-operation Amdahl fit expects:
# a (0, 0) baseline, the malloc-delay sweep at free delay 0, the
# free-delay sweep at malloc delay 0, and a joint grid spanning the range.
DELAY_SWEEP=(100 10000 100000 177828 316228 562341 1000000 1778279 3162278 5623413 10000000)
JOINT=(10000 1000000 10000000)

COMBINATIONS=("0 0")
for delay in ${DELAY_SWEEP[@]}; do
  COMBINATIONS+=("$delay 0" "0 $delay")
done
for malloc_delay in ${JOINT[@]}; do
  for free_delay in ${JOINT[@]}; do
    COMBINATIONS+=("$malloc_delay $free_delay")
  done
done

for example in ${EXAMPLES[@]}; do
  for combination in ${COMBINATIONS[@]}; do
    read -r malloc_delay free_delay <<< "$combination"
    echo "=============================="
    echo "Running example: $example"
    echo "Using allocator: FlatterScatter, malloc delay: $malloc_delay ns, free delay: $free_delay ns"
    echo "=============================="
    bash run_folder.sh build/$example $FLAGSFOLDER/${example}.flags $PROFILE "$malloc_delay" "$free_delay"
    echo ""
  done
done
