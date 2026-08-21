#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("FoilLCT" "KelvinHelmholtz")
# Allocation delays in nanoseconds, injected at run time by mallocMC through
# the MALLOCMC_SLEEP_TIME environment variable (run_folder.sh sets it), so one
# build per example serves the whole sweep.
SLEEP_TIMES=(100 10000 17783 31623 56235 100000 177828 316228 562341 1000000 1778279 3162278 5623413 10000000)

for example in ${EXAMPLES[@]}; do
  for sleep_time in ${SLEEP_TIMES[@]}; do
    echo "=============================="
    echo "Running example: $example"
    echo "Using allocator: FlatterScatter, sleep_time: $sleep_time ns"
    echo "=============================="
    bash run_folder.sh build/$example $FLAGSFOLDER/${example}.flags $PROFILE $sleep_time
    echo ""
  done
done
