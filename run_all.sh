#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("KelvinHelmholtz" "FoilLCT")
# Allocation delays in nanoseconds, injected at run time by mallocMC through
# the MALLOCMC_SLEEP_TIME environment variable (run_folder.sh sets it), so one
# build per example serves the whole sweep.
SLEEP_TIMES=(0 1000 1300 1800 2400 3100 4200 5000 7500 10000 13000 18000 24000 31000 42000 50000 75000 100000 130000 180000 240000 310000 420000 500000 750000)

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
