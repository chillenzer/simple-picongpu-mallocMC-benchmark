#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("FoilLCT")
# Allocation delays in nanoseconds, injected at run time by mallocMC through
# the MALLOCMC_SLEEP_TIME environment variable (run_folder.sh sets it), so one
# build per example serves the whole sweep.
SLEEP_TIMES=(100 10000 12589 15849 19953 25119 31623 39811 50119 63096 79433 100000 125893 158489 199526 251189 316228 398107 501187 630957 794328)

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
