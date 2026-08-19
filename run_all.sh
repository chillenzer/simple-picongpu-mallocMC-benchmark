#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("KelvinHelmholtz" "FoilLCT")
# Keep in sync with ALGORITHM/SLEEP_TIMES in setup.sh.
ALGORITHM="FlatterScatter"
SLEEP_TIMES=(0 1000 10000 50000 100000 500000)
ALGORITHMS=()
for sleep_time in "${SLEEP_TIMES[@]}"; do
  ALGORITHMS+=("${ALGORITHM}-sleep${sleep_time}")
done

for example in ${EXAMPLES[@]}; do
  for algorithm in ${ALGORITHMS[@]}; do
    echo "=============================="
    echo "Running example: $example"
    echo "Using algorithm: $algorithm"
    echo "=============================="
    DATETIME=$(date +"%Y-%m-%d %H:%M:%S")
    bash run_folder.sh build/$example/$algorithm $FLAGSFOLDER/${example}.flags $PROFILE
    echo ""
  done
done
