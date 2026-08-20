#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("KelvinHelmholtz" "FoilLCT")
# Keep in sync with ALGORITHM/SLEEP_TIMES in setup.sh.
ALGORITHM="FlatterScatter"
SLEEP_TIMES=(0 1000 1300 1800 2400 3100 4200 5000 7500 10000 13000 18000 24000 31000 42000 50000 75000 100000 130000 180000 240000 310000 420000 500000 750000)
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
