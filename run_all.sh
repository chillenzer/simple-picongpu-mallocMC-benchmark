#!/bin/bash

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

EXAMPLES=("KelvinHelmholtz" "FoilLCT")
ALGORITHMS=("FlatterScatter" "ScatterAlloc" "Gallatin")

mkdir -p output
for example in ${EXAMPLES[@]}; do
  mkdir -p output/$example
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
