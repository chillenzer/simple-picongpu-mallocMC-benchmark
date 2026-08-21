#!/usr/bin/env bash

set -x
set -e

FOLDER="output/rosi-sleeptimes"
FILENAME="$FOLDER/run_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
mkdir -p "$FOLDER"

echo "========================" | tee -a $FILENAME
echo "Logging environment" | tee -a $FILENAME
echo "========================" | tee -a $FILENAME

git log -n1 | tee -a $FILENAME
git diff | tee -a $FILENAME
nvidia-smi | tee -a $FILENAME
hostname | tee -a $FILENAME

echo "========================" | tee -a $FILENAME
echo "Starting run" | tee -a $FILENAME
echo "========================" | tee -a $FILENAME

bash run_all.sh profiles/rosi-v100.profile flags/ 2>&1 | tee -a $FILENAME
