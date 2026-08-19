#!/usr/bin/env bash

set -x
set -e

FOLDER="output/hal-sleeptimes"
FILENAME="$FOLDER/run_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
mkdir -p "$FOLDER"

echo "========================" >>$FILENAME
echo "Logging environment" >>$FILENAME
echo "========================" >>$FILENAME

git log -n1 >>$FILENAME
git diff >>$FILENAME
nvidia-smi >>$FILENAME
spack find --loaded >>$FILENAME
env >>$FILENAME
hwinfo >>$FILENAME

echo "========================" >>$FILENAME
echo "Starting run" >>$FILENAME
echo "========================" >>$FILENAME

bash run_all.sh profiles/hal.sh flags/ 2>&1 | tee -a $FILENAME
