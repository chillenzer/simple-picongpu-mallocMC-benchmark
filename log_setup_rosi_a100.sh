#!/usr/bin/env bash

set -x
set -e

FOLDER="output/rosi-a100-sleeptimes"
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# setup.sh reuses an existing src/ and build/ and only re-runs the parts that
# are not up to date any more; delete them manually for a fully clean run.
mkdir -p "$FOLDER"

echo "========================" | tee -a $FILENAME
echo "Loading environment" | tee -a $FILENAME
echo "========================" | tee -a $FILENAME

module load hopper GCCcore/14.3.0 git/2.50.1

echo "========================" | tee -a $FILENAME
echo "Logging environment" | tee -a $FILENAME
echo "========================" | tee -a $FILENAME

git log -n1 | tee -a $FILENAME
git diff | tee -a $FILENAME
hostname | tee -a $FILENAME
env | tee -a $FILENAME

echo "========================" | tee -a $FILENAME
echo "Starting setup" | tee -a $FILENAME
echo "========================" | tee -a $FILENAME

bash setup.sh profiles/rosi-a100.profile param/ 2>&1 | tee -a $FILENAME
