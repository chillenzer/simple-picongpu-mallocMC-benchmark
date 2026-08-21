#!/usr/bin/env bash

set -x
set -e

FOLDER="output/rosi-sleeptimes"
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# setup.sh reuses an existing src/ and build/ and only re-runs the parts that
# are not up to date any more; delete them manually for a fully clean run.
mkdir -p "$FOLDER"

echo "========================" >>$FILENAME
echo "Loading environment" >>$FILENAME
echo "========================" >>$FILENAME

module load hopper GCCcore/14.3.0 git/2.50.1

echo "========================" >>$FILENAME
echo "Logging environment" >>$FILENAME
echo "========================" >>$FILENAME

git log -n1 >>$FILENAME
git diff >>$FILENAME
env >>$FILENAME

echo "========================" >>$FILENAME
echo "Starting setup" >>$FILENAME
echo "========================" >>$FILENAME

bash setup.sh profiles/rosi-v100.profile param/ 2>&1 | tee -a $FILENAME
