#!/usr/bin/env bash

set -x
set -e

FOLDER="output/hal-sleeptimes"
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# setup.sh reuses an existing src/ and build/ and only re-runs the parts that
# are not up to date any more; delete them manually for a fully clean run.
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
echo "Starting setup" >>$FILENAME
echo "========================" >>$FILENAME

bash setup.sh profiles/hal.sh param/ 2>&1 | tee -a $FILENAME
