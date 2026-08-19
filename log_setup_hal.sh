#!/usr/bin/env bash

set -x
set -e

FILENAME="setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
rm -rf src build

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
