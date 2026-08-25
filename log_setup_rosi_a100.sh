#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile, modules) come from
# config.json.
MACHINE=rosi-a100
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
PROFILE=$(python3 config.py get "machines.$MACHINE.profile")
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# setup.sh reuses an existing src/ and build/ and only re-runs the parts that
# are not up to date any more; delete them manually for a fully clean run.
mkdir -p "$FOLDER"

echo "========================" | tee -a "$FILENAME"
echo "Loading environment" | tee -a "$FILENAME"
echo "========================" | tee -a "$FILENAME"

while IFS= read -r MODULE; do
  module load "$MODULE"
done < <(python3 config.py list "machines.$MACHINE.modules")

echo "========================" | tee -a "$FILENAME"
echo "Logging environment" | tee -a "$FILENAME"
echo "========================" | tee -a "$FILENAME"

git log -n1 | tee -a "$FILENAME"
git diff | tee -a "$FILENAME"
hostname | tee -a "$FILENAME"
env | tee -a "$FILENAME"

echo "========================" | tee -a "$FILENAME"
echo "Starting setup" | tee -a "$FILENAME"
echo "========================" | tee -a "$FILENAME"

bash setup.sh "$PROFILE" param/ 2>&1 | tee -a "$FILENAME"
