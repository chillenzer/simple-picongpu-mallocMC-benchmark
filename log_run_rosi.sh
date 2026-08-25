#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile) come from config.json.
MACHINE=rosi
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
PROFILE=$(python3 config.py get "machines.$MACHINE.profile")
FILENAME="$FOLDER/run_$(date --rfc-3339=seconds | sed 's/ /_/g')_${SLURM_JOB_ID}.txt"
mkdir -p "$FOLDER"

echo "========================" | tee -a "$FILENAME"
echo "Logging environment" | tee -a "$FILENAME"
echo "========================" | tee -a "$FILENAME"

git log -n1 | tee -a "$FILENAME"
git diff | tee -a "$FILENAME"
nvidia-smi | tee -a "$FILENAME"
hostname | tee -a "$FILENAME"

echo "========================" | tee -a "$FILENAME"
echo "Starting run" | tee -a "$FILENAME"
echo "========================" | tee -a "$FILENAME"

bash run_all.sh "$PROFILE" flags/ 2>&1 | tee -a "$FILENAME"
