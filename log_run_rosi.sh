#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile) come from config.json.
MACHINE=rosi
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
FILENAME="$FOLDER/run_$(date --rfc-3339=seconds | sed 's/ /_/g')_${SLURM_JOB_ID:-local}.txt"
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

# One slurm job runs one full-sweep repetition: the first argument is the
# repetition number (REP), REPEATS is the total number of repetitions of
# the series (environment variable, default 1). Submit one job per
# repetition with the same REPEATS, e.g.
#
#   sbatch log_run_rosi.sh 1   # with REPEATS=3 in the environment
#   sbatch log_run_rosi.sh 2
#   sbatch log_run_rosi.sh 3
#
# (All slurm allocation options come from the sbatch invocation itself; this
# script carries no #SBATCH directives.) Without a repetition number the job
# runs all remaining repetitions serially.
make runs MACHINE="$MACHINE" REPEATS="${REPEATS:-1}" REP="${1:-}" 2>&1 | tee -a "$FILENAME"
