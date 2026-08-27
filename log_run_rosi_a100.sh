#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile) come from config.json.
MACHINE=rosi-a100
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
# The session log belongs to sessions/ (a free-text backup, never re-read
# by the analysis).
FILENAME="$FOLDER/sessions/run_$(date --rfc-3339=seconds | sed 's/ /_/g')_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$FOLDER/sessions"

# The machine-readable provenance line (schema 1, kind "setup"); the
# analysis skips session logs.
python3 logmeta.py setup --machine "$MACHINE" >>"$FILENAME"

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
# the series (environment variable, default 1), and PHASE (environment
# variable, default arms) selects the sweep's delay combinations. Submit one
# job per repetition with the same REPEATS, e.g.
#
#   sbatch log_run_rosi_a100.sh 1   # with REPEATS=3 in the environment
#   sbatch log_run_rosi_a100.sh 2
#   sbatch log_run_rosi_a100.sh 3
#
# (All slurm allocation options come from the sbatch invocation itself; this
# script carries no #SBATCH directives.) Without a repetition number the job
# runs all remaining repetitions serially. The run stamps make the phases
# incremental: an arms job after an initial sweep re-runs only the new
# combinations.
make runs MACHINE="$MACHINE" PHASE="${PHASE:-arms}" REPEATS="${REPEATS:-1}" REP="${1:-}" 2>&1 | tee -a "$FILENAME"
