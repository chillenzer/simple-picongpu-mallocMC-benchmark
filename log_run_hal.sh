#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile) come from config.json.
MACHINE=hal
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
# The session log belongs to sessions/ (a free-text backup, never re-read
# by the analysis).
FILENAME="$FOLDER/sessions/run_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
mkdir -p "$FOLDER/sessions"

{
  # The machine-readable provenance line (schema 1, kind "setup"); the
  # analysis skips session logs.
  python3 logmeta.py setup --machine "$MACHINE"
  echo "========================"
  echo "Logging environment"
  echo "========================"
  git log -n1
  git diff
  nvidia-smi
  spack find --loaded
  env
  hwinfo
} >>"$FILENAME"

{
  echo "========================"
  echo "Starting run"
  echo "========================"
} >>"$FILENAME"

# REPEATS (default 1), REP (default: all repetitions) and PHASE (default
# arms) are environment variables, like the Makefile's invocation values.
# Each finished (combination, repetition) writes a stamp, so an interrupted
# sweep continues where it stopped, and a phase extension re-runs only the
# new combinations.
make runs MACHINE="$MACHINE" PHASE="${PHASE:-arms}" REPEATS="${REPEATS:-1}" REP="${REP:-}" 2>&1 | tee -a "$FILENAME"
