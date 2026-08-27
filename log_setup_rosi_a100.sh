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
# Session logs live in a folder of their own: they are a free-text backup
# of the launch environment, never re-read by the analysis (which parses
# the top-level run logs only), and never superseded.
FILENAME="$FOLDER/sessions/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# make reuses an existing src/ and build/ and only re-runs the parts that are
# not up to date any more; `make clean` / `make distclean` remove them for a
# fully clean run.
mkdir -p "$FOLDER/sessions"

# The machine-readable provenance line (schema 1, kind "setup"); the
# analysis skips session logs.
python3 logmeta.py setup --machine "$MACHINE" >>"$FILENAME"

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

make build PROFILE="$PROFILE" PARAM_DIR=param 2>&1 | tee -a "$FILENAME"
