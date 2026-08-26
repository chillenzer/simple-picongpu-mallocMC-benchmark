#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile, modules) come from
# config.json.
MACHINE=rosi
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
PROFILE=$(python3 config.py get "machines.$MACHINE.profile")
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# make reuses an existing src/ and build/ and only re-runs the parts that are
# not up to date any more; `make clean` / `make distclean` remove them for a
# fully clean run.
mkdir -p "$FOLDER"

{
  echo "========================"
  echo "Loading environment"
  echo "========================"
} >>"$FILENAME"

while IFS= read -r MODULE; do
  module load "$MODULE"
done < <(python3 config.py list "machines.$MACHINE.modules")

{
  echo "========================"
  echo "Logging environment"
  echo "========================"
  git log -n1
  git diff
  env
} >>"$FILENAME"

{
  echo "========================"
  echo "Starting setup"
  echo "========================"
} >>"$FILENAME"

make build PROFILE="$PROFILE" PARAM_DIR=param 2>&1 | tee -a "$FILENAME"
