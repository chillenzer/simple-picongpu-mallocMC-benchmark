#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

FOLDER="output/rosi-sleeptimes"
FILENAME="$FOLDER/setup_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

# setup.sh reuses an existing src/ and build/ and only re-runs the parts that
# are not up to date any more; delete them manually for a fully clean run.
mkdir -p "$FOLDER"

{
  echo "========================"
  echo "Loading environment"
  echo "========================"
} >>"$FILENAME"

module load hopper GCCcore/14.3.0 git/2.50.1

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

bash setup.sh profiles/rosi-v100.profile param/ 2>&1 | tee -a "$FILENAME"
