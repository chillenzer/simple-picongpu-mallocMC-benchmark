#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -x
set -e

# The machine-specific values (output folder, profile) come from config.yaml.
MACHINE=hal
FOLDER=$(python3 config.py get "machines.$MACHINE.output")
PROFILE=$(python3 config.py get "machines.$MACHINE.profile")
FILENAME="$FOLDER/run_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
mkdir -p "$FOLDER"

{
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

bash run_all.sh "$PROFILE" flags/ 2>&1 | tee -a "$FILENAME"
