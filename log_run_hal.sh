#!/usr/bin/env bash

set -x
set -e

FOLDER="output/hal-sleeptimes"
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

bash run_all.sh profiles/hal.sh flags/ 2>&1 | tee -a "$FILENAME"
