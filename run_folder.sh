#!/bin/bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -e
set -x

# Absolutize only relative paths: the Makefile passes them relative.
case "$1" in
  /*) FOLDER=$1 ;;
  *) FOLDER="$(pwd -P)/$1" ;;
esac

case "$2" in
  /*) FLAGSFILE=$2 ;;
  *) FLAGSFILE="$(pwd -P)/$2" ;;
esac
PROFILE=$3
MALLOC_DELAY=${4:-0}
FREE_DELAY=${5:-0}
# An optional 6th argument selects one line of the flags file (1-based):
# the per-grid logs of run_stamp.sh each record one grid run only.
LINE=${6:-}
declare -a FLAGS

# The profile path is a command-line argument, so shellcheck cannot follow it.
# shellcheck disable=SC1090
source "$(pwd -P)/$PROFILE"

OUTPUT="$(pwd -P)/$OUTPUT"
mkdir -p "$(dirname "${OUTPUT}")"

WD=$(pwd -P)

cd "$FOLDER"
while IFS="" read -r line || [ -n "$line" ]; do
  FLAGS+=("$line")
done <"$FLAGSFILE"

for INDEX in "${!FLAGS[@]}"; do
  LINE_NO=$((INDEX + 1))
  if [ -n "$LINE" ] && [ "$LINE_NO" -ne "$LINE" ]; then
    continue
  fi
  FLAG=${FLAGS[INDEX]}
  # Each flags-file line is one full command line and must be word-split
  # into picongpu's arguments, so the unquoted expansion is intentional.
  # shellcheck disable=SC2086
  MALLOCMC_MALLOC_DELAY="$MALLOC_DELAY" MALLOCMC_FREE_DELAY="$FREE_DELAY" bin/picongpu $FLAG
done

cd "$WD"
