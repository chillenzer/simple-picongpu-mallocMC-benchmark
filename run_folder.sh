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
  /*) PROFILE=$2 ;;
  *) PROFILE="$(pwd -P)/$2" ;;
esac
MALLOC_DELAY=${3:-0}
FREE_DELAY=${4:-0}
# The 5th argument is one full picongpu command line (one of the example's
# structured flag_lines, serialized by config.py), word-split into picongpu's
# arguments by the unquoted expansion below.
LINE=$5

# The profile path is a command-line argument, so shellcheck cannot follow it.
# shellcheck disable=SC1090
source "$PROFILE"

OUTPUT="$(pwd -P)/$OUTPUT"
mkdir -p "$(dirname "${OUTPUT}")"

WD=$(pwd -P)

cd "$FOLDER"
# The command line must be word-split into picongpu's arguments, so the
# unquoted expansion is intentional.
# shellcheck disable=SC2086
MALLOCMC_MALLOC_DELAY="$MALLOC_DELAY" MALLOCMC_FREE_DELAY="$FREE_DELAY" bin/picongpu $LINE

cd "$WD"
