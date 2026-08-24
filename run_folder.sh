#!/bin/bash

set -e
set -x

# Absolutize only relative paths: run_all.sh passes the flags file absolute.
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

for FLAG in "${FLAGS[@]}"; do
  # Each flags-file line is one full command line and must be word-split
  # into picongpu's arguments, so the unquoted expansion is intentional.
  # shellcheck disable=SC2086
  MALLOCMC_MALLOC_DELAY="$MALLOC_DELAY" MALLOCMC_FREE_DELAY="$FREE_DELAY" bin/picongpu $FLAG
done

cd "$WD"
