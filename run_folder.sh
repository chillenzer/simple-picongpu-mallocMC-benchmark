#!/bin/bash

set -e
set -x

# Absolutize only relative paths: run_all.sh passes the flags file absolute.
case $1 in
  /*) FOLDER=$1 ;;
  *)  FOLDER=$(pwd -P)/$1 ;;
esac

case $2 in
  /*) FLAGSFILE=$2 ;;
  *)  FLAGSFILE=$(pwd -P)/$2 ;;
esac
PROFILE=$3
SLEEP_TIME=${4:-0}
declare -a FLAGS

source $(pwd -P)/$PROFILE

OUTPUT=$(pwd -P)/$OUTPUT
mkdir -p $(dirname ${OUTPUT})

WD=$(pwd -P)

cd $FOLDER
while IFS="" read -r line || [ -n "$line" ]; do
  FLAGS+=("$line")
done <"$FLAGSFILE"

for FLAG in "${FLAGS[@]}"; do
  MALLOCMC_SLEEP_TIME="$SLEEP_TIME" bin/picongpu $FLAG
done

cd $WD
