#!/bin/bash

set -e
set -x

FOLDER=$1
FLAGSFILE=$2
OUTPUT=$3
PROFILE=$4
declare -a FLAGS

OUTPUT=$(pwd -P)/$OUTPUT
mkdir -p $(dirname ${OUTPUT})

WD=$(pwd -P)

cd $FOLDER
while IFS="" read -r line || [ -n "$line" ]; do
  FLAGS+=("$line")
done <"$FLAGSFILE"

for FLAG in "${FLAGS[@]}"; do
  bin/picongpu $FLAG >(tee -a ${OUTPUT}.log) 2> >(tee -a ${OUTPUT}.err >&2)
done

cd $WD
