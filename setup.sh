#!/bin/bash

set -e

PROFILE=$1
PARAM_DIR=$2

MALLOCMC_URL="https://github.com/chillenzer/mallocMC"
MALLOCMC_SRC="src/picongpu/thirdParty/mallocMC"
MALLOCMC_HASH="origin/add-GallatinCuda"
PICONGPU_URL="https://github.com/ComputationalRadiationPhysics/picongpu"
PICONGPU_SRC="src/picongpu"
PICONGPU_HASH="dev"
EXAMPLES=("KelvinHelmholtz" "FoilLCT")
ALGORITHMS=("FlatterScatter" "ScatterAlloc" "Gallatin")

MALLOCMC_SRC=$(pwd -P)/$MALLOCMC_SRC
PICONGPU_SRC=$(pwd -P)/$PICONGPU_SRC
FLAGS="-DCMAKE_CXX_FLAGS=-I${MALLOCMC_SRC}/thirdParty/gallatin/include -DCMAKE_CUDA_FLAGS=-I${MALLOCMC_SRC}/thirdParty/gallatin/include -Dalpaka_CXX_STANDARD=20"

function clone() {
  URL=$1
  DEST=$2
  HASH=$3

  WD=$(pwd -P)

  git clone $URL $DEST

  # Yes, we could the following directly in the `clone` command but we'd have to look up the syntax.
  cd $DEST
  git checkout $HASH
  git submodule init
  git submodule update

  cd $WD
}

function clone_src() {
  mkdir -p src
  clone $PICONGPU_URL $PICONGPU_SRC $PICONGPU_HASH
  # We patch in our custom version including Gallatin.
  rm -rf $MALLOCMC_SRC
  clone $MALLOCMC_URL $MALLOCMC_SRC $MALLOCMC_HASH
}

function create_input() {
  SRC=$1
  DEST=$2
  ALGO=$3
  EXAMPLE=$4

  pic-create $SRC $DEST
  if [ -d "$PARAM_DIR/${ALGO}" ] && [ -n "$(ls -A $PARAM_DIR/${ALGO}/*.param 2>/dev/null)" ]; then
    cp $PARAM_DIR/${ALGO}/*.param $DEST/include/picongpu/param/
  fi
  if [ -d "$PARAM_DIR/${EXAMPLE}" ] && [ -n "$(ls -A $PARAM_DIR/${EXAMPLE}/*.param 2>/dev/null)" ]; then
    cp $PARAM_DIR/${EXAMPLE}/*.param $DEST/include/picongpu/param/
  fi
  if [ -d "$PARAM_DIR/${EXAMPLE}/${ALGO}" ] && [ -n "$(ls -A $PARAM_DIR/${EXAMPLE}/${ALGO}/*.param 2>/dev/null)" ]; then
    cp $PARAM_DIR/${EXAMPLE}/${ALGO}/*.param $DEST/include/picongpu/param/
  fi
}

function prepare_inputs() {
  mkdir -p build
  for example in ${EXAMPLES[@]}; do
    mkdir -p build/$example
    for algorithm in ${ALGORITHMS[@]}; do
      create_input $PICONGPU_SRC/share/picongpu/examples/$example build/$example/$algorithm $algorithm $example
    done
  done
}

function build_from_input() {
  DEST=$1

  WD=$(pwd -P)

  cd $DEST
  HERE=$(pwd -P)
  export CMAKE_PREFIX_PATH=$MALLOCMC_SRC:$CMAKE_PREFIX_PATH
  # A little bit dirty, mallocMC's CMakeLists.txt is not exactly clean:
  pic-build -c "$FLAGS"

  cd $WD
}

function build() {
  mkdir -p build
  for example in ${EXAMPLES[@]}; do
    mkdir -p build/$example
    for algorithm in ${ALGORITHMS[@]}; do
      build_from_input build/$example/$algorithm
    done
  done
}

function prepare_environment() {
  sed -i 's|PICSRC=.*|PICSRC='"$PICONGPU_SRC"'|g' $PROFILE
  source $PROFILE
}

function main() {
  clone_src
  prepare_environment
  prepare_inputs
  build
}

main
