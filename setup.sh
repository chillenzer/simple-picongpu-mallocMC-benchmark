#!/bin/bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

set -e

PROFILE=$1
PARAM_DIR=$2

# add-delay branch: FlatterScatter delay support, run-time malloc/free delays
# via the MALLOCMC_MALLOC_DELAY / MALLOCMC_FREE_DELAY environment variables
# (read into the device-side allocator by mallocMC::Allocator::alloc), and
# the delays as busy-waits on the device global timer (replacing the
# __nanosleep intrinsic, whose wake-up guarantee was too weak for a
# controlled delay).
MALLOCMC_URL="https://github.com/chillenzer/mallocMC"
MALLOCMC_SRC="src/picongpu/thirdParty/mallocMC"
MALLOCMC_HASH="2eb8a18e3298afc18060edcd6e24b09316e2ee18"
PICONGPU_URL="https://github.com/ComputationalRadiationPhysics/picongpu"
PICONGPU_SRC="src/picongpu"
PICONGPU_HASH="6e7d58bb97300ac74cd7a09e13b3c03fdd3863ae"
EXAMPLES=("KelvinHelmholtz" "FoilLCT")
ALGORITHMS=("FlatterScatter" "ScatterAlloc" "Gallatin")

MALLOCMC_SRC="$(pwd -P)/$MALLOCMC_SRC"
PICONGPU_SRC="$(pwd -P)/$PICONGPU_SRC"
# Nasty little bug here: GCC has a constexpr std::source_location but nvcc does not.
# So, Boost gets confused and tries to use std::source_location constexpr.
CXX_FLAGS="-DBOOST_DISABLE_CURRENT_LOCATION"
FLAGS="-DCMAKE_CXX_FLAGS=\"$CXX_FLAGS\" -DCMAKE_CUDA_FLAGS=\"$CXX_FLAGS\" -Dalpaka_CXX_STANDARD=20 -DmallocMC_USE_Gallatin=ON_ALLOW_FETCH"

function clone() {
  URL=$1
  DEST=$2
  HASH=$3

  WD=$(pwd -P)

  git clone "$URL" "$DEST"

  # Yes, we could the following directly in the `clone` command but we'd have to look up the syntax.
  cd "$DEST"
  git checkout "$HASH"
  git submodule init
  git submodule update

  cd "$WD"
}

function ensure_src() {
  DEST=$1
  URL=$2
  HASH=$3

  if [ -d "$DEST/.git" ]; then
    # Reuse the existing checkout: only sync when the pinned hash changed.
    if [ "$(git -C "$DEST" rev-parse HEAD)" != "$HASH" ]; then
      echo "Updating $DEST to ${HASH:0:8} ..."
      git -C "$DEST" fetch --quiet
      git -C "$DEST" checkout --quiet "$HASH"
    fi
    git -C "$DEST" submodule update --init --force --quiet
    echo "Using $DEST @ ${HASH:0:8}."
  else
    if [ -e "$DEST" ]; then
      echo "Replacing $DEST (not a git checkout) ..."
      rm -rf "$DEST"
    fi
    echo "Cloning $DEST @ ${HASH:0:8} ..."
    clone "$URL" "$DEST" "$HASH"
  fi
}

function prepare_src() {
  mkdir -p src
  ensure_src "$PICONGPU_SRC" "$PICONGPU_URL" "$PICONGPU_HASH"

  # We want full control over the version, so we patch in our own.
  ensure_src "$MALLOCMC_SRC" "$MALLOCMC_URL" "$MALLOCMC_HASH"
}

function hash_dir() {
  # md5 over all file contents of a directory tree; prints nothing if missing.
  if [ -d "$1" ]; then
    find "$1" -type f -exec md5sum {} + | md5sum | awk '{print $1}'
  fi
}

function input_fingerprint() {
  # Everything that determines the content of an input directory:
  # the PIConGPU pin (pic-create template), the example, the algorithm and
  # the overlay parameter files.
  EXAMPLE=$1
  ALGO=$2
  {
    echo "$PICONGPU_HASH"
    echo "$EXAMPLE"
    echo "$ALGO"
    hash_dir "$PARAM_DIR/$ALGO"
    hash_dir "$PARAM_DIR/$EXAMPLE"
    hash_dir "$PARAM_DIR/$EXAMPLE/$ALGO"
  } | md5sum | awk '{print $1}'
}

function create_input() {
  SRC=$1
  DEST=$2
  EXAMPLE=$3
  ALGO=$4

  FINGERPRINT=$(input_fingerprint "$EXAMPLE" "$ALGO")
  if [ -f "$DEST/.input-stamp" ] && [ "$(cat "$DEST/.input-stamp")" = "$FINGERPRINT" ]; then
    echo "Input $DEST is up to date; keeping."
    return 0
  fi

  echo "Preparing input $DEST ..."
  rm -rf "$DEST"
  pic-create "$SRC" "$DEST"
  # The algorithm's mallocMC.param (the creation policy), the example's
  # parameters, and any per-(example, algorithm) overrides; later levels
  # win on name clashes.
  find "$PARAM_DIR"/* -type f \
    -wholename "$PARAM_DIR/${ALGO}/"'*'".param" \
    -exec cp -v {} "$DEST/include/picongpu/param/" \;
  find "$PARAM_DIR"/* -type f \
    -wholename "$PARAM_DIR/${EXAMPLE}/"'*'".param" \
    -exec cp -v {} "$DEST/include/picongpu/param/" \;
  find "$PARAM_DIR"/* -type f \
    -wholename "$PARAM_DIR/${EXAMPLE}/${ALGO}/"'*'".param" \
    -exec cp -v {} "$DEST/include/picongpu/param/" \;
  echo "$FINGERPRINT" >"$DEST/.input-stamp"
  echo "Prepared input $DEST."
}

function prepare_inputs() {
  mkdir -p build
  for example in "${EXAMPLES[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
      create_input "$PICONGPU_SRC/share/picongpu/examples/$example" "build/$example/$algorithm" "$example" "$algorithm"
    done
  done
}

function toolchain_fingerprint() {
  # Toolchain versions as loaded by the profile. A change here (for example
  # after updating the profile) invalidates all builds.
  for TOOL in gcc cmake nvcc; do
    if command -v "$TOOL" >/dev/null 2>&1; then
      echo "$TOOL: $("TOOL" --version 2>/dev/null | sed -n 1p)"
    fi
  done
  return 0
}

function build_fingerprint() {
  # Everything that determines whether the build is still valid:
  # the input (via its stamp), the build flags, the mallocMC pin (whose
  # headers are compiled into the binary), the profile and the toolchain.
  DEST=$1
  {
    if [ -f "$DEST/.input-stamp" ]; then
      cat "$DEST/.input-stamp"
    else
      echo "missing-input-stamp"
    fi
    echo "$FLAGS"
    echo "$MALLOCMC_HASH"
    md5sum "$PROFILE" | awk '{print $1}'
    toolchain_fingerprint
  } | md5sum | awk '{print $1}'
}

function build_from_input() {
  DEST=$1

  WD=$(pwd -P)

  FINGERPRINT=$(build_fingerprint "$DEST")
  if [ -f "$DEST/.build-stamp" ] && [ "$(cat "$DEST/.build-stamp")" = "$FINGERPRINT" ] &&
    [ -x "$DEST/bin/picongpu" ]; then
    echo "Build $DEST is up to date; skipping."
    return 0
  fi

  cd "$DEST"
  export CMAKE_PREFIX_PATH="$MALLOCMC_SRC:$CMAKE_PREFIX_PATH"
  # A little bit dirty, mallocMC's CMakeLists.txt is not exactly clean:
  pic-build -c "$FLAGS"

  cd "$WD"
  echo "$FINGERPRINT" >"$DEST/.build-stamp"
}

function build() {
  mkdir -p build
  for example in "${EXAMPLES[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
      build_from_input "build/$example/$algorithm"
    done
  done
}

function prepare_environment() {
  sed -i 's|PICSRC=.*|PICSRC='"$PICONGPU_SRC"'|g' "$PROFILE"
  # The profile path is a command-line argument, so shellcheck cannot follow it.
  # shellcheck disable=SC1090
  source "$PROFILE"
}

function main() {
  prepare_src
  prepare_environment
  prepare_inputs
  build
}

main
