#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# Worker of the Makefile run targets. One invocation is one run: the
# example's whole flags file once, from the matching build, with one (malloc
# delay, free delay) combination in the environment.
#
#   run_stamp.sh <machine> <repeats> <example> <algorithm> <malloc-ns> <free-ns> <rep>
#
# The run is recorded in one self-contained log in the machine's output
# directory (a metadata header - machine, the start datetime, the commit,
# the pinned and the checked-out dependency hashes, the sha256 of the
# binary used, the host, the slurm job and node list when running under
# slurm, the GPU and its driver, the loaded modules and the compiler -
# followed by the run's full output), and its success is stamped in
# run-stamps/<machine>/<example>/<algorithm>/<malloc>_<free>/rep-<rep>.stamp
# (content: the log path). An existing stamp is what makes `make runs`
# skip the run.

set -e

MACHINE=$1
REPEATS=$2
EXAMPLE=$3
ALGORITHM=$4
MALLOC_DELAY=$5
FREE_DELAY=$6
REP=$7

BIN="build/${EXAMPLE}/${ALGORITHM}/bin/picongpu"
if [ ! -x "$BIN" ]; then
  echo "run_stamp.sh: no $BIN; run the machine's setup first (make build PROFILE=...)" >&2
  exit 1
fi
if [ "$REP" -lt 1 ] || [ "$REP" -gt "$REPEATS" ]; then
  echo "run_stamp.sh: rep $REP is outside 1..$REPEATS" >&2
  exit 1
fi

PROFILE=$(python3 config.py get "machines.${MACHINE}.profile")
OUTDIR=$(python3 config.py get "machines.${MACHINE}.output")
# The pins of config.json, not the checkout: they are what the binary was
# built from (the `# source:` line of the header, when the build wrote one,
# records the actually checked-out hashes instead).
PIC_PIN=$(python3 config.py get dependencies.picongpu.hash)
MC_PIN=$(python3 config.py get dependencies.mallocmc.hash)
SOURCE_STAMP="build/${EXAMPLE}/${ALGORITHM}/.source-stamp"

# The machine's toolchain (modules, spack packages) is loaded once here, so
# the header records the compiler and modules the run will use; the
# re-source by run_folder.sh then only re-affirms the same environment.
# shellcheck disable=SC1090
source "$PROFILE"
if command -v "${CXX:-cc}" >/dev/null 2>&1; then
  CXX_FIRST="$(${CXX:-cc} --version 2>/dev/null | head -n 1 || true)"
else
  CXX_FIRST=unavailable
fi
GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | tr '\n' ', ' | sed 's/, $//')"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_${MACHINE}_${EXAMPLE}_${ALGORITHM}_m${MALLOC_DELAY}_f${FREE_DELAY}_r${REP}_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"

echo "Running ${EXAMPLE}/${ALGORITHM} m=${MALLOC_DELAY} ns, f=${FREE_DELAY} ns, rep ${REP}/${REPEATS} [${MACHINE}]"
echo "  log: $LOG"

{
  echo "# run: ${EXAMPLE}/${ALGORITHM}, malloc delay ${MALLOC_DELAY} ns, free delay ${FREE_DELAY} ns, rep ${REP}/${REPEATS} [${MACHINE}]"
  echo "# machine: ${MACHINE}"
  echo "# started: $(date --rfc-3339=seconds)"
  echo "# commit: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "# picongpu: ${PIC_PIN:0:8} (pinned)"
  echo "# mallocmc: ${MC_PIN:0:8} (pinned)"
  if [ -f "$SOURCE_STAMP" ]; then
    echo "# source: $(tr '\n' ' ' <"$SOURCE_STAMP")"
  fi
  echo "# binary: $(sha256sum "$BIN" | awk '{print $1}')  $BIN"
  echo "# host: $(hostname 2>/dev/null || echo unavailable)"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "# slurm_job: ${SLURM_JOB_ID}"
    if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
      echo "# slurm_nodes: ${SLURM_JOB_NODELIST}"
    fi
  fi
  echo "# gpu: ${GPU_INFO:-unavailable}"
  echo "# modules: ${LOADEDMODULES:-unavailable}"
  echo "# cxx: ${CXX_FIRST}"
  echo
} >"$LOG"

bash run_folder.sh "build/${EXAMPLE}/${ALGORITHM}" "flags/${EXAMPLE}.flags" "$PROFILE" "$MALLOC_DELAY" "$FREE_DELAY" >>"$LOG" 2>&1

STAMP_DIR="run-stamps/${MACHINE}/${EXAMPLE}/${ALGORITHM}/${MALLOC_DELAY}_${FREE_DELAY}"
mkdir -p "$STAMP_DIR"
printf '%s\n' "$LOG" >"$STAMP_DIR/rep-${REP}.stamp"
