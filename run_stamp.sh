#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# Worker of the Makefile run targets. One invocation is one run: the
# example's whole flags file once, from the matching build, with one (malloc
# delay, free delay) combination in the environment.
#
#   run_stamp.sh <machine> <repeats> <example> <algorithm> <malloc-ns> <free-ns> <rep>
#
# The run is recorded in one self-contained log per grid run (one line of
# the flags file) in the machine's output directory:
#
#   <outdir>/run_<machine>_<Ex>_<Algo>_m<M>_f<F>_r<rep>_<line-sha8>_<time>.txt
#
# Each log carries a two-line header (a human one-liner `# run:` line and
# the self-describing `# metadata:` JSON line, emitted by logmeta.py)
# followed by that grid run's `set -x` trace.
#
# Runs are append-only: re-running a (combination, repetition) writes a new
# vintage of the logs next to the older ones, nothing is removed, and the
# new logs' metadata (the commit, the binary's sha256, the flags line) is
# what tells the vintages apart in the analysis. Its success is stamped in
# run-stamps/<machine>/<example>/<algorithm>/<malloc>_<free>/rep-<rep>.stamp
# (content: one log path per line), which therefore also points to the
# run's current vintage. An existing stamp is what makes `make runs` skip
# the run.

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
FLAGSFILE="flags/${EXAMPLE}.flags"

mkdir -p "$OUTDIR"
PREFIX="run_${MACHINE}_${EXAMPLE}_${ALGORITHM}_m${MALLOC_DELAY}_f${FREE_DELAY}_r${REP}_"

# One log per grid run (one line of the flags file); the line numbers
# mirror run_folder.sh's loop over the same file.
mapfile -t LINES <"$FLAGSFILE"
TOTAL=${#LINES[@]}
LOGS=()

for i in "${!LINES[@]}"; do
  LINE_NO=$((i + 1))
  LINE_SHA=$(printf '%s' "${LINES[i]}" | sha256sum | awk '{print $1}' | cut -c1-8)
  LOG="$OUTDIR/${PREFIX}${LINE_SHA}_$(date --rfc-3339=seconds | sed 's/ /_/g').txt"
  # A re-run that lands in the very same second as the previous vintage
  # would reuse its log name; append a counter so no log is clobbered.
  if [ -e "$LOG" ]; then
    SUFFIX=1
    while [ -e "${LOG%.txt}.${SUFFIX}.txt" ]; do
      SUFFIX=$((SUFFIX + 1))
    done
    LOG="${LOG%.txt}.${SUFFIX}.txt"
  fi
  LOGS+=("$LOG")
  echo "Running ${EXAMPLE}/${ALGORITHM} m=${MALLOC_DELAY} ns, f=${FREE_DELAY} ns, rep ${REP}/${REPEATS} line ${LINE_NO}/${TOTAL} [${MACHINE}]"
  echo "  log: $LOG"
  {
    echo "# run: ${EXAMPLE}/${ALGORITHM} m=${MALLOC_DELAY} ns f=${FREE_DELAY} ns rep ${REP}/${REPEATS} line ${LINE_NO}/${TOTAL} [${MACHINE}]"
    python3 logmeta.py run --machine "$MACHINE" --repeats "$REPEATS" --rep "$REP" \
      --example "$EXAMPLE" --algorithm "$ALGORITHM" \
      --malloc-delay "$MALLOC_DELAY" --free-delay "$FREE_DELAY" \
      --line "$LINE_NO" --flags "$FLAGSFILE" --binary "$BIN"
    echo
  } >"$LOG"
  bash run_folder.sh "build/${EXAMPLE}/${ALGORITHM}" "$FLAGSFILE" "$PROFILE" "$MALLOC_DELAY" "$FREE_DELAY" "$LINE_NO" >>"$LOG" 2>&1
done

STAMP_DIR="run-stamps/${MACHINE}/${EXAMPLE}/${ALGORITHM}/${MALLOC_DELAY}_${FREE_DELAY}"
mkdir -p "$STAMP_DIR"
printf '%s\n' "${LOGS[@]}" >"$STAMP_DIR/rep-${REP}.stamp"
