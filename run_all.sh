#!/bin/bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# What to run (examples, algorithms, delay sweep) comes from config.json;
# `check` validates it and the lookups below read the individual values into
# the variables this script already uses. (run_all.sh has no `set -e`, so
# the check's failure must abort explicitly.)
if ! python3 config.py check; then
  echo "run_all.sh: config.json is invalid (see above)" >&2
  exit 1
fi
mapfile -t EXAMPLES < <(python3 config.py list examples)
mapfile -t ALGORITHMS < <(python3 config.py list algorithms)
# The (malloc_delay, free_delay) combinations are in nanoseconds and injected
# at run time by mallocMC through the MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY environment variables (run_folder.sh sets them), so one
# build per example serves the whole sweep. The sweep values themselves (the
# single-delay arms and the joint grid) live in config.json, and the design
# of the sweep is documented in the README.
mapfile -t DELAY_SWEEP < <(python3 config.py list delays.arms.values)
mapfile -t JOINT < <(python3 config.py list delays.joint.values)
mapfile -t BASELINE < <(python3 config.py list delays.baseline)

COMBINATIONS=("${BASELINE[0]} ${BASELINE[1]}")
for delay in "${DELAY_SWEEP[@]}"; do
  COMBINATIONS+=("$delay 0" "0 $delay")
done
for malloc_delay in "${JOINT[@]}"; do
  for free_delay in "${JOINT[@]}"; do
    COMBINATIONS+=("$malloc_delay $free_delay")
  done
done

if [ "${1:-}" = "--check" ]; then
  # Print the resolved run matrix and stop without running anything; used to
  # validate the configuration before a long benchmark.
  printf 'examples:     %s\n' "${EXAMPLES[*]}"
  printf 'algorithms:   %s\n' "${ALGORITHMS[*]}"
  printf 'combinations: %d\n' "${#COMBINATIONS[@]}"
  printf '%s\n' "${COMBINATIONS[@]}"
  exit 0
fi

PROFILE=$1
FLAGSFOLDER=$(pwd -P)/$2

echo "All combinations:"
echo "${COMBINATIONS[@]}"
for example in "${EXAMPLES[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    # Quoted: the combinations contain spaces ("100 0"), and unquoted
    # ${COMBINATIONS[@]} word-splits each of them into bare numbers, so the
    # free delay is lost and defaults to 0.
    for combination in "${COMBINATIONS[@]}"; do
      read -r malloc_delay free_delay <<<"$combination"
      echo "=============================="
      echo "Running example: $example"
      echo "Using allocator: $algorithm, malloc delay: $malloc_delay ns, free delay: $free_delay ns"
      echo "=============================="
      bash run_folder.sh "build/$example/$algorithm" "$FLAGSFOLDER/${example}.flags" "$PROFILE" "$malloc_delay" "$free_delay"
      echo ""
    done
  done
done
