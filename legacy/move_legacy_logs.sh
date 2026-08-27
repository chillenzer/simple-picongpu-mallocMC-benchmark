#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# One-time relocation of the pre-redesign benchmark logs into legacy/logs/,
# where legacy/make_legacy_results.py freezes them (make legacy-results).
#
#   - the pure legacy per-cluster output directories are moved whole;
#   - the output directories of the sweep machines configured in
#     config.json are carved out: only the legacy-era files are moved
#     (a file is new format if and only if it carries the `# metadata:`
#     JSON line of the redesigned logs), so the directories stay in place
#     for the new runs;
#   - new-format files are never touched, and already moved directories
#     are left alone, so re-running the script is a no-op.
#
# Run it from any directory; it operates on the repository root.

set -e

cd "$(dirname "$0")/.."

mkdir -p legacy/logs

# The legacy per-cluster directories of the frozen hardware map (see
# make_legacy_results.py), moved whole.
for dir in hal hemera hemera-a100 hemera-v100 lumi jedi hal-sleeptimes-nanosleep; do
  if [ -d "output/$dir" ]; then
    mv "output/$dir" "legacy/logs/$dir"
    echo "moved output/$dir -> legacy/logs/$dir"
  fi
done

# The sweep machines' output directories: move the legacy-era files only.
while IFS= read -r outdir; do
  name=$(basename "$outdir")
  if [ ! -d "$outdir" ]; then
    continue
  fi
  mkdir -p "legacy/logs/$name"
  moved=0
  for file in "$outdir"/*; do
    [ -f "$file" ] || continue
    if ! grep -q '^# metadata: ' "$file"; then
      mv "$file" "legacy/logs/$name/"
      moved=$((moved + 1))
    fi
  done
  echo "carved $outdir: $moved legacy-era file(s) -> legacy/logs/$name"
done < <(python3 -c "import json; [print(m['output']) for m in json.load(open('config.json'))['machines'].values()]")

echo "done: freeze the moved logs with make legacy-results"
