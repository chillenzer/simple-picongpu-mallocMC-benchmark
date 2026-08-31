"""Audit the raw microbenchmark CSVs for missing x-values and their causes.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Walks every run's per-allocator performance CSVs (under the `microbench.data`
path of `config.json`, one `results-<jobid>/` directory per run) and reports,
per file, which x-values (allocation size / range / thread count) are missing
and why: a timeout marker, a bare (crashed) partial line, or an absent file.

Usage:
    python3 analysis/audit_microbench.py                # `make microbench-audit`
    python3 analysis/audit_microbench.py <data-dir>     # a custom raw data root
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TIMEOUT_MARK = "Ran longer than"
ARROW = "----->"
# The five per-operation statistics (mean, std-dev, min, max, median; mirrors
# make_microbench.STAT_COLUMN_NAMES) that a valid result row appends after x.
N_STATS = 5

# The three allocation tests and their per-allocator CSV filename patterns
# (from make_microbench.py).
TESTS = [
    (
        "alloc_cost",
        "performance",
        re.compile(r"^perf_(?P<op>alloc|free)_(?P<alloc>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"),
    ),
    (
        "alloc_cost_mixed",
        "mixed_performance",
        re.compile(r"^perf_mixed_(?P<op>alloc|free)_(?P<alloc>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"),
    ),
    (
        "alloc_cost_scaling",
        "scaling",
        re.compile(r"^scale_(?P<op>alloc|free)_(?P<alloc>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"),
    ),
]


def load_config() -> dict:
    """Load the harness configuration.

    Returns:
        dict: the parsed `config.json`.

    """
    with (REPO_ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_number(text: str) -> bool:
    """Whether a string parses as a number.

    Args:
        text: the string to test.

    Returns:
        bool: True when `text` is a number.

    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def classify(line: str) -> tuple[str, str | None]:
    """Classify one raw data line of a per-allocator CSV.

    A valid result row is an x value (an integer, or a "lo-hi" range) followed
    by exactly the five statistics, all numeric. A line that carries the suite's
    timeout marker is a timeout; a bare or partial line ("<x>,", the C++ process
    died before appending its stats) is a crash.

    Args:
        line: the CSV line to classify.

    Returns:
        tuple: (kind, x), where kind is one of "ok", "timeout", "crash",
        "other", or "empty" (a blank line) and x is the line's first field.

    """
    line = line.strip()
    if not line:
        return ("empty", None)
    parts = line.split(",")
    prefix = parts[0]
    if len(parts) == 1 + N_STATS and all(_is_number(value) for value in parts[1:]):
        return ("ok", prefix)
    if TIMEOUT_MARK in line or ARROW in line:
        return ("timeout", prefix)
    if len(parts) < 1 + N_STATS:
        return ("crash", prefix)
    return ("other", prefix)


def audit_file(path: Path) -> dict:
    """Audit one per-allocator CSV, bucketing every data line by its cause.

    Args:
        path: the CSV to audit.

    Returns:
        dict: the file's header and its data lines, bucketed under the keys
        "ok", "timeout", "crash" and "other" (each a list of (x, line)).

    """
    lines = path.read_text(encoding="utf-8").splitlines()
    buckets: dict[str, list[tuple[str | None, str]]] = {"ok": [], "timeout": [], "crash": [], "other": []}
    for line in lines[1:]:
        kind, x = classify(line)
        if kind in buckets:
            buckets[kind].append((x, line))
    return {
        "header": lines[0] if lines else "",
        "n_data": max(len(lines) - 1, 0),
        "ok": buckets["ok"],
        "timeout": buckets["timeout"],
        "crash": buckets["crash"],
        "other": buckets["other"],
    }


def _report_file(path: Path, test: str, total: dict) -> None:
    """Print one per-allocator CSV's audit line and its dropped rows, updating totals.

    Args:
        path: the CSV that was audited.
        test: the test's short name (for the report prefix).
        total: the running grand totals, updated in place.

    """
    audit = audit_file(path)
    total["files"] += 1
    for key in ("ok", "timeout", "crash", "other"):
        total[key] += len(audit[key])
    flag = "" if (audit["timeout"] or audit["crash"] or audit["other"]) else "  <-- MISSING"
    print(
        f"  [{test}] {path.name}: {len(audit['ok'])} ok, "
        f"{len(audit['timeout'])} timeout, {len(audit['crash'])} crash, "
        f"{len(audit['other'])} other{flag}"
    )
    for x, line in audit["timeout"]:
        print(f"      timeout   x={x}: {line[:80]}")
    for x, line in audit["crash"]:
        print(f"      crash     x={x}: {line[:60]!r}")
    for x, line in audit["other"]:
        print(f"      other     x={x}: {line[:60]!r}")


def _report_run(run: Path, total: dict) -> None:
    """Print the per-file report of one run, accumulating the running totals.

    Args:
        run: the `results-<jobid>` directory of one run.
        total: the running grand totals, updated in place.

    """
    print(f"\n=== run {run.name} ===")
    for test, subdir, file_re in TESTS:
        perf_dir = run / "tests" / "alloc_tests" / "results" / subdir
        if not perf_dir.is_dir():
            print(f"  [{test}] ABSENT directory: {perf_dir.relative_to(run)}")
            continue
        for path in sorted(perf_dir.glob("*.csv")):
            if file_re.match(path.name) is None:
                print(f"  [{test}] UNRECOGNIZED file name: {path.name}")
                continue
            _report_file(path, test, total)


def main(argv: list[str] | None = None) -> int:
    """Audit the raw microbenchmark CSVs and report the missing x-values.

    Args:
        argv: the command-line arguments (after the program name).

    Returns:
        int: the process exit code (0 on success).

    """
    data = REPO_ROOT / load_config().get("microbench", {}).get("data", "")
    parser = argparse.ArgumentParser(
        description="Audit the raw microbenchmark CSVs for missing x-values and their causes."
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=data,
        help="the suite's raw data directory (default: the microbench.data path of config.json)",
    )
    args = parser.parse_args(argv)
    data_dir = args.data

    total = {"files": 0, "ok": 0, "timeout": 0, "crash": 0, "other": 0}
    runs = sorted(p for p in data_dir.glob("results-*") if p.is_dir()) if data_dir.is_dir() else []
    if not runs:
        print(f"no results-* run directories under {data_dir}", file=sys.stderr)
        return 1
    for run in runs:
        _report_run(run, total)
    missing = total["timeout"] + total["crash"] + total["other"]
    print("\n=== TOTALS ===")
    print(
        f"files={total['files']}  ok={total['ok']}  timeout={total['timeout']}  "
        f"crash={total['crash']}  other={total['other']}  missing={missing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
