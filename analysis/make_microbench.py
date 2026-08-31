"""Freeze the microbenchmark results into the frozen microbench table.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The microbenchmark suite (the microbenchmarks/memmansurvey submodule, the
`microbench` section of `config.json`) measures the native per-call cost of
the GPU device allocators with no imposed delay: the `alloc_tests`
performance runs time every allocation (and every free) at each allocation
size and report the mean and the spread per size, in milliseconds per
operation (the host-side timer, kernel launches included). The suite's raw
result CSVs live under `microbenchmarks/data/` (git-ignored, on the
benchmark machine, one `results-<jobid>/` directory per run; the layout is
the suite's, see microbenchmarks/README.md) and are frozen once into this
HDF5 file so the main analysis (analysis/compute_results.py) can read the
native per-call costs from a stable table.

Input: for every run of `microbench.runs` (the jobid and the hardware it
ran on), the per-allocator CSVs

    data/results-<jobid>/tests/alloc_tests/results/performance/
        perf_<alloc|free>_<Allocator>_<num>_<lo>-<hi>.csv

(header `AllocationSize (in Byte), mean, std-dev, min, max, median`, one
row per allocation size). A run whose directory is absent is skipped with
a note (a partial freeze is legitimate); a line that is not a numeric
result row (the suite's timeout marker) is dropped.

Output: the frozen file (the `microbench.frozen` path of `config.json`)
with

- alloc_cost: one row per (run, allocator, operation, size) with the
  per-operation statistics in milliseconds,
- file attributes: created_utc, git_commit, sources (one SHA-256 per
  input file), runs (jobid -> hardware), protocol (per (jobid, operation):
  the allocation count and the size range of the run), missing (the run
  directories that were absent).

Usage:
    python3 analysis/make_microbench.py           # `make microbench-results`
    python3 analysis/make_microbench.py --check   # `make microbench-verify`:
                                                  # reparse and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- used to read the git commit
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from results_io import load_results, read_attrs, read_table, write_results

REPO_ROOT = Path(__file__).resolve().parent.parent
# One row of a performance CSV: the allocation size and the five
# statistics (mean, std-dev, min, max, median), all numeric.
RESULT_ROW_RE = re.compile(r"^\d+,(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?,){4}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")
# The file name of a per-allocator performance CSV (the suite's layout).
PERF_FILE_RE = re.compile(r"^perf_(?P<operation>alloc|free)_(?P<allocator>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$")
# The CSV column names, mapped to the frozen table's column names (the
# unit of every time column is the suite's milliseconds per operation).
COLUMNS = {
    "AllocationSize (in Byte)": "size_bytes",
    "mean": "mean_ms",
    "std-dev": "std_ms",
    "min": "min_ms",
    "max": "max_ms",
    "median": "median_ms",
}
ALLOC_COST_COLUMNS = [
    "jobid",
    "hardware",
    "allocator",
    "operation",
    "size_bytes",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "median_ms",
]


def repo_root() -> Path:
    """Return the repository root (the parent of the `analysis/` directory).

    Returns:
        Path: the repository root.

    """
    return REPO_ROOT


def load_config() -> dict:
    """Load the harness configuration.

    Returns:
        dict: the parsed `config.json`.

    """
    with (repo_root() / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_entries() -> list[tuple[int, str]]:
    """Return the configured microbenchmark runs, in config order.

    Returns:
        list[tuple[int, str]]: the (jobid, hardware) pairs.

    """
    runs = load_config().get("microbench", {}).get("runs", [])
    return [(int(run["jobid"]), str(run["hardware"])) for run in runs if isinstance(run, dict) and "jobid" in run]


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, hex encoded.

    Args:
        path: the file to hash.

    Returns:
        str: the hex digest.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perf_files(job_dir: Path) -> list[tuple[Path, str, str, int, int, int]]:
    """Return the job's per-allocator performance CSVs, in sorted name order.

    Args:
        job_dir: the `results-<jobid>` directory of one run.

    Returns:
        list[tuple[Path, str, str, int, int, int]]: (path, operation,
        allocator, num, lo, hi) per parseable CSV.

    """
    perf_dir = job_dir / "tests" / "alloc_tests" / "results" / "performance"
    if not perf_dir.is_dir():
        return []
    entries = []
    for path in sorted(perf_dir.glob("perf_*.csv")):
        match = PERF_FILE_RE.match(path.name)
        if match is None:
            continue
        entries.append(
            (
                path,
                match["operation"],
                match["allocator"],
                int(match["num"]),
                int(match["lo"]),
                int(match["hi"]),
            )
        )
    return entries


def _parse_csv(path: Path) -> pd.DataFrame:
    """Parse one per-allocator performance CSV into its statistics rows.

    Lines that are not numeric result rows (the suite's timeout marker)
    are dropped, so a timed-out size simply has no row.

    Args:
        path: the CSV to parse.

    Returns:
        pd.DataFrame: one row per allocation size, the columns
        `size_bytes`, `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `median_ms`.

    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("AllocationSize"):
        return pd.DataFrame(columns=list(COLUMNS.values()))
    text = "\n".join([lines[0]] + [line for line in lines[1:] if RESULT_ROW_RE.match(line)])
    data = pd.read_csv(StringIO(text), skipinitialspace=True)
    return data.rename(columns=COLUMNS)[list(COLUMNS.values())]


def _git_commit() -> str:
    """Return the short git commit of the repository, empty when unavailable.

    Returns:
        str: the short commit hash, or "".

    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except OSError, subprocess.CalledProcessError:
        return ""


def collect() -> tuple[pd.DataFrame, list[str], dict[int, str], dict[str, dict[str, object]], list[int]]:
    """Parse every configured run's performance CSVs into the frozen table.

    Returns:
        tuple: (the alloc_cost table, the source manifest (one
        "results-<jobid>/<relative path>:sha256" entry per file, in the
        frozen table order), the runs map (jobid -> hardware, the frozen
        subset), the protocol map ("jobid/operation" -> {num, range}), the
        jobids whose directory was absent).

    """
    data_dir = repo_root() / load_config().get("microbench", {}).get("data", "")
    frames: list[pd.DataFrame] = []
    manifest: list[str] = []
    runs: dict[int, str] = {}
    protocol: dict[str, dict[str, object]] = {}
    missing: list[int] = []
    for jobid, hardware in _run_entries():
        job_dir = data_dir / f"results-{jobid}"
        files = _perf_files(job_dir) if job_dir.is_dir() else []
        if not files:
            print(f"microbench: no performance CSVs under {job_dir}; run {jobid} not frozen", file=sys.stderr)
            missing.append(jobid)
            continue
        runs[jobid] = hardware
        frame = pd.DataFrame()
        for path, operation, allocator, num, lo, hi in files:
            rows = _parse_csv(path)
            if rows.empty:
                continue
            manifest.append(f"results-{jobid}/{path.relative_to(job_dir)}:{_sha256(path)}")
            protocol.setdefault(f"{jobid}/{operation}", {"num": num, "range": f"{lo}-{hi}"})
            frame = pd.concat(
                [
                    frame,
                    rows.assign(jobid=jobid, hardware=hardware, allocator=allocator, operation=operation),
                ],
                ignore_index=True,
            )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=ALLOC_COST_COLUMNS), manifest, runs, protocol, missing
    table = pd.concat(frames, ignore_index=True)[ALLOC_COST_COLUMNS]
    return table, manifest, runs, protocol, missing


def read_alloc_cost(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a frozen file back: the alloc_cost table and the top-level attributes.

    Args:
        path: the frozen file.

    Returns:
        tuple: (the table in stored column order, the attribute name ->
        value map).

    """
    with load_results(path) as file:
        return read_table(file, "alloc_cost"), read_attrs(file)


def verify(output: Path) -> int:
    """Reparse the raw CSVs and compare against the frozen file.

    Args:
        output: the frozen file to verify.

    Returns:
        int: the process exit code (0 when the file matches the raw data).

    """
    if not output.is_file():
        print(f"no frozen file at {output}; run `make microbench-results` first", file=sys.stderr)
        return 1
    table, manifest, runs, protocol, _missing = collect()
    stored, attrs = read_alloc_cost(output)
    problems = []
    if str(attrs.get("sources", "")) != "; ".join(manifest):
        problems.append(
            "source manifest differs (the SHA-256 of one or more input files "
            f"changed: {len(manifest)} files now, "
            f"{len(str(attrs.get('sources', '')).split('; '))} stored)"
        )
    try:
        pd.testing.assert_frame_equal(table, stored.reset_index(drop=True), check_dtype=False, check_index_type=False)
    except AssertionError as error:
        problems.append(f"alloc_cost table differs (a CSV changed or the frozen parser drifted): {error}")
    if str(attrs.get("runs", "")) != json.dumps({str(key): value for key, value in runs.items()}, sort_keys=True):
        problems.append("runs attribute differs (a configured run is now frozen or not)")
    if str(attrs.get("protocol", "")) != json.dumps(protocol, sort_keys=True):
        problems.append("protocol attribute differs (the allocation count or size range of a run changed)")
    if problems:
        for problem in problems:
            print(f"microbench-verify: {problem}", file=sys.stderr)
        return 1
    print(f"microbench-verify: OK ({len(table)} rows, {len(manifest)} files, runs: {sorted(runs) or 'none'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build (or verify) the frozen microbenchmark results file.

    Args:
        argv: the command-line arguments (after the program name).

    Returns:
        int: the process exit code.

    """
    from datetime import UTC, datetime

    config = load_config()
    microbench = config.get("microbench", {})
    default_output = repo_root() / microbench.get("frozen", "")
    parser = argparse.ArgumentParser(description="Freeze the microbenchmark results into the frozen table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="destination file (default: %(default)s, the microbench.frozen path of config.json)",
    )
    parser.add_argument("--check", action="store_true", help="reparse the raw CSVs and compare against the existing file")
    args = parser.parse_args(argv)
    if args.check:
        return verify(args.output)
    table, manifest, runs, protocol, missing = collect()
    if manifest and table.empty:
        print("microbench: the performance CSVs hold no numeric result rows; nothing to freeze", file=sys.stderr)
        return 1
    if not manifest:
        data = repo_root() / microbench.get("data", "")
        print(f"microbench: no performance CSVs under {data}; nothing to freeze", file=sys.stderr)
        return 1
    attrs = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "sources": "; ".join(manifest),
        "runs": json.dumps({str(key): value for key, value in runs.items()}, sort_keys=True),
        "protocol": json.dumps(protocol, sort_keys=True),
        "missing": json.dumps(missing, sort_keys=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_results(args.output, {"alloc_cost": table}, attrs)
    summary = f"{len(table)} rows, {len(manifest)} files, runs: {sorted(runs)}, missing: {missing or 'none'}"
    print(f"wrote {args.output} ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
