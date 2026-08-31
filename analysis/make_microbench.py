"""Freeze the microbenchmark results into the frozen microbench table.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The microbenchmark suite (the microbenchmarks/memmansurvey submodule, the
`microbench` section of `config.json`) measures the native per-call cost of
the GPU device allocators with no imposed delay: the `alloc_tests` runs time
every allocation (and every free) and report the mean and the spread, in
milliseconds per operation (the host-side timer, kernel launches included).
Three allocation tests are frozen, each as its own table:

- alloc_cost (the `performance/` runs): the time per operation at each
  allocation size (the x column is `size_bytes`),
- alloc_cost_mixed (the `mixed_performance/` runs): the time per operation
  over each allocation size range (the x column is the `range`, "lo-hi"),
- alloc_cost_scaling (the `scaling/` runs): the time per operation at each
  thread count for a fixed allocation size (the x column is `num_threads`,
  `num_bytes` is the fixed size, from the file name).

The suite's raw result CSVs live under `microbenchmarks/data/` (git-ignored,
on the benchmark machine, one `results-<jobid>/` directory per run; the layout
is the suite's, see microbenchmarks/README.md) and are frozen once into this
HDF5 file so the main analysis (analysis/compute_results.py) can read the
native per-call costs from stable tables and the figure scripts
(analysis/plot_microbench.py) can draw them.

Input: for every run of `microbench.runs` (the jobid and the hardware it ran
on), the per-allocator CSVs under
`data/results-<jobid>/tests/alloc_tests/results/<performance|mixed_performance|scaling>/`,
one row per x value and the five statistics (mean, std-dev, min, max, median).
A run whose directory is absent is skipped with a note (a partial freeze is
legitimate); a line that is not a numeric result row (the suite's timeout
marker) is dropped.

Output: the frozen file (the `microbench.frozen` path of `config.json`) with

- alloc_cost, alloc_cost_mixed, alloc_cost_scaling: one row per (run,
  allocator, operation, x value) with the per-operation statistics,
- file attributes: created_utc, git_commit, sources (one SHA-256 per input
  file), runs (jobid -> hardware), protocol (per (jobid, test): the file
  count and the operations), missing (the run directories that were absent).

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
from collections.abc import Iterator
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
from results_io import load_results, read_attrs, read_table, write_results

REPO_ROOT = Path(__file__).resolve().parent.parent
# The five per-operation statistics (the unit of every time column is the
# suite's milliseconds per operation), CSV name -> frozen column name.
STAT_COLUMNS = {
    "mean": "mean_ms",
    "std-dev": "std_ms",
    "min": "min_ms",
    "max": "max_ms",
    "median": "median_ms",
}
STAT_COLUMN_NAMES = list(STAT_COLUMNS.values())
# The columns every frozen table carries to identify a row (set at freeze).
IDENTITY_COLUMNS = ("jobid", "hardware", "allocator", "operation")
ALLOC_COST_COLUMNS = ["jobid", "hardware", "allocator", "operation", "size_bytes", *STAT_COLUMN_NAMES]
ALLOC_COST_MIXED_COLUMNS = ["jobid", "hardware", "allocator", "operation", "range", *STAT_COLUMN_NAMES]
ALLOC_COST_SCALING_COLUMNS = [
    "jobid",
    "hardware",
    "allocator",
    "operation",
    "num_bytes",
    "num_threads",
    *STAT_COLUMN_NAMES,
]
# The three allocation tests: the perf CSV directory, the per-allocator file
# name, the x column's CSV header and frozen name, and the table columns.
TESTS = [
    {
        "name": "alloc_cost",
        "subdir": "performance",
        "file_re": re.compile(
            r"^perf_(?P<operation>alloc|free)_(?P<allocator>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"
        ),
        "x_header": "AllocationSize (in Byte)",
        "x_column": "size_bytes",
        "x_dtype": int,
        "columns": ALLOC_COST_COLUMNS,
    },
    {
        "name": "alloc_cost_mixed",
        "subdir": "mixed_performance",
        "file_re": re.compile(
            r"^perf_mixed_(?P<operation>alloc|free)_(?P<allocator>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"
        ),
        "x_header": "AllocationRange (in Byte)",
        "x_column": "range",
        "x_dtype": str,
        "columns": ALLOC_COST_MIXED_COLUMNS,
    },
    {
        "name": "alloc_cost_scaling",
        "subdir": "scaling",
        # The file name carries the fixed allocation size in the `num` field
        # (frozen as num_bytes); the x column (the thread count) is per row.
        "file_re": re.compile(
            r"^scale_(?P<operation>alloc|free)_(?P<allocator>.+)_(?P<num>\d+)_(?P<lo>\d+)-(?P<hi>\d+)\.csv$"
        ),
        "x_header": "NumThreads",
        "x_column": "num_threads",
        "x_dtype": int,
        "columns": ALLOC_COST_SCALING_COLUMNS,
    },
]
TEST_BY_NAME = {test["name"]: test for test in TESTS}


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


def _is_result_row(line: str) -> bool:
    """Whether a CSV line is a numeric result row (x value plus the stats).

    The x value may be an integer (a size or a thread count) or a range
    ("lo-hi"); only the trailing statistics must be numeric. This drops the
    suite's timeout marker lines.

    Args:
        line: the CSV line to test.

    Returns:
        bool: True for a numeric result row.

    """
    parts = line.split(",")
    if len(parts) != 1 + len(STAT_COLUMN_NAMES):
        return False
    return all(_is_number(value) for value in parts[1:])


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


def _test_files(job_dir: Path, spec: dict) -> Iterator[tuple[Path, str, str, int]]:
    """Yield the job's per-allocator CSVs of one test, in sorted name order.

    Args:
        job_dir: the `results-<jobid>` directory of one run.
        spec: the test's spec (its subdir and file name pattern).

    Yields:
        tuple: (path, operation, allocator, num) per parseable CSV.

    """
    perf_dir = job_dir / "tests" / "alloc_tests" / "results" / spec["subdir"]
    if not perf_dir.is_dir():
        return
    for path in sorted(perf_dir.glob("*.csv")):
        match = spec["file_re"].match(path.name)
        if match is not None:
            yield path, match["operation"], match["allocator"], int(match["num"])


def _parse_perf_csv(path: Path, spec: dict, num: int) -> pd.DataFrame:
    """Parse one per-allocator perf CSV into its statistics rows.

    Lines that are not numeric result rows (the suite's timeout marker) are
    dropped, so a timed-out x value simply has no row.

    Args:
        path: the CSV to parse.
        spec: the test's spec (its x column and the table columns).
        num: the file name's `num` field (the fixed allocation size for the
        scaling test, frozen as `num_bytes`).

    Returns:
        pd.DataFrame: one row per x value, the table's non-identity columns.

    """
    x_header, x_column, x_dtype = spec["x_header"], spec["x_column"], spec["x_dtype"]
    data_cols = [column for column in spec["columns"] if column not in IDENTITY_COLUMNS]
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith(x_header):
        return pd.DataFrame(columns=data_cols)
    text = "\n".join([lines[0]] + [line for line in lines[1:] if _is_result_row(line)])
    data = pd.read_csv(StringIO(text), skipinitialspace=True)
    data = data.rename(columns={x_header: x_column, **STAT_COLUMNS})
    for column in data_cols:
        if column == x_column:
            data[column] = data[column].astype(x_dtype)
        elif column == "num_bytes":
            data[column] = num
    return data[data_cols]


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


def _empty_table(name: str) -> pd.DataFrame:
    """Return an empty table with one test's columns.

    Args:
        name: the table's (test's) name.

    Returns:
        pd.DataFrame: an empty table with the test's columns.

    """
    return pd.DataFrame(columns=TEST_BY_NAME[name]["columns"])


def collect() -> tuple[dict[str, pd.DataFrame], list[str], dict[int, str], dict[str, dict], list[int]]:
    """Parse every configured run's perf CSVs into the frozen tables.

    Returns:
        tuple: (the frozen tables, by test name, the source manifest (one
        "results-<jobid>/<relative path>:sha256" entry per file, in the frozen
        tables' order), the runs map (jobid -> hardware, the frozen subset),
        the protocol map ("jobid/test" -> {files, operations}), the jobids
        whose directory was absent or held no perf CSVs).

    """
    data_dir = repo_root() / load_config().get("microbench", {}).get("data", "")
    frames: dict[str, list[pd.DataFrame]] = {test["name"]: [] for test in TESTS}
    manifest: list[str] = []
    runs: dict[int, str] = {}
    protocol: dict[str, dict] = {}
    missing: list[int] = []
    for jobid, hardware in _run_entries():
        job_dir = data_dir / f"results-{jobid}"
        if not job_dir.is_dir():
            print(f"microbench: no data under {job_dir}; run {jobid} not frozen", file=sys.stderr)
            missing.append(jobid)
            continue
        runs[jobid] = hardware
        frozen_any = False
        for spec in TESTS:
            files = list(_test_files(job_dir, spec))
            if not files:
                continue
            frozen_any = True
            tframes: list[pd.DataFrame] = []
            for path, operation, allocator, num in files:
                rows = _parse_perf_csv(path, spec, num)
                if rows.empty:
                    continue
                manifest.append(f"results-{jobid}/{path.relative_to(job_dir)}:{_sha256(path)}")
                entry = protocol.setdefault(f"{jobid}/{spec['name']}", {"files": 0, "operations": set()})
                entry["files"] += 1
                entry["operations"].add(operation)
                tframes.append(
                    rows.assign(jobid=jobid, hardware=hardware, allocator=allocator, operation=operation)[
                        spec["columns"]
                    ]
                )
            if tframes:
                frames[spec["name"]].append(pd.concat(tframes, ignore_index=True))
        if not frozen_any:
            print(f"microbench: no perf CSVs under {job_dir}; run {jobid} not frozen", file=sys.stderr)
            missing.append(jobid)
    for value in protocol.values():
        value["operations"] = sorted(value["operations"])
    tables = {
        name: (pd.concat(chunk, ignore_index=True) if chunk else _empty_table(name)) for name, chunk in frames.items()
    }
    return tables, manifest, runs, protocol, missing


def read_tables(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read a frozen file back: the frozen tables and the top-level attributes.

    Args:
        path: the frozen file.

    Returns:
        tuple: (the tables, by test name, in stored column order, and the
        attribute name -> value map).

    """
    with load_results(path) as file:
        tables = {
            spec["name"]: (read_table(file, spec["name"]) if spec["name"] in file else _empty_table(spec["name"]))
            for spec in TESTS
        }
        return tables, read_attrs(file)


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
    tables, manifest, runs, protocol, _missing = collect()
    stored, attrs = read_tables(output)
    problems = []
    if str(attrs.get("sources", "")) != "; ".join(manifest):
        problems.append(
            "source manifest differs (the SHA-256 of one or more input files "
            f"changed: {len(manifest)} files now, "
            f"{len(str(attrs.get('sources', '')).split('; '))} stored)"
        )
    for spec in TESTS:
        try:
            pd.testing.assert_frame_equal(
                tables[spec["name"]],
                stored[spec["name"]].reset_index(drop=True),
                check_dtype=False,
                check_index_type=False,
            )
        except AssertionError as error:
            problems.append(f"{spec['name']} table differs (a CSV changed or the frozen parser drifted): {error}")
    if str(attrs.get("runs", "")) != json.dumps({str(key): value for key, value in runs.items()}, sort_keys=True):
        problems.append("runs attribute differs (a configured run is now frozen or not)")
    if str(attrs.get("protocol", "")) != json.dumps(protocol, sort_keys=True):
        problems.append("protocol attribute differs (a run's test coverage changed)")
    if problems:
        for problem in problems:
            print(f"microbench-verify: {problem}", file=sys.stderr)
        return 1
    rows = sum(len(table) for table in tables.values())
    print(f"microbench-verify: OK ({rows} rows, {len(manifest)} files, runs: {sorted(runs) or 'none'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build (or verify) the frozen microbenchmark results file.

    Args:
        argv: the command-line arguments (after the program name).

    Returns:
        int: the process exit code.

    """
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
    parser.add_argument(
        "--check", action="store_true", help="reparse the raw CSVs and compare against the existing file"
    )
    args = parser.parse_args(argv)
    if args.check:
        return verify(args.output)
    tables, manifest, runs, protocol, missing = collect()
    if manifest and all(table.empty for table in tables.values()):
        print("microbench: the perf CSVs hold no numeric result rows; nothing to freeze", file=sys.stderr)
        return 0
    if not manifest:
        data = repo_root() / microbench.get("data", "")
        print(f"microbench: no perf CSVs under {data}; nothing to freeze", file=sys.stderr)
        return 0
    attrs = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "sources": "; ".join(manifest),
        "runs": json.dumps({str(key): value for key, value in runs.items()}, sort_keys=True),
        "protocol": json.dumps(protocol, sort_keys=True),
        "missing": json.dumps(missing, sort_keys=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_results(args.output, tables, attrs)
    rows = sum(len(table) for table in tables.values())
    per_test = ", ".join(f"{name}: {len(table)}" for name, table in tables.items())
    print(
        f"wrote {args.output} ({rows} rows [{per_test}], {len(manifest)} files, "
        f"runs: {sorted(runs)}, missing: {missing or 'none'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
