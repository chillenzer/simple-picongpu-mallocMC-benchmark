"""Freeze the legacy benchmark logs into legacy/legacy_results.h5.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The pre-redesign benchmark logs (the historical log layouts, the
directory-to-hardware map of the old comparison charts) are frozen once
into this HDF5 file so the main analysis (analysis/compute_results.py)
can read the legacy runs from a stable table instead of re-parsing the
raw logs with knowledge of every historical layout.

The log parsing below is a frozen copy of the historical behavior of
analysis/run_logs.py (commit 4c43718) and of the LEGACY_HARDWARE map of
analysis/compute_results.py; it must not change, or the frozen file
stops being comparable to the numbers computed before the cut.

Input: the legacy log directories under legacy/logs/ (created by
legacy/move_legacy_logs.sh, which moves them out of the historical
output/ tree). A subdirectory's name sets the run attribution:

- the output directory of a sweep machine from the config machines table
  (e.g. hal-sleeptimes, rosi-sleeptimes): the runs are attributed to the
  machine label and to the short hardware name (the last word of the
  machine's hardware title), as the sweep-machine parsing did;
- a legacy per-cluster directory (the frozen LEGACY_DIRECTORIES map): the
  runs are attributed to machine "" and the map's short hardware name,
  as the old LEGACY_HARDWARE parsing did;
- an archived-but-excluded directory (EXCLUDED_DIRECTORIES): the runs are
  parsed and archived, but attributed to machine "" and hardware "", so
  the main analysis drops them at its single exclusion choke point; the
  reason is recorded in the file attribute excluded_sources.

Output: legacy/legacy_results.h5 with

- runs: the parsed runs, in the same schema as the results file's runs
  table without rep (the main analysis numbers the repetitions in file
  order over the merged table). The first eleven columns are the
  historical schema and their values must not change; beyond them the
  freeze records the per-run metrics (the full and the initialisation
  runtimes, the number of simulation steps) and the provenance of the
  log file each run came from (parsed by log_meta.py from the
  historical log layouts; the empty string where a layout carries
  nothing). The row order is part of the contract (machine directories
  in config order, then the legacy directories in LEGACY_DIRECTORIES
  order, then the excluded directories last): the main analysis numbers
  rep in file order, so reordering rows would renumber the repetitions
  and shift the group statistics.
- file attributes: created_utc, the source manifest (one SHA-256 per
  input file), excluded_sources (the excluded directories and why),
  and attribution (per directory: machine and hardware).

Usage:
    python3 legacy/make_legacy_results.py           # (re)build the file
    python3 legacy/make_legacy_results.py --check   # `make legacy-verify`:
                                                    # reparse and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import h5py
import log_meta
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "legacy" / "logs"
DEFAULT_OUTPUT = REPO_ROOT / "legacy" / "legacy_results.h5"

# Frozen copy of the historical log parsing (analysis/run_logs.py as of
# commit 4c43718): the `set -x` trace of a run_folder.sh run, one record
# per bin/picongpu invocation, across the three historical layouts.
ALGORITHM = "FlatterScatter"
CD_CMD = "+ cd "
RUN_CMD = "bin/picongpu "
MALLOC_DELAY_CMD = "MALLOCMC_MALLOC_DELAY="
FREE_DELAY_CMD = "MALLOCMC_FREE_DELAY="
# Pre-rename logs: the malloc delay was then injected via MALLOCMC_SLEEP_TIME.
LEGACY_MALLOC_DELAY_CMD = "MALLOCMC_SLEEP_TIME="
VARIANT_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)-sleep(\d+)$")
BUILD_ALGO_CD_RE = re.compile(r"(?:^|/)build/(\w+)/(\w+)$")
BUILD_CD_RE = re.compile(r"(?:^|/)build/(\w+)$")
# The `-s` step count of one traced picongpu command (no other traced
# line carries it).
SIM_STEPS_RE = re.compile(r"(?:^|\s)-s (\d+)\b")
# The trailing `= <value> sec` of an `initialization time:` /
# `full simulation time:` line, in the shared format.
TIME_VALUE_RE = re.compile(r"= ([\d.]+) sec")

MALLOC_DELAY = "malloc_sleeptime"
FREE_DELAY = "free_sleeptime"
RUN_TIME = "runtime_s"
# The per-run metrics and the log provenance recorded beyond the
# historical schema: a frozen copy of the RUN_METRIC_COLUMNS and
# RUN_SOURCE_COLUMNS of analysis/results_io.py (the names are the
# contract; keep the two in sync).
METRIC_COLUMNS = ["full_runtime_s", "init_time_s", "sim_steps"]
SOURCE_COLUMNS = [
    "started_utc",
    "commit",
    "binary_sha256",
    "picongpu",
    "mallocmc",
    "gpu",
    "gpu_driver",
    "cuda_version",
    "cpu",
    "compiler",
    "host",
    "slurm_job",
]
# Same schema as the results file's runs table, without rep (which the
# main analysis assigns in file order over the merged table): the
# historical columns in their historical order, then the metrics, then
# the provenance.
RUNS_COLUMNS = [
    "machine",
    "hardware",
    "setup",
    "algorithm",
    "x",
    "y",
    "z",
    MALLOC_DELAY,
    FREE_DELAY,
    "configuration",
    RUN_TIME,
    *METRIC_COLUMNS,
    *SOURCE_COLUMNS,
]

# The legacy per-cluster output directories (directory name -> short
# hardware name of the comparison charts), the entries of the old
# LEGACY_HARDWARE map that are not a sweep machine's output directory, in
# the historical order.
LEGACY_DIRECTORIES: dict[str, str] = {
    "hal": "A30",
    "hemera": "A100",
    "hemera-a100": "A100",
    "hemera-v100": "V100",
    "lumi": "MI250X (1 GCD)",
    "jedi": "GH200",
}
# Archived-but-excluded directories: parsed and stored, but attributed to
# the empty hardware name so the main analysis drops them; the value is
# the reason recorded in the excluded_sources attribute.
EXCLUDED_DIRECTORIES: dict[str, str] = {
    "hal-sleeptimes-nanosleep": (
        "Archived nanosleep experiment; excluded from the benchmark "
        "analysis (as was already the case before the legacy cut)."
    ),
}


def repo_root() -> Path:
    """Return the repository root (the parent of the legacy/ directory).

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


def parse_setup(line: str) -> dict | None:
    """Parse the run context of a `cd` trace line; None if the line is unrelated.

    Returns:
        dict | None: the parsed setup context, or None if the line is unrelated.

    """
    path = line.rsplit(maxsplit=1)[-1]
    m = VARIANT_CD_RE.search(path)
    if m:
        # One build per (example, algorithm, sleeptime): the delay was
        # compiled into the binary (a malloc delay, no free delay).
        return {
            "setup": m[1],
            "algorithm": m[2],
            "malloc_sleeptime": int(m[3]),
            "free_sleeptime": 0,
            "configuration": "compile-time",
        }
    m = BUILD_ALGO_CD_RE.search(path)
    if m:
        # One build per (example, algorithm): the creation policy is
        # compiled into the binary; the delays are still injected at run
        # time via the MALLOCMC_*_DELAY environment variables.
        return {"setup": m[1], "algorithm": m[2]}
    m = BUILD_CD_RE.search(path)
    if m:
        return {"setup": m[1], "algorithm": ALGORITHM}
    return None


def parse_grid(line: str) -> dict[str, int]:
    """Parse the `-g` grid dimensions out of a picongpu command line.

    Returns:
        dict[str, int]: the grid dimensions, keyed by x, y, z.

    """
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            line.split(RUN_CMD, 1)[1].split("-g", 1)[1].split("-", maxsplit=1)[0].strip().split(" "),
            # 2-D grids have only two values; the zip truncates the keys to
            # the dimensions present (the missing one becomes NaN downstream).
            strict=False,
        )
    }


def parse_simulation_time(line: str) -> dict[str, float]:
    """Parse a `calculation  simulation time` line into a runtime dict.

    Returns:
        dict[str, float]: the simulation runtime in seconds.

    """
    return {"runtime in s": float(line.split("=")[1][: -len("sec")])}


def _update_delays(line: str, malloc_delay: int | None, free_delay: int | None) -> tuple[int | None, int | None]:
    """Update the remembered delay env values from a traced `set -x` line.

    Returns:
        tuple[int | None, int | None]: the updated (malloc_delay, free_delay).

    """
    if MALLOC_DELAY_CMD in line:
        malloc_delay = int(line.split(MALLOC_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    elif LEGACY_MALLOC_DELAY_CMD in line:
        malloc_delay = int(line.split(LEGACY_MALLOC_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    if FREE_DELAY_CMD in line:
        free_delay = int(line.split(FREE_DELAY_CMD, 1)[1].split(maxsplit=1)[0])
    return malloc_delay, free_delay


def _flush(pending: dict | None) -> dict | None:
    """Return the pending record when it measured a runtime, else None.

    Args:
        pending: the run's record in progress, or None.

    Returns:
        dict | None: the record, or None when it is absent or the run
        aborted before printing its calculation time.

    """
    if pending is not None and "runtime in s" in pending:
        return pending
    return None


def _time_line(value: str, pending: dict) -> bool:
    """Record one of the run's time lines on the pending record.

    The rosi logs run the combinations on several GPUs at once, so the
    time lines of the concurrent runs interleave; the first value of each
    line keeps the value of the run that is recorded.

    Args:
        value: the stripped log line.
        pending: the run's record in progress.

    Returns:
        bool: True when the line is the run's `full simulation time:`,
        which ends the record.

    """
    if value.startswith("calculation") and "simulation time" in value and "runtime in s" not in pending:
        pending.update(parse_simulation_time(value))
    elif value.startswith("full simulation time"):
        match = TIME_VALUE_RE.search(value)
        if match is not None and "full_runtime_s" not in pending:
            pending["full_runtime_s"] = float(match[1])
        return True
    elif value.startswith("initialization time") and "init_time_s" not in pending:
        match = TIME_VALUE_RE.search(value)
        if match is not None:
            pending["init_time_s"] = float(match[1])
    return False


def _start_run(
    line: str,
    context: dict,
    malloc_delay: int | None,
    free_delay: int | None,
) -> dict | None:
    """Start a pending record at one traced `bin/picongpu` command line.

    Args:
        line: the stripped `set -x` trace line.
        context: the run context of the current `build/...` directory.
        malloc_delay: the remembered malloc delay, or None.
        free_delay: the remembered free delay, or None.

    Returns:
        dict | None: the started record, or None when the line is no
        `bin/picongpu` command of a known context.

    """
    if RUN_CMD not in line or "setup" not in context:
        return None
    # In the run-time layout the env vars override the variant sleeptime;
    # in the per-variant layout they are absent.
    pending = dict(context) | parse_grid(line)
    steps = SIM_STEPS_RE.search(line)
    if steps is not None:
        pending["sim_steps"] = int(steps[1])
    if malloc_delay is not None or free_delay is not None:
        pending |= {
            "malloc_sleeptime": (malloc_delay if malloc_delay is not None else 0),
            "free_sleeptime": (free_delay if free_delay is not None else 0),
            "configuration": "run-time",
        }
    return pending


def parse_log(log_path: Path) -> Iterator[dict]:
    """Yield one record per picongpu run of a single legacy run log.

    The record is yielded when the run's `full simulation time:` line is
    seen (or, if a run ends without that line, at the next run context or
    at the end of the log), so the full and the initialisation runtimes,
    printed right after and long before the calculation time, belong to
    the same record. A run that aborted before printing its calculation
    time (a core-dump, an out-of-memory) carries no runtime and is not
    yielded.

    Yields:
        dict: one record per picongpu run of the log.

    """
    with log_path.open("r", encoding="utf-8") as file:
        context = {}
        pending = None
        malloc_delay = None
        free_delay = None
        for line in map(str.strip, file):
            if line.startswith(CD_CMD):
                # A new run context invalidates the remembered delay values
                # and, in case the run never printed its full time, ends
                # the pending record.
                setup = parse_setup(line)
                if setup is not None:
                    record = _flush(pending)
                    if record is not None:
                        yield record
                    pending = None
                    context = setup
                    malloc_delay = None
                    free_delay = None
            elif line.startswith("+ "):
                malloc_delay, free_delay = _update_delays(line, malloc_delay, free_delay)
                pending = _start_run(line, context, malloc_delay, free_delay) or pending
            elif pending is not None and _time_line(line, pending):
                record = _flush(pending)
                if record is not None:
                    yield record
                pending = None
        record = _flush(pending)
        if record is not None:
            yield record


def _files(log_dir: Path) -> list[Path]:
    """Return the files of one log directory, in sorted name order.

    Args:
        log_dir: the directory to list.

    Returns:
        list[Path]: the directory's files, sorted.

    """
    return sorted(path for path in log_dir.glob("*") if path.is_file())


def _parse_dir(
    log_dir: Path,
    *,
    picongpu_pin: str = "",
    mallocmc_pin: str = "",
    modules_hint: str = "",
) -> pd.DataFrame:
    """Parse one output directory's run logs, in sorted file order.

    The per-file construction mirrors the historical `parse_logs` /
    `run_to_df` exactly (including the `name` column): a zero-record file
    still contributes its `name` column to the concatenation, which is what
    upcasts the integer grid columns to float64 in pandas. That upcast is
    part of the historical values of the results file, so it must be kept.
    Beyond that, the provenance of each log (parsed by `log_meta.py`, its
    `generation` tag dropped) is merged onto all the records of that file.

    Args:
        log_dir: the directory of run logs to parse.
        picongpu_pin: the PIConGPU pin of `config.json`, the fallback
            dependency version of a log that carries none of its own.
        mallocmc_pin: the mallocMC pin of `config.json`, as for
            `picongpu_pin`.
        modules_hint: the machine's modules of `config.json` (" "-joined),
            the last-resort compiler fallback for a log that traces none.

    Returns:
        pd.DataFrame: the parsed runs, or an empty frame.

    """
    log_paths = _files(log_dir)
    if not log_paths:
        return pd.DataFrame()
    frames = []
    for path in log_paths:
        frame = pd.DataFrame(parse_log(path)).assign(name=str(path))
        meta = {
            column: value
            for column, value in log_meta.parse_log_metadata(
                path,
                picongpu_pin=picongpu_pin,
                mallocmc_pin=mallocmc_pin,
                modules_hint=modules_hint,
            ).items()
            if column != "generation"
        }
        frame = frame.assign(**meta)
        for column in METRIC_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frames.append(frame)
    tmp = pd.concat(frames)
    return tmp.assign(z=tmp.get("z", np.nan)).drop(columns=["name"]).rename(columns={"runtime in s": RUN_TIME})


def _machine_dirs() -> list[tuple[str, str, str]]:
    """Return the (output directory name, machine label, hardware) of the sweep machines.

    The hardware is the short name (the last word of the machine's
    hardware title), as the sweep-machine parsing always applied it.

    Returns:
        list[tuple[str, str, str]]: one entry per config machine, in config
        order.

    """
    return [
        (Path(machine["output"]).name, label, str(machine["hardware"]).rsplit(maxsplit=1)[-1])
        for label, machine in load_config()["machines"].items()
    ]


def _ordered_sources(machine_dirs: list[tuple[str, str, str]], present: set[str]) -> list[tuple[str, str, str]]:
    """Order the present source directories in the frozen row order.

    The sweep machines' output directories first (config order), then the
    legacy per-cluster directories (LEGACY_DIRECTORIES order, skipping the
    ones that are a machine output directory), then the excluded
    directories last.

    Args:
        machine_dirs: as from `_machine_dirs`.
        present: the directory names present under legacy/logs/.

    Returns:
        list[tuple[str, str, str]]: (directory name, machine, hardware).

    """
    machine_names = {name for name, _label, _hardware in machine_dirs}
    ordered = [(name, label, hardware) for name, label, hardware in machine_dirs if name in present]
    ordered += [
        (name, "", hardware)
        for name, hardware in LEGACY_DIRECTORIES.items()
        if name in present and name not in machine_names
    ]
    ordered += [(name, "", "") for name in sorted(EXCLUDED_DIRECTORIES) if name in present]
    return ordered


def _attributions() -> list[tuple[str, str, str]]:
    """Return the (directory name, machine label, hardware) of every source.

    Returns:
        list[tuple[str, str, str]]: the present source directories, in the
        frozen row order.

    Raises:
        SystemExit: on a missing legacy/logs/, an unknown directory, or no
        source at all.

    """
    machine_dirs = _machine_dirs()
    if not LOGS_DIR.is_dir():
        print(f"no {LOGS_DIR}; move the legacy logs there first (legacy/move_legacy_logs.sh)", file=sys.stderr)
        raise SystemExit(1)
    present = {path.name for path in LOGS_DIR.iterdir() if path.is_dir()}
    known = {name for name, _label, _hw in machine_dirs} | set(LEGACY_DIRECTORIES) | set(EXCLUDED_DIRECTORIES)
    unknown = sorted(present - known)
    if unknown:
        print(
            f"unknown legacy log directories (not a machine output, legacy or excluded directory): {unknown}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    ordered = _ordered_sources(machine_dirs, present)
    if not ordered:
        print(f"{LOGS_DIR} holds no legacy log directories; nothing to freeze", file=sys.stderr)
        raise SystemExit(1)
    return ordered


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, hex encoded.

    Args:
        path: the file to hash.

    Returns:
        str: the hex digest.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pins() -> tuple[str, str]:
    """Return the dependency pins of `config.json` (the fallback versions).

    Returns:
        tuple: the (picongpu, mallocmc) pins, the empty string when the
        config carries none.

    """
    dependencies = load_config().get("dependencies", {})
    pins = []
    for name in ("picongpu", "mallocmc"):
        dependency = dependencies.get(name)
        pins.append(dependency.get("hash", "") if isinstance(dependency, dict) else "")
    return tuple(pins)


def collect() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Parse every legacy log directory into the frozen runs table.

    Returns:
        tuple: (the runs table, the source manifest (one
        "directory/file:sha256" entry per file, in the frozen table
        order), the excluded_sources map (directory -> reason)).

    """
    config = load_config()
    picongpu_pin, mallocmc_pin = _pins()
    frames: list[pd.DataFrame] = []
    manifest: list[str] = []
    excluded: dict[str, str] = {}
    for name, machine, hardware in _attributions():
        log_dir = LOGS_DIR / name
        manifest.extend(f"{name}/{path.name}:{_sha256(path)}" for path in _files(log_dir))
        modules_hint = " ".join(config["machines"][machine].get("modules", [])) if machine else ""
        frame = _parse_dir(log_dir, picongpu_pin=picongpu_pin, mallocmc_pin=mallocmc_pin, modules_hint=modules_hint)
        if frame.empty:
            continue
        frame["machine"] = machine
        frame["hardware"] = hardware
        if not hardware:
            excluded[name] = EXCLUDED_DIRECTORIES[name]
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.assign(z=runs.get("z", np.nan))
    return runs[RUNS_COLUMNS], manifest, excluded


def _text_values(series: pd.Series) -> list[str]:
    """Return the stored form of a text column: missing values as the empty string.

    Args:
        series: the text column.

    Returns:
        list[str]: the stored values.

    """
    return ["" if pd.isna(value) else str(value) for value in series]


def _write_runs_table(file: h5py.File, table: pd.DataFrame) -> None:
    """Write the runs table to the file, in the results file's table format.

    The format mirrors analysis/results_io.py's table writer (one dataset
    per column, `column_order` attribute, int64/float64/vlen-string
    dtypes, missing text as the empty string), so the main analysis can
    read the table back with `results_io.read_table`.

    Args:
        file: the opened results file (write mode).
        table: the table to write.

    """
    group = file.create_group("runs")
    group.attrs["column_order"] = ",".join(str(column) for column in table.columns)
    for column in table.columns:
        series = table[column]
        if pd.api.types.is_integer_dtype(series):
            group.create_dataset(column, data=series.to_numpy(dtype=np.int64))
        elif pd.api.types.is_numeric_dtype(series):
            group.create_dataset(column, data=series.to_numpy(dtype=np.float64))
        else:
            values = np.array(_text_values(series), dtype=object)
            dataset = group.create_dataset(column, shape=(len(values),), dtype=h5py.special_dtype(vlen=str))
            dataset[...] = values


def read_runs(path: Path) -> tuple[pd.DataFrame, dict[str, str], h5py.File]:
    """Read a frozen file back: the runs table and the top-level attributes.

    Args:
        path: the frozen file.

    Returns:
        tuple: (the runs table in stored column order, the attribute
        name -> value map, the opened file).

    """
    file = h5py.File(path, "r")
    group = file["runs"]
    data = {}
    for column, dataset in group.items():
        if isinstance(dataset, h5py.Dataset):
            values = dataset[...]
            if values.dtype == object:
                values = np.array(
                    [
                        value.decode("utf-8")
                        if isinstance(value, (bytes, bytearray))
                        else ("" if value is None else str(value))
                        for value in values
                    ],
                    dtype=object,
                )
            data[column] = values
    columns = [c for c in str(group.attrs.get("column_order", "")).split(",") if c in data]
    columns += [c for c in data if c not in columns]
    runs = pd.DataFrame({c: data[c] for c in columns})
    attrs = {key: str(file.attrs[key]) for key in file.attrs}
    return runs, attrs, file


def verify(output: Path) -> int:
    """Reparse the logs and compare against the frozen file.

    Args:
        output: the frozen file to verify.

    Returns:
        int: the process exit code (0 when the file matches the logs).

    """
    if not output.is_file():
        print(f"no frozen file at {output}; run `make legacy-results` first", file=sys.stderr)
        return 1
    runs, manifest, excluded = collect()
    # Text columns are stored with missing values as the empty string (see the
    # table writer); apply the same normalization before comparing, since a
    # freshly parsed frame still carries the raw missing values.
    for column in runs.columns:
        if not (pd.api.types.is_integer_dtype(runs[column]) or pd.api.types.is_numeric_dtype(runs[column])):
            runs[column] = _text_values(runs[column])
    stored, attrs, file = read_runs(output)
    with file:
        problems = []
        if str(attrs.get("sources", "")) != "; ".join(manifest):
            problems.append(
                "source manifest differs (the SHA-256 of one or more input files "
                f"changed: {len(manifest)} files now, "
                f"{len(str(attrs.get('sources', '')).split('; '))} stored)"
            )
        try:
            pd.testing.assert_frame_equal(
                runs, stored.reset_index(drop=True), check_dtype=False, check_index_type=False
            )
        except AssertionError as error:
            problems.append(f"runs table differs (a log file changed or the frozen parser drifted): {error}")
        if str(attrs.get("excluded_sources", "")) != json.dumps(excluded, sort_keys=True):
            problems.append("excluded_sources attribute differs")
        if problems:
            for problem in problems:
                print(f"legacy-verify: {problem}", file=sys.stderr)
            return 1
    print(f"legacy-verify: OK ({len(runs)} runs, {len(manifest)} files, excluded: {len(excluded)} directories)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build (or verify) the frozen legacy results file.

    Args:
        argv: the command-line arguments (after the program name).

    Returns:
        int: the process exit code.

    """
    parser = argparse.ArgumentParser(description="Freeze the legacy benchmark logs into legacy/legacy_results.h5.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination file (default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="reparse the logs and compare against the existing file")
    args = parser.parse_args(argv)
    if args.check:
        return verify(args.output)
    if not LOGS_DIR.is_dir() or not any(path.is_dir() for path in LOGS_DIR.iterdir()):
        print(f"legacy: no logs under {LOGS_DIR}; nothing to freeze", file=sys.stderr)
        return 0
    runs, manifest, excluded = collect()
    attrs = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": "; ".join(manifest),
        "excluded_sources": json.dumps(excluded, sort_keys=True),
        "attribution": json.dumps(
            {name: {"machine": machine, "hardware": hardware} for name, machine, hardware in _attributions()},
            sort_keys=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as file:
        for key, value in attrs.items():
            file.attrs[key] = value
        _write_runs_table(file, runs)
    print(f"wrote {args.output} ({len(runs)} runs, {len(manifest)} files, excluded: {', '.join(excluded) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
