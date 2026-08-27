"""Emit the self-describing `# metadata:` line of the benchmark run logs.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Called by `run_stamp.sh` (one invocation per grid-run log) and by the
`log_{setup,run}_<machine>.sh` launchers (session logs); each invocation
prints exactly one output line, which the caller writes into the log's
header:

    # metadata: {"schema": 1, "kind": "run", ...}

The `run` mode carries everything that is known at run time: the run
context (example, algorithm, imposed delays, the repetition, the
flags-file line and a short hash of it), the host and the repository
state, the dependency pins of `config.json`, the sha256 of the binary
used, the build facts of the tree the binary was built from (read from
its `CMakeCache.txt`, the same tree `make clean` removes with the
binary), and a hardware snapshot (GPU, CPU, OS). The `setup` mode marks
a session log (build or run launch); the analysis skips such files.
Every single fact is best effort: an unavailable value is the string
"unavailable", never an error, so a metadata problem can never block a
benchmark run. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import] (probes system tools by absolute path only)
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = 1
UNAVAILABLE = "unavailable"
METADATA_PREFIX = "# metadata: "
CUDA_RELEASE = re.compile(r"Cuda compilation tools, release (?P<release>[\w.]+)")
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _probe(command: list[str]) -> str | None:
    """Run `command` and capture its stdout.

    Args:
        command: the command line to run (absolute paths only).

    Returns:
        str | None: the captured stdout, or `None` when the command is
            missing or fails.

    """
    try:
        # Every command is an absolute path (resolved with `shutil.which` or
        # read from the build tree), never user input.
        probe = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except OSError, subprocess.SubprocessError:
        return None
    if probe.returncode != 0:
        return None
    return probe.stdout


def _first_line(text: str | None) -> str | None:
    """Return the first non-blank line of `text`.

    Args:
        text: the text to inspect.

    Returns:
        str | None: the first non-blank line, or `None` when there is none.

    """
    if text is None:
        return None
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def _git_state() -> dict[str, object]:
    """Return the repository state (the HEAD commit and the dirtiness).

    Returns:
        dict: `commit` (the full 40-hex HEAD) and `dirty` (True/False),
            both at their "unavailable" placeholders when the repository
            state cannot be determined.

    """
    state: dict[str, object] = {"commit": UNAVAILABLE, "dirty": None}
    git = shutil.which("git")
    if git is None:
        return state
    commit = _first_line(_probe([git, "rev-parse", "HEAD"]))
    if commit is not None and FULL_COMMIT.fullmatch(commit):
        state["commit"] = commit
    status = _probe([git, "status", "--porcelain"])
    if status is not None:
        state["dirty"] = bool(status.strip())
    return state


def _pins() -> dict[str, str]:
    """Return the dependency pins from `config.json`.

    Returns:
        dict: the full pins of `picongpu` and `mallocmc` (at their
            "unavailable" placeholders when they cannot be read).

    """
    pins = dict.fromkeys(("picongpu", "mallocmc"), UNAVAILABLE)
    try:
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return pins
    dependencies = config.get("dependencies") if isinstance(config, dict) else None
    if not isinstance(dependencies, dict):
        return pins
    for name in pins:
        dependency = dependencies.get(name)
        if isinstance(dependency, dict) and isinstance(dependency.get("hash"), str):
            pins[name] = dependency["hash"]
    return pins


def _gpu_names() -> list[str]:
    """Return the GPU product names, empty when unknown.

    Returns:
        list: the product names of every GPU (via `nvidia-smi`), or an
            empty list when the tool is missing or fails.

    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    query = _probe([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"])
    if query is None:
        return []
    return [name.strip() for name in query.splitlines() if name.strip()]


def _gpu_driver() -> str:
    """Return the GPU driver version, the placeholder when unknown.

    Returns:
        str: the driver version of the first GPU (via `nvidia-smi`), at
        its "unavailable" placeholder when the tool is missing or fails.

    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return UNAVAILABLE
    query = _probe([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"])
    return _first_line(query) or UNAVAILABLE


def _cpu_model() -> str:
    """Return the CPU model name, at its "unavailable" placeholder when unknown.

    Returns:
        str: the `Model name` reported by `lscpu`, or the placeholder.

    """
    lscpu = shutil.which("lscpu")
    if lscpu is None:
        return UNAVAILABLE
    info = _probe([lscpu])
    if info is None:
        return UNAVAILABLE
    for line in info.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip() or UNAVAILABLE
    return UNAVAILABLE


def _os_pretty_name() -> str:
    """Return the OS name from `/etc/os-release`, the placeholder when unknown.

    Returns:
        str: the `PRETTY_NAME` of the operating system, or the placeholder.

    """
    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return UNAVAILABLE
    for line in os_release.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"') or UNAVAILABLE
    return UNAVAILABLE


def _hardware() -> dict[str, object]:
    """Return a snapshot of the host hardware (GPU, GPU driver, CPU, OS).

    Returns:
        dict: the GPU product names (a list, empty when unknown), the
        driver version of the first GPU, the CPU model name and the
        operating system, each at its "unavailable" placeholder when
        unknown.

    """
    return {"gpu": _gpu_names(), "gpu_driver": _gpu_driver(), "cpu": _cpu_model(), "os": _os_pretty_name()}


def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of `path`.

    Args:
        path: the file to hash.

    Returns:
        str: the hex digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_block(binary: str) -> dict[str, str]:
    """Return the block describing the binary that is run.

    Args:
        binary: the path of the PIConGPU binary (relative to the
            repository root).

    Returns:
        dict: the binary path and its sha256 hex digest (at its
            "unavailable" placeholder when the file cannot be read).

    """
    block = {"path": binary}
    try:
        block["sha256"] = _sha256(Path(binary))
    except OSError:
        block["sha256"] = UNAVAILABLE
    return block


def _read_cache(cache: Path) -> dict[str, str] | None:
    """Parse a `CMakeCache.txt` into a key/value mapping.

    Args:
        cache: the path of the CMakeCache.txt file.

    Returns:
        dict | None: the cache entries, or `None` when the file cannot
            be read.

    """
    try:
        text = cache.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(("//", "#")):
            continue
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        _, _, value = rest.partition("=")
        if key and key not in entries:
            entries[key] = value.strip()
    return entries


def _build_block(binary: str) -> str | dict[str, str]:
    """Return the block describing how the binary was built.

    The facts are read from the `CMakeCache.txt` of the build tree the
    binary lives in (its parent-of-parent directory), so they belong to
    this binary, not to whatever cmake happens to be on the machine.

    Args:
        binary: the path of the PIConGPU binary.

    Returns:
        str | dict: the build facts (the cache file, the C++ and CUDA
            compiler versions, the compiler flags, the build type), or
            "unavailable" when the cache file is missing.

    """
    parents = Path(binary).parents
    cache_path = parents[1] / "CMakeCache.txt" if len(parents) >= 2 else Path("CMakeCache.txt")
    cache = _read_cache(cache_path)
    if cache is None:
        return UNAVAILABLE
    compiler = UNAVAILABLE
    compiler_path = cache.get("CMAKE_CXX_COMPILER", "")
    if compiler_path and Path(compiler_path).is_absolute():
        compiler = _first_line(_probe([compiler_path, "--version"])) or compiler_path
    cuda = UNAVAILABLE
    cuda_path = cache.get("CMAKE_CUDA_COMPILER", "")
    if cuda_path and Path(cuda_path).is_absolute():
        driver = _probe([cuda_path, "--version"])
        if driver is not None:
            for line in driver.splitlines():
                match = CUDA_RELEASE.search(line)
                if match:
                    cuda = f"CUDA {match['release']}"
                    break
    return {
        "cache": str(cache_path),
        "compiler": compiler,
        "cuda": cuda,
        "cxx_flags": cache.get("CMAKE_CXX_FLAGS") or UNAVAILABLE,
        "cuda_flags": cache.get("CMAKE_CUDA_FLAGS") or UNAVAILABLE,
        "build_type": cache.get("CMAKE_BUILD_TYPE") or UNAVAILABLE,
    }


def _flag_lines(flags: Path) -> list[str] | None:
    r"""Read the lines of a flags file.

    The line split mirrors the `read -r line` loop of `run_folder.sh`
    (one line per `\n`, a trailing newline not producing an empty line),
    so the line numbers stay in step between the two.

    Args:
        flags: the path of the flags file.

    Returns:
        list | None: the lines, or `None` when the file cannot be read.

    """
    try:
        text = flags.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = text.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _run_block(args: argparse.Namespace) -> dict[str, object]:
    """Return the block describing this grid run.

    Args:
        args: the parsed command line (see `main`).

    Returns:
        dict: the run context (example, algorithm, imposed delays, the
            repetition), the flags-file line number, the line count, a
            short hash of the line, and the line itself.

    """
    block: dict[str, object] = {
        "setup": args.example,
        "algorithm": args.algorithm,
        "delays": [args.malloc_delay, args.free_delay],
        "rep": args.rep,
        "repeats": args.repeats,
        "line": args.line,
        "lines_total": UNAVAILABLE,
        "flag_sha": UNAVAILABLE,
        "command": UNAVAILABLE,
    }
    lines = _flag_lines(Path(args.flags))
    if lines is not None:
        block["lines_total"] = len(lines)
        if 1 <= args.line <= len(lines):
            line = lines[args.line - 1]
            block["command"] = line
            block["flag_sha"] = hashlib.sha256(line.encode("utf-8")).hexdigest()[:8]
    return block


def _metadata(kind: str, machine: str, extra: dict[str, object]) -> dict[str, object]:
    """Assemble the metadata dict.

    Args:
        kind: the metadata kind ("run" or "setup").
        machine: the machine label from the config machines table.
        extra: the kind-specific blocks (binary, build, run).

    Returns:
        dict: the metadata, ready for `json.dumps`.

    """
    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "kind": kind,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": machine,
        "hostname": socket.gethostname(),
    }
    metadata.update(_git_state())
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        metadata["slurm_job"] = slurm_job
    metadata["pins"] = _pins()
    metadata["hw"] = _hardware()
    metadata.update(extra)
    return metadata


def main() -> int:
    """Run the command line interface.

    Returns:
        int: the process exit status.

    """
    parser = argparse.ArgumentParser(description="Emit one self-describing '# metadata:' log line (schema 1).")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="the metadata line of one grid-run log")
    run.add_argument("--machine", required=True)
    run.add_argument("--repeats", type=int, required=True)
    run.add_argument("--rep", type=int, required=True)
    run.add_argument("--example", required=True)
    run.add_argument("--algorithm", required=True)
    run.add_argument("--malloc-delay", type=int, required=True)
    run.add_argument("--free-delay", type=int, required=True)
    run.add_argument("--line", type=int, required=True)
    run.add_argument("--flags", required=True)
    run.add_argument("--binary", required=True)

    setup = subcommands.add_parser("setup", help="the metadata line of a session log")
    setup.add_argument("--machine", required=True)

    args = parser.parse_args()
    extra: dict[str, object] = {}
    if args.command == "run":
        kind = "run"
        extra["binary"] = _binary_block(args.binary)
        extra["build"] = _build_block(args.binary)
        extra["run"] = _run_block(args)
    else:
        kind = "setup"
    print(METADATA_PREFIX + json.dumps(_metadata(kind, args.machine, extra), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
