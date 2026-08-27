"""Parse the provenance metadata out of one run log.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

`make_legacy_results.py` stores, on every frozen run, the provenance of the
log file the run came from: what was built, what was measured, and on what
hardware. The parser is generation-aware and defensive -- every field comes
out as the empty string when the log does not carry it:

- the redesign-era layout of `run_stamp.sh` (a `# <key>: <value>` header
  that carries the start datetime, the host, the GPU and its driver, the
  loaded modules, the compiler, the checked-out dependency hashes (the
  `source` line), the machine, the commit and the binary's sha256),
- the legacy "Logging environment" layout (a `git log -1` block, an
  `nvidia-smi` table, and -- on some machines -- an `hwinfo` dump and a
  full `env` dump),
- slurm build+run logs named `slurm-<job id>.out` (they embed the
  dependency checkouts as git output, and nothing else of provenance).

The start datetime of a legacy run log comes from its file name
(`run_<RFC 3339 timestamp>.txt`); a log with neither a start line in its
header nor a timestamped name carries no start datetime. The compiler is
read, in this priority, from the header's `cxx` line, from the traced
toolchain loading of the profile (`spack load gcc@...`,
`module load nvidia-compilers/...` or `.../gcc|GCCcore/...`), or from the
machine's modules of `config.json`. The PIConGPU / mallocMC versions of the
legacy logs are approximated by the pinned dependency hashes of
`config.json`, which the build harness checks out; the current layout
records the actual checked-out hashes instead.
"""

from __future__ import annotations

import re
from pathlib import Path

# The columns `parse_log_metadata` fills, in their order of the runs table
# (the returned dict carries the `generation` tag in addition).
META_COLUMNS = (
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
)

# `# <key>: <value>` header line of the current run-stamp layout.
HEADER_LINE_RE = re.compile(r"^# (\w+):\s*(.*)$")
# The full 40-hex commit line of a `git log -1` block (the first match is
# the logged commit itself; a diff hunk could only echo such a line later).
GIT_COMMIT_RE = re.compile(r"^commit ([0-9a-f]{40})\s*$")
# The two header banner lines of `nvidia-smi` (one per layout generation).
SMI_DRIVER_RE = re.compile(r"NVIDIA-SMI (\S+)\s+Driver Version:\s*(\S+)\s+CUDA Version:\s*(\S+)")
SMI_KMD_RE = re.compile(r"NVIDIA-SMI (\S+)\s+KMD Version:.*CUDA UMD Version:\s*(\S+)")
# A GPU row of the `nvidia-smi` table; on multi-GPU nodes only the first
# row is recorded.
SMI_GPU_RE = re.compile(r"^\|\s*\d+\s+(.+?)\s+(?:On|Off)\s*\|")
# A CPU line of an `hwinfo` dump.
HWINFO_CPU_RE = re.compile(r"^\s*model name\s*:\s*(.+)$")
# The RFC 3339 timestamp of a legacy run/setup log file name.
FILENAME_START_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})(\+\d{2}:\d{2})\.\w+$")
# The job id of a slurm output file name.
SLURM_NAME_RE = re.compile(r"slurm-(\d+)\.out$")
# The dependency checkouts of slurm build+run logs: one block per
# `git clone` (the main checkout, whose path ends in `src/picongpu`) or
# submodule checkout (`thirdParty/mallocMC`), ending in a 'switching to'
# or 'checked out' line with the full hash.
CLONE_RE = re.compile(r"Cloning into '([^']+)'")
CHECKOUT_RE = re.compile(r"(?:switching to|checked out) '([0-9a-f]{40})'")
# The checked-out hashes of the build's `.source-stamp` (the `# source:`
# header line): "picongpu <hash> mallocmc <hash>".
SOURCE_PICONGPU_RE = re.compile(r"picongpu (\S+)")
SOURCE_MALLOCMC_RE = re.compile(r"mallocmc (\S+)")
# Traced compiler loads of a profile (`set -x`), in priority order.
COMPILER_RE = (
    (re.compile(r"spack load gcc@([\d.]+)"), "gcc {m}"),
    (re.compile(r"module load \S*nvidia-compilers/([\w.-]+)"), "nvidia-compilers {m}"),
    (re.compile(r"module load (?:\S*[/?])?(?:gcc|GCCcore)/([\w.-]+)"), "gcc {m}"),
    (re.compile(r"module load (?:\S*[/?])?GCC/([\d.]+)"), "GCC {m}"),
)


def _filename_start(name: str) -> str:
    """Return the start datetime of one timestamped legacy log name, or "".

    Args:
        name: the log file name, e.g. `run_2026-08-20_14:58:01+02:00.txt`.

    Returns:
        str: the start datetime of the file name, in RFC 3339 form.

    """
    match = FILENAME_START_RE.search(name)
    return f"{match[1]}T{match[2]}{match[3]}" if match else ""


def _detect(stripped: list[str], name: str) -> tuple[str, list[str]]:
    """Detect the log generation and its `# key: value` header lines.

    Args:
        stripped: the log's lines, whitespace-stripped.
        name: the log file name.

    Returns:
        tuple: (the generation tag, the header lines, which are the
        current layout's `#` lines and empty for the other generations).

    """
    if stripped and stripped[0].startswith("# "):
        header = []
        for line in stripped:
            if line.startswith("# "):
                header.append(line)
            elif header:
                break
        return "run-stamp", header
    if any("Logging environment" in line[:160] for line in stripped[:200]):
        return "logging-env", []
    if SLURM_NAME_RE.search(name):
        return "slurm", []
    return "unknown", []


# The header keys whose value is the column's value verbatim.
_DIRECT_KEYS = {"started": "started_utc", "host": "host", "commit": "commit", "slurm_job": "slurm_job"}
# The header keys whose value is a leading whitespace-delimited token.
_TOKEN_KEYS = {"picongpu": "picongpu", "mallocmc": "mallocmc", "binary": "binary_sha256"}


def _stamp_source(meta: dict, value: str) -> None:
    """Copy the checked-out hashes of the `# source:` line onto `meta`.

    These supersede the pinned hashes of the `picongpu` / `mallocmc`
    header lines.

    Args:
        meta: the values collected so far.
        value: the line's value, "picongpu <hash> mallocmc <hash>".

    """
    pic = SOURCE_PICONGPU_RE.search(value)
    if pic:
        meta["picongpu"] = pic[1][:8]
    mc = SOURCE_MALLOCMC_RE.search(value)
    if mc:
        meta["mallocmc"] = mc[1][:8]


def _stamp_gpu(meta: dict, value: str) -> None:
    """Copy the first GPU's name and driver of the `# gpu:` line onto `meta`.

    Args:
        meta: the values collected so far.
        value: the line's value, "NVIDIA A30,610.43.02" per GPU, the GPUs
        joined with ", " (the csv name/driver pair itself has no space).

    """
    parts = [p for p in (part.strip() for part in value.split(",")) if p]
    meta["gpu"] = parts[0]
    if len(parts) > 1:
        meta["gpu_driver"] = parts[1]


def _run_stamp_meta(header: list[str]) -> dict:
    """Read the provenance values out of one run-stamp header.

    A value of `unavailable` is the header's sentinel of a probe that
    failed on the running machine; it is recorded as absent ("").

    Args:
        header: the header's `# key: value` lines.

    Returns:
        dict: the values the header carries, keyed by column name.

    """
    meta = {}
    for line in header:
        match = HEADER_LINE_RE.match(line)
        if match is None or match[2] == "unavailable":
            continue
        key, value = match[1], match[2]
        if key in _DIRECT_KEYS:
            meta[_DIRECT_KEYS[key]] = value
        elif key in _TOKEN_KEYS:
            meta[_TOKEN_KEYS[key]] = value.split(maxsplit=1)[0]
        elif key == "source":
            _stamp_source(meta, value)
        elif key == "gpu":
            _stamp_gpu(meta, value)
        elif key == "cxx":
            meta["compiler"] = value
    return meta


def _smi_gpu(line: str) -> str | None:
    """Return the model of the GPU of one `nvidia-smi` table row, or None.

    Args:
        line: a stripped log line.

    Returns:
        str | None: the GPU model of the row, or None if the line is no row.

    """
    gpu = SMI_GPU_RE.match(line)
    return gpu[1] if gpu else None


def _smi_banner(line: str) -> tuple[str, str] | None:
    """Return (driver, CUDA version) of one `nvidia-smi` banner line, or None.

    Args:
        line: a stripped log line.

    Returns:
        tuple | None: the driver and the CUDA version of the banner (in the
        `Driver Version:` layout the SMI version token is the driver, in
        the `KMD Version:` layout the KMD one), or None if the line is no
        banner.

    """
    driver = SMI_DRIVER_RE.search(line)
    if driver:
        return driver[2], driver[3]
    kmd = SMI_KMD_RE.search(line)
    return (kmd[1], kmd[2]) if kmd else None


def _scan_env_line(line: str, meta: dict) -> bool:
    """Remember the values of one legacy log line on `meta`.

    The first value of each field wins (the `git log -1` block precedes
    the `git diff` that could only echo look-alike lines later).

    Args:
        line: a stripped log line.
        meta: the values collected so far.

    Returns:
        bool: True once the commit and both GPU values are all recorded.

    """
    if not meta.get("commit"):
        commit = GIT_COMMIT_RE.match(line)
        if commit:
            meta["commit"] = commit[1]
    if not meta.get("gpu"):
        gpu = _smi_gpu(line)
        if gpu is not None:
            meta["gpu"] = gpu
    if not meta.get("gpu_driver"):
        banner = _smi_banner(line)
        if banner is not None:
            meta["gpu_driver"], meta["cuda_version"] = banner
    return bool(meta.get("commit") and meta.get("gpu") and meta.get("gpu_driver"))


def _logging_env_meta(stripped: list[str]) -> dict:
    """Extract the `git log`, `nvidia-smi` and `hwinfo` values of one legacy log.

    Args:
        stripped: the log's lines, whitespace-stripped.

    Returns:
        dict: the commit, the GPU (model, driver, CUDA version) and the CPU
        model, as far as the log carries them.

    """
    meta = {}
    for line in stripped:
        if _scan_env_line(line, meta):
            break
    if not meta.get("cpu"):
        # The hwinfo dump sits deep in the log (after the full `env` dump),
        # so the CPU gets its own single-regex pass.
        for line in stripped:
            cpu = HWINFO_CPU_RE.match(line)
            if cpu:
                meta["cpu"] = cpu[1]
                break
    return meta


def _slurm_meta(stripped: list[str], name: str) -> dict:
    """Extract the job id and the dependency checkouts of one slurm log.

    Args:
        stripped: the log's lines, whitespace-stripped.
        name: the log file name (`slurm-<job id>.out`).

    Returns:
        dict: the slurm job id, and the checked-out dependency hashes if
        the log rebuilt the dependencies.

    """
    meta = {"slurm_job": SLURM_NAME_RE.search(name)[1]}
    inside = ""
    for line in stripped:
        clone = CLONE_RE.search(line)
        if clone:
            path = clone[1]
            if path.endswith("/src/picongpu"):
                inside = "picongpu"
            elif path.endswith("/thirdParty/mallocMC"):
                inside = "mallocmc"
            else:
                inside = ""
        elif line.startswith("Submodule"):
            inside = "mallocmc" if "mallocMC" in line else ""
        elif inside and not meta.get(inside):
            checkout = CHECKOUT_RE.search(line)
            if checkout:
                meta[inside] = checkout[1][:8]
                inside = ""
    return meta


def _compiler(stripped: list[str], hint: str) -> str:
    """Read the compiler of the traced toolchain loading, or the config hint.

    Args:
        stripped: the log's lines, whitespace-stripped.
        hint: the machine's modules of `config.json` (" "-joined).

    Returns:
        str: the compiler, or the hint, or "" when nothing is found.

    """
    for pattern, fmt in COMPILER_RE:
        match = pattern.search("\n".join(stripped))
        if match:
            return fmt.format(m=match[1])
    return hint


def parse_log_metadata(
    log_path: Path,
    *,
    picongpu_pin: str = "",
    mallocmc_pin: str = "",
    modules_hint: str = "",
) -> dict:
    """Parse one run log's provenance metadata.

    Args:
        log_path: the log to read.
        picongpu_pin: the PIConGPU pin of `config.json`, applied as the
        dependency version of a log that carries none of its own.
        mallocmc_pin: the mallocMC pin of `config.json`, applied as for
        `picongpu_pin`.
        modules_hint: the machine's modules of `config.json`, the last-resort
        compiler fallback (" "-joined).

    Returns:
        dict: one value per `META_COLUMNS` entry ("" when the log carries
        no value), in addition to the `generation` tag.

    """
    meta = dict.fromkeys(META_COLUMNS, "")
    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        lines = [line.rstrip("\n") for line in file]
    stripped = [line.strip() for line in lines]
    generation, header = _detect(stripped, log_path.name)
    meta["generation"] = generation
    if generation == "run-stamp":
        meta.update(_run_stamp_meta(header))
    elif generation == "logging-env":
        meta.update(_logging_env_meta(stripped))
    elif generation == "slurm":
        meta.update(_slurm_meta(stripped, log_path.name))
    meta["started_utc"] = meta["started_utc"] or _filename_start(log_path.name)
    if not meta["compiler"]:
        meta["compiler"] = _compiler(stripped, modules_hint)
    if not meta["picongpu"]:
        meta["picongpu"] = picongpu_pin[:8] if picongpu_pin else ""
    if not meta["mallocmc"]:
        meta["mallocmc"] = mallocmc_pin[:8] if mallocmc_pin else ""
    return meta
