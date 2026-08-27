"""Read and validate `config.json`, the single source of harness configuration.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The build harness (the Makefile and the per-machine log_*.sh launchers)
parses no config format itself; it calls this helper to look up individual
values:

    python3 config.py get dependencies.picongpu.hash
    python3 config.py list examples
    python3 config.py list run-matrix [initial|arms]
    python3 config.py check

`check` validates the structure and the file references (flag files,
parameter files, profiles) so that a typo fails fast with a clear message
instead of a silently wrong benchmark run. Run it from the repository root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path("config.json")

# The phases of the two-stage sweep: the fast first scan and the full
# design (the Makefile's PHASE invocation value takes the same keys).
SWEEP_PHASES = {"initial", "arms"}


def _fail(message: str) -> None:
    """Print `message` to stderr and exit with status 1.

    Args:
        message: the error message.

    """
    print(f"config.json: {message}", file=sys.stderr)
    sys.exit(1)


def _load() -> dict:
    """Load `config.json` from the current directory.

    Returns:
        dict: the parsed configuration.

    """
    if not CONFIG_PATH.is_file():
        _fail(f"{CONFIG_PATH} not found (run from the repository root)")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        _fail("top level must be a mapping")
    return data


def _walk(data: dict, dotted: str) -> object | None:
    """Follow the dotted path `dotted` into `data`.

    Args:
        data: the parsed configuration.
        dotted: the dotted key path (e.g. "delays.arms.values").

    Returns:
        object | None: the value at the path, or None when it is absent.

    """
    node: object = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _lookup(data: dict, dotted: str) -> object:
    """Return the value at `dotted`, failing when the path is absent.

    Args:
        data: the parsed configuration.
        dotted: the dotted key path.

    Returns:
        object: the value at the path.

    """
    value = _walk(data, dotted)
    if value is None:
        _fail(f"unknown key '{dotted}'")
    return value


def _is_str_list(value: object, *, non_empty: bool = True) -> bool:
    """Whether `value` is a list of strings (optionally non-empty).

    Args:
        value: the value to test.
        non_empty: whether an empty list is acceptable.

    Returns:
        bool: whether `value` fits.

    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return False
    return not non_empty or bool(value)


def _is_int_list(value: object, *, non_empty: bool = True) -> bool:
    """Whether `value` is a list of integers (optionally non-empty), bools excluded.

    Args:
        value: the value to test.
        non_empty: whether an empty list is acceptable.

    Returns:
        bool: whether `value` fits.

    """
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return False
    return not non_empty or bool(value)


def _check_run_matrix(data: dict, errors: list[str]) -> None:
    """Validate examples, algorithms, and the delay sweep.

    The phase-1 arm subset (`delays.arms.initial`) and the optional joint
    grid (`delays.joint.values`) must be subsets of the arm ladder, so that
    every combination of a phase is a combination of the full design.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    for dotted in ("examples", "algorithms"):
        if _is_str_list(_walk(data, dotted)):
            continue
        errors.append(f"{dotted}: must be a non-empty list of strings")
    baseline = _walk(data, "delays.baseline")
    if not _is_int_list(baseline) or len(baseline) != 2:
        errors.append("delays.baseline: must be a list of two integers (malloc, free)")
    arms = _walk(data, "delays.arms.values")
    if not _is_int_list(arms):
        errors.append("delays.arms.values: must be a non-empty list of integers (nanoseconds)")
        return
    arm_set = set(arms)
    initial = _walk(data, "delays.arms.initial")
    if initial is not None and (not _is_int_list(initial) or not set(initial) <= arm_set):
        errors.append("delays.arms.initial: must be a non-empty subset of delays.arms.values")
    joint = _walk(data, "delays.joint.values")
    if joint is not None and (not _is_int_list(joint, non_empty=False) or not set(joint) <= arm_set):
        errors.append("delays.joint.values: must be a subset of delays.arms.values (optional, may be empty)")


def _check_build(data: dict, errors: list[str]) -> None:
    """Validate the dependency pins and the build flags.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    for name in ("picongpu", "mallocmc"):
        for field in ("url", "path", "hash"):
            dotted = f"dependencies.{name}.{field}"
            if not isinstance(_walk(data, dotted), str):
                errors.append(f"{dotted}: must be a string")
    if not isinstance(_walk(data, "build.cxx_flags"), str):
        errors.append("build.cxx_flags: must be a string")
    if not _is_str_list(_walk(data, "build.extra_cmake_flags"), non_empty=False):
        errors.append("build.extra_cmake_flags: must be a list of strings")


def _check_machines(data: dict, errors: list[str]) -> None:
    """Validate the per-machine table and the profile file references.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    machines = _walk(data, "machines")
    if not isinstance(machines, dict) or not machines:
        errors.append("machines: must be a non-empty mapping")
        return
    for name, machine in machines.items():
        if not isinstance(machine, dict):
            errors.append(f"machines.{name}: must be a mapping")
            continue
        for field in ("profile", "output", "hardware"):
            if isinstance(machine.get(field), str):
                continue
            errors.append(f"machines.{name}.{field}: must be a string")
        if not _is_str_list(machine.get("modules", []), non_empty=False):
            errors.append(f"machines.{name}.modules: must be a list of strings (optional)")
        profile = machine.get("profile")
        if isinstance(profile, str):
            _missing_file(Path(profile), f"machines.{name}.profile", errors)


def _missing_file(path: Path, dotted: str, errors: list[str]) -> None:
    """Record `dotted` in `errors` when `path` does not exist.

    Args:
        path: the expected file.
        dotted: the dotted key path (for the error message).
        errors: accumulates the problems found.

    """
    if not path.is_file():
        errors.append(f"{dotted}: file not found: {path}")


def _check_benchmark_files(data: dict, errors: list[str]) -> None:
    """Validate the per-example flag files and per-algorithm parameter files.

    A missing file means the benchmark would run with the wrong (or no)
    configuration, so it is a configuration error.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    examples = _walk(data, "examples")
    if isinstance(examples, list):
        for example in examples:
            if isinstance(example, str):
                _missing_file(Path("flags") / f"{example}.flags", "examples", errors)
    algorithms = _walk(data, "algorithms")
    if isinstance(algorithms, list):
        for algorithm in algorithms:
            if isinstance(algorithm, str):
                _missing_file(Path("param") / algorithm / "mallocMC.param", "algorithms", errors)


def _run_matrix(data: dict, phase: str = "arms") -> list[str]:
    """Return the (malloc, free) delay combinations of one sweep phase.

    The phases of the two-stage sweep are `initial` (the baseline plus the
    `delays.arms.initial` arm subset, or the full ladder when the subset is
    not configured) and `arms` (baseline, full ladder, and the joint grid
    when one is configured). The initial phase's combinations are a subset
    of the arms phase's, so the run stamps let a later `arms` sweep pick up
    where the `initial` sweep stopped.

    The order mirrors the historical run_all.sh sweep: the baseline first,
    then each arm value twice (malloc delay, then free delay, the other held
    at 0), then the full joint grid.

    Args:
        data: the parsed configuration.
        phase: the sweep phase ("initial" or "arms").

    Returns:
        list[str]: one "<malloc>_<free>" token per combination.

    Raises:
        ValueError: if `phase` is not a known phase.

    """
    if phase not in SWEEP_PHASES:
        msg = f"unknown sweep phase '{phase}' (expected initial or arms)"
        raise ValueError(msg)
    baseline = data["delays"]["baseline"]
    arms_spec = data["delays"]["arms"]
    arms = arms_spec.get("initial", arms_spec["values"]) if phase == "initial" else arms_spec["values"]
    joint = []
    if phase == "arms":
        joint = data["delays"].get("joint", {}).get("values", []) or []
    combinations = [f"{baseline[0]}_{baseline[1]}"]
    for delay in arms:
        combinations += [f"{delay}_0", f"0_{delay}"]
    combinations += [f"{malloc_delay}_{free_delay}" for malloc_delay in joint for free_delay in joint]
    return combinations


def _cmd_check() -> None:
    """Validate the structure and file references of `config.json`.

    Prints every problem found and exits with status 1 when there is any;
    prints a short OK line otherwise.
    """
    data = _load()
    errors: list[str] = []
    _check_run_matrix(data, errors)
    _check_build(data, errors)
    _check_machines(data, errors)
    _check_benchmark_files(data, errors)
    if errors:
        for error in errors:
            print(f"config.json: {error}", file=sys.stderr)
        sys.exit(1)
    print("config.json: OK")


def _print_scalar(value: object, dotted: str) -> None:
    """Print the looked-up scalar on one line.

    Args:
        value: the looked-up value.
        dotted: the dotted key path (for the error message).

    """
    if isinstance(value, (list, dict)):
        _fail(f"'{dotted}' is a {type(value).__name__}; use 'list' for a list")
    print(value)


def _print_list(value: object, dotted: str) -> None:
    """Print each element of the looked-up list on its own line.

    Args:
        value: the looked-up value.
        dotted: the dotted key path (for the error message).

    """
    if not isinstance(value, list):
        _fail(f"'{dotted}' is not a list")
    for item in value:
        if isinstance(item, (list, dict)):
            _fail(f"'{dotted}' contains a non-scalar item")
        print(item)


def _cmd_list_run_matrix(rest: list[str]) -> None:
    """Print the run matrix of one sweep phase, one combination per line.

    Args:
        rest: the command line arguments after the "list" subcommand.

    """
    if len(rest) > 2:
        _fail("usage: config.py list run-matrix [initial|arms]")
    phase = rest[1] if len(rest) == 2 else "arms"
    if phase not in SWEEP_PHASES:
        _fail(f"unknown sweep phase '{phase}' (expected initial or arms)")
    for combination in _run_matrix(_load(), phase):
        print(combination)


def main() -> int:
    """Dispatch the subcommand on the command line.

    Returns:
        int: the process exit status.

    """
    args = sys.argv[1:]
    if not args:
        _fail("usage: config.py {get|list|check} [dotted.key]")
    command, rest = args[0], args[1:]
    if command == "check":
        if rest:
            _fail("usage: config.py check")
        _cmd_check()
        return 0
    if command not in {"get", "list"}:
        _fail(f"unknown command '{command}' (expected get, list or check)")
    if rest and rest[0] == "run-matrix":
        if command != "list":
            _fail("'run-matrix' is a list, not a scalar key")
        _cmd_list_run_matrix(rest)
        return 0
    if len(rest) != 1:
        _fail(f"usage: config.py {command} <dotted.key>")
    value = _lookup(_load(), rest[0])
    if command == "get":
        _print_scalar(value, rest[0])
    else:
        _print_list(value, rest[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
