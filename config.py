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
    python3 config.py flag-lines <Example>

`check` validates the structure and the file references (parameter files,
profiles, the microbenchmark paths) so that a typo fails fast with a clear
message instead of a silently wrong benchmark run. Run it from the
repository root.

The example's picongpu command lines are stored as structured `flag_lines`
(entries of `examples`): one JSON object per command line, the keys in
command-line order, a value a scalar or a list of positive integers.
`flag-lines` (and the build) reconstruct the exact command lines from them:
a one-letter key serializes to its short form (-d), any longer key to its
long form (--periodic), a scalar to a single token, a list to a
space-joined token; `config.py` is the sole authority for the format.
"""

from __future__ import annotations

import json
import re
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


def _is_int_or_nonneg_int_list(value: object, *, non_empty: bool = True) -> bool:
    """Whether `value` is an integer or a list of non-negative integers (optionally non-empty list).

    A zero is a legitimate option value (a dimension that is not periodic,
    a zero size); the guard is against booleans, negatives, and empty lists.

    Args:
        value: the value to test.
        non_empty: whether an empty list is acceptable.

    Returns:
        bool: whether `value` fits.

    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    if isinstance(value, list):
        return all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value) and (
            not non_empty or bool(value)
        )
    return False


def flag_lines(definition: dict) -> list[str]:
    """Serialize the `flag_lines` of one example into picongpu command lines.

    One JSON object is one command line: the keys in insertion order become
    the options in command-line order. A one-letter key serializes to its
    short form (`d` -> `-d`), any longer key to its long form
    (`periodic` -> `--periodic`); a scalar value is a single token, a list a
    space-joined run of tokens.

    Args:
        definition: the example's `flag_lines` list (a list of mappings).

    Returns:
        list: one command line (a string) per entry.

    """
    lines = []
    for entry in definition:
        tokens = []
        for key, value in entry.items():
            tokens.append("-" + key if len(key) == 1 else "--" + key)
            values = value if isinstance(value, list) else [value]
            tokens.extend(str(item) for item in values)
        lines.append(" ".join(tokens))
    return lines


def _check_flag_lines(flag_lines: list[object], dotted: str, errors: list[str]) -> None:
    """Validate one example's `flag_lines` list.

    Args:
        flag_lines: the `flag_lines` value (checked here).
        dotted: the dotted path of the example (for the error messages).
        errors: accumulates the problems found.

    """
    for line_index, line in enumerate(flag_lines):
        line_dotted = f"{dotted}.flag_lines[{line_index}]"
        if not isinstance(line, dict) or not line:
            errors.append(f"{line_dotted}: each line must be a non-empty option mapping")
            continue
        for key, value in line.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z]+", key):
                errors.append(f"{line_dotted}: option keys must be [A-Za-z]+ strings")
            if not _is_int_or_nonneg_int_list(value, non_empty=True):
                errors.append(f"{line_dotted}.{key}: must be a non-negative integer or a non-empty list of them")


def _check_examples(data: dict, errors: list[str]) -> None:
    """Validate the `examples` list (name + flag_lines per example).

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    examples = _walk(data, "examples")
    if not isinstance(examples, list) or not examples:
        errors.append("examples: must be a non-empty list of {name, flag_lines}")
        return
    names = set()
    for index, example in enumerate(examples):
        dotted = f"examples[{index}]"
        if not isinstance(example, dict):
            errors.append(f"{dotted}: each example must be a {{name, flag_lines}} mapping")
            continue
        name = example.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{dotted}.name: must be a non-empty string")
        elif name in names:
            errors.append(f"{dotted}.name: example name '{name}' is not unique")
        else:
            names.add(name)
        flag_lines_ = example.get("flag_lines")
        if not isinstance(flag_lines_, list) or not flag_lines_:
            errors.append(f"{dotted}.flag_lines: must be a non-empty list of option mappings")
        else:
            _check_flag_lines(flag_lines_, dotted, errors)


def _check_algorithms(data: dict, errors: list[str]) -> None:
    """Validate the `algorithms` list.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    if not _is_str_list(_walk(data, "algorithms")):
        errors.append("algorithms: must be a non-empty list of strings")


def _check_delays(data: dict, errors: list[str]) -> None:
    """Validate the delay sweep (optional; absent means baseline-only).

    The phase-1 arm subset (`delays.arms.initial`) and the optional joint
    grid (`delays.joint.values`) must be subsets of the arm ladder, so that
    every combination of a phase is a combination of the full design. All of
    the ladder, the subset, and the joint grid may be empty, in which case a
    phase is baseline-only.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    delays = _walk(data, "delays")
    if delays is None:
        return
    if not isinstance(delays, dict):
        errors.append("delays: must be a mapping (or be omitted for a baseline-only run)")
        return
    baseline = delays.get("baseline", [0, 0])
    if not _is_int_list(baseline) or len(baseline) != 2:
        errors.append("delays.baseline: must be a list of two integers (malloc, free)")
    arms = _walk(data, "delays.arms.values")
    if arms is None:
        return
    if not _is_int_list(arms, non_empty=False):
        errors.append("delays.arms.values: must be a list of integers (nanoseconds)")
        return
    arm_set = set(arms)
    initial = _walk(data, "delays.arms.initial")
    if initial is not None and (not _is_int_list(initial, non_empty=False) or not set(initial) <= arm_set):
        errors.append("delays.arms.initial: must be a subset of delays.arms.values (optional, may be empty)")
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


def _check_people(data: dict, errors: list[str]) -> None:
    """Validate the optional people table (login to name/orcid mappings).

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    people = _walk(data, "people")
    if people is None:
        return
    if not isinstance(people, dict):
        errors.append("people: must be a mapping of login to {name, orcid}")
        return
    for login, person in people.items():
        if not isinstance(person, dict):
            errors.append(f"people.{login}: must be a mapping")
            continue
        orcid = person.get("orcid")
        if orcid is not None and (
            not isinstance(orcid, str) or not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid)
        ):
            errors.append(f"people.{login}.orcid: must be an ORCID iD (NNNN-NNNN-NNNN-NNNC) or omitted")
        name = person.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            errors.append(f"people.{login}.name: must be a non-empty string or omitted")
        if person.get("orcid") is None and person.get("name") is None:
            errors.append(f"people.{login}: needs at least one of name, orcid")


def _check_microbench(data: dict, errors: list[str]) -> None:
    """Validate the microbenchmark section (the memmansurvey pin and data).

    The raw results and the frozen table are machine data (git-ignored), so
    only the paths and the published run list are validated here; the data
    directories themselves are not required to exist. The data and frozen
    paths must not lie inside the submodule checkout, which is pinned code.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    microbench = _walk(data, "microbench")
    if microbench is None:
        return
    if not isinstance(microbench, dict):
        errors.append("microbench: must be a mapping")
        return
    _check_microbench_paths(microbench, errors)
    _check_microbench_runs(microbench.get("runs"), errors)


def _check_microbench_paths(microbench: dict, errors: list[str]) -> None:
    """Validate the submodule/data/frozen fields of the microbench section.

    Each must be a non-empty string, and the data/frozen paths must not lie
    inside the submodule checkout (which is pinned code, not data).

    Args:
        microbench: the parsed `microbench` mapping.
        errors: accumulates the problems found.

    """
    errors.extend(
        f"microbench.{field}: must be a non-empty string"
        for field in ("submodule", "data", "frozen")
        if not isinstance(microbench.get(field), str) or not microbench.get(field)
    )
    submodule = microbench.get("submodule")
    if isinstance(submodule, str) and submodule:
        for field in ("data", "frozen"):
            path = microbench.get(field)
            if isinstance(path, str) and path and _is_inside(path, submodule):
                errors.append(f"microbench.{field}: must not lie inside the submodule checkout '{submodule}'")


def _is_inside(path: str, root: str) -> bool:
    """Return True if `path` equals `root` or is nested inside it.

    Args:
        path: the path to check.
        root: the root path.

    Returns:
        bool: True when `path` is `root` or nested under it.

    """
    return path == root or path.startswith(root.rstrip("/") + "/")


def _check_microbench_runs(runs: object, errors: list[str]) -> None:
    """Validate the microbench run list (a non-empty list of {jobid, hardware}).

    Args:
        runs: the `microbench.runs` value (checked here).
        errors: accumulates the problems found.

    """
    if not isinstance(runs, list) or not runs:
        errors.append("microbench.runs: must be a non-empty list of {jobid, hardware}")
        return
    jobids: list[int] = []
    for entry in runs:
        if not isinstance(entry, dict):
            errors.append("microbench.runs: each entry must be a {jobid, hardware} mapping")
            continue
        jobid = entry.get("jobid")
        if not isinstance(jobid, int) or isinstance(jobid, bool):
            errors.append(f"microbench.runs.{jobid!r}.jobid: must be an integer")
        else:
            jobids.append(jobid)
        hardware = entry.get("hardware")
        if not isinstance(hardware, str) or not hardware:
            errors.append(f"microbench.runs.{jobid!r}.hardware: must be a non-empty string")
    if len(set(jobids)) != len(jobids):
        errors.append("microbench.runs: jobid values must be unique")


def _check_benchmark_files(data: dict, errors: list[str]) -> None:
    """Validate the per-algorithm parameter files.

    A missing file means the benchmark would build with the wrong (or no)
    allocator configuration, so it is a configuration error. The example's
    picongpu command lines come from config.json itself (the structured
    `flag_lines`, validated in `_check_examples`), so there is no flag file
    to check.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
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
    delays = data.get("delays")
    if not isinstance(delays, dict):
        # An absent `delays` section is baseline-only: the (0, 0) pair only.
        return ["0_0"]
    baseline = delays.get("baseline", [0, 0])
    arms_spec = delays.get("arms") or {}
    arms = arms_spec.get("initial", arms_spec.get("values", [])) if phase == "initial" else arms_spec.get("values", [])
    joint = (delays.get("joint") or {}).get("values", []) or [] if phase == "arms" else []
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
    _check_examples(data, errors)
    _check_algorithms(data, errors)
    _check_delays(data, errors)
    _check_build(data, errors)
    _check_machines(data, errors)
    _check_microbench(data, errors)
    _check_people(data, errors)
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


def _cmd_flag_lines(rest: list[str]) -> None:
    """Print the serialized picongpu command lines of one example.

    Args:
        rest: the command line arguments after the "flag-lines" subcommand.

    """
    if len(rest) != 1:
        _fail("usage: config.py flag-lines <Example>")
    name = rest[0]
    data = _load()
    for example in data["examples"]:
        if isinstance(example, dict) and example.get("name") == name:
            for line in flag_lines(example["flag_lines"]):
                print(line)
            return
    _fail(f"example '{name}' not found")


def _dispatch_key(command: str, rest: list[str]) -> None:
    """Dispatch one `get` or `list` over a dotted key path to its own output.

    `run-matrix` and the examples list (objects now) have special printers;
    everything else is a scalar (`get`) or a list of scalars (`list`).

    Args:
        command: the subcommand ("get" or "list").
        rest: the command line arguments after the subcommand.

    """
    if rest and rest[0] == "run-matrix":
        if command != "list":
            _fail("'run-matrix' is a list, not a scalar key")
        _cmd_list_run_matrix(rest)
        return
    if len(rest) != 1:
        _fail(f"usage: config.py {command} <dotted.key>")
    dotted = rest[0]
    value = _lookup(_load(), dotted)
    if dotted == "examples":
        # The examples are objects; `list examples` prints the names only
        # (the Makefile's contract) and `get` is not for a list of objects.
        if command == "get":
            _fail("'examples' is a list of objects; use `list examples`")
        for example in value:
            print(example["name"] if isinstance(example, dict) else example)
        return
    if command == "get":
        _print_scalar(value, dotted)
    else:
        _print_list(value, dotted)


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
        _fail("usage: config.py {get|list|check|flag-lines} ...")
    command, rest = args[0], args[1:]
    if command == "check":
        if rest:
            _fail("usage: config.py check")
        _cmd_check()
        return 0
    if command == "flag-lines":
        _cmd_flag_lines(rest)
        return 0
    if command not in {"get", "list"}:
        _fail(f"unknown command '{command}' (expected get, list, flag-lines or check)")
    _dispatch_key(command, rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
