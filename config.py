"""Read and validate `config.json`, the single source of harness configuration.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The build harness (the Makefile and the per-machine log_*.sh launchers)
parses no config format itself; it calls this helper to look up individual
values:

    python3 config.py get machines.hal.hardware
    python3 config.py list examples
    python3 config.py list run-matrix [initial|arms]
    python3 config.py list commits
    python3 config.py list configs <Algorithm>
    python3 config.py list build-matrix
    python3 config.py commit <name> <picongpu|mallocmc> <url|hash|path>
    python3 config.py check
    python3 config.py flag-lines <Example>
    python3 config.py render-param <Algorithm> <config-name>

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

 The allocator's C++ is likewise data: each `configs.<Algorithm>.<name>`
 object carries the heap scalars (`heap`) and a named hash profile
 (`hash.profile`, a file under `param/<Algorithm>/profiles/`), and
 `render-param` renders the algorithm's `param/<Algorithm>/mallocMC.param.in`
 template into the `mallocMC.param` overlay from those two, the only place
 the harness's allocator C++ is generated. The `// heap-args:` header line of
 the template lists the template parameters in order, so the heap scalars
 bind to them by (case-insensitive) name; no C++ ever appears in config.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NoReturn

CONFIG_PATH = Path("config.json")

# The phases of the two-stage sweep: the fast first scan and the full
# design (the Makefile's PHASE invocation value takes the same keys).
SWEEP_PHASES = {"initial", "arms"}


def _fail(message: str) -> NoReturn:
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


def _check_dep_spec(dep: object, dotted: str, errors: list[str]) -> None:
    """Validate one dependency pin ({url, hash, path?}).

    Args:
        dep: the dependency mapping (checked here).
        dotted: the dotted key path (for the error messages).
        errors: accumulates the problems found.

    """
    if not isinstance(dep, dict):
        errors.append(f"{dotted}: must be a {{url, hash, path?}} mapping")
        return
    url = dep.get("url")
    if not isinstance(url, str) or not url:
        errors.append(f"{dotted}.url: must be a non-empty string")
    path = dep.get("path")
    if path is not None and (not isinstance(path, str) or not path):
        errors.append(f"{dotted}.path: must be a non-empty string or omitted (defaults to src/<commit>/...)")
    hash_ = dep.get("hash")
    if not isinstance(hash_, str) or not re.fullmatch(r"[0-9a-f]{40}", hash_):
        errors.append(f"{dotted}.hash: must be a 40-hex git commit")


def _check_commit_entry(commit: object, dotted: str, names: set[str], errors: list[str]) -> None:
    """Validate one `commits` entry ({name, picongpu, mallocmc}).

    Args:
        commit: the commit mapping (checked here).
        dotted: the dotted key path (for the error messages).
        names: the commit names seen so far (accumulated for the uniqueness
            check).
        errors: accumulates the problems found.

    """
    if not isinstance(commit, dict):
        errors.append(f"{dotted}: each commit must be a {{name, picongpu, mallocmc}} mapping")
        return
    name = commit.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{dotted}.name: must be a non-empty string")
    elif name in names:
        errors.append(f"{dotted}.name: commit name '{name}' is not unique")
    else:
        names.add(name)
    for dep in ("picongpu", "mallocmc"):
        _check_dep_spec(commit.get(dep), f"{dotted}.{dep}", errors)


def _check_commits(data: dict, errors: list[str]) -> None:
    """Validate the `commits` list (the dependency pairs to build and run).

    A `commit` is one {name, picongpu, mallocmc} pair: the logical commit
    name the harness builds and runs, and the two dependency pins it uses.
    When `commits` is absent the single legacy pair in `dependencies` is
    synthesised as one commit `default` (a config with both is an error).

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    commits = _walk(data, "commits")
    dependencies = _walk(data, "dependencies")
    if commits is not None and dependencies is not None:
        errors.append("use either 'commits' or the legacy 'dependencies' block, not both")
        return
    if commits is not None:
        if not isinstance(commits, list) or not commits:
            errors.append("commits: must be a non-empty list of {name, picongpu, mallocmc}")
            return
        names: set[str] = set()
        for index, commit in enumerate(commits):
            _check_commit_entry(commit, f"commits[{index}]", names, errors)
        _check_commit_path_collisions(commits, errors)
        return
    # Legacy single pair: validate it when present (a config with neither is
    # still valid until the build needs the pins; the Makefile reads commits).
    if dependencies is not None:
        for dep in ("picongpu", "mallocmc"):
            _check_dep_spec(
                dependencies.get(dep) if isinstance(dependencies, dict) else None, f"dependencies.{dep}", errors
            )


def _commit_dep_path(commit: object, dep: str) -> str | None:
    """Return one commit's resolved dependency path, defaulting it.

    Mirrors `_commit_field` for `field == "path"` (the same defaulting the
    Makefile applies via `config.py commit <name> <dep> path`), so the
    collision check sees exactly the paths the build would use: an explicit
    `path` when present, else `src/<name>/picongpu` /
    `src/<name>/picongpu/thirdParty/mallocMC`.

    Args:
        commit: a valid commit mapping ({name, picongpu, mallocmc}).
        dep: the dependency ("picongpu" or "mallocmc").

    Returns:
        str | None: the resolved path, or `None` when the dep is absent or
        the commit's name is unusable.

    """
    spec = commit.get(dep) if isinstance(commit.get(dep), dict) else {}
    path = spec.get("path")
    name = commit.get("name")
    base = name if (isinstance(name, str) and name) else ""
    if isinstance(path, str) and path:
        return path
    if not base:
        return None
    return f"src/{base}/picongpu" if dep == "picongpu" else f"src/{base}/picongpu/thirdParty/mallocMC"


def _check_commit_path_collisions(commits: list, errors: list[str]) -> None:
    """Validate that no two commits resolve to the same dependency path.

    The harness keeps one `src/...` checkout per (commit, dependency) — the
    `.dep-stamp` files and the source drivers are keyed on the resolved
    path. Two commits resolving to the same path (a shared `src/picongpu`,
    and therefore a shared nested `thirdParty/mallocMC`) collapses their
    stamp targets (make warns "overriding recipe"), and their builds clobber
    each other's checkouts, so the second commit cannot really be built.
    Each commit needs its own path (e.g. `src/<name>/picongpu`).

    Args:
        commits: the validated `commits` list (each a {name, picongpu,
            mallocmc} mapping).
        errors: accumulates the problems found.

    """
    for dep in ("picongpu", "mallocmc"):
        seen: dict[str, str] = {}
        for index, commit in enumerate(commits):
            if not isinstance(commit, dict):
                continue
            path = _commit_dep_path(commit, dep)
            if path is None:
                continue
            key = path.rstrip("/")
            name = commit.get("name")
            label = f"commits[{index}]" if not (isinstance(name, str) and name) else f"commits[{index}] ({name})"
            owner = seen.get(key)
            if owner is None:
                seen[key] = label
            else:
                errors.append(
                    f"{label}: `{dep}.path` resolves to '{path}', which "
                    f"collides with {owner} — the harness keeps one checkout per "
                    f"commit, so two commits must not share a path; give each "
                    f"commit its own (e.g. `src/<name>/picongpu`, "
                    f"`src/<name>/picongpu/thirdParty/mallocMC`) or drop one."
                )


def _commit_field(data: dict, commit_name: str, dep: str, field: str) -> str:
    """Resolve one dependency field of one commit, defaulting its path.

    Args:
        data: the parsed configuration.
        commit_name: the logical commit name to look up.
        dep: the dependency ("picongpu" or "mallocmc").
        field: the field ("url", "hash", or "path").

    Returns:
        str: the field's value, the path defaulted when omitted
        (`src/<commit>/picongpu` / `.../thirdParty/mallocMC`).

    """
    for commit in effective_commits(data):
        if commit.get("name") != commit_name:
            continue
        spec = commit.get(dep) if isinstance(commit.get(dep), dict) else {}
        value = spec.get(field)
        if field == "path" and (value is None or not value):
            if dep == "picongpu":
                return f"src/{commit_name}/picongpu"
            return f"src/{commit_name}/picongpu/thirdParty/mallocMC"
        if isinstance(value, str) and value:
            return value
        _fail(f"commit '{commit_name}' has no {dep}.{field}")
    _fail(f"commit '{commit_name}' not found")


def _cmd_commit_field(rest: list[str]) -> None:
    """Print one dependency field of one commit (`commit <name> <dep> <field>`).

    Args:
        rest: the command line arguments after the "commit" subcommand.

    """
    if len(rest) != 3 or rest[1] not in {"picongpu", "mallocmc"} or rest[2] not in {"url", "hash", "path"}:
        _fail("usage: config.py commit <commit-name> <picongpu|mallocmc> <url|hash|path>")
    print(_commit_field(_load(), rest[0], rest[1], rest[2]))


def effective_commits(data: dict) -> list[dict]:
    """Return the commits to build and run (synthesising the legacy pair).

    Args:
        data: the parsed configuration.

    Returns:
        list: one {name, picongpu, mallocmc} mapping per commit, in
        config order. When `commits` is absent, a single commit `default`
        is synthesised from the legacy `dependencies` pair.

    """
    commits = _walk(data, "commits")
    if isinstance(commits, list) and commits:
        return [commit for commit in commits if isinstance(commit, dict)]
    dependencies = _walk(data, "dependencies")
    dependencies = dependencies if isinstance(dependencies, dict) else {}
    return [{"name": "default", **{dep: dependencies.get(dep, {}) for dep in ("picongpu", "mallocmc")}}]


HEAP_ARGS_LINE = "// heap-args: "
CONFIG_DEFAULT = "default"
# The heap scalars a `configs` block may carry when the `configs` key is
# absent (the pre-migration fallback, reconstructed from the static
# parameter files the migration replaces); they are the defaults every
# shipped `default` config spells out explicitly today, so a legacy config
# without `configs` still resolves to today's allocator.
LEGACY_HEAP: dict[str, dict] = {
    "FlatterScatter": {"accessBlockSize": 134217728, "pageSize": 131072, "wasteFactor": 2},
    "ScatterAlloc": {
        "pageSize": 2097152,
        "accessBlockSize": 2147483648,
        "regionSize": 16,
        "wasteFactor": 2,
        "resetFreedPages": True,
    },
    "Gallatin": {"bytesPerSegment": 16777216, "smallestSlice": 16, "largestSlice": 4096},
}
LEGACY_HASH: dict[str, str] = {"FlatterScatter": "FsHashDefault", "ScatterAlloc": "HashDefault"}


def _canonical(name: str) -> str:
    """Return the case-insensitive match key for a heap key (`name`).

    Args:
        name: the key to normalize.

    Returns:
        str: the canonical form (`aB` -> `ab`).

    """
    return name.lower().replace("_", "")


def effective_configs(data: dict, algorithm: str) -> dict[str, dict]:
    """Return the resolved config entries of one algorithm (+ their names, in order).

    Each key is a config name and each value the config's full entry
    (its ``heap`` mapping and, when present, its ``hash`` object). A config
    whose ``heap`` is absent is filled from the pre-migration defaults
    (``LEGACY_HEAP``); when the whole ``configs`` section is absent a single
    ``default`` entry is synthesised from those defaults and, for an
    algorithm with a hash slot, from the legacy hash profile.

    Args:
        data: the parsed configuration.
        algorithm: the algorithm name.

    Returns:
        dict: config name to config entry, in config order (a copy).

    """
    algorithms = _walk(data, "algorithms")
    if not isinstance(algorithms, list) or algorithm not in algorithms:
        _fail(f"algorithm '{algorithm}' is not one of the configured algorithms")
    raw = _walk(data, "configs")
    entries = raw.get(algorithm) if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        legacy = LEGACY_HEAP.get(algorithm)
        if legacy is None:
            _fail(f"algorithm '{algorithm}' has no configs.{algorithm} entries (add at least one)")
        entry = {"heap": dict(legacy)}
        profile = LEGACY_HASH.get(algorithm)
        if profile:
            entry["hash"] = {"profile": profile}
        return {CONFIG_DEFAULT: entry}
    return {name: _resolved_entry(algorithm, name, entry) for name, entry in entries.items()}


def _resolved_entry(algorithm: str, name: str, entry: object) -> dict:
    """Return one config entry with its ``heap`` filled in.

    Args:
        algorithm: the algorithm (for the error messages).
        name: the config name (for the error messages).
        entry: the raw config entry.

    Returns:
        dict: a copy of the entry; its ``heap`` defaults from ``LEGACY_HEAP``
        when missing.

    """
    dotted = f"configs.{algorithm}.{name}"
    if not isinstance(entry, dict):
        _fail(f"{dotted}: each config must be a {{heap, hash?}} mapping")
    resolved = dict(entry)
    heap = resolved.get("heap")
    if not isinstance(heap, dict):
        legacy = LEGACY_HEAP.get(algorithm, {})
        resolved["heap"] = dict(legacy)
    return resolved


def _hash_profile(algorithm: str, config: dict) -> str | None:
    """Return the named hash profile of a config entry, or the policy default.

    Args:
        algorithm: the algorithm (for the error messages).
        config: one ``{heap, hash?}`` config entry.

    Returns:
        str | None: the selected profile's file name (no ``.hpp``), or
        ``None`` when the entry has no ``hash`` object — the policy's own
        default hash type then stands in (and the Gallatin template, which
        has no hash slot, has no such placeholder to fill).

    """
    if "hash" not in config:
        return None
    hash_ = config.get("hash")
    profile = hash_.get("profile") if isinstance(hash_, dict) else None
    if not isinstance(profile, str) or not profile:
        _fail(f"configs.{algorithm}: a 'hash' entry needs a non-empty string 'profile'")
    return profile


def _fmt_scalar(value: object) -> str:
    """Render one heap scalar as a C++ non-type template argument.

    Args:
        value: a config.json heap scalar.

    Returns:
        str: an integer with a ``U`` suffix, or ``true`` / ``false``.

    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value) + "U"
    return str(value)


def _heap_arg_list(heap: dict, order: list[str], dotted: str) -> str:
    """Render the ``{heapArgs}`` template argument list in the declared order.

    Args:
        heap: the config's heap mapping.
        order: the ordered heap keys from the template's ``// heap-args:`` line.
        dotted: the dotted path (for the error messages).

    Returns:
        str: the comma-separated, U-suffixed / bool-valued template arguments.

    """
    by_key = {_canonical(key): value for key, value in heap.items()}
    args = []
    for token in order:
        key = by_key.get(_canonical(token))
        if key is None:
            _fail(f"{dotted}: the template wants heap argument '{token}' but the config has no such key")
        args.append(_fmt_scalar(key))
    return ", ".join(args)


def render_param(data: dict, algorithm: str, config_name: str) -> str:
    """Render one config of one algorithm into its `mallocMC.param` C++.

    Reads the ``param/<Algorithm>/mallocMC.param.in`` template, reads the
    ordered heap keys from its leading ``// heap-args:`` line, and substitutes
    every ``{...}`` placeholder: ``{heapArgs}`` is the full (ordered)
    template-argument list, every other name is a single heap scalar matched
    case-insensitively to a key of the config's ``heap``, and ``{hashProfile}``
    is the selected profile's type name (the policy's default type when the
    config has no ``hash`` object). Any leftover placeholder is an error.

    Args:
        data: the parsed configuration.
        algorithm: the algorithm name.
        config_name: the config to render.

    Returns:
        str: the rendered C++ (no trailing newline management).

    """
    template_path = Path("param") / algorithm / "mallocMC.param.in"
    if not template_path.is_file():
        _fail(f"template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    order = _heap_arg_order(template)
    if not order:
        _fail(f"{template_path}: no '{HEAP_ARGS_LINE.strip()}' header line")
    configs = effective_configs(data, algorithm)
    entry = configs.get(config_name)
    if entry is None:
        _fail(f"config '{config_name}' not found for {algorithm} (have: {', '.join(configs)})")
    heap = entry.get("heap")
    if not isinstance(heap, dict) or not heap:
        _fail(f"configs.{algorithm}.{config_name}.heap: must be a non-empty mapping of scalars")
    dotted = f"configs.{algorithm}.{config_name}"
    hash_profile = _hash_profile(algorithm, entry)
    substitutions = {"heapArgs": _heap_arg_list(heap, order, dotted)}
    for key, value in heap.items():
        substitutions[key] = _fmt_scalar(value)
        substitutions[_canonical(key)] = _fmt_scalar(value)
    if hash_profile is not None:
        substitutions["hashProfile"] = hash_profile
    rendered = template
    for name, value in substitutions.items():
        rendered = rendered.replace("{" + name + "}", value)
    unresolved = {match for match in _template_placeholders(template) if match not in substitutions}
    if unresolved:
        _fail(f"{algorithm}/{config_name}: unresolved placeholder(s) in the template: {', '.join(sorted(unresolved))}")
    return rendered


def _template_placeholders(template: str) -> list[str]:
    """Return the ``{name}`` placeholder names a template uses (comments stripped).

    Args:
        template: the template C++ text.

    Returns:
        list[str]: every placeholder name appearing in the template body.

    """
    stripped = re.sub(r"//[^\n]*", "", template)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    return re.findall(r"\{([A-Za-z_]\w*)\}", stripped)


def _heap_arg_order(template: str) -> list[str]:
    """Return the ordered heap keys from a template's ``// heap-args:`` line.

    Args:
        template: the template C++ text.

    Returns:
        list[str]: the ordered heap keys, or ``[]`` when the line is absent.

    """
    for line in template.splitlines():
        if line.startswith(HEAP_ARGS_LINE):
            return line.split(HEAP_ARGS_LINE, 1)[1].split()
    return []


def _check_configs(data: dict, errors: list[str]) -> None:
    """Validate the ``configs`` section (one entry per algorithm, per config).

    For each configured algorithm: at least one config named ``default``;
    every hash profile names an existing ``param/<Algo>/profiles/<name>.hpp``;
    and every placeholder the algorithm's template needs (the ``// heap-args:``
    keys, the per-key heap placeholders, and ``{hashProfile}`` when the
    template uses one) resolves for every config of that algorithm. An
    algorithm with no ``configs`` entry gets an implicit ``default`` (the
    pre-migration fallback in ``effective_configs``), which is checked too.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    algorithms = _walk(data, "algorithms")
    if not isinstance(algorithms, list):
        return
    raw = _walk(data, "configs")
    if raw is not None and not isinstance(raw, dict):
        errors.append("configs: must be a mapping of algorithm to configs")
        return
    for algorithm in algorithms:
        if not isinstance(algorithm, str):
            continue
        _check_configs_algorithm(algorithm, raw, errors)
        entry = raw.get(algorithm) if isinstance(raw, dict) else None
        if isinstance(entry, dict) and entry and CONFIG_DEFAULT not in entry:
            errors.append(f"configs.{algorithm}: must contain a config named 'default'")


# The template's placeholder requirements: (ordered // heap-args keys, the
# other per-key heap placeholders, whether a {hashProfile} is used).
TemplateReqs = tuple[list[str], set[str], bool]


def _check_configs_algorithm(algorithm: str, raw: object, errors: list[str]) -> None:
    """Validate one algorithm's configs and its template's placeholder coverage.

    Args:
        algorithm: the algorithm name.
        raw: the parsed ``configs`` mapping (or None when absent).
        errors: accumulates the problems found.

    """
    entry = raw.get(algorithm) if isinstance(raw, dict) else None
    if entry is None or not isinstance(entry, dict) or not entry:
        return  # implicit single default; nothing config-specific to check
    reqs = _template_requirements(_algorithm_template(algorithm))
    for config_name, config in entry.items():
        _check_one_config(algorithm, config_name, config, reqs, errors)


def _algorithm_template(algorithm: str) -> str | None:
    """Return the algorithm's template C++ text, or None when it is absent.

    Args:
        algorithm: the algorithm name.

    Returns:
        str | None: the template text, or None when the file does not exist.

    """
    template_path = Path("param") / algorithm / "mallocMC.param.in"
    return template_path.read_text(encoding="utf-8") if template_path.is_file() else None


def _template_requirements(template: str | None) -> TemplateReqs:
    """Return one template's placeholder requirements.

    Args:
        template: the template text (None-checked already by the caller).

    Returns:
        TemplateReqs: the ordered ``// heap-args:`` keys, the other heap
        scalar placeholder names, and whether a ``{hashProfile}`` is used.

    """
    if template is None:
        return [], set(), False
    placeholders = set(_template_placeholders(template))
    wanted = {name for name in placeholders if name not in {"heapArgs", "hashProfile"}}
    return _heap_arg_order(template), wanted, "hashProfile" in placeholders


def _check_one_config(algorithm: str, config_name: str, config: object, reqs: TemplateReqs, errors: list[str]) -> None:
    """Validate one algorithm/config entry (heap scalars, hash profile).

    Args:
        algorithm: the algorithm name.
        config_name: the config name.
        config: the raw config entry.
        reqs: the template's placeholder requirements.
        errors: accumulates the problems found.

    """
    dotted = f"configs.{algorithm}.{config_name}"
    if not isinstance(config, dict):
        errors.append(f"{dotted}: must be a {{heap, hash?}} mapping")
        return
    _check_config_heap(dotted, config, reqs, errors)
    _check_config_hash(algorithm, dotted, config, reqs, errors)


def _check_config_heap(dotted: str, config: dict, reqs: TemplateReqs, errors: list[str]) -> None:
    """Validate a config entry's `heap` (presence, scalarity, coverage).

    Args:
        dotted: the dotted path of the config entry (for the error messages).
        config: the config entry (a mapping).
        reqs: the template's placeholder requirements.
        errors: accumulates the problems found.

    """
    order, wanted_keys, _wants_hash = reqs
    heap = config.get("heap")
    if not isinstance(heap, dict) or not heap:
        errors.append(f"{dotted}.heap: must be a non-empty mapping of scalars")
        return
    for key, value in heap.items():
        if not isinstance(value, (int, bool)):
            errors.append(f"{dotted}.heap.{key}: must be an integer or a boolean")
    resolved = {_canonical(name) for name in heap}
    missing = [token for token in order + sorted(wanted_keys) if _canonical(token) not in resolved]
    if missing:
        errors.append(f"{dotted}.heap: missing key(s) for the template: {', '.join(missing)}")


def _check_config_hash(algorithm: str, dotted: str, config: dict, reqs: TemplateReqs, errors: list[str]) -> None:
    """Validate a config entry's `hash.profile` (against the hash slot).

    Args:
        algorithm: the algorithm name.
        dotted: the dotted path of the config entry (for the error messages).
        config: the config entry (a mapping).
        reqs: the template's placeholder requirements.
        errors: accumulates the problems found.

    """
    if "hash" not in config:
        return
    _order, _wanted, wants_hash = reqs
    if not wants_hash:
        errors.append(f"{dotted}.hash: no hash slot in the '{algorithm}' template, so no 'hash' key")
        return
    hash_ = config.get("hash")
    profile = hash_.get("profile", None) if isinstance(hash_, dict) else None
    if not isinstance(profile, str) or not profile:
        errors.append(f"{dotted}.hash.profile: must be a non-empty string")
        return
    if not (Path("param") / algorithm / "profiles" / (profile + ".hpp")).is_file():
        errors.append(f"{dotted}.hash.profile: no file param/{algorithm}/profiles/{profile}.hpp")


def _cmd_list_configs(algorithm: str) -> None:
    """Print one algorithm's config names, one per line (in config order).

    Args:
        algorithm: the algorithm name.

    """
    for name in effective_configs(_load(), algorithm):
        print(name)


def _cmd_build_matrix() -> None:
    """Print the build matrix: one `build/<commit>/<Ex>/<Algo>/<cfg>` per line.

    The quadruples (commit, example, algorithm, config) in build order, in
    the same order the Makefile expands them.

    """
    data = _load()
    examples = [example["name"] for example in data.get("examples", []) if isinstance(example, dict)]
    algorithms = data.get("algorithms", [])
    for commit in effective_commits(data):
        name = commit.get("name")
        for example in examples:
            for algorithm in algorithms:
                for config in effective_configs(data, algorithm):
                    print(f"build/{name}/{example}/{algorithm}/{config}")


def _check_build(data: dict, errors: list[str]) -> None:
    """Validate the build flags.

    The dependency pins are validated by `_check_commits`.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
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
    """Validate the per-algorithm parameter templates and profile headers.

    A missing template means the benchmark would build with the wrong (or no)
    allocator configuration, so it is a configuration error. The example's
    picongpu command lines come from config.json itself (the structured
    `flag_lines`, validated in `_check_examples`), so there is no flag file
    to check. The hash profile headers referenced by a config's `hash.profile`
    are checked by `_check_configs`, which also checks that every template
    placeholder resolves for every config of its algorithm.

    Args:
        data: the parsed configuration.
        errors: accumulates the problems found.

    """
    algorithms = _walk(data, "algorithms")
    if isinstance(algorithms, list):
        for algorithm in algorithms:
            if isinstance(algorithm, str):
                _missing_file(Path("param") / algorithm / "mallocMC.param.in", f"algorithms.{algorithm}", errors)


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
    _check_commits(data, errors)
    _check_configs(data, errors)
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


def _print_named_objects(items: list[object], dotted: str, command: str) -> None:
    """Print the `name` field of a list of objects (the `list commits` contract).

    Args:
        items: the list of objects to print the names of.
        dotted: the dotted key path (for the error message).
        command: the subcommand ("get" or "list").

    """
    if command == "get":
        _fail(f"'{dotted}' is a list of objects; use `list {dotted}`")
    for item in items:
        print(item["name"] if isinstance(item, dict) else item)


def _dispatch_key(command: str, rest: list[str]) -> None:
    """Dispatch one `get` or `list` over a dotted key path to its own output.

    `run-matrix` and the list-of-object keys (`commits`, `examples`) have
    special printers; everything else is a scalar (`get`) or a list of
    scalars (`list`).

    Args:
        command: the subcommand ("get" or "list").
        rest: the command line arguments after the subcommand.

    """
    if rest and rest[0] == "run-matrix":
        if command != "list":
            _fail("'run-matrix' is a list, not a scalar key")
        _cmd_list_run_matrix(rest)
        return
    if rest and rest[0] == "build-matrix":
        if command != "list":
            _fail("'build-matrix' is a list, not a scalar key")
        if len(rest) > 1:
            _fail("usage: config.py list build-matrix")
        _cmd_build_matrix()
        return
    if len(rest) != 1:
        _fail(f"usage: config.py {command} <dotted.key>")
    dotted = rest[0]
    if dotted == "commits":
        # A config with a legacy `dependencies` block still lists its single
        # synthesised commit `default`, so the names go through
        # `effective_commits` (which applies the synthesis).
        _print_named_objects(effective_commits(_load()), dotted, command)
        return
    value = _lookup(_load(), dotted)
    if dotted == "examples":
        # The examples are objects; `list examples` prints the names only
        # (the Makefile's contract).
        _print_named_objects(value, dotted, command)
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


def _cmd_render_param(rest: list[str]) -> None:
    """Render one config of one algorithm into its `mallocMC.param` C++ on stdout.

    Args:
        rest: the command line arguments after the "render-param" subcommand.

    """
    if len(rest) != 2:
        _fail("usage: config.py render-param <Algorithm> <config-name>")
    print(render_param(_load(), rest[0], rest[1]))


def main() -> int:
    """Dispatch the subcommand on the command line.

    Returns:
        int: the process exit status.

    """
    args = sys.argv[1:]
    if not args:
        _fail("usage: config.py {get|list|check|flag-lines|render-param} ...")
    command, rest = args[0], args[1:]
    if command == "check":
        if rest:
            _fail("usage: config.py check")
        _cmd_check()
        return 0
    if command == "flag-lines":
        _cmd_flag_lines(rest)
        return 0
    if command == "commit":
        _cmd_commit_field(rest)
        return 0
    if command == "render-param":
        _cmd_render_param(rest)
        return 0
    if command == "list" and rest and rest[0] == "configs":
        if len(rest) != 2:
            _fail("usage: config.py list configs <Algorithm>")
        _cmd_list_configs(rest[1])
        return 0
    if command not in {"get", "list"}:
        _fail(f"unknown command '{command}' (expected get, list, flag-lines, commit, render-param or check)")
    _dispatch_key(command, rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
