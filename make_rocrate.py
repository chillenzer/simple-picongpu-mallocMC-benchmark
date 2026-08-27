"""Generate and check the RO-Crate metadata of this repository.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

The repository root is the crate root. `create` writes the RO-Crate
metadata file (default `ro-crate-metadata.json`, git-ignored like
`output/results.h5`) describing the benchmark as one research object,
rebuilt from its ground truth on every invocation:

- The harness as a workflow (the RO-Crate *Workflows and scripts*
  conventions and the Workflow RO-Crate profile): the `Makefile` is the
  main workflow, the shell and python scripts its steps, the make
  variables its input parameters, the run results its outputs.
- The benchmark runs as provenance (the Process Run profile): one
  `CreateAction` per grid-run log of the sweep machines' output
  directories, reading the log's self-describing `# metadata:` line —
  instrument the built binary, object the flags/config/parameter/
  profile files, `environment` the imposed delays and the slurm job,
  agent the run's user, result the log. Runs are append-only, so every
  vintage of a re-run is described; the one whose identity's run stamp
  lists it is the current vintage, the others are annotated as
  superseded.
- The analysis as actions: one `CreateAction` for `output/results.h5`
  (from the run logs and the frozen legacy table) and one per figure.

Three subcommands: `create` (write the metadata file), `check` (validate
a metadata file; built-in checks plus, when the `rocrate` package is
installed, its official ones) and `zip` (pack the full crate into a
verified `.crate.zip` archive). The metadata is a derived artifact: it
is generated after the fact, never written during a run, and every fact
is best effort, like `logmeta.py`'s — a missing source is omitted,
never an error. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] (probes git by absolute path only)
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None

try:
    from rocrate.rocrate import ROCrate
except ImportError:
    ROCrate = None

CRATE_URI = "https://w3id.org/ro/crate/1.3"
CRATE_CONTEXT = "https://w3id.org/ro/crate/1.3/context"
WORKFLOW_RUN_CONTEXT = "https://w3id.org/ro/terms/workflow-run"
PROCESS_RUN_PROFILE = "https://w3id.org/ro/wfrun/process/0.4"
WORKFLOW_PROFILE = "https://w3id.org/workflowhub/workflow-ro-crate/1.0"
LICENSE = "https://spdx.org/licenses/MIT"
METADATA_DEFAULT = "ro-crate-metadata.json"
RESULTS = "output/results.h5"
LEGACY_H5 = "legacy/legacy_results.h5"
STAMPS = "run-stamps"
UNAVAILABLE = "unavailable"

# The identity of one run, parsed from the run log's file name
# (example, algorithm, malloc delay, free delay, repetition).
type Identity = tuple[str, str, str, str, str]

# The harness steps the main workflow (the Makefile) executes; each must
# be a file of the repository (missing ones are skipped).
STEPS = [
    "make_rocrate.py",
    "run_stamp.sh",
    "run_folder.sh",
    "logmeta.py",
    "config.py",
    "log_setup_hal.sh",
    "log_run_hal.sh",
    "log_setup_rosi.sh",
    "log_run_rosi.sh",
    "log_setup_rosi_a100.sh",
    "log_run_rosi_a100.sh",
    "analysis/compute_results.py",
    "analysis/run_logs.py",
    "analysis/results_io.py",
    "analysis/allocation_model.py",
    "analysis/summarize_results.py",
    "analysis/plot_sweeps.py",
    "analysis/plot_shared_fits.py",
    "analysis/plot_foil_lct.py",
    "analysis/plot_kelvin_helmholtz.py",
    "legacy/make_legacy_results.py",
    "legacy/log_meta.py",
    "legacy/move_legacy_logs.sh",
]

# The main workflow's input parameters, as make variables: (name, type,
# valueRequired, description).
PARAMS = [
    ("MACHINE", "Text", True, "the sweep machine (a key of the config machines table)"),
    ("PROFILE", "Text", False, "the build profile (make build)"),
    ("PARAM_DIR", "Text", False, "the parameter overlay directory (make build)"),
    ("REPEATS", "Integer", False, "the number of full-sweep repetitions (make runs)"),
    ("REP", "Integer", False, "one repetition only (make runs; the slurm case)"),
    ("PHASE", "Text", False, "the sweep phase, initial or arms (make runs)"),
]


def ref(entity_id: str) -> dict:
    """Return a JSON-LD reference to `entity_id`.

    Args:
        entity_id: the referenced entity's `@id`.

    Returns:
        dict: the reference, `{"@id": entity_id}`.

    """
    return {"@id": entity_id}


def slug(text: str) -> str:
    """Return `text` as a safe `#`-fragment suffix.

    Args:
        text: the text to slugify.

    Returns:
        str: the slugged text (`/` and whitespace replaced).

    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", text)


def git_state() -> tuple[str, bool | None]:
    """Return the repository state (the HEAD commit and the dirtiness).

    Mirrors `logmeta._git_state` (the crate must agree with the run
    logs' metadata about the repository state).

    Returns:
        tuple: `commit` (the full 40-hex HEAD) and `dirty` (True/False),
            the commit at its "unavailable" placeholder when it cannot
            be determined.

    """
    commit, dirty = UNAVAILABLE, None
    git = shutil.which("git")
    if git is not None:
        rev_probe = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if rev_probe.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", rev_probe.stdout.strip()):
            commit = rev_probe.stdout.strip()
        status_probe = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [git, "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if status_probe.returncode == 0:
            dirty = bool(status_probe.stdout.strip())
    return commit, dirty


def load_config() -> dict:
    """Return the parsed `config.json`.

    Returns:
        dict: the configuration.

    Raises:
        SystemExit: if the file is missing or not a JSON object (the
            script is to be run from the repository root).

    """
    try:
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        msg = "make_rocrate.py: config.json not found or invalid; run from the repository root"
        raise SystemExit(msg) from None
    if not isinstance(config, dict):
        msg = "rocrate.py: config.json must be a JSON object"
        raise SystemExit(msg) from None
    return config


def first_description(path: Path) -> str | None:
    """Return a one-line description of a script file.

    The first docstring line of a python file, the first comment line
    (after the shebang and the SPDX header) of a shell file; `None`
    when the file carries none.

    Args:
        path: the script file to inspect.

    Returns:
        str | None: the first descriptive line, or `None`.

    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if path.suffix == ".py":
            if stripped.startswith(("'''", '"""')):
                docstring = stripped[3:]
                if docstring:
                    return docstring
                for next_line in lines[index + 1 :]:
                    if next_line.strip():
                        return next_line.strip()
                return None
        elif stripped.startswith("#") and not stripped.startswith("#!") and "SPDX-" not in stripped:
            text = stripped.lstrip("#").strip()
            if text:
                return text
    return None


def log_header(path: Path) -> tuple[str, dict] | None:
    """Return the run one-liner and metadata of a schema-1 run log.

    Args:
        path: the log file to inspect.

    Returns:
        tuple | None: the (`# run:` one-liner, metadata dict) of a run
            log (a `# run:` line plus a `# metadata:` JSON line of
            schema 1 and kind `run`), or `None` for any other file
            (session logs, pre-redesign logs, ...).

    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            runline = handle.readline().rstrip("\n")
            metaline = handle.readline().rstrip("\n")
    except OSError:
        return None
    if not runline.startswith("# run: ") or not metaline.startswith("# metadata: "):
        return None
    try:
        metadata = json.loads(metaline[len("# metadata: ") :])
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict) or metadata.get("schema") != 1 or metadata.get("kind") != "run":
        return None
    return runline, metadata


def identity(label: str, name: str) -> Identity | None:
    """Split a run log's file name into its run identity.

    Mirrors `analysis/compute_results.py::_identity_stamp_path`, so the
    crate keys the vintage state on the same identity make and the
    analysis use.

    Args:
        label: the sweep machine's label.
        name: the run log's file name
            (`run_<label>_<Ex>_<Algo>_m<M>_f<F>_r<I>_<...>.txt`).

    Returns:
        Identity | None: (example, algorithm, malloc delay, free
            delay, repetition), or `None` when the name carries no
            identity.

    """
    pattern = rf"^run_{re.escape(label)}_(?P<rest>.+)_m(?P<m>\d+)_f(?P<f>\d+)_r(?P<rep>\d+)_.*$"
    match = re.match(pattern, name)
    if match is None:
        return None
    example, _, algorithm = match["rest"].partition("_")
    if not example or not algorithm:
        return None
    return example, algorithm, match["m"], match["f"], match["rep"]


def run_stamp_path(label: str, identity: Identity) -> Path:
    """Return the run stamp path of one run identity.

    Args:
        label: the sweep machine's label.
        identity: the run's identity (example, algorithm, malloc
            delay, free delay, repetition).

    Returns:
        Path: the stamp path (not necessarily existing).

    """
    example, algorithm, malloc_ns, free_ns, rep = identity
    return Path(STAMPS) / label / example / algorithm / f"{malloc_ns}_{free_ns}" / f"rep-{rep}.stamp"


def stamp_state(stamp: Path, relpath: str) -> str:
    """Return the vintage state of one run log, keyed on its run stamp.

    The identity's stamp lists the log paths of its current vintage
    (`run_stamp.sh` rewrites it on every re-run), so a stamp not listing
    the log means the log is superseded; an identity without a stamp
    (fresh runs, or after `make clean-runs`) has no superseded vintages.

    Args:
        stamp: the run identity's stamp path.
        relpath: the log's path relative to the repository root.

    Returns:
        str: "current", "superseded" or "unstamped".

    """
    if not stamp.is_file():
        return "unstamped"
    listed = {line.strip() for line in stamp.read_text(encoding="utf-8").splitlines() if line.strip()}
    return "current" if relpath in listed else "superseded"


def vintage_text(state: str, stamp: Path) -> str:
    """Return the vintage annotation for a run action's description.

    Args:
        state: the `stamp_state` result.
        stamp: the run identity's stamp path.

    Returns:
        str: the annotation, e.g.
            "current vintage (the run stamp run-stamps/... lists this log)".

    """
    if state == "current":
        return f"current vintage (the run stamp {stamp} lists this log)"
    if state == "superseded":
        return f"superseded vintage (the run stamp {stamp} lists a newer vintage)"
    return "no run stamp for this identity (fresh runs, or after make clean-runs)"


def mtime_iso(path: Path) -> str:
    """Return a file's modification time as an ISO 8601 string.

    Args:
        path: the file to stat.

    Returns:
        str: the ISO 8601 string, the empty string when the file cannot
            be stat'ed.

    """
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    except OSError:
        return ""


class Crate:
    """Accumulate the crate's JSON-LD entities, keyed by `@id`.

    Also carries the build context of one generation: the config's
    `people` table (how a run's user is resolved to a Person entity) and
    the logins that resolution left without an ORCID.
    """

    def __init__(self) -> None:
        """Initialize an empty crate."""
        self._entities: dict[str, dict] = {}
        self.action_ids: list[str] = []
        self.people: dict = {}
        self.unresolved: set[str] = set()

    def has(self, entity_id: str) -> bool:
        """Return whether `entity_id` is in the crate.

        Args:
            entity_id: the entity id to look for.

        Returns:
            bool: whether the crate already carries the entity.

        """
        return entity_id in self._entities

    def add(self, entity: dict) -> None:
        """Add one entity (a duplicate `@id` is kept once).

        Args:
            entity: the entity, carrying its `@id`.

        """
        entity_id = entity["@id"]
        if entity_id not in self._entities:
            self._entities[entity_id] = entity
            if entity.get("@type") == "CreateAction":
                self.action_ids.append(entity_id)

    def data_paths(self, metadata_name: str) -> list[str]:
        """Return the referenced data-entity paths, in order.

        Args:
            metadata_name: the metadata file's name (excluded; it is
                the crate's own descriptor, not a data entity).

        Returns:
            list: the relative file and directory paths of the data
                entities (the contextual `#` fragments and IRI ids are
                not paths).

        """
        excluded = {metadata_name, "./"}
        return [
            entity_id
            for entity_id in self._entities
            if entity_id not in excluded and not entity_id.startswith("#") and "://" not in entity_id
        ]

    def graph(self, metadata_name: str) -> list[dict]:
        """Return the `@graph` (metadata descriptor and root first).

        Args:
            metadata_name: the metadata file's name (the descriptor's
                `@id`).

        Returns:
            list: the entities in logical order (JSON-LD order is not
                significant).

        """
        heads = [self._entities[entity_id] for entity_id in (metadata_name, "./") if entity_id in self._entities]
        heads_ids = {metadata_name, "./"}
        rest = [entity for entity_id, entity in self._entities.items() if entity_id not in heads_ids]
        return heads + rest


def add_file(
    crate: Crate, path: str, *, encoding: str | None = None, description: str | None = None, about: str | None = None
) -> bool:
    """Add a data entity for the file at `path`, if it exists.

    Args:
        crate: the crate to add it to.
        path: the file's path relative to the repository root.
        encoding: the IANA media type, when known.
        description: the entity's description.
        about: the id of the entity this file is about (e.g. `./`).

    Returns:
        bool: whether the entity was added.

    """
    file = Path(path)
    if not file.is_file():
        return False
    entity = {"@id": path, "@type": "File", "name": file.name}
    if encoding:
        entity["encodingFormat"] = encoding
    if description:
        entity["description"] = description
    if about:
        entity["about"] = ref(about)
    crate.add(entity)
    return True


def main() -> int:
    """Run the command line interface.

    Returns:
        int: the process exit status.

    """
    parser = argparse.ArgumentParser(description="Generate and check the RO-Crate metadata of this repository.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    create = subcommands.add_parser("create", help="write the crate metadata file")
    create.add_argument("--out", default=METADATA_DEFAULT, help="destination metadata file (default: %(default)s)")

    check = subcommands.add_parser(
        "check",
        help=(
            "validate a crate metadata file (built-in checks; the official rocrate package's load, "
            "when installed, as a warning)"
        ),
    )
    check.add_argument("metadata", nargs="?", default=METADATA_DEFAULT, help="the metadata file to check")

    zip_cmd = subcommands.add_parser("zip", help="pack the full crate into a verified .crate.zip archive")
    zip_cmd.add_argument("--out", default="ro-crate.crate.zip", help="destination archive (default: %(default)s)")

    args = parser.parse_args()
    if args.command == "create":
        return create_main(args.out)
    if args.command == "zip":
        return zip_main(args.out)
    return check_main(Path(args.metadata))


def _build_crate(metadata_name: str) -> tuple[Crate, list[str], dict]:
    """Run all the crate builders.

    Args:
        metadata_name: the metadata file's name the metadata descriptor
            should carry.

    Returns:
        tuple: (the crate with every entity, the note lines, the stats
            (run action count, superseded count, described machines)).

    """
    config = load_config()
    commit, dirty = git_state()
    crate = Crate()
    people = config.get("people")
    crate.people = people if isinstance(people, dict) else {}
    notes: list[str] = []
    run_log_ids: list[str] = []

    _add_context(crate)
    _add_harness(crate, commit, dirty=dirty)
    run_actions, superseded, machines_described = _add_machines(crate, config, notes, run_log_ids)
    _add_analysis(crate, run_log_ids, notes)
    _add_root(crate, config, metadata_name)
    stats = {"run_actions": run_actions, "superseded": superseded, "machines": machines_described}
    return crate, notes, stats


def create_main(out: str) -> int:
    """Generate the crate metadata file.

    Args:
        out: the destination path, relative to the repository root.

    Returns:
        int: the process exit status.

    """
    crate, notes, stats = _build_crate(Path(out).name)
    metadata = {"@context": [CRATE_CONTEXT, WORKFLOW_RUN_CONTEXT], "@graph": crate.graph(Path(out).name)}
    Path(out).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out}: {len(crate._entities)} entities, {stats['run_actions']} run actions "
        f"({stats['superseded']} superseded), {len(stats['machines'])} machine(s) described"
    )
    for note in notes:
        print(f"note: {note}")
    return 0


def _human_size(size: float) -> str:
    """Return a byte count in a human-readable form.

    Args:
        size: the size in bytes.

    Returns:
        str: e.g. "14.2 MB".

    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def zip_main(out: str) -> int:
    """Pack the full crate into a verified `.crate.zip` archive.

    The archive carries the crate's metadata (staged under the
    conventional name) and every data file the crate references, so
    unzipping it yields a valid crate root (the RO-Crate packaging
    convention). Because the metadata and the files are regenerated and
    staged together, the archive is then verified by unzipping it into
    a scratch directory and re-running the checks on the result.

    Args:
        out: the destination archive path.

    Returns:
        int: the process exit status.

    """
    crate, notes, _stats = _build_crate(METADATA_DEFAULT)
    for note in notes:
        print(f"note: {note}")
    stage = Path(tempfile.mkdtemp(prefix="rocrate-stage-"))
    verify_root = Path(tempfile.mkdtemp(prefix="rocrate-verify-"))
    try:
        missing = _stage_crate(crate, stage)
        if missing:
            for path in missing:
                print(f"error: the crate references {path}, which does not exist")
            return 1
        total_uncompressed = _write_zip(stage, out)
        status, total_compressed, files = _verify_zip(Path(out), verify_root)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(verify_root, ignore_errors=True)
    if status != 0:
        print(f"error: {out} failed verification on its own content; the archive is kept for inspection")
        return 1
    ratio = total_uncompressed / total_compressed if total_compressed else 1.0
    print(
        f"wrote {out}: {files} file(s); {_human_size(total_compressed)} "
        f"(uncompressed {_human_size(total_uncompressed)}, {ratio:.0f}x); "
        "verified: unzips to a valid crate"
    )
    return 0


def _stage_crate(crate: Crate, stage: Path) -> list[str]:
    """Copy the crate's data files and the metadata under `stage`.

    Args:
        crate: the crate whose data entities are staged.
        stage: the scratch directory to stage into.

    Returns:
        list: the referenced paths that do not exist (nothing is staged
            then, so the caller can report and abort).

    """
    paths = crate.data_paths(METADATA_DEFAULT)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        return missing
    for path in paths:
        source = Path(path)
        if source.is_dir():
            shutil.copytree(source, stage / path, dirs_exist_ok=True)
            continue
        (stage / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stage / path)
    metadata = {"@context": [CRATE_CONTEXT, WORKFLOW_RUN_CONTEXT], "@graph": crate.graph(METADATA_DEFAULT)}
    (stage / METADATA_DEFAULT).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return []


def _write_zip(stage: Path, out: str) -> int:
    """Write the staged crate as a deterministic `.crate.zip` archive.

    Args:
        stage: the staged crate root.
        out: the destination archive path.

    Returns:
        int: the staged content's size in bytes (uncompressed).

    """
    total = 0
    # A fixed entry timestamp keeps the archive byte-stable for a given
    # crate (research artifacts should rebuild identically).
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(stage.rglob("*")):
            if not file.is_file():
                continue
            info = zipfile.ZipInfo(file.relative_to(stage).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (file.stat().st_mode & 0o7777) << 16
            with file.open("rb") as handle:
                archive.writestr(info, handle.read())
            total += file.stat().st_size
    return total


def _verify_zip(out: Path, verify_root: Path) -> tuple[int, int, int]:
    """Unzip `out` into `verify_root` and re-run the checks on the result.

    Args:
        out: the archive to verify.
        verify_root: the scratch directory to unzip into.

    Returns:
        tuple: (the checks' exit status, the archive's size in bytes,
            the archive's entry count).

    """
    with zipfile.ZipFile(out) as archive:
        files = len(archive.namelist())
        archive.extractall(verify_root)
    status = check_main(verify_root / METADATA_DEFAULT)
    return status, out.stat().st_size, files


def _add_context(crate: Crate) -> None:
    """Add the profile, language, organization and license entities.

    Args:
        crate: the crate to add them to.

    """
    crate.add(
        {
            "@id": "https://schema.org/CompletedActionStatus",
            "@type": "ActionStatusEntity",
            "name": "Completed",
        }
    )
    process_run = {
        "@id": PROCESS_RUN_PROFILE,
        "@type": ["CreativeWork", "Profile"],
        "name": "Process Run Crate",
        "version": "0.4",
    }
    crate.add(process_run)
    workflow_profile = {
        "@id": WORKFLOW_PROFILE,
        "@type": ["CreativeWork", "Profile"],
        "name": "Workflow RO-Crate",
        "version": "1.0",
    }
    crate.add(workflow_profile)
    crate.add({"@id": "#python", "@type": "ComputerLanguage", "name": "Python 3", "url": "https://www.python.org/"})
    crate.add(
        {"@id": "#bash", "@type": "ComputerLanguage", "name": "Bash", "url": "https://www.gnu.org/software/bash/"}
    )
    crate.add(
        {
            "@id": "#gnu-make",
            "@type": "ComputerLanguage",
            "name": "GNU Make",
            "url": "https://www.gnu.org/software/make/",
        }
    )
    crate.add(
        {
            "@id": "#hzdr",
            "@type": "Organization",
            "name": "Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf",
            "url": "https://www.hzdr.de/",
        }
    )
    crate.add({"@id": LICENSE, "@type": "CreativeWork", "name": "MIT License", "alternateName": "MIT"})


def _makefile_entity(commit: str, *, dirty: bool | None) -> dict:
    """Return the Makefile's workflow entity.

    Args:
        commit: the repository's HEAD commit (the harness's version).
        dirty: the repository's dirtiness at generation time.

    Returns:
        dict: the Makefile entity (its `hasPart` and `creator` are
            filled in later).

    """
    entity = {
        "@id": "Makefile",
        "@type": ["File", "SoftwareSourceCode", "ComputationalWorkflow"],
        "name": "PIConGPU/mallocMC benchmark harness",
        "description": (
            "Builds the pinned PIConGPU with the pinned mallocMC per (example, algorithm), sweeps the "
            "run matrix per repetition into the append-only run logs (one self-describing log per grid "
            "run), and computes the numbers (output/results.h5) and figures from them. `make rocrate` "
            "generates this crate's metadata."
        ),
        "programmingLanguage": ref("#gnu-make"),
        "license": ref(LICENSE),
        "input": [ref(f"#param-{name}") for name, _, _, _ in PARAMS],
        "output": [ref("#param-run-logs"), ref("#param-results-h5"), ref("#param-figures")],
    }
    if commit != UNAVAILABLE:
        entity["version"] = commit
        if dirty is True:
            entity["description"] += " (the tree was dirty at crate generation time)"
    return entity


def _add_harness_steps(crate: Crate) -> list[str]:
    """Add each existing harness step as a source-code entity, and return them.

    Args:
        crate: the crate to add them to.

    Returns:
        list: the step paths, as the Makefile entity's `hasPart` targets.

    """
    steps = []
    for step in STEPS:
        if not Path(step).is_file():
            continue
        language = "#python" if step.endswith(".py") else "#bash"
        entity = {
            "@id": step,
            "@type": ["File", "SoftwareSourceCode"],
            "name": Path(step).name,
            "programmingLanguage": ref(language),
        }
        description = first_description(Path(step))
        if description:
            entity["description"] = description
        crate.add(entity)
        steps.append(step)
    return steps


def _add_harness_files(crate: Crate) -> None:
    """Add the harness's configuration, document and environment files.

    Args:
        crate: the crate to add them to.

    """
    add_file(
        crate,
        "config.json",
        encoding="application/json",
        description="the single source of truth for what the harness builds and runs "
        "(examples, algorithms, delay sweep, dependency pins, build flags, machines)",
    )
    add_file(crate, "README.md", encoding="text/markdown", about="./")
    if Path("flags").is_dir():
        for flags in sorted(Path("flags").glob("*.flags")):
            add_file(
                crate,
                str(flags),
                encoding="text/plain",
                description=f"one picongpu command line per run (the {flags.stem} example)",
            )
    if Path("param").is_dir():
        for param_file in sorted(Path("param").rglob("*.param")):
            add_file(crate, str(param_file), description="parameter overlay for the build (mallocMC configuration)")
    if Path("profiles").is_dir():
        for profile in sorted(Path("profiles").glob("*")):
            if profile.is_file():
                add_file(
                    crate,
                    str(profile),
                    description="HPC environment profile (modules, PIC_BACKEND, PICSRC) for one machine",
                )
    for env_file in ("environment.yml", "requirements.txt", "conda-lock.yml", "conda-linux-64.lock", "pyproject.toml"):
        add_file(crate, env_file, description="the committed analysis environment, pins, or code style")


def _add_harness(crate: Crate, commit: str, *, dirty: bool | None) -> None:
    """Add the harness as a workflow: the Makefile, its steps, its parameters.

    Args:
        crate: the crate to add them to (its `people` build context is
            the workflow's creators).
        commit: the repository's HEAD commit (the harness's version).
        dirty: the repository's dirtiness at generation time.

    """
    makefile = _makefile_entity(commit, dirty=dirty)
    makefile["hasPart"] = [ref(step) for step in _add_harness_steps(crate)]
    makefile["creator"] = [ref(item) for item in _people_ids(crate)] + [ref("#hzdr")]
    crate.add(makefile)
    for name, parameter_type, value_required, description in PARAMS:
        crate.add(
            {
                "@id": f"#param-{name}",
                "@type": "FormalParameter",
                "name": name,
                "additionalType": parameter_type,
                "valueRequired": value_required,
                "description": description,
            }
        )
    crate.add(
        {
            "@id": "#param-run-logs",
            "@type": "FormalParameter",
            "name": "run logs",
            "additionalType": "Dataset",
            "description": "the sweep machines' append-only grid-run logs, one directory per machine",
        }
    )
    crate.add(
        {
            "@id": "#param-results-h5",
            "@type": "FormalParameter",
            "name": "results",
            "additionalType": "File",
            "description": "output/results.h5, the single results table of the analysis",
        }
    )
    crate.add(
        {
            "@id": "#param-figures",
            "@type": "FormalParameter",
            "name": "figures",
            "additionalType": "Dataset",
            "description": "the analysis figures (figures/)",
        }
    )
    _add_harness_files(crate)


def _add_machines(crate: Crate, config: dict, notes: list[str], run_log_ids: list[str]) -> tuple[int, int, list[str]]:
    """Add one dataset, log entities and run actions per sweep machine.

    Args:
        crate: the crate to add them to (its `unresolved` build context
            becomes the note line about logins without an ORCID).
        config: the parsed `config.json`.
        notes: the note lines to return to the caller (appended in place).
        run_log_ids: the run log entity ids, for the analysis action
            (appended in place).

    Returns:
        tuple: (run action count, superseded vintage count, described
            machine labels).

    """
    run_actions = 0
    superseded = 0
    machines_described: list[str] = []
    machines = config.get("machines")
    if not isinstance(machines, dict):
        notes.append("no machines table in config.json; no run logs described")
        return 0, 0, machines_described
    for label, machine in machines.items():
        if not isinstance(machine, dict) or not isinstance(machine.get("output"), str):
            continue
        outdir = Path(machine["output"])
        if not outdir.is_dir():
            notes.append(f"machine {label!r}: output directory {outdir} does not exist; no log described")
            continue
        machines_described.append(label)
        actions_here, superseded_here = _machine_logs(crate, label, machine.get("hardware", label), outdir, run_log_ids)
        run_actions += actions_here
        superseded += superseded_here
    if crate.unresolved:
        notes.append(f"no ORCID resolvable for login(s) {', '.join(sorted(crate.unresolved))}; recorded by login only")
    return run_actions, superseded, machines_described


def _person(crate: Crate, login: str, orcid: str | None = None, name: str | None = None) -> str:
    """Add (or find) the Person entity of one login, and return its id.

    With an ORCID iD the entity id is the ORCID URL (the dereferenceable,
    persistent identity); without one it is the login-only fragment.

    Args:
        crate: the crate to add it to.
        login: the login the run happened as.
        orcid: the resolved ORCID iD, or `None`.
        name: the display name, when known.

    Returns:
        str: the person's entity id.

    """
    if orcid:
        crate_id = f"https://orcid.org/{orcid}"
        if not crate.has(crate_id):
            crate.add({"@id": crate_id, "@type": "Person", "name": name or login, "identifier": login})
        return crate_id
    crate_id = f"#user-{slug(login)}"
    if not crate.has(crate_id):
        crate.add({"@id": crate_id, "@type": "Person", "name": name or login})
    return crate_id


def _people_ids(crate: Crate) -> list[str]:
    """Create every configured person's entity, and return the entity ids.

    Args:
        crate: the crate (the people table is its build context).

    Returns:
        list: the person entity ids, in config order.

    """
    ids: list[str] = []
    for login, person in crate.people.items():
        if not isinstance(person, dict):
            continue
        orcid = person.get("orcid")
        if not isinstance(orcid, str) or not orcid:
            orcid = None
        name = person.get("name")
        if not isinstance(name, str) or not name:
            name = None
        ids.append(_person(crate, str(login), orcid, name))
    return ids


def _machine_logs(crate: Crate, label: str, hardware: str, outdir: Path, run_log_ids: list[str]) -> tuple[int, int]:
    """Describe one machine's run log directory, and add its dataset entity.

    Args:
        crate: the crate to add them to.
        label: the sweep machine's label.
        hardware: the machine's hardware title (config).
        outdir: the machine's output directory.
        run_log_ids: the run log entity ids (appended in place).

    Returns:
        tuple: (run action count, superseded vintage count) of this machine.

    """
    run_actions = 0
    superseded = 0
    file_ids: list[str] = []
    first_meta: dict | None = None
    for file in sorted(outdir.iterdir()):
        if not file.is_file():
            continue
        relpath = f"{outdir}/{file.name}"
        file_ids.append(relpath)
        header = log_header(file)
        if header is None:
            add_file(crate, relpath, encoding="text/plain")
            continue
        runline, metadata = header
        add_file(crate, relpath, encoding="text/plain", description=runline[len("# run: ") :])
        action, superseded_vintage = run_action(crate, label, metadata, relpath)
        if action is not None:
            crate.add(action)
            run_log_ids.append(relpath)
            run_actions += 1
            superseded += superseded_vintage
        if first_meta is None:
            first_meta = metadata
    sessions = outdir / "sessions"
    if sessions.is_dir():
        crate.add(
            {
                "@id": f"{outdir}/sessions/",
                "@type": "Dataset",
                "name": "sessions",
                "description": "the launchers' session logs (a free-text backup of the launch "
                "environment; never parsed, never superseded)",
            }
        )
        file_ids.append(f"{outdir}/sessions/")
    dataset = {
        "@id": f"{outdir}/",
        "@type": "Dataset",
        "name": f"{label} ({hardware}) run logs",
        "description": (
            f"the {label} sweep machine's append-only grid-run logs (one self-describing log per "
            "grid run per (combination, repetition); a re-run adds a new vintage, nothing is "
            f"ever removed). {run_actions} run(s) described, {superseded} superseded "
            "vintage(s); a run's identity's run stamp lists its current vintage's logs."
        ),
        "hasPart": [ref(item) for item in file_ids],
    }
    if first_meta is not None:
        _machine_entity(crate, label, hardware, first_meta)
        dataset["mentions"] = [ref(f"#machine-{label}")]
    crate.add(dataset)
    return run_actions, superseded


def _machine_entity(crate: Crate, label: str, hardware: str, metadata: dict) -> None:
    """Add the machine's equipment entity, from one of its run metadata.

    Args:
        crate: the crate to add it to.
        label: the sweep machine's label.
        hardware: the machine's hardware title (config).
        metadata: one of the machine's run metadata (best-effort host
            facts).

    """
    hw = metadata.get("hw")
    if not isinstance(hw, dict):
        hw = {}
    gpus = hw.get("gpu")
    if isinstance(gpus, list):
        gpus = ", ".join(str(gpu) for gpu in gpus if isinstance(gpu, str) and gpu != UNAVAILABLE) or UNAVAILABLE
    else:
        gpus = UNAVAILABLE
    description = (
        f"host {metadata.get('hostname', UNAVAILABLE)}; os {hw.get('os', UNAVAILABLE)}; "
        f"cpu {hw.get('cpu', UNAVAILABLE)}; gpus {gpus}; gpu driver {hw.get('gpu_driver', UNAVAILABLE)}"
    )
    crate.add(
        {
            "@id": f"#machine-{label}",
            "@type": "IndividualProduct",
            "name": f"{label} ({hardware})",
            "identifier": label,
            "description": description,
        }
    )


def _command_line(run: dict, name: str) -> str:
    """Return the run's command line, with its imposed delays prefixed.

    Args:
        run: the metadata's run block.
        name: the run log's file name (the command placeholder).

    Returns:
        str: the command line, `MALLOCMC_MALLOC_DELAY=<m>
            MALLOCMC_FREE_DELAY=<f> ` prefixed when the run block
            carries both delays.

    """
    command = str(run.get("command", name))
    delays = run.get("delays")
    if isinstance(delays, list) and len(delays) == 2 and all(isinstance(delay, int) for delay in delays):
        command = f"MALLOCMC_MALLOC_DELAY={delays[0]} MALLOCMC_FREE_DELAY={delays[1]} {command}"
    return command


def _environment_ids(crate: Crate, action_id: str, delays: object, slurm_job: object) -> list[str]:
    """Add the action's PropertyValue environment entities; return their ids.

    Args:
        crate: the crate to add them to.
        action_id: the action's id (the entities nest under it).
        delays: the run's imposed delays (a two-element list, or absent).
        slurm_job: the run's slurm job id (a string, or absent).

    Returns:
        list: the environment entity ids (empty when there is none).

    """
    environment_ids: list[str] = []
    for index, delay in enumerate(delays if isinstance(delays, list) else []):
        if not isinstance(delay, int):
            continue
        env_id = f"{action_id}#{'malloc' if index == 0 else 'free'}-delay"
        crate.add(
            {
                "@id": env_id,
                "@type": "PropertyValue",
                "name": f"MALLOCMC_{'MALLOC' if index == 0 else 'FREE'}_DELAY",
                "value": str(delay),
            }
        )
        environment_ids.append(env_id)
    if isinstance(slurm_job, str) and slurm_job:
        env_id = f"{action_id}#slurm-job"
        crate.add({"@id": env_id, "@type": "PropertyValue", "name": "SLURM_JOB_ID", "value": slurm_job})
        environment_ids.append(env_id)
    return environment_ids


def _resolve_person(crate: Crate, user: str, metadata: dict) -> tuple[str | None, str | None]:
    """Resolve one run's user to (the ORCID iD, the display name).

    The run's own metadata carries the most faithful identity; the
    config people table (the crate's build context) covers runs recorded
    before it existed.

    Args:
        crate: the crate (the people table is its build context).
        user: the login the run happened as.
        metadata: the log's `# metadata:` dict (schema 1, kind `run`).

    Returns:
        tuple: (the ORCID iD or `None`, the display name or `None`).

    """
    entry = crate.people.get(user)
    orcid = metadata.get("orcid")
    if not isinstance(orcid, str) or not orcid or orcid == UNAVAILABLE:
        orcid = None
    if orcid is None and isinstance(entry, dict):
        table_orcid = entry.get("orcid")
        if isinstance(table_orcid, str) and table_orcid:
            orcid = table_orcid
    name = entry.get("name") if isinstance(entry, dict) else None
    if not isinstance(name, str) or not name:
        name = None
    return orcid, name


def _attach_agent(crate: Crate, action: dict, metadata: dict) -> None:
    """Attach the run's agent (the user the run happened as) to `action`.

    A login without a resolvable ORCID is recorded by login only and
    added to `crate.unresolved` (`_add_machines` turns them into a note
    line).

    Args:
        crate: the crate (the Person entity is added to it, and its
            `unresolved` logins are tracked).
        action: the CreateAction to attach the agent to.
        metadata: the log's `# metadata:` dict (schema 1, kind `run`).

    """
    user = metadata.get("user")
    if not isinstance(user, str) or not user or user == UNAVAILABLE:
        return
    orcid, name = _resolve_person(crate, user, metadata)
    action["agent"] = ref(_person(crate, user, orcid, name))
    if orcid is None:
        crate.unresolved.add(user)


def run_action(crate: Crate, label: str, metadata: dict, relpath: str) -> tuple[dict | None, int]:
    """Build the CreateAction of one grid-run log (and its entities).

    Args:
        crate: the crate (the action's referents are added to it).
        label: the sweep machine's label.
        metadata: the log's `# metadata:` dict (schema 1, kind `run`).
        relpath: the log's path relative to the repository root.

    Returns:
        tuple: (the action, or `None` when the log's file name carries
            no identity at all, 1 when the action is a superseded
            vintage and 0 otherwise).

    """
    run = metadata.get("run")
    if not isinstance(run, dict):
        return None, 0
    name = Path(relpath).name
    match = identity(label, name)
    if match is None:
        return None, 0
    state = stamp_state(run_stamp_path(label, match), relpath)
    example, algorithm = match[0], match[1]
    stem = name[: -len(".txt")]
    action_id = f"#run-{slug(stem)}"
    binary_id = _binary_entity(crate, metadata, run)

    action = {
        "@id": action_id,
        "@type": "CreateAction",
        "name": f"{example}/{algorithm} (run log {name})",
        "description": f"{_command_line(run, name)}; vintage: {vintage_text(state, run_stamp_path(label, match))}",
        "actionStatus": ref("https://schema.org/CompletedActionStatus"),
        "instrument": ref(binary_id),
        "object": [
            ref(item)
            for item in ("config.json", f"flags/{example}.flags", f"param/{algorithm}/mallocMC.param")
            if Path(item).is_file()
        ],
        "result": ref(relpath),
    }
    start_time = metadata.get("ts")
    if isinstance(start_time, str) and start_time:
        action["startTime"] = start_time
    end_time = mtime_iso(Path(relpath))
    if end_time:
        action["endTime"] = end_time
    environment_ids = _environment_ids(crate, action_id, run.get("delays"), metadata.get("slurm_job"))
    if environment_ids:
        action["environment"] = [ref(env_id) for env_id in environment_ids]
    _attach_agent(crate, action, metadata)
    return action, 1 if state == "superseded" else 0


def _build_facts(build: object) -> list[str]:
    """Return the human-readable build facts of a metadata build block.

    Args:
        build: the metadata's build dict (absent or partial allowed).

    Returns:
        list: the facts in display order (empty when there is none).

    """
    if not isinstance(build, dict):
        return []
    facts: list[str] = []
    compiler = str(build.get("compiler", UNAVAILABLE))
    if compiler != UNAVAILABLE:
        facts.append(f"compiler {compiler.splitlines()[0]}")
    for key, title in (
        ("cuda", "CUDA"),
        ("build_type", "build type"),
        ("cxx_flags", "CXX flags"),
        ("cuda_flags", "CUDA flags"),
    ):
        value = build.get(key)
        if isinstance(value, str) and value != UNAVAILABLE:
            facts.append(f"{title} {value}")
    return facts


def _dependency_entities(crate: Crate, pins: object) -> tuple[list[str], str | None]:
    """Add the pinned dependencies' entities, and return their ids.

    Args:
        crate: the crate to add them to.
        pins: the metadata's pins dict (absent allowed).

    Returns:
        tuple: (the dependency entity ids in pin order, the full
            picongpu pin or `None` when absent).

    """
    if not isinstance(pins, dict):
        return [], None
    requirements: list[str] = []
    picongpu_pin: str | None = None
    for name in ("picongpu", "mallocmc"):
        pin = pins.get(name)
        if pin in {None, UNAVAILABLE}:
            continue
        requirements.append(_dependency_entity(crate, name, pin))
        if name == "picongpu":
            picongpu_pin = str(pin)
    return requirements, picongpu_pin


def _binary_entity(crate: Crate, metadata: dict, run: dict) -> str:
    """Add (or find) the binary the run used, and return its entity id.

    Args:
        crate: the crate to add it to.
        metadata: the run's metadata.
        run: the metadata's run block.

    Returns:
        str: the binary's entity id.

    """
    binary = metadata.get("binary")
    if not isinstance(binary, dict):
        binary = {}
    sha256 = binary.get("sha256")
    if not isinstance(sha256, str) or sha256 == UNAVAILABLE:
        return _unknown_binary(crate)
    crate_id = f"#binary-{sha256[:8]}"
    if crate.has(crate_id):
        return crate_id
    entity = {
        "@id": crate_id,
        "@type": "SoftwareApplication",
        "name": f"PIConGPU binary of {run.get('setup', '?')}/{run.get('algorithm', '?')} (sha256 {sha256[:12]})",
        "identifier": sha256,
    }
    facts = _build_facts(metadata.get("build"))
    if facts:
        entity["description"] = "; ".join(facts)
    requirements, picongpu_pin = _dependency_entities(crate, metadata.get("pins"))
    if picongpu_pin is not None:
        entity["softwareVersion"] = picongpu_pin
    if requirements:
        entity["softwareRequirements"] = [ref(item) for item in requirements]
    crate.add(entity)
    return crate_id


def _unknown_binary(crate: Crate) -> str:
    """Return the unknown-binary entity id (added once).

    Args:
        crate: the crate.

    Returns:
        str: the entity id.

    """
    crate_id = "#binary-unknown"
    if not crate.has(crate_id):
        crate.add(
            {
                "@id": crate_id,
                "@type": "SoftwareApplication",
                "name": "PIConGPU binary (sha256 unavailable in the run's metadata)",
            }
        )
    return crate_id


def _dependency_entity(crate: Crate, name: str, pin: str) -> str:
    """Add (or find) a pinned dependency's entity, and return its id.

    Args:
        crate: the crate to add it to.
        name: the config dependencies key ("picongpu" / "mallocmc").
        pin: the full pinned hash.

    Returns:
        str: the entity id.

    """
    crate_id = f"#dep-{name}-{pin[:8]}"
    if crate.has(crate_id):
        return crate_id
    title = "PIConGPU" if name == "picongpu" else "mallocMC"
    source = (
        "https://github.com/ComputationalRadiationPhysics/picongpu"
        if name == "picongpu"
        else "https://github.com/chillenzer/mallocMC"
    )
    crate.add({"@id": crate_id, "@type": "SoftwareApplication", "name": title, "url": source, "version": pin})
    return crate_id


def _add_analysis(crate: Crate, run_log_ids: list[str], notes: list[str]) -> None:
    """Add the results/legacy/figures entities and the analysis actions.

    Args:
        crate: the crate to add them to.
        run_log_ids: the run log entity ids (the results action's inputs).
        notes: the note lines to return to the caller (appended in place).

    """
    legacy_exists = add_file(
        crate,
        LEGACY_H5,
        encoding="application/x-hdf5",
        description="the frozen pre-redesign benchmark table (each frozen run is superseded by nothing), "
        "with a per-file SHA-256 source manifest",
    )
    if not legacy_exists:
        notes.append(f"{LEGACY_H5} does not exist; it is not described (build it with `make legacy-results`)")
    results_exists = add_file(
        crate,
        RESULTS,
        encoding="application/x-hdf5",
        description=(
            "the single results table of the analysis (runs, group statistics, fits, baselines, figure metadata)"
        ),
    )
    if not results_exists:
        notes.append(f"{RESULTS} does not exist; the analysis action is not described (run `make results`)")
        _add_figure_actions(crate, results_exists=False)
        return
    _results_action(crate, run_log_ids, legacy_exists=legacy_exists)
    _add_figure_actions(crate, results_exists=True)


def _results_action(crate: Crate, run_log_ids: list[str], *, legacy_exists: bool) -> None:
    """Add the CreateAction that computes `output/results.h5`.

    Args:
        crate: the crate to add it to.
        run_log_ids: the run log entity ids (the action's log inputs).
        legacy_exists: whether the frozen legacy table is described.

    """
    created, git_commit, sources = _results_attrs(Path(RESULTS))
    object_ids = [ref(item) for item in run_log_ids] + [ref("config.json")]
    if legacy_exists:
        object_ids.append(ref(LEGACY_H5))
    action = {
        "@id": "#analysis-results-h5",
        "@type": "CreateAction",
        "name": f"compute {RESULTS} from the run logs and the frozen legacy table",
        "description": (
            "parses the run logs (all vintages; superseded runs stay in every table and number — "
            "the current state of the world is runs[superseded] == 0) and the frozen legacy table"
        ),
        "instrument": ref("analysis/compute_results.py"),
        "object": object_ids,
        "result": ref(RESULTS),
    }
    if created:
        action["startTime"] = created
    end_time = mtime_iso(Path(RESULTS))
    if end_time:
        action["endTime"] = end_time
    if git_commit:
        action["description"] += f"; git commit {git_commit[:12]}"
    if sources:
        action["description"] += f"; sources {sources}"
    crate.add(action)


def _add_figure_actions(crate: Crate, *, results_exists: bool) -> None:
    """Add each figure's data entity, and its CreateAction when resolvable.

    Args:
        crate: the crate to add them to.
        results_exists: whether `output/results.h5` is described (the
            figures' actions read it).

    """
    if not Path("figures").is_dir():
        return
    for figure in sorted(Path("figures").glob("*.pdf")):
        instrument = _figure_instrument(figure.name)
        description = (
            "an analysis figure, computed from results.h5"
            if instrument is not None
            else "a historical analysis figure (not produced by the current harness scripts)"
        )
        add_file(crate, str(figure), encoding="application/pdf", description=description)
        if not results_exists or instrument is None or not Path(instrument).is_file():
            continue
        end_time = mtime_iso(figure)
        if not end_time:
            continue
        crate.add(
            {
                "@id": f"#figure-{figure.stem}",
                "@type": "CreateAction",
                "name": f"compute figures/{figure.name} from results.h5",
                "instrument": ref(instrument),
                "object": ref(RESULTS),
                "result": ref(str(figure)),
                "startTime": end_time,
                "endTime": end_time,
            }
        )


def _results_attrs(path: Path) -> tuple[str, str, str]:
    """Return the results file's recorded provenance attributes.

    Args:
        path: the results file.

    Returns:
        tuple: (created_utc, git_commit, sources); the empty string for
            each fact that is unavailable (`h5py` not installed, the
            file unreadable, or the attribute absent).

    """
    if h5py is None:
        return "", "", ""
    try:
        with h5py.File(path, "r") as file:
            attrs = {key: str(value) for key, value in file.attrs.items()}
    except OSError:
        return "", "", ""
    return attrs.get("created_utc", ""), attrs.get("git_commit", ""), attrs.get("sources", "")


def _figure_instrument(name: str) -> str | None:
    """Return the plotting script that produced a figure, by its file name.

    Args:
        name: the figure's file name.

    Returns:
        str | None: the script path, or `None` for a figure the harness
            does not plot.

    """
    if not name.endswith(".pdf"):
        return None
    if name == "foil_lct.pdf":
        return "analysis/plot_foil_lct.py"
    if name == "kelvin_helmholtz.pdf":
        return "analysis/plot_kelvin_helmholtz.py"
    if name.startswith("sweeps-shared-"):
        return "analysis/plot_shared_fits.py"
    if name.startswith("sweeps-"):
        return "analysis/plot_sweeps.py"
    return None


def _machine_output_refs(machines: object) -> list[dict]:
    """Return references to the existing machine output directories.

    Args:
        machines: the config's machines table (absent or partial allowed).

    Returns:
        list: one `ref` per machine whose output directory exists.

    """
    if not isinstance(machines, dict):
        return []
    return [
        ref(f"{machine['output']}/")
        for machine in machines.values()
        if isinstance(machine, dict) and isinstance(machine.get("output"), str) and Path(machine["output"]).is_dir()
    ]


def _add_root(crate: Crate, config: dict, metadata_name: str) -> None:
    """Add the root data entity and the metadata descriptor.

    Args:
        crate: the crate to add them to (its `people` build context is
            the crate's creators).
        config: the parsed `config.json` (the machine output directories).
        metadata_name: the metadata file's name (the descriptor's id).

    """
    parts: list[dict] = [ref("Makefile")]
    parts.extend(ref(path) for path in ("README.md", "config.json") if Path(path).is_file())
    parts.extend(_machine_output_refs(config.get("machines")))
    if Path(RESULTS).is_file():
        parts.append(ref(RESULTS))
    if Path(LEGACY_H5).is_file():
        parts.append(ref(LEGACY_H5))
    if Path("figures").is_dir():
        parts.extend(ref(str(figure)) for figure in sorted(Path("figures").glob("*.pdf")))
    root = {
        "@id": "./",
        "@type": "Dataset",
        "name": "PIConGPU Allocations using mallocMC",
        "description": (
            "The PIConGPU/mallocMC allocation-latency benchmark: the harness (the Makefile build/run "
            "orchestration, the run and analysis scripts, the configuration) and its benchmark data — "
            "the sweep machines' append-only grid-run logs (a re-run adds a new vintage, nothing is "
            "ever removed, and a run's identity's run stamp lists its current vintage's logs), the "
            "results and figures computed from them, and the frozen pre-redesign table. Superseded "
            "vintages stay in every number; the current state of the world is runs[superseded] == 0."
        ),
        "license": ref(LICENSE),
        "creator": [ref(item) for item in _people_ids(crate)] + [ref("#hzdr")],
        "mainEntity": ref("Makefile"),
        "hasPart": parts,
        "mentions": [ref(action_id) for action_id in crate.action_ids],
        "conformsTo": [ref(PROCESS_RUN_PROFILE), ref(WORKFLOW_PROFILE)],
    }
    crate.add(root)
    crate.add(
        {
            "@id": metadata_name,
            "@type": "CreativeWork",
            "conformsTo": ref(CRATE_URI),
            "about": ref("./"),
        }
    )


def _check_context(metadata: dict, errors: list[str]) -> None:
    """Check the @context carries the RO-Crate context, appending problems.

    Args:
        metadata: the parsed metadata file.
        errors: accumulates the problems found.

    """
    contexts = metadata.get("@context")
    if isinstance(contexts, str):
        contexts = [contexts]
    if not isinstance(contexts, list) or CRATE_CONTEXT not in contexts:
        errors.append(f"the @context does not carry the RO-Crate context {CRATE_CONTEXT}")


def _index_graph(graph: list, errors: list[str]) -> dict[str, dict]:
    """Index the @graph entities by their `@id`, appending problems.

    Args:
        graph: the metadata's @graph list.
        errors: accumulates the problems found.

    Returns:
        dict: the entities, keyed by `@id` (duplicates kept last).

    """
    entities: dict[str, dict] = {}
    for index, entity in enumerate(graph):
        if not isinstance(entity, dict) or "@id" not in entity:
            errors.append(f"graph element {index} is not an entity with an @id")
            continue
        entity_id = str(entity["@id"])
        if entity_id in entities:
            errors.append(f"duplicate @id {entity_id!r}")
        entities[entity_id] = entity
    return entities


def _check_descriptor(errors: list[str], metadata_path: Path, entities: dict[str, dict]) -> None:
    """Check the metadata descriptor entity, appending problems.

    Args:
        errors: accumulates the problems found.
        metadata_path: the metadata file (the descriptor's `@id` name).
        entities: the graph entities, keyed by `@id`.

    """
    descriptor = entities.get(metadata_path.name)
    if descriptor is None:
        errors.append(f"no metadata descriptor entity for {metadata_path.name!r}")
        return
    if "./" not in _refs(descriptor.get("about")):
        errors.append("the metadata descriptor's about does not point to the root data entity ./")
    if CRATE_URI not in _refs(descriptor.get("conformsTo")):
        errors.append("the metadata descriptor does not conform to the RO-Crate specification")


def _check_root_entity(errors: list[str], entities: dict[str, dict]) -> None:
    """Check the root data entity, appending problems.

    Args:
        errors: accumulates the problems found.
        entities: the graph entities, keyed by `@id`.

    """
    root_entity = entities.get("./")
    if root_entity is None:
        errors.append("no root data entity ./")
        return
    types = root_entity.get("@type")
    if "Dataset" not in (types if isinstance(types, list) else [types]):
        errors.append("the root data entity is not a Dataset")
    main_entity = _refs(root_entity.get("mainEntity"))
    if not main_entity or main_entity[0] not in entities:
        errors.append("the root data entity has no resolvable mainEntity (the workflow profile requires one)")
    errors.extend(
        f"the root's conformsTo target {profile_id} has no contextual entity"
        for profile_id in _refs(root_entity.get("conformsTo"))
        if profile_id not in entities
    )


def _check_create_action(errors: list[str], entity_id: str, entity: dict, entities: dict[str, dict]) -> None:
    """Check one CreateAction's fields, appending problems.

    Args:
        errors: accumulates the problems found.
        entity_id: the action's `@id`.
        entity: the action entity.
        entities: the graph entities, keyed by `@id`.

    """
    errors.extend(
        f"action {entity_id} has no {required}"
        for required in ("instrument", "result")
        if not _refs(entity.get(required))
    )
    for timestamp in ("startTime", "endTime"):
        value = entity.get(timestamp)
        if not isinstance(value, str):
            continue
        try:
            datetime.fromisoformat(value)
        except ValueError:
            errors.append(f"action {entity_id}'s {timestamp} {value!r} is not ISO 8601")
    for env_id in _refs(entity.get("environment")):
        env_entity = entities.get(env_id)
        if env_entity is not None and (not env_entity.get("name") or env_entity.get("value") is None):
            errors.append(f"environment entity {env_id} has no name/value")


def _check_entity(errors: list[str], root: Path, entity_id: str, entity: dict, entities: dict[str, dict]) -> None:
    """Check one entity's references, fields and on-disk existence.

    Args:
        errors: accumulates the problems found.
        root: the crate root directory (the data entities' base).
        entity_id: the entity's `@id`.
        entity: the entity.
        entities: the graph entities, keyed by `@id`.

    """
    errors.extend(
        f"entity {entity_id} references {target}, which is not in the crate"
        for target in _referenced(entity)
        # The core spec URI (the metadata descriptor's conformsTo) is a
        # fixed identifier that carries no contextual entity; only the
        # root's profile conformsTo targets must be described.
        if target != CRATE_URI and target not in entities
    )
    if entity.get("@type") == "CreateAction":
        _check_create_action(errors, entity_id, entity, entities)
    if entity_id == "./" or entity_id.startswith("#") or "://" in entity_id:
        return
    if not (root / entity_id).exists():
        kind = "directory" if entity_id.endswith("/") else "file"
        errors.append(f"data entity {entity_id} is not an existing {kind} in the crate")


def check_main(metadata_path: Path) -> int:
    """Validate one crate metadata file.

    Args:
        metadata_path: the metadata file to check (its directory is the
            crate root).

    Returns:
        int: the process exit status (1 on any error).

    """
    root = metadata_path.parent
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {metadata_path}: {exc}")
        return 1
    graph = metadata.get("@graph")
    if not isinstance(graph, list):
        print("error: the metadata has no @graph list")
        return 1
    errors: list[str] = []
    _check_context(metadata, errors)
    entities = _index_graph(graph, errors)
    _check_descriptor(errors, metadata_path, entities)
    _check_root_entity(errors, entities)
    for entity_id, entity in entities.items():
        _check_entity(errors, root, entity_id, entity, entities)
    for error in errors:
        print(f"error: {error}")
    # The official package's load is advisory only: its newest release
    # supports crate versions up to 1.2, so it cannot even load a correct
    # 1.3 crate. The built-in checks are the gate.
    for message in _official_check(root):
        print(f"warning (rocrate package): {message}")
    if errors:
        print(f"FAIL: {len(errors)} error(s) in {metadata_path}")
        return 1
    actions = sum(1 for entity in entities.values() if entity.get("@type") == "CreateAction")
    print(f"OK: {metadata_path} is valid ({len(entities)} entities, {actions} actions)")
    return 0


def _refs(value: object) -> list:
    """Return every @id in a JSON-LD property value, however nested.

    Args:
        value: a JSON-LD property value.

    Returns:
        list: the @ids in it, in order (duplicates kept).

    """
    ids: list = []
    if isinstance(value, dict):
        if isinstance(value.get("@id"), str):
            ids.append(value["@id"])
    elif isinstance(value, list):
        for item in value:
            ids.extend(_refs(item))
    return ids


def _referenced(entity: dict) -> list:
    """Return every entity id `entity`'s properties reference.

    Args:
        entity: the entity to inspect.

    Returns:
        list: the referenced ids (duplicates removed, order kept).

    """
    seen: list = []
    for value in entity.values():
        for target in _refs(value):
            if target not in seen:
                seen.append(target)
    return seen


def _official_check(root: Path) -> list[str]:
    """Run the official `rocrate` package's crate loading, when installed.

    The package carries no separate validator; loading the crate is its
    structural validation (metadata file, metadata descriptor, root data
    entity, entity dereferencing), and it raises on anything it cannot
    make sense of.

    Args:
        root: the crate root directory.

    Returns:
        list: the package's error messages (empty when the package is
            not installed or the crate loads cleanly).

    """
    if ROCrate is None:
        return []
    try:
        ROCrate(str(root))
    except Exception as exc:  # ruff: ignore[blind-except] (third-party API of unknown failure modes)
        return [f"the official rocrate package could not load the crate: {exc}"]
    return []


if __name__ == "__main__":
    sys.exit(main())
