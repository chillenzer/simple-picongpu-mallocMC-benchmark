<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# PIConGPU Allocations using mallocMC

This repository benchmarks the runtime of PIConGPU simulations compiled with
different configurations of the [mallocMC](https://github.com/chillenzer/mallocMC)
device allocator. The setup benchmarks the **FlatterScatter**,
**ScatterAlloc** and **Gallatin** creation policies, each while sweeping
(malloc delay, free delay) combinations: nanosecond delays injected into
every allocation and every free request, implemented in mallocMC as a
busy-wait on the 64-bit device global timer (injected at the top of
`DeviceAllocator::malloc` / `DeviceAllocator::free`, independent of the
creation policy; the `__nanosleep` intrinsic's wake-up guarantee turned out
too weak and produced step-like sweeps) to study the impact of allocation
and free latency on simulation runtime.

## Benchmarked configuration

- **Allocation policies**: `FlatterScatter`, `ScatterAlloc`, `Gallatin`
  (the `algorithms` list in `config.json` and
  `param/<Algorithm>/mallocMC.param`)
- **Delay combinations** (nanoseconds, the `delays` section of `config.json`;
  the Makefile run targets derive the full combination set from it,
  `python3 config.py list run-matrix`): a `(0, 0)`
  baseline, the malloc-delay sweep with free delay 0 and the free-delay sweep
  with malloc delay 0 over the 12 log-spaced values `100`…`100000000`
  (1/4-decade steps, skipping the intermediate steps between `1e5` and
  `1e6`), plus a 3x3 joint grid over `10000`, `1000000`, `10000000` (34 runs
  per example and algorithm)
- **Examples**: `KelvinHelmholtz` (3D, grid sizes 128^3, 256x128x128,
  256x128x256, 1500 steps) and `FoilLCT` (2D, 256x1280 cells, 2000 steps)
  (the `examples` list in `config.json` and `flags/*.flags`)
- **One build per (example, algorithm)**: the creation policy is compiled
  in via `param/<Algorithm>/mallocMC.param`; the delays are not a
  compile-time option, mallocMC reads them at run time from the
  `MALLOCMC_MALLOC_DELAY` and `MALLOCMC_FREE_DELAY` environment variables
  (set by `run_folder.sh` for every run). Changing the sweep therefore never
  requires recompiling.
- **Layout**: `build/<Example>/<Algorithm>/` holds one build; its flag
  lines are run once per combination:
  `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`

The Makefile pins the dependency versions (the `dependencies` section of
`config.json`):

| dependency | source | pinned to |
|------------|--------|-----------|
| PIConGPU | ComputationalRadiationPhysics/picongpu | `6e7d58bb` |
| mallocMC | chillenzer/mallocMC, `add-delay` branch (`BOOST_LANG_*` guards compatible with PIConGPU's vendored alpaka 2.0; run-time malloc/free delays via the `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables, injected as a busy-wait on the device global timer at the top of `DeviceAllocator::malloc` / `DeviceAllocator::free`; the delays are held on the `DeviceAllocator` itself, so every creation policy reacts) | `9de11602` |

## Repository layout

- `config.json` — the single source of truth for what the harness runs and
  builds: the `examples` and `algorithms` lists, the `delays` sweep (the
  (malloc, free) combination values, documented under
  *Changing what is benchmarked*), the pinned
  `dependencies` and the build flags, and the per-machine `machines` table
  (profile to source, output directory for the run logs, hardware name,
  modules to load).
- `config.py` — the python3 bridge the harness uses to read
  `config.json` (`get` / `list` lookups) and to validate it (`check` also
  verifies the referenced flag, parameter, and profile files). It parses the
  file with the stdlib `json` module, so no third-party package (no PyYAML,
  no yq) is needed in the cluster environment.
- `Makefile` — the build harness, the run orchestrator, and the analysis
  driver. The harness clones the pinned PIConGPU and mallocMC, prepares one
  input directory per (example, algorithm) (`pic-create` + parameter
  overlay) and builds it (`pic-build`); it reads its configuration from
  `config.json` (validated up front). The overlay copies
  `param/<Algorithm>/*.param` (the algorithm's `mallocMC.param`, i.e. the
  creation policy), `param/<Example>/*.param` and any per-(example,
  algorithm) overrides from `param/<Example>/<Algorithm>/`. It is invoked as
  `make build PROFILE=<profile> PARAM_DIR=param` (see *Usage* below);
  `make check` prints the resolved build and run configuration and stops.
  The run orchestrator sweeps the full run matrix (examples x algorithms x
  (malloc delay, free delay) combinations) once per *repetition*
  (`REPEATS`, default 1) of `make runs MACHINE=<machine>`; `REP=<i>`
  restricts an invocation to one repetition (the slurm case: one job per
  repetition). Every finished (combination, repetition) writes a stamp
  under `run-stamps/`, so an interrupted series resumes where it stopped.
  The stamps depend only on the example's flags file: rebuilding the
  binaries never invalidates finished runs, and `make clean` / `distclean`
  do not touch `run-stamps/` (`make clean-runs [MACHINE=<m>]` does). Each
  run gets one self-contained log in the machine's output directory (a
  metadata header — machine, commit, the pinned dependency hashes, the
  slurm job when running under slurm, and the sha256 of the binary used —
  plus the run's full output). The analysis driver builds the numbers and
  figures: `make` (the default goal) makes all figures plus the summary
  tables, `make results` only rebuilds `output/results.h5` from the run
  logs, `make summary` only prints the tables, and a single figure is built
  by its file name, e.g. `make figures/foil_lct.pdf`. The numbers are
  rebuilt from the run logs on every invocation; make does not list the log
  files themselves as prerequisites (their names are machine-specific).
- `run_folder.sh` — runs one already-built example folder, once per flag
  line; takes optional fourth (malloc delay, default `0`) and fifth (free
  delay, default `0`) arguments in nanoseconds, passed via the
  `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables.
- `log_{setup,run}_<machine>.sh` — per-machine launchers (hal, rosi,
  rosi-a100): log the environment, load the machine's modules (setup), and
  run `make build` / `make runs MACHINE=<machine>` with the machine's
  profile from the `machines` table in `config.json`, writing the session
  log to the machine's output directory.
- `profiles/` — HPC environment profiles (module/spack setup, `PIC_BACKEND`,
  `PICSRC`). One per machine.
- `flags/` — one `picongpu` command line per benchmark run, per example.
- `param/` — parameter files overlaying the example defaults: one
  `mallocMC.param` per algorithm (`param/<Algorithm>/`, defining the
  creation policy), example-specific files (`param/FoilLCT/`), and optional
  per-(example, algorithm) overrides (`param/<Example>/<Algorithm>/`).
- `legacy/` — the frozen pre-redesign benchmark data: the scripts
  (`move_legacy_logs.sh` relocates the historical output directories,
  `make_legacy_results.py` parses them with a frozen copy of the historical
  parser into `legacy_results.h5`); only the scripts and docs are
  committed, the logs and the frozen table are git-ignored. See
  `legacy/README.md` and `make legacy-results` / `make legacy-verify`.
- `analysis/run_logs.py` — shared parsing of the benchmark run logs: one
  record per `bin/picongpu` run (example, algorithm, grid, imposed delays,
  runtime) from the log's `# metadata:` JSON header and its `set -x`
  trace; the basis of the analysis scripts below.
- `analysis/results_io.py` — shared access to the results HDF5 file: the
  table read/write, the per-fit covariance groups, and the schema helpers
  (grid/scenario labels, the no-delay mask, the particle-memory model).
- `analysis/amdahl.py` — the Amdahl allocation model, its constrained
  1-D/2-operation fits (plus the combined fit that shares W and the
  malloc/free call counts across a scenario's algorithms while each keeps
  its own Amdahl terms), and the bootstrap sleeve (the full model is
  documented in the module docstring).
- `analysis/compute_results.py` — the single "numbers" entry point: parses
  the sweep machines' run logs (the `machines` table of `config.json`) and
  the legacy runs (from the frozen `legacy/legacy_results.h5`, required —
  build it with `make legacy-results`), all grouped by their short
  hardware name, and computes the group runtime statistics, the
  Amdahl fits of every (machine, example, algorithm, grid) sweep (with the
  parameter covariances), the combined (shared-parameter) fit of every
  (machine, example, grid) scenario spanned by at least two algorithms, the
  zero-delay baselines (sweep machines), and the FoilLCT/KelvinHelmholtz
  metadata (all no-delay runs, per short hardware name); writes everything
  to `output/results.h5` and prints nothing.
- `analysis/summarize_results.py` — prints the summary tables from
  `output/results.h5` (group statistics, fits, the combined fit vs the
  individual fits per scenario, Amdahl fractions, no-delay runtimes, figure
  metadata; `--raw` adds the parsed runs).
- `analysis/plot_sweeps.py` — one delay-sweep figure per sweep machine
  (`figures/sweeps-<machine>.pdf`: one row per algorithm, the malloc and
  free delay sweeps side by side, all axes sharing the x- and y-axes,
  fitted curve + bootstrap sleeve, and the scenario's combined fit as a
  heavy line where one exists).
 - `analysis/plot_shared_fits.py` — one forest figure per sweep machine
   (`figures/sweeps-shared-<machine>.pdf`: one row per (scenario, algorithm),
   the shared W / N_malloc / N_free values against each algorithm's
   individual fit, and the individual vs combined A_malloc / A_free, each
   A_* value annotated with its Amdahl fraction f, the share of the
   zero-delay runtime W + A_malloc + A_free the native cost occupies).
- `analysis/plot_foil_lct.py` — the FoilLCT bar chart of the no-delay runs
  (`figures/foil_lct.pdf`).
- `analysis/plot_kelvin_helmholtz.py` — the KelvinHelmholtz violin chart of
  the no-delay runs relative to the ScatterAlloc reference
  (`figures/kelvin_helmholtz.pdf`).
- `build/` — created by the Makefile; one CMake project per (example,
  algorithm).

## Usage

Everything runs on the HPC machine with the matching profile; the entry
points must be invoked from the repository root. The benchmark
configuration (examples, algorithms, delay sweep, dependency pins, build
flags, and the per-machine profiles and output directories) lives in
`config.json`. Both entry points validate it first and can stop right after
resolving it, which is the way to sanity-check a change before a long build
or benchmark:

```
make check             # includes the derived run matrix
python3 config.py list run-matrix
```

1. Build all examples and algorithms (slow the first time: one full
   PIConGPU build per example and algorithm):

   ```
   make build PROFILE=profiles/hal.sh PARAM_DIR=param
   ```

   That also patches `PICSRC=` in the given profile in place, pointing it
   at the local `src/` checkout.

   The build is incremental and can be re-run at any time; it reuses
   `src/` and `build/` and only repeats the parts that are out of date:

   - `src/` is re-cloned only if the directory is not a git checkout at the
     pinned commit; a changed pin triggers `git fetch && git checkout`.
   - An input directory (`build/<Ex>/<Algo>/`) is regenerated when the
     PIConGPU pin, the profile, or the `param/<Algo>*/`, `param/<Ex>*`
     files change (including a newly added parameter file; the
     `build/<Ex>/<Algo>/.input-stamp` records the prepared state).
   - A build (`pic-build`) is skipped when the input, the build flags, the
     mallocMC pin, the profile content and the toolchain versions
     (gcc/cmake/nvcc) are unchanged (`build/.profile-env`,
     `build/.toolchain` and `build/.build-flags` record these); otherwise
     `pic-build` runs incrementally. `make -j` builds the independent
     (example, algorithm) pairs in parallel; the default is serial.

    `make clean` removes `build/`, `figures/` and `output/results.h5`,
    `make distclean` removes `src/` as well; neither touches the run stamps
    (finished runs stay finished after a rebuild). Alternatively delete just
    a single `build/<Ex>/<Algo>/` folder, or only its `.input-stamp`, to
    rebuild one example and algorithm.

2. Run the benchmarks (the sweep, once per repetition):

   ```
   make runs MACHINE=hal                    # REPEATS=1 (the default)
   make runs MACHINE=hal REPEATS=3          # three full-sweep repetitions
   make runs MACHINE=rosi REPEATS=3 REP=2   # only repetition 2 (one slurm job)
   make full MACHINE=hal                    # build, then run
   make clean-runs [MACHINE=hal]            # forget finished runs (logs kept)
   ```

   One run is one (example, algorithm) build through one `(malloc delay,
   free delay)` combination derived from the `delays` section of
   `config.json`: every flag line of the example is run as
   `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...` from
   `build/<Ex>/<Algo>/` (via `run_folder.sh`), in sweep order (example,
   algorithm, repetition, then the delay combination). Each run writes one
   self-contained
   log `run_<machine>_<Ex>_<Algo>_m<M>_f<F>_r<I>_<time>.txt` to the machine's
   output directory (metadata header: machine, commit, the pinned
   dependencies, the slurm job id when running under slurm, and the sha256
   of the binary used; then the run's full output, including the
   `calculation ... simulation time:` line). Each finished run stamps
   `run-stamps/<machine>/<Ex>/<Algo>/<M>_<F>/rep-<I>.stamp` (its content:
   the log path), so an interrupted series resumes where it stopped, and a
   rebuild never invalidates finished runs. The launchers
   `log_run_<machine>.sh` do this with a session log (environment block +
   progress) on top; the rosi ones take the repetition number as their
   first argument (`sbatch log_run_rosi.sh 1`, with `REPEATS` in the
   environment) so each slurm job runs exactly one full-sweep repetition —
   all slurm allocation options come from the `sbatch` line, the scripts
   carry no `#SBATCH` directives.

Single run (one built folder, one combination, one flags file):

```
bash run_folder.sh build/FoilLCT/FlatterScatter flags/FoilLCT.flags profiles/hal.sh 10000 0
```

## Changing what is benchmarked

Most of it is now in `config.json` (run `python3 config.py check` after
editing; `make check` / `python3 config.py list run-matrix` show what
would be done).

- **Delay combinations**: edit the `delays` section of `config.json`
  (`baseline`, `arms.values`, `joint.values`); the Makefile run targets
  derive the full combination set from it: `baseline` is the `(0, 0)`
  reference run, `arms` are the single-delay sweeps (one value at a time with
  the other delay held at 0, on a log grid from `100` to `1e8` ns in
  1/4-decade steps, skipping the intermediate steps between `1e5` and `1e6`),
  and `joint` is a 3x3 grid coupling both delays so the two-operation fit is
  constrained off the arms.
  The delays are applied at run time via `MALLOCMC_MALLOC_DELAY` /
  `MALLOCMC_FREE_DELAY`, so changing the sweep requires no rebuild. The large
  delays let each Amdahl term `A*s0/(d+s0)` decay into its `1/d` tail so the
  asymptote `W + A + N*d` gets anchored; this matters most for free, whose
  native cost fades slowly.
- **Grids / steps / other picongpu flags**: edit `flags/<Example>.flags`
  (one command line per run).
- **Allocator configuration**: `param/<Algorithm>/mallocMC.param` defines
  the `DeviceHeap`; the creation policy and, for FlatterScatter, the heap
  config (`DefaultHeapConfig<block, page, waste>`) live there. Editing a
  file invalidates exactly the affected (example, algorithm) inputs.
- **PIConGPU / mallocMC version**: the `dependencies` section of
  `config.json` (URL, checkout path, pinned hash).
- **Build flags**: the `build` section of `config.json`: `cxx_flags` is
  passed to both `CMAKE_CXX_FLAGS` and `CMAKE_CUDA_FLAGS` (the Boost
  `std::source_location` constexpr bug is in the C++ compiled by nvcc, not a
  CUDA-specific flag) and `extra_cmake_flags` are appended verbatim.
- **New example**: add it to `examples` in `config.json`, provide
  `flags/<Example>.flags`, and optionally `param/<Example>/*.param`.
- **New algorithm**: add it to `algorithms` in `config.json` and provide
  `param/<Algorithm>/mallocMC.param`.
- **New machine**: add an entry to the `machines` table in `config.json`
  (profile to source, output directory for the run logs, hardware name for
  the figure titles, modules to load before the profile) and a pair of
  `log_{setup,run}_<machine>.sh` launchers based on the existing ones.

## Analysis

The analysis is split into *computing the numbers* and *drawing the
figures*, joined by the single HDF5 file `output/results.h5`:

- `compute_results.py` parses the run logs of the two worlds they live in:
  the sweep machines of the `machines` table of `config.json` (the new
  format, one `# metadata:` JSON line per log; the group statistics, the
  Amdahl fits and the zero-delay baselines are computed for these) and the
  frozen legacy table `legacy/legacy_results.h5` (the pre-redesign logs,
  read grouped by their short hardware name). All runs are grouped by that
  short hardware name (`A30`, `V100`, ...; the FoilLCT / KelvinHelmholtz
  metadata covers the no-delay runs of both worlds). It prints nothing.
- `summarize_results.py` prints the summary tables from
  `output/results.h5` (per machine: group statistics; the fits and the
  Amdahl fractions of runtime spent in allocations / frees; the combined
  (shared-parameter) fit of each multi-algorithm scenario compared
  algorithm by algorithm against the individual fits; the no-delay
  runtimes; the figure metadata). `--raw` also prints the parsed runs.
- one script per figure, each reading `output/results.h5` (all take
  `--show` to display the figure in a window): `plot_sweeps.py`
  (`figures/sweeps-<machine>.pdf`, `--machine` for one machine),
   `plot_shared_fits.py` (`figures/sweeps-shared-<machine>.pdf`, `--machine`
   for one machine), `plot_foil_lct.py` (`figures/foil_lct.pdf`),
  `plot_kelvin_helmholtz.py` (`figures/kelvin_helmholtz.pdf`).

The simplest way to run the whole thing (or any single figure) is the
`Makefile`:

```
make                      # all figures + the summary tables
make results              # only output/results.h5, from the run logs
make summary              # only the summary tables
make figures/foil_lct.pdf # one figure by file name (also:
make figures/sweeps-hal.pdf # figures/sweeps-<machine>.pdf,
make figures/sweeps-shared-hal.pdf # figures/sweeps-shared-<machine>.pdf
```

or the scripts directly (run from the repository root):

```
python3 analysis/compute_results.py
python3 analysis/summarize_results.py
python3 analysis/plot_foil_lct.py
```

### Reproducing the analysis

The Python dependencies of the analysis are declared in
`requirements.txt` (permissive manifest, lowest verified versions) and
rigorously pinned, including the Python version, by the content-hashed
`conda-lock.yml` lock of the `environment.yml` recipe. From it, the
committed `conda-linux-64.lock` is rendered: a plain list of the exact
package files, installable by any version of `micromamba`, `mamba`, or
`conda` without invoking a solver (`make env` installs from it).

- Primary path: `make env` creates the locked environment with the first
  of `mamba`, `micromamba`, `conda` found on PATH (mamba is the default;
  the install itself is solver-free for all of them). Force a specific
  tool with `make env ENV_TOOL=<tool>` (`micromamba` is a single static
  binary, no root needed,
  <https://micro.mamba.pm/api/micromamba/linux-64/latest>; on mamba 1.x,
  which cannot read the explicit lock, use `ENV_TOOL=conda`):

  ```
  make env                                  # create the locked environment
  micromamba activate mallocmc-bench        # or: conda / mamba activate
  make                                      # ...then run the analysis
  ```

  (or `make PY=<prefix>/bin/python3`; e.g.
  `micromamba prefix -n mallocmc-bench`). If you have packages in your
  pip *user* site (`pip install --user`), export `PYTHONNOUSERSITE=1` so
  the locked environment's versions take precedence.
- Refreshing the lock (after editing `environment.yml`):
  `pip install conda-lock`, then
  `conda-lock lock -f environment.yml -p linux-64 --micromamba` (updates
  `conda-lock.yml`) and `conda-lock render -p linux-64` (updates
  `conda-linux-64.lock`); review and commit `environment.yml` and both
  lock files together.
- pip path (no lock): `pip install -r requirements.txt`.
- This covers the Python analysis only. The C++ build harness has its own
  dependencies (CMake, compilers; the PIConGPU / mallocMC pins in
  `config.json`), and the pre-commit hooks are independently pinned in
  `.pre-commit-config.yaml`.

Notes:

- The fits are bounded `scipy.optimize.curve_fit` of the 1-D model
  `T(s) = W + N*s + A*s0/(s+s0)` (sweep on one delay) or the two-operation
  model `T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0)`
  (combination sweep); the full model, the bounds, and the diagonalization
  caveats of the bootstrap are documented in `analysis/amdahl.py`.
- The combined fit reuses the two-operation model but fits it once over all
  algorithms of a (machine, example, grid) scenario together: W, N_malloc
  and N_free are shared across the algorithms while each keeps its own
  `A_malloc/A_free` and `m0/f0`, so it tests whether one set of baseline and
  call-count parameters explains every algorithm's sweep. It is fit for
  every scenario with at least two algorithms and at least one varying
  delay, and is written to the `shared_fits` table (one row per
  scenario-algorithm) and the `shared_fit_cov/` groups.
- Each run is labeled with a `configuration` column (`compile-time`
  per-variant sweep vs `run-time` env-var sweep), so both log layouts can
  be analyzed side by side (pre-rename logs that used `MALLOCMC_SLEEP_TIME`
  parse into the malloc delay). Pass
  `compute_results.py --configuration run-time` (or `compile-time`) to
  compute the statistics and the fits only from runs of that configuration;
  the stored runs always stay complete.
- The Python analysis needs `numpy`, `pandas`, `scipy`, `matplotlib`,
  `seaborn`, and `h5py`; see "Reproducing the analysis" for the declared
  and locked versions.
- The comparison figures (`foil_lct.pdf`, `kelvin_helmholtz.pdf`) use the
  no-delay runs of *every* hardware: the frozen legacy runs
  (`legacy/legacy_results.h5`) plus the (0, 0) baseline runs of each sweep
  machine's delay combination sweeps. Both are grouped by the short
  hardware name (e.g. the legacy runs of the old `hal` cluster and the
  `hal` sweep machine's baselines are both plotted as `A30`).
- The figure row order is the `algorithms` list of `config.json` (recorded
  in the results file); the hardware display order of the comparison figures
  is fixed in `analysis/results_io.py`.

## Code style

The Python analysis code is formatted and linted with ruff (line length
120, see `pyproject.toml`); the shell scripts are formatted with shfmt
(2-space indent, indented case bodies) and checked with shellcheck. All of
this runs as pre-commit hooks on every commit:

```
pip install pre-commit
pre-commit install          # once per clone
pre-commit run --all-files  # or just commit; the hooks check the staged files
```
