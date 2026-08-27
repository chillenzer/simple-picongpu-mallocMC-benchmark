<!--
SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT
-->

# PIConGPU Allocations using mallocMC

This repository benchmarks the runtime of PIConGPU simulations compiled with
different configurations of the [mallocMC](https://github.com/chillenzer/mallocMC)
device allocator: the **FlatterScatter**, **ScatterAlloc** and
**Gallatin** creation policies, each while sweeping (malloc delay,
free delay) combinations — nanosecond delays imposed at run time on every
allocation and every free request — to study the impact of allocation and
free latency on simulation runtime (the injection mechanism and the
allocation model are documented in *The method*).

The sections are ordered for the reader: *The method* (what is measured
and what the numbers mean), *What is benchmarked* (what was run), *Where
the data lives* (what is and is not in the repository), and then how to
run, change, and analyze the benchmark. *Where to look* indexes the goals.

## Where to look

| You want to ...                         | Read                                              | Open first                          |
|-----------------------------------------|---------------------------------------------------|-------------------------------------|
| verify the measurement and the model    | *The method*                                      | `analysis/allocation_model.py`      |
| reproduce the numbers and figures       | *Where the data lives*, *Reproducing the analysis* | `make env`, then `make`            |
| run the benchmark, or add a machine     | *Running the benchmark*                           | the `machines` table of `config.json` |
| extend the benchmark                    | *Changing what is benchmarked*                    | `config.json`                       |
| use the data or the figures             | *Analysis*                                        | `output/results.h5`                 |
| contribute (code, logs, metadata)       | *Repository layout*, *Run logs*, *Code style*     | the file at hand                    |

## The method

The measured quantity is the runtime of the full PIConGPU calculation
(the `calculation ... simulation time:` line of the run's output) with
nanosecond delays imposed on every allocation and free request of the
run. The creation policy is compiled in, the delays are not: mallocMC
reads them at run time from the `MALLOCMC_MALLOC_DELAY` and
`MALLOCMC_FREE_DELAY` environment variables, so one build per (example,
algorithm) serves the whole sweep.

**The delay injection.** The delays live in the `add-delay` branch of
mallocMC (the pinned dependency of *What is benchmarked*): a busy-wait
on the 64-bit device global timer injected at the top of
`DeviceAllocator::malloc` / `DeviceAllocator::free` and held on the
`DeviceAllocator` itself, so every creation policy reacts. The
`__nanosleep` intrinsic was considered first; its wake-up guarantee
turned out too weak and produced step-like sweeps, so the busy-wait
replaced it.

**The allocation model.** Each of the N allocation calls on the
(serial, host-side) critical path takes its native cost c_a plus the
imposed delay s, so the runtime follows the straight line

```
T(s) = W + A + N*s,   A = N*c_a (native allocation time)
```

with W the runtime without any allocation cost. The native part is
negligible against large imposed delays but sets the small-delay
behaviour, so a sweep is fitted with the allocation model

```
T(s)  = W + N*s + A*s0/(s+s0)                 (one delay)
T(m,f) = W + N_m*m + N_f*f + A_m*m0/(m+m0)
         + A_f*f0/(f+f0)                      (both delays)
```

which has the straight line's zero-delay value T0 = W + A (one delay;
T0 = W + A_malloc + A_free with the two operations) and approaches its
large-delay asymptote W + N*s, the native cost fading over the sleeptime
scale s0 (m0, f0 per operation). The fitted parameters are therefore W
(baseline runtime), N (allocation calls per run; N_malloc / N_free per
operation), A (native allocation time; A_malloc / A_free per operation),
and the fade scale s0 (m0, f0). The fraction of the zero-delay runtime
T0 spent in the native cost is f = A/T0 (f_malloc = A_malloc/T0,
f_free = A_free/T0), always in [0, 1) by the fit's bounds.

The fit is a bounded `scipy.optimize.curve_fit` (W, N, A >= 0, s0 in
[0.05*s_min, 0.5*s_range]) with a robust linear solution over a log-s0
grid as the initial guess — reported as the fit when `curve_fit` does
not converge; the parameter errors are propagated from the fit
covariance, and the figures carry a bootstrap sleeve. When the native
per-allocation cost c_a (ns) is known, the fit can be constrained to
A = N*c_a and fit (W, N, s0) only. The model, the bounds, and the
bootstrap's caveats are documented in the `analysis/allocation_model.py`
module docstring, which is the single source of truth for the model.

**Why the sweep is shaped like this.** The delay arms run a log grid from
100 ns to 1e8 ns so the large delays anchor the asymptote W + N*s —
which matters most for free, whose native cost fades slowly — and they
omit the zero-delay point, so the (0, 0) baseline runs are outside every
fit and provide the T0 baselines and the paper figures instead. The 3x3
joint grid couples both delays so the two-operation fit is constrained
off the arms.

## What is benchmarked

- **Machines**: `hal` (NVIDIA A30), `rosi` (NVIDIA V100) and
  `rosi-a100` (NVIDIA A100); the rosi machines load the `hopper`,
  `GCCcore/14.3.0` and `git/2.50.1` modules before their profile. The
  `machines` table of `config.json` maps each machine label to its
  profile to source, output directory for the run logs, hardware name
  (the figure titles), and modules to load.
- **Allocation policies**: `FlatterScatter`, `ScatterAlloc`, `Gallatin`
  (the `algorithms` list in `config.json` and
  `param/<Algorithm>/mallocMC.param`)
- **Delay combinations** (nanoseconds, the `delays` section of
  `config.json`; the Makefile run targets derive the full combination set
  from it, `python3 config.py list run-matrix`): a `(0, 0)`
  baseline, the malloc-delay sweep with free delay 0 and the free-delay
  sweep with malloc delay 0 over the 12 log-spaced values
  `100`…`100000000` (1/4-decade steps, skipping the intermediate steps
  between `1e5` and `1e6`), plus a 3x3 joint grid over `10000`,
  `1000000`, `10000000` (34 combinations per example and algorithm)
- **Cost**: 2 examples x 3 algorithms x 34 combinations = 204 runs per
  machine per repetition, plus one full PIConGPU build per (example,
  algorithm) pair (slow the first time)
- **Examples**: `KelvinHelmholtz` (3D, grid sizes 128^3, 256x128x128,
  256x128x256, 1500 steps) and `FoilLCT` (2D, 256x1280 cells, 2000 steps)
  (the `examples` list in `config.json` and `flags/*.flags`)
- **One build per (example, algorithm)**: the creation policy is compiled
  in via `param/<Algorithm>/mallocMC.param`; the delays are not a
  compile-time option, mallocMC reads them at run time from the
  `MALLOCMC_MALLOC_DELAY` and `MALLOCMC_FREE_DELAY` environment variables
  (set by `run_folder.sh` for every run). Changing the sweep therefore
  never requires recompiling.
- **Layout**: `build/<Example>/<Algorithm>/` holds one build; its flag
  lines are run once per combination:
  `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`

The Makefile pins the dependency versions (the `dependencies` section of
`config.json`):

| dependency | source | pinned to |
|------------|--------|-----------|
| PIConGPU | ComputationalRadiationPhysics/picongpu | `6e7d58bb` |
| mallocMC | chillenzer/mallocMC, `add-delay` branch (`BOOST_LANG_*` guards compatible with PIConGPU's vendored alpaka 2.0; run-time malloc/free delays via the `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables, injected as a busy-wait on the device global timer at the top of `DeviceAllocator::malloc` / `DeviceAllocator::free`; the delays are held on the `DeviceAllocator` itself, so every creation policy reacts) | `9de11602` |

## Where the data lives

The repository holds the harness, the analysis, and the pins; the
benchmark data does not. Data lives on the benchmark machines and is
git-ignored; the derived artifacts are rebuilt from it by `make`.

- **In the repository**: `config.json` and the harness/analysis scripts,
  the pinned dependency hashes, the locked analysis environment, and the
  `legacy/` scripts and docs.
- **On the machines (never committed)**: the run logs in each machine's
  output directory (`output/<machine>-sleeptimes/`, incl. its `sessions/`
  folder of session logs), the run stamps (`run-stamps/`), and the frozen
  legacy data (`legacy/logs/`, `legacy/legacy_results.h5`). The legacy
  data is the piece a fresh checkout must obtain for the comparison
  figures: copy `legacy/logs/` from a machine that still holds the
  historical `output/` tree (or run `move_legacy_logs.sh` there), then
  `make legacy-results` builds the frozen table and `make legacy-verify`
  checks it (the full procedure is in `legacy/README.md`).
- **Derived (never committed, rebuilt by `make`)**: `output/results.h5`,
  the `figures/` PDFs, and `ro-crate-metadata.json`; `make clean` removes
  them.

On a fresh checkout without the machine data, `make` still runs end to
end: the sweep tables (`group_stats`, `fits`, `baselines`) come out empty
for the sweep machines, and the FoilLCT / KelvinHelmholtz figures and
their statistics are computed from the frozen legacy table alone. The
archived-but-excluded legacy runs (the `hal-sleeptimes-nanosleep`
experiment, outside the benchmark matrix) are parsed but attributed to
the empty hardware name and dropped by the analysis, which records the
exclusion in the results file's attributes.

## Running the benchmark

Everything runs on the HPC machine with the matching profile; the entry
points must be invoked from the repository root. The benchmark
configuration (examples, algorithms, delay sweep, dependency pins, build
flags, and the per-machine profiles and output directories) lives in
`config.json`. Both entry points validate it first and can stop right
after resolving it, which is the way to sanity-check a change before a
long build or benchmark:

```
make check             # includes the derived run matrix
python3 config.py list run-matrix
```

The Makefile variables:

| variable        | meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `MACHINE`       | a row of the `machines` table of `config.json` (hal, rosi, rosi-a100) |
| `PROFILE`       | the profile to source for the build (`profiles/<machine>.sh`)      |
| `PARAM_DIR`     | the parameter overlay directory (default `param`)                  |
| `REPEATS`       | full-sweep repetitions of `make runs` (default 1)                  |
| `REP`           | restrict the invocation to one repetition (the slurm case)         |

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

   `make clean` removes `build/`, `figures/`, `output/results.h5` and
   `ro-crate-metadata.json`, `make distclean` removes `src/` as well;
   neither touches the run stamps (finished runs stay finished after a
   rebuild). Alternatively delete just
   a single `build/<Ex>/<Algo>/` folder, or only its `.input-stamp`, to
   rebuild one example and algorithm.

2. Run the benchmarks (the sweep, once per repetition):

   ```
   make runs MACHINE=hal                    # REPEATS=1 (the default)
   make runs MACHINE=hal REPEATS=3          # three full-sweep repetitions
   make runs MACHINE=rosi REPEATS=3 REP=2   # only repetition 2 (one slurm job)
   make full MACHINE=hal                    # build, then run
   make clean-runs [MACHINE=hal]            # forget finished runs (re-runs add a vintage)
   ```

   On a slurm machine, the `log_run_<machine>.sh` launcher drives one slurm
   job per repetition: `sbatch log_run_rosi.sh 1` (with `REPEATS` in the
   environment) runs exactly one full-sweep repetition — all slurm
   allocation options come from the `sbatch` line, the scripts carry no
   `#SBATCH` directives.

   **How a run works.** One run is one (example, algorithm, combination,
   repetition) identity: one (example, algorithm) build through one
   `(malloc delay, free delay)` combination derived from the `delays`
   section of `config.json`, once per repetition, in sweep order (example,
   algorithm, repetition, then the delay combination). Every flag line of
   the example is run as
   `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`
   from `build/<Ex>/<Algo>/` (via `run_folder.sh`), and each flag line
   writes one self-contained log — its anatomy is in *Run logs*. Each
   finished run writes a stamp (in *Run logs* too): an existing stamp is
   what makes `make runs` skip a run, so an interrupted series resumes
   where it stopped, and the stamps depend only on the example's flags
   file, so a rebuild never invalidates finished runs
   (`make clean-runs [MACHINE=<m>]` forgets them). Runs are append-only:
   a re-run of a (combination, repetition) writes a new vintage of the
   logs next to the older ones — nothing is ever removed (a re-run in the
   very same second merely appends a counter to the file name) — and the
   analysis flags the older vintages `superseded = 1` (*Analysis*), so a
   re-run adds new rows instead of replacing anything.

Single run (one built folder, one combination, one flags file):

```
bash run_folder.sh build/FoilLCT/FlatterScatter flags/FoilLCT.flags profiles/hal.sh 10000 0
```

**Adding a machine.** Add an entry to the `machines` table in
`config.json` (profile to source, output directory for the run logs,
hardware name for the figure titles, modules to load before the profile)
and a pair of `log_{setup,run}_<machine>.sh` launchers based on the
existing ones; the machine's hardware name is what appears in its figure
titles.

## Run logs

`run_stamp.sh` writes the run logs (one invocation per run, through the
`make runs` targets), `logmeta.py` emits their self-describing metadata
line, and `analysis/run_logs.py` reads them back.

- **Name and location.** One log per grid run (one line of the example's
  flags file), in the machine's output directory:
  `run_<machine>_<Ex>_<Algo>_m<M>_f<F>_r<rep>_<line-sha8>_<time>.txt`; a
  same-second re-run merely appends a counter to the file name.
- **Header.** A two-line header: a human one-liner `# run:` line and the
  self-describing `# metadata:` JSON line (schema 1). The run metadata
  carries, all best effort ("unavailable" placeholders, never an error):
  `schema`, `kind` (`run`), `ts`, `machine`, `hostname`, `user`, the git
  state, `slurm_job` when running under slurm, `pins` (the `config.json`
  dependency pins), `hw` (the GPU names and their driver, the CPU model,
  the operating system), the build facts read from the binary's own
  `CMakeCache.txt` (compiler, CUDA, the build flags, the build type), the
  sha256 of the binary, and the run block (example, algorithm, the
  imposed `[malloc, free]` delays, the repetition, the flags-file line
  number and the line count, the 8-char sha of the line, and the line
  itself).
- **Body.** That grid run's `set -x` trace and full output, including the
  `calculation ... simulation time:` line the runtime is parsed from.
- **Session logs.** The `log_{setup,run}_<machine>.sh` launchers write a
  session log per launch to `<output dir>/sessions/` (a `# metadata:` line
  with `kind: setup`, plus the environment block and progress); the folder
  is a free-text backup that the analysis never reads and that runs are
  never superseded.
- **Stamps.** `run-stamps/<machine>/<Ex>/<Algo>/<M>_<F>/rep-<R>.stamp`,
  the run's log paths one per line — i.e. its current vintage.
- **Format rule.** A log is new format if and only if its `# metadata:`
  line's JSON has `"schema": 1`; the pre-redesign logs do not carry it and
  must be frozen under `legacy/` (`make legacy-results`) instead of being
  parsed.

## Changing what is benchmarked

Most of it is now in `config.json` (run `python3 config.py check` after
editing; `make check` / `python3 config.py list run-matrix` show what
would be done).

- **Delay combinations**: edit the `delays` section of `config.json`
  (`baseline`, `arms.values`, `joint.values`; the grid values are listed
  in *What is benchmarked*); the Makefile run targets derive the full
  combination set from it: `baseline` is the `(0, 0)` reference run,
  `arms` are the single-delay sweeps (one value at a time with the other
  delay held at 0), and `joint` is a 3x3 grid coupling both delays (why
  the grid is shaped like it is: *The method*). The delays are applied at
  run time via `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY`, so
  changing the sweep requires no rebuild.
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

**What the analysis needs of a setup** (otherwise the table comes out
empty):

| table           | requirement                                                              |
|-----------------|---------------------------------------------------------------------------|
| `fits`          | the group's runs span at least two distinct values of a delay (the 1-D model falls back to that one delay) |
| `shared_fits`   | the scenario (machine, example, grid) has at least two algorithms and at least one varying delay |
| `baselines`     | the sweep machine runs a (0, 0) baseline                                  |
| `foil` / `khi`  | zero-delay runs per hardware: the frozen legacy runs plus every sweep machine's (0, 0) baselines |

## Repository layout

- `config.json` — the single source of truth for what the harness runs and
  builds: the `examples` and `algorithms` lists, the `delays` sweep (the
  (malloc, free) combination values, *What is benchmarked*; how to edit
  them, *Changing what is benchmarked*), the pinned `dependencies` and the
  build flags, and the per-machine `machines` table (*What is
  benchmarked*).
- `config.py` — the python3 bridge the harness uses to read `config.json`
  (`get` / `list` lookups) and to validate it (`check` also verifies the
  referenced flag, parameter, and profile files). It parses the file with
  the stdlib `json` module, so no third-party package (no PyYAML, no yq)
  is needed in the cluster environment.
- `Makefile` — the build harness, the run orchestrator, and the analysis
  driver: it clones the pinned PIConGPU and mallocMC and, from
  `config.json` (validated up front), prepares one input directory per
  (example, algorithm) (the `param/` overlay copies `param/<Algorithm>/`,
  `param/<Example>/`, and any `param/<Example>/<Algorithm>/` files) and
  builds it (`pic-build`). The run orchestrator and the analysis driver
  are documented in *Running the benchmark* and *Analysis*; `make check`
  prints the resolved configuration and stops. The target comments in the
  header mirror this README.
- `run_folder.sh` — runs one already-built example folder, once per flag
  line; takes optional fourth (malloc delay, default `0`) and fifth (free
  delay, default `0`) arguments in nanoseconds, passed via the
  `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables,
  and an optional sixth argument that selects one flags-file line
  (1-based), so a per-grid log records one grid run.
- `logmeta.py` — emits the self-describing `# metadata:` JSON line
  (schema 1) of the run and session logs; every fact is best effort
  ("unavailable" placeholders, never an error). The key list is in
  *Run logs*.
- `make_rocrate.py` — generates and checks the RO-Crate metadata of the
  repository (`make rocrate`): the harness as a workflow, the benchmark
  runs as provenance, the analysis as actions (what exactly is described,
  see the *RO-Crate* section below).
- `log_{setup,run}_<machine>.sh` — per-machine launchers (hal, rosi,
  rosi-a100): log the environment, load the machine's modules (setup), and
  run `make build` / `make runs MACHINE=<machine>` with the machine's
  profile from the `machines` table in `config.json` (see *Running the
  benchmark*).
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
  committed, the logs and the frozen table are git-ignored. Where the data
  lives and how to obtain it: *Where the data lives*; the directory doc:
  `legacy/README.md`.
- `analysis/run_logs.py` — shared parsing of the benchmark run logs: one
  record per `bin/picongpu` run (example, algorithm, grid, imposed
  delays, runtime, plus the run's full and initialisation runtimes and
  its number of simulation steps) from the log's `# metadata:` JSON header
  and its `set -x` trace (the keys, *Run logs*), and the provenance of the
  log copied from the header onto every one of the log's runs; the basis
  of the analysis scripts below.
- `analysis/results_io.py` — shared access to the results HDF5 file: the
  table read/write, the per-fit covariance groups, and the schema helpers
  (grid/scenario labels, the no-delay mask, the particle-memory model);
  the single source of truth for the runs' column names.
- `analysis/allocation_model.py` — the allocation model, its constrained
  1-D/2-operation fits (plus the combined fit that shares W and the
  malloc/free call counts across a scenario's algorithms while each keeps
  its own saturation terms), and the bootstrap sleeve. The model is
  summarised in *The method*; the module docstring is the reference.
- `analysis/compute_results.py` — the single "numbers" entry point: parses
  the sweep machines' run logs (the `machines` table of `config.json`) and
  the legacy runs (from the frozen `legacy/legacy_results.h5`, required —
  build it with `make legacy-results`), all grouped by their short
  hardware name, and computes the group runtime statistics, the
  allocation-model fits of every (machine, example, algorithm, grid) sweep
  (with the parameter covariances), the combined (shared-parameter) fit of
  every (machine, example, grid) scenario spanned by at least two
  algorithms, the zero-delay baselines (sweep machines), and the
  FoilLCT/KelvinHelmholtz figure statistics (all zero-delay runs, per short
  hardware name); writes everything to `output/results.h5` (*Analysis*)
  and prints nothing.
- `analysis/summarize_results.py` — prints the summary tables from
  `output/results.h5` (group statistics, fits, the combined fit vs the
  individual fits per scenario, the fractions of runtime spent in
  allocations / frees, the No-delay runtimes table, the figure
  statistics; `--raw` adds the parsed runs).
- `analysis/plot_sweeps.py` — one delay-sweep figure per sweep machine
  (`figures/sweeps-<machine>.pdf`: one row per algorithm, the malloc and
  free delay sweeps side by side, all axes sharing the x- and y-axes,
  fitted curve + bootstrap sleeve, and the scenario's combined fit as a
  heavy line where one exists).
- `analysis/plot_shared_fits.py` — one forest figure per sweep machine
  (`figures/sweeps-shared-<machine>.pdf`: one row per (scenario, algorithm),
  the shared W / N_malloc / N_free values against each algorithm's
  individual fit, and the individual vs combined A_malloc / A_free,
  each A_* value annotated with its fraction f = A/T0, the share of the
  zero-delay runtime T0 = W + A_malloc + A_free the operation's native
  cost occupies).
- `analysis/plot_foil_lct.py` — the FoilLCT bar chart of the zero-delay
  runs (`figures/foil_lct.pdf`).
- `analysis/plot_kelvin_helmholtz.py` — the KelvinHelmholtz violin chart of
  the zero-delay runs relative to the ScatterAlloc reference
  (`figures/kelvin_helmholtz.pdf`).
- `build/` — created by the Makefile; one CMake project per (example,
  algorithm).

## Analysis

The analysis is split into *computing the numbers* and *drawing the
figures*, joined by the single HDF5 file `output/results.h5`:

- `compute_results.py` parses the run logs of the two sources they live
  in: the sweep machines of the `machines` table of `config.json` (the
  new format, one `# metadata:` JSON line per log; the group statistics,
  the allocation-model fits and the zero-delay baselines are computed for
  these) and the frozen legacy table `legacy/legacy_results.h5` (the
  pre-redesign logs, read grouped by their short hardware name). All runs
  are grouped by that short hardware name (`A30`, `V100`, ...; the
  FoilLCT / KelvinHelmholtz figure statistics cover the zero-delay runs of
  both sources). It prints nothing. Every row of the `runs` table carries
  the timing metrics and the provenance of the log the run came from
  (tabulated below); a row whose log comes from an older vintage of the
  same identity carries `superseded = 1`, derived from the run stamps
  (*Run logs*), 0 otherwise (all frozen legacy runs are 0). Nothing
  downstream filters those rows out: the superseded vintages stay in every
  table and every number, and the group statistics and the fits are
  computed over all rows; a consumer that wants the current state of the
  world selects `runs[runs["superseded"] == 0]`. The single source of
  truth for the column names is `analysis/results_io.py`.
- `summarize_results.py` prints the summary tables from
  `output/results.h5` (first the total run count with its superseded
  share, `Runs: N (M superseded)`, and the excluded archived runs, when
  any; then per machine: group statistics; the fits and the
  fractions of runtime spent in allocations / frees; the combined
  (shared-parameter) fit of each multi-algorithm scenario compared
  algorithm by algorithm against the individual fits; the No-delay
  runtimes; the figure statistics). `--raw` also prints the parsed
  runs.
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

**The results file.** `output/results.h5` holds one group per table, one
dataset per column (numeric columns float64, int64 where the column
carries no NaN, text columns variable-length strings, a missing text
value `""`); the fits' parameter vectors and covariances are stored next
to their row:

| name          | content                                                                                          |
|---------------|---------------------------------------------------------------------------------------------------|
| `runs`        | every parsed run, the columns below                                                              |
| `group_stats` | per (machine, setup, algorithm, grid, delays): the runtime's count, mean, std, min, p25, p50, p75, max |
| `baselines`   | the zero-delay (0, 0) runtime IQR per group (sweep machines)                                      |
| `fits`        | the allocation-model fits, one row per group (+ the `fits/cov/<machine>/<setup>/<algorithm>/<grid>/` subgroups) |
| `shared_fits` | the combined (shared-parameter) fits, one row per (scenario, algorithm) (+ the `shared_fit_cov/<machine>/<setup>/<grid>/` subgroups) |
| `foil`        | the FoilLCT bar chart's distributions over the zero-delay runs, per hardware                      |
| `foil_pvalue` | the Kruskal p-values behind the FoilLCT chart, per hardware                                        |
| `khi`         | the KelvinHelmholtz violin chart's statistics, per (hardware, estimated memory)                   |

The file's top-level attributes record the provenance: `created_utc`,
`git_commit`, `sources` (the source log directories), `sweep_machines`,
`machine_titles` (machine label -> hardware title), `algorithm_order`
(the `algorithms` list of `config.json`, the row order of all figures),
and the exclusion of the archived-but-excluded legacy runs
(*Where the data lives*), `excluded_sources` and `excluded_runs`.

**The `runs` table** (one row per parsed run, sweep machines' logs plus
the frozen legacy table):

| columns                                                                                     | meaning                                                                                  |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `machine`, `hardware`                                                                        | the sweep machine's label (empty for the legacy runs stored without one) and the short paper-figure hardware name |
| `setup`, `algorithm`, `x`, `y`, `z`                                                         | the example, the algorithm, and the grid (2-D runs carry an empty `z`)                    |
| `malloc_sleeptime`, `free_sleeptime`                                                        | the imposed (malloc, free) delays in nanoseconds (empty for the pre-redesign legacy runs, which carry no delay columns) |
| `configuration`                                                                             | `run-time` (env-var delay) vs `compile-time` (per-variant) sweep — the `--configuration` filter analyzes one layout at a time |
| `runtime_s`                                                                                 | the calculation runtime (the `calculation ... simulation time:` line)                     |
| `full_runtime_s`, `init_time_s`, `sim_steps`                                                | the run's full simulation time, its initialisation time, and its `-s` step count (the metrics the pre-redesign record dropped) |
| `rep`                                                                                       | the row's position among all logs of its (machine, setup, algorithm, grid, delay) group in file order — a re-run of one nominal repetition appears as extra rows in the group and shifts the later rows' `rep` |
| `nominal_rep`                                                                               | the repetition the run declared (the metadata's `rep`) — what selects the declared repetition |
| `started_utc`, `commit`, `binary_sha256`, `picongpu`, `mallocmc`, `gpu`, `gpu_driver`, `cuda_version`, `cpu`, `compiler`, `host`, `slurm_job` | the provenance of the log file the run came from (*Run logs*; per log, repeated on all of its runs, empty where the log or its metadata carries nothing) |
| `log`                                                                                       | the log's own file name                                                                  |
| `flag_sha`                                                                                  | the 8-char sha of the flags-file line the run came from (the filter that separates a flags-edit re-run when binary and commit are identical) |
| `hw_os`, `cxx_flags`, `cuda_flags`, `build_type`                                            | the build- and host-facts the metadata records                                           |
| `superseded`                                                                                | the vintage flag (the paragraph above)                                                   |

**The figures** (built from `output/results.h5`, one script each):

- `figures/sweeps-<machine>.pdf` — the per-machine delay sweeps: one row
  per algorithm, the malloc sweep (free delay 0) on the left and the free
  sweep (malloc delay 0) on the right, all axes sharing the x- and
  y-axes; the median with IQR error bars per delay, the fitted model with
  its bootstrap sleeve, the extrapolation to zero native cost, and the
  scenario's combined fit as a heavy line.
- `figures/sweeps-shared-<machine>.pdf` — the per-machine forest of the
  combined (shared-parameter) fit: one row per (scenario, algorithm), the
  individual W / N_malloc / N_free (dots with error bars) against the
  values fitted once across the algorithms, and the individual vs
  combined A_malloc / A_free, each A_* annotated with its fraction
  f = A/T0.
- `figures/foil_lct.pdf` — the FoilLCT bar chart of the zero-delay runs:
  one bar per allocator (in the file's `algorithm_order`), median with
  IQR error bar; the Kruskal significance is in the `foil_pvalue` table.
- `figures/kelvin_helmholtz.pdf` — the KelvinHelmholtz violin chart of
  the zero-delay runs relative to the ScatterAlloc reference runtime, one
  violin per allocator, one column per estimated particle memory.

Notes:

- The fits are bounded `scipy.optimize.curve_fit` of the 1-D model
  `T(s) = W + N*s + A*s0/(s+s0)` (sweep on one delay) or the two-operation
  model `T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0)`
  (combination sweep); the full model, the bounds, and the diagonalization
  caveats of the bootstrap are documented in `analysis/allocation_model.py`.
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
  zero-delay runs of *every* hardware: the frozen legacy runs
  (`legacy/legacy_results.h5`) plus the (0, 0) baseline runs of each sweep
  machine's delay combination sweeps. Both are grouped by the short
  hardware name (e.g. the legacy runs of the old `hal` cluster and the
  `hal` sweep machine's baselines are both plotted as `A30`).
- The figure row order is the `algorithms` list of `config.json` (recorded
  in the results file); the hardware display order of the comparison figures
  is fixed in `analysis/results_io.py`.

## Reproducing the analysis

The data prerequisites — what a fresh clone can and cannot produce, and
where the data to produce it lives — are in *Where the data lives*; here
is the environment only.

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

## RO-Crate

The benchmark is also an [RO-Crate](https://www.researchobject.org/ro-crate/):
`make rocrate` (via `make_rocrate.py`) generates the metadata file
`ro-crate-metadata.json` (git-ignored, removed by `make clean`) at the
repository root and validates it. The metadata is a derived artifact: it is
rebuilt from the ground truth on every invocation, exactly like
`results.h5`, and never written during a run. The crate describes the
repository as one research object:

- **The harness as a workflow** (the RO-Crate *Workflows and scripts*
  conventions; the metadata requirements of the Workflow RO-Crate profile):
  the `Makefile` is the main workflow, the shell/python scripts its steps,
  `MACHINE`/`PROFILE`/`PARAM_DIR`/`REPEATS`/`REP` its input parameters,
  the run logs, `results.h5` and the figures its outputs.
- **The runs as provenance** (the Process Run profile): one `CreateAction`
  per grid-run log, reading the log's self-describing metadata line —
  instrument the binary used (sha256, build facts, the PIConGPU/mallocMC
  pins), object the flags/config/parameter/profile files, `environment`
  the imposed delay and the slurm job, agent the run's user, result the
  log. Runs are append-only, so every vintage is described: a log whose
  identity's run stamp does not list it is annotated as superseded, on the
  same keys the analysis uses.
- **The analysis as actions**: one `CreateAction` for `results.h5` (from
  the run logs and the frozen legacy table, all vintages) and one per
  figure.

The declared conformance is RO-Crate 1.3 plus the two profile statements
on the root data entity. The crate is deliberately *not* packaged for
WorkflowHub ingestion: the pipeline is a Makefile, not one of the workflow
languages WorkflowHub supports. The built-in validation runs with the
target; the check additionally loads the crate with the official `rocrate`
package when it is installed (`pip install rocrate`) — reported as a
warning only, since its newest release supports crate versions up to 1.2.

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

## Troubleshooting

- `make check` fails before building or running: `config.json` (or a file
  it references) is invalid — `python3 config.py check` prints the
  details.
- The sweep tables (`group_stats`, `fits`, `baselines`) are empty: this
  checkout has no run logs — the data lives on the machines, *Where the
  data lives*.
- The FoilLCT / KelvinHelmholtz figures are empty or missing hardware:
  the frozen legacy table was not built (`make legacy-results`, *Where the
  data lives*); a sweep machine contributes only if it ran its (0, 0)
  baseline.
- The analysis picks up the wrong Python package versions: use the locked
  environment (`make env`) or export `PYTHONNOUSERSITE=1` (*Reproducing
  the analysis*).
- A configured machine has no output directory yet (e.g. `rosi-a100`
  before its first run): the figures and the RO-Crate skip it and print a
  note.
- `make clean` removes `build/`, `figures/`, `output/results.h5` and
  `ro-crate-metadata.json`; `make distclean` removes `src/` as well;
  neither touches the run stamps — only `make clean-runs` removes
  finished runs (*Running the benchmark*).
