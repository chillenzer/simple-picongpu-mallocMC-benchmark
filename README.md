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
performance model are documented in *The method*).

The sections are ordered for the reader: *The method* (what is measured
and what the numbers mean), *What is benchmarked* (what was run), *Where
the data lives* (what is and is not in the repository), and then how to
run, change, and analyze the benchmark. *Where to look* indexes the goals.

## Where to look

| You want to ...                         | Read                                              | Open first                          |
|-----------------------------------------|---------------------------------------------------|-------------------------------------|
| verify the measurement and the model    | *The method*                                      | `analysis/performance_model.py`      |
| understand the small-delay KHI anomaly  | *Limitations*                                     | `figures/kelvin_helmholtz.pdf`       |
| reproduce the numbers and figures       | *Where the data lives*, *Reproducing the analysis* | `make env`, then `make`            |
| run the benchmark, or add a machine     | *Running the benchmark*                           | the `machines` table of `config.json` |
| extend the benchmark                    | *Changing what is benchmarked*                    | `config.json`                       |
| use the data or the figures             | *Analysis*                                        | `output/results.h5`                 |
| contribute (code, logs, metadata)       | *Repository layout*, *Run logs*, *Code style*     | the file at hand                    |

## The method

The measured quantity is the runtime of the full PIConGPU calculation
(the `calculation ... simulation time:` line of the run's output) with
nanosecond delays imposed on every allocation and free request of the
run. The creation policy and the allocator config are compiled in, the
delays are not: mallocMC reads them at run time from the
`MALLOCMC_MALLOC_DELAY` and `MALLOCMC_FREE_DELAY` environment variables,
so one build per (commit, example, algorithm, config) serves the whole
sweep.

**The delay injection.** The delays live in the `add-delay` branch of
mallocMC (the pinned dependency of *What is benchmarked*): a busy-wait
on the 64-bit device global timer injected at the top of
`DeviceAllocator::malloc` / `DeviceAllocator::free` and held on the
`DeviceAllocator` itself, so every creation policy reacts. The
`__nanosleep` intrinsic was considered first; its wake-up guarantee
turned out too weak and produced step-like sweeps, so the busy-wait
replaced it.

**The performance model.** With no imposed delay the measured runtime is
the simulation's real runtime with each allocator — the `(0, 0)`
baseline — and that end-to-end runtime is the benchmark's headline
algorithm comparison. The delay arms characterize how the allocator
interacts with the pipeline: each of the N calls on the critical path
takes the imposed delay s, so at large delay the runtime is the straight
line `T = W + N*s` (W the no-delay runtime). The delay does not simply
add, though — part of it is hidden by the parallel work that runs while
the allocator spins — so the absorbed part fades over the sleeptime scale
s0 and a sweep is fitted with the performance model

```
T(s)  = W + N*s + A*exp(-s/s0)                (one delay)
T(m,f) = W + N_m*m + N_f*f + A_m*exp(-m/m0)
         + A_f*exp(-f/f0)                     (both delays)
```

which equals the baseline T0 = W + A (one delay; T0 = W + A_malloc +
A_free with both) at zero delay and approaches the asymptote W + N*s,
where every call's delay is exposed. The fitted parameters are W
(baseline runtime), N (calls per run; N_malloc / N_free per operation), A
(the total delay absorbed per run; A_malloc / A_free per operation), and
the fade scale s0 (m0, f0), the e-folding delay of the absorption. The
exponential was chosen as the model's fade term from a family of
candidates with the same endpoints (hyperbola, Lorentzian,
truncated-linear, quadratic); the measured comparison is
`figures/fade-models.pdf`. The per-call absorbed slack c = A/N is how
much imposed delay per call the pipeline hides before it reaches the
runtime. The ratio f = A/T0 (f_malloc = A_malloc/T0, f_free = A_free/T0)
is a convention-dependent slack ratio — absorbed delay over zero-delay
runtime — not a runtime budget and not the native allocation cost: the
absorbed delay A measures the pipeline's slack, and a run with more
in-flight work has a larger f even if the allocator's native cost is
unchanged.

The fit is a bounded `scipy.optimize.curve_fit` (W, N, A >= 0, s0 in
[0.05*s_min, 0.5*s_range]) with a robust linear solution over a log-s0
grid as the initial guess — reported as the fit when `curve_fit` does
not converge; the parameter errors are propagated from the fit
covariance, and the figures carry a bootstrap sleeve. The
microbenchmark suite of `make_microbench.py` measures the native per-call
cost c_a (ns) independently; where its frozen table supplies a cost for a
group's hardware and allocator, the unconstrained fit above stays the
primary (absorbed-slack) reading and the fits row also reports the native
cost and the absorbed slack relative to it (`slack_over_native_*`), while a
separate `A = N*c_a` constrained fit — fitting (W, N, s0) only, the
`fits_ca` table — gives the native-cost budget. Without matching
microbenchmark data a group stays unconstrained. The model, the bounds,
the constraint, and the bootstrap's caveats are documented in the
`analysis/performance_model.py` module docstring, which is the single
source of truth for the model.

**Why the sweep is shaped like this.** The delay arms run a log grid from
100 ns to 1e8 ns so the large delays anchor the asymptote W + N*s —
which matters most for free, whose absorbed delay fades slowly — and they
omit the zero-delay point, so the (0, 0) baseline runs are outside every
fit and provide the T0 baselines and the paper figures instead. The arms
are run in two phases (*Running the benchmark*): a log-dense first scan
through the bend band plus both asymptote anchors — enough to check the
interim fit early — and the full ladder as an incremental extension. An
optional joint grid couples both delays so the two-operation fit is
constrained off the arms; it is empty by default, because a sparse
ladder plus a joint grid is near-degenerate for the fit.

## What is benchmarked

- **Machines**: `hal` (NVIDIA A30), `rosi` (NVIDIA V100) and
  `rosi-a100` (NVIDIA A100); the rosi machines load the `hopper`,
  `GCCcore/14.3.0` and `git/2.50.1` modules before their profile. The
  `machines` table of `config.json` maps each machine label to its
  profile to source, output directory for the run logs, hardware name
  (the figure titles), and modules to load.
- **Allocation policies**: `FlatterScatter`, `ScatterAlloc`, `Gallatin`
  (the `algorithms` list in `config.json` and
  `param/<Algorithm>/mallocMC.param.in`, the template each algorithm's
  allocator is rendered from — *Changing what is benchmarked*)
- **Commits**: the `commits` section of `config.json` names each
  independent dependency pair — a PIConGPU checkout plus a mallocMC
  checkout — by a logical name (e.g. `default`). The Cartesian matrix
  spans every commit (`make build` always builds all of them;
  `make runs COMMIT=<name>` runs one) cross every example, algorithm,
  config, and delay combination. With a single commit the logical name
  is `default` and the harness behaves as before.
- **Configs**: the `configs` section of `config.json` names, per
  algorithm, the allocator variants to benchmark. Each config is a set
  of scalar heap parameters (the `heap` object) plus, for algorithms
  that expose a separate hash template slot, a named hash profile (the
  `hash.profile` name, a header under `param/<Algorithm>/profiles/`).
  The `default` config reproduces the allocator's shipped parameters,
  so a single-commit, single-config benchmark is unchanged. `ScatterAlloc`
  ships three non-default variants (`16MiB-page`, `tuned-hash`, and
  `hash-from-page`, the latter demonstrating a hash derived from the
  heap parameters) for the cross-config comparison figure.
- **Delay combinations** (nanoseconds, the `delays` section of
  `config.json`; the Makefile run targets derive the combination set of
  each sweep phase from it, `python3 config.py list run-matrix
  [initial|arms]`): a `(0, 0)` baseline plus the arm ladders — the
  malloc-delay sweep with free delay 0 and the free-delay sweep with
  malloc delay 0 over the 12 log-spaced values `100`…`100000000`
  (1/4-decade steps, skipping the intermediate steps between `1e5` and
  `1e6`) — in the `arms` phase (25 combinations, the default; the
  optional joint grid is empty by default), or the log-dense `initial`
  subset of the ladders (17 combinations). **Baseline-only**: when the
  `delays` section is absent (or the arm ladders are empty) the
  benchmark runs only the `(0, 0)` baseline — *Changing what is
  benchmarked* — and the fit/absorption figures are skipped, the
  zero-delay baselines plus the two comparison figures (*Analysis*)
  being the primary output.
- **Cost**: per machine per repetition, (commits x 2 examples x 3
  algorithms x 4 configs) x the phase's combinations; with one commit
  and one config that is the old 2 examples x 3 algorithms x 17
  initial-phase combinations = 102 runs, or x 25 arms-phase
  combinations = 150 runs. Each (example, algorithm, config) is one
  build, so the config count multiplies the build count too.
- **Examples**: `KelvinHelmholtz` (3D, grid sizes 128^3, 256x128x128,
  256x128x256, 1500 steps) and `FoilLCT` (2D, 256x1280 cells, 2000 steps);
  their command lines are the `flag_lines` of the `examples` objects in
  `config.json` (*Changing what is benchmarked*).
- **Layout**: `build/<Commit>/<Example>/<Algorithm>/<Config>/` holds one
  build; its flag lines are run once per combination:
  `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`.

The Makefile resolves each commit's dependency versions from the `commits`
section of `config.json` (the shipped pins; `make check` prints the
table):

| commit   | dependency | source | pinned to |
|----------|------------|--------|-----------|
| `default`| PIConGPU | ComputationalRadiationPhysics/picongpu | `6e7d58bb` |
| `default`| mallocMC | chillenzer/mallocMC, `add-delay` branch (`BOOST_LANG_*` guards compatible with PIConGPU's vendored alpaka 2.0; run-time malloc/free delays via the `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables, injected as a busy-wait on the device global timer at the top of `DeviceAllocator::malloc` / `DeviceAllocator::free`; the delays are held on the `DeviceAllocator` itself, so every creation policy reacts) | `9de11602` |


## Where the data lives

The repository holds the harness, the analysis, and the pins; the
benchmark data does not. Data lives on the benchmark machines and is
git-ignored; the derived artifacts are rebuilt from it by `make`.

- **In the repository**: `config.json` and the harness/analysis scripts,
  the pinned dependency hashes, the locked analysis environment, and the
  `legacy/` scripts and docs.
- **On the machines (never committed)**: the run logs in each machine's
  output directory (`output/<machine>-sleeptimes/`, incl. its `sessions/`
  folder of session logs), the run stamps (`run-stamps/`), the frozen
  legacy data (`legacy/logs/`, `legacy/legacy_results.h5`), and the
  microbenchmark data (the memmansurvey allocation-test CSVs under
  `microbenchmarks/data/`, frozen into
  `microbenchmarks/microbench_results.h5`). The legacy data is the piece a
  fresh checkout must obtain for the comparison figures: copy
  `legacy/logs/` from a machine that still holds the historical `output/`
  tree (or run `move_legacy_logs.sh` there), then `make legacy-results`
  builds the frozen table and `make legacy-verify` checks it (the full
  procedure is in `legacy/README.md`); `make microbench-results` and
  `make microbench-verify` do the same for the microbenchmark CSVs.
- **Derived (never committed, rebuilt by `make`)**: `output/results.h5`,
  the `figures/` PDFs, `ro-crate-metadata.json`, and
  `ro-crate.crate.zip`; `make clean` removes them.

On a fresh checkout without the machine data, `make` still runs end to
end: each step skips its absent data instead of failing. The freezes
(`make freeze`) skip a source whose raw data is absent; `results` computes
the tables from whatever frozen tables exist; and each figure family
(`make figures-picongpu`, `make figures-microbench`) skips when its data is
absent, so `make figures` produces only the figures its data supports.
With the legacy data present, the sweep tables (`group_stats`, `fits`,
`baselines`) come out for the sweep machines and the FoilLCT /
KelvinHelmholtz figures and their statistics are computed from the frozen
legacy table. The archived-but-excluded legacy runs (the
`hal-sleeptimes-nanosleep` experiment, outside the benchmark matrix) are
parsed but attributed to the empty hardware name and dropped by the
analysis, which records the exclusion in the results file's attributes.

## Running the benchmark

Everything runs on the HPC machine with the matching profile; the entry
points must be invoked from the repository root. The benchmark
configuration (examples, algorithms, delay sweep, dependency pins, build
flags, and the per-machine profiles and output directories) lives in
`config.json`. Both entry points validate it first and can stop right
after resolving it, which is the way to sanity-check a change before a
long build or benchmark:

```
make check                              # includes both sweep phases' matrices
python3 config.py list run-matrix       # the arms phase (the default)
python3 config.py list run-matrix initial
```

The Makefile variables:

| variable        | meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `MACHINE`       | a row of the `machines` table of `config.json` (hal, rosi, rosi-a100) |
| `PROFILE`       | the profile to source for the build (`profiles/<machine>.sh`)      |
| `PARAM_DIR`     | the parameter overlay directory (default `param`)                  |
| `REPEATS`       | full-sweep repetitions of `make runs` (default 1)                  |
| `REP`           | restrict the invocation to one repetition (the slurm case)         |
| `PHASE`         | `initial` (the fast first scan) or `arms` (the full ladder, the default); the run stamps make the phases incremental |
| `COMMIT`        | restrict the run targets to one commit (a name of `config.json`'s `commits`); `make build` always builds every commit |

1. Build all commits, examples, and algorithms (slow the first time: one
   full PIConGPU build per commit, example, algorithm, and config):

    ```
    make build PROFILE=profiles/hal.sh PARAM_DIR=param
    ```

    That also patches `PICSRC=` in the given profile in place, pointing it
    at the first commit's PIConGPU checkout (the tool path the
    `pic-create` step is driven from — *The method*, the build note); the
    other commits' toolchain comes from the profile the first commit loads.
    Each build's allocator `DeviceHeap` is rendered by
    `python3 config.py render-param <Algorithm> <config>` from the
    `param/<Algorithm>/mallocMC.param.in` template and the config's scalars,
    and the algorithm's `param/<Algorithm>/profiles/` headers are copied in
    — so the allocator is assembled from `config.json`, not from a checked-in
    `mallocMC.param` (*Changing what is benchmarked*).

    The build is incremental and can be re-run at any time; it reuses
    `src/` and `build/` and only repeats the parts that are out of date:

    - `src/<commit>/picongpu` is re-cloned only if the directory is not a
      git checkout at the commit's pinned PIConGPU hash; a changed pin
      triggers `git fetch && git checkout` (one checkout per commit).
    - An input directory (`build/<Commit>/<Ex>/<Algo>/<cfg>/`) is rendered
      and regenerated when the rendered allocator parameters, the profile,
      or the `param/<Algo>*/`, `param/<Ex>*` files change (the
      `build/.../include/.input-stamp` records the prepared state, keyed on
      the rendered parameters' fingerprint).
    - A build (`pic-build`) is skipped when the input, the build flags, the
      mallocMC pin, the profile content and the toolchain versions
      (gcc/cmake/nvcc) are unchanged (`build/.profile-env`,
      `build/.toolchain` and `build/.build-flags` record these); otherwise
      `pic-build` runs incrementally. `make -j` builds the independent
      (commit, example, algorithm, config) quadruples in parallel; the
      default is serial.

    `make clean` removes `build/`, `figures/`, `output/results.h5`,
    `ro-crate-metadata.json` and `ro-crate.crate.zip`, `make distclean`
    removes `src/` as well; neither touches the run stamps (finished runs
    stay finished after a rebuild). Alternatively delete just
    a single `build/<Commit>/<Ex>/<Algo>/<cfg>/` folder, or only its
    `.input-stamp`, to rebuild one quadruple.

2. Run the benchmarks (the sweep, once per repetition):

    ```
    make runs MACHINE=hal                    # arms phase, REPEATS=1 (the defaults), all commits
    make runs MACHINE=hal PHASE=initial      # the fast first scan (17 combinations)
    make runs MACHINE=hal REPEATS=3          # three full-sweep repetitions
    make runs MACHINE=rosi REPEATS=3 REP=2   # only repetition 2 (one slurm job)
    make runs MACHINE=hal COMMIT=<name>      # only that commit (a name in config.json)
    make full MACHINE=hal                    # build, then run (PHASE and COMMIT pass through)
    make clean-runs [MACHINE=hal] [COMMIT=<name>]  # forget finished runs (re-runs add a vintage)
    make sweep-status MACHINE=hal            # which runs are stamped (arms phase)
    ```

   The sweep is two phases (`PHASE`, default `arms`), from the `delays`
   section of `config.json`: `initial` runs the baseline plus the
   `delays.arms.initial` arm subset (17 combinations: enough to anchor the
   asymptote and the bend band of the fit without the near-degenerate joint
   points), and `arms` runs the baseline plus the full arm ladder plus the
   joint grid when one is configured (25 combinations, the joint grid
   currently empty). The phases are incremental through the run stamps: run
   `PHASE=initial` first, check the interim fit (*Analysis*), then run
   `PHASE=arms` to extend the same series — it re-runs only the eight
   combinations without an up-to-date stamp. Per machine, the initial phase
   costs about 80 h (hal) or 41 h (rosi) per repetition, and the extension
   about 67 h or 35 h, against 172 h / 89 h for the old 34-combination
   design.

   On a slurm machine, the `log_run_<machine>.sh` launcher drives one slurm
   job per repetition: `sbatch log_run_rosi.sh 1` (with `REPEATS` in the
   environment) runs exactly one full-sweep repetition — all slurm
   allocation options come from the `sbatch` line, the scripts carry no
   `#SBATCH` directives.

    **How a run works.** One run is one (commit, example, algorithm,
    config, combination, repetition) identity: one (example, algorithm,
    config) build through one `(malloc delay, free delay)` combination
    derived from the `delays` section of `config.json`, once per
    repetition, in sweep order (commit, example, algorithm, config,
    repetition, then the delay combination). Every flag line of the
    example — the `flag_lines` of the example's object in `config.json`,
    serialized by `python3 config.py flag-lines <Example>` — is run as
    `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`
    from `build/<Commit>/<Ex>/<Algo>/<cfg>/` (via `run_folder.sh`), and
    each flag line writes one self-contained log — its anatomy is in
    *Run logs*. Each finished run writes a stamp (in *Run logs* too): an
    existing stamp is what makes `make runs` skip a run, so an interrupted
    series resumes where it stopped, and the stamps depend only on the
    example's flag lines *as serialized* (a fingerprint stamp of
    `config.py flag-lines <Ex>`), so a rebuild never invalidates finished
    runs (`make clean-runs [MACHINE=<m>] [COMMIT=<name>]` forgets them).
    Runs are append-only: a re-run of a (combination, repetition) writes a
    new vintage of the logs next to the older ones — nothing is ever
    removed (a re-run in the very same second merely appends a counter to
    the file name) — and the analysis flags the older vintages
    `superseded = 1` (*Analysis*), so a re-run adds new rows instead of
    replacing anything.

Single run (one built folder, one combination, one serialized flag line):

```
bash run_folder.sh build/default/FoilLCT/FlatterScatter/default profiles/hal.sh 10000 0 \
    "-g 256 1280 -d 1 1 --periodic 1 0 -s 2000 -p 1"
```

(`run_folder.sh <folder> <profile> <malloc-ns> <free-ns> <line>`; the fifth
argument is one of the example's serialized `flag_lines`, word-split into
`picongpu`'s arguments, and the delays default to `0` when omitted.)
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

- **Name and location.** One log per grid run (one of the example's
  serialized flag lines), in the machine's output directory:
  `run_<machine>_<commit8>_<Ex>_<Algo>_<config>_m<M>_f<F>_r<rep>_<line-sha8>_<time>.txt`;
  a same-second re-run merely appends a counter to the file name.
  `<commit8>` is the first eight hex digits of the commit's PIConGPU hash
  (the value that separates the commits' runs on disk), and `<config>` the
  allocator config name of the (commit, config) build that ran.
- **Header.** A two-line header: a human one-liner `# run:` line and the
  self-describing `# metadata:` JSON line (schema 1). The run metadata
  carries, all best effort ("unavailable" placeholders, never an error):
  `schema`, `kind` (`run`), `ts`, `machine`, `hostname`, `user`, `orcid`
  (the runner's ORCID iD from `$ORCID`, when exported — *RO-Crate*), the
  git state, `slurm_job` when running under slurm, `pins` (the run's
  commit's dependency pins from `config.json`), `hw` (the GPU names and
  their driver, the CPU model, the operating system), the build facts read
  from the binary's own `CMakeCache.txt` (compiler, CUDA, the build flags,
  the build type), the sha256 of the binary, and the run block (example,
  algorithm, the **commit** and the **config** of the run, the imposed
  `[malloc, free]` delays, the repetition, the serialized flag line and the
  total line count, and the 8-char sha of the line).
- **Body.** That grid run's `set -x` trace and full output, including the
  `calculation ... simulation time:` line the runtime is parsed from.
- **Session logs.** The `log_{setup,run}_<machine>.sh` launchers write a
  session log per launch to `<output dir>/sessions/` (a `# metadata:` line
  with `kind: setup`, plus the environment block and progress); the folder
  is a free-text backup that the analysis never reads and that runs are
  never superseded.
- **Stamps.** `run-stamps/<machine>/<commit>/<Ex>/<Algo>/<config>/<M>_<F>/rep-<R>.stamp`,
  the run's log paths one per line — i.e. its current vintage. (The
  `<commit>` and `<config>` levels in the path are what make the series
  separable on disk; after the switch to them, a machine-side
  `run-stamps/` from before the redesign is ignored and finished runs
  re-run as new vintages unless `make clean-runs` is run first.)
- **Format rule.** A log is new format if and only if its `# metadata:`
  line's JSON has `"schema": 1`; the pre-redesign logs do not carry it and
  must be frozen under `legacy/` (`make legacy-results`) instead of being
  parsed.

## Changing what is benchmarked

Most of it is now in `config.json` (run `python3 config.py check` after
editing; `make check` / `python3 config.py list run-matrix` show what
would be done). The table of knobs is:

| Goal | Edit |
|---|---|
| Change a heap number (`pageSize`, …) | `configs.<Algo>.<cfg>.heap.<field>` in `config.json` |
| Switch to another hash scheme | `configs.<Algo>.<cfg>.hash.profile` (a name → `param/<Algo>/profiles/<name>.hpp`) |
| Hash derived from heap parameters | point `hash.profile` at a header that uses `T_HeapConfig` (e.g. `HashFromHeap`); no extra key |
| Tune a hash constant | edit a number in `param/<Algo>/profiles/Hash*.hpp` (or add a new header) |
| New PIConGPU / mallocMC version sweep | append a commit entry to `commits` |
| New grid / steps / flags | edit the `flag_lines` of the example in `config.json` |
| New allocation policy | append to `algorithms` + create `param/<Algorithm>/` (a `.in` template + optional `profiles/`) |

- **Delay combinations**: edit the `delays` section of `config.json`
  (`baseline`, `arms.values`, `arms.initial`, `joint.values`; the grid
  values are listed in *What is benchmarked*); the Makefile run targets
  derive the combination set of each sweep phase from it: `baseline` is
  the `(0, 0)` reference run, `arms.values` is the arm ladder (the
  single-delay sweeps, one value at a time with the other delay held at
  0, on a log grid from `100` to `1e8` ns in 1/4-decade steps),
  `arms.initial` is the phase-1 subset of the ladder (a log-dense ladder
  through the bend band plus both asymptote anchors, so the fit is fully
  anchored with fewer, cheaper runs), and `joint` is an optional grid
  coupling both delays that constrains the two-operation fit off the arms
  (empty by default: a sparse arm ladder plus a joint grid is
  near-degenerate for the 7-parameter fit, so add joint values only with
  the full ladder, and keep them on the ladder). The delays are applied
  at run time via `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY`, so
  changing the sweep requires no rebuild (why the grid is shaped like it
  is: *The method*).
  - **Baseline-only**: the `delays` section may be **absent** (or carry
    only an empty arm ladder) — then the benchmark runs just the `(0, 0)`
    baseline. The Makefile print for `make check` reports `runs: baseline
    only (no delay sweep configured)` instead of the two phase lines,
    `PHASE` is ignored, the analysis fit / absorption sections are
    suppressed (*Analysis*), and the sweep figures are skipped. The
    zero-delay baselines plus the `configs-` / `commits-` comparison
    figures (*Analysis*) are the primary output for a baseline-only run
    series; with no delay arms there is nothing to fit.
- **Grids / steps / other picongpu flags**: edit the `flag_lines` of the
  example's object in `config.json`. Each `flag_lines` entry is a mapping
  of picongpu option name → integer scalar or list of integers (e.g.
  `{"d": [1,1,1], "g": [128,128,128], "periodic": [1,1,1], "s": 1500}`),
  serialized to the command line in JSON insertion order: a one-character
  key becomes `-<key>`, a longer key `--<key>`; a scalar is one token, a
  list is space-joined. Edit the mapping, not the serialized text, and
  `python3 config.py flag-lines <Example>` is the sole authority for what
  each run's command line will be. (Editing the `flag_lines` of an
  example changes the flag-lines fingerprint and therefore invalidates
  the existing run stamps of that example's runs.)
- **Allocator configuration**: each algorithm is one
  `param/<Algorithm>/mallocMC.param.in` assembly template plus an optional
  `param/<Algorithm>/profiles/` of user-authored C++ headers (hash
  profiles, heap-layout structs). The template declares its heap scalars
  in a leading `// heap-args: <a> <b> ...` comment (the camelCase names of
  the non-type template parameters); the config's `heap` object supplies
  the values (camelCase keys, matched to the template's declared args);
  `python3 config.py render-param <Algorithm> <config-name>` substitutes
  them in place, with `U` suffix on integers and `true` / `false` on
  booleans, into the `param/<Algorithm>/mallocMC.param` that `pic-build`
  compiles. Every hash profile is a `template<class T_HeapConfig>` struct
  referenced by name (the `hash.profile` value) and instantiated against
  the heap struct, so a hash may be a fixed constant
  (`HashDefault` / `HashTuned`), derived from the heap parameters
  (`HashFromHeap`), or forwarded from mallocMC's own default
  (`FsHashDefault` for `FlatterScatter`). Gallatin has no hash profile;
  the template carries only the three heap scalars. `python3 config.py
  list configs <Algorithm>` lists the config names (in JSON order);
  `make check` fails on any template placeholder a config does not cover.
- **New commit**: append an entry to the `commits` section of
  `config.json` — `{name, picongpu: {url, hash, path}, mallocmc: {url,
  hash, path}}` (the paths default to `src/<name>/picongpu` / `.../
  thirdParty/mallocMC` when absent; the hashes must be 40-hex). The
  logical name is what the runs, logs, and the analysis tables carry
  (`run.commit` in the metadata, `dep_commit` in results; the short PIConGPU
  hash appears in the log file name and in the `commits-` figure labels).
  The two names may be mixed freely in a single run matrix: the full
  cross product `commits × examples × algorithms × configs × (malloc,
  free)` runs, with `make build` always building every commit and `COMMIT=`
  in `run` / `full` / `clean-runs` / `sweep-status` restricting a single
  commit's runs.
- **New config variant**: add a name under `configs.<Algorithm>` in
  `config.json` (a `heap` object of the algorithm's scalars and, where the
  algorithm exposes a hash template slot, a `hash.profile` name). For a
  new hash formula, add `param/<Algorithm>/profiles/<Profile>.hpp`
  (header with the standard `SPDX` header) and reference its name in
  `hash.profile`; for a fixed-constant hash variant, a profile header is
  still required (the profile is the C++ surface the renderer instantiates;
  the renderer itself only substitutes scalars and the profile name, never
  arbitrary C++). `python3 config.py check` fails until the new profile
  file exists.
- **PIConGPU / mallocMC version**: see *New commit* (the `dependencies`
  section no longer exists; a config with both `dependencies` and
  `commits` is rejected by `python3 config.py check`).
- **Build flags**: the `build` section of `config.json`: `cxx_flags` is
  passed to both `CMAKE_CXX_FLAGS` and `CMAKE_CUDA_FLAGS` (the Boost
  `std::source_location` constexpr bug is in the C++ compiled by nvcc, not a
  CUDA-specific flag) and `extra_cmake_flags` are appended verbatim. The
  build flags are global (not per commit); a per-commit CMake flag is not
  an implemented extension point.
- **New example**: add it to `examples` in `config.json` (its name and
  `flag_lines`), and optionally `param/<Example>/*.param` for per-example
  parameter overlays.
- **New algorithm**: add it to `algorithms` in `config.json` and provide
  `param/<Algorithm>/mallocMC.param.in` (the template) plus a `default`
  entry under `configs.<Algorithm>` (and the `profiles/` headers the
  template references).

**What the analysis needs of a setup** (otherwise the table comes out
empty):

| table           | requirement                                                              |
|-----------------|---------------------------------------------------------------------------|
| `fits`          | the group's runs (per `dep_commit` and `config`) span at least two distinct values of a delay (the 1-D model falls back to that one delay) |
| `shared_fits`   | the scenario (machine, `dep_commit`, `config`, example, grid) has at least two algorithms and at least one varying delay |
| `baselines`     | the sweep machine runs a (0, 0) baseline                                  |
| `foil` / `khi`  | zero-delay runs per hardware: the frozen legacy runs plus every sweep machine's (0, 0) baselines |
| `configs-` figure | a (machine, commit) with an algorithm carrying at least two configs       |
| `commits-` figure | a (machine, config) with at least two commits                             |

## Repository layout

- `config.json` — the single source of truth for what the harness runs and
  builds: the `examples` (name plus serialized-able `flag_lines`) and
  `algorithms` lists, the `commits` (per-commit PIConGPU/mallocMC pins), the
  `configs` (per-algorithm allocator variants: heap scalars and hash-profile
  names), the `delays` sweep (optional; absent = baseline-only), the build
  flags, the per-machine `machines` table (*What is benchmarked*), the
  `microbench` table, and the optional `people` table (login to name and
  ORCID iD; the harness uses it to identify the run's operator in the
  RO-Crate metadata, and `config.py check` validates the entries,
  *RO-Crate*).
- `config.py` — the python3 bridge the harness uses to read `config.json`
  (`get` / `list` lookups) and to validate it (`check` also validates the
  `configs` against each algorithm's `param/<Algorithm>/mallocMC.param.in`
  template and the referenced `profiles/` headers). The same module is the
  sole authority for the two rendered views: `flag-lines <Example>` prints
  the serialized command lines of the example (the source of the run-stamp
  fingerprint and of `logmeta.py`'s metadata) and `render-param
  <Algorithm> <config-name>` prints the rendered `mallocMC.param` C++ into
  the build's include dir. It parses the file with the stdlib `json`
  module, so no third-party package (no PyYAML, no yq) is needed in the
  cluster environment.
- `Makefile` — the build harness, the run orchestrator, and the analysis
  driver: it clones each commit's PIConGPU and mallocMC and, from
  `config.json` (validated up front), prepares one input directory per
  (commit, example, algorithm, config) quadruple — rendering
  `param/<Algorithm>/mallocMC.param` from the `param/<Algorithm>/
  mallocMC.param.in` template and the config's scalars, copying the
  algorithm's `profiles/` in, and overlaying any `param/<Example>/*.param`
  and `param/<Example>/<Algorithm>/*.param` files — and builds it
  (`pic-build`). The run orchestrator and the analysis driver
  are documented in *Running the benchmark* and *Analysis*; `make
  rocrate` / `make crate-zip` in *RO-Crate*; `make check` prints the
  resolved configuration (the commits and their short hashes, the per-
  algorithm configs, the build-matrix count, and the run-matrix lines — or
  the baseline-only line when the delay arms are empty) and stops. The
  target comments in the header mirror this README.
- `run_folder.sh` — runs one already-built example folder, once per flag
  line; takes `folder`, `profile`, optional fourth (malloc delay, default
  `0`) and fifth (free delay, default `0`) arguments in nanoseconds, passed
  via the `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment
  variables, and sixth the one serialized flag line (word-split into
  `picongpu`'s arguments), so a per-grid log records one grid run.
- `run_stamp.sh` — the per-run worker of the `make runs` targets: one
  invocation is one (commit, example, algorithm, config, combination,
  repetition) identity; it sources the example's `flag_lines` via
  `config.py flag-lines`, computes the `<commit8>` token from the
  commit's PIConGPU hash, names the log and stamp paths with the commit
  and config in them, and invokes `logmeta.py run --commit <c> --config
  <cfg>` for the metadata line before calling `run_folder.sh`.
- `logmeta.py` — emits the self-describing `# metadata:` JSON line
  (schema 1) of the run and session logs; every fact is best effort
  ("unavailable" placeholders, never an error). The key list is in
  *Run logs*.
- `make_rocrate.py` — generates, checks, and packs the RO-Crate metadata
  of the repository (`make rocrate`, `make crate-zip`): the harness as a
  workflow, the benchmark runs as provenance, the analysis as actions
  (what exactly is described, see the *RO-Crate* section below).
- `log_{setup,run}_<machine>.sh` — per-machine launchers (hal, rosi,
  rosi-a100): log the environment, load the machine's modules (setup), and
  run `make build` / `make runs MACHINE=<machine>` with the machine's
  profile from the `machines` table in `config.json` (the sweep invocation
  values `REPEATS`, `REP`, `PHASE` and `COMMIT` pass through as environment
  variables; see *Running the benchmark*).
- `profiles/` — HPC environment profiles (module/spack setup, `PIC_BACKEND`,
  `PICSRC`). One per machine.
- `param/` — parameter files overlaying the example defaults, plus the
  allocator assembly templates: one `mallocMC.param.in` per algorithm
  (`param/<Algorithm>/`) and, under `param/<Algorithm>/profiles/`, the
  user-authored C++ headers (the hash profile and, where the template
  references one, the heap-layout struct) that
  `config.py render-param` instantiates into the build's
  `include/picongpu/param/mallocMC.param`; example-specific files
  (`param/FoilLCT/`), and optional per-(example, algorithm) overrides
  (`param/<Example>/<Algorithm>/`). No `mallocMC.param` files are checked
  in — they are rendered at build time, and the profile headers are the
  only checked-in C++.
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
- `analysis/performance_model.py` — the performance model, its constrained
  1-D/2-operation fits (plus the combined fit that shares W and the
  malloc/free call counts across a scenario's algorithms while each keeps
  its own saturation terms), and the bootstrap sleeve. The model is
  summarised in *The method*; the module docstring is the reference.
- `analysis/compute_results.py` — the single "numbers" entry point: parses
  the sweep machines' run logs (the `machines` table of `config.json`) and
  the legacy runs (from the frozen `legacy/legacy_results.h5`, when it is
  present — build it with `make legacy-results`), all grouped by their short
  hardware name, and copies the microbenchmark allocation-cost tables (from
  the frozen `microbenchmarks/microbench_results.h5`, when it is present —
  build it with `make microbench-results`); then computes the group runtime
  statistics, the performance-model fits of every (machine, `dep_commit`,
  `config`, example, algorithm, grid) sweep (with the parameter covariances),
  the combined (shared-parameter) fit of every (machine, `dep_commit`,
  `config`, example, grid) scenario spanned by at least two algorithms, the
  zero-delay baselines (sweep machines), the per-arm absorbed-delay slack
  (the plateau deficit d and the per-call c = d/N), and the
  FoilLCT/KelvinHelmholtz figure statistics (all zero-delay runs, per short
  hardware name); writes everything to `output/results.h5` (*Analysis*) and
  prints nothing. The runs always carry all rows; the statistics and the
  fits are restricted by `--configuration` (run-time / compile-time),
  `--dep-commit <name>` (one commit), and `--config <name>` (one allocator
  config). Every table also carries the two benchmark dimension columns
  `dep_commit` (the logical commit name, metadata `run.commit`) and `config`
  (the allocator config name, metadata `run.config`), and the file records
  `commit_order` and a per-algorithm `config_order` attribute for the
  figure ordering.
- `analysis/summarize_results.py` — prints the summary tables from
  `output/results.h5` (group statistics, fits, the combined fit vs the
  individual fits per scenario, the slack-ratio summary (f = A/T0, the
  absorbed delay over the zero-delay runtime), the per-arm absorbed-delay
  slack (d, c = d/N), the No-delay runtimes table, the figure statistics,
  and the three microbenchmark allocation-cost tables (native, mixed
  workload, thread scaling); `--raw` adds the parsed runs). When the run
  series has no delay sweep (baseline-only), the fit / shared-fit /
  slack-ratio / absorption sections are suppressed and a single note is
  printed, so the zero-delay baselines are the headline.
- `analysis/plot_sweeps.py` — one delay-sweep figure per sweep machine
  (`figures/sweeps-<machine>.pdf`: one row per algorithm, the malloc and
  free delay sweeps side by side, all axes sharing the x- and y-axes,
  fitted curve + bootstrap sleeve, and the scenario's combined fit as a
  heavy line where one exists).
- `analysis/plot_shared_fits.py` — one forest figure per sweep machine
  (`figures/sweeps-shared-<machine>.pdf`: one row per (scenario, algorithm),
  the shared W / N_malloc / N_free values against each algorithm's
   individual fit, and the individual vs combined A_malloc / A_free,
   each A_* value annotated with its slack ratio f = A/T0, the absorbed
   delay's share of the zero-delay runtime T0 = W + A_malloc + A_free).
- `analysis/plot_configs.py` — the per-machine allocator-config comparison:
  for each (machine, commit) with an algorithm carrying at least two
  configs, `figures/configs-<machine>.pdf` (one row per scenario, one
  column per such algorithm; the zero-delay p50 runtime as a bar per config,
  the default config's bar hatched as the reference, per-row shared y-axes).
  `--dep-commit all` writes one figure per (machine, commit), each named
  `configs-<machine>-<commit>.pdf`. Skips (a note, no file) on a
  single-config axis.
- `analysis/plot_commits.py` — the per-machine dependency-commit
  comparison: for each (machine, config) with at least two commits,
  `figures/commits-<machine>.pdf` (one row per (scenario, algorithm); the
  baseline column marks each commit's zero-delay p50 with its PIConGPU short
  hex, and the sweep column draws one line per commit's log-alog malloc-delay
  sweep with its fitted model curve + sleeve). `--config all` writes one
  figure per (machine, config), each named `commits-<machine>-<config>.pdf`.
  Skips (a note, no file) on a single-commit axis. Both of these comparison
  figures consume the baselines only, so they are the primary output of a
  baseline-only run series.
- `analysis/plot_foil_lct.py` — the FoilLCT bar chart of the zero-delay
  runs (`figures/foil_lct.pdf`).
- `analysis/plot_kelvin_helmholtz.py` — the KelvinHelmholtz violin chart of
  the zero-delay runs relative to the ScatterAlloc reference
  (`figures/kelvin_helmholtz.pdf`).
- `analysis/plot_fade_models.py` — the performance-model fade-term
  comparison figure (`figures/fade-models.pdf`).
- `analysis/plot_microbench.py` — the microbenchmark allocation-cost figures
  (the native, mixed-workload, and thread-scaling costs, plus the allocator
  legend; from the frozen tables in `output/results.h5`).
- `analysis/plot_microbench_misc.py` — the microbenchmark diagnostic figures
  (the allocator utilisation and the allocation-graph figures; from the
  suite's raw CSVs under `microbenchmarks/data/`).
- `build/` — created by the Makefile; one CMake project per (commit,
  example, algorithm, config), and the per-commit `src/` dependency
  checkouts.

## Analysis

The analysis is split into *computing the numbers* and *drawing the
figures*, joined by the single HDF5 file `output/results.h5`:

- `compute_results.py` parses the run logs of the two sources they live
  in: the sweep machines of the `machines` table of `config.json` (the
  new format, one `# metadata:` JSON line per log; the group statistics,
  the performance-model fits and the zero-delay baselines are computed for
  these) and the frozen legacy table `legacy/legacy_results.h5` (the
  pre-redesign logs, read grouped by their short hardware name; both are
  optional, and the analysis runs without either), and copies the
  microbenchmark allocation-cost tables from the frozen
  `microbenchmarks/microbench_results.h5` (optional). All runs are grouped
  by that short hardware name (`A30`, `V100`, ...; the FoilLCT /
  KelvinHelmholtz figure statistics cover the zero-delay runs of both
  sources). It prints nothing. Every row of the `runs` table carries
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
   any; then per machine: group statistics; the fits and the slack-ratio
   summary (f = A/T0, the absorbed delay over the zero-delay runtime);
   the combined (shared-parameter) fit of each multi-algorithm scenario
   compared algorithm by algorithm against the individual fits; the
   per-arm absorbed-delay slack (the plateau deficit d and the per-call
   c = d/N); the No-delay runtimes; the figure statistics). `--raw` also
   prints the parsed runs.
- one script per figure, each reading `output/results.h5` (all take
  `--show` to display the figure in a window): `plot_sweeps.py`
  (`figures/sweeps-<machine>.pdf`, `--machine` for one machine),
  `plot_shared_fits.py` (`figures/sweeps-shared-<machine>.pdf`, `--machine`
  for one machine), `plot_foil_lct.py` (`figures/foil_lct.pdf`),
  `plot_kelvin_helmholtz.py` (`figures/kelvin_helmholtz.pdf`),
  `plot_fade_models.py` (`figures/fade-models.pdf`), and the microbenchmark
  figures: `plot_microbench.py` (the allocation-cost figures, from the frozen
  table in `results.h5`) and `plot_microbench_misc.py` (the diagnostic
  figures, from the suite's raw CSVs).

The simplest way to run the whole thing (or any single figure) is the
`Makefile`:

```
make                      # all figures + the summary tables
make freeze               # freeze the raw sources (the PIConGPU legacy logs
                          #   and the microbenchmark CSVs) into their frozen
                          #   tables; a source with no data is skipped
make freeze-verify        # check the frozen tables against their raw data
make results              # only output/results.h5, from the run logs and the
                          #   frozen tables (a separate step: run `make freeze`
                          #   first to refresh the frozen tables)
make summary              # only the summary tables
make figures              # all figures (PIConGPU + microbench); a family
                           #   whose data is absent is skipped
make figures-picongpu     # only the PIConGPU figures
make figures-microbench   # only the microbenchmark figures
make figures-configs      # the per-machine config comparison figures
make figures-commits      # the per-machine commit comparison figures
make figures/foil_lct.pdf # one figure by file name (also:
make figures/sweeps-hal.pdf # figures/sweeps-<machine>.pdf,
make figures/sweeps-shared-hal.pdf # figures/sweeps-shared-<machine>.pdf,
make figures/configs-hal.pdf # figures/configs-<machine>.pdf,
make figures/commits-hal.pdf # figures/commits-<machine>.pdf
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
 | `group_stats` | per (machine, `dep_commit`, `config`, setup, algorithm, grid, delays): the runtime's count, mean, std, min, p25, p50, p75, max |
 | `baselines`   | the zero-delay (0, 0) runtime IQR per (machine, `dep_commit`, `config`, setup, grid, algorithm) group (sweep machines)  |
 | `absorption`  | the per-arm absorbed-delay slack, one row per (machine, `dep_commit`, `config`, group, arm): the plateau deficit `d` (s) and the per-call `c = d/N` (µs), from the raw runs (gauge-invariant) |
 | `fits`        | the performance-model fits, one row per group (+ the `fits/cov/<machine>/<dep_commit>/<config>/<setup>/<algorithm>/<grid>/` subgroups) |
 | `shared_fits` | the combined (shared-parameter) fits, one row per (machine, `dep_commit`, `config`, scenario, algorithm) (+ the `shared_fit_cov/<machine>/<dep_commit>/<config>/<setup>/<grid>/` subgroups) |
 | `foil`        | the FoilLCT bar chart's distributions over the zero-delay runs, per hardware                      |
 | `foil_pvalue` | the Kruskal p-values behind the FoilLCT chart, per hardware                                        |
| `khi`         | the KelvinHelmholtz violin chart's statistics, per (hardware, estimated memory)                   |

The file's top-level attributes record the provenance: `created_utc`,
`git_commit`, `sources` (the source log directories), `sweep_machines`,
`machine_titles` (machine label -> hardware title), `algorithm_order`
(the `algorithms` list of `config.json`, the row order of all figures),
`commit_order` (the `commits` names of `config.json`, the order of
`dep_commit` in the runs and in the `commits-` figure labels),
`config_order` (a JSON object mapping each algorithm to its config
names, the order of `config` in the runs and in the `configs-` figure
bars), and the exclusion of the archived-but-excluded legacy runs
(*Where the data lives*), `excluded_sources` and `excluded_runs`.

**The `runs` table** (one row per parsed run, sweep machines' logs plus
the frozen legacy table):

| columns                                                                                     | meaning                                                                                  |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `machine`, `hardware`                                                                        | the sweep machine's label (empty for the legacy runs stored without one) and the short paper-figure hardware name |
| `dep_commit`, `config`                                                                       | the two benchmark dimensions: the logical commit name of the run (metadata `run.commit`, the `commits` entry the binary was built from; empty for pre-Phase-2 logs) and the allocator config name (metadata `run.config`) |
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
   its bootstrap sleeve, the extrapolation to the zero-delay baseline
   (A_malloc = A_free = 0), and the scenario's combined fit as a heavy
   line.
 - `figures/sweeps-shared-<machine>.pdf` — the per-machine forest of the
   combined (shared-parameter) fit: one row per (scenario, algorithm), the
    individual W / N_malloc / N_free (dots with error bars) against the
    values fitted once across the algorithms, and the individual vs
    combined A_malloc / A_free, each A_* annotated with its slack ratio
    f = A/T0 (the absorbed delay over the zero-delay runtime).
 - `figures/configs-<machine>.pdf` — the per-machine allocator-config
   comparison: one row per scenario, one column per algorithm that carries
   at least two configs; the zero-delay p50 runtime per config as a bar
   (x = config name, in the file's `config_order`), the default config's
   hatched bar as the reference, per-row shared y-axes. Skipped when no
   (machine, commit) has an algorithm with two or more configs.
 - `figures/commits-<machine>.pdf` — the per-machine dependency-commit
   comparison: one row per (scenario, algorithm); the baseline column marks
   each commit's zero-delay p50 (ordinal x, log y) annotated with the
   commit's PIConGPU short hex, and the sweep column draws one line per
   commit's malloc-delay sweep with its fitted model curve + sleeve.
   Skipped when the (machine, config) data has only one commit. Both
   comparison figures consume the baselines only, so they are the primary
   output of a baseline-only run series.
 - `figures/foil_lct.pdf` — the FoilLCT bar chart of the zero-delay runs:
   one bar per allocator (in the file's `algorithm_order`), median with
   IQR error bar; the Kruskal significance is in the `foil_pvalue` table.
- `figures/kelvin_helmholtz.pdf` — the KelvinHelmholtz violin chart of the
  zero-delay runs relative to the ScatterAlloc reference runtime, one
  violin per allocator, one column per estimated particle memory.
- `figures/fade-models.pdf` — the candidate fade-term comparison that
  selects the model's fade shape: the candidate fade shapes g(u), the
  per-group shape effect (delta-SSR) against a reference fit of the base
  shape, and the aggregate ranking that chooses the exponential (see
  *The method*, "The performance model").
- `figures/native-cost.pdf` — the absorbed-slack vs native-cost comparison:
  one panel per operation, each fitted group plotted as (native per-call
  cost `c_a`, absorbed slack `A/N`) in µs per call, with the `A/N = c_a`
  reference line (from the `fits` table and the frozen microbenchmark cost;
  skipped with a note when no matching cost exists).
- `figures/microbench-allocation.pdf`, `microbench-allocation-mixed.pdf`,
  `microbench-allocation-scaling.pdf` — the microbenchmark's native
  per-call allocation costs (by allocation size, by size range, and by
  thread count at each fixed size), one panel per allocator/operation
  (from the frozen tables in `output/results.h5`).
- `figures/microbench-legend.pdf` — the allocator legend for the
  microbenchmark figures (the colour/marker each allocator uses).
- `figures/microbench-utilisation.pdf`, `microbench-graph.pdf` — the
  microbenchmark diagnostic figures (the per-allocator utilisation over the
  allocation sizes, and the allocation-graph figure; from the suite's raw
  CSVs under `microbenchmarks/data/`).

Notes:

- The fits are bounded `scipy.optimize.curve_fit` of the 1-D model
  `T(s) = W + N*s + A*exp(-s/s0)` (sweep on one delay) or the two-operation
  model `T(m, f) = W + N_m*m + N_f*f + A_m*exp(-m/m0) + A_f*exp(-f/f0)`
  (combination sweep); the full model, the candidate fade family, the
  bounds, and the diagonalization caveats of the bootstrap are documented
  in `analysis/performance_model.py`.
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

## Limitations

The reported quantities and their scope are defined in *The method*; this
section records the known limitations of the data and of the reporting.

- **The small-delay anomaly on the hal KelvinHelmholtz runs.** At imposed
  delays below roughly 10 ms, the KelvinHelmholtz runs on the `hal` machine
  (all three grids, both allocators) rise *above* the nominal
  baseline-plus-stall line, by up to ~4 % of the total runtime (a
  small-delay amplification hump, present in essentially every run of a
  delay cell). Its mechanism is not yet understood. Two consequences: those
  points are not described by the model's absorption term, and
  delay-sweep statements about the KHI allocation behaviour should be made
  from the large-delay asymptote rather than the small-delay points.
- **Arms without delay absorption.** A few arms show the opposite of
  absorption — their large-delay line extrapolates *above* the baseline
  (the free arms of rosi FoilLCT and rosi KHI 128^3, and the hal KHI 128^3
  ScatterAlloc arms) — and one hal FoilLCT free arm is non-monotonic. These
  arms are excluded from the absorption summary and flagged in the results
  file.
- **Slow-fade rosi KelvinHelmholtz arms.** On `rosi` the absorption fades
  over tens of milliseconds, beyond the measured delay range; there the
  plateau deficit is an extrapolation with a larger error.
- **The allocator comparison is conditional.** The relative per-call slack
  `Δc = Δ(d/N)` between two allocators is well defined and reported, but it
  equals a *relative native cost* only if the pipeline's hiding capacity is
  the same for both allocators — an assumption the data do not measure.
  Under it, the ranking differs between the two larger KHI grids; the data
  alone do not settle the ranking.
- **An absolute per-call allocation cost is not measured by the sweep.**
  The per-call absorbed slack `c = d/N` is the pipeline's hiding capacity,
  not the native cost; the native per-call cost comes from the independent
  microbenchmark (`c_a`, and the `A = N*c_a` fit — *The method*), which is
  currently measured on the A100 only.
- **The raw data are not in this repository.** The benchmark data lives on
  the HPC machines and in the released archive; *Where the data lives*
  describes what a fresh checkout can and cannot produce.

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
   `MACHINE`/`PROFILE`/`PARAM_DIR`/`REPEATS`/`REP`/`PHASE`/`COMMIT` its input
   parameters, the run logs, `results.h5` and the figures its outputs.
 - **The runs as provenance** (the Process Run profile): one `CreateAction`
   per grid-run log, reading the log's self-describing metadata line —
   instrument the binary used (sha256, build facts, the PIConGPU/mallocMC
   pins), object the `config.json` and the rendered parameter/profile files,
   `environment` the imposed delay and the slurm job, agent the run's user,
   result the log. Runs are append-only, so every vintage is described: a log
   whose identity's run stamp does not list it is annotated as superseded, on
   the same keys the analysis uses.
- **The analysis as actions**: one `CreateAction` for `results.h5` (from
  the run logs and the frozen legacy table, all vintages) and one per
  figure.
- **The operator as agent**: the `agent` of every run's `CreateAction`
  is the user the run happened as, resolved through the `people` table
  of `config.json` (login to name and ORCID iD); a run whose own metadata
  carries the ORCID uses that. The harness never manages ORCID: export
  `ORCID=…` in the environment under which you launch runs (module,
  profile, or shell) to have the persistent identifier recorded in the
  log. Where no ORCID can be resolved, the identity degrades to the login
  only, and `make rocrate` prints a `note:` naming the affected logins —
  the starting point for growing the table. The root and the workflow
  also list the configured people (with the ORCID iD as the entity's
  identifier, so they stay identifyable beyond any cluster's login
  scheme) as `creator`, alongside the institution.

The declared conformance is RO-Crate 1.3 plus the two profile statements
on the root data entity. The crate is deliberately *not* packaged for
WorkflowHub ingestion: the pipeline is a Makefile, not one of the workflow
languages WorkflowHub supports. The built-in validation runs with the
target; the check additionally loads the crate with the official `rocrate`
package when it is installed (`pip install rocrate`) — reported as a
warning only, since its newest release supports crate versions up to 1.2.

**Packaging (`make crate-zip`).** The crate is also packaged as a
portable, self-describing archive following the RO-Crate packaging
convention: `ro-crate.crate.zip` (git-ignored, removed by `make clean`)
carries the metadata — staged under the conventional
`ro-crate-metadata.json` name — and every data file the crate references
(the run logs, `results.h5`, the frozen legacy table, the figures, the
harness), so unzipping the archive yields a valid crate root. The target
proves this: it unzips the freshly written archive into a scratch
directory and re-runs the checks on the result before declaring success.
The run logs are mostly repetitive text and compress heavily (measured
~15-20x), so the full archive of a long run series stays on the order of
tens of MB; entry timestamps are fixed, so a given crate builds a
byte-identical archive.

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
  data lives*. Which combinations are already stamped (and therefore
  which a `make runs` would still run): `make sweep-status
  MACHINE=<machine>`.
- The FoilLCT / KelvinHelmholtz figures are empty or missing hardware:
  the frozen legacy table was not built (`make legacy-results`, *Where the
  data lives*); a sweep machine contributes only if it ran its (0, 0)
  baseline.
- The microbenchmark figures are missing: the microbenchmark CSVs were not
  frozen (`make microbench-results`, *Where the data lives*) — the
  allocation-cost figures come from the frozen table (so they also need
  `make results` after the freeze) and the diagnostic figures from the raw
  CSVs under `microbenchmarks/data/`.
- The analysis picks up the wrong Python package versions: use the locked
  environment (`make env`) or export `PYTHONNOUSERSITE=1` (*Reproducing
  the analysis*).
- A configured machine has no output directory yet (e.g. `rosi-a100`
  before its first run): the figures and the RO-Crate skip it and print a
  note.
- `make clean` removes `build/`, `figures/`, `output/results.h5`,
  `ro-crate-metadata.json` and `ro-crate.crate.zip`; `make distclean`
  removes `src/` as well; neither touches the run stamps — only
  `make clean-runs` removes finished runs (*Running the benchmark*).
