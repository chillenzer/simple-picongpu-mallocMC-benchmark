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
  (see `ALGORITHMS` in `run_all.sh` and `param/<Algorithm>/mallocMC.param`)
- **Delay combinations** (nanoseconds, see `COMBINATIONS` in `run_all.sh`):
  a `(0, 0)` baseline, the malloc-delay sweep with free delay 0 and the
  free-delay sweep with malloc delay 0 over the 12 log-spaced values
  `100`…`100000000` (1/4-decade steps, skipping the intermediate steps
  between `1e5` and `1e6`), plus a 3x3 joint grid over `10000`, `1000000`,
  `10000000` (34 runs per example and algorithm)
- **Examples**: `KelvinHelmholtz` (3D, grid sizes 128^3, 256x128x128,
  256x128x256, 1500 steps) and `FoilLCT` (2D, 256x1280 cells, 2000 steps)
  (see `EXAMPLES` in `run_all.sh` and `flags/*.flags`)
- **One build per (example, algorithm)**: the creation policy is compiled
  in via `param/<Algorithm>/mallocMC.param`; the delays are not a
  compile-time option, mallocMC reads them at run time from the
  `MALLOCMC_MALLOC_DELAY` and `MALLOCMC_FREE_DELAY` environment variables
  (set by `run_all.sh` / `run_folder.sh` for every run). Changing the sweep
  therefore never requires recompiling.
- **Layout**: `build/<Example>/<Algorithm>/` holds one build; its flag
  lines are run once per combination:
  `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`

`setup.sh` pins the dependency versions:

| dependency | source | pinned to |
|------------|--------|-----------|
| PIConGPU | ComputationalRadiationPhysics/picongpu | `6e7d58bb` |
| mallocMC | chillenzer/mallocMC, `add-delay` branch (`BOOST_LANG_*` guards compatible with PIConGPU's vendored alpaka 2.0; run-time malloc/free delays via the `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables, injected as a busy-wait on the device global timer at the top of `DeviceAllocator::malloc` / `DeviceAllocator::free`) | `2eb8a18e` |

## Repository layout

- `setup.sh` — clones the pinned PIConGPU and mallocMC, prepares one input
  directory per (example, algorithm) (`pic-create` + parameter overlay) and
  builds it (`pic-build`). The overlay copies `param/<Algorithm>/*.param`
  (the algorithm's `mallocMC.param`, i.e. the creation policy),
  `param/<Example>/*.param` and any per-(example, algorithm) overrides from
  `param/<Example>/<Algorithm>/`.
- `run_all.sh` — for every example, every algorithm and every `(malloc
  delay, free delay)` combination in `COMBINATIONS`, runs the example's
  flag lines from the matching build with both environment variables set.
- `run_folder.sh` — runs one already-built example folder, once per flag
  line; takes optional fourth (malloc delay, default `0`) and fifth (free
  delay, default `0`) arguments in nanoseconds, passed via the
  `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY` environment variables.
- `profiles/` — HPC environment profiles (module/spack setup, `PIC_BACKEND`,
  `PICSRC`). One per machine.
- `flags/` — one `picongpu` command line per benchmark run, per example.
- `param/` — parameter files overlaying the example defaults: one
  `mallocMC.param` per algorithm (`param/<Algorithm>/`, defining the
  creation policy), example-specific files (`param/FoilLCT/`), and optional
  per-(example, algorithm) overrides (`param/<Example>/<Algorithm>/`).
- `analysis/run_logs.py` — shared parsing of the benchmark run logs: one
  record per `bin/picongpu` run (example, algorithm, grid, imposed delays,
  runtime) from the raw `set -x` trace of any of the historical log layouts;
  the basis of the analysis scripts below.
- `analysis/analyse_sleeptimes.py` — reads the raw `run_all.sh` logs directly
  (no pre-filtering). For a sweep that varies only one delay it fits the
  (example, algorithm, grid) group to the Amdahl model
  `T(s) = W + N*s + A*s0/(s+s0)` (large-s Amdahl line `W + N*s` plus a
  small-s native-cost correction `A`); for a (malloc, free) combination
  sweep it fits the two-operation model
  `T(m, f) = W + N_m*m + N_f*f + A_m*m0/(m+m0) + A_f*f0/(f+f0)` and reports
  the runtime fractions spent in allocations and in frees separately
  (`f_malloc = A_m/T0`, `f_free = A_f/T0`, `T0 = W + A_m + A_f`). Both fits
  use a bounded `scipy.optimize.curve_fit` (bounds `W, N, A >= 0` keep each
  `f` in `[0, 1)`), and print the fractions, the call counts `N`, and
  `W`/`A`, each with a standard error (the full model is documented in the
  script docstring). It plots runtime against the delay per example,
  algorithm, grid and held delay with IQR error bars and the fitted curve
  overlaid, in one figure per cluster (see `CLUSTERS` in the script) titled
  by the hardware the runs were made on: one row per algorithm present in
  the data (`ALGORITHM_ORDER`), two columns (malloc | free delay sweep).
  Each run is labeled with a `configuration` column (`compile-time`
  per-variant sweep vs `run-time` env-var sweep), so both log layouts can be
  analyzed side by side (pre-rename logs that used `MALLOCMC_SLEEP_TIME`
  parse into the malloc delay). Set the module variable `CONFIGURATION` to
  `"run-time"` or `"compile-time"` to compute the statistics, plot and fit
  only from runs of that configuration (`None` uses all; the printed parsed
  results stay complete).
- `analysis/produce_figures.py` — produces the paper plots of the original
  three-algorithm comparison (see note below).
- `build/` — created by `setup.sh`; one CMake project per (example,
  algorithm).

## Usage

Everything runs on the HPC machine with the matching profile; the scripts
must be invoked from the repository root.

1. Build all examples and algorithms (slow the first time: one full
   PIConGPU build per example and algorithm):

   ```
   bash setup.sh profiles/hal.sh param
   ```

   `setup.sh` also patches `PICSRC=` in the given profile in place.

   `setup.sh` is incremental and can be re-run at any time; it reuses
   `src/` and `build/` and only repeats the parts that are out of date:

   - `src/` is re-cloned only if the directory is not a git checkout at the
     pinned commit; a changed pin triggers `git fetch && git checkout`.
   - An input directory (`build/<Ex>/<Algo>/`) is regenerated when the
     PIConGPU pin or the `param/<Algo>*/`, `param/<Ex>*` files change
     (`.input-stamp` records this).
   - A build (`pic-build`) is skipped when the input, the build `FLAGS`,
     the profile and the toolchain versions (gcc/cmake/nvcc) are unchanged
     (`.build-stamp` records this); otherwise `pic-build` runs
     incrementally.

   To force a clean state, delete `src/` and `build/` (or just a single
   `build/<Ex>/<Algo>/` folder, or only a `.build-stamp` to re-run
   `pic-build` for one example and algorithm).

2. Run the benchmarks:

   ```
   bash run_all.sh profiles/hal.sh flags
   ```

     For every example, every algorithm and every `(malloc delay, free
     delay)` combination in `COMBINATIONS`, each flag line is run as
     `MALLOCMC_MALLOC_DELAY=<M> MALLOCMC_FREE_DELAY=<F> bin/picongpu ...`
     from `build/<Ex>/<Algo>/`. Stdout contains one
     `Running example: <Ex> / Using allocator: <Algo>, malloc delay: <M> ns,
     free delay: <F> ns` section per run; each run's result is picongpu's
     `calculation ... simulation time:` line.

Single run (one built folder, one combination):

```
bash run_folder.sh build/FoilLCT/FlatterScatter flags/FoilLCT.flags profiles/hal.sh 10000 0
```

## Changing what is benchmarked

- **Delay combinations**: edit `COMBINATIONS` (and the `DELAY_SWEEP` /
  `JOINT` that build it) in `run_all.sh`. The delays are applied at run time
  via `MALLOCMC_MALLOC_DELAY` / `MALLOCMC_FREE_DELAY`, so changing the sweep
  requires no rebuild.
- **Grids / steps / other picongpu flags**: edit `flags/<Example>.flags`
  (one command line per run).
- **Allocator configuration**: `param/<Algorithm>/mallocMC.param` defines
  the `DeviceHeap`; the creation policy and, for FlatterScatter, the heap
  config (`DefaultHeapConfig<block, page, waste>`) live there. Editing a
  file invalidates exactly the affected (example, algorithm) inputs.
- **PIConGPU / mallocMC version**: the `*_URL` / `*_HASH` variables at the
  top of `setup.sh`.
- **New example**: add it to `EXAMPLES` in both scripts, provide
  `flags/<Example>.flags`, and optionally `param/<Example>/*.param`.
- **New algorithm**: add it to `ALGORITHMS` in both scripts and provide
  `param/<Algorithm>/mallocMC.param`.

## Analysis

All of the analysis reads the raw run logs directly (no pre-filtering); the
shared parsing lives in `analysis/run_logs.py`.

`analysis/analyse_sleeptimes.py` reads the raw
`output/<cluster>-sleeptimes/run_*` logs (every historical layout) and plots
the runtime against the delay (log-log, median with IQR error
bars) per example, algorithm, grid and held delay, with the fitted Amdahl
curve overlaid, in one figure per cluster (see `CLUSTERS` in the script)
titled by the hardware the runs were made on, with one row per algorithm
present in the data (a single-algorithm dataset yields the plain
two-axis figure). It also fits every (example, algorithm, grid) sweep to
the model and prints the extracted fraction of runtime spent in
allocations / frees:

```
python3 analysis/analyse_sleeptimes.py
```

`analysis/produce_figures.py` produces the paper plots of the original
three-algorithm comparison from the no-delay runs: the per-cluster
`output/<cluster>/` run logs (one full `run_all.sh` repetition per file) plus
the (0, 0) baseline runs of the delay-combination sweeps (the
`hal-sleeptimes` and `rosi-sleeptimes` dirs; the nanosleep runs are
excluded). It writes the FoilLCT bar chart (`figures/foil_lct.pdf`), the
KelvinHelmholtz violin chart of the runtime relative to the ScatterAlloc
baseline (`figures/kelvin_helmholtz.pdf`), and prints the per-grid timing
statistics. The delay-combination sweeps themselves are analyzed by
`analyse_sleeptimes.py`, which saves one figure per cluster to
`figures/<cluster>.pdf`.

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
