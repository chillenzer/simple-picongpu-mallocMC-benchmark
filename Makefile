# Makefile
#
# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT
#
# Four purposes, one Makefile:
#
# 1. The benchmark build harness (rewritten from setup.sh): clones the
#    pinned PIConGPU and mallocMC into src/, prepares one input directory
#    per (example, algorithm) (pic-create template + parameter overlay) and
#    builds each one (pic-build). What to build (examples, algorithms,
#    dependency pins, build flags) is read from config.json through
#    config.py and validated up front:
#
#      make build PROFILE=profiles/hal.sh PARAM_DIR=param   # clone + inputs + builds
#      make check                                           # resolved configuration
#
# 2. The benchmark run orchestrator (rewritten from run_all.sh): sweeps one
#    full repetition of one sweep phase (examples x algorithms x the phase's
#    (malloc, free) delay combinations, the `delays` section of config.json)
#    per invocation, one (combination, repetition) at a time:
#
#      make runs MACHINE=hal                          # full (arms) sweep
#      make runs MACHINE=hal PHASE=initial            # fast first scan
#      make runs MACHINE=rosi REPEATS=3 REP=2         # one repetition (= one slurm job)
#      make full MACHINE=hal                          # build, then run
#      make clean-runs [MACHINE=hal]                  # forget finished runs
#      make sweep-status MACHINE=hal [PHASE=...]      # which runs are stamped
#
#    PHASE selects the delay combinations: `initial` (the baseline plus the
#    delays.arms.initial subset) and `arms` (the default: the full arm ladder
#    plus the joint grid when one is configured). The phases are incremental:
#    an `arms` sweep after an `initial` one re-runs only the new combinations.
#    Each finished (example, algorithm, combination, repetition) writes a
#    stamp under run-stamps/, which makes an interrupted run series (one
#    `make runs` invocation) resumable.
#    The stamps depend only on the example's flags file, so rebuilding the
#    binaries never invalidates finished runs (and `make clean` /
#    `distclean` do not touch run-stamps/). Every run gets one
#    self-contained log per grid run (one line of the example's flags
#    file), run_<machine>_<Ex>_<Algo>_m<M>_f<F>_r<I>_<line-sha8>_<time>.txt
#    in the machine's output directory: a human one-liner `# run:` line,
#    the self-describing `# metadata:` JSON line emitted by logmeta.py
#    (machine, commit, the pinned dependency hashes, the slurm job when
#    running under slurm, the sha256 of the binary used, the build facts
#    of the binary's tree, the host hardware and the user the run happens
#    as), and that grid run's full output. Runs are append-only:
#    re-running a (combination, repetition)
#    writes a new vintage of the logs next to the older ones, nothing is
#    ever removed, and the stamp content lists the log paths of the
#    run's current vintage, which the analysis uses to flag the older
#    vintages. The session logs of the log_*.sh launchers go to
#    <output dir>/sessions/ (a free-text backup, never parsed).
#
# 3. The analysis driver: builds the benchmark numbers (output/results.h5)
#    and the figures (figures/) from the run logs. The runs come from two
#    sources: the sweep machines' output directories (config machines
#    table, the new-format logs with the `# metadata:` JSON line) and the
#    frozen legacy table legacy/legacy_results.h5, built once with
#    `make legacy-results` from the legacy/ tree (see legacy/README.md).
#    The numbers are rebuilt from both on every invocation: make
#    deliberately does not list files of the output/ or legacy/ data as
#    prerequisites, because their names are machine-specific (run names,
#    timestamps) and not safe to parse as make dependencies. Every figure
#    is independent: build any one of them by its file name, e.g.
#    `make figures/foil_lct.pdf` or `make figures/sweeps-hal.pdf`, or all
#    of them with `make` (the default goal, which also prints the summary
#    tables). Rebuild only the numbers with `make results`.
#
# 4. The RO-Crate generator: describes the benchmark and its data as one
#    research object (ro-crate-metadata.json at the repository root,
#    git-ignored): the harness as a workflow, the benchmark runs as
#    provenance (one CreateAction per grid-run log, superseded vintages
#    included) and the analysis as actions. The metadata is a derived
#    artifact, rebuilt from the run logs, stamps, results.h5 and figures
#    on every invocation, and validated in the same target:
#
#      make rocrate
#
# `make clean` removes everything generated (build/, figures/,
# output/results.h5 and ro-crate-metadata.json); `make distclean` removes
# src/ as well. Neither touches the run stamps (run-stamps/), which are
# the record of finished runs; `make clean-runs` removes them.
#
# From the repository root, with the machine's environment loaded (in
# practice via the log_setup_<machine>.sh launchers) for the harness.
#
# The harness targets are incremental: a target re-runs only when its
# prerequisites change (the dependency pins, the parameter overlay files,
# the build flags, the profile content, the toolchain versions). Like
# setup.sh, `make build` without -j runs the targets serially;
# `make build -j` builds the independent (example, algorithm) pairs in
# parallel. The run targets are serial by construction (one GPU): `runs`
# fans out to one sub-make per (combination, repetition) stamp in sweep
# order, and a stamp that is already up to date is skipped.

.DEFAULT_GOAL := all

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -e -c
.DELETE_ON_ERROR:

PY      := python3
RESULTS := output/results.h5
FIGDIR  := figures

CONFIG := config.json

# --- harness configuration: validated up front, read through config.py ----

ifneq ($(strip $(shell $(PY) config.py check >/dev/null 2>&1; echo $$?)),0)
$(error config.json failed validation; run `python3 config.py check` for details)
endif

EXAMPLES   := $(shell $(PY) config.py list examples)
ALGORITHMS := $(shell $(PY) config.py list algorithms)

PICONGPU_PATH  := $(shell $(PY) config.py get dependencies.picongpu.path)
PICONGPU_URL   := $(shell $(PY) config.py get dependencies.picongpu.url)
PICONGPU_HASH  := $(shell $(PY) config.py get dependencies.picongpu.hash)
PICONGPU_ABS   := $(CURDIR)/$(PICONGPU_PATH)
PICONGPU_SHORT := $(shell printf '%.8s' $(PICONGPU_HASH))
# add-delay branch: run-time malloc/free delays for every creation policy
# (FlatterScatter, Scatter, Gallatin) via the MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY environment variables (read into the device allocator
# by mallocMC::Allocator::alloc), the delays as busy-waits on the device
# global timer. We pin our fork, not picongpu's own mallocMC copy.
MALLOCMC_PATH  := $(shell $(PY) config.py get dependencies.mallocmc.path)
MALLOCMC_URL   := $(shell $(PY) config.py get dependencies.mallocmc.url)
MALLOCMC_HASH  := $(shell $(PY) config.py get dependencies.mallocmc.hash)
MALLOCMC_ABS   := $(CURDIR)/$(MALLOCMC_PATH)
MALLOCMC_SHORT := $(shell printf '%.8s' $(MALLOCMC_HASH))

# Nasty little bug here: GCC has a constexpr std::source_location but nvcc
# does not. So Boost gets confused and tries to use std::source_location
# constexpr.
CXX_FLAGS         := $(shell $(PY) config.py get build.cxx_flags)
EXTRA_CMAKE_FLAGS := $(shell $(PY) config.py list build.extra_cmake_flags)
FLAGS := -DCMAKE_CXX_FLAGS="$(CXX_FLAGS)" -DCMAKE_CUDA_FLAGS="$(CXX_FLAGS)" $(EXTRA_CMAKE_FLAGS)

# csh exports PROFILE (the path to the user's csh profile); an inherited
# value must not silently stand in for the machine profile we source.
# (PARAM_DIR gets the same treatment; it is not normally exported, but
# guarding it costs nothing.)
ifeq ($(origin PROFILE),environment)
PROFILE =
endif
ifeq ($(origin PARAM_DIR),environment)
PARAM_DIR =
endif
PROFILE   ?=
PARAM_DIR ?= param
PARAM_DIR := $(patsubst %/,%,$(strip $(PARAM_DIR)))

# Runs are addressed by the machines table of config.json; REPEATS is how
# many full-sweep repetitions the series is made of, and REP restricts an
# invocation to one repetition (the slurm case: one job per repetition).
# Like PROFILE, MACHINE is an invocation value rather than a configuration
# key, so a stray exported value must not stand in for it.
ifeq ($(origin MACHINE),environment)
MACHINE =
endif
MACHINE ?=
REPEATS ?= 1
REP ?=

# PHASE selects the sweep's delay combinations from the `delays` section of
# config.json. `initial` is the fast first scan (the baseline plus the
# delays.arms.initial subset); `arms` (the default) is the full arm ladder
# plus the joint grid when one is configured. The phases are incremental:
# the run stamps of an `initial` sweep remain up to date for an `arms` sweep,
# so a series can start minimal and be extended in place. Like MACHINE,
# PHASE is an invocation value rather than a configuration key, so a stray
# exported value must not stand in for it.
ifeq ($(origin PHASE),environment)
PHASE =
endif
PHASE   ?= arms
ifneq ($(filter-out initial arms,$(PHASE)),)
$(error specify PHASE=initial or arms (got '$(PHASE)'))
endif

# The (malloc, free) delay combinations of the phase, one
# "<malloc>_<free>" token per line, derived from config.json by config.py.
COMBOS := $(shell $(PY) config.py list run-matrix $(PHASE))

# Only `runs` and `full` need a machine; `clean-runs` also accepts an empty
# one (it then removes the stamps of every machine).
ifneq ($(filter runs full sweep-status,$(MAKECMDGOALS)),)
ifeq ($(strip $(MACHINE)),)
$(error specify MACHINE=<machine> from the config machines table, e.g. make runs MACHINE=hal)
endif
endif

# Stamps: written by the phony drivers below, consumed as prerequisites.
PICONGPU_STAMP    := $(PICONGPU_ABS)/.dep-stamp
MALLOCMC_STAMP    := $(MALLOCMC_ABS)/.dep-stamp
PROFILE_ENV_STAMP := build/.profile-env
TOOLCHAIN_STAMP   := build/.toolchain
FLAGS_STAMP       := build/.build-flags

# Only the benchmark goal (`build`) and the explicit per-pair targets under
# build/ run on the HPC machine and need its profile; the default goal and
# all analysis goals, check and clean do not.
ifneq ($(filter build%,$(MAKECMDGOALS)),)
ifeq ($(PROFILE),)
$(error specify PROFILE=<machine profile>, e.g. make build PROFILE=profiles/hal.sh)
endif
endif

.PHONY: all build check clean distclean results summary figures \
	figures-sweeps figures-shared picongpu-src mallocmc-src env-check env \
	runs full clean-runs legacy-results legacy-verify rocrate sweep-status

# --- per (example, algorithm) harness targets ------------------------------

PAIRS      := $(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(e)/$(a)))
BUILD_DIRS := $(addprefix build/,$(PAIRS))
BINARIES   := $(addsuffix /bin/picongpu,$(BUILD_DIRS))

# One (example, algorithm) pair: $1 = example, $2 = algorithm, $3 = pair.
#
# The input is regenerated when the PIConGPU pin (the pic-create template),
# the profile (PICSRC feeds pic-create's tool path), or the parameter overlay
# changes. The overlay prerequisites are the concrete *.param files plus the
# three param directories (their mtime changes when a file is added or
# removed, so a newly added parameter file is picked up).
#
# The build is skipped while the input, the mallocMC pin (the headers are
# compiled into the binary), the profile content, the build flags and the
# toolchain versions are all unchanged; targeting the binary itself also
# covers a manually deleted bin/picongpu.
define pair_rules
build/$(3)/.input-stamp: $(PICONGPU_STAMP) $(PROFILE_ENV_STAMP) \
	$(wildcard $(PARAM_DIR)/$(2)/*.param) \
	$(wildcard $(PARAM_DIR)/$(1)/*.param) \
	$(wildcard $(PARAM_DIR)/$(1)/$(2)/*.param) \
	$(if $(wildcard $(PARAM_DIR)/$(2)/.),$(PARAM_DIR)/$(2)) \
	$(if $(wildcard $(PARAM_DIR)/$(1)/.),$(PARAM_DIR)/$(1)) \
	$(if $(wildcard $(PARAM_DIR)/$(1)/$(2)/.),$(PARAM_DIR)/$(1)/$(2))
	@echo "Preparing input build/$(3) ..."
	@rm -rf build/$(3)
	@mkdir -p build/$(3)
	@source "$(PROFILE)"
	pic-create "$(PICONGPU_ABS)/share/picongpu/examples/$(1)" "build/$(3)"
	# The algorithm's mallocMC.param (the creation policy), the example's
	# parameters, and any per-(example, algorithm) overrides; later levels
	# win on name clashes.
	find $(PARAM_DIR)/* -type f -wholename '$(PARAM_DIR)/$(2)/*.param' -exec cp -v {} "build/$(3)/include/picongpu/param/" ';'
	find $(PARAM_DIR)/* -type f -wholename '$(PARAM_DIR)/$(1)/*.param' -exec cp -v {} "build/$(3)/include/picongpu/param/" ';'
	find $(PARAM_DIR)/* -type f -wholename '$(PARAM_DIR)/$(1)/$(2)/*.param' -exec cp -v {} "build/$(3)/include/picongpu/param/" ';'
	@printf 'input ready for %s/%s\n' "$(1)" "$(2)" >"build/$(3)/.input-stamp"
	@echo "Prepared input build/$(3)."

build/$(3)/bin/picongpu: build/$(3)/.input-stamp $(MALLOCMC_STAMP) $(PROFILE_ENV_STAMP) $(TOOLCHAIN_STAMP) $(FLAGS_STAMP)
	@source "$(PROFILE)"
	cd "build/$(3)"
	export CMAKE_PREFIX_PATH="$(MALLOCMC_ABS):$${CMAKE_PREFIX_PATH:-}"
	pic-build -c "$(FLAGS)"
endef

$(foreach p,$(PAIRS),$(eval $(call pair_rules,$(firstword $(subst /, ,$(p))),$(lastword $(subst /, ,$(p))),$(p))))

# --- run stamps: one per (combination, repetition) --------------------------
#
# A run is one (example, algorithm) build through one (malloc delay, free
# delay) combination: the example's whole flags file once, with the
# combination's delays in the environment. The sweep order is example, then
# algorithm, then repetition, and within a repetition the combination order
# of `config.py list run-matrix`, so every repetition is a full sweep and
# when REPEATS is finished, every build has finished it.
#
# The stamp is the record that the run happened (its content is the
# run's log paths, one per line, i.e. the paths of its current vintage),
# and it is what makes an interrupted series resumable. Its only
# prerequisite is the example's flags file: editing flags/<Ex>.flags
# invalidates exactly that example's stamped runs, and nothing that
# `make build` touches (pins, profile, toolchain, a rebuild) ever
# invalidates a finished run. The recipe itself refuses to start while the
# build's binary is missing.
#
# The repetitions of this invocation (all of 1..REPEATS, or only REP in the
# slurm case: one job per repetition) are expanded at parse time into one
# explicit target per (combination, repetition).
REPS := $(strip $(shell seq 1 '$(REPEATS)' 2>/dev/null))

# $1 = malloc delay (ns), $2 = free delay (ns), $3 = example, $4 = algorithm,
# $5 = repetition number. The recipe carries no shell of its own (the rule
# is generated by $(eval $(call ...)) and is therefore expanded twice), it
# just hands the parameters to run_stamp.sh, which does the rest.
define run_stamp_rules
run-stamps/$(MACHINE)/$(3)/$(4)/$(1)_$(2)/rep-$(5).stamp: flags/$(3).flags
	@bash run_stamp.sh "$(MACHINE)" "$(REPEATS)" "$(3)" "$(4)" "$(1)" "$(2)" "$(5)"
endef

$(foreach c,$(COMBOS),$(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(foreach r,$(REPS),$(eval $(call run_stamp_rules,$(firstword $(subst _, ,$(c))),$(lastword $(subst _, ,$(c))),$(e),$(a),$(r)))))))

# --- run targets -------------------------------------------------------------

ifeq ($(strip $(REP)),)
RUN_REPS := $(REPS)
else
RUN_REPS := $(REP)
endif
RUN_STAMPS := $(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(foreach i,$(RUN_REPS),$(addprefix run-stamps/$(MACHINE)/$(e)/$(a)/,$(addsuffix /rep-$(i).stamp,$(COMBOS))))))

# The fan-out: one sub-make per (combination, repetition) stamp, in sweep
# order, serially (one GPU; -j cannot parallelize a single recipe). A stamp
# that is up to date (present and not older than the flags file, so the
# freshness test make itself would apply) is skipped without a sub-make;
# everything else goes through make's normal freshness rules.
runs:
	@python3 config.py get "machines.$(MACHINE).output" >/dev/null || { echo "make runs: MACHINE=$(MACHINE) is not in the config machines table" >&2; exit 1; }
	@if [ -z "$(REPS)" ]; then
	  echo "make runs: REPEATS must be a positive integer (got '$(REPEATS)')" >&2
	  exit 1
	fi
	@if [ -n "$(REP)" ]; then
	  case "$$(printf '%s' "$(REP)" | tr -d '0-9')" in
	    "") : ;;
	    *) echo "make runs: REP must be a positive integer (got '$(REP)')" >&2; exit 1 ;;
	  esac
	  if [ "$(REP)" -lt 1 ] || [ "$(REP)" -gt "$(REPEATS)" ]; then
	    echo "make runs: REP=$(REP) is outside 1..$(REPEATS)" >&2
	    exit 1
	  fi
	fi
	@STAMPED=0
	@for stamp in $(RUN_STAMPS); do
	  EX=$$(printf '%s\n' "$$stamp" | cut -d/ -f3)
	  if [ -f "$$stamp" ] && [ ! "flags/$$EX.flags" -nt "$$stamp" ]; then
	    STAMPED=$$(($$STAMPED + 1))
	    continue
	  fi
	  $(MAKE) --no-print-directory "$$stamp"
	done
	@echo "runs [$(MACHINE)]: $${STAMPED} of $(words $(RUN_STAMPS)) runs already stamped."

# The one-command machine-side flow: build the harness, then run the whole
# series. The analysis (figures, summary) stays a plain `make`, which is
# where it belongs once the logs are on a laptop.
full:
	@$(MAKE) build PROFILE="$$(python3 config.py get machines.$(MACHINE).profile)" PARAM_DIR=param
	@$(MAKE) runs

# Remove the run stamps (of all machines, or of one): the next `make runs`
# then repeats those series, as a new vintage next to the existing logs.
# No log is ever removed by make.
clean-runs:
	@if [ -n "$(MACHINE)" ]; then
	  python3 config.py get "machines.$(MACHINE).output" >/dev/null || { echo "make clean-runs: MACHINE=$(MACHINE) is not in the config machines table" >&2; exit 1; }
	  rm -rf "run-stamps/$(MACHINE)"
	  echo "removed run-stamps/$(MACHINE)"
	else
	  rm -rf run-stamps
	  echo "removed run-stamps"
	fi

# Which runs of the machine and phase are already stamped, and which are
# still pending: one `pending <stamp>` line for each unstamped
# (combination, repetition), then an X of Y summary. The combination set is
# the phase's, the repetition set the REPEATS/REP one, exactly as for
# `runs`, so `make sweep-status MACHINE=hal PHASE=arms` shows what an arms
# sweep would still do on top of an initial sweep.
sweep-status:
	@STAMPED=0
	@for stamp in $(RUN_STAMPS); do
	  if [ -f "$$stamp" ]; then
	    STAMPED=$$(($$STAMPED + 1))
	  else
	    echo "pending $$stamp"
	  fi
	done
	@echo "sweep-status [$(MACHINE)] phase $(PHASE): $${STAMPED} of $(words $(RUN_STAMPS)) runs stamped (REPEATS=$(REPEATS)$(if $(REP), REP=$(REP)))"

# --- top-level targets -----------------------------------------------------

all: figures summary

build: $(BINARIES)
	@echo "=== binaries ==="
	@sha256sum $(BINARIES)

check:
	$(PY) config.py check
	@printf 'examples:   %s\n' "$(EXAMPLES)"
	@printf 'algorithms: %s\n' "$(ALGORITHMS)"
	@printf 'runs:       %s combinations per repetition (initial): %s\n' \
		"$$(python3 config.py list run-matrix initial | wc -l | tr -d ' ')" \
		"$$(python3 config.py list run-matrix initial | tr '\n' ' ' | sed 's/ *$$//')"
	@printf 'runs:       %s combinations per repetition (arms):    %s\n' \
		"$$(python3 config.py list run-matrix arms | wc -l | tr -d ' ')" \
		"$$(python3 config.py list run-matrix arms | tr '\n' ' ' | sed 's/ *$$//')"
	@printf 'picongpu:   %s @ %s\n' "$(PICONGPU_ABS)" "$(PICONGPU_SHORT)"
	@printf 'mallocmc:   %s @ %s\n' "$(MALLOCMC_ABS)" "$(MALLOCMC_SHORT)"
	# Each value comes through a double-quoted command substitution, so any
	# spaces, quotes or leading dashes in them stay one single shell word.
	@printf 'flags:      -DCMAKE_CXX_FLAGS=%s -DCMAKE_CUDA_FLAGS=%s %s\n' \
		"$$(python3 config.py get build.cxx_flags)" \
		"$$(python3 config.py get build.cxx_flags)" \
		"$$(python3 config.py list build.extra_cmake_flags | tr '\n' ' ' | sed 's/ *$$//')"

# --- analysis targets --------------------------------------------------------

# The analysis Python environment is pinned by two committed lock files
# (see the README, "Reproducing the analysis"):
#   conda-lock.yml       the conda-lock lock (the regeneration source;
#                        read by `conda-lock install`)
#   conda-linux-64.lock  the rendered per-platform *explicit* lock, a
#                        plain list of exact package URLs that
#                        `make env` installs from: any version of
#                        micromamba, mamba, or conda installs it without
#                        invoking a solver
# `make env` creates the environment with the first of these tools found
# on PATH (mamba is the default; it must be new enough to read the
# explicit lock, i.e. mamba 2.x -- mamba 1.x looks for a package named
# after the lock file; use ENV_TOOL=conda for such installs):
#   mamba       default
#   micromamba  standalone binary, no root needed
#               (https://micro.mamba.pm/api/micromamba/linux-64/latest)
#   conda       base conda, e.g. for older mamba installs
# (All three take the same `create -f <explicit lock>` form; because the
# lock is explicit, none of them invoke a solver.) Force a specific tool
# with `make env ENV_TOOL=<tool>`. Re-running is safe for micromamba and
# mamba (the environment is updated in place); conda reports the
# environment as existing (conda env remove first).
ENV_NAME   := mallocmc-bench
ENV_LOCK   := conda-linux-64.lock
MAMBA      := $(shell command -v mamba 2>/dev/null)
MICROMAMBA := $(shell command -v micromamba 2>/dev/null)
CONDA      := $(shell command -v conda 2>/dev/null)

ifeq ($(ENV_TOOL),)
ifeq ($(MAMBA),)
ifeq ($(MICROMAMBA),)
ENV_TOOL := conda
else
ENV_TOOL := micromamba
endif
else
ENV_TOOL := mamba
endif
endif

env:
ifeq ($(ENV_TOOL),micromamba)
ifdef MICROMAMBA
	$(MICROMAMBA) create -n $(ENV_NAME) -f $(ENV_LOCK) -y
else
	@echo "make env: micromamba not found on PATH (ENV_TOOL=micromamba);"
	@echo "           download it from https://micro.mamba.pm/api/micromamba/linux-64/latest"
	@echo "           or re-run with ENV_TOOL=conda / ENV_TOOL=mamba."
	@exit 1
endif
else ifeq ($(ENV_TOOL),conda)
ifdef CONDA
	$(CONDA) create -n $(ENV_NAME) -f $(ENV_LOCK) -y
else
	@echo "make env: conda not found on PATH (ENV_TOOL=conda); install it with"
	@echo "           Miniforge (https://github.com/conda-forge/miniforge), or"
	@echo "           re-run with ENV_TOOL=micromamba / ENV_TOOL=mamba."
	@exit 1
endif
else ifdef MAMBA
	$(MAMBA) create -n $(ENV_NAME) -f $(ENV_LOCK) -y
else
	@echo "make env: mamba not found on PATH (ENV_TOOL=mamba); install it with"
	@echo "           conda install -c conda-forge mamba, or re-run with"
	@echo "           ENV_TOOL=micromamba / ENV_TOOL=conda."
	@exit 1
endif

# Freeze the pre-redesign benchmark logs once into legacy/legacy_results.h5
# (see legacy/README.md). The raw logs and the frozen table are git-ignored;
# only the legacy/ scripts and docs are committed. Re-running is safe: the
# table is rebuilt from legacy/logs/ every time, and `legacy-verify` reports
# if any input file has changed since the freeze.
legacy-results:
	$(PY) legacy/make_legacy_results.py

legacy-verify:
	$(PY) legacy/make_legacy_results.py --check

results:
	$(PY) analysis/compute_results.py --output $(RESULTS)

summary: results
	$(PY) analysis/summarize_results.py --results $(RESULTS)

figures: figures-sweeps figures-shared \
	$(FIGDIR)/foil_lct.pdf $(FIGDIR)/kelvin_helmholtz.pdf

# The family targets run the plotting scripts unfiltered (all machines,
# all scenarios), so they work even when figures/ does not exist yet.
figures-sweeps: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS)

figures-shared: results
	$(PY) analysis/plot_shared_fits.py --results $(RESULTS)

$(FIGDIR)/foil_lct.pdf: results
	$(PY) analysis/plot_foil_lct.py --results $(RESULTS)

$(FIGDIR)/kelvin_helmholtz.pdf: results
	$(PY) analysis/plot_kelvin_helmholtz.py --results $(RESULTS)

# Build a single figure by name: `make figures/sweeps-hal.pdf` or
# `make figures/sweeps-shared-hal.pdf`.
$(FIGDIR)/sweeps-%.pdf: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS) --machine $*

$(FIGDIR)/sweeps-shared-%.pdf: results
	$(PY) analysis/plot_shared_fits.py --results $(RESULTS) --machine $*

# The RO-Crate metadata (see the header): generate it and run make_rocrate.py's
# checks on the result (built-in validation, plus the official rocrate
# package's crate loading when it is installed, reported as a warning).
rocrate:
	$(PY) make_rocrate.py create --out ro-crate-metadata.json
	$(PY) make_rocrate.py check ro-crate-metadata.json

clean:
	rm -rf $(FIGDIR) $(RESULTS) build
	rm -f ro-crate-metadata.json

distclean: clean
	rm -rf src

# --- harness dependency sources, environment, toolchain ---------------------

# Clone and keep the pinned sources in sync. The phony driver runs on every
# make invocation (a changed pin must be noticed even though the stamp
# exists) and rewrites the stamp only when the checkout actually changed;
# the stamp is what invalidates the downstream inputs and builds.
picongpu-src:
	@mkdir -p src
	@if [ -d "$(PICONGPU_ABS)/.git" ]; then
	  if [ "$$(git -C "$(PICONGPU_ABS)" rev-parse HEAD)" != "$(PICONGPU_HASH)" ]; then
	    echo "Updating $(PICONGPU_ABS) to $(PICONGPU_SHORT) ..."
	    git -C "$(PICONGPU_ABS)" fetch --quiet
	    git -C "$(PICONGPU_ABS)" checkout --quiet "$(PICONGPU_HASH)"
	  fi
	  git -C "$(PICONGPU_ABS)" submodule update --init --force --quiet
	  echo "Using $(PICONGPU_ABS) @ $(PICONGPU_SHORT)."
	  if [ "$$(cat "$(PICONGPU_STAMP)" 2>/dev/null)" != "$(PICONGPU_HASH)" ]; then
	    echo "$(PICONGPU_HASH)" >"$(PICONGPU_STAMP)"
	  fi
	else
	  if [ -e "$(PICONGPU_ABS)" ]; then
	    echo "Replacing $(PICONGPU_ABS) (not a git checkout) ..."
	    rm -rf "$(PICONGPU_ABS)"
	  fi
	  echo "Cloning $(PICONGPU_ABS) @ $(PICONGPU_SHORT) ..."
	  git clone "$(PICONGPU_URL)" "$(PICONGPU_ABS)"
	  git -C "$(PICONGPU_ABS)" checkout "$(PICONGPU_HASH)"
	  git -C "$(PICONGPU_ABS)" submodule init
	  git -C "$(PICONGPU_ABS)" submodule update
	  echo "$(PICONGPU_HASH)" >"$(PICONGPU_STAMP)"
	fi

# mallocMC lives inside the picongpu tree, so its clone waits for the
# picongpu checkout to exist.
mallocmc-src: picongpu-src
	@if [ -d "$(MALLOCMC_ABS)/.git" ]; then
	  if [ "$$(git -C "$(MALLOCMC_ABS)" rev-parse HEAD)" != "$(MALLOCMC_HASH)" ]; then
	    echo "Updating $(MALLOCMC_ABS) to $(MALLOCMC_SHORT) ..."
	    git -C "$(MALLOCMC_ABS)" fetch --quiet
	    git -C "$(MALLOCMC_ABS)" checkout --quiet "$(MALLOCMC_HASH)"
	  fi
	  echo "Using $(MALLOCMC_ABS) @ $(MALLOCMC_SHORT)."
	  if [ "$$(cat "$(MALLOCMC_STAMP)" 2>/dev/null)" != "$(MALLOCMC_HASH)" ]; then
	    echo "$(MALLOCMC_HASH)" >"$(MALLOCMC_STAMP)"
	  fi
	else
	  if [ -e "$(MALLOCMC_ABS)" ]; then
	    echo "Replacing $(MALLOCMC_ABS) (not a git checkout) ..."
	    rm -rf "$(MALLOCMC_ABS)"
	  fi
	  echo "Cloning $(MALLOCMC_ABS) @ $(MALLOCMC_SHORT) ..."
	  git clone "$(MALLOCMC_URL)" "$(MALLOCMC_ABS)"
	  git -C "$(MALLOCMC_ABS)" checkout "$(MALLOCMC_HASH)"
	  git -C "$(MALLOCMC_ABS)" submodule init
	  git -C "$(MALLOCMC_ABS)" submodule update
	  echo "$(MALLOCMC_HASH)" >"$(MALLOCMC_STAMP)"
	fi

# The stamp files are side effects of the phony drivers above; the trivial
# recipes only tie them into the dependency graph (their mtime is what the
# inputs and builds key off). The recipe is what matters: make refreshes a
# target's mtime after running its recipe, but not for recipe-less targets,
# so a pin change (a rewritten stamp) would not invalidate the dependents
# otherwise.
$(PICONGPU_STAMP): picongpu-src
	@true
$(MALLOCMC_STAMP): mallocmc-src
	@true

# Prepare the build environment once per make invocation: patch PICSRC= in
# the profile in place (idempotent, applied only when it would change the
# file, so an untouched profile keeps its content), stamp the profile
# content and the build flags, and probe the toolchain the profile loads
# (gcc/cmake/nvcc versions). The probe runs on every invocation, so a module
# update is caught even when the profile file is otherwise unchanged.
env-check:
	@mkdir -p build
	sed 's|PICSRC=.*|PICSRC=$(PICONGPU_ABS)|g' "$(PROFILE)" >build/.profile.patched
	if [ "$$(md5sum build/.profile.patched | awk '{print $$1}')" = "$$(md5sum "$(PROFILE)" | awk '{print $$1}')" ]; then
	  rm -f build/.profile.patched
	else
	  mv build/.profile.patched "$(PROFILE)"
	fi
	if [ "$$(md5sum "$(PROFILE)" | awk '{print $$1}')" != "$$(cat "$(PROFILE_ENV_STAMP)" 2>/dev/null)" ]; then
	  md5sum "$(PROFILE)" | awk '{print $$1}' >"$(PROFILE_ENV_STAMP)"
	fi
	if [ "$$(printf '%s' "$(FLAGS)" | md5sum | awk '{print $$1}')" != "$$(cat "$(FLAGS_STAMP)" 2>/dev/null)" ]; then
	  printf '%s' "$(FLAGS)" | md5sum | awk '{print $$1}' >"$(FLAGS_STAMP)"
	fi
	@source "$(PROFILE)"
	for TOOL in gcc cmake nvcc; do
	  if command -v "$$TOOL" >/dev/null 2>&1; then
	    echo "$$TOOL: $$( $$TOOL --version 2>/dev/null | sed -n 1p)"
	  fi
	done >build/.toolchain.tmp
	if [ -f "$(TOOLCHAIN_STAMP)" ] && \
	  [ "$$(md5sum build/.toolchain.tmp | awk '{print $$1}')" = "$$(md5sum "$(TOOLCHAIN_STAMP)" | awk '{print $$1}')" ]; then
	  rm -f build/.toolchain.tmp
	else
	  mv build/.toolchain.tmp "$(TOOLCHAIN_STAMP)"
	fi

$(PROFILE_ENV_STAMP) $(TOOLCHAIN_STAMP) $(FLAGS_STAMP): env-check
	@true
