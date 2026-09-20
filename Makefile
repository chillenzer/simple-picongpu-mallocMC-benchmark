# Makefile
#
# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT
#
# Four purposes, one Makefile:
#
# 1. The benchmark build harness (rewritten from setup.sh): clones the
#    pinned PIConGPU and mallocMC into src/, prepares one input directory
#    per (commit, example, algorithm, config) (pic-create template + a
#    picongpu/param/mallocMC.param rendered from config.json's
#    configs.<Algorithm>.<config> + parameter overlay) and builds each one
#    (pic-build). What to build (commits, examples, algorithms, the allocator
#    configs, build flags) is read from config.json through config.py and
#    validated up front:
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
#    The stamps depend only on the example's flag-lines fingerprint
#    (the structured flag_lines of config.json, see build/flag-lines.*),
#    so rebuilding the binaries never invalidates finished runs (and
#    `make clean` / `distclean` do not touch run-stamps/). Every run gets
#    one self-contained log per grid run (one line of the example's flag
#    lines), run_<machine>_<commit8>_<Ex>_<Algo>_<config>_m<M>_f<F>_r<I>_<line-sha8>_<time>.txt
#    in the machine's output directory (<commit8> is the commit's PIConGPU
#    hash, <config> the allocator config): a human one-liner `# run:` line,
#    the self-describing `# metadata:` JSON line emitted by logmeta.py
#    (machine, commit, the pinned dependency hashes, the slurm job when
#    running under slurm, the sha256 of the binary used, the build facts
#    of the binary's tree, the host hardware, the user the run happens
#    as, and the user's ORCID iD when the runner has exported $ORCID),
#    and that grid run's full output. Runs are append-only:
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
#    The native per-call allocation costs of the microbenchmark suite
#    (the microbenchmarks/memmansurvey submodule, kept in sync by
#    `make microbench-src`) come from a third, optional source: the frozen
#    table microbenchmarks/microbench_results.h5, built once with
#    `make microbench-results` from the raw CSVs under microbenchmarks/
#    data/ (see microbenchmarks/README.md); when it is absent (a fresh
#    checkout without the machine data), the analysis runs without it.
#    The PIConGPU figure families (the sweeps, the shared fits, the
#    configs- and commits- comparison figures, and the setup figures) each
#    read output/results.h5 and skip gracefully when their comparison axis
#    has at most one distinct value (a single-config / single-commit checkout,
#    or a baseline-only series for the sweep families): the configs- and
#    commits- figures are the primary output of a baseline-only run series.
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
#      make rocrate       # generate and validate the metadata
#      make crate-zip     # pack the full crate into a verified .crate.zip
#
# `make clean` removes everything generated (build/, figures/,
# output/results.h5, ro-crate-metadata.json and ro-crate.crate.zip);
# `make distclean` removes src/ as well. Neither touches the run stamps
# (run-stamps/), which are the record of finished runs; `make clean-runs`
# removes them.
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

# A single space (the `$(space)` used to split a path into its word fields).
empty :=
space := $(empty) $(empty)

# --- harness configuration: validated up front, read through config.py ----

ifneq ($(strip $(shell $(PY) config.py check >/dev/null 2>&1; echo $$?)),0)
$(error config.json failed validation; run `python3 config.py check` for details)
endif

EXAMPLES   := $(shell $(PY) config.py list examples)
ALGORITHMS := $(shell $(PY) config.py list algorithms)

# The configs of each algorithm, one `CONFIGS_OF_<algo>` line per algorithm
# (in config order). These are what the build/run matrix is expanded with:
# a build is one (commit, example, algorithm, config) quadruple.
define CONFIGS_OF
CONFIGS_OF_$(1) := $(shell $(PY) config.py list configs $(1))
endef
$(foreach a,$(ALGORITHMS),$(eval $(call CONFIGS_OF,$(a))))

# One commit is one {name, {picongpu, mallocmc} deps} pair from config.json;
# the harness keeps one PIConGPU checkout per commit (and the mallocMC fork
# nested inside it, pinned to the add-delay branch: run-time malloc/free
# delays for every creation policy via the MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY environment variables, read into the device allocator
# by mallocMC::Allocator::alloc as busy-waits on the device global timer. We
# pin our fork, not picongpu's own mallocMC copy.)
COMMITS := $(shell $(PY) config.py list commits)

# Per-commit dependency pins, resolved from config.json (paths defaulted to
# src/<commit>/picongpu[...]). Each value is a self-contained shell call
# (referencing a sibling variable inside the same eval block would expand
# before it is set). The first commit (config order, typically the `default`
# one) provides the profile's PICSRC and the pic-create/pic-build tooling
# that every commit's build is driven with (per-commit toolchains are a
# documented extension point, not implemented); the per-commit checkout and
# mallocMC prefix are what make the builds differ. $1 = commit name.
define COMMIT_DEPS
PICONGPU_PATH_$(1)    := $(shell $(PY) config.py commit $(1) picongpu path)
PICONGPU_ABS_$(1)     := $(CURDIR)/$(shell $(PY) config.py commit $(1) picongpu path)
PICONGPU_URL_$(1)     := $(shell $(PY) config.py commit $(1) picongpu url)
PICONGPU_HASH_$(1)    := $(shell $(PY) config.py commit $(1) picongpu hash)
PICONGPU_SHORT_$(1)   := $(shell printf '%.8s' $(shell $(PY) config.py commit $(1) picongpu hash))
MALLOCMC_PATH_$(1)    := $(shell $(PY) config.py commit $(1) mallocmc path)
MALLOCMC_ABS_$(1)     := $(CURDIR)/$(shell $(PY) config.py commit $(1) mallocmc path)
MALLOCMC_URL_$(1)     := $(shell $(PY) config.py commit $(1) mallocmc url)
MALLOCMC_HASH_$(1)    := $(shell $(PY) config.py commit $(1) mallocmc hash)
MALLOCMC_SHORT_$(1)   := $(shell printf '%.8s' $(shell $(PY) config.py commit $(1) mallocmc hash))
endef
$(foreach c,$(COMMITS),$(eval $(call COMMIT_DEPS,$(c))))
# The first commit's PIConGPU checkout is the PICSRC every build is driven
# with (see the profile's PICSRC patching in `env-check` below).
_first_commit := $(firstword $(COMMITS))
PICONGPU_ABS := $(PICONGPU_ABS_$(_first_commit))

# The microbenchmark suite (microbenchmarks/memmansurvey, a git submodule;
# its raw results and frozen table live under microbenchmarks/, the
# microbench section of config.json): the pin is kept in sync like the
# cloned dependency sources, and the freeze step is driven like the legacy
# one (analysis/make_microbench.py, `make microbench-results`).
MICROBENCH_PATH  := $(shell $(PY) config.py get microbench.submodule)
MICROBENCH_STAMP := build/.microbench

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
PROFILE_ENV_STAMP := build/.profile-env
TOOLCHAIN_STAMP   := build/.toolchain
FLAGS_STAMP       := build/.build-flags

# Per-commit dependency stamps (the checkout of each commit's PIConGPU and
# the mallocMC fork nested in it). The stamps live inside the checkouts
# themselves, so they are removed with the checkout by `make distclean`.
define COMMIT_STAMPS
PICONGPU_STAMP_$(1)  := $(PICONGPU_ABS_$(1))/.dep-stamp
MALLOCMC_STAMP_$(1)  := $(MALLOCMC_ABS_$(1))/.dep-stamp
endef
$(foreach c,$(COMMITS),$(eval $(call COMMIT_STAMPS,$(c))))

# The example's picongpu command lines come from config.json (the structured
# flag_lines), not from a flags/ file. This fingerprint stamp is the
# run-stamps' only prerequisite: it is rewritten (only) when the example's
# flag_lines in config.json change, so editing them invalidates exactly that
# example's stamped runs and nothing a rebuild touches.
build/flag-lines.%.stamp: $(CONFIG)
	@mkdir -p build
	HASH=$$($(PY) config.py flag-lines $* | sha256sum | awk '{print $$1}')
	if [ "$$HASH" != "$$(cat $@ 2>/dev/null)" ]; then
	  printf '%s' "$$HASH" >"$@"
	  echo "flag lines changed for $* ($$HASH)."
	fi

# Only the benchmark goal (`build`) and the explicit per-pair targets under
# build/ run on the HPC machine and need its profile; the default goal and
# all analysis goals, check and clean do not.
ifneq ($(filter build%,$(MAKECMDGOALS)),)
ifeq ($(PROFILE),)
$(error specify PROFILE=<machine profile>, e.g. make build PROFILE=profiles/hal.sh)
endif
endif

.PHONY: all build check clean distclean results summary figures \
	figures-picongpu figures-sweeps figures-shared figures-configs figures-commits figures-microbench \
	$(foreach c,$(COMMITS),picongpu-src-$(c) mallocmc-src-$(c)) microbench-src env-check env runs full \
	clean-runs freeze freeze-verify \
	legacy-results legacy-verify microbench-results microbench-verify microbench-audit \
	rocrate crate-zip sweep-status

# --- per (commit, example, algorithm) harness targets -----------------------

# The (commit, example, algorithm, config) quadruples, one
# "<c>/<e>/<a>/<cfg>" per build dir. The build dir is build/<c>/<e>/<a>/<cfg>
# and the binary build/<c>/<e>/<a>/<cfg>/bin/picongpu.
QUADS      := $(foreach c,$(COMMITS),$(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(foreach g,$(CONFIGS_OF_$(a)),$(c)/$(e)/$(a)/$(g)))))
BUILD_DIRS := $(addprefix build/,$(QUADS))
BINARIES   := $(addsuffix /bin/picongpu,$(BUILD_DIRS))

# One (commit, example, algorithm, config) quadruple: $1 = commit, $2 =
# example, $3 = algorithm, $4 = config, $5 = build dir
# (build/<c>/<e>/<a>/<cfg>).
#
# The input is regenerated when the commit's PIConGPU pin (the pic-create
# template), the profile (PICSRC feeds pic-create's tool path), or the
# parameter inputs change: the algorithm's mallocMC.param.in template, its
# profile headers (copied into the build tree), config.json (it holds both
# the config's heap scalars and its hash profile name), or the example's own
# *.param overlay files (their own param directories, so a newly added
# parameter file is picked up).
#
# The build is skipped while the input, the commit's mallocMC pin (the
# headers are compiled into the binary), the profile content, the build
# flags and the toolchain versions are all unchanged; targeting the binary
# itself also covers a manually deleted bin/picongpu.
define pair_rules
build/$(5)/.input-stamp: $(PICONGPU_STAMP_$(1)) $(PROFILE_ENV_STAMP) \
	$(CONFIG) \
	$(PARAM_DIR)/$(3)/mallocMC.param.in \
	$(wildcard $(PARAM_DIR)/$(3)/profiles/*.hpp) \
	$(wildcard $(PARAM_DIR)/$(2)/*.param) \
	$(wildcard $(PARAM_DIR)/$(2)/$(3)/*.param) \
	$(if $(wildcard $(PARAM_DIR)/$(3)/.),$(PARAM_DIR)/$(3)) \
	$(if $(wildcard $(PARAM_DIR)/$(2)/.),$(PARAM_DIR)/$(2)) \
	$(if $(wildcard $(PARAM_DIR)/$(2)/$(3)/.),$(PARAM_DIR)/$(2)/$(3))
	@echo "Preparing input build/$(5) ... ($(3)/$(4))"
	@rm -rf build/$(5)
	@mkdir -p build/$(5)
	@source "$(PROFILE)"
	pic-create "$(PICONGPU_ABS)/share/picongpu/examples/$(2)" "build/$(5)"
	@mkdir -p "build/$(5)/include/picongpu/param"
	# The algorithm's allocator: the mallocMC.param is rendered from its
	# template (param/<algo>/mallocMC.param.in) by config.py, the profiles
	# it includes are copied in, and the example's own parameters (and any
	# per-(example, algorithm) overrides) are overlaid on top; later levels
	# win on name clashes.
	$(PY) config.py render-param "$(3)" "$(4)" >"build/$(5)/include/picongpu/param/mallocMC.param"
	if [ -d "$(PARAM_DIR)/$(3)/profiles" ]; then cp -r -- "$(PARAM_DIR)/$(3)/profiles" "build/$(5)/include/picongpu/param/"; fi
	find $(PARAM_DIR)/* -type f -wholename '$(PARAM_DIR)/$(2)/*.param' -exec cp -v {} "build/$(5)/include/picongpu/param/" ';'
	find $(PARAM_DIR)/* -type f -wholename '$(PARAM_DIR)/$(2)/$(3)/*.param' -exec cp -v {} "build/$(5)/include/picongpu/param/" ';'
	@printf 'input ready for %s (%s/%s)\n' "$(5)" "$(3)" "$(4)" >"build/$(5)/.input-stamp"
	@echo "Prepared input build/$(5)."

build/$(5)/bin/picongpu: build/$(5)/.input-stamp $(MALLOCMC_STAMP_$(1)) $(PROFILE_ENV_STAMP) $(TOOLCHAIN_STAMP) $(FLAGS_STAMP)
	@source "$(PROFILE)"
	cd "build/$(5)"
	export CMAKE_PREFIX_PATH="$(MALLOCMC_ABS_$(1)):$${CMAKE_PREFIX_PATH:-}"
	pic-build -c "$(FLAGS)"
endef

$(foreach t,$(QUADS),$(eval $(call pair_rules,$(word 1,$(subst /,$(space),$(t))),$(word 2,$(subst /,$(space),$(t))),$(word 3,$(subst /,$(space),$(t))),$(word 4,$(subst /,$(space),$(t))),$(t))))

# --- run stamps: one per (combination, repetition) --------------------------
#
# A run is one (commit, example, algorithm) build through one (malloc delay,
# free delay) combination: the example's whole flag lines once (the
# structured flag_lines of config.json), with the combination's delays in
# the environment. The sweep order is commit, then example, then algorithm,
# then repetition, and within a repetition the combination order of
# `config.py list run-matrix`, so every repetition is a full sweep and when
# REPEATS is finished, every build has finished it.
#
# The stamp is the record that the run happened (its content is the
# run's log paths, one per line, i.e. the paths of its current vintage),
# and it is what makes an interrupted series resumable. Its only
# prerequisite is the example's flag-lines fingerprint stamp
# (build/flag-lines.<Ex>.stamp): editing the example's flag_lines in
# config.json invalidates exactly that example's stamped runs, and nothing
# that `make build` touches (pins, profile, toolchain, a rebuild) ever
# invalidates a finished run. The recipe itself refuses to start while the
# build's binary is missing.
#
# The repetitions of this invocation (all of 1..REPEATS, or only REP in the
# slurm case: one job per repetition) are expanded at parse time into one
# explicit target per (combination, repetition).
REPS := $(strip $(shell seq 1 '$(REPEATS)' 2>/dev/null))

# COMMIT restricts a `runs`/`full`/`sweep-status`/`clean-runs` invocation to
# one commit (the slurm case: parallelise the commits across nodes; the
# single-commit case is the default). Like MACHINE, it is an invocation
# value, not a configuration key.
ifeq ($(origin COMMIT),environment)
COMMIT =
endif
COMMIT ?=
ifneq ($(filter runs full sweep-status clean-runs,$(MAKECMDGOALS)),)
  ifneq ($(strip $(COMMIT)),)
    ifeq ($(filter $(COMMIT),$(COMMITS)),$(COMMIT))
    else
      $(error specify COMMIT=<name> from the config commits list, e.g. make runs MACHINE=hal COMMIT=default (got '$(COMMIT)'))
    endif
  endif
endif
RUN_COMMITS := $(if $(strip $(COMMIT)),$(strip $(COMMIT)),$(COMMITS))

# $1 = commit, $2 = example, $3 = algorithm, $4 = config, $5 = malloc delay
# (ns), $6 = free delay (ns), $7 = repetition number. The recipe carries no
# shell of its own (the rule is generated by $(eval $(call ...)) and is
# therefore expanded twice), it just hands the parameters to run_stamp.sh,
# which does the rest.
define run_stamp_rules
run-stamps/$(MACHINE)/$(1)/$(2)/$(3)/$(4)/$(5)_$(6)/rep-$(7).stamp: build/flag-lines.$(2).stamp
	@bash run_stamp.sh "$(MACHINE)" "$(REPEATS)" "$(1)" "$(2)" "$(3)" "$(4)" "$(5)" "$(6)" "$(7)"
endef

$(foreach c,$(RUN_COMMITS),$(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(foreach g,$(CONFIGS_OF_$(a)),$(foreach m,$(COMBOS),$(foreach r,$(REPS),$(eval $(call run_stamp_rules,$(c),$(e),$(a),$(g),$(firstword $(subst _, ,$(m))),$(lastword $(subst _, ,$(m))),$(r)))))))))

# --- run targets -------------------------------------------------------------

ifeq ($(strip $(REP)),)
RUN_REPS := $(REPS)
else
RUN_REPS := $(REP)
endif
RUN_STAMPS := $(foreach c,$(RUN_COMMITS),$(foreach e,$(EXAMPLES),$(foreach a,$(ALGORITHMS),$(foreach g,$(CONFIGS_OF_$(a)),$(foreach i,$(RUN_REPS),$(addprefix run-stamps/$(MACHINE)/$(c)/$(e)/$(a)/$(g)/,$(addsuffix /rep-$(i).stamp,$(COMBOS))))))))

# The fan-out: one sub-make per (combination, repetition) stamp, in sweep
# order, serially (one GPU; -j cannot parallelize a single recipe). A stamp
# that is up to date (present and not older than the example's flag-lines
# fingerprint, so the freshness test make itself would apply) is skipped
# without a sub-make; everything else goes through make's normal freshness
# rules.
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
	  EX=$$(printf '%s\n' "$$stamp" | cut -d/ -f4)
	  if [ -f "$$stamp" ] && [ ! "build/flag-lines.$$EX.stamp" -nt "$$stamp" ]; then
	    STAMPED=$$(($$STAMPED + 1))
	    continue
	  fi
	  $(MAKE) --no-print-directory "$$stamp"
	done
	@echo "runs [$(MACHINE)]: $${STAMPED} of $(words $(RUN_STAMPS)) runs already stamped${if $(COMMIT), (commit $(COMMIT))}."

# The one-command machine-side flow: build the harness, then run the whole
# series. The analysis (figures, summary) stays a plain `make`, which is
# where it belongs once the logs are on a laptop.
full:
	@$(MAKE) build PROFILE="$$(python3 config.py get machines.$(MACHINE).profile)" PARAM_DIR=param
	@$(MAKE) runs

# Remove the run stamps (of all machines, or of one; and, when a machine is
# named together with a COMMIT, of just that commit's): the next `make runs`
# then repeats those series, as a new vintage next to the existing logs.
# No log is ever removed by make.
clean-runs:
	@if [ -n "$(MACHINE)" ]; then
	  python3 config.py get "machines.$(MACHINE).output" >/dev/null || { echo "make clean-runs: MACHINE=$(MACHINE) is not in the config machines table" >&2; exit 1; }
	  if [ -n "$(COMMIT)" ]; then
	    rm -rf "run-stamps/$(MACHINE)/$(COMMIT)"
	    echo "removed run-stamps/$(MACHINE)/$(COMMIT)"
	  else
	    rm -rf "run-stamps/$(MACHINE)"
	    echo "removed run-stamps/$(MACHINE)"
	  fi
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
	@for a in $(ALGORITHMS); do
	  printf 'configs    %s: %s\n' "$$a" "$$(python3 config.py list configs $$a | tr '\n' ' ' | sed 's/ *$$//')"
	done
	@if [ "$$(python3 config.py list run-matrix arms | wc -l | tr -d ' ')" = "1" ]; then
	  printf 'runs:       baseline only (no delay sweep configured)\n'
	else
	  printf 'runs:       %s combinations per repetition (initial): %s\n' \
	    "$$(python3 config.py list run-matrix initial | wc -l | tr -d ' ')" \
	    "$$(python3 config.py list run-matrix initial | tr '\n' ' ' | sed 's/ *$$//')"
	  printf 'runs:       %s combinations per repetition (arms):    %s\n' \
	    "$$(python3 config.py list run-matrix arms | wc -l | tr -d ' ')" \
	    "$$(python3 config.py list run-matrix arms | tr '\n' ' ' | sed 's/ *$$//')"
	fi
	@printf 'builds:     %s (commit/example/algorithm/config quadruples)\n' "$(words $(BINARIES))"
	@for c in $(COMMITS); do
	  printf 'commit %-14s picongpu %s @ %s\n' "$$c" "$$(python3 config.py commit $$c picongpu path)" "$$(printf '%.8s' $$(python3 config.py commit $$c picongpu hash))"
	  printf '             mallocmc %s @ %s\n' "$$(python3 config.py commit $$c mallocmc path)" "$$(printf '%.8s' $$(python3 config.py commit $$c mallocmc hash))"
	done
	@printf 'microbench: %s @ %s\n' "$(MICROBENCH_PATH)" \
		"$$(git -C "$(MICROBENCH_PATH)" rev-parse --short HEAD 2>/dev/null || echo 'not initialised')"
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

# Freeze the microbenchmark results (the memmansurvey allocation-test CSVs
# under microbenchmarks/data/, the runs of the microbench section of
# config.json) once into the frozen table (microbenchmarks/
# microbench_results.h5): the main analysis reads the native per-call
# allocation costs from it. Re-running is safe: the table is rebuilt from
# the raw CSVs every time, and `microbench-verify` reports if any input
# file has changed since the freeze.
microbench-results:
	$(PY) analysis/make_microbench.py

microbench-verify:
	$(PY) analysis/make_microbench.py --check

# Audit the raw microbenchmark CSVs for missing x-values and their causes
# (the per-allocator files whose rows the freeze drops: a timeout marker, a
# crashed partial line, or an absent file), reading the raw data directory
# from the microbench section of config.json and printing a per-file report
# plus a grand total.
microbench-audit:
	$(PY) analysis/audit_microbench.py

# Freeze every raw source (the PIConGPU legacy logs and the microbenchmark
# CSVs) into its frozen table. The freezes are separate steps from `results`;
# each source freezes independently and is skipped when its raw data is
# absent (a fresh checkout without the machine data), so this runs cleanly on
# any checkout. Use `freeze-verify` to report if any input file has changed
# since the last freeze.
freeze: legacy-results microbench-results

freeze-verify: legacy-verify microbench-verify

# Compute the analysis tables (output/results.h5) from the sweep machines'
# run logs and the frozen tables (legacy + microbench, built by `freeze`).
# The freezes are a separate step; run `make freeze` first to refresh them.
results:
	$(PY) analysis/compute_results.py --output $(RESULTS)

summary: results
	$(PY) analysis/summarize_results.py --results $(RESULTS)

# The full figure set: both the PIConGPU figures and the microbenchmark
# figures. Each family depends on `results` and skips gracefully when its
# data is absent (a fresh checkout without the machine data), so `make
# figures` runs cleanly on any checkout.
figures: figures-picongpu figures-microbench

# The PIConGPU figures: the per-machine sweep fits, the shared fits, the two
# comparison figure families (configs and commits), and the setup-specific
# figures (FoilLCT, KelvinHelmholtz, the fade models, the native-cost
# comparison). The family targets run the plotting scripts unfiltered (all
# machines, all scenarios, all configs/commits), so they work even when
# figures/ does not exist yet.
figures-picongpu: figures-sweeps figures-shared figures-configs figures-commits \
	$(FIGDIR)/foil_lct.pdf $(FIGDIR)/kelvin_helmholtz.pdf $(FIGDIR)/fade-models.pdf \
	$(FIGDIR)/native-cost.pdf

figures-sweeps: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS)

figures-shared: results
	$(PY) analysis/plot_shared_fits.py --results $(RESULTS)

# The per-machine allocator-config comparison: one figure per (machine,
# commit) with an algorithm carrying two or more configs (skips otherwise).
figures-configs: results
	$(PY) analysis/plot_configs.py --results $(RESULTS) --dep-commit all

# The per-machine dependency-commit comparison: one figure per (machine,
# config) with two or more commits (skips otherwise).
figures-commits: results
	$(PY) analysis/plot_commits.py --results $(RESULTS) --config all

# The microbenchmark figures: the allocation-cost figures (from the frozen
# table in results.h5) and the diagnostic figures (from the suite's raw CSVs
# under microbench.data). The microbench data is an optional source: the
# allocation-cost figures are drawn only when `results` carries the frozen
# tables (built by `make freeze`), and the diagnostic figures only when the
# raw CSVs are present. Both plotting scripts skip gracefully when their data
# is absent (a fresh checkout without the machine data), so this target runs
# cleanly on any checkout.
figures-microbench: results
	$(PY) analysis/plot_microbench.py --results $(RESULTS) && \
	$(PY) analysis/plot_microbench_misc.py

$(FIGDIR)/foil_lct.pdf: results
	$(PY) analysis/plot_foil_lct.py --results $(RESULTS)

$(FIGDIR)/kelvin_helmholtz.pdf: results
	$(PY) analysis/plot_kelvin_helmholtz.py --results $(RESULTS)

$(FIGDIR)/fade-models.pdf: results
	$(PY) analysis/plot_fade_models.py --results $(RESULTS)

# The absorbed-slack-vs-native-cost comparison (figures/native-cost.pdf).
# Data-optional: with no frozen microbenchmark table the script prints a note
# and writes no file, so the fit stays unconstrained and the figure is skipped.
$(FIGDIR)/native-cost.pdf: results
	$(PY) analysis/plot_native_cost.py --results $(RESULTS)

# Build a single figure by name: `make figures/sweeps-hal.pdf` or
# `make figures/sweeps-shared-hal.pdf`.
$(FIGDIR)/sweeps-%.pdf: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS) --machine $*

$(FIGDIR)/sweeps-shared-%.pdf: results
	$(PY) analysis/plot_shared_fits.py --results $(RESULTS) --machine $*

# A single machine's comparison figures (the single-commit/single-config form,
# `make figures/configs-hal.pdf`): `configs-<machine>.pdf` uses the first
# commit, `commits-<machine>.pdf` the `default` config. (The all-commits /
# all-configs variants carry a `-<commit>` / `-<config>` suffix and are built
# by the unfiltered family targets above.)
$(FIGDIR)/configs-%.pdf: results
	$(PY) analysis/plot_configs.py --results $(RESULTS) --machine $*

$(FIGDIR)/commits-%.pdf: results
	$(PY) analysis/plot_commits.py --results $(RESULTS) --machine $*

# The RO-Crate metadata (see the header): generate it and run make_rocrate.py's
# checks on the result (built-in validation, plus the official rocrate
# package's crate loading when it is installed, reported as a warning).
rocrate:
	$(PY) make_rocrate.py create --out ro-crate-metadata.json
	$(PY) make_rocrate.py check ro-crate-metadata.json

# The RO-Crate as a portable, self-describing archive (the RO-Crate
# packaging convention): the full crate - the metadata and every data file
# it references - zipped with a fixed entry time, then verified by
# unzipping and re-running the checks on the result.
crate-zip:
	$(PY) make_rocrate.py zip --out ro-crate.crate.zip

clean:
	rm -rf $(FIGDIR) $(RESULTS) build
	rm -f ro-crate-metadata.json ro-crate.crate.zip

distclean: clean
	rm -rf src

# --- harness dependency sources, environment, toolchain ---------------------

# Clone and keep each commit's pinned PIConGPU checkout in sync. The phony
# driver runs on every make invocation (a changed pin must be noticed even
# though the stamp exists) and rewrites the stamp only when the checkout
# actually changed; the stamp is what invalidates the downstream inputs and
# builds. One driver per commit (the checkout lives at
# PICONGPU_ABS_$(1)). $1 = commit name.
define picongpu_src_rules
picongpu-src-$(1):
	@mkdir -p src
	@if [ -d "$(PICONGPU_ABS_$(1))/.git" ]; then
	  if [ "$$(git -C "$(PICONGPU_ABS_$(1))" rev-parse HEAD)" != "$(PICONGPU_HASH_$(1))" ]; then
	    echo "Updating $(PICONGPU_ABS_$(1)) [$(1)] to $(PICONGPU_SHORT_$(1)) ..."
	    git -C "$(PICONGPU_ABS_$(1))" fetch --quiet
	    git -C "$(PICONGPU_ABS_$(1))" checkout --quiet "$(PICONGPU_HASH_$(1))"
	  fi
	  git -C "$(PICONGPU_ABS_$(1))" submodule update --init --force --quiet
	  echo "Using $(PICONGPU_ABS_$(1)) [$(1)] @ $(PICONGPU_SHORT_$(1))."
	  if [ "$$(cat "$(PICONGPU_STAMP_$(1))" 2>/dev/null)" != "$(PICONGPU_HASH_$(1))" ]; then
	    echo "$(PICONGPU_HASH_$(1))" >"$(PICONGPU_STAMP_$(1))"
	  fi
	else
	  if [ -e "$(PICONGPU_ABS_$(1))" ]; then
	    echo "Replacing $(PICONGPU_ABS_$(1)) [$(1)] (not a git checkout) ..."
	    rm -rf "$(PICONGPU_ABS_$(1))"
	  fi
	  echo "Cloning $(PICONGPU_ABS_$(1)) [$(1)] @ $(PICONGPU_SHORT_$(1)) ..."
	  git clone "$(PICONGPU_URL_$(1))" "$(PICONGPU_ABS_$(1))"
	  git -C "$(PICONGPU_ABS_$(1))" checkout "$(PICONGPU_HASH_$(1))"
	  git -C "$(PICONGPU_ABS_$(1))" submodule init
	  git -C "$(PICONGPU_ABS_$(1))" submodule update
	  echo "$(PICONGPU_HASH_$(1))" >"$(PICONGPU_STAMP_$(1))"
	fi
endef
$(foreach c,$(COMMITS),$(eval $(call picongpu_src_rules,$(c))))

# mallocMC lives inside the commit's picongpu tree, so its clone waits for
# that commit's picongpu checkout to exist. $1 = commit name.
define mallocmc_src_rules
mallocmc-src-$(1): picongpu-src-$(1)
	@if [ -d "$(MALLOCMC_ABS_$(1))/.git" ]; then
	  if [ "$$(git -C "$(MALLOCMC_ABS_$(1))" rev-parse HEAD)" != "$(MALLOCMC_HASH_$(1))" ]; then
	    echo "Updating $(MALLOCMC_ABS_$(1)) [$(1)] to $(MALLOCMC_SHORT_$(1)) ..."
	    git -C "$(MALLOCMC_ABS_$(1))" fetch --quiet
	    git -C "$(MALLOCMC_ABS_$(1))" checkout --quiet "$(MALLOCMC_HASH_$(1))"
	  fi
	  echo "Using $(MALLOCMC_ABS_$(1)) [$(1)] @ $(MALLOCMC_SHORT_$(1))."
	  if [ "$$(cat "$(MALLOCMC_STAMP_$(1))" 2>/dev/null)" != "$(MALLOCMC_HASH_$(1))" ]; then
	    echo "$(MALLOCMC_HASH_$(1))" >"$(MALLOCMC_STAMP_$(1))"
	  fi
	else
	  if [ -e "$(MALLOCMC_ABS_$(1))" ]; then
	    echo "Replacing $(MALLOCMC_ABS_$(1)) [$(1)] (not a git checkout) ..."
	    rm -rf "$(MALLOCMC_ABS_$(1))"
	  fi
	  echo "Cloning $(MALLOCMC_ABS_$(1)) [$(1)] @ $(MALLOCMC_SHORT_$(1)) ..."
	  git clone "$(MALLOCMC_URL_$(1))" "$(MALLOCMC_ABS_$(1))"
	  git -C "$(MALLOCMC_ABS_$(1))" checkout "$(MALLOCMC_HASH_$(1))"
	  git -C "$(MALLOCMC_ABS_$(1))" submodule init
	  git -C "$(MALLOCMC_ABS_$(1))" submodule update
	  echo "$(MALLOCMC_HASH_$(1))" >"$(MALLOCMC_STAMP_$(1))"
	fi
endef
$(foreach c,$(COMMITS),$(eval $(call mallocmc_src_rules,$(c))))

# The microbenchmark suite is a git submodule (the microbench section of
# config.json), pinned by the gitlink of the repository HEAD: the driver
# initialises it when absent, re-checks it out at the pinned commit when it
# has drifted (a pull of the outer repository moves the pin), and
# initialises the suite's own framework submodules (its .gitmodules). The
# stamp records the pinned commit, like the dependency stamps; nothing in
# the PIConGPU build keys off it (the suite is not compiled into the
# benchmark binaries).
microbench-src:
	@if [ ! -d "$(MICROBENCH_PATH)/.git" ]; then
	  if [ -e "$(MICROBENCH_PATH)" ]; then
	    echo "Replacing $(MICROBENCH_PATH) (not a git checkout) ..."
	    rm -rf "$(MICROBENCH_PATH)"
	  fi
	  echo "Initialising $(MICROBENCH_PATH) ..."
	  git submodule update --init --force -- "$(MICROBENCH_PATH)"
	else
	  PINNED=$$(git ls-tree HEAD -- "$(MICROBENCH_PATH)" | awk '{print $$3}')
	  if [ "$$(git -C "$(MICROBENCH_PATH)" rev-parse HEAD)" != "$$PINNED" ]; then
	    echo "Updating $(MICROBENCH_PATH) to $$(printf '%.8s' $$PINNED) ..."
	    git submodule update --force -- "$(MICROBENCH_PATH)"
	  fi
	fi
	git -C "$(MICROBENCH_PATH)" submodule update --init --force --quiet
	if [ "$$(cat "$(MICROBENCH_STAMP)" 2>/dev/null)" != "$$(git -C "$(MICROBENCH_PATH)" rev-parse HEAD)" ]; then
	  git -C "$(MICROBENCH_PATH)" rev-parse HEAD >"$(MICROBENCH_STAMP)"
	fi
	@echo "Using $(MICROBENCH_PATH) @ $$(printf '%.8s' "$$(git -C "$(MICROBENCH_PATH)" rev-parse HEAD)")."

# The stamp files are side effects of the per-commit phony drivers above;
# the trivial recipes only tie them into the dependency graph (their mtime
# is what the inputs and builds key off). The recipe is what matters: make
# refreshes a target's mtime after running its recipe, but not for
# recipe-less targets, so a pin change (a rewritten stamp) would not
# invalidate the dependents otherwise. $1 = commit name.
define stamp_rules
$(PICONGPU_STAMP_$(1)): picongpu-src-$(1)
	@true
$(MALLOCMC_STAMP_$(1)): mallocmc-src-$(1)
	@true
endef
$(foreach c,$(COMMITS),$(eval $(call stamp_rules,$(c))))
$(MICROBENCH_STAMP): microbench-src
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
