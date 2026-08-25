# Makefile
#
# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT
#
# Build harness (rewritten from setup.sh). Clones the pinned PIConGPU and
# mallocMC into src/, prepares one input directory per (example, algorithm)
# (pic-create template + parameter overlay) and builds each one (pic-build).
# What to build (examples, algorithms, dependency pins, build flags) is read
# from config.json through config.py and validated up front.
#
# From the repository root, with the machine's environment loaded (in
# practice via the log_setup_<machine>.sh launchers):
#
#   make PROFILE=profiles/hal.sh PARAM_DIR=param    # clone + inputs + builds
#   make check                                      # resolved configuration
#   make clean                                      # remove build/
#   make distclean                                  # remove build/ and src/
#
# Incremental: a target re-runs only when its prerequisites change (the
# dependency pins, the parameter overlay files, the build flags, the profile
# content, the toolchain versions). Like setup.sh, `make` without arguments
# runs the targets serially; `make -j` builds the independent (example,
# algorithm) pairs in parallel.

.DEFAULT_GOAL := all

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -e -c
.DELETE_ON_ERROR:

CONFIG := config.json

# --- configuration: validated up front, then read through config.py -------

ifneq ($(strip $(shell python3 config.py check >/dev/null 2>&1; echo $$?)),0)
$(error config.json failed validation; run `python3 config.py check` for details)
endif

EXAMPLES   := $(shell python3 config.py list examples)
ALGORITHMS := $(shell python3 config.py list algorithms)

PICONGPU_PATH  := $(shell python3 config.py get dependencies.picongpu.path)
PICONGPU_URL   := $(shell python3 config.py get dependencies.picongpu.url)
PICONGPU_HASH  := $(shell python3 config.py get dependencies.picongpu.hash)
PICONGPU_ABS   := $(CURDIR)/$(PICONGPU_PATH)
PICONGPU_SHORT := $(shell printf '%.8s' $(PICONGPU_HASH))
# add-delay branch: run-time malloc/free delays for every creation policy
# (FlatterScatter, Scatter, Gallatin) via the MALLOCMC_MALLOC_DELAY /
# MALLOCMC_FREE_DELAY environment variables (read into the device allocator
# by mallocMC::Allocator::alloc), the delays as busy-waits on the device
# global timer. We pin our fork, not picongpu's own mallocMC copy.
MALLOCMC_PATH  := $(shell python3 config.py get dependencies.mallocmc.path)
MALLOCMC_URL   := $(shell python3 config.py get dependencies.mallocmc.url)
MALLOCMC_HASH  := $(shell python3 config.py get dependencies.mallocmc.hash)
MALLOCMC_ABS   := $(CURDIR)/$(MALLOCMC_PATH)
MALLOCMC_SHORT := $(shell printf '%.8s' $(MALLOCMC_HASH))

# Nasty little bug here: GCC has a constexpr std::source_location but nvcc
# does not. So Boost gets confused and tries to use std::source_location
# constexpr.
CXX_FLAGS         := $(shell python3 config.py get build.cxx_flags)
EXTRA_CMAKE_FLAGS := $(shell python3 config.py list build.extra_cmake_flags)
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

# Stamps: written by the phony drivers below, consumed as prerequisites.
PICONGPU_STAMP    := $(PICONGPU_ABS)/.dep-stamp
MALLOCMC_STAMP    := $(MALLOCMC_ABS)/.dep-stamp
PROFILE_ENV_STAMP := build/.profile-env
TOOLCHAIN_STAMP   := build/.toolchain
FLAGS_STAMP       := build/.build-flags

# Every goal except check/clean/distclean needs a profile (the default goal
# `all` does too).
NON_EXEMPT_GOALS := $(filter-out check clean distclean,$(MAKECMDGOALS))
ifeq ($(MAKECMDGOALS),)
NON_EXEMPT_GOALS := all
endif
ifneq ($(NON_EXEMPT_GOALS),)
ifeq ($(PROFILE),)
$(error specify PROFILE=<machine profile>, e.g. make PROFILE=profiles/hal.sh)
endif
endif

.PHONY: all check clean distclean picongpu-src mallocmc-src env-check

# --- per (example, algorithm) targets -------------------------------------

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

# --- top-level targets -----------------------------------------------------

all: $(BINARIES)

check:
	python3 config.py check
	@printf 'examples:   %s\n' "$(EXAMPLES)"
	@printf 'algorithms: %s\n' "$(ALGORITHMS)"
	@printf 'picongpu:   %s @ %s\n' "$(PICONGPU_ABS)" "$(PICONGPU_SHORT)"
	@printf 'mallocmc:   %s @ %s\n' "$(MALLOCMC_ABS)" "$(MALLOCMC_SHORT)"
	# Each value comes through a double-quoted command substitution, so any
	# spaces, quotes or leading dashes in them stay one single shell word.
	@printf 'flags:      -DCMAKE_CXX_FLAGS=%s -DCMAKE_CUDA_FLAGS=%s %s\n' \
		"$$(python3 config.py get build.cxx_flags)" \
		"$$(python3 config.py get build.cxx_flags)" \
		"$$(python3 config.py list build.extra_cmake_flags | tr '\n' ' ' | sed 's/ *$$//')"

clean:
	rm -rf build

distclean: clean
	rm -rf src

# --- dependency sources, environment, toolchain -----------------------------

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
