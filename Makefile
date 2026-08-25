# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# Build the benchmark numbers (output/results.h5) and the figures
# (figures/). The numbers are rebuilt from the run logs on every
# invocation: make deliberately does not list files of the output/
# tree as prerequisites, because their names are machine-specific
# (run-all job names, timestamps) and not safe to parse as make
# dependencies. Every figure is independent: build any one of them by
# its file name, e.g. `make figures/foil_lct.pdf` or
# `make figures/sweeps-hal.pdf`, or all of them with `make` (the
# default goal, which also prints the summary tables). Rebuild only
# the numbers with `make results`.

PY      := python3
RESULTS := output/results.h5
FIGDIR  := figures

.PHONY: all results summary figures figures-sweeps figures-runtime-stack clean

all: figures summary

results:
	$(PY) analysis/compute_results.py --output $(RESULTS)

summary: results
	$(PY) analysis/summarize_results.py --results $(RESULTS)

figures: figures-sweeps figures-runtime-stack \
	$(FIGDIR)/foil_lct.pdf $(FIGDIR)/kelvin_helmholtz.pdf

# The family targets run the plotting scripts unfiltered (all machines,
# all scenarios), so they work even when figures/ does not exist yet.
figures-sweeps: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS)

figures-runtime-stack: results
	$(PY) analysis/plot_runtime_stack.py --results $(RESULTS)

$(FIGDIR)/foil_lct.pdf: results
	$(PY) analysis/plot_foil_lct.py --results $(RESULTS)

$(FIGDIR)/kelvin_helmholtz.pdf: results
	$(PY) analysis/plot_kelvin_helmholtz.py --results $(RESULTS)

# Build a single figure by name: `make figures/sweeps-hal.pdf` or
# `make figures/runtime-stack-<setup>-<grid>.pdf`.
$(FIGDIR)/sweeps-%.pdf: results
	$(PY) analysis/plot_sweeps.py --results $(RESULTS) --machine $*

$(FIGDIR)/runtime-stack-%.pdf: results
	$(PY) analysis/plot_runtime_stack.py --results $(RESULTS) --name $*

clean:
	rm -rf $(FIGDIR) $(RESULTS)
