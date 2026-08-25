# SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
# SPDX-License-Identifier: MIT

# Build the benchmark numbers (output/results.h5) and the figures
# (figures/). Every figure is independent: build any one of them by its
# file name, e.g. `make figures/foil_lct.pdf` or
# `make figures/sweeps-hal.pdf`, or all of them with `make` (the default
# goal, which also prints the summary tables). Regenerate the numbers
# from the run logs with `make results`.

PY      := python3
RESULTS := output/results.h5
FIGDIR  := figures
LOGS    := $(wildcard output/*/*)

.PHONY: all results summary figures figures-sweeps figures-runtime-stack clean

all: figures summary

results: $(RESULTS)

$(RESULTS): $(wildcard analysis/*.py) config.json $(LOGS)
	$(PY) analysis/compute_results.py --output $(RESULTS)

$(FIGDIR)/foil_lct.pdf: $(RESULTS)
	$(PY) analysis/plot_foil_lct.py --results $(RESULTS)

$(FIGDIR)/kelvin_helmholtz.pdf: $(RESULTS)
	$(PY) analysis/plot_kelvin_helmholtz.py --results $(RESULTS)

$(FIGDIR)/sweeps-%.pdf: $(RESULTS)
	$(PY) analysis/plot_sweeps.py --results $(RESULTS) --machine $*

$(FIGDIR)/runtime-stack-%.pdf: $(RESULTS)
	$(PY) analysis/plot_runtime_stack.py --results $(RESULTS) --name $*

# The families below enumerate their members (machines, scenarios) from
# the results file, so `make` stays correct for new machines or grids.
figures-sweeps: $(RESULTS)
	$(PY) analysis/plot_sweeps.py --results $(RESULTS)

figures-runtime-stack: $(RESULTS)
	$(PY) analysis/plot_runtime_stack.py --results $(RESULTS)

summary: $(RESULTS)
	$(PY) analysis/summarize_results.py --results $(RESULTS)

figures: $(RESULTS) $(FIGDIR)/foil_lct.pdf $(FIGDIR)/kelvin_helmholtz.pdf figures-sweeps figures-runtime-stack

clean:
	rm -f $(RESULTS) $(FIGDIR)/*.pdf
