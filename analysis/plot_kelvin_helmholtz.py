"""The KelvinHelmholtz violin chart from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws
the paper violin chart of the original three-algorithm comparison: the
no-delay KelvinHelmholtz runs of every hardware, the runtime relative to
the ScatterAlloc reference runtime (per (hardware, memory) group, from the
results file's `khi` table), one violin per allocator, one column per
estimated particle memory. The chart's metadata (reference runtime,
outlier count, Kruskal p-value, FlatterScatter median) is in the `khi`
table; `summarize_results.py` prints it. Saved to
`figures/kelvin_helmholtz.pdf`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from results_io import (
    HARDWARE_ORDER,
    RESULTS,
    RUN_TIME,
    algorithm_order,
    load_results,
    no_delay_mask,
    particle_memory_gb,
    read_table,
)

mpl.use("pdf")

FIGURES = Path("figures")
MEM_LABEL = "estimated particle memory consumption in GB"
YMIN, YMAX = 0.9, 1.1


def make_figure(runs: pd.DataFrame, khi: pd.DataFrame, algorithms: list[str]) -> plt.Figure:
    """Draw the KelvinHelmholtz violin chart.

    Args:
        runs: the runs table of the results file.
        khi: the `khi` table of the results file (reference runtimes).
        algorithms: the allocators' hue order.

    Returns:
        plt.Figure: the chart.

    """
    khi_runs = runs[no_delay_mask(runs)]
    khi_runs = khi_runs[khi_runs["setup"] == "KelvinHelmholtz"]
    frame = pd.DataFrame(
        {
            "hardware": khi_runs["hardware"],
            "algorithm": khi_runs["algorithm"],
            MEM_LABEL: [particle_memory_gb(x, y, z) for x, y, z in khi_runs[["x", "y", "z"]].to_numpy()],
        }
    )
    reference = khi.set_index(["hardware", "memory_gb"])["reference_runtime"]
    reference = reference.reindex(frame.set_index(["hardware", MEM_LABEL]).index).to_numpy()
    frame["relative runtime"] = khi_runs[RUN_TIME].to_numpy() / reference
    ax = sns.catplot(
        frame,
        kind="violin",
        x="hardware",
        y="relative runtime",
        hue="algorithm",
        col=MEM_LABEL,
        sharey=True,
        legend_out=False,
        split=False,
        hue_order=algorithms,
        order=HARDWARE_ORDER,
        orient="v",
    )
    ax.refline(y=1)
    ax.set(ylim=(YMIN, YMAX))
    plt.tight_layout()
    return ax.figure


def main(*, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the KelvinHelmholtz violin chart from one results file.

    Args:
        show: display the figure in a window (blocking).
        results: the results file, e.g. `output/results.h5`.

    Returns:
        int: the process exit code.

    """
    try:
        file = load_results(results)
    except OSError as err:
        print(f"{err}\nno results file: run `python3 analysis/compute_results.py` first", file=sys.stderr)
        return 1
    with file:
        runs = read_table(file, "runs")
        khi = read_table(file, "khi")
        algorithms = algorithm_order(file)
    khi_runs = runs[no_delay_mask(runs)]
    if not len(khi_runs[khi_runs["setup"] == "KelvinHelmholtz"]):
        print("no no-delay KelvinHelmholtz runs in the results file", file=sys.stderr)
        return 1
    FIGURES.mkdir(exist_ok=True)
    fig = make_figure(runs, khi, algorithms)
    fig.savefig(FIGURES / "kelvin_helmholtz.pdf")
    print(f"wrote {FIGURES / 'kelvin_helmholtz.pdf'}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The KelvinHelmholtz violin chart of the no-delay runs from output/results.h5 "
        "(figures/kelvin_helmholtz.pdf)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the figure in a window (blocking); by default it is only saved",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results file (default: %(default)s)",
    )
    args = parser.parse_args()
    sys.exit(main(show=args.show, results=args.results))
