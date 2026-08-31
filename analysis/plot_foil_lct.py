"""The FoilLCT bar chart from the computed results.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws
the paper bar chart of the original three-algorithm comparison: the
zero-delay FoilLCT runs of every hardware, median with IQR error bar, one
bar per allocator (in the file's `algorithm_order`). The chart's
significance test (Kruskal p-value) is stored in the results file's `foil`
and `foil_pvalue` tables; `summarize_results.py` prints it. Saved to
`figures/foil_lct.pdf`.
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
    read_table,
)

mpl.use("pdf")

FIGURES = Path("figures")
# The bar chart's value column, as labelled on the axis.
VALUE_LABEL = "runtime (s)"


def make_figure(runs: pd.DataFrame, algorithms: list[str]) -> plt.Figure:
    """Draw the FoilLCT bar chart.

    Args:
        runs: the runs table of the results file.
        algorithms: the allocators' hue order.

    Returns:
        plt.Figure: the chart.

    """
    foil = runs[no_delay_mask(runs)]
    foil = foil[foil["setup"] == "FoilLCT"][["hardware", RUN_TIME, "algorithm"]].rename(columns={RUN_TIME: VALUE_LABEL})
    ax = sns.barplot(
        foil,
        x="hardware",
        y=VALUE_LABEL,
        estimator="median",
        hue="algorithm",
        errorbar="pi",
        order=HARDWARE_ORDER,
        hue_order=algorithms,
    )
    plt.tight_layout()
    return ax.figure


def main(*, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the FoilLCT bar chart from one results file.

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
        algorithms = algorithm_order(file)
    foil = runs[no_delay_mask(runs)]
    if not len(foil[foil["setup"] == "FoilLCT"]):
        print("no zero-delay FoilLCT runs in the results file; nothing to draw", file=sys.stderr)
        return 0
    FIGURES.mkdir(exist_ok=True)
    fig = make_figure(runs, algorithms)
    fig.savefig(FIGURES / "foil_lct.pdf")
    print(f"wrote {FIGURES / 'foil_lct.pdf'}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The FoilLCT bar chart of the zero-delay runs from output/results.h5 (figures/foil_lct.pdf)."
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
