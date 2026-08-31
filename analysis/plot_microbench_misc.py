"""The microbenchmark's diagnostic figures (utilisation and graph workloads).

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Draws the two diagnostic figures of the microbenchmark suite (the
microbenchmarks/memmansurvey submodule) that the suite's data release
produces with its `plot_synthetic_misc.py`:

- `figures/microbench-utilisation.pdf`: the GPU heap memory utilisation, in
  percent, at each allocation size, of every allocator,
- `figures/microbench-graph.pdf`: the time per operation, in milliseconds, of
  the graph workloads (build and update, plain and range-restricted, on the
  orkut graphs), one bar per (graph scenario, allocator).

These read the suite's raw result CSVs directly (under `microbench.data` of
`config.json`, the git-ignored `microbenchmarks/data/`), unlike the
allocation-cost figures: the figures are diagnostic matrices of the allocator
suite that do not map to a tidy table of the results file. The important
allocators are drawn at full opacity, the rest collapsed into a min-max sleeve
(utilisation) or a min-max background bar (graph). A run whose directory is
absent is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("pdf")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES = Path("figures")
# The allocators the suite measured, in their (shared) colour order, and the
# subset the figures highlight.
ALLOCATORS = [
    "CUDA",
    "Gallatin",
    "Ouroboros-C-S",
    "Ouroboros-C-VA",
    "Ouroboros-C-VL",
    "Ouroboros-P-S",
    "Ouroboros-P-VA",
    "Ouroboros-P-VL",
    "RegEff-AW",
    "RegEff-C",
    "RegEff-CF",
    "RegEff-CFM",
    "RegEff-CM",
    "ScatterAlloc",
    "mallocMC",
    "XMalloc",
]
IMPORTANT_ALLOCATORS = ["Gallatin", "mallocMC", "RegEff-AW", "CUDA"]
ALPHA_IMPORTANT = 1.0
ALPHA_UNIMPORTANT = 0.3
# One colour per allocator, in the ALLOCATORS order (shared by both figures).
_ALLOCATOR_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#6a51a3",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#333333",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#c49c94",
    "#17becf",
    "#ff9896",
    "#4d4d4d",
    "#9edae5",
]
ALLOCATOR_COLOR = dict(zip(ALLOCATORS, _ALLOCATOR_COLORS, strict=True))


def display_name(allocator: str) -> str:
    """Return the figure label of an allocator (mallocMC is shown as FlatterScatter).

    Args:
        allocator: the allocator name.

    Returns:
        str: the display label.

    """
    return "FlatterScatter" if allocator == "mallocMC" else allocator


def _jobids() -> list[int]:
    """Return the suite's run jobids, from the microbench section of `config.json`.

    Returns:
        list[int]: the jobids, in config order.

    """
    with (REPO_ROOT / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    return [int(run["jobid"]) for run in config.get("microbench", {}).get("runs", []) if "jobid" in run]


def _results_dir(data_dir: Path, jobid: int) -> Path:
    """Return the results directory of one run.

    Args:
        data_dir: the suite's raw data directory.
        jobid: the run jobid.

    Returns:
        Path: the `results-<jobid>/results` directory.

    """
    return data_dir / f"results-{jobid}" / "results"


def plot_min_max_sleeve(ax: plt.Axes, series: list[tuple[pd.Series, np.ndarray]]) -> None:
    """Fill the min-max envelope of the non-important allocators' values.

    Args:
        ax: the axis to draw into.
        series: the (x, y) series of the non-important allocators.

    """
    if not series:
        return
    points = pd.DataFrame(
        {"x": np.concatenate([x.to_numpy() for x, _ in series]), "y": np.concatenate([y for _, y in series])}
    )
    bounds = points.groupby("x")["y"].agg(["min", "max"]).sort_index()
    ax.fill_between(bounds.index, bounds["min"], bounds["max"], color="0.6", alpha=ALPHA_UNIMPORTANT, label="others")


def plot_utilisation(data_dir: Path, jobids: list[int]) -> plt.Figure:
    """Draw the GPU heap memory utilisation figure.

    Args:
        data_dir: the suite's raw data directory (`microbench.data`).
        jobids: the run jobids.

    Returns:
        plt.Figure: the chart.

    """
    fig, ax = plt.subplots(figsize=(0.9 * 5.05, 0.9 * 2.63))
    others: list[tuple[pd.Series, np.ndarray]] = []
    for jobid in jobids:
        for filename in sorted(_results_dir(data_dir, jobid).glob("*oom*")):
            data = pd.read_csv(filename, index_col=0, header=None).T.drop("BaseLine", axis=1)
            x = data["Bytes"].astype(int)
            for column in ALLOCATORS:
                if column not in data.columns:
                    continue
                if column in IMPORTANT_ALLOCATORS:
                    ax.errorbar(
                        x,
                        data[column],
                        linestyle="none",
                        marker="o",
                        color=ALLOCATOR_COLOR[column],
                        label=display_name(column),
                    )
                else:
                    others.append((x, data[column].to_numpy()))
    plot_min_max_sleeve(ax, others)
    ax.set_ylim(0, 103)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Allocation size [bytes]")
    ax.set_ylabel("Memory utilisation [%]")
    return fig


def _graph_key(filename: Path) -> str:
    """Return the graph scenario key of a graph result file.

    Args:
        filename: the graph result file path.

    Returns:
        str: the scenario key (the file name without the timestamp and the
        "update" tokens).

    """
    return "_".join(element for element in filename.name[: -len(".csv")].split("_")[3:] if element != "update")


def _gather_graph_files(filenames: list[Path]) -> pd.DataFrame:
    """Gather the per-allocator times of the graph result files.

    Args:
        filenames: the graph result files.

    Returns:
        pd.DataFrame: the (scenario -> allocator -> time) table, only the
        allocators the suite knows.

    """
    data = pd.concat(
        [pd.read_csv(filename, index_col=0).T for filename in filenames],
        keys=[_graph_key(filename) for filename in filenames],
    ).droplevel(1, axis=0)
    return data[list(set(data.columns).intersection(ALLOCATORS))]


def _graph_table(data_dir: Path, jobids: list[int]) -> pd.DataFrame:
    """Build the combined (scenario -> allocator -> time) table across the runs.

    The runs cover the same scenarios; each reports the allocators it tested.
    A (scenario, allocator) entry takes the first run that reports a value.

    Args:
        data_dir: the suite's raw data directory.
        jobids: the run jobids.

    Returns:
        pd.DataFrame: the (scenario -> allocator -> time) table.

    """
    frames = []
    for jobid in jobids:
        files = sorted(_results_dir(data_dir, jobid).glob("*graph*"))
        if files:
            frames.append((jobid, _gather_graph_files(files)))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat([table for _, table in frames], keys=[jobid for jobid, _ in frames], axis=1).dropna(
        how="all", axis=0
    )
    table = pd.DataFrame(index=combined.index, columns=combined.columns.get_level_values(1).unique())
    for allocator in table.columns:
        per_run = combined.xs(allocator, axis=1, level=1)
        table[allocator] = per_run.bfill(axis=1).iloc[:, 0]
    return table


def plot_graph(data_dir: Path, jobids: list[int]) -> plt.Figure:
    """Draw the graph-workload time-per-operation figure.

    Args:
        data_dir: the suite's raw data directory (`microbench.data`).
        jobids: the run jobids.

    Returns:
        plt.Figure: the chart.

    """
    fig, ax = plt.subplots(figsize=(0.9 * 5.05, 0.9 * 2.63))
    data = _graph_table(data_dir, jobids)
    important = [allocator for allocator in IMPORTANT_ALLOCATORS if allocator in data.columns]
    others = [allocator for allocator in data.columns if allocator not in IMPORTANT_ALLOCATORS]
    if others:
        others_min = data[others].min(axis=1)
        others_max = data[others].max(axis=1)
        ax.bar(
            data.index,
            (others_max - others_min).to_numpy(),
            bottom=others_min.to_numpy(),
            width=0.9,
            color="#999999",
            alpha=ALPHA_UNIMPORTANT,
        )
    if important:
        (
            data[important]
            .rename(columns={"mallocMC": "FlatterScatter"})
            .sort_index(axis=1)
            .plot(
                kind="bar",
                ax=ax,
                color=[ALLOCATOR_COLOR[allocator] for allocator in important],
                width=0.9 / max(len(important), 1),
                legend=False,
            )
        )
    ax.set_ylabel("Time per op. [ms]")
    ax.set_yscale("log")
    ax.set_xlabel("Graph scenario")
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment="right")
    return fig


def main(*, show: bool = False, data: Path | None = None) -> int:
    """Draw the microbenchmark's diagnostic figures from the raw result CSVs.

    Args:
        show: display the figures in a window (blocking).
        data: the suite's raw data directory (default: the `microbench.data`
        path of `config.json`).

    Returns:
        int: the process exit code.

    """
    if data is None:
        with (REPO_ROOT / "config.json").open(encoding="utf-8") as handle:
            data = REPO_ROOT / json.load(handle).get("microbench", {}).get("data", "")
    jobids = _jobids()
    if not any(_results_dir(data, jobid).is_dir() for jobid in jobids):
        print(f"no microbenchmark data under {data}; nothing to draw", file=sys.stderr)
        return 0
    FIGURES.mkdir(exist_ok=True)
    utilisation = plot_utilisation(data, jobids)
    utilisation.tight_layout()
    utilisation_path = FIGURES / "microbench-utilisation.pdf"
    utilisation.savefig(utilisation_path)
    print(f"wrote {utilisation_path}")
    graph = plot_graph(data, jobids)
    graph.tight_layout()
    graph_path = FIGURES / "microbench-graph.pdf"
    graph.savefig(graph_path)
    print(f"wrote {graph_path}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The microbenchmark's diagnostic figures (memory utilisation, graph workloads) "
        "from the suite's raw result CSVs (figures/microbench-utilisation.pdf, microbench-graph.pdf)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the figures in a window (blocking); by default they are only saved",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="the suite's raw data directory (default: the microbench.data path of config.json)",
    )
    args = parser.parse_args()
    sys.exit(main(show=args.show, data=args.data))
