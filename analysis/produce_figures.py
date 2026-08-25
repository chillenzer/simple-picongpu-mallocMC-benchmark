"""Benchmark figures and runtime statistics from the run_all.sh cluster logs.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads the per-run no-delay simulation times from the `output/<cluster>/`
logs — the old-layout three-algorithm runs plus the (0, 0) baseline runs of
the delay-combination sweeps (the `*-sleeptimes` dirs, excluding the
nanosleep runs) — writes `figures/foil.pdf` (FoilLCT) and `figures/khi.pdf`
(KelvinHelmholtz), and prints the per-grid timing statistics together with
the metadata of both figures.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("pdf")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from run_logs import FREE_DELAY, MALLOC_DELAY, parse_logs
from scipy.stats import kruskal

HARDWARE = {
    "hal": "A30",
    "hemera": "A100",
    "hemera-a100": "A100",
    "hemera-v100": "V100",
    "lumi": "MI250X (1 GCD)",
    "jedi": "GH200",
    # the (0, 0) baseline runs of the delay-combination sweeps, read as well
    # (the `hal-sleeptimes-nanosleep` dir is deliberately left out):
    "hal-sleeptimes": "A30",
    "rosi-sleeptimes": "V100",
}
HARDWARE_ORDER = ["V100", "A100", "A30", "GH200", "MI250X (1 GCD)"]
ALGORITHM_ORDER = ["ScatterAlloc", "FlatterScatter", "Gallatin"]
MEM_LABEL = "estimated particle memory consumption in GB"
YMIN, YMAX = 0.9, 1.1

OUTPUT = Path("output")
SIZE_OF_PARTICLE = 30
TYPICAL_PARTICLES_PER_CELL = 25
NUMBER_OF_SPECIES = 2

MEMORY_PER_CELL = SIZE_OF_PARTICLE * TYPICAL_PARTICLES_PER_CELL * NUMBER_OF_SPECIES

REFERENCE_ALGORITHM = "ScatterAlloc"


def _no_delay_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the no-delay baseline runs (malloc and free delay both 0).

    Runs parsed from the delay-combination sweep carry `malloc_sleeptime` /
    `free_sleeptime`; only their (0, 0) baseline is a no-delay run. The
    old-layout runs carry no delay columns, so they all pass through.

    Returns:
        pd.DataFrame: the no-delay subset of the runs.

    """
    for col in (MALLOC_DELAY, FREE_DELAY):
        if col in df.columns:
            df = df[df[col] == 0]
    return df


def read_data(cluster: Path) -> pd.DataFrame:
    """Parse a cluster's run logs into its no-delay per-run records.

    Returns:
        pd.DataFrame: one record per no-delay run, tagged with the cluster's hardware.

    """
    files = [f for f in sorted(cluster.glob("*")) if f.is_file()]
    if not files:
        return pd.DataFrame()
    df = parse_logs(files)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "setup": "benchmark",
            "x": "grid_x",
            "y": "grid_y",
            "z": "grid_z",
            "runtime in s": "runtime in seconds",
        }
    )
    df = _no_delay_runs(df)
    drop = [c for c in ("name", "malloc_sleeptime", "free_sleeptime", "configuration") if c in df.columns]
    return df.drop(columns=drop).assign(hardware=HARDWARE[cluster.name])


def read_timings() -> pd.DataFrame:
    """Read all clusters into a long (hardware, benchmark, grid, algorithm) frame.

    Returns:
        pd.DataFrame: one row per parsed run; `run_id` numbers the runs of a
        (hardware, benchmark, grid, algorithm) group in file order.

    """
    clusters = [c for c in sorted(OUTPUT.glob("*")) if c.is_dir() and c.name in HARDWARE]
    frames = [read_data(cluster) for cluster in clusters]
    frames = [f for f in frames if not f.empty]
    columns = ["hardware", "benchmark", "grid_x", "grid_y", "grid_z", "algorithm", "run_id", "runtime in seconds"]
    if not frames:
        return pd.DataFrame(columns=columns)
    timings = pd.concat(frames, ignore_index=True)
    timings["run_id"] = timings.groupby(
        ["hardware", "benchmark", "algorithm", "grid_x", "grid_y", "grid_z"], dropna=False, sort=False
    ).cumcount()
    return timings[columns]


def memory(grid_sizes: pd.DataFrame) -> np.ndarray:
    """Estimate particle memory in GB for each grid size.

    Returns:
        np.ndarray: the estimated particle memory in GB for each grid size.

    """
    return np.ceil(np.prod(grid_sizes, axis=1) * MEMORY_PER_CELL / 1024**3).astype(int)


def statistical_timings(timings: pd.DataFrame) -> pd.DataFrame:
    """Describe the runtime of every (hardware, grid, algorithm) group.

    Returns:
        pd.DataFrame: the runtime description of every (hardware, grid, algorithm) group.

    """
    return (
        timings.set_index(
            [
                "hardware",
                "benchmark",
                "grid_x",
                "grid_y",
                "grid_z",
                "algorithm",
                "run_id",
            ]
        )
        .unstack("run_id")
        .apply(pd.Series.describe, axis=1)
    )


def compute_baselines(timings: pd.DataFrame) -> pd.Series:
    """Compute the median ScatterAlloc runtime per (hardware, memory) group.

    Returns:
        pd.Series: the median ScatterAlloc runtime per (hardware, memory) group.

    """
    return timings.groupby(["hardware", MEM_LABEL]).apply(
        lambda x: np.percentile(
            x.set_index("algorithm", append=False, drop=True)["runtime in seconds"][REFERENCE_ALGORITHM],
            50,
        )
    )


def print_results(results: pd.DataFrame | pd.Series, name: str) -> None:
    """Print a results frame under a banner."""
    print("+++++++++++++++++++++++++++++++++++")
    print(name)
    print("+++++++++++++++++++++++++++++++++++")
    print(results)
    print()


def plot_foil(timings: pd.DataFrame) -> pd.Series:
    """Plot the FoilLCT bar chart and return its significance metadata.

    Returns:
        pd.Series: the Kruskal significance metadata of the chart.

    """
    plt.figure()
    ax = sns.barplot(
        timings,
        x="hardware",
        y="runtime in seconds",
        estimator="median",
        hue="algorithm",
        errorbar="pi",
        order=HARDWARE_ORDER,
        hue_order=ALGORITHM_ORDER,
    )
    plt.tight_layout()
    ax.get_figure().savefig("figures/foil.pdf")
    return compute_significance(
        timings.assign(**{MEM_LABEL: 1})[
            (timings["algorithm"] == "FlatterScatter") + (timings["algorithm"] == "ScatterAlloc")
        ],
        "runtime in seconds",
    ).droplevel(MEM_LABEL)


def outlier_mask(timings: pd.Series, safety_factor: float = 1.5) -> pd.Series:
    """Flag values outside the Tukey fences as outliers.

    Returns:
        pd.Series: a boolean mask of the values outside the Tukey fences.

    """
    # according to Tukey's criterion
    perc_25, perc_75 = timings.describe()[["25%", "75%"]]
    interval = (
        perc_25 - safety_factor * (perc_75 - perc_25),
        perc_75 + safety_factor * (perc_75 - perc_25),
    )
    return (timings < interval[0]) + (timings > interval[1])


def compute_significance(timings: pd.DataFrame, name: str) -> pd.Series:
    """Compute the Kruskal p-value of `name` across algorithms per group.

    Returns:
        pd.Series: the Kruskal p-value of `name` across algorithms per group.

    """

    def pvalue(frame: pd.DataFrame) -> float:
        samples = frame.groupby("algorithm")[name].agg(list).to_numpy()
        return kruskal(*samples, nan_policy="omit").pvalue

    return timings.groupby(["hardware", MEM_LABEL]).apply(pvalue, include_groups=False)


def plot_khi(timings: pd.DataFrame) -> pd.DataFrame:
    """Plot the KelvinHelmholtz violin chart and return its metadata.

    Returns:
        pd.DataFrame: the chart's metadata (reference runtime, outliers,
        kruskal p-value, relative runtime).

    """
    timings = (
        timings.assign(**{MEM_LABEL: memory(timings[["grid_x", "grid_y", "grid_z"]])})
        .reset_index(drop=True)
        .drop(["grid_x", "grid_y", "grid_z"], axis=1)
    )
    timings["outlier"] = timings.groupby(["hardware", "algorithm", MEM_LABEL])["runtime in seconds"].transform(
        outlier_mask
    )
    baselines = compute_baselines(timings)
    timings["relative runtime"] = timings["runtime in seconds"] / timings.set_index(["hardware", MEM_LABEL]).index.map(
        baselines
    )
    ax = sns.catplot(
        timings,
        kind="violin",
        x="hardware",
        y="relative runtime",
        hue="algorithm",
        col=MEM_LABEL,
        sharey=True,
        legend_out=False,
        split=False,
        hue_order=ALGORITHM_ORDER,
        order=HARDWARE_ORDER,
        orient="v",
    )
    ax.refline(y=1)
    ax.set(ylim=(YMIN, YMAX))
    plt.tight_layout()
    ax.savefig("figures/khi.pdf")
    flatter_vs_scatter = timings[(timings["algorithm"] == "FlatterScatter") + (timings["algorithm"] == "ScatterAlloc")]
    return pd.concat(
        [
            baselines,
            flatter_vs_scatter.groupby(["hardware", MEM_LABEL]).sum()["outlier"],
            compute_significance(flatter_vs_scatter, "relative runtime"),
            flatter_vs_scatter[flatter_vs_scatter["algorithm"] == "FlatterScatter"]
            .groupby(["hardware", MEM_LABEL])["relative runtime"]
            .median(),
        ],
        keys=[
            "reference runtime in seconds",
            "outliers",
            "kruskal p-value",
            "relative runtime",
        ],
        axis=1,
    )


def main() -> None:
    """Read the timings, draw both figures and print the statistics."""
    timings = read_timings()

    stats = statistical_timings(timings)
    foil_metadata = plot_foil(timings[timings["benchmark"] == "FoilLCT"].drop(columns="benchmark"))
    khi_metadata = plot_khi(timings[timings["benchmark"] == "KelvinHelmholtz"].drop(columns="benchmark"))

    print_results(stats, "Timings")
    print_results(foil_metadata, "Foil Metadata")
    print_results(khi_metadata, "KHI Metadata")


if __name__ == "__main__":
    main()
