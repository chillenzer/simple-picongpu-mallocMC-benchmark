from pathlib import Path
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import parse
from functools import reduce
from itertools import cycle
import seaborn as sns
from scipy.stats import kruskal


def fresh_markers():
    return cycle(("o", "s", "v", "p", "^", "8", ">", "<"))


def fresh_colours():
    return cycle(
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )


MARKER = fresh_markers()
COLOR = fresh_colours()
HARDWARE = {
    "hal": "NVIDIA A30",
    "hemera": "NVIDIA A100",
    "lumi": "AMD MI250X (1 GCD)",
    "jedi": "NVIDIA GH200",
}
MEM_LABEL = "estimated particle memory consumption in GB"


def generate_new_style():
    return {"marker": next(MARKER), "color": next(COLOR)}


STYLE = {
    algorithm: generate_new_style()
    for algorithm in ["FlatterScatter", "Gallatin", "ScatterAlloc"]
}
OUTPUT = Path("output")
SIZE_OF_PARTICLE = 30
TYPICAL_PARTICLES_PER_CELL = 25
NUMBER_OF_SPECIES = 2

MEMORY_PER_CELL = SIZE_OF_PARTICLE * TYPICAL_PARTICLES_PER_CELL * NUMBER_OF_SPECIES

REFERENCE_ALGORITHM = "ScatterAlloc"


def parse_header(header):
    parsed = parse.parse(
        "Running example: {example}\nUsing algorithm: {algorithm}",
        header.strip("=").strip(),
    )
    if parsed is None:
        return None
    return parsed["example"], parsed["algorithm"]


def parse_tuple(text):
    return text.strip()[0], tuple(map(int, text.strip()[1:].strip().split(" ")))


def parse_log(log):
    results = {}
    for text in log.split("+ bin/picongpu ")[1:]:
        flags = parse.parse(
            "-{first:tuple} -{second:tuple} --periodic {periodic} -s {steps:d}{:s}{}",
            text,
            {"tuple": parse_tuple},
        )
        runtime = parse.findall(
            "calculation  simulation time:{}= {seconds:f} sec", text
        )
        key = flags["first"][1] if flags["first"][0] == "g" else flags["second"][1]
        if len(key) == 2:
            key = key + (np.nan,)
        try:
            results[key] = next(runtime)["seconds"]
        except StopIteration:
            results[key] = np.nan
    results = pd.Series(results)
    results.index.names = ("grid_x", "grid_y", "grid_z")
    return results


def pairs(iterable):
    try:
        first = next(iterable)
    except TypeError:
        iterable = iter(iterable)
        try:
            first = next(iterable)
        except StopIteration:
            return

    while True:
        try:
            second = next(iterable)
            yield first, second
            first = next(iterable)
        except StopIteration:
            return StopIteration()


def parse_full(file):
    with file.open("r") as f:
        text = f.read()
    return pd.DataFrame(
        {
            key: parse_log(log)
            for header, log in pairs(text.split("\n==============================\n"))
            if (key := parse_header(header)) is not None
        }
    )


def read_data(cluster):
    files = list(cluster.glob("*"))
    return pd.concat([parse_full(file) for file in files], axis=1)


def read_timings():
    clusters = list(OUTPUT.glob("*"))
    timings = pd.concat(
        [read_data(cluster) for cluster in clusters],
        axis=1,
        keys=[HARDWARE[cluster.name] for cluster in clusters],
    )
    names = ["hardware", "benchmark", "algorithm"]
    timings.columns.names = names
    timings = timings.T.reset_index(drop=False).set_index(names, append=True).T
    timings = (
        timings.stack(names)
        .reorder_levels(
            ["hardware", "benchmark", "grid_x", "grid_y", "grid_z", "algorithm"]
        )
        .sort_index()
    )
    timings.columns.names = ["run_id"]
    return (
        timings.stack()
        .reset_index(drop=False)
        .rename({0: "runtime in seconds"}, axis=1)
        .set_index("benchmark", drop=True)
    )


def memory(grid_sizes):
    return np.ceil(np.prod(grid_sizes, axis=1) * MEMORY_PER_CELL / 1024**3).astype(int)


def statistical_timings(timings):
    return (
        timings.reset_index(drop=False)
        .set_index(
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


def compute_baselines(timings):
    return timings.groupby(["hardware", MEM_LABEL], axis=0).apply(
        lambda x: np.percentile(
            x.set_index("algorithm", append=False, drop=True)["runtime in seconds"][
                "ScatterAlloc"
            ],
            50,
        )
    )


def print_results(results, name):
    print("+++++++++++++++++++++++++++++++++++")
    print(name)
    print("+++++++++++++++++++++++++++++++++++")
    print(results)
    print()


def plot_foil(timings):
    plt.figure()
    ax = sns.barplot(
        timings, x="hardware", y="runtime in seconds", hue="algorithm", errorbar="pi"
    )
    ax.get_figure().savefig("figures/foil.pdf")
    return compute_significance(
        timings.assign(**{MEM_LABEL: 1}), "runtime in seconds"
    ).droplevel(MEM_LABEL)


def plot_khi(timings):
    plt.figure()
    timings = (
        timings.assign(**{MEM_LABEL: memory(timings[["grid_x", "grid_y", "grid_z"]])})
        .reset_index(drop=True)
        .drop(["grid_x", "grid_y", "grid_z"], axis=1)
    )
    baselines = compute_baselines(timings)
    timings["relative runtime"] = (
        timings.reset_index(drop=False)
        .set_index(
            baselines.index.names + ["algorithm", "run_id"], append=False, drop=True
        )
        .unstack(["algorithm", "run_id"])
        .div(baselines, axis=0)["runtime in seconds"]
        .stack(["algorithm", "run_id"])
        .reset_index(drop=True)
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
    )
    ax.refline(y=1)
    ax.set(ylim=(0.95, 1.05))
    ax.savefig("figures/khi.pdf")
    metadata = pd.concat(
        [baselines, compute_significance(timings, "relative runtime")],
        keys=["reference runtime in seconds", "kruskal p-value"],
        axis=1,
    )
    return metadata


def compute_significance(timings, name):
    return (
        timings.set_index(["algorithm", "run_id"])
        .groupby(["hardware", MEM_LABEL])
        .apply(
            lambda x: kruskal(
                *x[[]].unstack("run_id").to_numpy(),
                nan_policy="omit",
            ).pvalue
        )
    )


def main():
    timings = read_timings()
    stats = statistical_timings(timings)
    print_results(stats, "Timings")
    foil_metadata = plot_foil(timings.loc(axis=0)["FoilLCT"])
    print(foil_metadata)
    khi_metadata = plot_khi(timings.loc(axis=0)["KelvinHelmholtz"])
    print(khi_metadata)


if __name__ == "__main__":
    main()
