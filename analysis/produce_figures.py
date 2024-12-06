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
        try:
            results[key] = next(runtime)["seconds"]
        except StopIteration:
            results[key] = np.nan
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
    return {
        key: parse_log(log)
        for header, log in pairs(text.split("\n==============================\n"))
        if (key := parse_header(header)) is not None
    }


def extract(data, example):
    khi = {key[1]: data[key] for key in filter(lambda key: key[0] == example, data)}
    keys = list(khi.keys())
    if len(keys) > 0:
        return pd.concat([khi[key] for key in keys], axis=1, keys=keys)
    return pd.DataFrame()


def read_data(cluster):
    files = cluster.glob("*")
    tmp = [parse_full(file) for file in files]
    data = {
        key: (pd.concat(frames, axis=1).T.describe().T)
        for key in reduce(set.union, map(dict.keys, tmp), set())
        if len(
            (
                frames := list(
                    filter(
                        lambda series: not series.dropna(how="all").empty,
                        map(
                            lambda x: pd.Series(x.get(key, None))
                            .to_frame()
                            .astype(float),
                            tmp,
                        ),
                    )
                )
            )
        )
        > 0
    }
    return {"khi": extract(data, "KelvinHelmholtz"), "foil": extract(data, "FoilLCT")}


def make_dict_of_frames(iterable):
    # make sure we don't exhaust an iterator
    iterable = list(iterable)
    return {
        str(key): pd.concat(
            **{
                k: v
                for k, v in zip(
                    ["objs", "keys"],
                    reduce(
                        lambda acc, new: (acc[0] + [new[2]], acc[1] + [new[1]]),
                        filter(lambda x: x[0] == key and not x[2].empty, iterable),
                        ([], []),
                    ),
                )
            }
        )
        for key in np.unique(list(map(lambda x: x[0], iterable)))
    }


def errorbar(ax, x, y, yerr, **kwargs):
    points = ax.errorbar(x, y, yerr, linestyle="none", **kwargs)
    ax.plot(x, y, alpha=0.3, color=points.lines[0].get_color())
    return points


def normalise_cluster(results, name):
    cluster = results.loc(axis=0)[name]
    cluster = (
        cluster.assign(
            memory=np.prod(cluster.index.to_frame().values, axis=1)
            * MEMORY_PER_CELL
            / 1024**3
        )
        .set_index("memory", drop=True)
        .loc(axis=1)[:, ["mean", "std"]]
    )
    normalised = cluster.loc(axis=1)["ScatterAlloc", ["mean"]].to_numpy() / cluster

    # linear error propagation
    term1 = (
        cluster.loc(axis=1)["ScatterAlloc", ["std"]].to_numpy()
        / cluster.loc(axis=1)[:, "mean"].to_numpy()
    )
    # this one stays a data frame
    term2 = (
        cluster.loc(axis=1)["ScatterAlloc", ["mean"]].to_numpy()
        / cluster.loc(axis=1)[:, "mean"].to_numpy() ** 2
        * cluster.loc(axis=1)[:, "std"]
    )

    # it's convenient to use pandas' fillna,
    # so we do multiple assignments to create separate data frames
    # to `fillna` individually
    normalised.loc(axis=1)[:, "std"] = term1
    normalised.loc(axis=1)[:, "std"] = normalised.loc(axis=1)[:, "std"].fillna(0)
    normalised.loc(axis=1)[:, "std"] += term2.fillna(0)
    return normalised


def plot_khi(results):
    print("KHI results:\n===========================")
    print(results)
    fig, ax = plt.subplots()
    algorithms = sorted(results.droplevel(1, axis=1).columns.unique())
    # This is our baseline
    algorithms.remove("ScatterAlloc")
    devices = sorted(results.droplevel([1, 2, 3], axis=0).index.unique())
    scaling_factor = 3
    for device, fillstyle in zip(
        devices, ["full", "none", "left", "right", "bottom", "top"]
    ):
        try:
            normalised = normalise_cluster(results, device)
        except KeyError:
            print(f"No valid data found for KHI on {device}!")
            continue
        for algorithm in algorithms:
            tmp = normalised.loc(axis=1)[algorithm].reset_index(drop=False)
            if not tmp["mean"].isna().all():
                errorbar(
                    ax,
                    tmp["memory"] / scaling_factor,
                    tmp["mean"],
                    tmp["std"],
                    label=f"{device}: {algorithm}",
                    fillstyle=fillstyle,
                    capsize=3,
                    **STYLE[algorithm],
                )
    ax.axhline(1, color=STYLE["ScatterAlloc"]["color"], label="ScatterAlloc")

    ax.set_xscale("log", base=2)

    def adjust_for_scaling_factor_formatter(x, pos):
        # return rf"${scaling_factor} \times 2^" "{" f"{int(np.round(np.log2(x)))}" "}$"
        return f"{scaling_factor * x}"

    # Set the custom formatter for the x-axis tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(adjust_for_scaling_factor_formatter))
    #    ax.xaxis.set_major_formatter(
    #        plt.FuncFormatter(lambda x, pos: f"2^{int(np.log2(x))}")
    #    )
    ax.set_ylabel("Speedup to ScatterAlloc")
    ax.set_xlabel("Estimated particle memory consumption in GB")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/khi.pdf")


def plot_foil(results):
    print("FoilLCT results:\n===========================")
    print(results)
    print()
    print("Speedup to ScatterAlloc:")
    clusters = results.droplevel([1, 2], axis=0).index.unique()
    print(
        pd.concat(
            [normalise_cluster(results, cluster) for cluster in clusters],
            keys=clusters,
        )
        .drop("ScatterAlloc", axis=1)
        .droplevel(1, axis=0)
    )
    ax = (
        results.droplevel([1, 2], axis=0)
        .loc(axis=1)[:, ["mean", "std"]]
        .stack(0, future_stack=True)
        .unstack(1)
        .plot.bar(
            y="mean",
            yerr="std",
            color={key: val["color"] for key, val in STYLE.items()},
        )
    )
    ax.set_ylabel("Main loop runtime in s")
    ax.tick_params(axis="x", labelrotation=0)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig("figures/foil.pdf")


def main():
    results = dict(
        map(
            lambda x: print(x)
            or (
                x[0],
                x[1].rename(
                    {
                        "hal": "NVIDIA A30",
                        "hemera": "NVIDIA A100",
                        "lumi": "AMD MI250X",
                    },
                    axis=0,
                ),
            ),
            make_dict_of_frames(
                (key, cluster.name, val)
                for cluster in OUTPUT.glob("*")
                for key, val in read_data(cluster).items()
            ).items(),
        )
    )
    plot_khi(results["khi"])
    plot_foil(results["foil"])


if __name__ == "__main__":
    main()
