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


def parse_header(header):
    parsed = parse.parse(
        "Running example: {example}\nUsing algorithm: {algorithm}", header.strip()
    )
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
        parse_header(header): parse_log(log)
        for header, log in pairs(text.split("==============================")[1:])
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
        key: (
            pd.concat(
                map(
                    lambda x: pd.Series(x.get(key, None)).to_frame().astype(float), tmp
                ),
                axis=1,
            )
            .T.describe()
            .T
        )
        for key in reduce(set.union, map(dict.keys, tmp), set())
    }
    khi = extract(data, "KelvinHelmholtz")
    foil = extract(data, "FoilLCT")
    return {"khi": khi, "foil": foil}


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
                        filter(lambda x: x[0] == key, iterable),
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
        cluster.assign(volume=np.prod(cluster.index.to_frame().values, axis=1))
        .set_index("volume", drop=True)
        .loc(axis=1)[:, ["mean", "std"]]
    )
    normalised = cluster.loc(axis=1)["ScatterAlloc", ["mean"]].to_numpy() / cluster - 1

    # linear error propagation
    normalised.loc(axis=1)[:, "std"] = (
        cluster.loc(axis=1)["ScatterAlloc", ["std"]].to_numpy()
        / cluster.loc(axis=1)[:, "mean"].to_numpy()
        + cluster.loc(axis=1)["ScatterAlloc", ["mean"]].to_numpy()
        / cluster.loc(axis=1)[:, "mean"].to_numpy() ** 2
        * cluster.loc(axis=1)[:, "std"].to_numpy()
    )
    return normalised


def plot_khi(results):
    fig, ax = plt.subplots()
    algorithms = sorted(results.droplevel(1, axis=1).columns.unique())
    for cluster, fillstyle in zip(["hal", "hemera"], ["full", "none"]):
        normalised = normalise_cluster(results, cluster)
        for i, algorithm in enumerate(algorithms):
            tmp = normalised.loc(axis=1)[algorithm].reset_index(drop=False)
            errorbar(
                ax,
                tmp["volume"],
                tmp["mean"],
                tmp["std"],
                label=f"{cluster}: {algorithm}",
                fillstyle=fillstyle,
                capsize=3,
                **STYLE[algorithm],
            )
    ax.set_xscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Speedup to ScatterAlloc")
    ax.set_xlabel("Grid volume in number of cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/khi.pdf")


def plot_foil(results):
    ax = (
        results.droplevel([1, 2], axis=0)
        .rename({"hal": "NVIDIA A30", "hemera": "NVIDIA A100"}, axis=0)
        .loc(axis=1)[:, ["mean", "std"]]
        .stack(0)
        .unstack(1)
        .plot.bar(y="mean", yerr="std")
    )
    ax.set_ylabel("Main loop runtime in s")
    ax.tick_params(axis="x", labelrotation=0)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig("figures/foil.pdf")


def main():
    results = make_dict_of_frames(
        (key, cluster.name, val)
        for cluster in OUTPUT.glob("*")
        for key, val in read_data(cluster).items()
    )
    plot_khi(results["khi"])
    plot_foil(results["foil"])


if __name__ == "__main__":
    main()
