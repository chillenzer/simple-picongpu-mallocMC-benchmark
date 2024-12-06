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


def memory(results):
    return np.prod(results.index.to_frame().values, axis=1) * MEMORY_PER_CELL / 1024**3


def plot_khi(results):
    fig, ax = plt.subplots()
    algorithms = sorted(results.droplevel(1, axis=1).columns.unique())
    devices = sorted(results.droplevel([1, 2, 3], axis=0).index.unique())
    scaling_factor = 3
    for device, fillstyle in zip(
        devices, ["full", "none", "left", "right", "bottom", "top"]
    ):
        for algorithm in algorithms:
            try:
                tmp = results.loc(axis=0)[device].loc(axis=1)[algorithm]
            except KeyError:
                continue
            if not tmp["mean"].isna().all():
                x = memory(tmp)
                arg = np.argsort(x)
                x = x[arg] / scaling_factor
                y = tmp["50%"].to_numpy()[arg]
                yerr = np.abs(tmp[["25%", "75%"]].to_numpy()[arg] - y.reshape(-1, 1)).T
                errorbar(
                    ax,
                    x[x >= 3 / scaling_factor - 0.1],
                    y[x >= 3 / scaling_factor - 0.1],
                    yerr[:, x >= 3 / scaling_factor - 0.1],
                    label=f"{device}: {algorithm}",
                    fillstyle=fillstyle,
                    capsize=3,
                    **STYLE[algorithm],
                )
    ax.axhline(1, color=STYLE["ScatterAlloc"]["color"], label="ScatterAlloc")

    ax.set_xscale("log", base=2)

    def adjust_for_scaling_factor_formatter(x, pos):
        # return rf"${scaling_factor} \times 2^" "{" f"{int(np.round(np.log2(x)))}" "}$"
        return f"{int(scaling_factor * x)}"

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
    tmp = (
        results.droplevel([1, 2], axis=0)
        .loc(axis=1)[:, ["50%", "25%", "75%"]]
        .stack(0, future_stack=True)
        .unstack(1)
    )
    values = tmp.loc(axis=1)[["50%"]]
    errors = tmp.loc(axis=1)[["25%", "75%"]]
    errors.loc(axis=1)["25%"] -= values.to_numpy()
    errors.loc(axis=1)["75%"] -= values.to_numpy()

    ax = values.plot.bar(
        y="50%",
        yerr=errors,
        color={key: val["color"] for key, val in STYLE.items()},
    )
    ax.set_ylabel("Main loop runtime in s")
    ax.tick_params(axis="x", labelrotation=0)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig("figures/foil.pdf")


def statistical_timings():
    return dict(
        map(
            lambda x: (
                x[0],
                x[1].rename(
                    {
                        "hal": "NVIDIA A30",
                        "hemera": "NVIDIA A100",
                        "lumi": "AMD MI250X",
                        "jedi": "NVIDIA GH200",
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


def divide_dicts(d1, d2):
    return {key: d1[key] / d2[key] for key in set(d1.keys()).intersection(d2.keys())}


def compute_speedup(timings, ref_algorithm):
    ref = {
        example: times
        for (example, algo), times in timings.items()
        if algo == ref_algorithm
    }
    return {
        key: divide_dicts(ref[key[0]], val)
        for key, val in timings.items()
        if key[1] != ref_algorithm and key[0] in ref
    }


def read_speedups(cluster):
    files = cluster.glob("*")
    tmp = [compute_speedup(parse_full(file), REFERENCE_ALGORITHM) for file in files]
    algorithms = reduce(set.union, map(dict.keys, tmp), set())
    data = {
        key: (pd.concat(frames, axis=1).T.describe().T)
        for key in algorithms
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


def statistical_speedups():
    return dict(
        map(
            lambda x: (
                x[0],
                x[1].rename(
                    {
                        "hal": "NVIDIA A30",
                        "hemera": "NVIDIA A100",
                        "lumi": "AMD MI250X",
                        "jedi": "NVIDIA GH200",
                    },
                    axis=0,
                ),
            ),
            make_dict_of_frames(
                (key, cluster.name, val)
                for cluster in OUTPUT.glob("*")
                for key, val in read_speedups(cluster).items()
            ).items(),
        )
    )


def print_results(results, name):
    print("+++++++++++++++++++++++++++++++++++")
    print(name)
    print("+++++++++++++++++++++++++++++++++++")
    print()
    for key, val in results.items():
        print(key)
        print("===================================")
        print(val)
        print()


def main():
    timings = statistical_timings()
    print_results(timings, "Timings")
    speedups = statistical_speedups()
    print_results(speedups, "Speedups")
    plot_khi(speedups["khi"])
    plot_foil(timings["foil"])


if __name__ == "__main__":
    main()
