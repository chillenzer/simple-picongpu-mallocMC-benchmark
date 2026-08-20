from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CD_CMD = "cd build/"
LOG_PATHS = (Path("output") / "hal-sleeptimes").glob("run_*")


def parse_setup(log_content: str):
    relevant_line = log_content.split("\n", maxsplit=1)[0]
    return {
        "setup": relevant_line.split("/")[0],
        "algorithm": relevant_line.split("/")[1].split("-")[0],
        "sleeptime": int(relevant_line.split("/")[1].split("-")[1][len("sleep") :]),
    }


def parse_grid(log_content: str):
    return {
        key: int(val)
        for key, val in zip(
            ("x", "y", "z"),
            log_content.split("bin/picongpu")[1].split("\n")[0].split("-g")[1].split("-")[0].strip().split(" "),
        )
    }


def parse_simulation_time(log_content: str):
    return {"runtime in s": float(log_content.split("calculation ")[1].split("\n")[0].split("=")[1][: -len("sec")])}


def parse_run(log_content: str):
    return parse_setup(log_content) | parse_grid(log_content) | parse_simulation_time(log_content)


def parse_log(log_path: Path):
    with log_path.open("r") as file:
        return map(parse_run, file.read().split(CD_CMD)[1:])


def run_to_df(run: dict):
    return pd.DataFrame(run["runs"]).assign(name=run["name"])


def runs_to_df(runs: Iterable[dict]):
    return pd.concat(map(run_to_df, runs))


def parse_logs(log_paths: Iterable[Path]):
    return runs_to_df({"name": p, "runs": parse_log(p)} for p in log_paths)


def simple_statistics(full_results: pd.DataFrame):
    return full_results.groupby(list(set(full_results.columns) - {"runtime in s", "name"})).describe()


def simple_plot(simple_results: pd.DataFrame):
    x = simple_results.reset_index("sleeptime", drop=False)["sleeptime"].to_numpy()
    ye_min, y, ye_max = simple_results["runtime in s"][["25%", "50%", "75%"]].to_numpy().T
    fig, ax = plt.subplots(1, 1)
    (line,) = ax.plot(x, y, linestyle="-", alpha=0.3)
    ax.errorbar(x, y, yerr=(y - ye_min, ye_max - y), linestyle="none", marker="o", color=line.get_color())
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig


def main(log_paths: Iterable[PathLike]):
    full_results = parse_logs(map(Path, log_paths))
    print(full_results)
    simple_results = simple_statistics(full_results)
    print(simple_results)
    _ = simple_plot(simple_results)
    plt.show()


if __name__ == "__main__":
    main(LOG_PATHS)
