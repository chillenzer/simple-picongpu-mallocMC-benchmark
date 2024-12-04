from pathlib import Path
import pandas as pd
import numpy as np
import parse

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
    return pd.concat([khi[key] for key in keys], axis=1, keys=keys)


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
        for key in tmp[0]
    }
    khi = extract(data, "KelvinHelmholtz")
    foil = extract(data, "FoilLCT")
    return khi, foil


def main():
    clusters = OUTPUT.glob("*")
    results = {cluster.name: read_data(cluster) for cluster in clusters}
    print(results)
    breakpoint()


if __name__ == "__main__":
    main()
