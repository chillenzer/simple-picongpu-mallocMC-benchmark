"""Shared access to the benchmark results HDF5 file.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

`compute_results.py` writes all of the benchmark numbers to a single HDF5
file (`output/results.h5`); `summarize_results.py` and the `plot_*.py`
figure scripts read them back. A table is stored as a group holding one
dataset per column: numeric columns are stored as float64 (int64 when the
column carries no NaN), text columns as variable-length strings, and a
missing text value is stored as ``""``. The fits' parameter vectors and
covariances are stored next to their fit row, one subgroup each under
``fits/cov/<machine>/<setup>/<algorithm>/<grid>``.

The file's top-level attributes record the provenance: the creation time,
the git commit, the source log directories, and `algorithm_order`, the
`algorithms` list of `config.json`, which all figures use as their row
order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from run_logs import FREE_DELAY, MALLOC_DELAY

# The default location of the results file; the scripts are invoked from
# the repository root, so this is relative to the working directory.
RESULTS = Path("output/results.h5")
# The runtime column of the parsed runs.
RUN_TIME = "runtime_s"
# Hardware display order of the comparison figures (GPU generation, the
# newest last).
HARDWARE_ORDER = ("V100", "A100", "A30", "GH200", "MI250X (1 GCD)")
# Particle memory model of the KelvinHelmholtz figure: bytes per particle,
# typical particles per cell, and the number of species.
SIZE_OF_PARTICLE = 30
TYPICAL_PARTICLES_PER_CELL = 25
NUMBER_OF_SPECIES = 2
MEMORY_PER_CELL = SIZE_OF_PARTICLE * TYPICAL_PARTICLES_PER_CELL * NUMBER_OF_SPECIES


def no_delay_mask(runs: pd.DataFrame) -> pd.Series:
    """Boolean mask of the baseline runs (no injected delay).

    Old-layout runs carry no delay columns at all, so every one of them is
    a baseline run; of the delay-injected runs only the (0, 0) combination
    is.

    Args:
        runs: the parsed runs table of `compute_results.py`.

    Returns:
        pd.Series: True for the baseline runs.

    """
    if MALLOC_DELAY not in runs.columns:
        return pd.Series([True] * len(runs.index), index=runs.index)
    malloc_delay = runs[MALLOC_DELAY]
    free_delay = runs[FREE_DELAY]
    return (malloc_delay.isna() & free_delay.isna()) | ((malloc_delay == 0) & (free_delay == 0))


def grid_label(x: float, y: float, z: float) -> str:
    """Compact grid label; a missing (2-D) dimension is dropped.

    Args:
        x: the x grid dimension.
        y: the y grid dimension.
        z: the z grid dimension (NaN for 2-D runs).

    Returns:
        str: the grid label, e.g. `256x1280` or `128x128x128`.

    """
    dims = []
    for dim in (x, y, z):
        if dim is None:
            continue
        value = float(dim)
        if not np.isfinite(value):
            continue
        dims.append(str(int(value)) if value.is_integer() else str(value))
    return "x".join(dims)


def scenario_key(setup: str, x: float, y: float, z: float) -> tuple:
    """Canonical key of a (setup, grid) scenario, without the algorithm.

    Grid dimensions are normalized so the same scenario matches even when a
    dimension is an int in one place and a whole-valued float in the other;
    a missing (2-D) ``z`` maps to ``None``.

    Args:
        setup: the scenario's setup name.
        x: the scenario's x grid dimension.
        y: the scenario's y grid dimension.
        z: the scenario's z grid dimension (NaN for 2-D).

    Returns:
        tuple: (setup, x, y, z) with normalized grid dimensions.

    """

    def norm(v: float | None) -> float | None:
        if v is None:
            return None
        v = float(v)
        if not np.isfinite(v):
            return None
        return int(v) if v.is_integer() else v

    return (setup, norm(x), norm(y), norm(z))


def scenario_name(setup: str, x: float, y: float, z: float) -> str:
    """Human-readable scenario name, e.g. `FoilLCT 256x1280`.

    Args:
        setup: the scenario's setup name.
        x: the scenario's x grid dimension.
        y: the scenario's y grid dimension.
        z: the scenario's z grid dimension (NaN for 2-D).

    Returns:
        str: the scenario name.

    """
    return f"{setup} {grid_label(x, y, z)}"


def scenario_filename(setup: str, x: float, y: float, z: float) -> str:
    """Filename of one scenario's runtime-stack figure.

    Args:
        setup: the scenario's setup name.
        x: the scenario's x grid dimension.
        y: the scenario's y grid dimension.
        z: the scenario's z grid dimension (NaN for 2-D).

    Returns:
        str: the PDF filename, e.g. `runtime-stack-FoilLCT-256x1280.pdf`.

    """
    return f"runtime-stack-{scenario_name(setup, x, y, z).replace(' ', '-')}.pdf"


def particle_memory_gb(x: float, y: float, z: float) -> int:
    """Estimated particle memory consumption in GB of one grid.

    Args:
        x: the x grid dimension.
        y: the y grid dimension.
        z: the z grid dimension (NaN for 2-D runs, where it contributes no
        volume).

    Returns:
        int: the estimated particle memory in GB.

    """
    dims = [v for v in (float(x), float(y), float(z)) if np.isfinite(v)]
    return int(np.ceil(np.prod(dims) * MEMORY_PER_CELL / 1024**3))


def _write_table(file: h5py.File, name: str, table: pd.DataFrame) -> None:
    group = file.create_group(name)
    # Record the column order: the default link index sorts the datasets by
    # name, so the file alone would lose the table's original order.
    group.attrs["column_order"] = ",".join(str(column) for column in table.columns)
    for column in table.columns:
        series = table[column]
        if pd.api.types.is_integer_dtype(series):
            group.create_dataset(column, data=series.to_numpy(dtype=np.int64))
        elif pd.api.types.is_numeric_dtype(series):
            group.create_dataset(column, data=series.to_numpy(dtype=np.float64))
        else:
            # A missing text value (None, NaN) is stored as the empty string.
            values = np.array(
                ["" if (value is None or pd.isna(value)) else str(value) for value in series], dtype=object
            )
            dataset = group.create_dataset(column, shape=(len(values),), dtype=h5py.special_dtype(vlen=str))
            dataset[...] = values


def _read_column(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[...]
    if values.dtype == object:
        # h5py hands variable-length strings back as bytes; decode them.
        values = np.array(
            [
                value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else ("" if value is None else str(value))
                for value in values
            ],
            dtype=object,
        )
    return values


def write_results(
    path: str | Path,
    tables: dict[str, pd.DataFrame],
    attrs: dict[str, Any] | None = None,
    fit_covs: list[tuple[tuple, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Write all result tables (and the fit covariances) to a fresh HDF5 file.

    Args:
        path: the destination file, e.g. `output/results.h5`.
        tables: the result tables, by group name.
        attrs: the file's top-level attributes.
        fit_covs: per fitted group, (key, fit_params, pcov); the key is the
        (machine, setup, algorithm, grid label) group path.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for key, value in (attrs or {}).items():
            file.attrs[key] = value
        for name, table in tables.items():
            _write_table(file, name, table)
        for key, fit_params, pcov in fit_covs or []:
            write_fit_cov(file, key, fit_params, pcov)


def load_results(path: str | Path) -> h5py.File:
    """Open a results file for reading.

    Args:
        path: the results file, e.g. `output/results.h5`.

    Returns:
        h5py.File: the opened file (a context manager).

    """
    return h5py.File(path, "r")


def read_table(file: h5py.File, name: str) -> pd.DataFrame:
    """Read one result table back into a DataFrame.

    Args:
        file: an opened results file.
        name: the table's group name.

    Returns:
        pd.DataFrame: the table's columns, in their stored order.

    Raises:
        KeyError: if the table is not in the file.

    """
    if name not in file:
        msg = f"table {name!r} not found in {file.filename}"
        raise KeyError(msg)
    group = file[name]
    data = {column: _read_column(dataset) for column, dataset in group.items() if isinstance(dataset, h5py.Dataset)}
    columns = [str(column) for column in str(group.attrs.get("column_order", "")).split(",") if column in data]
    columns += [column for column in group if column in data and column not in columns]
    return pd.DataFrame({column: data[column] for column in columns})


def read_attrs(file: h5py.File) -> dict[str, Any]:
    """Read the file's top-level attributes.

    Args:
        file: an opened results file.

    Returns:
        dict[str, Any]: the top-level attributes.

    """
    return {key: file.attrs[key] for key in file.attrs}


def algorithm_order(file: h5py.File) -> list[str]:
    """Return the figure rows' algorithm order, as recorded in the file.

    Args:
        file: an opened results file.

    Returns:
        list[str]: the algorithms, in figure row order.

    """
    return [algorithm for algorithm in str(file.attrs.get("algorithm_order", "")).split(",") if algorithm]


def write_fit_cov(file: h5py.File, key: tuple, fit_params: np.ndarray, pcov: np.ndarray) -> None:
    """Store one fitted group's parameters and covariance.

    Args:
        file: an opened results file (in write mode).
        key: the (machine, setup, algorithm, grid label) group path.
        fit_params: the fitted parameter vector.
        pcov: the parameter covariance matrix.

    """
    group = file.create_group("fits/cov/" + "/".join(map(str, key)))
    group.create_dataset("fit_params", data=np.asarray(fit_params, dtype=float))
    group.create_dataset("pcov", data=np.asarray(pcov, dtype=float))


def read_fit_covs(file: h5py.File) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Read every stored fit's parameters and covariance.

    Args:
        file: an opened results file.

    Returns:
        dict[tuple, tuple]: per (machine, setup, algorithm, grid label)
        key, the (fit_params, pcov) pair.

    """
    covs: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    root = file.get("fits/cov")
    if root is None:
        return covs

    def walk(group: h5py.Group, prefix: tuple) -> None:
        if "fit_params" in group:
            covs[prefix] = (group["fit_params"][...], group["pcov"][...])
            return
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                walk(item, (*prefix, name))

    walk(root, ())
    return covs
