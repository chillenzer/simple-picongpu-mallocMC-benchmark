"""Comparison figure of the performance model's candidate fade shapes.

SPDX-FileCopyrightText: 2024-2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
SPDX-License-Identifier: MIT

Reads `output/results.h5` (the output of `compute_results.py`) and draws one
figure (`figures/fade-models.pdf`) that showcases the candidate fade shapes of
the performance model and the measured result of choosing the exponential (see
qa-fade-term.md, Q3). The model is

    T = W + N_m*m + N_f*f + A_m*g(m/m0) + A_f*g(f/f0),

where `g` is one of the candidate fade shapes in `performance_model.FADE_SHAPES`
(each normalised so that g(0) = 1 and g(u -> inf) = 0). The four panels:

  (a) the candidate fade shapes g(u) against the reduced delay u = s/s0, so
      the difference between the models -- and the tail each claims beyond the
      data -- is visible at a glance;
  (b) the malloc arm of the focus group: the measured data and the fitted
      curve of every candidate, showing that the shapes are hard to tell apart
      within the measured delay range;
  (c) the per-group shape effect: the delta-SSR of every candidate against a
      fresh hyperbola refit, in units of the residual variance, one dot per
      (group, candidate);
  (d) the mean shape effect per candidate over all groups: the exponential (E)
      and the original hyperbola (H) are the two best shapes and are
      statistically tied, while the Lorentzian (L) and truncated (T) shapes are
      clearly rejected; E is chosen as the model for its finite total absorbed
      cost (integral = A*s) and its gauge-invariant scale s.

Each candidate is re-fit on every group on its own terms: a log grid over both
fade scales seeds a bounded curve_fit (a wider scale window than the stored-
model fit, so every candidate reaches its own optimum). The comparison is
model-agnostic: it does not depend on which fade shape is stored as the
default. Run: `python3 analysis/plot_fade_models.py`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import performance_model
from performance_model import BEST_FADE, EPS_S, FADE_SHAPES
from results_io import RESULTS, RUN_TIME, load_results, read_table
from run_logs import FREE_DELAY, GROUP_KEYS, MALLOC_DELAY
from scipy.optimize import OptimizeWarning, curve_fit

mpl.use("pdf")

FIGURES = Path("figures")

# The candidate fade shapes, in display order. The exponential is the selected
# best model (performance_model.BEST_FADE) and is highlighted throughout.
CANDIDATES = ("hyperbola", "exponential", "lorentzian", "truncated", "quadratic")
LABELS = {
    "hyperbola": "H  hyperbola  A s0/(s+s0)",
    "exponential": "E  exponential  A e^(-s/s0)",
    "lorentzian": "L  Lorentzian  A s0^2/(s^2+s0^2)",
    "truncated": "T  truncated  A max(1-s/s0, 0)",
    "quadratic": "Q  quadratic  A max(1-s/s0, 0)^2",
}
# One colour per candidate; the best model (exponential) is drawn in red and
# heavier than the others.
COLORS = {
    "hyperbola": "0.45",
    "exponential": "tab:red",
    "lorentzian": "tab:blue",
    "truncated": "tab:green",
    "quadratic": "tab:purple",
}
# The panel (b) focus group, chosen for a wide malloc-delay range; the first
# two-operation group with the largest malloc delay is used when absent.
FOCUS = ("hal", "KelvinHelmholtz", "FlatterScatter", 256.0, 128.0, 128.0)


def _group_entry(key: tuple, frame: pd.DataFrame) -> dict | None:
    """Build one group's fitting inputs from its runs.

    Args:
        key: the (machine, setup, algorithm, x, y, z) group key.
        frame: the group's runs.

    Returns:
        dict | None: the name, the key, the (m, f) delay arrays in seconds,
        and the runtime array in seconds; None when the group has fewer than
        3 usable points.

    """
    machine, setup, algorithm, x, y, z = key
    grp = frame.dropna(subset=[MALLOC_DELAY, FREE_DELAY, RUN_TIME])
    if grp.shape[0] < 3:
        return None
    m = grp[MALLOC_DELAY].to_numpy(float) * 1e-9
    f = grp[FREE_DELAY].to_numpy(float) * 1e-9
    t = grp[RUN_TIME].to_numpy(float)
    z_label = "" if pd.isna(z) else f"x{int(z)}"
    name = f"{machine[:4]} {setup[:4]} {algorithm[:4]} {int(x)}x{int(y)}{z_label}"
    return {"key": key, "name": name, "m": m, "f": f, "t": t}


def _is_two_d(entry: dict) -> bool:
    """Whether a group varies both delays (a two-operation fit).

    Args:
        entry: the group's fitting inputs.

    Returns:
        bool: True when both the malloc and the free delay take two or more
        distinct values.

    """
    return np.unique(entry["m"]).size >= 2 and np.unique(entry["f"]).size >= 2


def _scale_window(s: np.ndarray) -> tuple[float, float]:
    """Compute the wide fade-scale search window of one delay arm (seconds).

    The comparison fits each candidate over a scale range an order of
    magnitude wider than the stored-model fit, so every candidate reaches its
    own optimum and the shapes are compared on equal terms (as in Q3).

    Args:
        s: the arm's delays in seconds.

    Returns:
        tuple: (the grid lower bound, the search upper bound) in seconds.

    """
    s_min_pos = s[s > 0].min() if np.any(s > 0) else s.max()
    lo = 1e-7
    hi = 10.0 * max(0.5 * (float(s.max()) - float(s.min())), 1.5 * 0.05 * s_min_pos, EPS_S)
    return lo, hi


def _grid_guess_2arm(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    m: np.ndarray, f: np.ndarray, t: np.ndarray, g: Callable, lo_m: float, hi_m: float, lo_f: float, hi_f: float
) -> tuple[np.ndarray, float, float]:
    """Grid-search the fade scales of the two-operation model for one candidate.

    For each fixed (s_m, s_f) the model is linear in (W, N_m, N_f, A_m, A_f),
    so one least-squares call per cell.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        g: the candidate's fade shape.
        lo_m: the malloc-scale lower bound (s).
        hi_m: the malloc-scale upper bound (s).
        lo_f: the free-scale lower bound (s).
        hi_f: the free-scale upper bound (s).

    Returns:
        tuple: the best cell's least-squares solution and its (s_m, s_f).

    """
    best = None
    for sm in np.logspace(np.log10(lo_m), np.log10(hi_m), 15):
        col_m = g(m / sm)
        for sf in np.logspace(np.log10(lo_f), np.log10(hi_f), 15):
            X = np.vstack([np.ones_like(m), m, f, col_m, g(f / sf)]).T
            sol, *_ = np.linalg.lstsq(X, t, rcond=None)
            ssr = float(np.sum(np.square(t - X @ sol)))
            if best is None or ssr < best[0]:
                best = (ssr, sol, float(sm), float(sf))
    return best[1], best[2], best[3]


def _polish(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    m: np.ndarray, f: np.ndarray, t: np.ndarray, g: Callable, p0: list[float], bounds: tuple
) -> tuple | None:
    """Polish the grid guess with a bounded curve_fit of all seven parameters.

    Args:
        m: the malloc delays in seconds.
        f: the free delays in seconds.
        t: the runtimes in seconds.
        g: the candidate's fade shape.
        p0: the initial parameter vector.
        bounds: the (lower, upper) parameter bounds.

    Returns:
        tuple | None: (the residual sum of squares, the fitted parameter
        vector), or None when the fit fails to converge.

    """

    def model(x: tuple, W: float, Nm: float, Nf: float, Am: float, sm: float, Af: float, sf: float) -> np.ndarray:  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        return W + Nm * x[0] + Nf * x[1] + Am * g(x[0] / sm) + Af * g(x[1] / sf)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _pcov = curve_fit(model, (m, f), t, p0=p0, bounds=bounds, maxfev=40000)
    except RuntimeError, ValueError:
        return None
    params = np.asarray(popt, dtype=float)
    pred = model((m, f), *params)
    return float(np.sum(np.square(t - pred))), params


def _fit_candidate(entry: dict, fade: str) -> tuple[float, np.ndarray] | None:
    """Fit one candidate fade shape to one group on its own terms.

    Args:
        entry: the group's fitting inputs (see `_group_entry`).
        fade: the candidate's fade shape name.

    Returns:
        tuple | None: (the residual sum of squares, the fitted parameter
        vector (W, N_m, N_f, A_m, s_m, A_f, s_f)), or None when the fit
        fails to converge.

    """
    m, f, t = entry["m"], entry["f"], entry["t"]
    g = FADE_SHAPES[fade]
    lo_m, hi_m = _scale_window(m)
    lo_f, hi_f = _scale_window(f)
    sol, sm0, sf0 = _grid_guess_2arm(m, f, t, g, lo_m, hi_m, lo_f, hi_f)
    eps = 1e-9 * max(1.0, float(np.max(np.abs(t))))
    p0 = [max(v, eps) for v in (*sol[:4], sm0, sol[4], sf0)]
    bounds = ([0.0, 0.0, 0.0, 0.0, EPS_S, 0.0, EPS_S], [np.inf, np.inf, np.inf, np.inf, hi_m, np.inf, hi_f])
    return _polish(m, f, t, g, p0, bounds)


def _fit_all(entries: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit every candidate on every group and measure the per-group shape effect.

    The shape effect of a candidate is its delta-SSR against a fresh
    hyperbola refit of the same group, in units of that refit's residual
    variance, so the stored model's drift along the flat direction is removed
    and the deltas are pure shape effects.

    Args:
        entries: the loaded groups.

    Returns:
        tuple: (the per-(group, candidate) fitted parameter vectors, the
        per-(group, candidate) shape effects in sigma^2), keyed by
        `"<group>/<candidate>"`.

    """
    fits: dict[str, np.ndarray] = {}
    deltas: dict[str, float] = {}
    for entry in entries:
        baseline = _fit_candidate(entry, "hyperbola")
        if baseline is None:
            continue
        ssr_h, _ = baseline
        dof = max(entry["t"].size - 7, 1)  # the two-operation model has 7 parameters
        for cand in CANDIDATES:
            result = _fit_candidate(entry, cand)
            if result is None:
                continue
            ssr_c, params = result
            key = f"{entry['name']}/{cand}"
            fits[key] = params
            deltas[key] = (ssr_c - ssr_h) / (ssr_h / dof)
    return fits, deltas


def _focus_entry(entries: list[dict]) -> dict:
    """Pick the panel (b) focus group.

    Args:
        entries: the loaded groups.

    Returns:
        dict: the focus group, the named `FOCUS` group when present, else the
        two-operation group with the largest malloc delay.

    """
    want = set(FOCUS)
    for entry in entries:
        key = entry["key"]
        if {key[0], key[1], key[2], key[3], key[4]} == want and (pd.isna(key[5]) or key[5] == FOCUS[5]):
            return entry
    two_d = [entry for entry in entries if _is_two_d(entry)]
    return max(two_d, key=lambda entry: float(entry["m"].max()))


def _malloc_arm(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the focus group's malloc arm (free delay 0): medians and IQR.

    Args:
        entry: the focus group's fitting inputs.

    Returns:
        tuple: the sorted malloc delays (s), the runtime median (s), and the
        lower and upper IQR bounds (s), one row per distinct delay.

    """
    m, f, t = entry["m"], entry["f"], entry["t"]
    arm = (f == 0) & (m > 0)
    m, t = m[arm], t[arm]
    order = np.argsort(m)
    m, t = m[order], t[order]
    unique = np.unique(m)
    med = np.array([np.median(t[m == v]) for v in unique])
    p25 = np.array([np.percentile(t[m == v], 25) for v in unique])
    p75 = np.array([np.percentile(t[m == v], 75) for v in unique])
    return unique, med, p25, p75


def _malloc_curve(entry: dict, params: np.ndarray, fade: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute the candidate's model curve on the focus group's malloc arm.

    Args:
        entry: the focus group's fitting inputs.
        params: the candidate's fitted parameter vector.
        fade: the candidate's fade shape name.

    Returns:
        tuple: (the log-spaced malloc-delay grid (s), the model runtime (s)).

    """
    m_min = float(entry["m"][entry["m"] > 0].min())
    grid = np.geomspace(max(m_min, 1e-5), 1.5 * float(entry["m"].max()), 200)
    held = np.zeros_like(grid)
    if params.size == 7:
        return grid, performance_model.model_2d(grid, held, params, fade=fade)
    return grid, performance_model.model_1d(grid, params[0], params[1], params[2], params[3], fade=fade)


def _panel_shapes(ax: plt.Axes) -> None:
    """Draw panel (a): the candidate fade shapes g(u) against the reduced delay.

    Args:
        ax: the axis to fill.

    """
    u = np.linspace(0.0, 6.0, 300)
    for cand in CANDIDATES:
        width = 2.4 if cand == BEST_FADE else 1.3
        alpha = 1.0 if cand == BEST_FADE else 0.8
        ax.plot(u, FADE_SHAPES[cand](u), color=COLORS[cand], linewidth=width, alpha=alpha, label=LABELS[cand])
    ax.set_xlabel("reduced delay u = s/s0")
    ax.set_ylabel("fade shape g(u) = F(s)/A")
    ax.set_title("(a) candidate fade shapes")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _panel_fit(ax: plt.Axes, entries: list[dict], fits: dict[str, np.ndarray]) -> None:
    """Draw panel (b): the focus group's malloc arm, data and candidate curves.

    Args:
        ax: the axis to fill.
        entries: the loaded groups.
        fits: the per-(group, candidate) fitted parameter vectors.

    """
    focus = _focus_entry(entries)
    m, med, p25, p75 = _malloc_arm(focus)
    ax.errorbar(
        m,
        med,
        yerr=(med - p25, p75 - med),
        linestyle="none",
        marker="o",
        markersize=4,
        color="0.25",
        label="measured (malloc arm)",
        zorder=3,
    )
    for cand in CANDIDATES:
        params = fits.get(f"{focus['name']}/{cand}")
        if params is None:
            continue
        width = 2.4 if cand == BEST_FADE else 1.2
        alpha = 1.0 if cand == BEST_FADE else 0.7
        grid, curve = _malloc_curve(focus, params, cand)
        ax.plot(grid, curve, color=COLORS[cand], linewidth=width, alpha=alpha, label=LABELS[cand])
    ax.set_xlabel("malloc delay (s)")
    ax.set_ylabel("runtime (s)")
    ax.set_title(f"(b) {focus['name']}: every shape fits the measured range")
    ax.legend(fontsize=7, loc="upper left")


def _panel_per_group(ax: plt.Axes, entries: list[dict], deltas: dict[str, float]) -> None:
    """Draw panel (c): the per-group shape effect (delta-SSR vs the fresh H refit).

    Args:
        ax: the axis to fill.
        entries: the loaded groups.
        deltas: the per-(group, candidate) shape effect in sigma^2.

    """
    for cand in CANDIDATES:
        ys, xs = [], []
        for i, entry in enumerate(entries):
            value = deltas.get(f"{entry['name']}/{cand}")
            if value is None:
                continue
            ys.append(i)
            xs.append(value)
        best = cand == BEST_FADE
        ax.scatter(
            xs,
            ys,
            s=22,
            color=COLORS[cand],
            alpha=1.0 if best else 0.75,
            edgecolors="0.3" if best else "none",
            linewidths=0.8 if best else 0.0,
            zorder=3 if best else 2,
            label=LABELS[cand],
        )
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([entry["name"] for entry in entries], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"shape effect $\Delta$SSR vs fresh H refit ($\sigma^2$)")
    ax.set_title("(c) per-group shape effect (negative = better than H)")
    ax.grid(visible=True, alpha=0.3, axis="x")
    ax.legend(fontsize=7, loc="lower right")


def _panel_ranking(ax: plt.Axes, entries: list[dict], deltas: dict[str, float]) -> None:
    """Draw panel (d): the mean shape effect per candidate over all groups.

    Args:
        ax: the axis to fill.
        entries: the loaded groups.
        deltas: the per-(group, candidate) shape effect in sigma^2.

    """
    names, means, stds, colors = [], [], [], []
    for cand in CANDIDATES:
        values = [deltas[f"{entry['name']}/{cand}"] for entry in entries if f"{entry['name']}/{cand}" in deltas]
        if not values:
            continue
        names.append(LABELS[cand].split("  ", 1)[0])
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
        colors.append(COLORS[cand])
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = [colors[i] for i in order]
    ax.barh(names, means, xerr=stds, color=colors, edgecolor="0.3", linewidth=0.5)
    for i, (mean, std) in enumerate(zip(means, stds, strict=True)):
        ax.text(mean + std + 0.1, i, f"{mean:+.2f}", va="center", fontsize=8)
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"mean shape effect vs fresh H refit ($\sigma^2$, over all groups)")
    ax.set_title("(d) aggregate ranking (lowest = best fade term)")
    ax.grid(visible=True, alpha=0.3, axis="x")


def _load_entries(runs: pd.DataFrame) -> list[dict]:
    """Group the runs into fitting inputs, as the pipeline does.

    Args:
        runs: the runs table of the results file.

    Returns:
        list: the per-group fitting inputs of the delay-injected groups.

    """
    entries: list[dict] = []
    for key, frame in runs.groupby(["machine", *GROUP_KEYS], dropna=False):
        if frame[MALLOC_DELAY].isna().all() and frame[FREE_DELAY].isna().all():
            continue  # a non-sweep machine: no delay runs
        entry = _group_entry(key, frame)
        if entry is not None:
            entries.append(entry)
    return entries


def main(*, show: bool = False, results: Path = RESULTS) -> int:
    """Draw the fade-model comparison figure from one results file.

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
    for column in ("machine", "setup", "algorithm"):
        runs[column] = [value.decode() if isinstance(value, bytes) else value for value in runs[column]]
    entries = _load_entries(runs)
    if not entries:
        print("no delay-injected runs found; nothing to draw", file=sys.stderr)
        return 0
    fits, deltas = _fit_all(entries)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), layout="constrained")
    fig.suptitle("Performance-model fade terms: the candidate family and the measured choice of the exponential")
    _panel_shapes(axes[0, 0])
    _panel_fit(axes[0, 1], entries, fits)
    _panel_per_group(axes[1, 0], entries, deltas)
    _panel_ranking(axes[1, 1], entries, deltas)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "fade-models.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Candidate fade-shape comparison figure from output/results.h5 (figures/fade-models.pdf)."
    )
    parser.add_argument("--show", action="store_true", help="display the figure in a window (blocking)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="the results file (default: %(default)s)")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    sys.exit(main(show=args.show, results=args.results))
