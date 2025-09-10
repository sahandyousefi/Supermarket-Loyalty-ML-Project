"""Static figures for the report and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TEXT = "#191919"
MUTED = "#666666"
GRID = "#e6e6e6"
# Fixed categorical order: one hue per model family, never cycled.
FAMILY_COLORS = {"linear": "#257de0", "tree": "#f47230", "deep": "#00b894"}
SINGLE = "#257de0"


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x" if ax.get_xlabel() else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", metadata={})
    plt.close(fig)
    return path


def _family_legend(target, families, loc: str = "upper right") -> None:
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=FAMILY_COLORS[f], label=f)
        for f in FAMILY_COLORS
        if f in set(families)
    ]
    target.legend(
        handles=handles, frameon=False, fontsize=8, loc=loc, ncol=len(handles)
    )


def plot_group_means(table: pd.DataFrame, group: str, path: str | Path) -> Path:
    """Mean spend per group with bootstrap intervals."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    y = np.arange(len(table))
    ax.hlines(y, table["ci_low"], table["ci_high"], color=SINGLE, linewidth=2)
    ax.plot(table["avg_spend"], y, "o", color=SINGLE, markersize=6)
    ax.set_yticks(y, table[group].astype(str))
    ax.invert_yaxis()
    _style(ax, "Mean annual spend by loyalty tenure (95% bootstrap CI)", "spend", "")
    return _save(fig, path)


def plot_driver_shares(table: pd.DataFrame, path: str | Path) -> Path:
    """Share of explained variance attributed to each feature."""
    fig, ax = plt.subplots(figsize=(6, 3.0))
    t = table.sort_values("r2_share")
    ax.barh(t["feature"], t["r2_share"], color=SINGLE, height=0.6)
    for yv, v in enumerate(t["r2_share"]):
        ax.text(v, yv, f" {v:.3f}", va="center", fontsize=8, color=TEXT)
    _style(ax, "Shapley decomposition of R squared (LMG)", "share of R squared", "")
    return _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path: str | Path) -> Path:
    """Cross-validated RMSE with one standard deviation across folds."""
    t = table.sort_values("rmse_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(t) + 1.2))
    for i, row in t.iterrows():
        color = FAMILY_COLORS.get(row["family"], MUTED)
        ax.hlines(
            i,
            row["rmse_mean"] - row["rmse_std"],
            row["rmse_mean"] + row["rmse_std"],
            color=color,
            linewidth=2,
        )
        ax.plot(row["rmse_mean"], i, "o", color=color, markersize=6)
    ax.set_yticks(range(len(t)), t["model"])
    _style(ax, "Repeated 5-fold CV on train.csv", "RMSE (lower is better)", "")
    _family_legend(ax, t["family"])
    return _save(fig, path)


def plot_simulation_study(table: pd.DataFrame, path: str | Path) -> Path:
    """Excess RMSE over the Bayes error, one panel per scenario."""
    scenarios = list(dict.fromkeys(table["scenario"]))
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(3.0 * len(scenarios), 3.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    models = list(dict.fromkeys(table["model"]))
    for ax, sc in zip(axes, scenarios):
        part = table[table["scenario"] == sc].set_index("model").reindex(models)
        colors = [FAMILY_COLORS.get(f, MUTED) for f in part["family"]]
        ax.barh(range(len(models)), part["relative_excess"], color=colors, height=0.6)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        _style(ax, sc, "excess RMSE / Bayes RMSE", "")
    _family_legend(fig, table["family"])
    fig.suptitle(
        "Simulation study: distance from the irreducible error",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_conditional_coverage(table: pd.DataFrame, target: float, path) -> Path:
    """Coverage by predicted-spend decile for each interval method."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for color, (method, part) in zip(
        ["#257de0", "#f47230", "#00b894"], table.groupby("method", sort=False)
    ):
        ax.plot(
            part["group"],
            part["coverage"],
            "-o",
            color=color,
            linewidth=2,
            markersize=5,
            label=method,
        )
    ax.axhline(target, color=MUTED, linewidth=1, linestyle="--")
    ax.text(
        table["group"].max(), target, " target", va="bottom", fontsize=8, color=MUTED
    )
    ax.set_ylim(min(0.5, table["coverage"].min() - 0.05), 1.02)
    _style(ax, "Coverage by predicted spend decile", "", "coverage")
    ax.set_xlabel("decile of predicted spend", color=MUTED, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, path)


def plot_liability(draws: np.ndarray, plug_in: float, path: str | Path) -> Path:
    """Distribution of total reward cost from the Monte Carlo forecast."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(draws, bins=50, color=SINGLE, edgecolor="white", linewidth=0.5)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    top = ax.get_ylim()[1]
    mean = float(draws.mean())
    # Labels go on opposite sides so they stay readable when the lines are close.
    for v, label, color in [(mean, "expected", TEXT), (plug_in, "plug-in", "#f47230")]:
        ax.axvline(v, color=color, linewidth=1.5)
        left = (v <= plug_in) if label == "expected" else (v < mean)
        ax.text(
            v,
            top * 0.97,
            f"{label} " if left else f" {label}",
            fontsize=8,
            color=TEXT,
            ha="right" if left else "left",
            va="top",
        )
    ax.axvspan(lo, hi, color=SINGLE, alpha=0.08)
    _style(ax, "Total reward cost, 90% interval shaded", "reward cost", "draws")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_quantile_reliability(
    taus, shares: dict[str, np.ndarray], path: str | Path
) -> Path:
    """Observed against nominal quantile level, one line per forecaster."""
    palette = ["#257de0", "#f47230", "#00b894", "#8e6bd8", "#c0392b", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
    for (name, share), color in zip(shares.items(), palette):
        ax.plot(taus, share, marker="o", markersize=3, linewidth=1.4, color=color)
        ax.annotate(
            name,
            (taus[-1], share[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="center",
        )
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1)
    _style(ax, "Quantile reliability", "Nominal level", "Observed share below")
    return _save(fig, path)


def plot_effect_estimates(table: pd.DataFrame, path: str | Path) -> Path:
    """Forest plot of effect estimates with 95% intervals, ATE above ATT."""
    rows = table.sort_values(["estimand", "method"], ascending=[True, False])
    labels = [f"{m} ({e.upper()})" for m, e in zip(rows["method"], rows["estimand"])]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.0))
    ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.errorbar(
        rows["estimate"],
        y,
        xerr=[rows["estimate"] - rows["ci_lower"], rows["ci_upper"] - rows["estimate"]],
        fmt="o",
        color=SINGLE,
        ecolor=SINGLE,
        elinewidth=1.4,
        capsize=0,
        markersize=4,
    )
    ax.set_yticks(y, labels)
    _style(ax, "Effect of the promotion on annual spend", "Effect (95% interval)")
    return _save(fig, path)


def plot_ale(curves: list[pd.DataFrame], path: str | Path) -> Path:
    """One ALE panel per feature, with a rug of the bin edges."""
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.2 * len(curves), 3.6), squeeze=False
    )
    for ax, curve in zip(axes[0], curves):
        ax.plot(curve["value"], curve["effect"], color=SINGLE, linewidth=1.6)
        ax.plot(
            curve["value"], np.full(len(curve), curve["effect"].min()), "|", color=MUTED
        )
        name = str(curve["feature"].iloc[0])
        _style(ax, name, name, "Accumulated local effect")
    return _save(fig, path)


def plot_ice(
    grid: np.ndarray, curves: np.ndarray, feature: str, path: str | Path
) -> Path:
    """Centred ICE curves in grey with the centred partial dependence on top."""
    centred = curves - curves[:, [0]]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for row in centred:
        ax.plot(grid, row, color=MUTED, linewidth=0.5, alpha=0.35)
    ax.plot(grid, centred.mean(axis=0), color=SINGLE, linewidth=2.0)
    _style(ax, f"Centred ICE: {feature}", feature, "Change in forecast")
    return _save(fig, path)
