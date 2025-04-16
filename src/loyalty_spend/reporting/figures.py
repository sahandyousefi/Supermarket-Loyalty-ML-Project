"""Static figures for the report and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
