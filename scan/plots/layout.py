from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def draw_label_legend(
    ax: Axes,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
    *,
    ignore_zero: bool,
    max_items: int,
) -> None:
    ax.set_axis_off()
    items = []
    for key, name, rgba in labels:
        if ignore_zero and int(key) == 0:
            continue
        items.append((int(key), str(name), rgba))
    if max_items > 0:
        items = items[: int(max_items)]
    if not items:
        return

    handles = [
        Patch(facecolor=rgba, edgecolor="none", label=f"{key}: {name}")
        for key, name, rgba in items
    ]
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        handleheight=1.0,
        labelspacing=0.4,
        borderaxespad=0.0,
    )


def add_colorbar(
    *, fig: Figure, cax: Axes, cmap: str, vmin: float, vmax: float
) -> None:
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="vertical")


def build_side_axes(
    *,
    fig: Figure,
    nrows_maps: int,
    ncols: int,
    ncols_total: int,
    side_col: int,
    side: str,
    want_side_col: bool,
    want_cbar: bool,
    want_legend: bool,
    side_width_ratio: float,
) -> tuple[list[float] | None, Any, Axes | None, Axes | None]:
    width_ratios = None
    if want_side_col:
        width_ratios = (
            ([side_width_ratio] + [1.0] * ncols)
            if side == "left"
            else ([1.0] * ncols + [side_width_ratio])
        )

    gs = fig.add_gridspec(
        nrows=nrows_maps,
        ncols=ncols_total,
        width_ratios=width_ratios,
        wspace=0.02,
        hspace=0.0,
    )

    ax_cbar = None
    ax_legend = None
    if want_side_col:
        ax_side_container = fig.add_subplot(gs[:nrows_maps, side_col])
        ax_side_container.set_axis_off()
        if want_cbar:
            ax_cbar = inset_axes(
                ax_side_container,
                width="55%",
                height=("55%" if want_legend else "70%"),
                loc="upper center" if want_legend else "center",
                borderpad=0.0,
            )
        if want_legend:
            ax_legend = inset_axes(
                ax_side_container,
                width="100%",
                height=("40%" if want_cbar else "90%"),
                loc="lower center" if want_cbar else "center",
                borderpad=0.0,
            )
    return width_ratios, gs, ax_cbar, ax_legend


def annotate_time(*, fig: Figure, time_s: float, black_bg: bool) -> None:
    fig.text(
        0.01,
        0.99,
        f"t={time_s:.2f}s",
        ha="left",
        va="top",
        color=("white" if black_bg else "black"),
    )
