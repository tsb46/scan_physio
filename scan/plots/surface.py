from __future__ import annotations

import dataclasses
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from nilearn import plotting
import numpy as np

from scan.plots.layout import add_colorbar
from scan.plots.layout import annotate_time
from scan.plots.layout import build_side_axes
from scan.plots.layout import draw_label_legend


class FrameMapSource(Protocol):
    n_frames: int

    def get_frame_maps(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclasses.dataclass(frozen=True)
class SurfaceOverlay:
    left_plot: np.ndarray
    right_plot: np.ndarray
    cmap: Colormap | str
    labels: list[tuple[int, str, tuple[float, float, float, float]]]
    vmin: float | None
    vmax: float | None
    output_tag: str


@dataclasses.dataclass(frozen=True)
class SurfaceCameraConfig:
    elev: float | None = None
    azim: float | None = None


@dataclasses.dataclass(frozen=True)
class SurfaceRenderOptions:
    views: list[str]
    figure_size: tuple[float, float]
    dpi: int
    ncols: int | None = None
    cmap: str = "coolwarm"
    vmin: float | None = None
    vmax: float | None = None
    mesh_alpha: float = 1.0
    surf_zoom: float = 1.8
    black_bg: bool = False
    colorbar: bool = True
    colorbar_side: str = "right"
    title_template: str | None = None
    source_format: str = "unknown"
    atlas_view_type: str = "contour"
    atlas_ignore_zero: bool = True
    atlas_legend: bool = False
    atlas_legend_max_items: int = 40
    label_ignore_zero: bool = True
    label_legend: bool = False
    label_legend_max_items: int = 40
    time_annotate: bool = False
    time_s: float | None = None
    global_camera: SurfaceCameraConfig = dataclasses.field(
        default_factory=SurfaceCameraConfig
    )
    left_camera: SurfaceCameraConfig = dataclasses.field(
        default_factory=SurfaceCameraConfig
    )
    right_camera: SurfaceCameraConfig = dataclasses.field(
        default_factory=SurfaceCameraConfig
    )


@dataclasses.dataclass(frozen=True)
class RenderedSurfaceFigure:
    figure: Figure
    panel_axes: list[Axes]
    colorbar_ax: Axes | None
    legend_ax: Axes | None


def _sample_finite_values(
    data: np.ndarray, *, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    flat = np.asarray(data, dtype=float).ravel()
    n = int(flat.size)
    if n == 0 or max_samples <= 0:
        return np.asarray([], dtype=float)
    if n <= max_samples:
        vals = flat[np.isfinite(flat)]
        return vals.astype(float, copy=False)

    collected: list[np.ndarray] = []
    remaining = max_samples
    for _ in range(25):
        if remaining <= 0:
            break
        draw = min(max(remaining * 2, 1024), 200_000)
        idx = rng.integers(0, n, size=draw, dtype=np.int64)
        vals = flat[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > remaining:
            vals = vals[:remaining]
        collected.append(vals.astype(float, copy=False))
        remaining -= int(vals.size)

    if not collected:
        return np.asarray([], dtype=float)
    return np.concatenate(collected, axis=0)


def compute_intensity_bounds(
    metric_source: FrameMapSource | None,
    *,
    selected_index: int,
    intensity_mode: str,
    p_low: float,
    p_high: float,
    max_samples: int,
    max_total_samples: int,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    if metric_source is None:
        return None, None
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    rng = np.random.default_rng(0)
    vmin: float | None = None
    vmax: float | None = None
    if intensity_mode == "global" and metric_source.n_frames > 1:
        samples_per_frame = max(
            1, int(max_total_samples // max(metric_source.n_frames, 1))
        )
        collected: list[np.ndarray] = []
        for frame_index in range(metric_source.n_frames):
            left_map, right_map = metric_source.get_frame_maps(frame_index)
            collected.append(
                _sample_finite_values(
                    np.concatenate((left_map, right_map), axis=0),
                    max_samples=samples_per_frame,
                    rng=rng,
                )
            )
        all_samples = (
            np.concatenate([sample for sample in collected if sample.size > 0], axis=0)
            if collected
            else np.asarray([], dtype=float)
        )
        if all_samples.size > 0:
            vmin = float(np.percentile(all_samples, p_low))
            vmax = float(np.percentile(all_samples, p_high))
    else:
        left_map, right_map = metric_source.get_frame_maps(selected_index)
        samples = _sample_finite_values(
            np.concatenate((left_map, right_map), axis=0),
            max_samples=max_samples,
            rng=rng,
        )
        if samples.size > 0:
            vmin = float(np.percentile(samples, p_low))
            vmax = float(np.percentile(samples, p_high))
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def validate_map_against_mesh(values: np.ndarray, surf_mesh: Any, *, kind: str) -> None:
    coords = np.asarray(surf_mesh[0])
    if int(values.size) != int(coords.shape[0]):
        raise ValueError(
            f"{kind} vertex count ({int(values.size)}) does not match mesh vertices ({int(coords.shape[0])})"
        )


def _apply_surf_zoom(ax: Axes, zoom: float) -> None:
    try:
        z = float(zoom)
    except Exception:
        return
    if not np.isfinite(z) or z <= 0:
        return
    try:
        box_aspect = getattr(cast(Any, ax), "set_box_aspect", None)
        if callable(box_aspect):
            box_aspect(None, zoom=z)
            return
        if hasattr(cast(Any, ax), "dist"):
            setattr(cast(Any, ax), "dist", max(1.0, 10.0 / z))
    except Exception:
        return


def _resolve_surface_view(
    options: SurfaceRenderOptions, *, view: str, hemi: str
) -> str | tuple[float, float]:
    hemi_camera = options.left_camera if hemi == "L" else options.right_camera
    elev = hemi_camera.elev
    if elev is None:
        elev = options.global_camera.elev
    azim = hemi_camera.azim
    if azim is None:
        azim = options.global_camera.azim
    if elev is None and azim is None:
        return view
    if elev is None or azim is None:
        raise ValueError(
            "Explicit surface camera control requires both elevation and azimuth. "
            "Provide both global camera values or a complete per-hemisphere override."
        )
    return float(elev), float(azim)


def render_surface_figure(
    *,
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    stat_map_left: np.ndarray | None,
    stat_map_right: np.ndarray | None,
    overlay: SurfaceOverlay | None,
    options: SurfaceRenderOptions,
    index: int,
) -> RenderedSurfaceFigure:
    panels = [(str(view), hemi) for view in options.views for hemi in ("L", "R")]
    if not panels:
        raise RuntimeError("No surface panels to render")

    ncols = int(options.ncols) if options.ncols is not None else len(panels)
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")
    nrows_maps = int(np.ceil(len(panels) / ncols))

    overlay_is_atlas = overlay is not None and overlay.output_tag == "atlas"
    if overlay_is_atlas:
        legend_enabled = bool(options.atlas_legend and overlay and overlay.labels)
        legend_labels = overlay.labels if overlay is not None else []
        legend_max_items = int(options.atlas_legend_max_items)
        ignore_zero = bool(options.atlas_ignore_zero)
    else:
        legend_enabled = bool(options.label_legend and overlay and overlay.labels)
        legend_labels = overlay.labels if overlay is not None else []
        legend_max_items = int(options.label_legend_max_items)
        ignore_zero = bool(options.label_ignore_zero)

    want_cbar = bool(options.colorbar) and (
        options.vmin is not None and options.vmax is not None
    )
    want_legend = legend_enabled
    side = str(options.colorbar_side)
    want_side_col = want_cbar or want_legend
    ncols_total = ncols + (1 if want_side_col else 0)
    surf_col0 = 1 if (want_side_col and side == "left") else 0
    side_col = 0 if (want_side_col and side == "left") else (ncols_total - 1)

    side_width_ratio = 0.08
    if want_legend and want_cbar:
        side_width_ratio = 0.18
    elif want_legend:
        side_width_ratio = 0.22

    fig = plt.figure(figsize=options.figure_size)
    if options.black_bg:
        fig.patch.set_facecolor("black")

    width_ratios, gs, ax_cbar, ax_legend = build_side_axes(
        fig=fig,
        nrows_maps=nrows_maps,
        ncols=ncols,
        ncols_total=ncols_total,
        side_col=side_col,
        side=side,
        want_side_col=want_side_col,
        want_cbar=want_cbar,
        want_legend=want_legend,
        side_width_ratio=side_width_ratio,
    )
    _ = width_ratios

    axes_flat = [
        fig.add_subplot(gs[row, col + surf_col0], projection="3d")
        for row in range(nrows_maps)
        for col in range(ncols)
    ]

    if options.time_annotate:
        if options.time_s is None:
            raise ValueError("time_annotate requires time_s to be provided")
        annotate_time(fig=fig, time_s=float(options.time_s), black_bg=options.black_bg)

    for panel_idx, (view, hemi) in enumerate(panels):
        if panel_idx >= len(axes_flat):
            break
        ax = axes_flat[panel_idx]
        surf_mesh = surf_left_mesh if hemi == "L" else surf_right_mesh
        surf_sulc = surf_left_sulc if hemi == "L" else surf_right_sulc
        stat_map = stat_map_left if hemi == "L" else stat_map_right
        resolved_view = _resolve_surface_view(options, view=str(view), hemi=str(hemi))
        title = None
        if options.title_template:
            title = str(options.title_template).format(
                index=int(index),
                panel=int(panel_idx),
                view=str(view),
                hemi=str(hemi),
                time=("" if options.time_s is None else f"{float(options.time_s):.3f}"),
                format=options.source_format,
            )

        if stat_map is not None:
            plotting.plot_surf_stat_map(
                surf_mesh,
                stat_map,
                hemi=("left" if hemi == "L" else "right"),
                view=cast(Any, resolved_view),
                cmap=str(options.cmap),
                alpha=float(options.mesh_alpha),
                vmin=options.vmin,
                vmax=options.vmax,
                colorbar=False,
                title=title,
                figure=fig,
                axes=ax,
                bg_map=surf_sulc,
                bg_on_data=False,
            )
        elif title is not None:
            ax.set_title(title)

        if overlay is not None:
            roi_map = overlay.left_plot if hemi == "L" else overlay.right_plot
            if (
                overlay.output_tag == "atlas"
                and str(options.atlas_view_type) == "contour"
            ):
                plotting.plot_surf_contours(
                    surf_mesh,
                    np.ma.masked_invalid(roi_map),
                    cmap=cast(Any, overlay.cmap),
                    alpha=float(options.mesh_alpha),
                    title=None,
                    figure=fig,
                    axes=ax,
                    bg_map=surf_sulc,
                    bg_on_data=False,
                )
            else:
                plotting.plot_surf_roi(
                    surf_mesh,
                    np.ma.masked_invalid(roi_map),
                    hemi=("left" if hemi == "L" else "right"),
                    view=cast(Any, resolved_view),
                    vmin=overlay.vmin,
                    vmax=overlay.vmax,
                    cmap=cast(Any, overlay.cmap),
                    alpha=float(options.mesh_alpha),
                    colorbar=False,
                    figure=fig,
                    axes=ax,
                    bg_on_data=False,
                    bg_map=surf_sulc,
                )

        try:
            for child in ax.get_children():
                if child.__class__.__name__ == "Poly3DCollection":
                    child_any = cast(Any, child)
                    child_any.set_edgecolor("none")
                    child_any.set_linewidth(0)
        except Exception:
            pass
        _apply_surf_zoom(ax, float(options.surf_zoom))

    if ax_cbar is not None and options.vmin is not None and options.vmax is not None:
        add_colorbar(
            fig=fig,
            cax=ax_cbar,
            cmap=str(options.cmap),
            vmin=float(options.vmin),
            vmax=float(options.vmax),
        )
    if ax_legend is not None and legend_labels:
        draw_label_legend(
            ax_legend,
            legend_labels,
            ignore_zero=ignore_zero,
            max_items=legend_max_items,
        )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95)
    return RenderedSurfaceFigure(
        figure=fig,
        panel_axes=axes_flat[: len(panels)],
        colorbar_ax=ax_cbar,
        legend_ax=ax_legend,
    )
