"""Plotting utilities for scan_physio."""

from scan.plots.surface import RenderedSurfaceFigure
from scan.plots.surface import SurfaceCameraConfig
from scan.plots.surface import SurfaceOverlay
from scan.plots.surface import SurfaceRenderOptions
from scan.plots.surface import compute_intensity_bounds
from scan.plots.snapshot import SurfaceSnapshotPlotter
from scan.plots.snapshot import build_snapshot_plotter
from scan.plots.snapshot import load_snapshot_scene
from scan.plots.surface import render_surface_figure
from scan.plots.snapshot import select_snapshot_output_path
from scan.plots.surface import validate_map_against_mesh
from scan.plots.network_summary import plot_network_summary

__all__ = [
    "RenderedSurfaceFigure",
    "SurfaceCameraConfig",
    "SurfaceOverlay",
    "SurfaceSnapshotPlotter",
    "SurfaceRenderOptions",
    "compute_intensity_bounds",
    "build_snapshot_plotter",
    "load_snapshot_scene",
    "render_surface_figure",
    "select_snapshot_output_path",
    "validate_map_against_mesh",
    "plot_network_summary",
]
