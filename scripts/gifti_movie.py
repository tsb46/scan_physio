"""Render paired GIFTI metric time series to frames and optionally encode an MP4.

This is the movie-oriented sibling to scripts/gifti_snapshot.py and preserves the
same base surface visualization defaults for GIFTI metric inputs.

- Input: paired GIFTI metric files (.func.gii)
- Surface rendering: inflated fsLR32k GIFTI surfaces

Video encoding uses an ffmpeg executable. The script will try, in order:
1) --ffmpeg path (if provided)
2) system ffmpeg on PATH
3) imageio-ffmpeg (if installed; see optional dependency extra `viz`)

Examples
--------
# Basic movie
uv run python scripts/gifti_movie.py --input sub-01_bold_lh.func.gii --output bold.mp4

# Restrict frames and keep rendered PNGs
uv run python scripts/gifti_movie.py --input sub-01_bold_lh.func.gii --start 0 --stop 20 \
  --keep-frames --frames-dir tmp/gifti_frames

# Multiple views with explicit camera control
uv run python scripts/gifti_movie.py --input-left lh.func.gii --input-right rh.func.gii \
  --surf-views lateral medial --surf-elev 15 --surf-azim 210 --output bold_views.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import plotting, surface
from nilearn.plotting.cm import mix_colormaps


@dataclasses.dataclass(frozen=True)
class _MetricSource:
    n_frames: int
    output_hint: Path
    get_frame_maps: Callable[[int], tuple[np.ndarray, np.ndarray]]


def _validate_percentiles(p_low: float, p_high: float) -> tuple[float, float]:
    if not (0.0 <= p_low <= 100.0 and 0.0 <= p_high <= 100.0):
        raise ValueError(f"Percentiles must be in [0, 100], got {p_low}, {p_high}")
    if p_low >= p_high:
        raise ValueError(
            f"Lower percentile must be < upper percentile, got {p_low}, {p_high}"
        )
    return float(p_low), float(p_high)


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


def _load_gifti_image(path: Path, *, kind: str) -> GiftiImage:
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, GiftiImage):
        raise TypeError(f"Expected {kind} GIFTI image, got {type(loaded)}")
    return loaded


def _get_gifti_frame(img: GiftiImage, index: int) -> np.ndarray:
    if index < 0 or index >= len(img.darrays):
        raise ValueError(f"frame index out of range: {index}")
    return np.asarray(img.darrays[index].data, dtype=float).ravel()


def _toggle_hemi_token(name: str) -> str | None:
    replacements = (
        ("lh.", "rh."),
        ("rh.", "lh."),
        ("_lh_", "_rh_"),
        ("_rh_", "_lh_"),
        ("_lh.", "_rh."),
        ("_rh.", "_lh."),
        ("-lh.", "-rh."),
        ("-rh.", "-lh."),
        (".lh_", ".rh_"),
        (".rh_", ".lh_"),
        (".L.", ".R."),
        (".R.", ".L."),
        ("_L_", "_R_"),
        ("_R_", "_L_"),
        ("_L.", "_R."),
        ("_R.", "_L."),
    )
    for old, new in replacements:
        if old in name:
            return name.replace(old, new, 1)
    return None


def _detect_hemi_token(name: str) -> str | None:
    left_tokens = ("lh.", "_lh_", "_lh.", "-lh.", ".lh_", ".L.", "_L_", "_L.")
    right_tokens = ("rh.", "_rh_", "_rh.", "-rh.", ".rh_", ".R.", "_R_", "_R.")
    if any(token in name for token in left_tokens):
        return "left"
    if any(token in name for token in right_tokens):
        return "right"
    return None


def _pair_from_single_path(path: Path, *, suffix: str) -> tuple[Path, Path]:
    if path.exists():
        mate_name = _toggle_hemi_token(path.name)
        if mate_name is None:
            raise ValueError(
                f"Could not infer hemisphere mate from {path}. Provide explicit left/right files."
            )
        mate = path.with_name(mate_name)
        if not mate.exists():
            raise FileNotFoundError(f"Could not find inferred hemisphere mate: {mate}")
        hemi = _detect_hemi_token(path.name)
        if hemi == "left":
            return path, mate
        if hemi == "right":
            return mate, path
        raise ValueError(
            f"Could not determine left/right hemisphere from {path}. Provide explicit left/right files."
        )

    candidate_sets = (
        (
            path.with_name(f"{path.name}_lh{suffix}"),
            path.with_name(f"{path.name}_rh{suffix}"),
        ),
        (
            path.with_name(f"{path.name}.lh{suffix}"),
            path.with_name(f"{path.name}.rh{suffix}"),
        ),
        (
            path.with_name(f"{path.name}-lh{suffix}"),
            path.with_name(f"{path.name}-rh{suffix}"),
        ),
    )
    for left_path, right_path in candidate_sets:
        if left_path.exists() and right_path.exists():
            return left_path, right_path
    raise FileNotFoundError(
        f"Could not resolve paired GIFTI files from {path}. Provide explicit left/right files."
    )


def _resolve_gifti_pair(
    *,
    single_path: Path | None,
    left_path: Path | None,
    right_path: Path | None,
) -> tuple[Path, Path]:
    if left_path is not None or right_path is not None:
        if left_path is None or right_path is None:
            raise ValueError(
                "Provide both --input-left and --input-right, or just --input."
            )
        return left_path, right_path
    if single_path is None:
        raise ValueError("No GIFTI metric input provided")
    return _pair_from_single_path(single_path, suffix=".func.gii")


def _load_gifti_metric_source(args: argparse.Namespace) -> _MetricSource:
    left_path, right_path = _resolve_gifti_pair(
        single_path=cast(Path | None, args.input),
        left_path=cast(Path | None, args.input_left),
        right_path=cast(Path | None, args.input_right),
    )
    left_img = _load_gifti_image(left_path, kind="metric")
    right_img = _load_gifti_image(right_path, kind="metric")

    n_frames = len(left_img.darrays)
    if n_frames <= 0:
        raise ValueError("No frames found in left hemisphere GIFTI")
    if len(right_img.darrays) != n_frames:
        raise ValueError(
            "Left and right hemisphere GIFTI files must have the same number of frames"
        )

    left_sizes = {int(np.asarray(darr.data).size) for darr in left_img.darrays}
    right_sizes = {int(np.asarray(darr.data).size) for darr in right_img.darrays}
    if len(left_sizes) != 1 or len(right_sizes) != 1:
        raise ValueError(
            "Each GIFTI hemisphere must have consistent vertex counts across frames"
        )

    def _get_frame_maps(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        return _get_gifti_frame(left_img, frame_index), _get_gifti_frame(
            right_img, frame_index
        )

    return _MetricSource(
        n_frames=n_frames,
        output_hint=left_path,
        get_frame_maps=_get_frame_maps,
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
    args: argparse.Namespace, *, view: str, hemi: str
) -> str | tuple[float, float]:
    hemi_name = "left" if hemi == "L" else "right"
    elev = getattr(args, f"surf_elev_{hemi_name}")
    if elev is None:
        elev = args.surf_elev
    azim = getattr(args, f"surf_azim_{hemi_name}")
    if azim is None:
        azim = args.surf_azim
    if elev is None and azim is None:
        return view
    if elev is None or azim is None:
        raise ValueError(
            "Explicit surface camera control requires both elevation and azimuth. "
            "Provide both global --surf-elev/--surf-azim values or a complete per-hemisphere override."
        )
    return float(elev), float(azim)


def _add_colorbar(
    *, fig: Figure, cax: Axes, cmap: str, vmin: float, vmax: float
) -> None:
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="vertical")


def _validate_map_against_mesh(
    values: np.ndarray, surf_mesh: Any, *, kind: str
) -> None:
    coords = np.asarray(surf_mesh[0])
    if int(values.size) != int(coords.shape[0]):
        raise ValueError(
            f"{kind} vertex count ({int(values.size)}) does not match mesh vertices ({int(coords.shape[0])})"
        )


def _strip_suffixes(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _derive_default_output(path: Path) -> Path:
    stem = _strip_suffixes(
        path.name,
        (
            ".func.gii",
            ".shape.gii",
            ".nii.gz",
            ".nii",
            ".gii",
        ),
    )
    return path.with_name(f"{stem}.mp4")


def _iter_frame_indices(
    n_frames: int, start: int, stop: int | None, step: int
) -> list[int]:
    stop_ = n_frames if stop is None else min(stop, n_frames)
    if start < 0 or start >= n_frames:
        raise ValueError(f"start must be in [0, {n_frames - 1}], got {start}")
    if stop_ <= start:
        raise ValueError(f"stop must be > start; got start={start} stop={stop_}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return list(range(start, stop_, step))


def _find_ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_ffmpeg(
    *,
    ffmpeg: str,
    frames_pattern: str,
    fps: float,
    output: Path,
    crf: int,
    preset: str,
    scale: str | None,
) -> None:
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        frames_pattern,
    ]

    vf_filters: list[str] = []
    if scale:
        vf_filters.append(f"scale={scale}")
    vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _compute_frame_bounds(
    source: _MetricSource,
    *,
    frame_index: int,
    p_low: float,
    p_high: float,
    max_samples: int,
    rng: np.random.Generator,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    left_map, right_map = source.get_frame_maps(frame_index)
    samples = _sample_finite_values(
        np.concatenate((left_map, right_map), axis=0),
        max_samples=max_samples,
        rng=rng,
    )
    vmin = float(np.percentile(samples, p_low)) if samples.size > 0 else None
    vmax = float(np.percentile(samples, p_high)) if samples.size > 0 else None
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def _compute_global_bounds(
    source: _MetricSource,
    *,
    frame_indices: list[int],
    p_low: float,
    p_high: float,
    max_total_samples: int,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    rng = np.random.default_rng(0)
    samples_per_frame = max(1, int(max_total_samples // max(len(frame_indices), 1)))
    collected: list[np.ndarray] = []
    for frame_index in frame_indices:
        left_map, right_map = source.get_frame_maps(frame_index)
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
    vmin = float(np.percentile(all_samples, p_low)) if all_samples.size > 0 else None
    vmax = float(np.percentile(all_samples, p_high)) if all_samples.size > 0 else None
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def _compute_bg_facecolors(*, n_faces: int, alpha: float) -> np.ndarray:
    bg = plt.get_cmap("gray_r")(np.full(n_faces, 0.5, dtype=float))
    bg[:, 3] = float(alpha) * bg[:, 3]
    return bg


def _update_poly3d_facecolors(
    *,
    poly: Poly3DCollection,
    faces: np.ndarray,
    stat_map_vertices: np.ndarray,
    cmap: Any,
    vmin: float,
    vmax: float,
    bg_facecolors: np.ndarray,
) -> None:
    face_vals = np.mean(np.asarray(stat_map_vertices, dtype=float)[faces], axis=1)
    kept = ~np.isnan(face_vals)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        finite = face_vals[np.isfinite(face_vals)]
        if finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = -1.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

    scaled = (face_vals - float(vmin)) / (float(vmax) - float(vmin))
    scaled = np.clip(scaled, 0.0, 1.0)
    surf_colors = cmap(scaled)
    surf_colors[~kept, 3] = 0.0
    face_colors = mix_colormaps(surf_colors, bg_facecolors)
    face_colors = np.clip(face_colors, 0.0, 1.0)
    poly.set_facecolors(face_colors)  # type: ignore


def _default_surface_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "template" / filename


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render paired GIFTI metric data to surface frames and optionally encode a movie.",
    )
    p.add_argument(
        "--input",
        required=False,
        type=Path,
        default=None,
        help="A single GIFTI hemisphere file or prefix used to infer the mate file.",
    )
    p.add_argument(
        "--input-left",
        type=Path,
        default=None,
        help="Left hemisphere metric GIFTI (.func.gii).",
    )
    p.add_argument(
        "--input-right",
        type=Path,
        default=None,
        help="Right hemisphere metric GIFTI (.func.gii).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output movie path (.mp4). Defaults to a path derived from the metric input.",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory to write frames. Defaults to a temp directory.",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Do not delete frames directory when done (only applies to temp dir).",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Only render frames; do not run ffmpeg.",
    )

    p.add_argument(
        "--surf-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.inflated.32k_fs_LR.surf.gii"),
        help="Left hemisphere inflated surface GIFTI.",
    )
    p.add_argument(
        "--surf-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.inflated.32k_fs_LR.surf.gii"),
        help="Right hemisphere inflated surface GIFTI.",
    )
    p.add_argument(
        "--surf-views",
        nargs="+",
        default=["lateral", "medial"],
        help="One or more nilearn surface views. Each view renders both hemispheres as separate panels.",
    )
    p.add_argument(
        "--surf-elev",
        type=float,
        default=None,
        help=(
            "Explicit camera elevation passed through nilearn as part of a "
            "(elev, azim) view tuple. Overrides --surf-views when paired with --surf-azim."
        ),
    )
    p.add_argument(
        "--surf-azim",
        type=float,
        default=None,
        help=(
            "Explicit camera azimuth passed through nilearn as part of a "
            "(elev, azim) view tuple. Overrides --surf-views when paired with --surf-azim."
        ),
    )
    p.add_argument(
        "--surf-elev-left",
        type=float,
        default=None,
        help="Per-hemisphere override for camera elevation (LEFT).",
    )
    p.add_argument(
        "--surf-elev-right",
        type=float,
        default=None,
        help="Per-hemisphere override for camera elevation (RIGHT).",
    )
    p.add_argument(
        "--surf-azim-left",
        type=float,
        default=None,
        help="Per-hemisphere override for camera azimuth (LEFT).",
    )
    p.add_argument(
        "--surf-azim-right",
        type=float,
        default=None,
        help="Per-hemisphere override for camera azimuth (RIGHT).",
    )
    p.add_argument(
        "--surf-zoom",
        type=float,
        default=1.8,
        help="Zoom factor for mplot3d surface panels.",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="Number of columns when rendering multiple panels.",
    )

    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    p.add_argument("--vmin", type=float, default=None, help="Lower bound for colormap.")
    p.add_argument("--vmax", type=float, default=None, help="Upper bound for colormap.")
    p.add_argument(
        "--intensity-mode",
        choices=["global", "frame"],
        default="global",
        help="How to set vmin/vmax when not explicitly provided.",
    )
    p.add_argument(
        "--auto-percentiles",
        nargs=2,
        type=float,
        metavar=("PLOW", "PHIGH"),
        default=(1.0, 99.0),
        help="Percentiles for automatic vmin/vmax (default: 1 99).",
    )
    p.add_argument(
        "--auto-max-samples",
        type=int,
        default=200_000,
        help="Max random samples when estimating frame percentiles (default: 200000).",
    )
    p.add_argument(
        "--auto-max-total-samples",
        type=int,
        default=2_000_000,
        help="Max total random samples used to estimate global percentiles (default: 2000000).",
    )
    p.add_argument(
        "--black-bg",
        action="store_true",
        help="Use black figure background.",
    )
    p.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colorbar (default: true).",
    )
    p.add_argument(
        "--colorbar-side",
        choices=["right", "left"],
        default="right",
        help="Where to place the colorbar when enabled (default: right).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional title template. Use {frame}, {index}, {panel}, {view}, {hemi}, {time} placeholders.",
    )
    p.add_argument(
        "--time-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate movie with time in seconds (default: false).",
    )
    p.add_argument("--tr", type=float, default=None, help="Repetition time in seconds.")
    p.add_argument(
        "--t0-trs",
        type=float,
        default=0.0,
        help="Starting time in TRs for index 0.",
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved frames (default: 150).",
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 720),
        help="Frame size in pixels (default: 1280 720).",
    )

    p.add_argument(
        "--start", type=int, default=0, help="First frame index (default: 0)."
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index (exclusive). Defaults to end.",
    )
    p.add_argument("--step", type=int, default=1, help="Frame step (default: 1).")

    p.add_argument(
        "--fps", type=float, default=10.0, help="Frames per second (default: 10)."
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Path to ffmpeg executable (optional).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x264 CRF quality (lower=better, larger files). Default: 23.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 preset (default: medium).",
    )
    p.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Optional ffmpeg scale W:H (e.g. 960:-2) to downscale during encoding.",
    )
    p.add_argument(
        "--fast-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse surface artists across frames when vmin/vmax stay fixed. "
            "Automatically falls back when per-frame auto scaling is active."
        ),
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print periodic progress updates (default: true).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if bool(args.time_annotate) and args.tr is None:
        raise ValueError("--tr is required when using --time-annotate")

    source = _load_gifti_metric_source(args)
    output_path = (
        Path(args.output)
        if args.output is not None
        else _derive_default_output(source.output_hint)
    )

    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))
    probe_left, probe_right = source.get_frame_maps(0)
    _validate_map_against_mesh(probe_left, surf_left_mesh, kind="Left metric")
    _validate_map_against_mesh(probe_right, surf_right_mesh, kind="Right metric")

    frame_indices = _iter_frame_indices(
        source.n_frames,
        int(args.start),
        cast(int | None, args.stop),
        int(args.step),
    )
    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )
    global_vmin: float | None = None
    global_vmax: float | None = None
    if args.intensity_mode == "global":
        global_vmin, global_vmax = _compute_global_bounds(
            source,
            frame_indices=frame_indices,
            p_low=p_low,
            p_high=p_high,
            max_total_samples=int(args.auto_max_total_samples),
            vmin_arg=cast(float | None, args.vmin),
            vmax_arg=cast(float | None, args.vmax),
        )

    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)
    panels = [(str(view), hemi) for view in args.surf_views for hemi in ("L", "R")]
    if not panels:
        raise RuntimeError("No surface panels to render")
    ncols = int(args.ncols) if args.ncols is not None else len(panels)
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")
    nrows_maps = int(np.ceil(len(panels) / ncols))
    want_cbar = bool(args.colorbar)
    side = str(args.colorbar_side)
    ncols_total = ncols + (1 if want_cbar else 0)
    surf_col0 = 1 if (want_cbar and side == "left") else 0
    side_col = 0 if (want_cbar and side == "left") else (ncols_total - 1)
    width_ratios = None
    if want_cbar:
        width_ratios = (
            ([0.08] + [1.0] * ncols) if side == "left" else ([1.0] * ncols + [0.08])
        )

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gifti_frames_")
        frames_dir = Path(temp_dir_obj.name)
    else:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
    keep_temp_frames = bool(args.keep_frames) or bool(args.no_video)

    can_fast_render = bool(args.fast_render)
    if args.intensity_mode == "frame" and (args.vmin is None or args.vmax is None):
        can_fast_render = False

    try:
        frame_rng = np.random.default_rng(0)
        t_all0 = time.perf_counter()

        if can_fast_render:
            fixed_vmin = global_vmin
            fixed_vmax = global_vmax
            if fixed_vmin is None or fixed_vmax is None:
                fixed_vmin, fixed_vmax = _compute_frame_bounds(
                    source,
                    frame_index=frame_indices[0],
                    p_low=p_low,
                    p_high=p_high,
                    max_samples=int(args.auto_max_samples),
                    rng=frame_rng,
                    vmin_arg=cast(float | None, args.vmin),
                    vmax_arg=cast(float | None, args.vmax),
                )
            if fixed_vmin is None or fixed_vmax is None:
                fixed_vmin, fixed_vmax = -1.0, 1.0

            fig = plt.figure(figsize=figsize)
            if bool(args.black_bg):
                fig.patch.set_facecolor("black")
            gs = fig.add_gridspec(
                nrows=nrows_maps,
                ncols=ncols_total,
                width_ratios=width_ratios,
                wspace=0.02,
                hspace=0.0,
            )
            axes_flat = [
                fig.add_subplot(gs[row, col + surf_col0], projection="3d")
                for row in range(nrows_maps)
                for col in range(ncols)
            ]
            ax_cbar = None
            if want_cbar:
                ax_side_container = fig.add_subplot(gs[:nrows_maps, side_col])
                ax_side_container.set_axis_off()
                ax_cbar = inset_axes(
                    ax_side_container,
                    width="55%",
                    height="70%",
                    loc="center",
                    borderpad=0.0,
                )

            left_coords, left_faces = surf_left_mesh
            right_coords, right_faces = surf_right_mesh
            left_faces = np.asarray(left_faces, dtype=np.int64)
            right_faces = np.asarray(right_faces, dtype=np.int64)
            bg_left = _compute_bg_facecolors(
                n_faces=int(left_faces.shape[0]), alpha=0.5
            )
            bg_right = _compute_bg_facecolors(
                n_faces=int(right_faces.shape[0]), alpha=0.5
            )
            cmap_obj = plt.get_cmap(str(args.cmap))

            init_left, init_right = source.get_frame_maps(frame_indices[0])
            panel_polys: list[Poly3DCollection] = []
            panel_faces: list[np.ndarray] = []
            panel_bg: list[np.ndarray] = []
            time_text = None
            if bool(args.time_annotate):
                color = "white" if bool(args.black_bg) else "black"
                time_text = fig.text(0.01, 0.99, "", ha="left", va="top", color=color)

            for panel_idx, (view, hemi) in enumerate(panels):
                ax = axes_flat[panel_idx]
                surf_mesh = surf_left_mesh if hemi == "L" else surf_right_mesh
                stat_map = init_left if hemi == "L" else init_right
                resolved_view = _resolve_surface_view(args, view=view, hemi=hemi)
                title = None
                if args.title:
                    time_sec = (
                        (float(args.t0_trs) + float(frame_indices[0])) * float(args.tr)
                        if args.tr is not None
                        else None
                    )
                    title = str(args.title).format(
                        frame=0,
                        index=int(frame_indices[0]),
                        panel=panel_idx,
                        view=view,
                        hemi=hemi,
                        time=("" if time_sec is None else f"{time_sec:.3f}"),
                    )
                plotting.plot_surf_stat_map(
                    surf_mesh,
                    stat_map,
                    hemi=("left" if hemi == "L" else "right"),
                    view=cast(Any, resolved_view),
                    cmap=str(args.cmap),
                    vmin=fixed_vmin,
                    vmax=fixed_vmax,
                    colorbar=False,
                    title=title,
                    figure=fig,
                    axes=ax,
                )
                if ax.collections:
                    try:
                        ax.collections[0].set_edgecolor("none")
                        ax.collections[0].set_linewidth(0)
                    except Exception:
                        pass
                _apply_surf_zoom(ax, float(args.surf_zoom))
                poly = next(
                    (
                        child
                        for child in ax.get_children()
                        if isinstance(child, Poly3DCollection)
                    ),
                    None,
                )
                if poly is None:
                    raise RuntimeError("Failed to initialize surface collection")
                panel_polys.append(poly)
                panel_faces.append(left_faces if hemi == "L" else right_faces)
                panel_bg.append(bg_left if hemi == "L" else bg_right)

            if ax_cbar is not None:
                _add_colorbar(
                    fig=fig,
                    cax=ax_cbar,
                    cmap=str(args.cmap),
                    vmin=float(fixed_vmin),
                    vmax=float(fixed_vmax),
                )
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95)

            for frame_number, frame_index in enumerate(frame_indices):
                t0 = time.perf_counter()
                left_map, right_map = source.get_frame_maps(frame_index)
                time_sec = (
                    (float(args.t0_trs) + float(frame_index)) * float(args.tr)
                    if args.tr is not None
                    else None
                )
                if time_text is not None:
                    label = "t = ? s" if time_sec is None else f"t={time_sec:.2f}s"
                    time_text.set_text(label)
                for panel_idx, (view, hemi) in enumerate(panels):
                    if args.title:
                        axes_flat[panel_idx].set_title(
                            str(args.title).format(
                                frame=frame_number,
                                index=frame_index,
                                panel=panel_idx,
                                view=view,
                                hemi=hemi,
                                time=("" if time_sec is None else f"{time_sec:.3f}"),
                            )
                        )
                    _update_poly3d_facecolors(
                        poly=panel_polys[panel_idx],
                        faces=panel_faces[panel_idx],
                        stat_map_vertices=(left_map if hemi == "L" else right_map),
                        cmap=cmap_obj,
                        vmin=float(fixed_vmin),
                        vmax=float(fixed_vmax),
                        bg_facecolors=panel_bg[panel_idx],
                    )
                frame_path = frames_dir / f"frame_{frame_number:05d}.png"
                fig.savefig(str(frame_path), dpi=dpi, facecolor=fig.get_facecolor())
                if bool(args.progress):
                    n_done = frame_number + 1
                    n_total = len(frame_indices)
                    if n_done == 1 or (n_done % 5) == 0 or n_done == n_total:
                        elapsed = time.perf_counter() - t0
                        total_elapsed = time.perf_counter() - t_all0
                        avg = total_elapsed / max(n_done, 1)
                        remaining = avg * (n_total - n_done)
                        print(
                            f"Rendered frame {n_done}/{n_total} (index={frame_index}) in {elapsed:.2f}s; avg {avg:.2f}s/frame; ETA {remaining / 60:.1f} min"
                        )
            plt.close(fig)
        else:
            for frame_number, frame_index in enumerate(frame_indices):
                t0 = time.perf_counter()
                left_map, right_map = source.get_frame_maps(frame_index)
                frame_vmin = global_vmin
                frame_vmax = global_vmax
                if args.intensity_mode == "frame":
                    frame_vmin, frame_vmax = _compute_frame_bounds(
                        source,
                        frame_index=frame_index,
                        p_low=p_low,
                        p_high=p_high,
                        max_samples=int(args.auto_max_samples),
                        rng=frame_rng,
                        vmin_arg=cast(float | None, args.vmin),
                        vmax_arg=cast(float | None, args.vmax),
                    )

                fig = plt.figure(figsize=figsize)
                if bool(args.black_bg):
                    fig.patch.set_facecolor("black")
                gs = fig.add_gridspec(
                    nrows=nrows_maps,
                    ncols=ncols_total,
                    width_ratios=width_ratios,
                    wspace=0.02,
                    hspace=0.0,
                )
                axes_flat = [
                    fig.add_subplot(gs[row, col + surf_col0], projection="3d")
                    for row in range(nrows_maps)
                    for col in range(ncols)
                ]
                ax_cbar = None
                if want_cbar:
                    ax_side_container = fig.add_subplot(gs[:nrows_maps, side_col])
                    ax_side_container.set_axis_off()
                    ax_cbar = inset_axes(
                        ax_side_container,
                        width="55%",
                        height="70%",
                        loc="center",
                        borderpad=0.0,
                    )

                time_sec = (
                    (float(args.t0_trs) + float(frame_index)) * float(args.tr)
                    if args.tr is not None
                    else None
                )
                if bool(args.time_annotate):
                    fig.text(
                        0.01,
                        0.99,
                        "t = ? s" if time_sec is None else f"t={time_sec:.2f}s",
                        ha="left",
                        va="top",
                        color=("white" if bool(args.black_bg) else "black"),
                    )

                for panel_idx, (view, hemi) in enumerate(panels):
                    ax = axes_flat[panel_idx]
                    surf_mesh = surf_left_mesh if hemi == "L" else surf_right_mesh
                    stat_map = left_map if hemi == "L" else right_map
                    resolved_view = _resolve_surface_view(args, view=view, hemi=hemi)
                    title = None
                    if args.title:
                        title = str(args.title).format(
                            frame=frame_number,
                            index=frame_index,
                            panel=panel_idx,
                            view=view,
                            hemi=hemi,
                            time=("" if time_sec is None else f"{time_sec:.3f}"),
                        )
                    plotting.plot_surf_stat_map(
                        surf_mesh,
                        stat_map,
                        hemi=("left" if hemi == "L" else "right"),
                        view=cast(Any, resolved_view),
                        cmap=str(args.cmap),
                        vmin=frame_vmin,
                        vmax=frame_vmax,
                        colorbar=False,
                        title=title,
                        figure=fig,
                        axes=ax,
                    )
                    if ax.collections:
                        try:
                            ax.collections[0].set_edgecolor("none")
                            ax.collections[0].set_linewidth(0)
                        except Exception:
                            pass
                    _apply_surf_zoom(ax, float(args.surf_zoom))

                if ax_cbar is not None:
                    draw_vmin = frame_vmin
                    draw_vmax = frame_vmax
                    if draw_vmin is None or draw_vmax is None:
                        finite = np.concatenate((left_map, right_map), axis=0)
                        finite = finite[np.isfinite(finite)]
                        if finite.size:
                            draw_vmin = float(np.percentile(finite, p_low))
                            draw_vmax = float(np.percentile(finite, p_high))
                        else:
                            draw_vmin, draw_vmax = -1.0, 1.0
                    _add_colorbar(
                        fig=fig,
                        cax=ax_cbar,
                        cmap=str(args.cmap),
                        vmin=float(draw_vmin),
                        vmax=float(draw_vmax),
                    )

                fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95)
                frame_path = frames_dir / f"frame_{frame_number:05d}.png"
                fig.savefig(str(frame_path), dpi=dpi)
                plt.close(fig)
                if bool(args.progress):
                    n_done = frame_number + 1
                    n_total = len(frame_indices)
                    if n_done == 1 or (n_done % 5) == 0 or n_done == n_total:
                        elapsed = time.perf_counter() - t0
                        total_elapsed = time.perf_counter() - t_all0
                        avg = total_elapsed / max(n_done, 1)
                        remaining = avg * (n_total - n_done)
                        print(
                            f"Rendered frame {n_done}/{n_total} (index={frame_index}) in {elapsed:.2f}s; avg {avg:.2f}s/frame; ETA {remaining / 60:.1f} min"
                        )

        if args.no_video:
            print(f"Rendered {len(frame_indices)} frame(s) to: {frames_dir}")
            return 0

        ffmpeg = _find_ffmpeg(cast(str | None, args.ffmpeg))
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found. Install system ffmpeg, or add optional extra 'viz' "
                "(imageio-ffmpeg) and re-run. You can also use --no-video to only render frames."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            ffmpeg=ffmpeg,
            frames_pattern=str(frames_dir / "frame_%05d.png"),
            fps=float(args.fps),
            output=output_path,
            crf=int(args.crf),
            preset=str(args.preset),
            scale=cast(str | None, args.scale),
        )
        print(f"Wrote movie: {output_path}")
        return 0
    finally:
        if temp_dir_obj is not None and (not keep_temp_frames):
            temp_dir_obj.cleanup()
        elif temp_dir_obj is not None and keep_temp_frames:
            print(f"Kept frames at: {frames_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
