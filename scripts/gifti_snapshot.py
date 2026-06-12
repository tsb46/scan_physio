"""Render a static surface snapshot from CIFTI or GIFTI data as a PNG.

This script keeps the shared surface plotting/layout logic in one place while
supporting separate CIFTI and GIFTI loading paths.

- CIFTI metric input: .dtseries.nii or .dscalar.nii
- GIFTI metric input: paired lh/rh .func.gii files
- CIFTI overlay input: dense label atlas .dlabel.nii
- GIFTI overlay input: paired lh/rh .label.gii files
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Callable, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
import nibabel as nib
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import surface

from scan.plots import SurfaceCameraConfig
from scan.plots import SurfaceOverlay
from scan.plots import SurfaceRenderOptions
from scan.plots import compute_intensity_bounds as compute_surface_intensity_bounds
from scan.plots import render_surface_figure
from scan.plots import validate_map_against_mesh


@dataclasses.dataclass(frozen=True)
class _CiftiFrameSpec:
    frame_axis_first: bool
    n_frames: int


@dataclasses.dataclass(frozen=True)
class _MetricSource:
    n_frames: int
    output_hint: Path
    get_frame_maps: Callable[[int], tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass(frozen=True)
class _OverlayData:
    left_plot: np.ndarray
    right_plot: np.ndarray
    cmap: Colormap | str
    labels: list[tuple[int, str, tuple[float, float, float, float]]]
    vmin: float | None
    vmax: float | None
    output_hint: Path
    output_tag: str


@dataclasses.dataclass(frozen=True)
class _SceneData:
    metric_source: _MetricSource | None
    overlay: _OverlayData | None
    source_format: str


@dataclasses.dataclass(frozen=True)
class _GiftiLabelPair:
    left_path: Path | None
    right_path: Path | None


def _validate_percentiles(p_low: float, p_high: float) -> tuple[float, float]:
    if not (0.0 <= p_low <= 100.0 and 0.0 <= p_high <= 100.0):
        raise ValueError(f"Percentiles must be in [0, 100], got {p_low}, {p_high}")
    if p_low >= p_high:
        raise ValueError(
            f"Lower percentile must be < upper percentile, got {p_low}, {p_high}"
        )
    return float(p_low), float(p_high)


def _normalize_rgba(
    rgba4: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    mx = max(rgba4)
    if mx > 1.0 and mx <= 255.0:
        return tuple(float(x) / 255.0 for x in rgba4)  # type: ignore[return-value]
    return rgba4


def _copy_cmap_with_transparent_bad(cmap: Colormap | str) -> Colormap | str:
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    try:
        cmap_obj = cast(Any, cmap_obj).copy()
    except Exception:
        pass
    try:
        cast(Any, cmap_obj).set_bad((0.0, 0.0, 0.0, 0.0))
    except Exception:
        pass
    return cmap_obj


def _sample_overlay_colors(
    cmap_name: str, count: int
) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []
    cmap_obj = plt.get_cmap(cmap_name)

    def _rgba(value: float) -> tuple[float, float, float, float]:
        color = cmap_obj(value)
        return (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3]),
        )

    if count == 1:
        return [_rgba(0.5)]
    samples = np.linspace(0.0, 1.0, num=count, endpoint=False)
    return [_rgba(float(sample)) for sample in samples]


def _infer_cifti_axes(
    img: nib.cifti2.cifti2.Cifti2Image,
) -> tuple[Any, Any, _CiftiFrameSpec]:
    from nibabel.cifti2 import cifti2_axes

    ax0 = cifti2_axes.from_index_mapping(img.header.get_index_map(0))
    ax1 = cifti2_axes.from_index_mapping(img.header.get_index_map(1))
    if isinstance(ax0, cifti2_axes.BrainModelAxis) and not isinstance(
        ax1, cifti2_axes.BrainModelAxis
    ):
        frame_axis, brain_axis = ax1, ax0
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=False, n_frames=int(frame_axis.size)
        )
    elif isinstance(ax1, cifti2_axes.BrainModelAxis) and not isinstance(
        ax0, cifti2_axes.BrainModelAxis
    ):
        frame_axis, brain_axis = ax0, ax1
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=True, n_frames=int(frame_axis.size)
        )
    else:
        raise ValueError(
            "Expected one BrainModelAxis and one frame axis (SeriesAxis/ScalarAxis). "
            f"Got axis0={type(ax0)} axis1={type(ax1)}"
        )
    return frame_axis, brain_axis, frame_spec


def _get_cifti_frame_vector(
    img: nib.cifti2.cifti2.Cifti2Image,
    *,
    frame_index: int,
    frame_spec: _CiftiFrameSpec,
) -> np.ndarray:
    dataobj = img.dataobj
    if frame_spec.frame_axis_first:
        vec = np.asanyarray(dataobj[frame_index, :])
    else:
        vec = np.asanyarray(dataobj[:, frame_index])
    return np.asarray(vec, dtype=float).ravel()


def _extract_cortex_structures(brain_axis: Any) -> dict[str, tuple[slice, Any]]:
    structures: dict[str, tuple[slice, Any]] = {}
    for struct_name, struct_slice, struct_bm in brain_axis.iter_structures():
        structures[str(struct_name)] = (struct_slice, struct_bm)
    return structures


def _brain_to_hemi_vertices(
    *,
    frame_vec: np.ndarray,
    structures: dict[str, tuple[slice, Any]],
    structure_name: str,
) -> np.ndarray:
    if structure_name not in structures:
        raise ValueError(f"Structure {structure_name} not found in CIFTI brain models")
    struct_slice, struct_bm = structures[structure_name]
    vals = np.asarray(frame_vec[struct_slice], dtype=float)
    struct_bm_any = cast(Any, struct_bm)
    vertex = np.asarray(struct_bm_any.vertex, dtype=np.int64)
    nverts_dict = getattr(struct_bm_any, "nvertices", None)
    if isinstance(nverts_dict, dict) and structure_name in nverts_dict:
        n_verts = int(nverts_dict[structure_name])
    else:
        n_verts = int(vertex.max()) + 1 if vertex.size else int(vals.size)
    if int(vertex.size) != int(vals.size):
        raise ValueError(
            f"BrainModel vertex index length ({int(vertex.size)}) does not match values length ({int(vals.size)})"
        )
    out = np.full(n_verts, np.nan, dtype=float)
    out[vertex] = vals
    return out


def _load_cifti_metric_source(path: Path, *, index: int) -> _MetricSource:
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, nib.cifti2.cifti2.Cifti2Image):
        raise TypeError(f"Expected CIFTI-2 image, got {type(loaded)}")

    img = loaded
    _frame_axis, brain_axis, frame_spec = _infer_cifti_axes(img)
    structures = _extract_cortex_structures(brain_axis)
    n_frames = int(frame_spec.n_frames)
    if n_frames <= 0:
        raise ValueError("No frames found")
    if index < 0 or index >= n_frames:
        raise ValueError(f"--index must be in [0, {n_frames - 1}], got {index}")

    def _get_frame_maps(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        if frame_index < 0 or frame_index >= n_frames:
            raise ValueError(f"frame index out of range: {frame_index}")
        frame_vec = _get_cifti_frame_vector(
            img, frame_index=frame_index, frame_spec=frame_spec
        )
        left_map = _brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
        )
        right_map = _brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
        )
        return left_map, right_map

    return _MetricSource(
        n_frames=n_frames, output_hint=path, get_frame_maps=_get_frame_maps
    )


def _extract_cifti_label_legend(
    atlas_img: nib.cifti2.cifti2.Cifti2Image,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    try:
        from nibabel.cifti2 import cifti2_axes

        ax0 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(0))
        ax1 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(1))
        label_axis = None
        if isinstance(ax0, cifti2_axes.LabelAxis):
            label_axis = ax0
        elif isinstance(ax1, cifti2_axes.LabelAxis):
            label_axis = ax1
        if label_axis is not None:
            labels_any = cast(Any, getattr(label_axis, "label", None))
            if labels_any is not None and len(labels_any) > 0:
                entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
                for key, value in cast(Any, labels_any[0]).items():
                    try:
                        name, rgba = value
                        rgba4 = tuple(float(x) for x in rgba)
                    except Exception:
                        continue
                    if len(rgba4) != 4:
                        continue
                    entries.append(
                        (int(key), str(name), _normalize_rgba(cast(Any, rgba4)))
                    )
                entries.sort(key=lambda item: item[0])
                return entries
    except Exception:
        pass

    try:
        idx_map0 = atlas_img.header.get_index_map(0)
        named_maps = cast(Any, getattr(idx_map0, "named_maps", None))
        if named_maps is not None and len(named_maps) > 0:
            label_table = cast(Any, getattr(named_maps[0], "label_table", None))
            if label_table is not None:
                entries = []
                for key, entry in cast(Any, label_table).items():
                    rgba = getattr(entry, "rgba", None)
                    if rgba is None:
                        continue
                    rgba4 = tuple(float(x) for x in rgba)
                    if len(rgba4) != 4:
                        continue
                    name = (
                        getattr(entry, "label", None)
                        or getattr(entry, "name", None)
                        or str(entry)
                    )
                    entries.append(
                        (int(key), str(name), _normalize_rgba(cast(Any, rgba4)))
                    )
                entries.sort(key=lambda item: item[0])
                return entries
    except Exception:
        pass
    return []


def _load_cifti_overlay(args: argparse.Namespace) -> _OverlayData | None:
    atlas_path = cast(Path | None, args.atlas)
    if atlas_path is None:
        return None
    if not atlas_path.exists():
        raise FileNotFoundError(str(atlas_path))
    atlas_img = nib_load(str(atlas_path))
    if not isinstance(atlas_img, nib.cifti2.cifti2.Cifti2Image):
        raise TypeError(f"Expected atlas CIFTI-2 image, got {type(atlas_img)}")

    _frame_axis, brain_axis, frame_spec = _infer_cifti_axes(atlas_img)
    frame_vec = _get_cifti_frame_vector(atlas_img, frame_index=0, frame_spec=frame_spec)
    structures = _extract_cortex_structures(brain_axis)
    labels = _extract_cifti_label_legend(atlas_img) if bool(args.atlas_legend) else []
    return _prepare_label_overlay(
        left_labels=_brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
        ),
        right_labels=_brain_to_hemi_vertices(
            frame_vec=frame_vec,
            structures=structures,
            structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
        ),
        labels=labels,
        cmap_name=str(args.atlas_cmap),
        ignore_zero=bool(args.atlas_ignore_zero),
        use_label_table=bool(args.atlas_legend) and len(labels) > 0,
        output_hint=atlas_path,
        output_tag="atlas",
    )


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


def _strip_suffixes(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


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


def _detect_hemi_token(name: str) -> Literal["left", "right"] | None:
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
    kind: str,
) -> tuple[Path, Path]:
    if left_path is not None or right_path is not None:
        if left_path is None or right_path is None:
            raise ValueError(
                f"Provide both --{kind}-left and --{kind}-right, or just --{kind}."
            )
        return left_path, right_path
    if single_path is None:
        raise ValueError(f"No {kind} input provided")
    suffix = ".func.gii" if kind == "input" else ".label.gii"
    return _pair_from_single_path(single_path, suffix=suffix)


def _resolve_gifti_pairs(
    *,
    single_paths: list[Path] | None,
    left_paths: list[Path] | None,
    right_paths: list[Path] | None,
    kind: str,
) -> list[_GiftiLabelPair]:
    single_paths = single_paths or []
    left_paths = left_paths or []
    right_paths = right_paths or []

    if single_paths and (left_paths or right_paths):
        raise ValueError(
            f"Do not mix --{kind} with --{kind}-left/--{kind}-right in the same command."
        )

    pairs: list[_GiftiLabelPair] = []
    if left_paths or right_paths:
        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Provide the same number of --{kind}-left and --{kind}-right arguments."
            )
        for left_path, right_path in zip(left_paths, right_paths):
            pairs.append(_GiftiLabelPair(left_path=left_path, right_path=right_path))
        return pairs

    for single_path in single_paths:
        if kind == "label" and single_path.exists():
            hemi = _detect_hemi_token(single_path.name)
            if hemi == "left":
                pairs.append(_GiftiLabelPair(left_path=single_path, right_path=None))
                continue
            if hemi == "right":
                pairs.append(_GiftiLabelPair(left_path=None, right_path=single_path))
                continue

        left_path, right_path = _resolve_gifti_pair(
            single_path=single_path,
            left_path=None,
            right_path=None,
            kind=kind,
        )
        pairs.append(_GiftiLabelPair(left_path=left_path, right_path=right_path))
    return pairs


def _empty_overlay_like(size: int) -> np.ndarray:
    return np.full((size,), np.nan, dtype=float)


def _load_gifti_metric_source(args: argparse.Namespace) -> _MetricSource | None:
    input_single = cast(Path | None, args.input)
    input_left = cast(Path | None, args.input_left)
    input_right = cast(Path | None, args.input_right)
    if input_single is None and input_left is None and input_right is None:
        return None

    left_path, right_path = _resolve_gifti_pair(
        single_path=input_single,
        left_path=input_left,
        right_path=input_right,
        kind="input",
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

    index = int(args.index)
    if index < 0 or index >= n_frames:
        raise ValueError(f"--index must be in [0, {n_frames - 1}], got {index}")
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
        n_frames=n_frames, output_hint=left_path, get_frame_maps=_get_frame_maps
    )


def _extract_gifti_label_legend(
    img: GiftiImage,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    label_table = getattr(img, "labeltable", None)
    labels_any = getattr(label_table, "labels", None)
    if labels_any is None:
        return []

    entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for entry in labels_any:
        try:
            key = int(getattr(entry, "key"))
            name = str(
                getattr(entry, "label", None) or getattr(entry, "name", None) or key
            )
            rgba_any = getattr(entry, "rgba", None)
            if rgba_any is None:
                rgba_any = (
                    getattr(entry, "red", 0.0),
                    getattr(entry, "green", 0.0),
                    getattr(entry, "blue", 0.0),
                    getattr(entry, "alpha", 1.0),
                )
            rgba = tuple(float(x) for x in cast(Any, rgba_any))
        except Exception:
            continue
        if len(rgba) != 4:
            continue
        entries.append((key, name, _normalize_rgba(cast(Any, rgba))))
    entries.sort(key=lambda item: item[0])
    return entries


def _label_display_name(
    *,
    left_path: Path,
    label_value: int,
    label_lookup: dict[int, tuple[str, tuple[float, float, float, float]]],
    prefix_with_file: bool,
) -> str:
    stem = _strip_suffixes(left_path.name, (".label.gii", ".gii"))
    if label_value in label_lookup:
        base_name = str(label_lookup[label_value][0])
        if base_name in {"", "???", str(label_value)}:
            base_name = stem
        elif prefix_with_file:
            base_name = f"{stem}: {base_name}"
        return base_name
    return stem if prefix_with_file else f"{stem}: {label_value}"


def _merge_label_legends(
    *legend_lists: list[tuple[int, str, tuple[float, float, float, float]]],
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    merged: dict[int, tuple[str, tuple[float, float, float, float]]] = {}
    for legend in legend_lists:
        for key, name, rgba in legend:
            if key not in merged or merged[key][0] in {"", "???", str(key)}:
                merged[key] = (name, rgba)
    return [(key, name, rgba) for key, (name, rgba) in sorted(merged.items())]


def _load_gifti_overlay(args: argparse.Namespace) -> _OverlayData | None:
    label_single = cast(list[Path] | None, args.label)
    label_left = cast(list[Path] | None, args.label_left)
    label_right = cast(list[Path] | None, args.label_right)
    if not label_single and not label_left and not label_right:
        return None

    label_pairs = _resolve_gifti_pairs(
        single_paths=label_single,
        left_paths=label_left,
        right_paths=label_right,
        kind="label",
    )
    if not label_pairs:
        return None

    if len(label_pairs) == 1:
        left_path = label_pairs[0].left_path
        right_path = label_pairs[0].right_path
        left_img = (
            _load_gifti_image(left_path, kind="label")
            if left_path is not None
            else None
        )
        right_img = (
            _load_gifti_image(right_path, kind="label")
            if right_path is not None
            else None
        )
        if left_img is None and right_img is None:
            return None
        if left_img is not None and len(left_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")
        if right_img is not None and len(right_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")

        fallback_size = 0
        if left_img is not None:
            fallback_size = int(np.asarray(left_img.darrays[0].data).size)
        elif right_img is not None:
            fallback_size = int(np.asarray(right_img.darrays[0].data).size)

        left_labels = (
            np.asarray(left_img.darrays[0].data, dtype=float).ravel()
            if left_img is not None
            else _empty_overlay_like(fallback_size)
        )
        right_labels = (
            np.asarray(right_img.darrays[0].data, dtype=float).ravel()
            if right_img is not None
            else _empty_overlay_like(fallback_size)
        )

        labels = []
        if bool(args.label_legend):
            labels = _merge_label_legends(
                _extract_gifti_label_legend(left_img) if left_img is not None else [],
                _extract_gifti_label_legend(right_img) if right_img is not None else [],
            )
        return _prepare_label_overlay(
            left_labels=left_labels,
            right_labels=right_labels,
            labels=labels,
            cmap_name=str(args.label_cmap),
            ignore_zero=bool(args.label_ignore_zero),
            use_label_table=bool(args.label_legend) and len(labels) > 0,
            output_hint=left_path if left_path is not None else cast(Path, right_path),
            output_tag="label",
        )

    combined_left: np.ndarray | None = None
    combined_right: np.ndarray | None = None
    combined_entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
    pending_entries: list[tuple[int, str]] = []
    next_label = 1
    prefix_with_file = len(label_pairs) > 1

    for pair in label_pairs:
        left_img = (
            _load_gifti_image(pair.left_path, kind="label")
            if pair.left_path is not None
            else None
        )
        right_img = (
            _load_gifti_image(pair.right_path, kind="label")
            if pair.right_path is not None
            else None
        )
        if left_img is None and right_img is None:
            continue
        if left_img is not None and len(left_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")
        if right_img is not None and len(right_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")

        if left_img is not None:
            left_data = np.asarray(left_img.darrays[0].data, dtype=float).ravel()
        else:
            assert right_img is not None
            left_data = _empty_overlay_like(
                int(np.asarray(right_img.darrays[0].data).size)
            )
        if right_img is not None:
            right_data = np.asarray(right_img.darrays[0].data, dtype=float).ravel()
        else:
            assert left_img is not None
            right_data = _empty_overlay_like(
                int(np.asarray(left_img.darrays[0].data).size)
            )

        if combined_left is None:
            combined_left = np.full(left_data.shape, np.nan, dtype=float)
            combined_right = np.full(right_data.shape, np.nan, dtype=float)
        else:
            assert combined_right is not None
            if (
                combined_left.shape != left_data.shape
                or combined_right.shape != right_data.shape
            ):
                raise ValueError(
                    "All label GIFTI files must have matching vertex counts"
                )

        label_lookup = {
            key: (name, rgba)
            for key, name, rgba in _merge_label_legends(
                _extract_gifti_label_legend(left_img) if left_img is not None else [],
                _extract_gifti_label_legend(right_img) if right_img is not None else [],
            )
        }
        unique_values = np.unique(np.concatenate((left_data, right_data), axis=0))
        for label_value in unique_values:
            if not np.isfinite(label_value):
                continue
            label_int = int(label_value)
            if bool(args.label_ignore_zero) and label_int == 0:
                continue
            left_mask = left_data == label_value
            right_mask = right_data == label_value
            if not np.any(left_mask) and not np.any(right_mask):
                continue
            combined_left[left_mask] = float(next_label)
            combined_right[right_mask] = float(next_label)
            pending_entries.append(
                (
                    next_label,
                    _label_display_name(
                        left_path=pair.left_path
                        if pair.left_path is not None
                        else cast(Path, pair.right_path),
                        label_value=label_int,
                        label_lookup=label_lookup,
                        prefix_with_file=prefix_with_file,
                    ),
                )
            )
            next_label += 1

    if combined_left is None or combined_right is None:
        return None

    colors = _sample_overlay_colors(str(args.label_cmap), len(pending_entries))
    combined_entries = [
        (label_id, name, colors[idx])
        for idx, (label_id, name) in enumerate(pending_entries)
    ]
    cmap = ListedColormap(
        [rgba for _key, _name, rgba in combined_entries], name="label_multi"
    )
    cmap = cast(Colormap, _copy_cmap_with_transparent_bad(cmap))
    legend_entries = combined_entries if bool(args.label_legend) else []
    return _OverlayData(
        left_plot=combined_left,
        right_plot=combined_right,
        cmap=cmap,
        labels=legend_entries,
        vmin=(1.0 if combined_entries else None),
        vmax=(float(len(combined_entries)) if combined_entries else None),
        output_hint=label_pairs[0].left_path
        if label_pairs[0].left_path is not None
        else cast(Path, label_pairs[0].right_path),
        output_tag="label",
    )


def _prepare_label_overlay(
    *,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
    cmap_name: str,
    ignore_zero: bool,
    use_label_table: bool,
    output_hint: Path,
    output_tag: str,
) -> _OverlayData:
    left = np.asarray(left_labels, dtype=float).copy()
    right = np.asarray(right_labels, dtype=float).copy()
    if ignore_zero:
        left[left == 0] = np.nan
        right[right == 0] = np.nan

    if use_label_table and len(labels) > 0:
        entries = labels
        if ignore_zero:
            entries = [entry for entry in entries if int(entry[0]) != 0]
        keys = np.array([int(key) for key, _name, _rgba in entries], dtype=np.int64)
        colors = [rgba for _key, _name, rgba in entries]
        if keys.size > 0:
            order = np.argsort(keys)
            keys = keys[order]
            colors = [colors[int(i)] for i in order]
        cmap = ListedColormap(colors, name=f"{output_tag}_label_table")
        cmap = cast(Colormap, _copy_cmap_with_transparent_bad(cmap))

        def _remap(arr: np.ndarray) -> np.ndarray:
            values = np.asarray(arr, dtype=float)
            out = np.full(values.shape, np.nan, dtype=float)
            finite_mask = np.isfinite(values)
            if not np.any(finite_mask) or keys.size == 0:
                return out
            label_vals = values[finite_mask].astype(np.int64, copy=False)
            idx = np.searchsorted(keys, label_vals)
            ok = (idx >= 0) & (idx < keys.size) & (keys[idx] == label_vals)
            tmp = np.full(label_vals.shape, np.nan, dtype=float)
            # nilearn treats 0 as background for ROI maps, so visible labels
            # must be remapped onto 1..N rather than 0..N-1.
            tmp[ok] = idx[ok].astype(float) + 1.0
            out[finite_mask] = tmp
            return out

        left_plot = _remap(left)
        right_plot = _remap(right)
        return _OverlayData(
            left_plot=left_plot,
            right_plot=right_plot,
            cmap=cmap,
            labels=entries,
            vmin=(1.0 if len(entries) > 0 else None),
            vmax=(float(len(entries)) if len(entries) > 0 else None),
            output_hint=output_hint,
            output_tag=output_tag,
        )

    cmap = _copy_cmap_with_transparent_bad(cmap_name)
    finite_values = np.concatenate(
        (left[np.isfinite(left)], right[np.isfinite(right)]), axis=0
    )
    return _OverlayData(
        left_plot=left,
        right_plot=right,
        cmap=cmap,
        labels=labels,
        vmin=(float(np.min(finite_values)) if finite_values.size > 0 else None),
        vmax=(float(np.max(finite_values)) if finite_values.size > 0 else None),
        output_hint=output_hint,
        output_tag=output_tag,
    )


def _derive_default_output(
    path: Path, *, index: int, suffixes: tuple[str, ...]
) -> Path:
    return path.with_name(f"{_strip_suffixes(path.name, suffixes)}_idx-{index}.png")


def _derive_default_overlay_output(
    path: Path, *, tag: str, suffixes: tuple[str, ...]
) -> Path:
    return path.with_name(f"{_strip_suffixes(path.name, suffixes)}_{tag}.png")


def _select_output_path(args: argparse.Namespace, scene: _SceneData) -> Path:
    if args.output is not None:
        return Path(args.output)
    if scene.metric_source is not None:
        return _derive_default_output(
            scene.metric_source.output_hint,
            index=int(args.index),
            suffixes=(
                ".dtseries.nii",
                ".dscalar.nii",
                ".func.gii",
                ".nii.gz",
                ".nii",
                ".gii",
            ),
        )
    if scene.overlay is not None:
        return _derive_default_overlay_output(
            scene.overlay.output_hint,
            tag=scene.overlay.output_tag,
            suffixes=(
                ".dlabel.nii",
                ".label.gii",
                ".nii.gz",
                ".nii",
                ".gii",
            ),
        )
    raise RuntimeError("No metric or overlay input available to derive output path")


def _default_surface_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "template" / filename


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a static snapshot PNG from CIFTI or GIFTI surface data.",
    )
    p.add_argument(
        "--format",
        choices=["auto", "gifti", "cifti"],
        default="auto",
        help="Input format selection. Defaults to auto-detect.",
    )
    p.add_argument(
        "--input",
        required=False,
        type=Path,
        default=None,
        help="CIFTI metric file or a single GIFTI hemisphere file/prefix to infer the mate file.",
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
        help="Output PNG path. Defaults to a path derived from the metric or overlay input.",
    )
    p.add_argument(
        "--index", type=int, default=0, help="Frame index to display (default: 0)."
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
        "--sulc-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.sulc.32k_fs_LR.surf.gii"),
        help="Left hemisphere sulcal depth GIFTI.",
    )
    p.add_argument(
        "--sulc-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.sulc.32k_fs_LR.surf.gii"),
        help="Right hemisphere sulcal depth GIFTI.",
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
        help=(
            "Per-hemisphere override for camera elevation (LEFT). "
            "If not provided, falls back to --surf-elev."
        ),
    )
    p.add_argument(
        "--surf-elev-right",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera elevation (RIGHT). "
            "If not provided, falls back to --surf-elev."
        ),
    )
    p.add_argument(
        "--surf-azim-left",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera azimuth (LEFT). "
            "If not provided, falls back to --surf-azim."
        ),
    )
    p.add_argument(
        "--surf-azim-right",
        type=float,
        default=None,
        help=(
            "Per-hemisphere override for camera azimuth (RIGHT). "
            "If not provided, falls back to --surf-azim."
        ),
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
        "--black-bg", action="store_true", help="Use black figure background."
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
        help="Optional title template. Use {index}, {panel}, {view}, {hemi}, {time}, {format} placeholders.",
    )
    p.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help="Optional CIFTI dense label atlas (.dlabel.nii) overlay.",
    )
    p.add_argument(
        "--mesh-alpha", type=float, default=1.0, help="Alpha for CIFTI mesh."
    )
    p.add_argument(
        "--atlas-cmap",
        type=str,
        default="tab20",
        help="Colormap for CIFTI atlas overlay.",
    )
    p.add_argument(
        "--atlas-ignore-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat atlas label 0 as background (default: true).",
    )
    p.add_argument(
        "--atlas-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render a legend using label names/colors from the atlas.",
    )
    p.add_argument(
        "--atlas-legend-max-items",
        type=int,
        default=40,
        help="Max CIFTI atlas legend entries to show (default: 40).",
    )
    p.add_argument(
        "--atlas-view-type",
        type=str,
        default="contour",
        choices=["contour", "continuous"],
        help=(
            "How to render CIFTI atlas overlays. 'contour' draws colored borders between "
            "atlas regions, while 'continuous' colors the full region interiors."
        ),
    )

    p.add_argument(
        "--label",
        type=Path,
        action="append",
        default=None,
        help="GIFTI label path or prefix for mate-file inference. Repeat to overlay multiple label files.",
    )
    p.add_argument(
        "--label-left",
        type=Path,
        action="append",
        default=None,
        help="Left hemisphere label GIFTI (.label.gii). Repeat alongside --label-right to overlay multiple label pairs.",
    )
    p.add_argument(
        "--label-right",
        type=Path,
        action="append",
        default=None,
        help="Right hemisphere label GIFTI (.label.gii). Repeat alongside --label-left to overlay multiple label pairs.",
    )
    p.add_argument(
        "--label-cmap",
        type=str,
        default="tab20",
        help="Colormap for GIFTI label overlay.",
    )
    p.add_argument(
        "--label-ignore-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat label 0 as background (default: true).",
    )
    p.add_argument(
        "--label-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render a legend using label names/colors from the GIFTI label table.",
    )
    p.add_argument(
        "--label-legend-max-items",
        type=int,
        default=40,
        help="Max GIFTI label legend entries to show (default: 40).",
    )

    p.add_argument(
        "--time-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate snapshot with time in seconds (default: false).",
    )
    p.add_argument("--tr", type=float, default=None, help="Repetition time in seconds.")
    p.add_argument(
        "--t0-trs", type=float, default=0.0, help="Starting time in TRs for index 0."
    )

    p.add_argument(
        "--dpi", type=int, default=150, help="DPI for saved PNG (default: 150)."
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 720),
        help="Figure size in pixels (default: 1280 720).",
    )
    p.add_argument(
        "--sulc-file-reverse-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reverse the sign of the sulcal depth file (default: true).",
    )
    return p


def _detect_format(args: argparse.Namespace) -> str:
    if str(args.format) != "auto":
        return str(args.format)
    if args.input_left is not None or args.input_right is not None:
        return "gifti"
    if (
        args.label is not None
        or args.label_left is not None
        or args.label_right is not None
    ):
        return "gifti"
    if args.atlas is not None:
        return "cifti"
    input_path = cast(Path | None, args.input)
    if input_path is not None:
        name = input_path.name
        if name.endswith(
            (".dtseries.nii", ".dscalar.nii", ".dlabel.nii", ".nii", ".nii.gz")
        ):
            return "cifti"
        if name.endswith(".gii") or any(
            token in name for token in ("lh.", "rh.", ".L.", ".R.")
        ):
            return "gifti"
    raise ValueError(
        "Could not infer input format. Use --format gifti or --format cifti."
    )


def _load_scene(args: argparse.Namespace) -> _SceneData:
    source_format = _detect_format(args)
    has_cifti_overlay = args.atlas is not None
    has_gifti_overlay = any(
        value is not None for value in (args.label, args.label_left, args.label_right)
    )
    if has_cifti_overlay and has_gifti_overlay:
        raise ValueError(
            "Use either --atlas or --label/--label-left/--label-right, not both"
        )
    if source_format == "cifti":
        metric_source = None
        input_path = cast(Path | None, args.input)
        if input_path is not None:
            metric_source = _load_cifti_metric_source(input_path, index=int(args.index))
        overlay = _load_cifti_overlay(args) if has_cifti_overlay else None
        if metric_source is None and overlay is None:
            raise ValueError("CIFTI mode requires --input and/or --atlas")
        return _SceneData(
            metric_source=metric_source, overlay=overlay, source_format=source_format
        )

    metric_source = _load_gifti_metric_source(args)
    overlay = (
        _load_cifti_overlay(args) if has_cifti_overlay else _load_gifti_overlay(args)
    )
    if metric_source is None and overlay is None:
        raise ValueError("GIFTI mode requires metric input and/or label or atlas input")
    return _SceneData(
        metric_source=metric_source, overlay=overlay, source_format=source_format
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scene = _load_scene(args)

    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))
    surf_left_sulc = surface.load_surf_data(str(args.sulc_left))
    surf_right_sulc = surface.load_surf_data(str(args.sulc_right))
    if bool(args.sulc_file_reverse_sign):
        surf_left_sulc = -surf_left_sulc
        surf_right_sulc = -surf_right_sulc

    selected_left: np.ndarray | None = None
    selected_right: np.ndarray | None = None
    if scene.metric_source is not None:
        selected_left, selected_right = scene.metric_source.get_frame_maps(
            int(args.index)
        )
        validate_map_against_mesh(selected_left, surf_left_mesh, kind="Left metric")
        validate_map_against_mesh(selected_right, surf_right_mesh, kind="Right metric")
    if scene.overlay is not None:
        validate_map_against_mesh(
            scene.overlay.left_plot, surf_left_mesh, kind="Left overlay"
        )
        validate_map_against_mesh(
            scene.overlay.right_plot, surf_right_mesh, kind="Right overlay"
        )

    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )
    vmin, vmax = compute_surface_intensity_bounds(
        scene.metric_source,
        selected_index=int(args.index),
        intensity_mode=str(args.intensity_mode),
        p_low=p_low,
        p_high=p_high,
        max_samples=int(args.auto_max_samples),
        max_total_samples=int(args.auto_max_total_samples),
        vmin_arg=args.vmin,
        vmax_arg=args.vmax,
    )

    time_s: float | None = None
    if bool(args.time_annotate):
        if scene.metric_source is None:
            raise ValueError("--time-annotate requires metric input")
        if args.tr is None:
            raise ValueError("--tr is required when --time-annotate is true")
        time_s = (float(args.t0_trs) + float(args.index)) * float(args.tr)

    overlay = None
    if scene.overlay is not None:
        overlay = SurfaceOverlay(
            left_plot=scene.overlay.left_plot,
            right_plot=scene.overlay.right_plot,
            cmap=scene.overlay.cmap,
            labels=scene.overlay.labels,
            vmin=scene.overlay.vmin,
            vmax=scene.overlay.vmax,
            output_tag=scene.overlay.output_tag,
        )

    options = SurfaceRenderOptions(
        views=[str(view) for view in args.surf_views],
        figure_size=(
            int(args.size[0]) / int(args.dpi),
            int(args.size[1]) / int(args.dpi),
        ),
        dpi=int(args.dpi),
        ncols=int(args.ncols) if args.ncols is not None else None,
        cmap=str(args.cmap),
        vmin=vmin,
        vmax=vmax,
        mesh_alpha=float(args.mesh_alpha),
        surf_zoom=float(args.surf_zoom),
        black_bg=bool(args.black_bg),
        colorbar=bool(args.colorbar),
        colorbar_side=str(args.colorbar_side),
        title_template=args.title,
        source_format=scene.source_format,
        atlas_view_type=str(args.atlas_view_type),
        atlas_ignore_zero=bool(args.atlas_ignore_zero),
        atlas_legend=bool(args.atlas_legend),
        atlas_legend_max_items=int(args.atlas_legend_max_items),
        label_ignore_zero=bool(args.label_ignore_zero),
        label_legend=bool(args.label_legend),
        label_legend_max_items=int(args.label_legend_max_items),
        time_annotate=bool(args.time_annotate),
        time_s=time_s,
        global_camera=SurfaceCameraConfig(elev=args.surf_elev, azim=args.surf_azim),
        left_camera=SurfaceCameraConfig(
            elev=args.surf_elev_left, azim=args.surf_azim_left
        ),
        right_camera=SurfaceCameraConfig(
            elev=args.surf_elev_right, azim=args.surf_azim_right
        ),
    )

    rendered = render_surface_figure(
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        stat_map_left=selected_left,
        stat_map_right=selected_right,
        overlay=overlay,
        options=options,
        index=int(args.index),
    )

    output_path = _select_output_path(args, scene)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.figure.savefig(str(output_path), dpi=int(args.dpi))
    plt.close(rendered.figure)

    print(f"Wrote snapshot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
