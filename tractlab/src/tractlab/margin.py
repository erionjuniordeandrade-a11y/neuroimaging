"""World-space envelope around continuous streamlines.

This is a cosmetic display envelope, not a navigation or resection boundary.
The previous implementation rounded vertices into the tracking grid, used the
minimum voxel spacing as an isotropic radius, and then applied L1 dilation.
That can place a nominal 3 mm shell more than 7 mm from a path on a 1 x 1 x
3 mm grid.

The implementation below samples each continuous segment by bounded arclength,
builds an isotropic world-space distance field on a small orthonormal raster,
and extracts its surface.  The path sample spacing and raster spacing are
fixed and published so the mesh distance error is bounded by

    path_spacing / 2 + sqrt(3) * raster_spacing / 2.

The tracking affine is used only to identify the input grid; streamline points
are already world millimetres.  Consequently anisotropic, rotated, and sheared
tracking grids do not receive an invalid orthogonal-grid EDT treatment.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

from .grid import Grid


MARGIN_PATH_SAMPLE_SPACING_MM = 0.5
MARGIN_RASTER_SPACING_MM = 0.5
# A bounded allocation protects the viewer from an accidental whole-brain
# world raster.  The function refuses rather than silently increasing spacing
# and hiding a larger geometric error.
MARGIN_MAX_RASTER_VOXELS = 12_000_000
# Sampling is fixed at 0.5 mm so the published geometric error bound remains
# true.  Refuse paths that would require unbounded arrays at that resolution.
MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE = 1_000_000
MARGIN_MAX_PATH_SAMPLES_TOTAL = 2_000_000
# A product cap alone is vulnerable to integer overflow and an extreme thin
# slab can still create surprising temporary arrays.  Bound each axis before
# converting the shape to an integer dtype.
MARGIN_MAX_RASTER_DIM = 4096
MARGIN_MESH_ERROR_BOUND_MM = (
    MARGIN_PATH_SAMPLE_SPACING_MM / 2.0
    + np.sqrt(3.0) * MARGIN_RASTER_SPACING_MM / 2.0
)


def margin_error_bound_mm(
    *,
    path_spacing_mm: float = MARGIN_PATH_SAMPLE_SPACING_MM,
    raster_spacing_mm: float = MARGIN_RASTER_SPACING_MM,
) -> float:
    """Return the declared upper bound for world-space mesh distance error."""
    path_spacing = float(path_spacing_mm)
    raster_spacing = float(raster_spacing_mm)
    if not np.isfinite(path_spacing) or path_spacing <= 0.0:
        raise ValueError("path spacing must be finite and > 0")
    if not np.isfinite(raster_spacing) or raster_spacing <= 0.0:
        raise ValueError("raster spacing must be finite and > 0")
    return path_spacing / 2.0 + np.sqrt(3.0) * raster_spacing / 2.0


def _validate_line(line: np.ndarray) -> np.ndarray:
    arr = np.asarray(line, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"streamline must have shape (N, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite vertex in streamline")
    return arr


def _simplify_collinear(line: np.ndarray) -> np.ndarray:
    """Drop duplicate/same-direction collinear vertices without changing path."""
    arr = _validate_line(line)
    if arr.shape[0] <= 2:
        return arr.copy()
    keep = [arr[0]]
    for point in arr[1:]:
        if np.array_equal(point, keep[-1]):
            continue
        keep.append(point)
        while len(keep) >= 3:
            a, b, c = keep[-3:]
            first = b - a
            second = c - b
            n_first = float(np.linalg.norm(first))
            n_second = float(np.linalg.norm(second))
            if n_first == 0.0 or n_second == 0.0:
                keep.pop(-2)
                continue
            cross_norm = float(np.linalg.norm(np.cross(first, second)))
            if cross_norm <= 1e-10 * max(n_first * n_second, 1.0):
                # Preserve a cusp/reversal, which is a real path change.
                if float(np.dot(first, second)) >= 0.0:
                    keep.pop(-2)
                    continue
            break
    return np.asarray(keep, dtype=np.float64)


def _sample_polyline(line: np.ndarray, spacing_mm: float) -> np.ndarray:
    if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
        raise ValueError("margin sampling spacing must be finite and > 0")
    arr = _simplify_collinear(line)
    if arr.shape[0] <= 1:
        return arr
    seg = np.diff(arr, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("margin path length is not finite")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if not np.isfinite(total):
        raise ValueError("margin path length is not finite")
    if total == 0.0:
        return arr[:1].copy()
    ratio = total / float(spacing_mm)
    if not np.isfinite(ratio):
        raise ValueError("margin path samples would overflow at fixed spacing")
    n = max(1, int(np.ceil(ratio)))
    if n + 1 > MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE:
        raise ValueError(
            "margin path samples per streamline exceed bounded allocation "
            f"({n + 1} > {MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE}); "
            "refusing a coarser undocumented spacing"
        )
    distances = np.linspace(0.0, total, n + 1)
    ids = np.searchsorted(cumulative, distances, side="right") - 1
    ids = np.clip(ids, 0, len(seg) - 1)
    denom = lengths[ids]
    frac = np.divide(
        distances - cumulative[ids],
        denom,
        out=np.zeros_like(distances),
        where=denom > 0.0,
    )
    return arr[ids] + frac[:, None] * seg[ids]


def _path_sample_count(line: np.ndarray, spacing_mm: float) -> int:
    """Return the fixed-spacing sample count without allocating the samples."""
    if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
        raise ValueError("margin sampling spacing must be finite and > 0")
    arr = _simplify_collinear(line)
    if arr.shape[0] <= 1:
        return int(arr.shape[0])
    lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("margin path length is not finite")
    total = float(np.sum(lengths))
    if not np.isfinite(total):
        raise ValueError("margin path length is not finite")
    if total == 0.0:
        return 1
    ratio = total / float(spacing_mm)
    if not np.isfinite(ratio):
        raise ValueError("margin path samples would overflow at fixed spacing")
    n = max(1, int(np.ceil(ratio)))
    count = n + 1
    if count > MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE:
        raise ValueError(
            "margin path samples per streamline exceed bounded allocation "
            f"({count} > {MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE}); "
            "refusing a coarser undocumented spacing"
        )
    return count


def _length_mm(line: np.ndarray) -> float:
    arr = _validate_line(line)
    if arr.shape[0] <= 1:
        return 0.0
    lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = float(np.sum(lengths))
    if not np.isfinite(total):
        raise ValueError("margin path length is not finite")
    return total


def _world_raster(
    samples: np.ndarray,
    margin_mm: float,
    *,
    spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (volume, origin, shape) for a chunked world-space distance field."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] == 0:
        raise ValueError(f"margin samples must have shape (N, 3), got {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("margin samples must be finite")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
        raise ValueError("margin raster spacing must be finite and > 0")
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("margin raster radius must be finite and >= 0")
    radius = float(margin_mm) + MARGIN_PATH_SAMPLE_SPACING_MM / 2.0
    pad = radius + spacing_mm
    if not np.isfinite(radius) or not np.isfinite(pad):
        raise ValueError("margin raster extent is not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.floor((samples.min(axis=0) - pad) / spacing_mm) * spacing_mm
        hi = np.ceil((samples.max(axis=0) + pad) / spacing_mm) * spacing_mm
        span = hi - lo
        raw_shape = span / spacing_mm
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(raw_shape).all():
        raise ValueError("margin raster extent is not finite or exceeds bounded dimensions")
    if np.any(raw_shape < 0.0) or np.any(raw_shape > float(MARGIN_MAX_RASTER_DIM - 1)):
        largest = float(np.max(raw_shape))
        raise ValueError(
            "margin raster dimension exceeds bounded world raster "
            f"(requested axis span {largest:g} voxels, limit "
            f"{MARGIN_MAX_RASTER_DIM})"
        )
    shape = np.floor(raw_shape + 0.5).astype(np.int64) + 1
    if np.any(shape < 1) or np.any(shape > MARGIN_MAX_RASTER_DIM):
        raise ValueError("margin raster dimension is outside bounded range")
    n_voxels = math.prod(int(s) for s in shape)
    if n_voxels > MARGIN_MAX_RASTER_VOXELS:
        raise ValueError(
            "margin envelope exceeds bounded world raster; reduce the display "
            f"path subset (requested {n_voxels} voxels, limit "
            f"{MARGIN_MAX_RASTER_VOXELS}; declared error "
            f"<={margin_error_bound_mm():.3f} mm at supported resolution)"
        )
    tree = cKDTree(samples)
    volume = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    # Process x slabs so the temporary meshgrid and KD-tree query remain bounded.
    ny, nz = int(shape[1]), int(shape[2])
    slab_x = max(1, min(int(shape[0]), 64))
    ys = lo[1] + np.arange(ny, dtype=np.float64) * spacing_mm
    zs = lo[2] + np.arange(nz, dtype=np.float64) * spacing_mm
    for x0 in range(0, int(shape[0]), slab_x):
        x1 = min(int(shape[0]), x0 + slab_x)
        xs = lo[0] + np.arange(x0, x1, dtype=np.float64) * spacing_mm
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        distances, _ = tree.query(coords, k=1)
        volume[x0:x1] = (distances <= radius).reshape(x1 - x0, ny, nz)
    return volume, lo, shape


def build_margin_hull(
    streamlines: list[np.ndarray],
    grid: Grid,
    *,
    margin_mm: float = 5.0,
    step: int = 2,
    max_streamlines: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a world-space cosmetic envelope mesh.

    Up to ``max_streamlines`` fibres are selected by deterministic descending
    arclength for display cost.  The subset is cosmetic and is not a clinical
    clearance calculation or a resection boundary.  ``step`` remains in the
    API for compatibility; marching cubes always uses unit steps so its mesh
    error remains covered by :func:`margin_error_bound_mm`.
    """
    if not streamlines:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    try:
        margin = float(margin_mm)
    except (TypeError, ValueError):
        raise ValueError("margin_mm must be a number") from None
    if not (0.5 <= margin <= 30.0) or not np.isfinite(margin):
        raise ValueError("margin_mm must be finite and in [0.5, 30]")
    if int(max_streamlines) <= 0:
        raise ValueError("max_streamlines must be > 0")
    if int(step) <= 0:
        raise ValueError("step must be > 0")
    if np.asarray(grid.affine).shape != (4, 4) or not np.isfinite(grid.affine).all():
        raise ValueError("grid affine must be a finite 4x4 matrix")

    validated = [_validate_line(line) for line in streamlines]
    nonempty = [line for line in validated if line.shape[0] > 0]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    lines = nonempty
    if len(lines) > int(max_streamlines):
        lengths = np.asarray([_length_mm(line) for line in lines], dtype=np.float64)
        order = np.argsort(-lengths, kind="stable")
        lines = [lines[int(i)] for i in order[: int(max_streamlines)]]

    total_sample_count = 0
    for line in lines:
        total_sample_count += _path_sample_count(
            line, MARGIN_PATH_SAMPLE_SPACING_MM
        )
        if total_sample_count > MARGIN_MAX_PATH_SAMPLES_TOTAL:
            raise ValueError(
                "margin total path samples exceed bounded allocation "
                f"({total_sample_count} > {MARGIN_MAX_PATH_SAMPLES_TOTAL}); "
                "refusing a coarser undocumented spacing"
            )
    samples = np.concatenate(
        [_sample_polyline(line, MARGIN_PATH_SAMPLE_SPACING_MM) for line in lines],
        axis=0,
    )
    if samples.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    volume, origin, _shape = _world_raster(
        samples,
        margin,
        spacing_mm=MARGIN_RASTER_SPACING_MM,
    )
    if not volume.any() or min(volume.shape) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    try:
        verts_grid, faces, _, _ = measure.marching_cubes(
            volume.astype(np.float32),
            level=0.5,
            spacing=(MARGIN_RASTER_SPACING_MM,) * 3,
            step_size=1,
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    verts_world = (verts_grid + origin[None, :]).astype("<f4")
    return verts_world, faces.astype("<u4")
