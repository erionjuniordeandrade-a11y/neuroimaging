"""Containment diagnostics on source streamline geometry.

This module measures whether original world-mm centerlines remain in a supplied
ACT/support mask. It never changes a display mesh, dilates support, or reads a
viewer payload: an apparent exit remains an explicit diagnostic result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .grid import Grid, in_bounds, load_grid, voxel_to_world, world_to_voxel
from .surface import validate_surface_grid


@dataclass(frozen=True)
class ContainmentDiagnostic:
    """Counts and nearest-support-centre error only; never copies path geometry.

    ``max_nearest_support_center_mm`` is the maximum world-mm distance to a
    support *voxel centre*. It is not a distance to a mask boundary or a
    clearance measurement.
    """

    streamline_count: int
    source_vertex_count: int
    sample_count: int
    outside_streamline_count: int
    outside_sample_count: int
    max_nearest_support_center_mm: float

    def summary(self) -> dict[str, int | float]:
        return {
            "streamline_count": self.streamline_count,
            "source_vertex_count": self.source_vertex_count,
            "sample_count": self.sample_count,
            "outside_streamline_count": self.outside_streamline_count,
            "outside_sample_count": self.outside_sample_count,
            "max_nearest_support_center_mm": self.max_nearest_support_center_mm,
        }


def _default_step_mm(grid: Grid) -> float:
    spacing = np.linalg.norm(np.asarray(grid.affine, dtype=np.float64)[:3, :3], axis=0)
    step = float(spacing.min()) / 2.0
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("support grid has no positive sampling step")
    return step


def _dense_source_line(line: np.ndarray, spacing_mm: float) -> np.ndarray:
    points = np.asarray(line, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("streamline must be shaped (N,3)")
    if points.shape[0] == 0:
        return points
    if not np.isfinite(points).all():
        raise ValueError("non-finite vertex in source streamline")
    if points.shape[0] == 1:
        return points
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    count = np.maximum(1, np.ceil(lengths / spacing_mm).astype(np.int64))
    total = int(count.sum())
    segment_ids = np.repeat(np.arange(len(count)), count)
    local = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    fraction = local / count[segment_ids]
    return np.vstack([points[segment_ids] + segments[segment_ids] * fraction[:, None], points[-1:]])


def _nearest_support_center_distance_mm(
    outside_world: np.ndarray,
    support: np.ndarray,
    grid: Grid,
) -> float:
    """Return nearest positive-support voxel-centre distance, never boundary distance."""
    if outside_world.size == 0:
        return 0.0
    support_ijk = np.argwhere(support)
    if support_ijk.size == 0:
        raise ValueError("support mask is empty")
    support_world = voxel_to_world(grid, support_ijk)
    distances, _ = cKDTree(support_world).query(outside_world, k=1)
    return float(np.max(distances))


def _positive_support_mask(support_mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    raw_support = np.asarray(support_mask)
    if raw_support.shape != shape:
        raise ValueError(f"support shape {raw_support.shape} != grid shape {shape}")
    if raw_support.dtype.kind not in "biuf":
        raise ValueError("support mask must be boolean or real-valued")
    # ACT/support labels are positive. Casting -1 or NaN directly to bool would
    # fabricate support and conceal an outside-path diagnostic.
    return np.isfinite(raw_support) & (raw_support > 0)


def diagnose_streamline_containment(
    streamlines: Iterable[np.ndarray],
    support_mask: np.ndarray,
    grid: Grid,
    *,
    step_mm: float | None = None,
) -> ContainmentDiagnostic:
    """Diagnose source-line exits from an ACT/support mask without altering it.

    Dense sampling catches an unsupported interval between original vertices;
    callers must pass source centerlines, not resampled display splines. A
    point outside the field of view remains outside and is never clamped.
    """
    validate_surface_grid(grid)
    support = _positive_support_mask(support_mask, tuple(grid.shape))
    if not bool(support.any()):
        raise ValueError("support mask is empty")
    spacing = _default_step_mm(grid) if step_mm is None else float(step_mm)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("step_mm must be finite and > 0")

    line_count = 0
    source_vertex_count = 0
    sample_count = 0
    outside_streamline_count = 0
    outside_samples: list[np.ndarray] = []
    for line in streamlines:
        source = np.asarray(line, dtype=np.float64)
        if source.ndim != 2 or source.shape[1:] != (3,):
            raise ValueError("streamline must be shaped (N,3)")
        line_count += 1
        source_vertex_count += int(source.shape[0])
        dense = _dense_source_line(source, spacing)
        sample_count += int(dense.shape[0])
        if dense.shape[0] == 0:
            continue
        voxel = world_to_voxel(grid, dense)
        ijk = np.rint(voxel).astype(np.int64)
        in_grid = in_bounds(grid, ijk)
        contained = np.zeros(dense.shape[0], dtype=bool)
        if bool(in_grid.any()):
            valid = ijk[in_grid]
            contained[in_grid] = support[valid[:, 0], valid[:, 1], valid[:, 2]]
        outside = ~contained
        if bool(outside.any()):
            outside_streamline_count += 1
            outside_samples.append(dense[outside])

    outside_world = np.vstack(outside_samples) if outside_samples else np.empty((0, 3), dtype=float)
    return ContainmentDiagnostic(
        streamline_count=line_count,
        source_vertex_count=source_vertex_count,
        sample_count=sample_count,
        outside_streamline_count=outside_streamline_count,
        outside_sample_count=int(outside_world.shape[0]),
        max_nearest_support_center_mm=_nearest_support_center_distance_mm(
            outside_world, support, grid,
        ),
    )


def diagnose_tck_containment(
    tck_path: str,
    support_path: str,
    *,
    step_mm: float | None = None,
) -> ContainmentDiagnostic:
    """Load original TCK centerlines and diagnose them against a support NIfTI.

    The result contains aggregate counts and nearest-support-voxel-centre mm
    only; it is not a boundary-distance/clearance measurement. It intentionally
    exposes no subject identifiers, paths, or copied points.
    """
    support_img = nib.load(support_path)
    raw_support = np.asarray(support_img.dataobj)
    if raw_support.ndim != 3:
        raise ValueError(f"support mask must be 3-D, got shape {raw_support.shape}")
    if raw_support.dtype.kind not in "biuf":
        raise ValueError("support mask must be boolean or real-valued")
    support = np.isfinite(raw_support) & (raw_support > 0)
    grid = validate_surface_grid(load_grid(support_path))
    original_streamlines = nib.streamlines.load(tck_path).streamlines
    return diagnose_streamline_containment(
        original_streamlines, support, grid, step_mm=step_mm,
    )
