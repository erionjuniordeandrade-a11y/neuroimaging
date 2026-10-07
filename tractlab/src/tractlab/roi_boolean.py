"""Boolean ROI algebra for Commit-mode iFOD2 tracking.

Maps interactive paint layers to MRtrix ``tckgen`` ROI flags:

  SEED → union of seed strokes → one ``-seed_image`` (required; empty fails loud)
  AND  → each non-empty region separate → one ``-include`` per region
  OR   → union of OR strokes → one additional ``-include`` (omit if empty)
  NOT  → union of NOT strokes → one ``-exclude`` (omit if empty)

The client never supplies paths or flags — only world-mm points + radii.
Rasterization reuses ``seed.rasterize_points`` (no-clamp OOB, world-mm distance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectotomy import reject_cavity_in_request
from .grid import Grid
from .seed import rasterize_points


class RoiCompileError(ValueError):
    """Invalid or empty ROI request — map to HTTP 400 at the API boundary."""


MAX_ROI_REGIONS = 64
MAX_ROI_POINTS_TOTAL = 50_000


@dataclass(frozen=True)
class Stroke:
    """One continuous paint stroke: world-mm points + brush radius (mm)."""

    points_mm: np.ndarray  # (N, 3)
    radius_mm: float

    def __post_init__(self) -> None:
        pts = np.atleast_2d(np.asarray(self.points_mm, dtype=np.float64))
        object.__setattr__(self, "points_mm", pts)
        if pts.ndim != 2 or pts.shape[-1] != 3:
            raise RoiCompileError("points_mm must be (N,3) world coordinates")
        if not (0.0 < float(self.radius_mm) <= 20.0):
            raise RoiCompileError("radius_mm out of range (0, 20]")


# One AND *region* is a list of strokes that are unioned within the region.
AndRegion = list[Stroke]


@dataclass(frozen=True)
class RoiLayers:
    seed: list[Stroke]
    and_regions: list[AndRegion] = field(default_factory=list)
    or_regions: list[Stroke] = field(default_factory=list)
    not_regions: list[Stroke] = field(default_factory=list)


@dataclass(frozen=True)
class CompiledRois:
    seed_mask: np.ndarray
    and_masks: list[np.ndarray]
    or_mask: np.ndarray | None
    not_mask: np.ndarray | None


def _union_strokes(grid: Grid, strokes: list[Stroke]) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for s in strokes:
        if s.points_mm.size == 0:
            continue
        mask |= rasterize_points(grid, s.points_mm, float(s.radius_mm))
    return mask


def compile_roi_layers(
    grid: Grid,
    layers: RoiLayers,
    *,
    seed_mask_override: np.ndarray | None = None,
) -> CompiledRois:
    """Rasterize layers into masks. Empty SEED raises; empty optional roles omit.

    ``seed_mask_override`` — when set (named preset), replaces painted SEED.
    Paint SEED may still be empty in that mode.
    """
    if seed_mask_override is not None:
        if seed_mask_override.shape != grid.shape:
            raise RoiCompileError("seed_mask_override shape mismatch")
        seed_mask = np.asarray(seed_mask_override, dtype=bool)
        if int(seed_mask.sum()) == 0:
            raise RoiCompileError("preset seed is empty")
    else:
        if not layers.seed:
            raise RoiCompileError("seed is required — paint a SEED region before tracking")
        seed_mask = _union_strokes(grid, layers.seed)
        if int(seed_mask.sum()) == 0:
            raise RoiCompileError("seed painted no in-grid voxels")

    and_masks: list[np.ndarray] = []
    for region in layers.and_regions:
        if not region:
            continue
        m = _union_strokes(grid, region)
        if int(m.sum()) == 0:
            # empty after OOB discard — skip rather than emit a zero include
            # that would reject every streamline
            continue
        and_masks.append(m)

    or_mask = None
    if layers.or_regions:
        m = _union_strokes(grid, layers.or_regions)
        if int(m.sum()) > 0:
            or_mask = m

    not_mask = None
    if layers.not_regions:
        m = _union_strokes(grid, layers.not_regions)
        if int(m.sum()) > 0:
            not_mask = m

    return CompiledRois(
        seed_mask=seed_mask,
        and_masks=and_masks,
        or_mask=or_mask,
        not_mask=not_mask,
    )


def build_tckgen_roi_args(
    *,
    seed_path: str,
    and_paths: list[str] | None = None,
    or_path: str | None = None,
    not_path: str | None = None,
) -> list[str]:
    """Build the argv fragment for ROI flags (no shell, paths server-owned)."""
    if not seed_path:
        raise RoiCompileError("seed_path is required")
    argv: list[str] = ["-seed_image", seed_path]
    for p in and_paths or []:
        argv.extend(["-include", p])
    if or_path:
        argv.extend(["-include", or_path])
    if not_path:
        argv.extend(["-exclude", not_path])
    return argv


def _stroke_from_dict(obj: dict) -> Stroke:
    if not isinstance(obj, dict):
        raise RoiCompileError("ROI entry must be an object with points_mm + radius_mm")
    pts = obj.get("points_mm")
    radius = obj.get("radius_mm")
    if not isinstance(pts, list):
        raise RoiCompileError("points_mm must be a list")
    if not isinstance(radius, (int, float)):
        raise RoiCompileError("radius_mm must be a number")
    if not pts:
        raise RoiCompileError("ROI region has no points")
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RoiCompileError("points_mm must be [[x,y,z], ...]")
    if not np.isfinite(arr).all():
        raise RoiCompileError("points_mm must contain only finite numbers")
    if len(arr) > 20_000:
        raise RoiCompileError("too many paint points in one region")
    return Stroke(points_mm=arr, radius_mm=float(radius))


def layers_from_request(req: dict, *, require_seed: bool = True) -> RoiLayers:
    """Parse the POST /api/track body into RoiLayers.

    Backward compatible:
      {"seed": {"points_mm": [...], "radius_mm": 4}}
    Full boolean:
      seed + optional and (list of regions) / or (list of strokes) / not (list of strokes)
    Named preset track:
      {"seed_preset": "seed_fa_r", ...} with require_seed=False
    """
    reject_cavity_in_request(req)
    seed_strokes: list[Stroke] = []
    total_points = 0

    def parse_stroke(obj: dict) -> Stroke:
        nonlocal total_points
        stroke = _stroke_from_dict(obj)
        total_points += len(stroke.points_mm)
        if total_points > MAX_ROI_POINTS_TOTAL:
            raise RoiCompileError(
                f"too many paint points in request (max {MAX_ROI_POINTS_TOTAL})"
            )
        return stroke

    seed_obj = req.get("seed")
    if isinstance(seed_obj, dict):
        pts = seed_obj.get("points_mm")
        if isinstance(pts, list) and pts:
            seed_strokes = [parse_stroke(seed_obj)]
        elif require_seed:
            raise RoiCompileError("seed.points_mm missing/empty")
    elif require_seed:
        raise RoiCompileError("seed is required")

    and_regions: list[AndRegion] = []
    raw_and = req.get("and", [])
    if not isinstance(raw_and, list):
        raise RoiCompileError("'and' must be a list of regions")
    if len(raw_and) > MAX_ROI_REGIONS:
        raise RoiCompileError(f"too many AND regions (max {MAX_ROI_REGIONS})")
    for entry in raw_and:
        # each entry is one region (one stroke object with its own points)
        and_regions.append([parse_stroke(entry)])

    or_regions: list[Stroke] = []
    raw_or = req.get("or", [])
    if not isinstance(raw_or, list):
        raise RoiCompileError("'or' must be a list of regions")
    if len(raw_or) > MAX_ROI_REGIONS:
        raise RoiCompileError(f"too many OR regions (max {MAX_ROI_REGIONS})")
    for entry in raw_or:
        or_regions.append(parse_stroke(entry))

    not_regions: list[Stroke] = []
    raw_not = req.get("not", [])
    if not isinstance(raw_not, list):
        raise RoiCompileError("'not' must be a list of regions")
    if len(raw_not) > MAX_ROI_REGIONS:
        raise RoiCompileError(f"too many NOT regions (max {MAX_ROI_REGIONS})")
    for entry in raw_not:
        not_regions.append(parse_stroke(entry))

    return RoiLayers(
        seed=seed_strokes,
        and_regions=and_regions,
        or_regions=or_regions,
        not_regions=not_regions,
    )
