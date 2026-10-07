"""Reference-aligned cropped grid and scalar packing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from neuro_core.grid import lps_to_ras_affine


DEFAULT_MAX_VOXELS = 64_000_000
MIN_SPACING_MM = 0.4
MAX_SPACING_MM = 1.5


@dataclass(frozen=True)
class GridPlan:
    image: sitk.Image
    requested_spacing_mm: float
    spacing_mm: float
    spacing_raised: bool
    crop: dict | None


def first_volume(image: sitk.Image) -> tuple[sitk.Image, str]:
    if image.GetDimension() == 3:
        return image, "all"
    if image.GetDimension() != 4:
        raise ValueError("only 3D and 4D image series are supported")
    size = list(image.GetSize())
    size[3] = 0
    return sitk.Extract(image, size, [0, 0, 0, 0]), "volume 0"


def _largest_component(mask: sitk.Image) -> np.ndarray:
    labels = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(mask)))
    return labels == 1


def ct_head_mask(reference: sitk.Image, opening_mm: float = 2.0) -> np.ndarray:
    """Head foreground for the CT crop: > -500 HU, opened, largest connected component.

    The opening (radius ~opening_mm per axis, 0 along coarse axes) detaches thin
    table/headrest shells from the head so they fall outside the largest component.
    The mask only bounds the crop box; no voxel values are changed.
    """
    raw = sitk.GetArrayFromImage(reference) > -500
    if not raw.any():
        return raw
    binary = sitk.GetImageFromArray(raw.astype(np.uint8))
    binary.CopyInformation(reference)
    radius = [int(round(opening_mm / spacing)) for spacing in reference.GetSpacing()]
    opened = sitk.BinaryMorphologicalOpening(binary, radius, sitk.sitkBall) if any(radius) else binary
    head = _largest_component(opened)
    if not head.any():
        head = _largest_component(binary)
    return head


def _box_corners(lower: np.ndarray, upper: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(x), float(y), float(z))
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])]


def _ras_bounds_of_reference_indices(reference: sitk.Image, lower: np.ndarray,
                                     upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray([reference.TransformContinuousIndexToPhysicalPoint(point)
                          for point in _box_corners(lower, upper)])
    ras = corners * np.array([-1.0, -1.0, 1.0])
    return ras.min(axis=0), ras.max(axis=0)


def mask_ras_bounds(mask: np.ndarray, image: sitk.Image, margin_mm: float = 0.0) -> list[float]:
    """Return the RAS+ axis-aligned bounds of a mask's occupied voxel centres, with margin."""
    positions = np.argwhere(mask.astype(bool))
    if not len(positions):
        raise ValueError("crop mask is empty")
    lower = positions.min(axis=0)[::-1].astype(float)
    upper = positions.max(axis=0)[::-1].astype(float)
    ras_lower, ras_upper = _ras_bounds_of_reference_indices(image, lower, upper)
    return [*(ras_lower - margin_mm).tolist(), *(ras_upper + margin_mm).tolist()]


def common_grid(reference: sitk.Image, kind: str, requested_spacing: float,
                max_voxels: int = DEFAULT_MAX_VOXELS,
                crop_ras: list[float] | None = None,
                crop_source: str | None = None,
                crop_margin_mm: float = 0.0) -> GridPlan:
    if not math.isfinite(requested_spacing) or not MIN_SPACING_MM <= requested_spacing <= MAX_SPACING_MM:
        raise ValueError(f"--spacing must be between {MIN_SPACING_MM:g} and {MAX_SPACING_MM:g} mm")
    if max_voxels < 1:
        raise ValueError("--max-voxels must be at least 1")
    if crop_ras is not None:
        if len(crop_ras) != 6 or not all(math.isfinite(float(value)) for value in crop_ras):
            raise ValueError("crop RAS box must contain six finite coordinates")
        if any(crop_ras[i] > crop_ras[i + 3] for i in range(3)):
            raise ValueError("crop RAS lower bounds must not exceed upper bounds")
        if not crop_source:
            raise ValueError("crop source is required when a crop box is supplied")
    elif crop_source is not None:
        raise ValueError("crop source was supplied without a crop box")
    if not math.isfinite(crop_margin_mm) or crop_margin_mm < 0:
        raise ValueError("--margin-mm must be a finite non-negative value")
    if kind == "CT":
        foreground = ct_head_mask(reference)
    else:
        foreground = sitk.GetArrayFromImage(sitk.OtsuThreshold(reference)) != 0
    positions = np.argwhere(foreground)
    if not len(positions):
        raise ValueError("reference volume has no detectable head foreground")
    lower_index = positions.min(axis=0)[::-1].astype(float)
    upper_index = positions.max(axis=0)[::-1].astype(float)
    reference_spacing = np.asarray(reference.GetSpacing(), dtype=float)
    # Preserve the legacy head box when no explicit crop is requested.
    head_lower_index = lower_index - 10.0 / reference_spacing
    head_upper_index = upper_index + 10.0 / reference_spacing
    crop_record = None
    crop_bounds_ras = None
    if crop_ras is None:
        lower_index, upper_index = head_lower_index, head_upper_index
    else:
        head_ras_lower, head_ras_upper = _ras_bounds_of_reference_indices(
            reference, head_lower_index, head_upper_index)
        ras_lower = np.maximum(head_ras_lower, np.asarray(crop_ras[:3], dtype=float))
        ras_upper = np.minimum(head_ras_upper, np.asarray(crop_ras[3:], dtype=float))
        if np.any(ras_upper < ras_lower):
            raise ValueError("requested crop does not intersect the head foreground box")
        crop_bounds_ras = (ras_lower, ras_upper)
        crop_record = {"ras_mm": [*map(float, ras_lower), *map(float, ras_upper)], "source": crop_source,
                       "margin_mm": float(crop_margin_mm)}

    if crop_bounds_ras is None:
        extent_mm = (upper_index - lower_index) * reference_spacing
    else:
        # An oblique reference lattice's enclosing index-space cuboid extends
        # outside an axis-aligned RAS crop. Use a RAS-aligned lattice for an
        # explicit crop so every output sample stays inside the effective box.
        ras_lower, ras_upper = crop_bounds_ras
        extent_mm = ras_upper - ras_lower
    if np.any(extent_mm < 0):
        raise ValueError("requested crop has no extent after intersection with the head foreground box")

    def dimensions(spacing: float) -> tuple[int, int, int]:
        if crop_bounds_ras is not None:
            return tuple(int(math.floor(float(value) / spacing + 1e-12)) + 1 for value in extent_mm)
        return tuple(int(math.ceil(float(value) / spacing - 1e-12)) + 1 for value in extent_mm)

    dims = dimensions(requested_spacing)
    voxel_count = math.prod(dims)
    spacing = float(requested_spacing)
    if voxel_count > max_voxels:
        if crop_ras is not None:
            raise ValueError(f"crop grid requires {voxel_count:,} voxels, exceeding --max-voxels {max_voxels:,}; "
                             "increase --spacing or tighten the crop/margin")
        low, high = requested_spacing, max(requested_spacing, float(np.max(extent_mm)))
        minimum_dims = tuple(1 if extent <= 0 else 2 for extent in extent_mm)
        minimum_voxels = math.prod(minimum_dims)
        if max_voxels < minimum_voxels:
            raise ValueError(f"--max-voxels {max_voxels:,} is too small for this grid; at least "
                             f"{minimum_voxels:,} voxels are required")
        for _ in range(64):
            middle = (low + high) / 2.0
            if math.prod(dimensions(middle)) > max_voxels:
                low = middle
            else:
                high = middle
        spacing = high
        dims = dimensions(spacing)
    grid = sitk.Image(list(dims), sitk.sitkFloat32)
    grid.SetSpacing((spacing,) * 3)
    if crop_bounds_ras is None:
        grid.SetDirection(reference.GetDirection())
        grid.SetOrigin(reference.TransformContinuousIndexToPhysicalPoint(tuple(lower_index)))
    else:
        ras_lower, _ = crop_bounds_ras
        grid.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        grid.SetOrigin((-float(ras_lower[0]), -float(ras_lower[1]), float(ras_lower[2])))
    return GridPlan(image=grid, requested_spacing_mm=float(requested_spacing), spacing_mm=spacing,
                    spacing_raised=spacing > requested_spacing + 1e-9, crop=crop_record)


def affine_ras(grid: sitk.Image) -> list[list[float]]:
    return lps_to_ras_affine(grid.GetDirection(), grid.GetSpacing(), grid.GetOrigin()).tolist()


def resample_to_grid(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform, kind: str) -> np.ndarray:
    default = -1000.0 if kind == "CT" else 0.0
    aligned = sitk.Resample(image, grid, transform, sitk.sitkLinear, default, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(aligned)


def pack_scalar(array: np.ndarray, kind: str) -> tuple[np.ndarray, float, float]:
    if kind == "CT":
        return np.clip(np.rint(array), -32768, 32767).astype(np.int16), 1.0, 0.0
    # Map the full range: clipping at a percentile would flatten small bright
    # structures such as an enhancing lesion. uint16 keeps ample precision.
    minimum = float(np.min(array))
    high = float(np.max(array))
    slope = (high - minimum) / 65535.0 if high > minimum else 1.0
    packed = np.clip(np.rint((array - minimum) / slope), 0, 65535).astype(np.uint16)
    return packed, slope, minimum


def scalar_stats(array: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(array)), "max": float(np.max(array)),
            "p01": float(np.percentile(array, 1)), "p99": float(np.percentile(array, 99))}
