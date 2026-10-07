"""Grid / affine contract — the single source of truth for voxel<->world.

⛔ For NIfTI, the authoritative IJK->world matrix is nibabel's ``img.affine``
(the NIfTI sform/qform selection), NOT MRtrix ``mrinfo -transform``. For this
case the two disagree by 42.3 mm at the volume center because
``mrinfo -transform`` prints a normalized +diagonal matrix with strides handled
separately, whereas the NIfTI array affine is L/P/S with a negative diagonal.
Using the wrong one silently moves a painted seed to a different part of the
brain while every API type still looks valid. Native MIF inputs use the
validated, stride-aware canonical reconstruction in ``load_mif_grid``; no MIF
transform is substituted for a NIfTI array affine. See docs/DESIGN-slice1.md v2
#2.
"""

from __future__ import annotations

# The implementation lives in neuro_core so capsule and tractlab share one contract.
from neuro_core.grid import _parse_mrinfo_numbers  # noqa: F401  (tests pin the parser)
from neuro_core.grid import (
    Grid,
    assert_grids_match,
    grid_from_image,
    grid_id,
    in_bounds,
    load_grid,
    load_image_grid,
    load_mif_grid,
    unknown_units_assumption_from_manifest,
    validate_affine,
    voxel_to_world,
    world_to_voxel,
)

__all__ = [
    "Grid",
    "assert_grids_match",
    "grid_from_image",
    "grid_id",
    "in_bounds",
    "load_grid",
    "load_image_grid",
    "load_mif_grid",
    "unknown_units_assumption_from_manifest",
    "validate_affine",
    "voxel_to_world",
    "world_to_voxel",
]
