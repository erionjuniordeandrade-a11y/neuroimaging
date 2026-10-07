"""ROI seed rasterization — world-mm paint points -> a voxel mask NIfTI.

The browser sends world-mm points + a radius in mm (never pixels/voxels). The
server owns the single affine and rasterizes here. Two rules from the review:

  1. NEVER clamp out-of-bounds voxels onto the volume face — discard them. A
     clamp fabricates ROI membership on the boundary in both directions.
  2. Build the output NIfTI from the REFERENCE header so qform/sform and their
     codes are preserved; ``nib.Nifti1Image(data, affine)`` alone drops them and
     tckgen may then read a subtly different geometry. See DESIGN v2 #7.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib

from .grid import Grid, world_to_voxel, in_bounds


def rasterize_points(
    grid: Grid,
    points_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Return a bool mask (grid.shape) of voxels within radius_mm of any point.

    Distance is measured in true world mm (handles the oblique/anisotropic case
    correctly), not in voxel units. OOB voxels are discarded, never clamped.
    """
    points_mm = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
    if points_mm.shape[-1] != 3:
        raise ValueError("points_mm must be (...,3) world coordinates")
    if not (radius_mm > 0):
        raise ValueError("radius_mm must be > 0")

    mask = np.zeros(grid.shape, dtype=bool)

    # voxel-space bbox for each point, then exact world-distance test inside it.
    centers_vox = world_to_voxel(grid, points_mm)
    # per-axis voxel reach = ceil(radius / min column norm) is loose but safe;
    # compute per-axis spacing from the affine column norms.
    spacing = np.linalg.norm(grid.affine[:3, :3], axis=0)  # mm per voxel step, per axis
    reach = np.ceil(radius_mm / spacing).astype(int) + 1

    for c_vox, p_mm in zip(centers_vox, points_mm):
        lo = np.floor(c_vox).astype(int) - reach
        hi = np.ceil(c_vox).astype(int) + reach
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(grid.shape) - 1)
        if np.any(lo > hi):
            continue
        ii, jj, kk = np.meshgrid(
            np.arange(lo[0], hi[0] + 1),
            np.arange(lo[1], hi[1] + 1),
            np.arange(lo[2], hi[2] + 1),
            indexing="ij",
        )
        cand = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)
        # exact world-distance test
        from .grid import voxel_to_world

        world = voxel_to_world(grid, cand)
        d = np.linalg.norm(world - p_mm, axis=-1)
        hit = cand[d <= radius_mm]
        if hit.size:
            mask[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    return mask


def write_seed_nifti(
    mask: np.ndarray,
    reference_path: str,
    out_path: str,
) -> dict:
    """Write ``mask`` as a u8 NIfTI copying the reference header (qform/sform/codes).

    Returns a dict of what was preserved, for the round-trip assertion in tests.
    """
    ref = nib.load(reference_path)
    if mask.shape != ref.shape[:3]:
        raise ValueError("mask shape != reference shape")

    out = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, header=ref.header.copy())
    out.set_data_dtype(np.uint8)
    # preserve both transforms and their codes explicitly
    qform, qcode = ref.get_qform(coded=True)
    sform, scode = ref.get_sform(coded=True)
    out.set_qform(qform, code=int(qcode) if qcode is not None else 0)
    out.set_sform(sform, code=int(scode) if scode is not None else 0)
    nib.save(out, out_path)

    return {
        "n_voxels": int(mask.sum()),
        "qcode": int(qcode) if qcode is not None else 0,
        "scode": int(scode) if scode is not None else 0,
    }
