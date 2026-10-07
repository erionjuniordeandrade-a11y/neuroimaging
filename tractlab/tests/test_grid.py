"""Real-grid tests for the voxel<->world contract.

These run against the materialized b0 in the stable case tree. They encode the
42.3 mm trap from the adversarial review: if anyone swaps the nibabel affine for
the mrinfo +diagonal matrix, the center-voxel and sabotage tests fail loudly.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from tractlab.grid import (
    load_grid,
    voxel_to_world,
    world_to_voxel,
    in_bounds,
    grid_id,
)

B0 = os.path.expanduser(
    "~/tractlab-data/cases/local-case/nifti/b0.nii.gz"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(B0), reason="materialized b0 not present (run Phase 0)"
)


def test_axcodes_and_diagonal_are_lps_negative():
    """The authoritative affine is L/P/S with a negative diagonal.

    This is the assertion that catches a swap to mrinfo's +diagonal matrix.
    """
    g = load_grid(B0)
    assert g.shape == (185, 185, 109)
    assert g.axcodes == ("L", "P", "S")
    assert list(np.sign(np.diag(g.affine)[:3])) == [-1.0, -1.0, 1.0]


def test_center_voxel_maps_to_verified_world_point():
    """Center voxel [92,92,54] -> the value verified on 2026-08-02.

    If the mrinfo +diagonal matrix were used, this lands at [-29.16,-33.96,-3.68]
    instead — a 42.3 mm error. This test is that tripwire.
    """
    g = load_grid(B0)
    world = voxel_to_world(g, np.array([92, 92, 54]))
    np.testing.assert_allclose(world, [-0.332, -11.309, 17.344], atol=1e-2)


def test_roundtrip_world_voxel_is_identity():
    g = load_grid(B0)
    rng = np.random.default_rng(0)
    ijk = rng.integers(0, [185, 185, 109], size=(200, 3))
    back = world_to_voxel(g, voxel_to_world(g, ijk))
    np.testing.assert_allclose(back, ijk, atol=1e-6)


def test_in_bounds_never_clamps():
    """Out-of-bounds voxels are flagged, not snapped to the face."""
    g = load_grid(B0)
    ijk = np.array([[0, 0, 0], [184, 184, 108], [-1, 0, 0], [185, 0, 0], [92, 92, 200]])
    mask = in_bounds(g, ijk)
    assert list(mask) == [True, True, False, False, False]


def test_affine_x_flip_sabotage_changes_the_mapping():
    """SABOTAGE gate: flipping the x-axis of the affine MUST move the seed.

    A coordinate chain that ignores the affine (identity / clamped) would map a
    point identically before and after the flip. Requiring a large displacement
    proves the affine is actually load-bearing — the laterality sentinel.

    ⚠ The probe MUST be genuinely lateral. The center voxel is near the x=0
    midline (world-x = -0.33 mm), so an x-flip barely moves it (~0.7 mm) — a
    midline point is a useless laterality sentinel, the same trap that makes
    "flipud alone" and the near-midline cst_l_seed poor L/R checks.
    """
    g = load_grid(B0)
    lateral = np.array([160, 92, 54])  # far in i -> large |world-x|
    good = voxel_to_world(g, lateral)
    assert abs(good[0]) > 60.0, "probe is not lateral enough to test laterality"

    flipped_affine = g.affine.copy()
    flipped_affine[0, :] *= -1  # sabotage: negate the world-x row
    homog = np.concatenate([lateral.astype(float), [1.0]])
    bad = (homog @ flipped_affine.T)[:3]

    displacement = np.linalg.norm(good - bad)
    assert displacement > 40.0, (
        f"x-flip only moved a lateral point {displacement:.1f} mm; the affine is "
        "not load-bearing in the coordinate chain — laterality is unprotected."
    )


def test_grid_id_is_stable_and_geometry_bound():
    g = load_grid(B0)
    gid = grid_id(g)
    assert len(gid) == 64
    assert grid_id(load_grid(B0)) == gid  # deterministic
