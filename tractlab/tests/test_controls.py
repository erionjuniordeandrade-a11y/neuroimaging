"""M8 — quantitative slice-1 acceptance gates (coordinate-integrity).

The load-bearing one is the NEGATIVE control: painting a LEFT seed vs a RIGHT
seed must produce laterally-separated, largely-disjoint bundles. A positive-only
test cannot tell a working pipeline from one that returns everything regardless
of where you paint (clamped coords, identity affine, ignored seed). This runs the
FULL painted-seed stack (seed.py -> track.py) and asserts the result actually
depends on the seed location.

Deterministic fixtures run nthreads=0 (serial) so the assertions are stable.
"""

from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import pytest

from tractlab.grid import load_grid, voxel_to_world, world_to_voxel
from tractlab.seed import rasterize_points, write_seed_nifti
from tractlab.track import run_tckgen, TrackParams, Outcome, TCKGEN

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
FOD = f"{CASE}/nifti/wmfod_norm.mif"
MASK = f"{CASE}/nifti/mask_up.nii.gz"
FA_L = f"{CASE}/tracts/roi/fa_l_seed.nii.gz"
FA_R = f"{CASE}/tracts/roi/fa_r_seed.nii.gz"

needs = pytest.mark.skipif(
    not (os.path.exists(FOD) and os.path.exists(TCKGEN) and os.path.exists(FA_L)),
    reason="FOD / tckgen / ROI fixtures absent",
)


def _roi_centroid_world(grid, roi_path):
    d = np.asarray(nib.load(roi_path).dataobj) > 0
    ijk = np.argwhere(d).mean(axis=0)
    return voxel_to_world(grid, ijk)


def _paint_and_track(grid, world_mm, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mask = rasterize_points(grid, np.asarray([world_mm]), radius_mm=6.0)
    seed_path = os.path.join(out_dir, "seed.nii.gz")
    write_seed_nifti(mask, MASK, seed_path)
    return run_tckgen(FOD, seed_path, MASK, out_dir,
                      TrackParams(seeds=20_000, select=1200, nthreads=0), timeout_s=60)


def _mean_world_x(grid, tck_path):
    pts = np.concatenate(list(nib.streamlines.load(tck_path).streamlines), axis=0)
    return float(pts[:, 0].mean())


def _visited_voxels(grid, tck_path):
    pts = np.concatenate(list(nib.streamlines.load(tck_path).streamlines), axis=0)
    ijk = np.round(world_to_voxel(grid, pts)).astype(int)
    inb = np.all((ijk >= 0) & (ijk < np.asarray(grid.shape)), axis=1)
    ijk = ijk[inb]
    return set(map(tuple, ijk))


@needs
def test_negative_control_left_vs_right_are_lateralized_and_disjoint(tmp_path):
    g = load_grid(MASK)
    lw = _roi_centroid_world(g, FA_L)
    rw = _roi_centroid_world(g, FA_R)
    # sanity: the two fixtures really are on opposite sides in world-x
    assert np.sign(lw[0]) != np.sign(rw[0]), "fixtures not on opposite sides"

    lres = _paint_and_track(g, lw, str(tmp_path / "L"))
    rres = _paint_and_track(g, rw, str(tmp_path / "R"))
    assert lres.outcome is Outcome.OK and rres.outcome is Outcome.OK
    assert lres.n_accepted > 50 and rres.n_accepted > 50

    lx = _mean_world_x(g, lres.tck_path)
    rx = _mean_world_x(g, rres.tck_path)
    # laterality preserved: bundles sit on the same side as their seed
    assert np.sign(lx) == np.sign(lw[0]), f"left bundle drifted to x={lx:.1f}"
    assert np.sign(rx) == np.sign(rw[0]), f"right bundle drifted to x={rx:.1f}"
    assert abs(lx - rx) > 20.0, "left/right bundles are not spatially separated"

    # largely disjoint: Jaccard of visited voxels is low (not 'returns everything')
    vl, vr = _visited_voxels(g, lres.tck_path), _visited_voxels(g, rres.tck_path)
    jacc = len(vl & vr) / max(1, len(vl | vr))
    assert jacc < 0.15, f"left/right bundles overlap too much (Jaccard={jacc:.2f})"


@needs
def test_same_seed_repeats_are_stable_nthreads0(tmp_path):
    """A/A: same painted seed, serial, twice -> identical geometry.

    Bounds the stochastic variability that the negative control must exceed.
    """
    g = load_grid(MASK)
    w = _roi_centroid_world(g, FA_R)
    a = _paint_and_track(g, w, str(tmp_path / "a"))
    b = _paint_and_track(g, w, str(tmp_path / "b"))
    assert a.outcome is Outcome.OK and b.outcome is Outcome.OK
    va = _visited_voxels(g, a.tck_path)
    vb = _visited_voxels(g, b.tck_path)
    jacc = len(va & vb) / max(1, len(va | vb))
    assert jacc > 0.95, f"serial A/A not reproducible (Jaccard={jacc:.2f})"
