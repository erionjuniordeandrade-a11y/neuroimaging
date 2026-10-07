"""Phantom suite for the CR peel core (ARCHITECT-BRIEF decision 8a).

Sphere-in-shell phantom: a bright "scalp" shell around a "brain" sphere, a thin bright
ring (vessel) at a known depth under the envelope and a thick bright blob at the same
depth (must NOT be labelled a vessel). Every expectation below is a geometric fact of
the phantom, not a number copied from the code under test.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tractlab.cr.peel import PeelParams, run_peel

VX = 1.0                  # mm, isotropic
SHAPE = (144, 144, 144)
CENTER = np.array([72.0, 72.0, 66.0])
R_HEAD = 60.0             # outer scalp radius (head-scale: must dwarf the 8 mm direction smoothing)
SHELL = 6.0               # scalp thickness → envelope at R_HEAD
GAP = 3.0                 # dark CSF-like gap between scalp and brain
R_BRAIN = R_HEAD - SHELL - GAP          # 31 mm → scalp→brain distance = 9 mm
VESSEL_DEPTH = 12.0                     # ring sits R_HEAD − 12 = 28 mm from centre
RING_Z = 12.0                           # ring plane: z = centre + 12 (upper convexity)
RING_HALF_WIDTH = 0.9                   # ~1.8 mm thick ⇒ "thin"
BLOB_RADIUS = 4.0                       # 8 mm thick ⇒ not thin
I_AIR, I_GAP, I_BRAIN, I_SCALP, I_BRIGHT = 0.0, 50.0, 400.0, 900.0, 1500.0


def _grid(shape, center):
    x, y, z = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    return x - center[0], y - center[1], z - center[2]


def _phantom() -> tuple[np.ndarray, np.ndarray]:
    x, y, z = _grid(SHAPE, CENTER)
    r = np.sqrt(x * x + y * y + z * z)
    vol = np.full(SHAPE, I_AIR, np.float32)
    vol[r <= R_HEAD] = I_SCALP
    vol[r <= R_HEAD - SHELL] = I_GAP
    brain = r <= R_BRAIN
    # textured brain (smooth, deterministic, σ≈20): percentile thresholds are meaningless on a flat field
    from scipy import ndimage as ndi
    noise = ndi.gaussian_filter(np.random.default_rng(0).normal(size=SHAPE), 1.5)
    noise *= 20.0 / noise[brain].std()
    vol[brain] = (I_BRAIN + noise[brain]).astype(np.float32)
    ring = (np.abs(r - (R_HEAD - VESSEL_DEPTH)) <= RING_HALF_WIDTH) & (np.abs(z - RING_Z) <= RING_HALF_WIDTH)
    vol[ring] = I_BRIGHT
    blob_centre = np.array([0.0, -(R_HEAD - VESSEL_DEPTH), 0.0])   # on the −y side, same depth
    rb = np.sqrt((x - blob_centre[0]) ** 2 + (y - blob_centre[1]) ** 2 + (z - blob_centre[2]) ** 2)
    vol[rb <= BLOB_RADIUS] = I_BRIGHT
    return vol, brain


@pytest.fixture(scope="module")
def peel():
    vol, brain = _phantom()
    params = PeelParams(depths_mm=tuple(float(d) for d in range(0, 25)))
    return run_peel(vol, brain, VX, params)


def _radius(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v - CENTER, axis=1) * VX


def test_marker_measures_scalp_to_brain_distance(peel):
    # marker = first depth column entering the brain; envelope sits ~half a voxel outside
    # the scalp sphere, so truth is SHELL + GAP (+≤1 voxel), quantised to the 1 mm stack
    assert SHELL + GAP - 0.5 <= peel.marker_depth_mm <= SHELL + GAP + 1.5
    p10, p90 = peel.marker_p10_p90_mm
    assert p90 - p10 <= 1.0


def test_vertex_count_and_topology_constant_across_depths(peel):
    n_d = len(peel.depths_mm)
    assert peel.verts_vox.shape == (n_d, len(peel.verts_vox[0]), 3)
    assert peel.grey.shape == peel.vmax.shape == peel.vessel.shape == (n_d, peel.verts_vox.shape[1])
    assert peel.faces.ndim == 2 and peel.faces.shape[1] == 3
    assert peel.verts_vox.shape[1] > 5000


def test_projection_lands_on_each_depth_isosurface(peel):
    assert float(peel.depth_residual_mm.max()) < 0.2
    # exact-offset property: every depth-d vertex sits d mm inside the depth-0 envelope
    v0 = peel.verts_vox[0]
    upper = v0[:, 2] > CENTER[2] + 15.0          # constant vertex ids ⇒ same selection at every depth
    r_env = float(np.median(_radius(v0[upper])))
    assert abs(r_env - R_HEAD) < 1.5             # envelope ≈ scalp sphere (mask/smoothing bias < 1 voxel)
    for k, d in enumerate(peel.depths_mm):
        err = np.abs(_radius(peel.verts_vox[k][upper]) - (r_env - d))
        # depth-0 mesh rides the voxel staircase of the signed EDT; deeper level sets are smooth
        assert np.percentile(err, 95) < 0.8 * VX, f"depth {d}: p95 offset error {np.percentile(err, 95):.2f} mm"


def test_no_folded_faces_at_any_depth(peel):
    assert int(peel.folded_faces.max()) == 0
    assert int(peel.folded_faces_convexity.max()) == 0


def test_vessel_ring_appears_only_at_its_depth_column(peel):
    depths = list(peel.depths_mm)
    k_on = depths.index(VESSEL_DEPTH)
    v_on = peel.verts_vox[k_on]
    ring_idx = np.nonzero(np.abs(v_on[:, 2] - (CENTER[2] + RING_Z)) <= 1.0)[0]
    # keep only ring vertices away from the blob (blob is on the −y side)
    ring_idx = ring_idx[v_on[ring_idx, 1] > CENTER[1] - 10]
    assert len(ring_idx) > 50
    assert peel.vessel[k_on, ring_idx].mean() > 0.8
    # ring half-width 0.9 + trilinear smear ~1 voxel + slab 1 mm ⇒ invisible beyond ±4 mm
    for d_off in (VESSEL_DEPTH - 4, VESSEL_DEPTH + 4):
        k = depths.index(d_off)
        assert peel.vessel[k, ring_idx].mean() < 0.1, f"ring visible at depth {d_off}"


def test_thick_blob_is_bright_but_not_a_vessel(peel):
    k_on = list(peel.depths_mm).index(VESSEL_DEPTH)
    v = peel.verts_vox[k_on]
    blob_xyz = CENTER + np.array([0.0, -(R_HEAD - VESSEL_DEPTH), 0.0])
    blob_idx = np.nonzero(np.linalg.norm(v - blob_xyz, axis=1) <= 2.0)[0]
    assert len(blob_idx) > 5
    assert peel.vmax[k_on, blob_idx].mean() > peel.vessel_thr
    assert peel.vessel[k_on, blob_idx].mean() < 0.2


def test_grey_is_a_single_sample_not_a_slab_max(peel):
    # 1 mm above the ring the ±1 mm slab still reaches it (vmax bright) but the single
    # sample at the vertex does not (grey ≈ brain): a slab-max grey would fail this
    depths = list(peel.depths_mm)
    k = depths.index(VESSEL_DEPTH - 1)
    v = peel.verts_vox[k]
    ring_idx = np.nonzero(np.abs(v[:, 2] - (CENTER[2] + RING_Z)) <= 0.5)[0]
    ring_idx = ring_idx[v[ring_idx, 1] > CENTER[1] - 10]
    assert len(ring_idx) > 30
    assert np.median(peel.vmax[k, ring_idx]) > 0.5 * I_BRIGHT
    assert np.median(peel.vmax[k, ring_idx] - peel.grey[k, ring_idx]) > 0.3 * I_BRIGHT
    k6 = depths.index(VESSEL_DEPTH + 6)
    inb = peel.in_brain[k6]
    assert inb.mean() > 0.9
    assert abs(np.median(peel.grey[k6][inb]) - I_BRAIN) < 5.0


def test_pinched_envelope_triggers_the_fold_detector():
    """Negative control: two fused spheres have a concave waist where inward trajectories
    converge; the detector must report folds there, and they must not touch the convexity
    columns that define the surgical view. A detector that stays silent here is broken."""
    # centres ±38 mm, R 40: waist ring radius 12.5 mm (≈19 mm after envelope smoothing), so
    # the 15 mm level set is a thin neck whose faces fold/bridge; the 30 mm level set does
    # not exist at the waist at all
    shape = (184, 112, 112)
    centre = np.array([92.0, 56.0, 52.0])
    x, y, z = _grid(shape, centre)
    r_a = np.sqrt((x - 38) ** 2 + y * y + z * z)
    r_b = np.sqrt((x + 38) ** 2 + y * y + z * z)
    vol = np.full(shape, I_AIR, np.float32)
    head = (r_a <= 40) | (r_b <= 40)
    vol[head] = I_SCALP
    brain = (r_a <= 30) | (r_b <= 30)
    vol[brain] = I_BRAIN
    res = run_peel(vol, brain, VX, PeelParams(depths_mm=(0.0, 5.0, 10.0, 15.0)))
    assert res.folded_faces.tolist()[:3] == [0, 0, 0]
    assert int(res.folded_faces[3]) > 0
    assert int(res.folded_faces_convexity.max()) == 0


def test_unreachable_depth_fails_closed():
    """A level set deeper than the head exists nowhere: refuse, never return a surface
    that silently misses its depth."""
    vol, brain = _phantom()
    with pytest.raises(ValueError, match="residual"):
        run_peel(vol, brain, VX, PeelParams(depths_mm=(0.0, R_HEAD + 5.0)))


def test_overlay_is_confined_to_the_superficial_band(peel):
    lo, hi = peel.band_mm
    assert lo <= VESSEL_DEPTH <= hi                    # the ring sits inside the band by construction
    for k, d in enumerate(peel.depths_mm):
        if d < lo or d > hi:
            assert not peel.vessel[k].any(), f"tint outside the band at {d} mm"


def test_rejects_empty_brain_mask():
    vol, brain = _phantom()
    with pytest.raises(ValueError):
        run_peel(vol, np.zeros_like(brain), VX)


@pytest.mark.skipif(not os.environ.get("TRACTLAB_CR_T1C"), reason="local-only real-data smoke")
def test_real_data_smoke():
    """Decision 8b: env-var path to an unstripped T1c (+ _mask sibling); never in CI."""
    import nibabel as nib

    path = os.environ["TRACTLAB_CR_T1C"]
    im = nib.load(path)
    vx = float(im.header.get_zooms()[0])
    vol = np.asarray(im.dataobj, np.float32)
    brain = np.asarray(nib.load(path.replace(".nii.gz", "_mask.nii.gz")).dataobj) > 0
    res = run_peel(vol, brain, vx, PeelParams(depths_mm=(10.0, 15.0, 20.0, 25.0, 30.0)))
    assert 8.0 < res.marker_depth_mm < 25.0
    assert float(res.depth_residual_mm.max()) < 0.2
    # the surgical view (upper convexity) must never fold; the cap rim (ear/temporal
    # concavities) may — it is reported, not hidden
    assert int(res.folded_faces_convexity.max()) == 0, res.folded_faces_convexity.tolist()
    assert res.folded_faces.max() < 0.05 * len(res.faces), res.folded_faces.tolist()
