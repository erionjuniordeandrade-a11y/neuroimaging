"""Tests for M2 volume, M3 seed, M5 pack. Real b0 where needed."""

from __future__ import annotations

import os
import numpy as np
import pytest

from tractlab.grid import load_grid, voxel_to_world
from tractlab.volume import build_u8_volume
from tractlab.seed import rasterize_points, write_seed_nifti
from tractlab.pack import resample_polyline, pack_streamlines, unpack_streamlines

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
B0 = f"{CASE}/nifti/b0.nii.gz"
MASK = f"{CASE}/nifti/mask_up.nii.gz"
needs_data = pytest.mark.skipif(not os.path.exists(B0), reason="Phase 0 data absent")


# ---------------- M2 volume ----------------

@needs_data
def test_u8_volume_shape_and_range():
    v = build_u8_volume(B0, MASK)
    assert v.data_u8.shape == (185, 185, 109)
    assert v.data_u8.dtype == np.uint8
    assert v.data_u8.max() == 255 and v.data_u8.min() == 0
    assert v.window[1] > v.window[0]


@needs_data
def test_u8_bytes_are_i_fastest_and_roundtrip():
    v = build_u8_volume(B0, MASK)
    buf = v.to_bytes()
    assert len(buf) == 185 * 185 * 109
    back = np.frombuffer(buf, dtype=np.uint8).reshape((185, 185, 109), order="F")
    np.testing.assert_array_equal(back, v.data_u8)


@needs_data
def test_volume_id_deterministic_and_content_bound():
    v1 = build_u8_volume(B0, MASK)
    v2 = build_u8_volume(B0, MASK)
    assert v1.volume_id == v2.volume_id
    # perturb one voxel -> different volume_id (content bound)
    import copy
    d = v1.data_u8.copy()
    d[0, 0, 0] = 255 - d[0, 0, 0]
    from tractlab.volume import U8Volume
    v3 = U8Volume(v1.grid, d, v1.window, v1.source_sha256)
    assert v3.volume_id != v1.volume_id


# ---------------- M3 seed ----------------

@needs_data
def test_rasterize_sphere_hits_center_voxel_and_is_bounded():
    g = load_grid(B0)
    center_world = voxel_to_world(g, np.array([92, 92, 54]))
    mask = rasterize_points(g, center_world[None, :], radius_mm=3.0)
    assert mask[92, 92, 54]           # center is hit
    assert 1 <= mask.sum() <= 400     # a 3mm sphere on 1.3mm voxels is small
    assert mask.shape == g.shape


@needs_data
def test_rasterize_discards_oob_never_clamps():
    g = load_grid(B0)
    # a point far outside the FOV must produce an EMPTY mask, not a face smear
    far = voxel_to_world(g, np.array([400, 400, 400]))
    mask = rasterize_points(g, far[None, :], radius_mm=3.0)
    assert mask.sum() == 0
    # specifically: no boundary face got lit
    assert not mask[-1, :, :].any() and not mask[:, -1, :].any()


@needs_data
def test_write_seed_preserves_header_codes(tmp_path):
    g = load_grid(B0)
    cw = voxel_to_world(g, np.array([92, 92, 54]))
    mask = rasterize_points(g, cw[None, :], radius_mm=4.0)
    out = str(tmp_path / "seed.nii.gz")
    info = write_seed_nifti(mask, B0, out)
    import nibabel as nib
    ref = nib.load(B0)
    got = nib.load(out)
    _, ref_qc = ref.get_qform(coded=True)
    _, got_qc = got.get_qform(coded=True)
    assert int(got_qc or 0) == int(ref_qc or 0)          # qform code preserved
    np.testing.assert_allclose(got.affine, ref.affine, atol=1e-6)
    assert info["n_voxels"] == int(mask.sum())


# ---------------- M5 pack ----------------

def test_resample_gives_exactly_k_points():
    line = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
    r = resample_polyline(line, 50)
    assert r.shape == (50, 3)
    np.testing.assert_allclose(r[0], [0, 0, 0])
    np.testing.assert_allclose(r[-1], [10, 10, 0])


def test_pack_roundtrip_fixed_k():
    lines = [np.random.default_rng(i).random((np.random.default_rng(i).integers(5, 40), 3))
             for i in range(20)]
    buf, header = pack_streamlines(lines, k=64, space_id="dwi-v1")
    assert header["lineCount"] == 20
    k = int(header["pointsPerLine"])
    back = unpack_streamlines(buf, 20, k)
    assert back.shape == (20, k, 3)


def test_pack_respects_minlength_on_source():
    short = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], dtype=float)
    long = np.column_stack([np.linspace(0, 100, 50), np.zeros(50), np.zeros(50)])
    buf, header = pack_streamlines(
        [short, long], k=64, space_id="x", minlength_mm=20.0,
    )
    assert header["lineCount"] == 1
    assert header["nDropSourceShort"] == 1
    assert float(header["sourceMinLengthMm"]) >= 20.0 - 1e-3
    k = int(header["pointsPerLine"])
    back = unpack_streamlines(buf, 1, k)
    from tractlab.pack import polyline_length_mm
    assert polyline_length_mm(back[0]) >= 20.0 - 0.5


def test_pack_rejects_nonfinite():
    bad = [np.array([[0, 0, 0], [np.nan, 1, 2]], dtype=float)]
    with pytest.raises(ValueError, match="non-finite"):
        pack_streamlines(bad, k=16, space_id="x")
