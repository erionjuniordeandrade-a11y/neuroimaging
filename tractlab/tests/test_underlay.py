"""FA / DEC underlay volumes — grid-matched, fail-closed affine gate."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tractlab.grid import load_grid
from tractlab.volume import (
    build_u8_volume,
    build_rgb_u8_volume,
    assert_grid_match,
)

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
B0 = f"{CASE}/nifti/b0.nii.gz"
MASK = f"{CASE}/nifti/mask_up.nii.gz"
FA = f"{CASE}/nifti/fa_dwi.nii.gz"
DEC = f"{CASE}/nifti/fod_dec.nii.gz"
MANIFEST = os.path.expanduser(
    "~/tractlab/cases/local-case/manifest.json"
)

needs = pytest.mark.skipif(
    not (os.path.exists(FA) and os.path.exists(DEC) and os.path.exists(B0)),
    reason="FA/DEC/b0 underlay assets absent",
)


@needs
def test_fa_and_dec_match_tracking_grid():
    g = load_grid(B0)
    assert_grid_match(FA, g)
    assert_grid_match(DEC, g)


@needs
def test_fa_u8_builds():
    vol = build_u8_volume(FA, MASK, lo_pct=1.0, hi_pct=99.0)
    assert vol.data_u8.shape == (185, 185, 109)
    assert vol.data_u8.dtype == np.uint8
    assert vol.data_u8.max() > 0
    assert len(vol.to_bytes()) == 185 * 185 * 109


@needs
def test_dec_rgb_builds_three_planes():
    g = load_grid(B0)
    rgb = build_rgb_u8_volume(DEC, g)
    assert rgb.data_u8.shape == (185, 185, 109, 3)
    buf = rgb.to_bytes()
    n = 185 * 185 * 109
    assert len(buf) == n * 3
    # some non-zero colour somewhere
    assert any(buf[i] > 0 for i in range(0, len(buf), max(1, len(buf) // 5000)))


@needs
def test_dec_affine_mismatch_fails(tmp_path):
    import nibabel as nib

    g = load_grid(B0)
    img = nib.load(DEC)
    bad_aff = np.array(img.affine, copy=True)
    bad_aff[0, 3] += 50.0  # 5 cm shift
    bad = tmp_path / "bad_dec.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), bad_aff), str(bad))
    with pytest.raises(ValueError, match="affine"):
        build_rgb_u8_volume(str(bad), g)


@needs
def test_manifest_lists_fa_dec():
    with open(MANIFEST) as f:
        man = json.load(f)
    assert "fa" in man["inputs"] and "dec" in man["inputs"]
    root = man["case_root"]
    assert os.path.isfile(os.path.join(root, man["inputs"]["fa"]["path"]))
    assert os.path.isfile(os.path.join(root, man["inputs"]["dec"]["path"]))
