"""Explore bank: prebuilt ACT extracts + in-memory filter."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tractlab.grid import load_grid, voxel_to_world
from tractlab.bank import (
    _hits_mask,
    discover_banks,
    load_prebuilt_bundle,
    filter_bank_memory,
    tckedit_filter,
)

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
MANIFEST = os.path.expanduser(
    "~/tractlab/cases/local-case/manifest.json"
)
BANK100 = f"{CASE}/tracts/connectome/pre/freesurfer/path-to-wm-v2/wholebrain_act_ifod2_100000.tck"
CST_BANK = f"{CASE}/tracts/bank/cst_r_motor_pons.tck"
MASK = f"{CASE}/nifti/mask_up.nii.gz"
FA_R = f"{CASE}/tracts/roi/fa_r_seed.nii.gz"

needs = pytest.mark.skipif(
    not (os.path.exists(BANK100) and os.path.exists(MANIFEST)),
    reason="100k bank or manifest absent",
)
needs_cst = pytest.mark.skipif(not os.path.exists(CST_BANK), reason="CST bank absent")


def test_bank_hit_default_visits_every_vertex():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    track = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ])
    inv_affine = np.eye(4)
    assert _hits_mask(track, mask, inv_affine)
    assert not _hits_mask(track, mask, inv_affine, stride=3)


@needs
def test_discover_banks_from_manifest():
    with open(MANIFEST) as f:
        man = json.load(f)
    banks = discover_banks(man["case_root"], man["inputs"])
    assert "bank_cst_r" in banks or "bank_fat_r" in banks or "bank_fa_r" in banks


@needs
def test_true_cst_bank_is_default_on_this_3t():
    """3T case ships true CST as default Explore bank — not live Commit."""
    with open(MANIFEST) as f:
        man = json.load(f)
    banks = discover_banks(man["case_root"], man["inputs"])
    assert "bank_cst_r" in banks
    cst = banks["bank_cst_r"]
    assert cst.default is True
    assert cst.role == "true_cst"
    assert cst.n_streamlines is not None and cst.n_streamlines >= 500


@needs_cst
def test_load_prebuilt_cst_bank():
    # Default: full analytic population (no length-rank truncate)
    lines, meta = load_prebuilt_bundle(CST_BANK)
    assert meta["engine"].startswith("BANK")
    assert len(lines) >= 500
    assert meta["n_corpus"] == len(lines)
    assert meta["n_loaded"] == len(lines)
    # full-length morphology: deep inferior reach (brainstem)
    zmin = np.array([float(np.asarray(s)[:, 2].min()) for s in lines])
    assert (zmin < -10).mean() > 0.8
    # Explicit display cap still available for tests
    capped, meta2 = load_prebuilt_bundle(CST_BANK, max_keep=100)
    assert len(capped) <= 100
    assert meta2["n_corpus"] >= len(capped)


@needs
def test_filter_bank_memory_fa_r():
    import nibabel as nib

    grid = load_grid(MASK)
    seed = np.asarray(nib.load(FA_R).dataobj) > 0
    lines, meta = filter_bank_memory(
        bank_path=BANK100,
        grid=grid,
        seed=seed,
        and_masks=[],
        or_mask=None,
        not_mask=None,
        minlength=30.0,
        max_keep=500,
    )
    assert meta["n_corpus"] == 100_000
    assert meta["n_kept"] >= 0
    # should find some FA-R hits in ACT bank
    assert meta["n_kept"] > 10
    assert len(lines) <= 500


@needs
def test_tckedit_oracle_matches_order_of_magnitude(tmp_path):
    """In-memory filter kept count ≈ tckedit (same seed include)."""
    import nibabel as nib

    grid = load_grid(MASK)
    seed = np.asarray(nib.load(FA_R).dataobj) > 0
    lines, meta = filter_bank_memory(
        bank_path=BANK100,
        grid=grid,
        seed=seed,
        and_masks=[],
        or_mask=None,
        not_mask=None,
        minlength=30.0,
        max_keep=100_000,
        hit_stride=1,  # denser hit test for closer oracle match
    )
    out = str(tmp_path / "oracle.tck")
    n_cli = tckedit_filter(
        bank_path=BANK100,
        out_tck=out,
        include_paths=[FA_R],
        exclude_path=None,
        minlength=30.0,
    )
    # stride=1 memory filter can still miss edge voxels vs tckedit voxelisation;
    # require same order of magnitude, not bit equality
    assert n_cli > 0 and meta["n_kept"] > 0
    ratio = meta["n_kept"] / max(n_cli, 1)
    assert 0.25 < ratio < 4.0, f"memory {meta['n_kept']} vs tckedit {n_cli}"
