"""Peri-lesional recovery: zone mask + filter hits lesion neighbourhood only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tractlab.grid import load_grid
from tractlab.recovery import (
    filter_near_lesion_memory,
    lesion_proximity_mask,
    load_lesion_bool,
)

CASE = Path.home() / "tractlab-data/cases/local-case"
LESION = CASE / "tracts/roi/lesion1_dwi.nii.gz"
CORPUS = CASE / (
    "tracts/connectome/pre/freesurfer/path-to-wm-v2/"
    "wholebrain_act_ifod2_100000.tck"
)
needs = pytest.mark.skipif(
    not (LESION.is_file() and CORPUS.is_file()),
    reason="case corpus/lesion absent",
)


def test_lesion_proximity_mask_grows_with_radius():
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = 2.0  # 2 mm voxels
    m = np.zeros((20, 20, 20), dtype=bool)
    m[10, 10, 10] = True
    z0 = lesion_proximity_mask(m, aff, 0.0)
    z4 = lesion_proximity_mask(m, aff, 4.0)
    assert int(z0.sum()) == 1
    assert int(z4.sum()) > int(z0.sum())


@needs
def test_load_lesion_matches_case_grid():
    # b0 / mask grid authority for tracking
    grid = load_grid(str(CASE / "nifti/b0.nii.gz"))
    les = load_lesion_bool(str(LESION), grid)
    assert les.shape == grid.shape
    assert les.any()


@needs
def test_filter_near_lesion_returns_hits_with_stamp_meta():
    grid = load_grid(str(CASE / "nifti/b0.nii.gz"))
    les = load_lesion_bool(str(LESION), grid)
    zone = lesion_proximity_mask(les, grid.affine, 8.0)
    kept, meta = filter_near_lesion_memory(
        bank_path=str(CORPUS),
        grid=grid,
        lesion_zone=zone,
        minlength=10.0,
        maxlength=250.0,
        max_keep=500,  # cap for test speed after filter
    )
    assert meta["role"] == "recovery_perilesional"
    assert "not named" in meta["label"].lower() or "RECOVERY" in meta["label"]
    assert meta["n_corpus"] > 0
    # Should find something within 8 mm on this case
    assert meta["n_kept"] > 0
    assert len(kept) > 0
    assert len(kept) <= 500
