"""Named seed presets — catalog from manifest, path-safe, grid-matched."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import nibabel as nib
import pytest

from tractlab.grid import load_grid
from tractlab.presets import (
    PresetError,
    load_preset_catalog,
    catalog_public,
    load_preset_mask,
)

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
MASK = f"{CASE}/nifti/mask_up.nii.gz"
MANIFEST = os.path.expanduser(
    "~/tractlab/cases/local-case/manifest.json"
)

needs = pytest.mark.skipif(
    not (os.path.exists(MASK) and os.path.exists(MANIFEST)),
    reason="case data or manifest absent",
)


@needs
def test_catalog_lists_seed_presets():
    with open(MANIFEST) as f:
        man = json.load(f)
    grid = load_grid(os.path.join(man["case_root"], man["inputs"]["mask"]["path"]))
    cat = load_preset_catalog(man["case_root"], man["inputs"], grid)
    assert "seed_fa_r" in cat
    assert "seed_fa_l" in cat
    assert "seed_handknob_r" in cat
    pub = catalog_public(cat)
    assert all("path" not in p for p in pub)
    assert all(p["nVoxels"] > 0 for p in pub)
    fa_r = cat["seed_fa_r"]
    assert fa_r.label == "FA-R"
    # right-hemisphere seed should have positive x in RAS-ish DWI space
    # (this case is L/P/S affine — centroid x sign is measured, not assumed)
    assert abs(fa_r.centroid_mm[0]) > 5.0


@needs
def test_live_handknob_is_not_labelled_cst():
    """Live hand-knob preset must not claim CST (brainstem not reconstructible)."""
    from tractlab.presets import load_preset_recipe_masks

    with open(MANIFEST) as f:
        man = json.load(f)
    grid = load_grid(os.path.join(man["case_root"], man["inputs"]["mask"]["path"]))
    cat = load_preset_catalog(man["case_root"], man["inputs"], grid)
    assert "seed_cst_r" not in cat  # removed false CST claim
    hk = cat["seed_handknob_r"]
    assert "cst" not in hk.label.lower()
    assert "not" in hk.note.lower() or "NOT" in hk.note
    seed, ands, not_m = load_preset_recipe_masks(cat, "seed_handknob_r")
    assert int(seed.sum()) > 1000
    assert not_m is not None and int(not_m.sum()) > 0
    assert hk.centroid_mm[2] > 20.0


@needs
def test_load_preset_mask_matches_file():
    with open(MANIFEST) as f:
        man = json.load(f)
    grid = load_grid(os.path.join(man["case_root"], man["inputs"]["mask"]["path"]))
    cat = load_preset_catalog(man["case_root"], man["inputs"], grid)
    m = load_preset_mask(cat, "seed_fa_r")
    assert m.dtype == bool or m.dtype == np.bool_
    assert int(m.sum()) == cat["seed_fa_r"].n_voxels


@needs
def test_unknown_preset_raises():
    with open(MANIFEST) as f:
        man = json.load(f)
    grid = load_grid(os.path.join(man["case_root"], man["inputs"]["mask"]["path"]))
    cat = load_preset_catalog(man["case_root"], man["inputs"], grid)
    with pytest.raises(PresetError, match="unknown"):
        load_preset_mask(cat, "seed_does_not_exist")


@needs
def test_path_traversal_refused(tmp_path):
    with open(MANIFEST) as f:
        man = json.load(f)
    grid = load_grid(os.path.join(man["case_root"], man["inputs"]["mask"]["path"]))
    # plant a malicious relative path in a synthetic inputs dict
    bad = {"seed_evil": {"path": "../../../etc/passwd"}}
    with pytest.raises(PresetError, match="path"):
        load_preset_catalog(str(tmp_path), bad, grid)


@needs
def test_relative_escape_dotdot_refused(tmp_path):
    grid = load_grid(MASK)
    # even without real file: ".." in path components is refused before open
    with pytest.raises(PresetError, match="path"):
        load_preset_catalog(str(tmp_path), {"seed_x": {"path": "foo/../../etc/passwd"}}, grid)
