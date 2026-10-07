"""A1 atlas prep: registry refusal, auto-QC can fail, approve gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tractlab.atlas_prep import (
    load_registry,
    resolve_wanted,
    auto_qc_priors,
    approve_atlas_qc,
    approve_parcellation_qc,
    DEFAULT_PRIORS,
)
from tractlab.priors import discover_priors, atlas_prior_qc_ok

CASE = Path.home() / "tractlab-data/cases/local-case"
MASK = CASE / "nifti/mask_up.nii.gz"
needs = pytest.mark.skipif(not MASK.is_file(), reason="case absent")


def test_registry_loads_and_refuses_unknown():
    reg = load_registry()
    assert 14 in reg and "Corticospinal Tract R" in reg[14]
    with pytest.raises(ValueError, match="NOT IN XTRACT REGISTRY"):
        resolve_wanted({"bogus": "Not A Real Tract Name"}, reg)


def test_default_priors_all_in_registry():
    reg = load_registry()
    resolved = resolve_wanted(DEFAULT_PRIORS, reg)
    assert len(resolved) == len(DEFAULT_PRIORS)
    assert resolved["cst_r"][1] == "Corticospinal Tract R"
    assert resolved["fmi"][1] == "Forceps Minor"
    assert resolved["fma"][1] == "Forceps Major"


@needs
def test_auto_qc_symmetry_broken_fixture(tmp_path):
    """Asymmetric CST pair must fail the symmetry gate."""
    # minimal volume on real mask shape
    m = np.asanyarray(nib.load(str(MASK)).dataobj) > 0
    aff = nib.load(str(MASK)).affine
    sh = m.shape
    # put mass only on right half for "L" and different for "R"
    vl = np.zeros(sh, dtype=np.float32)
    vr = np.zeros(sh, dtype=np.float32)
    mid = sh[0] // 2
    vl[mid + 10 : mid + 20, 40:50, 40:50] = 1.0  # only one side blob
    vr[10:15, 10:15, 10:15] = 1.0  # far elsewhere
    for name, vol in [("cst_l", vl), ("cst_r", vr)]:
        nib.save(nib.Nifti1Image(vol, aff), str(tmp_path / f"norm_{name}.nii.gz"))
    qc = auto_qc_priors(
        norm_dir=tmp_path,
        mask_path=str(MASK),
        keys=["cst_l", "cst_r"],
        symmetry_min=0.15,
        hull_min=0.0,  # ignore hull for this unit
    )
    assert not qc.cst_symmetry_ok
    assert not qc.ok


def test_discover_priors_fail_closed_without_qc(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    p = root / "norm_cst_r.nii.gz"
    p.write_bytes(b"not-a-nifti")
    man = {
        "atlas_prior_qc": {"approved_by": None},
        "inputs": {
            "norm_cst_r": {"path": "norm_cst_r.nii.gz", "official_name": "x"},
        },
    }
    assert discover_priors(str(root), man["inputs"], man) == {}
    assert not atlas_prior_qc_ok(man)


def test_discover_priors_fail_closed_missing_file(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = {
        "atlas_prior_qc": {
            "approved_by": "tester",
            "date": "2026-08-09",
            "sheet_sha": "abc",
        },
        "inputs": {
            "norm_cst_r": {"path": "missing.nii.gz", "label": "CST R"},
        },
    }
    assert discover_priors(str(root), man["inputs"], man) == {}


def test_signed_false_when_qc_scoped_to_other_derivation():
    man = {
        "active_derivation": "d1",
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "atlas_prior_qc": {
            "approved_by": "tester",
            "date": "2026-08-09",
            "sheet_sha": "abc",
            "derivation": "d0-old",
        },
    }
    assert atlas_prior_qc_ok(man) is False


def test_sign_raises_on_derivation_mismatch(tmp_path):
    # arrange a manifest whose unsigned sheet says d0 but active moved to d1
    root = tmp_path / "case"
    (root / "qc").mkdir(parents=True)
    (root / "qc" / "atlas_prior_contours.png").write_bytes(b"not-a-real-png")
    (root / "norm_cst_r.nii.gz").write_bytes(b"not-a-nifti")
    man = {
        "case_root": str(root),
        "active_derivation": "d1",
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "inputs": {
            "norm_cst_r": {"path": "norm_cst_r.nii.gz", "official_name": "CST R"},
        },
        "atlas_prior_qc": {
            "auto": {"ok": True},
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "sheet_path": "qc/atlas_prior_contours.png",
            "derivation": "d0",
        },
    }
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="derivation mismatch"):
        approve_atlas_qc(str(man_path), approved_by="tester")


def test_approve_parcellation_raises_on_derivation_mismatch(tmp_path):
    # arrange a manifest whose unsigned sheet says d0 but active moved to d1
    root = tmp_path / "case"
    (root / "qc").mkdir(parents=True)
    (root / "qc" / "parcellation_prior_qc.png").write_bytes(b"not-a-real-png")
    (root / "parc_schaefer200_yeo7.nii.gz").write_bytes(b"not-a-nifti")
    man = {
        "case_root": str(root),
        "active_derivation": "d1",
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "inputs": {
            "parc_schaefer200_yeo7": {"path": "parc_schaefer200_yeo7.nii.gz"},
        },
        "parcellation_qc": {
            "auto": {"ok": True},
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "sheet_path": "qc/parcellation_prior_qc.png",
            "derivation": "d0",
        },
    }
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="derivation mismatch"):
        approve_parcellation_qc(str(man_path), approved_by="tester")
