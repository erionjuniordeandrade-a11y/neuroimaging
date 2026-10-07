"""A0 registration trust: auto-check can fail; approve refuses bad auto."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tractlab.check_registration import (
    anatomical_plane,
    auto_sanity,
    run_check,
    approve_t1_qc,
    sha256_file,
)
from tractlab.grid import load_grid, grid_id

CASE = Path.home() / "tractlab-data/cases/local-case"
B0 = CASE / "nifti/b0.nii.gz"
T1 = CASE / "nifti/t1c_brain_dwi.nii.gz"
MASK = CASE / "nifti/mask_up.nii.gz"
needs = pytest.mark.skipif(not B0.is_file(), reason="case absent")


@needs
def test_auto_sanity_passes_real_case():
    man = json.loads(
        (Path.home() / "tractlab/cases/local-case/manifest.json").read_text()
    )
    gid = man.get("grid", {}).get("grid_id")
    s = auto_sanity(
        b0_path=str(B0), t1_path=str(T1), mask_path=str(MASK), expected_grid_id=gid,
    )
    assert s.shape_match
    assert s.grid_id_match
    assert s.ok


@needs
def test_broken_fixture_fails_auto_check(tmp_path):
    """Deliberately wrong T1 (zeros) must not pass auto sanity."""
    g = load_grid(str(B0))
    bad = tmp_path / "bad_t1.nii.gz"
    zeros = np.zeros(g.shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(zeros, g.affine), str(bad))
    s = auto_sanity(
        b0_path=str(B0), t1_path=str(bad), mask_path=str(MASK), expected_grid_id=None,
    )
    assert not s.ok


@needs
def test_run_check_writes_sheet(tmp_path):
    res = run_check(
        b0_path=str(B0), t1_path=str(T1), mask_path=str(MASK), out_dir=tmp_path,
    )
    assert res.sheet_path and Path(res.sheet_path).is_file()
    assert res.sheet_sha256 and len(res.sheet_sha256) == 64


def _marker_volume(axcodes: tuple[str, str, str]) -> np.ndarray:
    """Ones at a superior / anterior / patient-right corner."""
    vol = np.zeros((5, 5, 5), dtype=np.float32)
    i = 0 if axcodes[0] == "L" else 4  # i=0 is patient R when axis is L
    j = 4 if axcodes[1] == "A" else 0  # high j is A; low j is A when axis is P
    k = 4 if axcodes[2] == "S" else 0
    vol[i, j, k] = 1.0
    return vol


def test_anatomical_plane_las_vertex_up_radiological():
    vol = _marker_volume(("L", "A", "S"))
    axial = anatomical_plane(vol, 2, 4, ("L", "A", "S"))
    coronal = anatomical_plane(vol, 1, 4, ("L", "A", "S"))
    sagittal = anatomical_plane(vol, 0, 0, ("L", "A", "S"))
    # origin=upper: row 0 is top. Marker is anterior+superior+patient-R.
    assert axial[0, 0] == 1.0  # top-left = A + R
    assert coronal[0, 0] == 1.0  # top-left = S + R
    assert sagittal[0, -1] == 1.0  # top-right = S + A


def test_anatomical_plane_lps_matches_las_display():
    vol = _marker_volume(("L", "P", "S"))
    axial = anatomical_plane(vol, 2, 4, ("L", "P", "S"))
    coronal = anatomical_plane(vol, 1, 0, ("L", "P", "S"))
    sagittal = anatomical_plane(vol, 0, 0, ("L", "P", "S"))
    assert axial[0, 0] == 1.0
    assert coronal[0, 0] == 1.0
    assert sagittal[0, -1] == 1.0


def test_approve_refuses_without_sheet(tmp_path):
    man = {
        "case_root": str(tmp_path),
        "t1_qc": {"auto": {"ok": True}},
        "inputs": {},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(man))
    with pytest.raises(FileNotFoundError):
        approve_t1_qc(str(p), approved_by="tester")
