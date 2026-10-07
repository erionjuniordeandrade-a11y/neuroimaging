import hashlib
import json

import numpy as np
import nibabel as nib
import pytest

from tractlab import delta_qc


def _nii(data, zoom=2.0):
    aff = np.diag([zoom, zoom, zoom, 1.0])
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    image.header.set_xyzt_units("mm", "sec")
    return image


def test_constant_field_gives_exact_shift():
    field = _nii(np.full((8, 8, 8), 20.0))
    mask = _nii(np.ones((8, 8, 8)))
    s = delta_qc.compute_delta(field, mask, readout_time_s=0.05, pe_axis=1)
    assert abs(s.median_mm - 2.0) < 1e-6
    assert abs(s.p95_mm - 2.0) < 1e-6
    assert abs(s.max_mm - 2.0) < 1e-6
    assert s.n_vox == 8 * 8 * 8


def test_masked_voxels_excluded():
    field = _nii(np.full((8, 8, 8), 20.0))
    m = np.zeros((8, 8, 8))
    m[0, 0, 0] = 1
    s = delta_qc.compute_delta(field, _nii(m), readout_time_s=0.05, pe_axis=1)
    assert s.n_vox == 1
    assert abs(s.median_mm - 2.0) < 1e-6


def test_nan_field_voxels_are_excluded_not_fabricated():
    data = np.full((8, 8, 8), 20.0, dtype=np.float32)
    data[0, 0, 0] = np.nan
    field = _nii(data)
    mask = _nii(np.ones((8, 8, 8)))
    s = delta_qc.compute_delta(field, mask, readout_time_s=0.05, pe_axis=1)
    assert s.n_vox == 8 * 8 * 8 - 1
    assert abs(s.median_mm - 2.0) < 1e-6


def test_empty_mask_gives_honest_null_not_zero():
    field = _nii(np.full((8, 8, 8), 20.0))
    mask = _nii(np.zeros((8, 8, 8)))
    s = delta_qc.compute_delta(field, mask, readout_time_s=0.05, pe_axis=1)
    assert s.n_vox == 0
    assert np.isnan(s.median_mm)
    assert np.isnan(s.p95_mm)
    assert np.isnan(s.max_mm)


def test_sheet_block_is_unsigned_and_derivation_scoped(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    (case_root / "manifest.json").write_text(json.dumps({
        "case_id": "stub",
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "active_derivation": "d1",
        "inputs": {},
    }))
    field = _nii(np.full((8, 8, 8), 20.0))
    mask = _nii(np.ones((8, 8, 8)))
    stats = delta_qc.compute_delta(field, mask, readout_time_s=0.05, pe_axis=1)
    png_path = case_root / "qc" / "delta_qc.png"

    block = delta_qc.write_sheet(case_root, stats, png_path)

    assert block["approved_by"] is None
    assert block["date"] is None
    assert block["derivation"] == "d1"
    assert block["sheet_path"] == "qc/delta_qc.png"
    assert png_path.is_file()
    assert block["sheet_sha"] == hashlib.sha256(png_path.read_bytes()).hexdigest()
    assert block["auto"]["median_mm"] == stats.median_mm
    assert block["auto"]["n_vox"] == stats.n_vox


def test_write_sheet_derivation_none_for_unmigrated_manifest(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    (case_root / "manifest.json").write_text(json.dumps({"case_id": "stub"}))
    field = _nii(np.full((4, 4, 4), 20.0))
    mask = _nii(np.ones((4, 4, 4)))
    stats = delta_qc.compute_delta(field, mask, readout_time_s=0.05, pe_axis=1)

    block = delta_qc.write_sheet(case_root, stats, case_root / "qc" / "delta_qc.png")

    assert block["derivation"] is None
    assert block["approved_by"] is None


def test_cli_writes_unsigned_delta_qc_block_into_manifest(tmp_path):
    case_root = tmp_path / "case"
    (case_root / "raw" / "dwi").mkdir(parents=True)
    (case_root / "nifti").mkdir()

    nib.save(_nii(np.full((6, 6, 6), 10.0)), str(case_root / "field_hz.nii.gz"))
    nib.save(_nii(np.ones((6, 6, 6))), str(case_root / "nifti" / "mask.nii.gz"))
    nib.save(_nii(np.zeros((6, 6, 6))), str(case_root / "raw" / "dwi" / "dwi.nii.gz"))
    (case_root / "raw" / "dwi" / "dwi.json").write_text(
        json.dumps({"TotalReadoutTime": 0.05}))

    manifest = {
        "case_id": "stub",
        "case_root": str(case_root),
        "raw": {"dwi": {"path": "raw/dwi/dwi.nii.gz"}},
        "inputs": {"mask": {"path": "nifti/mask.nii.gz"}},
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "active_derivation": "d1",
    }
    (case_root / "manifest.json").write_text(json.dumps(manifest))

    rc = delta_qc.main([str(case_root), "--field", "field_hz.nii.gz", "--pe-axis", "1"])
    assert rc == 0

    updated = json.loads((case_root / "manifest.json").read_text())
    block = updated["delta_qc"]
    assert block["approved_by"] is None
    assert block["date"] is None
    assert block["derivation"] == "d1"
    assert block["sheet_path"] == "qc/delta_qc.png"
    assert abs(block["auto"]["median_mm"] - 1.0) < 1e-6  # 10 Hz * 0.05 s * 2 mm
    assert (case_root / "qc" / "delta_qc.png").is_file()


def test_cli_mask_override_is_used_not_manifest_mask(tmp_path):
    case_root = tmp_path / "case"
    (case_root / "raw" / "dwi").mkdir(parents=True)
    (case_root / "nifti").mkdir()
    nib.save(_nii(np.full((6, 6, 6), 10.0)), str(case_root / "field_hz.nii.gz"))
    nib.save(_nii(np.ones((6, 6, 6))), str(case_root / "nifti" / "mask.nii.gz"))
    native = np.zeros((6, 6, 6))
    native[0, 0, 0] = 1
    nib.save(_nii(native), str(case_root / "native_mask.nii.gz"))
    nib.save(_nii(np.zeros((6, 6, 6))), str(case_root / "raw" / "dwi" / "dwi.nii.gz"))
    (case_root / "raw" / "dwi" / "dwi.json").write_text(
        json.dumps({"TotalReadoutTime": 0.05}))
    (case_root / "manifest.json").write_text(json.dumps({
        "case_id": "stub",
        "raw": {"dwi": {"path": "raw/dwi/dwi.nii.gz"}},
        "inputs": {"mask": {"path": "nifti/mask.nii.gz"}},
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "active_derivation": "d1",
    }))
    rc = delta_qc.main([
        str(case_root), "--field", "field_hz.nii.gz",
        "--mask", "native_mask.nii.gz", "--pe-axis", "1",
    ])
    assert rc == 0
    block = json.loads((case_root / "manifest.json").read_text())["delta_qc"]
    assert block["auto"]["n_vox"] == 1


# --- Task 8 fix round 1: owner sign path for delta QC ----------------------

def _signable_case(tmp_path, *, block_derivation="d1", active_derivation="d1"):
    """A case with an unsigned delta_qc block + a real sheet file on disk."""
    case_root = tmp_path / "case"
    (case_root / "qc").mkdir(parents=True)
    (case_root / "qc" / "delta_qc.png").write_bytes(b"not-a-real-png")
    manifest = {
        "case_id": "stub",
        "case_root": str(case_root),
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "active_derivation": active_derivation,
        "inputs": {},
        "delta_qc": {
            "auto": {"median_mm": 1.73, "p95_mm": 2.1, "max_mm": 2.9, "n_vox": 100},
            "approved_by": None,
            "date": None,
            "sheet_sha": hashlib.sha256(b"not-a-real-png").hexdigest(),
            "sheet_path": "qc/delta_qc.png",
            "derivation": block_derivation,
        },
    }
    (case_root / "manifest.json").write_text(json.dumps(manifest))
    return case_root


def test_approve_delta_qc_happy_path_signs(tmp_path):
    case_root = _signable_case(tmp_path)
    live_sha = hashlib.sha256((case_root / "qc" / "delta_qc.png").read_bytes()).hexdigest()

    block = delta_qc.approve_delta_qc(case_root, approved_by="erion")

    assert block["approved_by"] == "erion"
    assert block["date"]  # ISO date string, non-empty
    assert block["sheet_sha"] == live_sha
    assert block["derivation"] == "d1"
    # persisted
    on_disk = json.loads((case_root / "manifest.json").read_text())["delta_qc"]
    assert on_disk["approved_by"] == "erion"


def test_approve_delta_qc_sheet_sha_mismatch_raises(tmp_path):
    case_root = _signable_case(tmp_path)
    with pytest.raises(ValueError, match="sheet_sha mismatch"):
        delta_qc.approve_delta_qc(case_root, approved_by="erion", sheet_sha="deadbeef")
    # manifest untouched by the failed sign
    assert json.loads((case_root / "manifest.json").read_text())["delta_qc"]["approved_by"] is None


def test_approve_delta_qc_derivation_mismatch_raises(tmp_path):
    case_root = _signable_case(tmp_path, block_derivation="d0-stale", active_derivation="d1")
    with pytest.raises(ValueError, match="derivation mismatch"):
        delta_qc.approve_delta_qc(case_root, approved_by="erion")
    assert json.loads((case_root / "manifest.json").read_text())["delta_qc"]["approved_by"] is None


def test_approve_delta_qc_missing_block_raises(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    (case_root / "manifest.json").write_text(json.dumps({
        "case_id": "stub", "case_root": str(case_root),
        "derivations": {"d1": {"kind": "rpe_pair"}}, "active_derivation": "d1",
        "inputs": {},
    }))
    with pytest.raises(ValueError, match="no delta_qc block"):
        delta_qc.approve_delta_qc(case_root, approved_by="erion")


def test_approve_delta_qc_missing_sheet_file_raises(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    manifest = {
        "case_id": "stub", "case_root": str(case_root),
        "derivations": {"d1": {"kind": "rpe_pair"}}, "active_derivation": "d1",
        "inputs": {},
        "delta_qc": {
            "auto": {"median_mm": 1.73}, "approved_by": None, "date": None,
            "sheet_sha": None, "sheet_path": "qc/delta_qc.png", "derivation": "d1",
        },
    }
    (case_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="QC sheet missing"):
        delta_qc.approve_delta_qc(case_root, approved_by="erion")


def test_approve_delta_qc_approved_by_never_auto_filled(tmp_path):
    """approved_by must come only from the caller's explicit arg."""
    case_root = _signable_case(tmp_path)
    with pytest.raises(TypeError):
        delta_qc.approve_delta_qc(case_root)  # no approved_by -> refuse to guess


def test_cli_approve_signs_from_terminal(tmp_path):
    case_root = _signable_case(tmp_path)
    rc = delta_qc.main([str(case_root), "--approve", "erion"])
    assert rc == 0
    block = json.loads((case_root / "manifest.json").read_text())["delta_qc"]
    assert block["approved_by"] == "erion"


def test_cli_approve_sheet_sha_mismatch_fails_loud(tmp_path):
    case_root = _signable_case(tmp_path)
    rc = delta_qc.main([str(case_root), "--approve", "erion", "--sheet-sha", "deadbeef"])
    assert rc == 1
    block = json.loads((case_root / "manifest.json").read_text())["delta_qc"]
    assert block["approved_by"] is None
