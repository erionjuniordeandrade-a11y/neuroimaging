"""E3/T11 — preflight criteria: one pass and one broken fixture each."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.preflight import CRITERION_IDS, run_preflight


AFF = np.diag([2.0, 2.0, 2.0, 1.0])
ANISO = np.diag([2.0, 2.0, 4.0, 1.0])
FLIP = np.diag([-2.0, 2.0, 2.0, 1.0])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nifti(path: Path, affine=AFF, shape=(8, 8, 8)):
    rng = np.random.RandomState(0)
    data = rng.rand(*shape).astype(np.float32)
    data[2:6, 2:6, 2:6] += 0.4
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))
    return path


def _mask(path: Path, affine=AFF, shape=(8, 8, 8)):
    m = np.zeros(shape, np.uint8)
    m[2:6, 2:6, 2:6] = 1
    image = nib.Nifti1Image(m, affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))
    return path


def _write_man(
    root: Path,
    evidence: dict | None = None,
    extra: dict | None = None,
    *,
    manifest_root: Path | None = None,
):
    man = {
        "case_id": "tmp-preflight",
        "case_root": str(root),
        "inputs": {
            "b0": {"path": "b0.nii.gz"},
            "t1": {"path": "t1.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
        },
    }
    if evidence is not None:
        man["acquisition_evidence"] = evidence
    if extra:
        man.update(extra)
    target_root = root if manifest_root is None else manifest_root
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "manifest.json").write_text(json.dumps(man))
    return root


def _base_case(tmp_path: Path):
    _nifti(tmp_path / "b0.nii.gz")
    _nifti(tmp_path / "t1.nii.gz")
    _mask(tmp_path / "mask.nii.gz")
    bvec = "\n".join([" ".join(["0"] * 8), " ".join(["0"] * 8), " ".join(["1"] * 8)]) + "\n"
    bval = "0 0 1000 1000 1000 1000 2000 2000\n"
    (tmp_path / "dwi.bvec").write_text(bvec)
    (tmp_path / "dwi.bval").write_text(bval)
    (tmp_path / "eddy_qc.json").write_text(json.dumps({"qc_motion": 0.2}))
    (tmp_path / "topup.done").write_text("ok\n")
    return tmp_path


def ev_file(root: Path, rel: str) -> dict:
    return {"path": rel, "sha256": _sha(root / rel)}


def ev_absent(reason: str, signed_by: str = "erion") -> dict:
    return {"absent_reason": reason, "signed_by": signed_by}


def c(report, cid: str):
    return next(x for x in report.criteria if x.id == cid)


def test_undeclared_evidence_is_untestable_never_pass(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path)
    r = run_preflight(str(tmp_path))
    assert [x.id for x in r.criteria] == list(CRITERION_IDS)
    for cid in CRITERION_IDS:
        assert c(r, cid).verdict == "untestable"
        assert "no declared evidence" in c(r, cid).evidence


def test_gradients_untestable_when_undeclared(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {})
    assert c(run_preflight(str(tmp_path)), "gradients").verdict == "untestable"


def test_gradients_fail_on_count_mismatch(tmp_path):
    _base_case(tmp_path)
    (tmp_path / "dwi.bvec").write_text("1 0\n0 1\n0 0\n")
    (tmp_path / "dwi.bval").write_text("0 1000 2000\n")
    _write_man(tmp_path, {"gradients": ev_file(tmp_path, "dwi.bvec")})
    crit = c(run_preflight(str(tmp_path)), "gradients")
    assert crit.verdict == "fail"
    assert "count" in crit.evidence.lower() or "mismatch" in crit.evidence.lower()


def test_gradients_pass_when_bvec_bval_agree(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"gradients": ev_file(tmp_path, "dwi.bvec")})
    crit = c(run_preflight(str(tmp_path)), "gradients")
    assert crit.verdict == "pass"
    assert "dwi.bvec" in crit.evidence


def test_run_preflight_uses_supplied_manifest_for_separate_data_root(tmp_path):
    from tractlab.grid import grid_id, load_grid

    data_root = tmp_path / "data"
    manifest_root = tmp_path / "case-control"
    data_root.mkdir()
    _base_case(data_root)
    registration = ev_file(data_root, "t1.nii.gz")
    registration["b0"] = "b0.nii.gz"
    registration["mask"] = "mask.nii.gz"
    _write_man(
        data_root,
        {
            "gradients": ev_file(data_root, "dwi.bvec"),
            "registration_t1_b0": registration,
        },
        extra={"grid": {"grid_id": grid_id(load_grid(str(data_root / "b0.nii.gz")))}},
        manifest_root=manifest_root,
    )

    manifest_path = manifest_root / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_before)
    report = run_preflight(str(data_root), manifest=manifest)

    assert not (data_root / "manifest.json").exists()
    assert c(report, "gradients").verdict == "pass"
    assert c(report, "registration_t1_b0").verdict == "pass"
    assert manifest_path.read_bytes() == manifest_before
    assert not (data_root / "preflight.json").exists()


def test_gradients_fail_on_sha_mismatch(tmp_path):
    _base_case(tmp_path)
    bad = ev_file(tmp_path, "dwi.bvec")
    bad["sha256"] = "00" * 32
    _write_man(tmp_path, {"gradients": bad})
    crit = c(run_preflight(str(tmp_path)), "gradients")
    assert crit.verdict == "fail"
    assert "sha" in crit.evidence.lower()


def test_shells_pass_on_multi_shell(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"shells": ev_file(tmp_path, "dwi.bval")})
    crit = c(run_preflight(str(tmp_path)), "shells")
    assert crit.verdict == "pass"
    assert "dwi.bval" in crit.evidence


def test_shells_degraded_on_single_shell(tmp_path):
    _base_case(tmp_path)
    (tmp_path / "dwi.bval").write_text("0 0 1000 1000 1000 1000 1000 1000\n")
    _write_man(tmp_path, {"shells": ev_file(tmp_path, "dwi.bval")})
    assert c(run_preflight(str(tmp_path)), "shells").verdict == "degraded"


def test_shells_degraded_when_b1000_cluster_is_the_only_dwi_shell(tmp_path):
    _base_case(tmp_path)
    (tmp_path / "dwi.bval").write_text("5 5 995 1000 1005 1000 995 1005\n")
    _write_man(tmp_path, {"shells": ev_file(tmp_path, "dwi.bval")})
    crit = c(run_preflight(str(tmp_path)), "shells")
    assert crit.verdict == "degraded"
    assert "single DWI shell" in crit.evidence


def test_isotropy_degraded_on_anisotropic_voxels(tmp_path):
    _base_case(tmp_path)
    _nifti(tmp_path / "b0.nii.gz", affine=ANISO)
    _write_man(tmp_path, {"isotropy": ev_file(tmp_path, "b0.nii.gz")})
    assert c(run_preflight(str(tmp_path)), "isotropy").verdict == "degraded"


def test_isotropy_pass_on_isotropic_voxels(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"isotropy": ev_file(tmp_path, "b0.nii.gz")})
    assert c(run_preflight(str(tmp_path)), "isotropy").verdict == "pass"


def test_orientation_fail_on_flipped_affine(tmp_path):
    _base_case(tmp_path)
    _nifti(tmp_path / "b0.nii.gz", affine=FLIP)
    ev = ev_file(tmp_path, "b0.nii.gz")
    ev["expected_affine"] = AFF.ravel().tolist()
    _write_man(tmp_path, {"orientation": ev})
    assert c(run_preflight(str(tmp_path)), "orientation").verdict == "fail"


def test_orientation_pass_when_affine_matches(tmp_path):
    _base_case(tmp_path)
    ev = ev_file(tmp_path, "b0.nii.gz")
    ev["expected_affine"] = AFF.ravel().tolist()
    _write_man(tmp_path, {"orientation": ev})
    assert c(run_preflight(str(tmp_path)), "orientation").verdict == "pass"


def test_pe_correction_degraded_when_absent_signed(tmp_path):
    _base_case(tmp_path)
    _write_man(
        tmp_path,
        {"pe_correction": ev_absent("no reverse-PE acquired", "erion")},
    )
    crit = c(run_preflight(str(tmp_path)), "pe_correction")
    assert crit.verdict == "degraded"
    assert "no reverse-PE" in crit.evidence


def test_pe_correction_untestable_when_absent_unsigned(tmp_path):
    _base_case(tmp_path)
    _write_man(
        tmp_path,
        {"pe_correction": {"absent_reason": "unknown", "signed_by": None}},
    )
    assert c(run_preflight(str(tmp_path)), "pe_correction").verdict == "untestable"


def test_pe_correction_pass_when_receipt_present(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"pe_correction": ev_file(tmp_path, "topup.done")})
    crit = c(run_preflight(str(tmp_path)), "pe_correction")
    assert crit.verdict == "pass"
    assert "topup.done" in crit.evidence


def test_registration_fail_on_zero_t1(tmp_path):
    from tractlab.grid import grid_id, load_grid

    _base_case(tmp_path)
    zeros = np.zeros((8, 8, 8), np.float32)
    image = nib.Nifti1Image(zeros, AFF)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(tmp_path / "t1.nii.gz"))
    ev = ev_file(tmp_path, "t1.nii.gz")
    ev["b0"] = "b0.nii.gz"
    ev["mask"] = "mask.nii.gz"
    gid = grid_id(load_grid(str(tmp_path / "b0.nii.gz")))
    _write_man(
        tmp_path,
        {"registration_t1_b0": ev},
        extra={"grid": {"grid_id": gid}},
    )
    assert c(run_preflight(str(tmp_path)), "registration_t1_b0").verdict == "fail"


def test_registration_pass_on_aligned_volumes(tmp_path):
    from tractlab.grid import grid_id, load_grid

    _base_case(tmp_path)
    ev = ev_file(tmp_path, "t1.nii.gz")
    ev["b0"] = "b0.nii.gz"
    ev["mask"] = "mask.nii.gz"
    gid = grid_id(load_grid(str(tmp_path / "b0.nii.gz")))
    _write_man(
        tmp_path,
        {"registration_t1_b0": ev},
        extra={"grid": {"grid_id": gid}},
    )
    crit = c(run_preflight(str(tmp_path)), "registration_t1_b0")
    assert crit.verdict == "pass"
    assert "auto_sanity" in crit.evidence or "t1.nii.gz" in crit.evidence


def test_registration_fail_on_manifest_grid_mismatch(tmp_path):
    _base_case(tmp_path)
    ev = ev_file(tmp_path, "t1.nii.gz")
    ev["b0"] = "b0.nii.gz"
    ev["mask"] = "mask.nii.gz"
    _write_man(
        tmp_path,
        {"registration_t1_b0": ev},
        extra={"grid": {"grid_id": "0" * 64}},
    )
    crit = c(run_preflight(str(tmp_path)), "registration_t1_b0")
    assert crit.verdict == "fail"
    assert "grid_id" in crit.evidence


def test_eddy_qc_signed_absence_is_untestable_with_reason(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"eddy_qc": ev_absent("eddy_quad never ran", signed_by="erion")})
    crit = c(run_preflight(str(tmp_path)), "eddy_qc")
    assert crit.verdict == "untestable"
    assert "eddy_quad never ran" in crit.evidence
    assert "erion" in crit.evidence


def test_eddy_qc_ingests_never_recomputes(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path, {"eddy_qc": ev_file(tmp_path, "eddy_qc.json")})
    crit = c(run_preflight(str(tmp_path)), "eddy_qc")
    assert crit.verdict == "pass"
    assert "qc.json" in crit.evidence or "eddy_qc.json" in crit.evidence


def test_eddy_qc_fail_on_sha_mismatch(tmp_path):
    _base_case(tmp_path)
    bad = ev_file(tmp_path, "eddy_qc.json")
    bad["sha256"] = "ff" * 32
    _write_man(tmp_path, {"eddy_qc": bad})
    crit = c(run_preflight(str(tmp_path)), "eddy_qc")
    assert crit.verdict == "fail"
    assert "sha" in crit.evidence.lower()


def test_report_to_json_roundtrip_shape(tmp_path):
    _base_case(tmp_path)
    _write_man(tmp_path)
    blob = run_preflight(str(tmp_path)).to_json()
    assert blob["schema"] == 2
    assert {c["id"] for c in blob["criteria"]} == set(CRITERION_IDS)
    assert all("verdict" in c and "evidence" in c for c in blob["criteria"])
