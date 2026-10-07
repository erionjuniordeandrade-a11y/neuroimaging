"""S-01/N-03 — canonical manifest threading + schema-v2 evidence fingerprints.

Synthetic fixtures only (small in-memory niftis under tmp_path).
"""
from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from tractlab.preflight import (
    FINGERPRINT_SCHEMA,
    build_receipt,
    compute_drift,
    run_preflight,
)

AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _bvec_bval(root, n=4):
    bvec = root / "dwi.bvec"
    bval = root / "dwi.bval"
    bvec.write_text("\n".join([" ".join(["0"] * n)] * 2 + [" ".join(["1"] * n)]) + "\n")
    bval.write_text(" ".join(["1000"] * n) + "\n")
    return bvec, bval


def _manifest(root):
    _bvec_bval(root)
    man = {
        "case_id": "tmp-identity",
        "case_root": str(root),
        "inputs": {"b0": {"path": "b0.nii.gz"}},
        "acquisition_evidence": {
            "gradients": {"path": "dwi.bvec"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(man))
    return man


def test_run_preflight_uses_the_passed_manifest_not_the_disk_copy(tmp_path):
    """The core S-01 fix: a server holding its own manifest object must score
    THAT object, never silently re-read a manifest.json that may differ."""
    man = _manifest(tmp_path)
    # Disk now disagrees with the in-memory object the caller holds.
    on_disk = dict(man)
    on_disk["acquisition_evidence"] = {}  # would make gradients "untestable"
    (tmp_path / "manifest.json").write_text(json.dumps(on_disk))

    report = run_preflight(str(tmp_path), man).to_json()
    grad = next(c for c in report["criteria"] if c["id"] == "gradients")
    assert grad["verdict"] == "pass"  # scored the passed-in `man`, not disk


def test_run_preflight_defaults_to_reading_disk_for_cli_callers(tmp_path):
    man = _manifest(tmp_path)
    report = run_preflight(str(tmp_path)).to_json()  # no manifest kwarg
    grad = next(c for c in report["criteria"] if c["id"] == "gradients")
    assert grad["verdict"] == "pass"


def test_report_carries_schema_v2_fingerprints_per_criterion(tmp_path):
    man = _manifest(tmp_path)
    report = run_preflight(str(tmp_path), man).to_json()
    assert report["schema"] == FINGERPRINT_SCHEMA == 2
    assert set(report["fingerprints"]) == {c["id"] for c in report["criteria"]}
    assert all(isinstance(v, str) and len(v) == 64 for v in report["fingerprints"].values())


def test_fingerprint_changes_when_bvec_bytes_change_even_if_verdict_repeats(tmp_path):
    man = _manifest(tmp_path)
    fp1 = run_preflight(str(tmp_path), man).to_json()["fingerprints"]["gradients"]
    # Same shape/verdict-relevant counts, different actual numbers -> same
    # verdict ("pass"), different bytes.
    (tmp_path / "dwi.bvec").write_text(
        "\n".join([" ".join(["0"] * 4)] * 2 + [" ".join(["0.999"] * 4)]) + "\n"
    )
    v1 = run_preflight(str(tmp_path), man)
    assert next(c for c in v1.criteria if c.id == "gradients").verdict == "pass"
    fp2 = v1.to_json()["fingerprints"]["gradients"]
    assert fp1 != fp2


def test_fingerprint_binds_the_bval_companion_not_just_the_declared_bvec(tmp_path):
    man = _manifest(tmp_path)
    fp1 = run_preflight(str(tmp_path), man).to_json()["fingerprints"]["gradients"]
    (tmp_path / "dwi.bval").write_text("2000 2000 2000 2000\n")
    fp2 = run_preflight(str(tmp_path), man).to_json()["fingerprints"]["gradients"]
    assert fp1 != fp2


def test_compute_drift_flags_same_verdict_changed_bytes_as_stale(tmp_path):
    """The literal S-01 claim: 'same verdict with changed bytes is stale'."""
    man = _manifest(tmp_path)
    stored = build_receipt(str(tmp_path), man)
    (tmp_path / "dwi.bvec").write_text(
        "\n".join([" ".join(["0"] * 4)] * 2 + [" ".join(["0.5"] * 4)]) + "\n"
    )
    verified = run_preflight(str(tmp_path), man).to_json()
    live_verdict = next(c for c in verified["criteria"] if c["id"] == "gradients")["verdict"]
    assert live_verdict == "pass"  # verdict unchanged...
    drift = compute_drift(stored, verified, active_derivation=stored.get("derivation"))
    assert "gradients" in drift  # ...but the fingerprint moved, so it's drift


def test_compute_drift_clean_when_nothing_changed(tmp_path):
    man = _manifest(tmp_path)
    stored = build_receipt(str(tmp_path), man)
    verified = run_preflight(str(tmp_path), man).to_json()
    assert compute_drift(stored, verified, active_derivation=stored.get("derivation")) == []


def test_old_schema_receipt_always_reports_schema_drift(tmp_path):
    """Old receipts require explicit new review — never auto-approved or
    silently reinterpreted under the new fingerprint contract."""
    man = _manifest(tmp_path)
    stored = build_receipt(str(tmp_path), man)
    stored["schema"] = 1  # simulate a receipt written before fingerprints existed
    stored.pop("fingerprints", None)
    verified = run_preflight(str(tmp_path), man).to_json()
    drift = compute_drift(stored, verified, active_derivation=stored.get("derivation"))
    assert "schema" in drift


def _nifti(path, shape=(8, 8, 8)):
    rng = np.random.RandomState(0)
    data = rng.rand(*shape).astype(np.float32)
    data[2:6, 2:6, 2:6] += 0.4
    image = nib.Nifti1Image(data, AFF)
    image.header.set_xyzt_units('mm')
    nib.save(image, str(path))


def _mask(path, shape=(8, 8, 8)):
    m = np.zeros(shape, np.uint8)
    m[2:6, 2:6, 2:6] = 1
    image = nib.Nifti1Image(m, AFF)
    image.header.set_xyzt_units('mm')
    nib.save(image, str(path))


def test_registration_criterion_reads_grid_id_from_passed_manifest(tmp_path):
    """N-03: registration_t1_b0 must use the caller's manifest object too,
    not a fresh disk read of manifest.json (the historical inconsistency)."""
    from tractlab.grid import grid_id, load_grid

    # Same recipe as test_preflight.py's passing registration fixture: both
    # volumes come from RandomState(0), so they are identical arrays.
    _nifti(tmp_path / "b0.nii.gz")
    _nifti(tmp_path / "t1.nii.gz")
    _mask(tmp_path / "mask.nii.gz")
    gid = grid_id(load_grid(str(tmp_path / "b0.nii.gz")))

    man = {
        "case_id": "tmp-reg",
        "case_root": str(tmp_path),
        "inputs": {"b0": {"path": "b0.nii.gz"}},
        "acquisition_evidence": {
            "registration_t1_b0": {
                "path": "t1.nii.gz", "b0": "b0.nii.gz", "mask": "mask.nii.gz",
            },
        },
        "grid": {"grid_id": gid},
    }
    # Disk manifest disagrees (wrong grid_id) — the passed-in object must win.
    on_disk = dict(man)
    on_disk["grid"] = {"grid_id": "0" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(on_disk))

    report = run_preflight(str(tmp_path), man).to_json()
    reg = next(c for c in report["criteria"] if c["id"] == "registration_t1_b0")
    assert reg["verdict"] == "pass"
