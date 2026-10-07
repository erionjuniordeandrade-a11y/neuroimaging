"""Standalone topup-field planner (D0 regen recipe). Never runs FSL in these tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tractlab import topup_field as tf


def _nii(path: Path, shape=(4, 4, 4, 2), zoom=1.7):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, np.diag([zoom, zoom, zoom, 1.0]))
    nib.save(img, str(path))


def _case(tmp_path, *, with_fmaps=True, ap_pe="j-", pa_pe="j", trt=0.04914, n_vol=2):
    root = tmp_path / "case"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    dwi = raw_dir / "dwi.nii.gz"
    _nii(dwi, shape=(4, 4, 4, 3))
    (raw_dir / "dwi.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": trt,
    }))
    raw = {"dwi": {"path": "raw/dwi.nii.gz"}}
    if with_fmaps:
        ap = raw_dir / "ap.nii.gz"
        pa = raw_dir / "pa.nii.gz"
        _nii(ap, shape=(4, 4, 4, n_vol))
        _nii(pa, shape=(4, 4, 4, n_vol))
        (raw_dir / "ap.json").write_text(json.dumps({
            "PhaseEncodingDirection": ap_pe,
            "TotalReadoutTime": trt,
        }))
        (raw_dir / "pa.json").write_text(json.dumps({
            "PhaseEncodingDirection": pa_pe,
            "TotalReadoutTime": trt,
        }))
        raw["fmap_dwi_ap"] = {"path": "raw/ap.nii.gz"}
        raw["fmap_dwi_pa"] = {"path": "raw/pa.nii.gz"}
    (root / "manifest.json").write_text(json.dumps({"case_id": "t", "raw": raw}))
    return root


def test_bids_pe_to_fsl_rows():
    assert tf.bids_pe_to_fsl("j-") == (0, -1, 0)
    assert tf.bids_pe_to_fsl("j") == (0, 1, 0)
    assert tf.bids_pe_to_fsl("i") == (1, 0, 0)
    with pytest.raises(ValueError, match="PhaseEncodingDirection"):
        tf.bids_pe_to_fsl("q")


def test_acqparams_repeats_one_row_per_volume():
    text = tf.acqparams_text(
        ap_pe="j-", pa_pe="j", readout_s=0.04914, n_ap=3, n_pa=3,
    )
    rows = [ln.split() for ln in text.strip().splitlines()]
    assert len(rows) == 6
    assert rows[0] == ["0", "-1", "0", "0.04914"]
    assert rows[3] == ["0", "1", "0", "0.04914"]


def test_plan_is_argv_only_and_default_is_dry(tmp_path):
    plan = tf.plan_topup_field(_case(tmp_path, n_vol=2))
    assert isinstance(plan, tf.TopupFieldPlan)
    assert plan.field_path.name == "field_hz.nii.gz"
    flat = sum(plan.commands, [])
    assert "fslmerge" in flat and "topup" in flat
    fout = next(t for t in flat if t == "--fout" or t.startswith("--fout="))
    fout_path = flat[flat.index(fout) + 1] if fout == "--fout" else fout.split("=", 1)[1]
    assert fout_path == str(plan.field_path)
    for cmd in plan.commands:
        for i, tok in enumerate(cmd):
            if i > 0 and cmd[i - 1] in ("--datain", "--imain", "--out", "--fout", "--config"):
                continue
            # paths and flags; no shell join
            assert isinstance(tok, str)
    # dry-run must not invoke the runner
    calls = []
    res = tf.run_topup_field(plan, apply=False, runner=lambda *a, **k: calls.append(a) or type("R", (), {"returncode": 0})())
    assert res.ok is True and res.ran is False and calls == []


def test_plan_without_fmaps_is_typed_null(tmp_path):
    out = tf.plan_topup_field(_case(tmp_path, with_fmaps=False))
    assert isinstance(out, tf.NoReversePE)
    assert "fmap" in out.reason


def test_plan_refuses_non_opposite_pe(tmp_path):
    with pytest.raises(ValueError, match="opposite"):
        tf.plan_topup_field(_case(tmp_path, ap_pe="j-", pa_pe="j-"))


def test_plan_refuses_readout_mismatch(tmp_path):
    root = _case(tmp_path)
    (root / "raw/pa.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j",
        "TotalReadoutTime": 0.03,
    }))
    with pytest.raises(ValueError, match="TotalReadoutTime"):
        tf.plan_topup_field(root)


def test_validate_field_hz_requires_finite_3d_and_receipt_units(tmp_path):
    field = tmp_path / "field_hz.nii.gz"
    data = np.ones((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(field))
    receipt = {"units": "Hz", "tool": "topup", "--fout": True}
    tf.validate_field_hz(field, expected_shape=(4, 4, 4), receipt=receipt)
    with pytest.raises(ValueError, match="units"):
        tf.validate_field_hz(field, expected_shape=(4, 4, 4), receipt={"units": "rad/s"})
    bad = tmp_path / "zero.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(bad))
    with pytest.raises(ValueError, match="all.zero"):
        tf.validate_field_hz(bad, expected_shape=(4, 4, 4), receipt=receipt)
