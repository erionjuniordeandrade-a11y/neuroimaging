"""E3/T12 — prepare CLI, /api/preflight, viewer chip. Case load never writes."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import urllib.error
import urllib.request

from tractlab.preflight import CRITERION_IDS, main, run_preflight
from tractlab.serve import serve

VIEWER = str(Path(__file__).resolve().parents[1] / "viewer")
AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nifti(path: Path, shape=(8, 8, 8)):
    rng = np.random.RandomState(0)
    data = rng.rand(*shape).astype(np.float32)
    data[2:6, 2:6, 2:6] += 0.4
    image = nib.Nifti1Image(data, AFF)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


def _mask(path: Path, shape=(8, 8, 8)):
    m = np.zeros(shape, np.uint8)
    m[2:6, 2:6, 2:6] = 1
    image = nib.Nifti1Image(m, AFF)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


def _fod(path: Path, shape=(8, 8, 8)):
    fod = np.zeros(shape + (45,), np.float32)
    fod[..., 0] = 2.0
    image = nib.Nifti1Image(fod, AFF)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


def _write_gradients(root: Path):
    (root / "dwi.bvec").write_text(
        "\n".join([" ".join(["0"] * 8), " ".join(["0"] * 8), " ".join(["1"] * 8)]) + "\n"
    )
    (root / "dwi.bval").write_text("0 0 1000 1000 1000 1000 2000 2000\n")


def _case(tmp_path: Path, *, with_fod: bool = False) -> Path:
    _nifti(tmp_path / "b0.nii.gz")
    _nifti(tmp_path / "t1.nii.gz")
    _mask(tmp_path / "mask.nii.gz")
    _write_gradients(tmp_path)
    man = {
        "case_id": "tmp-preflight-api",
        "case_root": str(tmp_path),
        "inputs": {
            "b0": {"path": "b0.nii.gz"},
            "t1": {"path": "t1.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
        },
        "acquisition_evidence": {
            "gradients": {"path": "dwi.bvec", "sha256": _sha(tmp_path / "dwi.bvec")},
        },
    }
    if with_fod:
        _fod(tmp_path / "fod.nii.gz")
        man["inputs"]["fod"] = {"path": "fod.nii.gz"}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return tmp_path


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _serve(manifest: Path):
    httpd, service = serve(str(manifest), VIEWER, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, service, base


def test_prepare_dry_run_prints_and_writes_nothing(tmp_path, capsys):
    root = _case(tmp_path)
    rc = main(["prepare", str(root)])
    assert rc == 0
    assert not (root / "preflight.json").exists()
    blob = json.loads(capsys.readouterr().out)
    assert blob["schema"] == 2
    assert blob["approved_by"] is None
    assert set(k["id"] for k in blob["criteria"]) == set(CRITERION_IDS)
    assert blob["produced_with"]["tractlab_commit"]
    assert blob["produced_with"]["python"]
    assert blob["produced_with"]["nibabel"]


def test_prepare_apply_writes_unsigned_receipt(tmp_path):
    root = _case(tmp_path)
    rc = main(["prepare", str(root), "--apply"])
    assert rc == 0
    path = root / "preflight.json"
    blob = json.loads(path.read_text())
    assert blob["schema"] == 2
    assert blob["approved_by"] is None
    assert {c["id"] for c in blob["criteria"]} == set(CRITERION_IDS)
    assert blob["produced_with"]["tractlab_commit"]


def test_prepare_apply_cannot_set_approved_by(tmp_path):
    root = _case(tmp_path)
    rc = main(["prepare", str(root), "--apply", "--approve", "erion"])
    assert rc != 0
    assert not (root / "preflight.json").exists()


def test_approve_refuses_without_receipt(tmp_path):
    root = _case(tmp_path)
    rc = main(["approve", str(root), "--who", "erion"])
    assert rc == 1
    assert not (root / "preflight.json").exists()


def test_approve_refuses_when_drifted(tmp_path):
    root = _case(tmp_path)
    assert main(["prepare", str(root), "--apply"]) == 0
    (root / "dwi.bvec").write_text("1 0\n0 1\n0 0\n")
    rc = main(["approve", str(root), "--who", "erion"])
    assert rc == 1
    blob = json.loads((root / "preflight.json").read_text())
    assert blob["approved_by"] is None


def test_apply_refuses_to_overwrite_signed_receipt(tmp_path):
    root = _case(tmp_path)
    assert main(["prepare", str(root), "--apply"]) == 0
    assert main(["approve", str(root), "--who", "erion"]) == 0
    before = (root / "preflight.json").read_text()
    rc = main(["prepare", str(root), "--apply"])
    assert rc == 1
    assert (root / "preflight.json").read_text() == before


def test_receipt_stamps_active_derivation_and_drifts_on_lineage_change(tmp_path):
    root = _case(tmp_path, with_fod=True)
    man = json.loads((root / "manifest.json").read_text())
    man["derivations"] = {"d1": {"kind": "rpe_pair"}}
    man["active_derivation"] = "d1"
    (root / "manifest.json").write_text(json.dumps(man))
    assert main(["prepare", str(root), "--apply"]) == 0
    blob = json.loads((root / "preflight.json").read_text())
    assert blob["derivation"] == "d1"
    d2 = root / "d2"
    d2.mkdir()
    _nifti(d2 / "b0.nii.gz")
    _nifti(d2 / "t1.nii.gz")
    _mask(d2 / "mask.nii.gz")
    _fod(d2 / "fod.nii.gz")
    man["active_derivation"] = "d2"
    man["derivations"]["d2"] = {
        "kind": "rpe_pair",
        "inputs": {
            name: {
                "path": f"d2/{name}.nii.gz",
                "sha256_by_field": {
                    "path": _sha(d2 / f"{name}.nii.gz"),
                },
                "bytes_by_field": {
                    "path": (d2 / f"{name}.nii.gz").stat().st_size,
                },
            }
            for name in ("b0", "t1", "mask", "fod")
        },
    }
    (root / "manifest.json").write_text(json.dumps(man))
    httpd, service, base = _serve(root / "manifest.json")
    try:
        status, body, _ = _get(base + "/api/preflight")
        assert status == 200
        payload = json.loads(body)
        assert "derivation" in payload["drift"]
    finally:
        httpd.shutdown()


def test_approve_writes_who_only_when_clean(tmp_path):
    root = _case(tmp_path)
    assert main(["prepare", str(root), "--apply"]) == 0
    rc = main(["approve", str(root), "--who", "erion"])
    assert rc == 0
    blob = json.loads((root / "preflight.json").read_text())
    assert blob["approved_by"] == "erion"
    assert blob.get("approved_date")


def test_api_preflight_absent_stored_no_drift(tmp_path):
    root = _case(tmp_path, with_fod=True)
    httpd, service, base = _serve(root / "manifest.json")
    try:
        status, body, _ = _get(base + "/api/preflight")
        assert status == 200
        payload = json.loads(body)
        assert payload["stored"] is None
        assert {c["id"] for c in payload["verified"]["criteria"]} == set(CRITERION_IDS)
        assert payload["drift"] == []
    finally:
        httpd.shutdown()


@pytest.mark.parametrize('receipt', ['{bad json', '{"schema":99,"approved_by":"reviewer"}', '{"schema":2,"criteria":[]}'])
def test_api_incompatible_receipt_is_typed_and_never_signed(tmp_path, receipt):
    root = _case(tmp_path, with_fod=True)
    (root / 'preflight.json').write_text(receipt)
    httpd, service, base = _serve(root / 'manifest.json')
    try:
        status, body, _ = _get(base + '/api/preflight')
        assert status == 409
        assert json.loads(body)['code'] == 'preflight_receipt_incompatible'
        assert 'stored' not in json.loads(body)
        assert (root / 'preflight.json').read_text() == receipt
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_preflight_reports_empty_drift_when_unchanged(tmp_path):
    root = _case(tmp_path, with_fod=True)
    assert main(["prepare", str(root), "--apply"]) == 0
    httpd, service, base = _serve(root / "manifest.json")
    try:
        status, body, _ = _get(base + "/api/preflight")
        assert status == 200
        payload = json.loads(body)
        assert payload["stored"] is not None
        assert payload["stored"]["approved_by"] is None
        assert payload["drift"] == []
    finally:
        httpd.shutdown()


def test_api_preflight_names_drifted_criterion(tmp_path):
    root = _case(tmp_path, with_fod=True)
    assert main(["prepare", str(root), "--apply"]) == 0
    (root / "dwi.bvec").write_text("1 0\n0 1\n0 0\n")
    httpd, service, base = _serve(root / "manifest.json")
    try:
        status, body, _ = _get(base + "/api/preflight")
        assert status == 200
        payload = json.loads(body)
        assert "gradients" in payload["drift"]
        live = next(c for c in payload["verified"]["criteria"] if c["id"] == "gradients")
        assert live["verdict"] == "fail"
    finally:
        httpd.shutdown()


def test_case_load_never_writes_preflight(tmp_path):
    root = _case(tmp_path, with_fod=True)
    man = root / "manifest.json"
    before_man = man.stat().st_mtime_ns
    assert not (root / "preflight.json").exists()
    httpd, service, base = _serve(man)
    try:
        _get(base + "/api/health")
        _get(base + "/api/preflight")
        assert not (root / "preflight.json").exists()
        assert man.stat().st_mtime_ns == before_man
    finally:
        httpd.shutdown()


def test_viewer_path_is_this_checkout_not_a_worktree_pin():
    expected = Path(__file__).resolve().parents[1] / "viewer"
    assert Path(VIEWER).resolve() == expected.resolve()


def test_viewer_has_preflight_chip_and_card():
    html = Path(VIEWER, "index.html").read_text()
    assert 'id="provChipPreflight"' in html
    assert 'id="provCard"' in html
    assert "/api/preflight" in html
    for state in ("signed", "unsigned", "absent"):
        assert state in html
