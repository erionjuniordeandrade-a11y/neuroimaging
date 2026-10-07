"""Single-flight lock cannot leak; cancel_active gives tckgen a SIGTERM grace."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.serve import TrackParams, serve

VIEWER = str(Path(__file__).resolve().parents[1] / "viewer")
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
Y00 = 0.28209479177387814


def _tiny_case(tmp_path, *, with_sidecar: bool = False):
    """Serveable tmp case: b0/mask/fod + one bank. Copied from test_serve_fidelity_block."""
    del with_sidecar  # lock tests never need a sidecar
    shape = (24, 24, 24)
    mask = np.zeros(shape, np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    b0 = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    fod = np.zeros(shape + (45,), dtype=np.float32)
    fod[..., 0] = 2.0
    peak = (fod[..., 0] * Y00).astype(np.float32)
    for name, data in (("mask", mask), ("b0", b0), ("fod", fod), ("peak", peak)):
        image = nib.Nifti1Image(data, AFF)
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, str(tmp_path / f"{name}.nii.gz"))

    lines = [
        np.array([[10.0 + i, 14.0, 14.0], [22.0 + i, 14.0, 14.0]], dtype=np.float32)
        for i in range(5)
    ]
    tck = tmp_path / "bank.tck"
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
        str(tck),
    )
    bank_sha = hashlib.sha256(tck.read_bytes()).hexdigest()
    fod_sha = hashlib.sha256((tmp_path / "fod.nii.gz").read_bytes()).hexdigest()
    prov = {
        "bank_sha256": bank_sha,
        "fod_sha256": fod_sha,
        "sources": {
            "bank_sha256": "recorded",
            "fod_sha256": "recorded",
        },
    }
    man = {
        "case_id": "tmp-fid",
        "case_root": str(tmp_path),
        "inputs": {
            "fod": {"path": "fod.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "b0": {"path": "b0.nii.gz"},
            "bank_test": {
                "path": "bank.tck",
                "label": "TEST",
                "engine": "fixture",
                "provenance": prov,
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return tmp_path / "manifest.json"


def _serve(manifest):
    httpd, service = serve(str(manifest), VIEWER, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, service, base


def test_lock_released_when_mkdtemp_fails(tmp_path, monkeypatch):
    man = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        monkeypatch.setattr(
            tempfile, "mkdtemp",
            lambda *a, **k: (_ for _ in ()).throw(OSError("ENOSPC")),
        )
        with pytest.raises(OSError):
            service.track({}, TrackParams())
        assert not service._lock.locked(), "single-flight lock leaked"
    finally:
        httpd.shutdown()


def test_cancel_cooperative_process_dies_fast(tmp_path):
    man = _tiny_case(tmp_path)
    httpd, service, _base = _serve(man)
    proc = None
    try:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        service._register_proc(proc, "job1")
        t0 = time.time()
        result = service.cancel_active()
        elapsed = time.time() - t0
        assert result == {"cancelled": True, "jobId": "job1"}
        assert proc.poll() is not None
        assert elapsed < 5, f"cooperative cancel took {elapsed:.3f}s"
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        httpd.shutdown()


def test_cancel_term_ignoring_process_gets_sigkill_after_grace(tmp_path):
    man = _tiny_case(tmp_path)
    httpd, service, _base = _serve(man)
    proc = None
    try:
        ready = tmp_path / "term_trap_ready"
        proc = subprocess.Popen(
            ["bash", "-c", f'trap "" TERM; echo ready > "{ready}"; sleep 30'],
            start_new_session=True,
        )
        deadline = time.time() + 2
        while not ready.exists():
            if time.time() > deadline:
                pytest.fail("TERM trap never armed")
            time.sleep(0.01)
        service._register_proc(proc, "job1")
        t0 = time.time()
        result = service.cancel_active()
        elapsed = time.time() - t0
        assert result["cancelled"] is True
        assert result.get("jobId") == "job1"
        assert proc.poll() is not None
        assert elapsed >= 2, f"grace window skipped ({elapsed:.3f}s)"
        assert elapsed < 5, f"cancel after SIGKILL took {elapsed:.3f}s"
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        httpd.shutdown()
