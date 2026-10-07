"""Audit remediation contract tests (S-02..S-08, SEC-01/02, N-03).

All fixtures are tiny synthetic cases under tmp_path — no private-case
fixtures, no network, loopback-only. Mirrors the fixture recipe already used
by tests/test_serve_fidelity_block.py and tests/test_serve_lock.py.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import nibabel as nib
import numpy as np
import pytest

from tractlab import derivation as dv
from tractlab.fidelity import build_sidecar, save_sidecar
from tractlab.serve import serve

VIEWER = str(Path(__file__).resolve().parents[1] / "viewer")
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
Y00 = 0.28209479177387814


# ── shared fixture: tiny serveable case with one bank ───────────────────────

def _tiny_case(tmp_path, *, n_lines: int = 5, with_sidecar: bool = False, derivations=None):
    shape = (24, 24, 24)
    mask = np.zeros(shape, np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    b0 = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    fod = np.zeros(shape + (45,), dtype=np.float32)
    fod[..., 0] = 2.0
    peak = (fod[..., 0] * Y00).astype(np.float32)
    for name, data in [('mask', mask), ('b0', b0), ('fod', fod), ('peak', peak)]:
        image = nib.Nifti1Image(data, AFF)
        image.header.set_xyzt_units('mm')
        nib.save(image, str(tmp_path / f'{name}.nii.gz'))

    lines = [
        np.array([[10.0 + i, 14.0, 14.0], [22.0 + i, 14.0, 14.0]], dtype=np.float32)
        for i in range(n_lines)
    ]
    tck = tmp_path / "bank.tck"
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)), str(tck),
    )
    bank_sha = hashlib.sha256(tck.read_bytes()).hexdigest()
    fod_sha = hashlib.sha256((tmp_path / "fod.nii.gz").read_bytes()).hexdigest()
    prov = {
        "bank_sha256": bank_sha,
        "fod_sha256": fod_sha,
        "sources": {"bank_sha256": "recorded", "fod_sha256": "recorded"},
    }
    man = {
        "case_id": "tmp-audit",
        "case_root": str(tmp_path),
        "inputs": {
            "fod": {"path": "fod.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "b0": {"path": "b0.nii.gz"},
            "bank_test": {
                "path": "bank.tck", "label": "TEST", "engine": "fixture",
                "provenance": prov,
            },
        },
    }
    if derivations is not None:
        man["derivations"] = derivations["derivations"]
        man["active_derivation"] = derivations["active_derivation"]
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    if with_sidecar:
        data = build_sidecar(
            tck, tmp_path / "fod.nii.gz", tmp_path / "peak.nii.gz", provenance=prov,
        )
        fid_dir = tmp_path / "fidelity"
        fid_dir.mkdir()
        (fid_dir / "fod_peak.nii.gz").write_bytes((tmp_path / "peak.nii.gz").read_bytes())
        save_sidecar(fid_dir / "bank_test.fidelity.npz", data)
    return tmp_path / "manifest.json", tck


def _serve(manifest, viewer_dir=VIEWER):
    httpd, service = serve(str(manifest), viewer_dir, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, service, base


def _hdr(headers, name):
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


# ── raw HTTP helpers with full header control (urllib can't spoof Host) ────

def _raw(base, method, path, *, body=None, headers=None, host=None):
    """Full manual control over Host/Content-Length — urllib won't let a
    caller override Host or send a deliberately malformed Content-Length."""
    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    hdrs = dict(headers or {})
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host if host is not None else f"{parts.hostname}:{parts.port}")
        body_bytes = body if isinstance(body, (bytes, type(None))) else body.encode()
        has_cl = any(k.lower() == "content-length" for k in hdrs)
        if body_bytes is not None and not has_cl:
            conn.putheader("Content-Length", str(len(body_bytes)))
        for k, v in hdrs.items():
            conn.putheader(k, v)
        conn.endheaders()
        if body_bytes:
            conn.send(body_bytes)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data, dict(resp.getheaders())
    finally:
        conn.close()


def _post_json(base, path, obj, **kw):
    extra_headers = kw.pop("headers", {})
    headers = {"Content-Type": "application/json", **extra_headers}
    return _raw(base, "POST", path, body=json.dumps(obj), headers=headers, **kw)


def _get(base, path, **kw):
    return _raw(base, "GET", path, **kw)


def _load_bank(base, service, bid="bank_test"):
    return _post_json(base, "/api/bank/load", {
        "gridId": service.grid_id, "volumeId": service.volume_id, "bankId": bid,
    })


# ── S-02/03/04/07: immutable result registry + resultId contract ──────────

@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_bank_load_publishes_result_headers(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, _body, headers = _load_bank(base, service)
        assert status == 200
        assert _hdr(headers, "X-resultId")
        assert _hdr(headers, "X-resultSchema") == "1"
        assert _hdr(headers, "X-sourcePopulation") == "bank:bank_test"
        assert _hdr(headers, "X-sourceDigest")
        assert _hdr(headers, "X-gridId") == service.grid_id
        assert _hdr(headers, "X-volumeId") == service.volume_id
        assert _hdr(headers, "X-nAnalyticFull") == "5"
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_export_and_margin_require_resultid_no_banked_fallback(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, headers = _load_bank(base, service)
        result_id = _hdr(headers, "X-resultId")
        assert result_id

        status, body, headers = _post_json(base, "/api/export/tck", {"resultId": result_id})
        assert status == 200
        assert _hdr(headers, "X-resultId") == result_id
        assert int(_hdr(headers, "X-nStreamlines")) == 5
        assert len(body) > 0

        status, body, headers = _post_json(base, "/api/margin", {
            "resultId": result_id, "marginMm": 5.0,
        })
        assert status == 200
        assert _hdr(headers, "X-resultId") == result_id

        # bankId is NEVER interpreted as a result — export requires resultId.
        status, body, _ = _post_json(base, "/api/export/tck", {"bankId": "bank_test"})
        assert status == 400
        err = json.loads(body)
        assert err.get("code") == "bad_request"

        status, body, _ = _post_json(base, "/api/margin", {"bankId": "bank_test"})
        assert status == 400
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_unknown_and_expired_resultid_are_typed_404(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/export/tck", {"resultId": "nope-not-real"})
        assert status == 404
        err = json.loads(body)
        assert err.get("code") == "result_unavailable"

        status, body, _ = _post_json(base, "/api/margin", {"resultId": "nope-not-real"})
        assert status == 404
        assert json.loads(body).get("code") == "result_unavailable"
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_each_bank_load_publishes_a_distinct_result_not_a_shared_last(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        _s1, _b1, h1 = _load_bank(base, service)
        _s2, _b2, h2 = _load_bank(base, service)
        r1, r2 = _hdr(h1, "X-resultId"), _hdr(h2, "X-resultId")
        assert r1 and r2 and r1 != r2
        # Both remain independently exportable (no single mutable "last").
        for rid in (r1, r2):
            status, _body, _hdrs = _post_json(base, "/api/export/tck", {"resultId": rid})
            assert status == 200
    finally:
        httpd.shutdown()


def test_rejected_bank_load_publishes_nothing(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/bank/load", {
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "bankId": "does-not-exist",
        })
        assert status == 400
        assert service.analytic_sources._latest is None
    finally:
        httpd.shutdown()


# ── S-05: current-bytes fidelity identity, not declared-hash comparison ────

def test_bank_load_fidelity_ok_when_bytes_match_sidecar(tmp_path):
    man, _tck = _tiny_case(tmp_path, with_sidecar=True)
    httpd, service, base = _serve(man)
    try:
        status, _body, headers = _load_bank(base, service)
        assert status == 200
        assert _hdr(headers, "X-fidelityStatus") == "ok"
    finally:
        httpd.shutdown()


def test_same_count_permuted_bank_bytes_refuse_fidelity(tmp_path):
    """S-05: a same-streamline-count permutation of the bank file must
    refuse — the sidecar is checked against CURRENT bytes, not a declared
    hash string that nobody re-verified against the file on disk."""
    from tractlab import evidence_identity

    man, tck = _tiny_case(tmp_path, with_sidecar=True)
    httpd, service, base = _serve(man)
    try:
        status, _body, headers = _load_bank(base, service)
        assert status == 200
        assert _hdr(headers, "X-fidelityStatus") == "ok"

        # Same count, streamlines permuted -> different file bytes, same
        # manifest-declared provenance strings (untouched).
        tck_obj = nib.streamlines.load(str(tck))
        original = list(tck_obj.streamlines)
        permuted = [original[-1]] + original[:-1]
        nib.streamlines.save(
            nib.streamlines.Tractogram(permuted, affine_to_rasmm=np.eye(4)), str(tck),
        )
        evidence_identity.clear_cache()  # deterministic re-hash regardless of mtime granularity

        # main refuses earlier: the named bank source hash changed (409).
        status, body, _ = _load_bank(base, service)
        assert status == 409
        err = json.loads(body)
        assert err.get("code") == "bank_source_changed"
    finally:
        httpd.shutdown()


def test_old_schema_sidecar_refuses_explicitly(tmp_path):
    """Schema advancement (S-05) refuses an old sidecar outright rather than
    reinterpreting it — never a fabricated/re-signed result."""
    man, tck = _tiny_case(tmp_path, with_sidecar=True)
    sidecar_path = tmp_path / "fidelity" / "bank_test.fidelity.npz"
    with np.load(str(sidecar_path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["schema"] = np.asarray(1, dtype=np.int64)  # simulate a pre-S-05 sidecar
    import zipfile
    import io as _io

    with zipfile.ZipFile(str(sidecar_path), "w", zipfile.ZIP_STORED) as zf:
        for k, v in arrays.items():
            buf = _io.BytesIO()
            np.lib.format.write_array(buf, v, allow_pickle=False)
            zf.writestr(k + ".npy", buf.getvalue())

    httpd, service, base = _serve(man)
    try:
        status, body, _ = _load_bank(base, service)
        assert status == 422
        err = json.loads(body)
        assert "schema" in err.get("reason", "")
    finally:
        httpd.shutdown()


# ── S-06: derivation.validate() at startup ─────────────────────────────────

def test_startup_refuses_malformed_derivations_block(tmp_path):
    man, _tck = _tiny_case(tmp_path, derivations={
        "derivations": {"d1": {"kind": "not-a-real-kind"}},
        "active_derivation": "d1",
    })
    with pytest.raises(dv.DerivationError):
        serve(str(man), VIEWER, port=0)


@pytest.mark.skip(reason="main supports multiple derivations by design (tests/test_derivation_delta.py).")
def test_startup_refuses_multiple_derivations(tmp_path):
    man, _tck = _tiny_case(tmp_path, derivations={
        "derivations": {"d1": {"kind": "uncorrected"}, "d2": {"kind": "rpe_pair"}},
        "active_derivation": "d1",
    })
    with pytest.raises(dv.DerivationError, match="path re-layout"):
        serve(str(man), VIEWER, port=0)


def test_startup_accepts_well_formed_single_lineage(tmp_path):
    man, _tck = _tiny_case(tmp_path, derivations={
        "derivations": {"d1": {"kind": "uncorrected"}},
        "active_derivation": "d1",
    })
    httpd, service, base = _serve(man)
    try:
        assert service.derivation_id() == "d1"
    finally:
        httpd.shutdown()


# ── S-08: typed client/server errors, never a closed socket or traceback ──

def test_non_dict_json_body_is_typed_400(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/track", ["not", "an", "object"])
        assert status == 400
        assert json.loads(body).get("code") == "bad_json_shape"
    finally:
        httpd.shutdown()


def test_bad_content_length_header_is_typed_400_not_a_crash(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _raw(
            base, "POST", "/api/track",
            body=b'{"seed": {}}',
            headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
        )
        assert status == 400
        assert json.loads(body).get("code") == "bad_content_length"
        # the connection must still be usable afterward (not a torn socket)
        status2, _b2, _h2 = _get(base, "/api/health")
        assert status2 == 200
    finally:
        httpd.shutdown()


def test_post_without_json_content_type_is_typed_400(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _raw(
            base, "POST", "/api/track",
            body=b'{"seed": {}}', headers={"Content-Type": "text/plain"},
        )
        assert status == 400
        assert json.loads(body).get("code") == "bad_content_type"
    finally:
        httpd.shutdown()


def test_params_must_be_an_object(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/track", {
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "seed": {"points_mm": [[0, 0, 0]], "radius_mm": 3.0},
            "params": ["oops"],
        })
        assert status == 400
        assert json.loads(body).get("code") == "bad_request"
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_result_registry_is_bounded(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        service.results.max_count = 2
        for _ in range(4):
            status, _b, _h = _load_bank(base, service)
            assert status == 200
        assert len(service.results) <= 2
    finally:
        httpd.shutdown()


# ── SEC-01: loopback Host/Origin + Fetch Metadata; static ancestry ─────────

def test_wrong_host_header_rejected(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _get(base, "/api/health", host="evil.example.com")
        assert status == 403
        assert json.loads(body).get("code") == "bad_host"
    finally:
        httpd.shutdown()


def test_cross_site_origin_rejected(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _get(base, "/api/health", headers={
            "Origin": "http://evil.example.com",
        })
        assert status == 403
        assert json.loads(body).get("code") == "bad_origin"
    finally:
        httpd.shutdown()


def test_cross_site_fetch_metadata_rejected(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _get(base, "/api/health", headers={"Sec-Fetch-Site": "cross-site"})
        assert status == 403
        assert json.loads(body).get("code") == "cross_site"
    finally:
        httpd.shutdown()


def test_cross_site_navigate_document_get_allowed(tmp_path):
    """SEC-01 refinement: Chrome sends Sec-Fetch-Site: cross-site with
    Sec-Fetch-Mode: navigate / Sec-Fetch-Dest: document for a top-level page
    load initiated by an extension or another site (no Origin header on a
    GET navigation). A browser-automation tool opening the viewer URL must
    not be refused — Fetch Metadata resource isolation allows navigations."""
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _get(base, "/index.html", headers={
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        })
        assert status == 200, body
    finally:
        httpd.shutdown()


def test_cross_site_navigate_post_still_rejected(tmp_path):
    """A cross-site POST (form-POST CSRF) must still be refused even with
    Sec-Fetch-Mode: navigate — the GET/HEAD-only carve-out excludes it."""
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/margin", {}, headers={
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        })
        assert status == 403
        assert json.loads(body).get("code") == "cross_site"
    finally:
        httpd.shutdown()


def test_cross_site_navigate_object_dest_rejected(tmp_path):
    """Sec-Fetch-Dest: object/embed is excluded from the navigation carve-out
    (an embedding frame/plugin, not a top-level navigation the user asked for)."""
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _get(base, "/index.html", headers={
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "object",
        })
        assert status == 403
        assert json.loads(body).get("code") == "cross_site"
    finally:
        httpd.shutdown()


def test_same_origin_request_allowed(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        parts = urlsplit(base)
        status, _body, _h = _get(base, "/api/health", headers={
            "Origin": f"http://{parts.hostname}:{parts.port}",
            "Sec-Fetch-Site": "same-origin",
        })
        assert status == 200
    finally:
        httpd.shutdown()


def test_non_browser_caller_with_no_origin_is_allowed(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, _body, _h = _get(base, "/api/health")
        assert status == 200
    finally:
        httpd.shutdown()


def test_static_ancestry_refuses_symlink_escape(tmp_path):
    """SEC-01: a symlink inside the static root pointing outside it must not
    be servable — realpath must resolve the symlink before the boundary
    check, not just string-compare the un-resolved path."""
    viewer_root = tmp_path / "viewer_root"
    viewer_root.mkdir()
    (viewer_root / "index.html").write_text("<html>ok</html>")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")
    try:
        os.symlink(str(secret), str(viewer_root / "escape.html"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man, viewer_dir=str(viewer_root))
    try:
        status, body, _ = _get(base, "/escape.html")
        assert status == 404
        assert b"do not serve me" not in body
        # a legitimate file in the root is still servable
        status, body, _ = _get(base, "/index.html")
        assert status == 200
    finally:
        httpd.shutdown()


# ── SEC-02: cancel must name the exact active job ──────────────────────────

def test_cancel_without_jobid_does_not_cancel_a_busy_job(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    proc = None
    try:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        service._register_proc(proc, "real-job-id")
        status, body, _ = _post_json(base, "/api/cancel", {})
        assert status == 200
        j = json.loads(body)
        assert j.get("cancelled") is False
        assert proc.poll() is None  # still running — not killed
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        httpd.shutdown()


def test_cancel_with_wrong_jobid_does_not_cancel(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    proc = None
    try:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        service._register_proc(proc, "real-job-id")
        status, body, _ = _post_json(base, "/api/cancel", {"jobId": "some-other-id"})
        assert status == 200
        j = json.loads(body)
        assert j.get("cancelled") is False
        assert j.get("reason") == "job_id_mismatch"
        assert proc.poll() is None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        httpd.shutdown()


def test_cancel_with_matching_jobid_cancels(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    proc = None
    try:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        service._register_proc(proc, "real-job-id")
        status, body, _ = _post_json(base, "/api/cancel", {"jobId": "real-job-id"})
        assert status == 200
        j = json.loads(body)
        assert j.get("cancelled") is True
        deadline = time.time() + 5
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        httpd.shutdown()


def test_track_jobid_must_be_a_uuid_string(tmp_path):
    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    try:
        status, body, _ = _post_json(base, "/api/track", {
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 3.0},
            "params": {}, "jobId": "not-a-uuid",
        })
        assert status == 400
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_server_close_frees_result_registry(tmp_path):
    from tractlab.results import ResultUnavailable

    man, _tck = _tiny_case(tmp_path)
    httpd, service, base = _serve(man)
    _s, _b, headers = _load_bank(base, service)
    result_id = _hdr(headers, "X-resultId")
    assert result_id
    httpd.shutdown()
    assert service.analytic_sources._latest is None
    with pytest.raises(ResultUnavailable):
        service.results.get(result_id)


# ── 2026-09-07 audit S-1/S-2/S-4/S-14: every served string honest about the
#    active derivation (CLAUDE.md: never an uncorrected warning on a corrected case)

def _lineage_case(tmp_path, *, kind, signed=False):
    """_tiny_case migrated to `kind`, carrying the ADR-0004 trap: an inherited
    acquisition.geom_floor_mm = 3.0 that predates any correction. `signed` adds
    a human-approved delta_qc block with a 0.6 mm median (the demo case's shape)."""
    manifest, _tck = _tiny_case(
        tmp_path, derivations={"derivations": {"d1": {"kind": kind}}, "active_derivation": "d1"},
    )
    man = json.loads(manifest.read_text())
    man["acquisition"] = {"geom_floor_mm": 3.0, "geom_floor_note": "recorded before correction"}
    if signed:
        man["delta_qc"] = {
            "auto": {"median_mm": 0.6033, "p95_mm": 3.87, "max_mm": 17.5, "n_vox": 351940},
            "approved_by": "erion", "date": "2026-08-18", "sheet_sha": "0" * 64,
            "sheet_path": "qc/delta_qc.png", "derivation": "d1",
        }
    manifest.write_text(json.dumps(man))
    return manifest


# "3 mm" as a bare figure — (?<![\d.]) so the 3 inside "0.6033" or "13 mm" never matches.
_UNCORRECTED_PATTERNS = ("no reverse-PE", r"(?<![\d.])~?3 mm", "True CST")


def _served_strings(base, service):
    """The human-readable surfaces the audit named: /api/health, /api/banks,
    the T1 header label and the connectotomy honesty copy."""
    from tractlab.connectotomy import connectotomy_note
    from tractlab.serve import t1_header_label

    _, health, _ = _get(base, "/api/health")
    _, banks, _ = _get(base, "/api/banks")
    out = {
        "health": json.loads(health)["uncertainty"],
        "banks": json.loads(banks)["disclaimer"],
        "t1": t1_header_label(service.manifest),
    }
    out["connectotomy"] = (
        connectotomy_note(service.geom_floor_mm)
        if service.geom_floor_mm is not None else service._geom_floor_error
    )
    return out


def test_rpe_pair_signed_case_serves_no_uncorrected_warning(tmp_path):
    manifest = _lineage_case(tmp_path, kind="rpe_pair", signed=True)
    httpd, service, base = _serve(manifest)
    try:
        served = _served_strings(base, service)
        for surface, text in served.items():
            for pat in _UNCORRECTED_PATTERNS:
                assert re.search(pat, text) is None, f"{surface} still says {pat!r}: {text}"
            assert "reverse-PE corrected" in text or surface == "connectotomy", (surface, text)
        assert "median shift 0.6 mm (signed delta QC)" in served["health"]
        assert "candidate identity, not verified anatomy" in served["banks"].lower()
        # S-14: the inherited acquisition floor (3.0) is NOT served; the signed median is.
        assert service.geom_floor_mm == pytest.approx(0.6033)
        assert "~0.6 mm" in served["connectotomy"]
        # /api/derivation and /api/health can never disagree
        _, deriv, _ = _get(base, "/api/derivation")
        assert json.loads(deriv)["floor_label"] in served["health"]
    finally:
        httpd.shutdown()


def test_rpe_pair_unsigned_case_refuses_the_inherited_floor_and_says_unsigned(tmp_path):
    manifest = _lineage_case(tmp_path, kind="rpe_pair", signed=False)
    httpd, service, base = _serve(manifest)
    try:
        served = _served_strings(base, service)
        assert service.geom_floor_mm is None
        assert "unsigned" in served["connectotomy"]
        for surface in ("health", "banks", "t1"):
            assert "no reverse-PE" not in served[surface]
            assert re.search(r"(?<![\d.])~?3 mm", served[surface]) is None
            assert "delta QC unsigned" in served[surface], (surface, served[surface])
    finally:
        httpd.shutdown()


def test_uncorrected_case_still_serves_the_uncorrected_label_everywhere(tmp_path):
    manifest = _lineage_case(tmp_path, kind="uncorrected")
    httpd, service, base = _serve(manifest)
    try:
        served = _served_strings(base, service)
        for surface in ("health", "banks", "t1"):
            assert "no reverse-PE — ~3 mm geometric floor" in served[surface], (surface, served[surface])
        assert service.geom_floor_mm == 3.0
        assert "~3 mm" in served["connectotomy"]
        assert "True CST" not in served["banks"]
    finally:
        httpd.shutdown()


def test_served_bank_labels_carry_no_anatomical_truth_claim(tmp_path):
    """S-13: the committed case manifests must not label a recipe-selected bank
    as 'True' anatomy; role=true_cst may stay internal."""
    import pathlib
    for rel in ("cases/demo-leipzig-sub-010005/manifest.json",
                "cases/local-case/manifest.json"):
        p = pathlib.Path(__file__).resolve().parents[1] / rel
        if not p.exists():
            continue
        man = json.loads(p.read_text())
        for key, spec in man.get("inputs", {}).items():
            if isinstance(spec, dict) and spec.get("label"):
                assert "True CST" not in spec["label"], (rel, key, spec["label"])
