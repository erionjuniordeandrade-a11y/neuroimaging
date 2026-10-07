"""M6 serve.py — loopback refusal, endpoints, stale-grid, single-flight."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import urllib.request
import urllib.error

import nibabel as nib
import numpy as np
import pytest

from tractlab.serve import (
    serve,
    make_handler,
    _resolve_case_path,
    _unpack_packed_streamlines,
    _validate_filter_minlength,
)
from tractlab.grid import voxel_to_world
from tractlab import derivation as dv

MANIFEST = os.path.expanduser("~/tractlab/cases/local-case/manifest.json")
VIEWER = os.path.expanduser("~/tractlab/viewer")
needs = pytest.mark.skipif(not os.path.exists(MANIFEST), reason="manifest absent")


def test_refuses_non_loopback_bind():
    with pytest.raises(ValueError, match="non-loopback"):
        serve(MANIFEST, VIEWER, host="0.0.0.0", port=8791)


def test_manifest_paths_cannot_escape_case_root(tmp_path):
    root = str(tmp_path / "case")
    os.makedirs(root)
    assert _resolve_case_path(root, "nifti/b0.nii.gz", what="b0").startswith(root)
    with pytest.raises(ValueError, match="escapes case_root"):
        _resolve_case_path(root, "../outside.nii.gz", what="b0")
    with pytest.raises(ValueError, match="relative"):
        _resolve_case_path(root, "/absolute.nii.gz", what="b0")


def test_packed_decode_uses_advertised_variable_k():
    from tractlab.pack import pack_streamlines

    x = np.linspace(0, 100, 201)
    line = np.column_stack([x, 20.0 * np.sin(x / 2.0), np.zeros_like(x)])
    buf, header = pack_streamlines([line], k=64, space_id="test", minlength_mm=100.0)
    assert int(header["pointsPerLine"]) > 64
    decoded = _unpack_packed_streamlines(buf, header)
    assert decoded.shape == (1, int(header["pointsPerLine"]), 3)


def test_filter_minlength_is_finite_and_bounded():
    assert _validate_filter_minlength(20) == 20.0
    for bad in (0, -1, 251, float("nan"), "not-a-number"):
        with pytest.raises(ValueError, match="filter minlength"):
            _validate_filter_minlength(bad)


@pytest.fixture
def server():
    # port=0: ephemeral bind — fixed ports collided across parallel suite
    # runs and TIME_WAIT between back-to-back runs (Errno 48 flakes).
    httpd, service = serve(MANIFEST, VIEWER, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{httpd.server_address[1]}", service
    httpd.shutdown()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _post(url, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_derivation_endpoint_reports_lineage_list_and_qc_shape():
    """Route-level JSON contract using only a synthetic manifest/service."""
    manifest = {
        "case_root": "/synthetic/case",
        "active_derivation": "d1",
        "inputs": {"b0": {"path": "d1/b0.nii.gz"}},
        "derivations": {
            "d1": {"kind": "rpe_pair"},
            "d2": {
                "kind": "uncorrected",
                "inputs": {"b0": {
                    "path": "d2/b0.nii.gz",
                    "sha256_by_field": {"path": "aa" * 32},
                    "bytes_by_field": {"path": 1},
                }},
            },
        },
        "delta_qc": {
            "auto": {"median_mm": 1.73},
            "approved_by": "erion",
            "derivation": "d1",
        },
    }

    class SyntheticService:
        pass

    service = SyntheticService()
    service.manifest = manifest
    Handler = make_handler(service, ".")
    handler = object.__new__(Handler)
    payload = {}
    handler.path = "/api/derivation"
    handler._runtime_current = lambda: True
    handler._enforce_browser_origin_policy = lambda: None
    handler._json = lambda code, obj: payload.update(status=code, body=obj)

    handler.do_GET()

    assert payload["status"] == 200
    body = payload["body"]
    assert body["active"] == "d1"
    assert body["kind"] == "rpe_pair"
    assert body["delta_qc_signed"] is True
    assert body["derivations"] == [
        {
            "id": "d1",
            "kind": "rpe_pair",
            "qc_signed": {
                "t1_qc": False,
                "atlas_prior_qc": False,
                "parcellation_qc": False,
                "delta_qc": True,
            },
        },
        {
            "id": "d2",
            "kind": "uncorrected",
            "qc_signed": {
                "t1_qc": False,
                "atlas_prior_qc": False,
                "parcellation_qc": False,
                "delta_qc": False,
            },
        },
    ]


@needs
def test_health_and_volume(server):
    base, service = server
    status, body, _ = _get(base + "/api/health")
    assert status == 200
    h = json.loads(body)
    assert h["gridId"] == service.grid_id and h["volumeId"] == service.volume_id
    if service.filter_bank_path:
        assert h["filterBankCount"] == int(nib.streamlines.load(service.filter_bank_path, lazy_load=True).header["count"])
    else:
        assert h["filterBankCount"] is None

    status, buf, hdr = _get(base + "/api/volume/b0")
    assert status == 200
    assert len(buf) == 185 * 185 * 109        # u8 volume, i-fastest
    assert hdr["X-Grid-Id"] == service.grid_id
    assert hdr["X-Volume-Id"] == service.volume_id


@needs
def test_t1_and_priors_fail_closed_without_qc(server):
    """Unsigned QC → no T1 volume (if gated) and empty priors list."""
    base, service = server
    # Priors must be empty without atlas_prior_qc.approved_by
    status, body, _ = _get(base + "/api/priors")
    assert status == 200
    j = json.loads(body)
    if not service.priors:
        assert j.get("available") is False or j.get("priors") == []
    # /api/priors/unknown/mesh always 404
    status, _, _ = _get(base + "/api/priors/norm_does_not_exist/mesh")
    assert status == 404


@needs
def test_stale_grid_rejected(server):
    base, service = server
    body = {"gridId": "wrong", "volumeId": service.volume_id,
            "seed": {"points_mm": [[0, 0, 0]], "radius_mm": 3.0}, "params": {}}
    status, body_b, _ = _post(base + "/api/track", body)
    assert status == 409
    err = json.loads(body_b)
    assert err.get("code") == "stale_volume"


@needs
def test_health_carries_recipe_hash(server):
    base, service = server
    status, body, _ = _get(base + "/api/health")
    assert status == 200
    h = json.loads(body)
    assert h.get("recipeHash") == service.recipe_hash
    assert len(h["recipeHash"]) == 16


@needs
def test_cancel_when_idle(server):
    base, _ = server
    status, body, _ = _post(base + "/api/cancel", {})
    assert status == 200
    j = json.loads(body)
    assert j.get("cancelled") is False


@needs
def test_empty_seed_is_400(server):
    """SEED required — empty paint fails loud at the API boundary (not empty_seed 200)."""
    base, service = server
    body = {"gridId": service.grid_id, "volumeId": service.volume_id,
            "seed": {"points_mm": [], "radius_mm": 4.0}, "params": {}}
    status, body_b, _ = _post(base + "/api/track", body)
    assert status == 400
    err = json.loads(body_b)
    assert "seed" in err["error"].lower()


@needs
def test_presets_catalog_and_volume(server):
    base, service = server
    status, body, _ = _get(base + "/api/presets")
    assert status == 200
    data = json.loads(body)
    ids = {p["id"] for p in data["presets"]}
    assert "seed_fa_r" in ids
    assert all("path" not in p for p in data["presets"])

    status, buf, hdr = _get(base + "/api/presets/seed_fa_r/volume")
    assert status == 200
    assert len(buf) == 185 * 185 * 109
    assert hdr["X-Preset-Id"] == "seed_fa_r"
    assert int(hdr["X-N-Voxels"]) > 0


@needs
def test_unknown_preset_volume_404(server):
    base, _ = server
    status, body, _ = _get(base + "/api/presets/seed_nope/volume")
    assert status == 404


@needs
def test_fa_and_dec_underlay_endpoints(server):
    base, service = server
    status, body, _ = _get(base + "/api/health")
    h = json.loads(body)
    assert h.get("hasFa") is True
    assert h.get("hasDec") is True

    status, buf, hdr = _get(base + "/api/volume/fa")
    assert status == 200
    assert len(buf) == 185 * 185 * 109
    assert hdr.get("X-Format") == "gray"

    status, buf, hdr = _get(base + "/api/volume/dec")
    assert status == 200
    assert len(buf) == 185 * 185 * 109 * 3
    assert hdr.get("X-Format") == "rgb-planes"
    assert hdr.get("X-Channels") == "3"


@needs
def test_track_roundtrip_binary(server):
    base, service = server
    cw = voxel_to_world(service.grid, np.array([92, 92, 54]))  # a WM point
    # use a validated lateral WM seed instead so we reliably get streamlines
    fa_r = voxel_to_world(service.grid, np.array([40, 92, 54]))
    body = {"gridId": service.grid_id, "volumeId": service.volume_id,
            "seed": {"points_mm": [list(map(float, fa_r))], "radius_mm": 6.0},
            "params": {"angle": 45, "minlength": 20}}
    status, buf, hdr = _post(base + "/api/track", body)
    assert status == 200
    assert hdr["X-outcome"] in ("ok", "empty_seed")
    if hdr["X-outcome"] == "ok":
        k = int(hdr["X-pointsPerLine"]); lc = int(hdr["X-lineCount"])
        tract_bytes = lc * k * 3 * 4           # float32 (L,K,3)
        source_off = int(hdr["X-sourceOrdinalOffset"])
        assert int(hdr["X-sourceOrdinalCount"]) == lc
        assert hdr["X-sourceOrdinalEncoding"] == "uint32le"
        assert hdr["X-sourceOrdinalBase"] == "0"
        assert hdr["X-sourcePopulation"] == "live-track"
        dist_off = hdr.get("X-distanceOffset") or ""
        if dist_off:                            # lesion present → distance block appended
            assert int(dist_off) == tract_bytes
            assert source_off == tract_bytes + lc * k * 4
            dist = np.frombuffer(buf, dtype="<f4", count=lc * k, offset=tract_bytes)
            assert dist.shape[0] == lc * k
            assert np.isfinite(dist).all() and (dist >= 0).all()
        else:
            assert source_off == tract_bytes
        assert len(buf) == source_off + lc * 4
        source_ordinals = np.frombuffer(buf, dtype="<u4", count=lc, offset=source_off)
        assert source_ordinals.shape[0] == lc
        assert np.unique(source_ordinals).shape[0] == lc
        assert int(source_ordinals.max()) < int(hdr["X-nFile"])
        arr = np.frombuffer(buf[:tract_bytes], dtype="<f4")
        assert np.isfinite(arr).all()


@needs
def test_single_flight_returns_409_when_busy(server):
    """Hold the service lock, a concurrent track must get 409 busy."""
    base, service = server
    service._lock.acquire()
    try:
        body = {"gridId": service.grid_id, "volumeId": service.volume_id,
                "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 3.0}, "params": {}}
        status, b, _ = _post(base + "/api/track", body)
        assert status == 409
        assert b"already running" in b
    finally:
        service._lock.release()


@needs
def test_connectotomy_payload_on_lesion_case(server):
    """3T case has a lesion — C1 report is present, assignment-free."""
    base, service = server
    assert service.ensure_connectotomy() is not None
    status, body, _ = _get(base + "/api/connectotomy")
    assert status == 200
    data = json.loads(body)
    assert data["cavity"] == "lesion"
    assert "floor_mm" in data
    assert data["banks"]
    blob = json.dumps(data).lower()
    for forbidden in ("assigned", "assignment", "schaefer", "yeo", "network",
                      "margin", "navigation", "at risk"):
        assert forbidden not in blob
    row = data["banks"][0]
    assert set(row) == {"id", "n_cut", "n_bank", "fraction"}
    assert row["n_bank"] >= row["n_cut"] >= 0


@needs
def test_connectotomy_404_without_lesion(server):
    """Absence is honest — no zero-cuts body when the cached report is missing."""
    base, service = server
    saved = service.connectotomy_report
    saved_ready = service._connectotomy_ready
    saved_idx = service._connectotomy_cut_idx
    service.connectotomy_report = None
    service._connectotomy_ready = True
    try:
        status, body, _ = _get(base + "/api/connectotomy")
        assert status == 404
        assert b"n_cut" not in body
        assert b"banks" not in body
    finally:
        service.connectotomy_report = saved
        service._connectotomy_ready = saved_ready
        service._connectotomy_cut_idx = saved_idx


@needs
def test_connectotomy_cut_subset_known_bank(server):
    base, service = server
    if service.ensure_connectotomy() is None:
        pytest.skip("no lesion cavity on this case")
    bid = service.connectotomy_report["banks"][0]["id"]
    status, buf, hdr = _get(base + f"/api/connectotomy/{bid}/cut")
    assert status == 200
    assert hdr.get("X-bankId") == bid
    assert hdr.get("X-connectotomy") == "cut-subset"
    assert hdr.get("X-cavity") == "lesion"
    c1_copy = " ".join(
        str(hdr.get(k) or "")
        for k in ("X-honesty", "X-connectotomy", "X-cavity", "X-label", "X-role")
    )
    for word in ("margin", "at risk", "ASSIGNED", "Schaefer", "Yeo"):
        assert word.lower() not in c1_copy.lower()


@needs
def test_connectotomy_cut_subset_carries_bank_source_hash(server):
    """A cut subset is that bank's streamlines, so it must state the bank's
    source hash exactly as /api/bank/load does. A saved review compares this
    on reopen; without it a changed bank would restore silently."""
    base, service = server
    if service.ensure_connectotomy() is None:
        pytest.skip("no lesion cavity on this case")
    bid = service.connectotomy_report["banks"][0]["id"]
    _, _, cut = _get(base + f"/api/connectotomy/{bid}/cut")
    cut_hash = cut.get("X-bankSourceHash")
    assert cut_hash, "cut subset must return X-bankSourceHash"
    assert re.fullmatch(r"[0-9a-f]{64}", cut_hash), cut_hash
    assert cut_hash == service.bank_source_hash(bid)
    status, _, load = _post(
        base + "/api/bank/load",
        {"gridId": service.grid_id, "volumeId": service.volume_id, "bankId": bid},
    )
    assert status == 200
    assert cut.get("X-bankSourceHash") == load.get("X-bankSourceHash")


@needs
def test_connectotomy_cut_unknown_bank_404(server):
    base, service = server
    if service.ensure_connectotomy() is None:
        pytest.skip("no lesion cavity on this case")
    status, _, _ = _get(base + "/api/connectotomy/bank_does_not_exist/cut")
    assert status == 404


@needs
def test_track_rejects_cavity_role(server):
    base, service = server
    body = {
        "gridId": service.grid_id,
        "volumeId": service.volume_id,
        "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 4.0, "role": "cavity"},
        "params": {},
    }
    status, body_b, _ = _post(base + "/api/track", body)
    assert status == 400
    err = json.loads(body_b)
    assert "cavity" in err["error"].lower()


def test_header_ascii_maps_unicode_separators():
    """Engine strings must not become 'prebuilt ? ACT ? bilat' in HTTP headers."""
    from tractlab.serve import header_ascii

    raw = "BANK | prebuilt · ACT iFOD2 10M → bilat frontal"
    out = header_ascii(raw)
    assert "?" not in out
    assert "->" in out
    assert "prebuilt" in out
    assert "bilat" in out


@needs
def test_bank_load_sends_body_outside_single_flight_lock():
    """A stalled or vanished reader must never hold the single-flight lock
    while the response body drains (observed 60 s lock wedge, 2026-08-17)."""
    httpd, service = serve(MANIFEST, VIEWER, port=0)
    port = httpd.server_address[1]
    H = httpd.RequestHandlerClass
    sends = []
    orig = H._send

    def spy(self, code, body=b"", ctype="application/octet-stream", extra=None):
        sends.append((code, len(body), service._lock.locked()))
        return orig(self, code, body, ctype, extra)

    H._send = spy
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, _, _ = _post(f"http://127.0.0.1:{port}/api/bank/load", {
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "bankId": next(iter(service.banks)),
        })
        assert status == 200
        big = [s for s in sends if s[0] == 200 and s[1] > 1000]
        assert big, "no large 200 send recorded"
        assert all(locked is False for _, _, locked in big), (
            "response body was sent while holding the single-flight lock"
        )
    finally:
        H._send = orig
        httpd.shutdown()


@needs
def test_stalled_reader_does_not_wedge_other_bank_loads():
    """Client that stops reading its bank response must not 409 everyone else."""
    import socket
    import time

    httpd, service = serve(MANIFEST, VIEWER, port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    stalled = socket.socket()
    try:
        big_bid = max(
            service.banks,
            key=lambda b: service.banks[b].n_streamlines or 0,
        )
        payload = json.dumps({
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "bankId": big_bid,
        }).encode()
        stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        stalled.connect(("127.0.0.1", port))
        stalled.sendall(
            b"POST /api/bank/load HTTP/1.1\r\nHost: t\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        # Never read the response. Wait out the compute phase (lock held
        # legitimately), then require the lock to free while the body is
        # still undrained in sendall.
        deadline = time.time() + 30.0
        while time.time() < deadline and not service._lock.locked():
            time.sleep(0.02)  # request reaches compute
        while time.time() < deadline and service._lock.locked():
            time.sleep(0.05)
        assert not service._lock.locked(), (
            "single-flight lock still held while a stalled reader drains"
        )
        status, _, _ = _post(f"http://127.0.0.1:{port}/api/bank/load", {
            "gridId": service.grid_id, "volumeId": service.volume_id,
            "bankId": next(iter(service.banks)),
        })
        assert status == 200, (
            "stalled reader wedged the single-flight lock: got 409"
        )
    finally:
        stalled.close()
        httpd.shutdown()


@needs
def test_concurrent_cst_bank_loads_both_succeed():
    """Multi-select CST-L + CST-R must both 200 — 409 eats the second chip."""
    import concurrent.futures

    httpd, service = serve(MANIFEST, VIEWER, port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        ids = [b for b in ("bank_cst_l", "bank_cst_r") if b in service.banks]
        assert ids == ["bank_cst_l", "bank_cst_r"]

        def load(bid):
            return bid, _post(f"http://127.0.0.1:{port}/api/bank/load", {
                "gridId": service.grid_id, "volumeId": service.volume_id,
                "bankId": bid,
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(load, ids))
        statuses = {bid: status for bid, (status, _body, _hdr) in results}
        assert statuses == {"bank_cst_l": 200, "bank_cst_r": 200}, statuses
        for bid, (status, _body, hdr) in results:
            assert hdr.get("X-outcome") == "ok", (bid, hdr.get("X-outcome"))
            assert int(hdr.get("X-lineCount") or 0) > 0, bid
    finally:
        httpd.shutdown()


# --- Task 15: connectotomy ready-race + honest invalid-cavity error ---------

def _reset_connectotomy(service):
    service._connectotomy_ready = False
    service.connectotomy_report = None
    service._connectotomy_cut_idx = {}
    if hasattr(service, "_connectotomy_error"):
        service._connectotomy_error = None


@needs
def test_concurrent_first_connectotomy_calls_both_get_report(server, monkeypatch):
    import time

    import tractlab.serve as sv

    _base, service = server
    _reset_connectotomy(service)

    def slow_compute(banks, cavity, grid, *, floor_mm):
        time.sleep(0.4)
        return {"synthetic": True, "floor_mm": floor_mm}, {}

    monkeypatch.setattr(sv, "compute_connectotomy", slow_compute)
    start = threading.Barrier(2)
    results = []

    def call():
        start.wait()
        results.append(service.ensure_connectotomy())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[0] is not None and results[1] is not None, results
    _reset_connectotomy(service)


@needs
def test_invalid_cavity_is_422_not_404(server, monkeypatch):
    import tractlab.serve as sv

    base, service = server
    _reset_connectotomy(service)

    def broken_load(path, grid):
        raise ValueError("lesion grid does not match tracking grid")

    monkeypatch.setattr(sv, "load_lesion_bool", broken_load)
    status, body, _ = _get(base + "/api/connectotomy")
    assert status == 422, (status, body[:200])
    payload = json.loads(body)
    assert payload["error"] == "connectotomy-invalid"
    assert "grid" in payload["reason"]

    # the cut endpoint must surface the same invalid state, not 404-absence
    status2, body2, _ = _get(base + "/api/connectotomy/bank_cst_r/cut")
    assert status2 == 422, (status2, body2[:200])
    _reset_connectotomy(service)


@needs
def test_absent_lesion_is_still_404(server, monkeypatch):
    _base, service = server
    _reset_connectotomy(service)
    monkeypatch.setattr(service, "_lesion_path", None)
    assert service.ensure_connectotomy() is None
    assert getattr(service, "_connectotomy_error", None) is None
    _reset_connectotomy(service)


@needs
def test_floorless_lesioned_case_names_refusal_everywhere(server, monkeypatch):
    """Grok+Sol batch-3: a floorless case must 422 the connectotomy AND name
    the clearance refusal — blank headers would equal a legitimate null."""
    base, service = server
    _reset_connectotomy(service)
    monkeypatch.setattr(service, "geom_floor_mm", None)
    monkeypatch.setattr(service, "_geom_floor_error",
                        "case manifest has no acquisition.geom_floor_mm")
    status, body, _ = _get(base + "/api/connectotomy")
    assert status == 422
    assert json.loads(body)["reason"].startswith("case manifest has no")
    from tractlab.serve import _clearance_headers
    hdr = _clearance_headers(None, {}, service._geom_floor_error)
    assert hdr["clearanceFloorMm"] == ""
    assert "geom_floor_mm" in hdr["clearanceRefusal"]
    _reset_connectotomy(service)


def test_geom_floor_overflow_int_refuses_not_crashes():
    from tractlab.clearance import geom_floor_mm as floor_of
    import pytest as _pytest

    with _pytest.raises(ValueError, match="geom_floor_mm"):
        floor_of({"acquisition": {"geom_floor_mm": 10 ** 400}})


@needs
def test_declared_but_missing_lesion_is_invalid_not_absent(server, monkeypatch):
    base, service = server
    _reset_connectotomy(service)
    monkeypatch.setattr(service, "_lesion_missing_error",
                        "manifest declares a lesion but the file is missing (x)")
    status, body, _ = _get(base + "/api/connectotomy")
    assert status == 422
    assert "missing" in json.loads(body)["reason"]
    _reset_connectotomy(service)


@needs
def test_non_cavity_valueerror_propagates_never_422(server, monkeypatch):
    """Sol/Grok batch-4: only typed CavityInvalid may become the client 422;
    an algorithmic ValueError is a server bug and must raise loudly."""
    import tractlab.serve as sv

    _base, service = server
    _reset_connectotomy(service)

    def buggy_compute(banks, cavity, grid, *, floor_mm):
        raise ValueError("algorithmic bug, not a cavity fault")

    monkeypatch.setattr(sv, "compute_connectotomy", buggy_compute)
    with pytest.raises(ValueError, match="algorithmic bug"):
        service.ensure_connectotomy()
    _reset_connectotomy(service)


# --- Task 7: /api/derivation — HUD floor label -------------------------------

def _unmigrated(manifest: dict) -> dict:
    """Strip lineage keys so migrate_manifest can run after S2 landed on the fixture."""
    out = copy.deepcopy(manifest)
    out.pop("derivations", None)
    out.pop("active_derivation", None)
    for block in ("t1_qc", "atlas_prior_qc", "parcellation_qc", "delta_qc"):
        qc = out.get(block)
        if isinstance(qc, dict):
            qc.pop("derivation", None)
    return out


@needs
def test_derivation_endpoint_unmigrated_manifest_reports_conservative_default(server, monkeypatch):
    base, service = server
    monkeypatch.setattr(service, "manifest", _unmigrated(service.manifest))
    status, body, _ = _get(base + "/api/derivation")
    assert status == 200
    j = json.loads(body)
    assert j["active"] is None
    assert j["kind"] is None
    assert j["delta_qc_signed"] is False
    assert j["floor_label"] == "no reverse-PE — ~3 mm geometric floor"


@needs
def test_derivation_endpoint_reports_unsigned_corrected(server, monkeypatch):
    base, service = server
    migrated = dv.migrate_manifest(_unmigrated(service.manifest), dv.KIND_RPE_PAIR)
    monkeypatch.setattr(service, "manifest", migrated)

    status, body, _ = _get(base + "/api/derivation")
    assert status == 200
    j = json.loads(body)
    assert j["kind"] == "rpe_pair"
    assert j["active"] == "d1"
    assert j["delta_qc_signed"] is False
    assert "unquantified" in j["floor_label"]
    assert j["floor_label"] == (
        "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
    )


@needs
def test_derivation_endpoint_reports_uncorrected_kind(server, monkeypatch):
    base, service = server
    migrated = dv.migrate_manifest(_unmigrated(service.manifest), dv.KIND_UNCORRECTED)
    monkeypatch.setattr(service, "manifest", migrated)

    status, body, _ = _get(base + "/api/derivation")
    assert status == 200
    j = json.loads(body)
    assert j["kind"] == "uncorrected"
    assert j["delta_qc_signed"] is False
    assert j["floor_label"] == "no reverse-PE — ~3 mm geometric floor"


@needs
def test_derivation_endpoint_reports_signed_corrected_with_median(server, monkeypatch):
    base, service = server
    migrated = dv.migrate_manifest(_unmigrated(service.manifest), dv.KIND_RPE_PAIR)
    migrated["delta_qc"] = {
        "auto": {"median_mm": 1.73, "p95_mm": 2.9, "max_mm": 4.1, "n_vox": 1000},
        "approved_by": "erion",
        "date": "2026-08-17",
        "sheet_sha": "abc123",
        "sheet_path": "qc/delta_qc.png",
        "derivation": "d1",
    }
    monkeypatch.setattr(service, "manifest", migrated)

    status, body, _ = _get(base + "/api/derivation")
    assert status == 200
    j = json.loads(body)
    assert j["kind"] == "rpe_pair"
    assert j["delta_qc_signed"] is True
    assert j["floor_label"] == "reverse-PE corrected — median shift 1.7 mm (signed delta QC)"


@needs
def test_derivation_endpoint_nan_median_reports_unsigned_not_contradictory(server, monkeypatch):
    """approved_by set but median_mm is NaN: delta_qc_signed and floor_label
    must agree that this is unsigned (Task 7 fix round 1)."""
    base, service = server
    migrated = dv.migrate_manifest(_unmigrated(service.manifest), dv.KIND_RPE_PAIR)
    migrated["delta_qc"] = {
        "auto": {"median_mm": float("nan"), "p95_mm": float("nan"), "max_mm": float("nan"), "n_vox": 0},
        "approved_by": "erion",
        "date": "2026-08-17",
        "sheet_sha": "abc123",
        "sheet_path": "qc/delta_qc.png",
        "derivation": "d1",
    }
    monkeypatch.setattr(service, "manifest", migrated)

    status, body, _ = _get(base + "/api/derivation")
    assert status == 200
    j = json.loads(body)
    assert j["delta_qc_signed"] is False
    assert j["floor_label"] == (
        "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
    )


# --- T1 QC fail-closed gate (plan 002) --------------------------------------

_AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _tiny_case(tmp_path, *, t1_qc=None, active_derivation=None):
    """Serveable tmp case: b0/mask/fod + manifest (fidelity-block pattern)."""
    shape = (24, 24, 24)
    mask = np.zeros(shape, np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    b0 = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    fod = np.zeros(shape + (45,), dtype=np.float32)
    fod[..., 0] = 2.0
    for name, data in (("mask", mask), ("b0", b0), ("fod", fod)):
        image = nib.Nifti1Image(data, _AFF)
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, str(tmp_path / f"{name}.nii.gz"))
    man = {
        "case_id": "tmp-t1-gate",
        "case_root": str(tmp_path),
        "inputs": {
            "fod": {"path": "fod.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "b0": {"path": "b0.nii.gz"},
        },
    }
    if t1_qc is not None:
        man["t1_qc"] = t1_qc
    if active_derivation is not None:
        man["active_derivation"] = active_derivation
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    return path


def _serve_tmp(manifest):
    httpd, service = serve(str(manifest), VIEWER, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, service


def test_approved_by_alone_no_longer_approves_t1(tmp_path):
    """approved_by alone no longer approves T1."""
    man = _tiny_case(tmp_path, t1_qc={"approved_by": "erion"})
    httpd, service = _serve_tmp(man)
    try:
        assert service.t1_qc_approved is False
    finally:
        httpd.shutdown()


def test_t1_qc_gate_refuses_stale_derivation_scope(tmp_path):
    signed = {
        "approved_by": "erion",
        "date": "2026-08-09",
        "sheet_sha": "abc",
        "derivation": "d1",
    }
    stale = tmp_path / "stale"
    stale.mkdir()
    httpd, service = _serve_tmp(_tiny_case(stale, t1_qc=signed, active_derivation="d2"))
    try:
        assert service.t1_qc_approved is False
    finally:
        httpd.shutdown()

    ok = tmp_path / "ok"
    ok.mkdir()
    httpd, service = _serve_tmp(_tiny_case(ok, t1_qc=signed, active_derivation="d1"))
    try:
        assert service.t1_qc_approved is True
    finally:
        httpd.shutdown()
