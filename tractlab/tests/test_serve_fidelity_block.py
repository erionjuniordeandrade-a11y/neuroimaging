"""E2/T8 — bank responses carry a per-display-streamline fidelity summary block.

Block (when status=ok): n_displayed × float32 frac_ge_R + n_displayed ×
float32 p5_ratio + n_displayed × uint8 flags (9 bytes/streamline).
Absent sidecar → status=absent and NO X-lowSupportCount (untested, never 0).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.fidelity import (
    FLAG_CROSSES_CAVITY,
    FLAG_CROSSES_LESION,
    R_GRID,
    build_sidecar,
    save_sidecar,
)

from tractlab.fidelity import DEFAULT_MIN_FRAC, DEFAULT_R, FidelityRefusal
from tractlab.serve import _pack_display_streamlines, serve

VIEWER = str(Path(__file__).resolve().parents[1] / "viewer")
D0 = os.path.expanduser("~/tractlab/cases/demo-leipzig-sub-010005")
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
Y00 = 0.28209479177387814


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _post(url, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _tiny_case(tmp_path, *, with_sidecar: bool, n_lines: int = 5):
    """Serveable tmp case: b0/mask/fod + one bank. Optional fidelity sidecar."""
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
        for i in range(n_lines)
    ]
    # Mark two streamlines as cavity / lesion crossings in the sidecar only
    # (flags come from the sidecar, not from live traversal at serve time).
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
    if with_sidecar:
        data = build_sidecar(
            tck, tmp_path / "fod.nii.gz", tmp_path / "peak.nii.gz",
            provenance=prov,
        )
        flags = np.asarray(data["flags"], dtype=np.uint8)
        flags[0] |= FLAG_CROSSES_LESION
        flags[1] |= FLAG_CROSSES_CAVITY
        data["flags"] = flags
        # Force one low-support row at DEFAULT_R so counts are non-zero.
        r_idx = int(np.where(np.isclose(R_GRID, DEFAULT_R))[0][0])
        data["frac_ge"][2, r_idx] = 0.0
        fid_dir = tmp_path / "fidelity"
        fid_dir.mkdir()
        (fid_dir / "fod_peak.nii.gz").write_bytes((tmp_path / "peak.nii.gz").read_bytes())
        save_sidecar(fid_dir / "bank_test.fidelity.npz", data)
    return tmp_path / "manifest.json"


def _serve(manifest):
    httpd, service = serve(str(manifest), VIEWER, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, service, base


def _load(base, service, bid="bank_test"):
    return _post(base + "/api/bank/load", {
        "gridId": service.grid_id,
        "volumeId": service.volume_id,
        "bankId": bid,
    })


def _hdr(headers, name):
    # urllib lower-cases; accept either
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def test_absent_sidecar_is_untested_never_zero(tmp_path):
    """No sidecar → X-fidelityStatus=absent and NO low-support count header."""
    man = _tiny_case(tmp_path, with_sidecar=False)
    httpd, service, base = _serve(man)
    try:
        status, body, headers = _load(base, service)
        assert status == 200
        assert _hdr(headers, "X-fidelityStatus") == "absent"
        assert _hdr(headers, "X-lowSupportCount") is None
        assert _hdr(headers, "X-fidelityOffset") in (None, "")
        n = int(_hdr(headers, "X-nDisplayed") or 0)
        assert n > 0
        # No 9-byte fidelity tail is claimed; lineage transport may still follow.
        assert len(body) > 0
    finally:
        httpd.shutdown()


def test_bank_response_exposes_exact_display_to_source_ordinals(tmp_path):
    """Every displayed line carries its zero-based source row in the bank."""
    man = _tiny_case(tmp_path, with_sidecar=False)
    httpd, service, base = _serve(man)
    try:
        status, body, headers = _load(base, service)
        assert status == 200
        n = int(_hdr(headers, "X-nDisplayed"))
        k = int(_hdr(headers, "X-pointsPerLine"))
        off = int(_hdr(headers, "X-sourceOrdinalOffset"))
        count = int(_hdr(headers, "X-sourceOrdinalCount"))
        assert _hdr(headers, "X-sourceOrdinalEncoding") == "uint32le"
        assert _hdr(headers, "X-sourceOrdinalBase") == "0"
        assert _hdr(headers, "X-sourcePopulation") == "bank:bank_test"
        assert count == n == 5
        assert off == n * k * 3 * 4
        ordinals = np.frombuffer(body[off : off + n * 4], dtype="<u4")
        np.testing.assert_array_equal(ordinals, np.arange(n, dtype=np.uint32))
        assert len(body) == off + n * 4
    finally:
        httpd.shutdown()


def test_packed_display_lineage_tracks_rows_dropped_by_minlength_guard():
    """Post-sample length drops must remove the same source-ordinal rows."""
    long_a = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    short = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    long_b = np.array([[0, 0, 0], [0, 12, 0]], dtype=float)

    body, headers, source_ordinals = _pack_display_streamlines(
        [long_a, short, long_b],
        np.array([17, 5, 91], dtype=np.int64),
        8,
        "test-space",
        minlength_mm=5.0,
    )

    assert headers["lineCount"] == 2
    assert headers["nDropSourceShort"] == 1
    np.testing.assert_array_equal(source_ordinals, np.array([17, 91]))
    assert len(body) == 2 * int(headers["pointsPerLine"]) * 3 * 4


def test_sidecar_appends_block_and_counts_match(tmp_path):
    """ok block length == n_displayed × 9; header counts match the arrays."""
    man = _tiny_case(tmp_path, with_sidecar=True)
    httpd, service, base = _serve(man)
    try:
        status, body, headers = _load(base, service)
        assert status == 200
        assert _hdr(headers, "X-fidelityStatus") == "ok"
        off = int(_hdr(headers, "X-fidelityOffset"))
        n = int(_hdr(headers, "X-nDisplayed"))
        assert n == 5
        block = body[off:]
        assert len(block) == n * 9
        frac = np.frombuffer(block[: n * 4], dtype="<f4")
        p5 = np.frombuffer(block[n * 4: n * 8], dtype="<f4")
        flags = np.frombuffer(block[n * 8:], dtype=np.uint8)
        assert frac.shape == (n,) and p5.shape == (n,) and flags.shape == (n,)
        min_frac = float(_hdr(headers, "X-fidelityMinFrac"))
        assert abs(min_frac - DEFAULT_MIN_FRAC) < 1e-6
        assert abs(float(_hdr(headers, "X-fidelityR")) - DEFAULT_R) < 1e-6
        # No sheet → pilot, and the header SAYS pilot.
        assert _hdr(headers, "X-fidelityOpSource") == "pilot"
        assert _hdr(headers, "X-fidelityApprovedBy") == ""
        low = int(((frac < min_frac) | ((flags & FLAG_CROSSES_CAVITY) != 0)).sum())
        n_les = int(((flags & FLAG_CROSSES_LESION) != 0).sum())
        n_cav = int(((flags & FLAG_CROSSES_CAVITY) != 0).sum())
        assert int(_hdr(headers, "X-lowSupportCount")) == low
        assert int(_hdr(headers, "X-crossesLesionCount")) == n_les
        assert int(_hdr(headers, "X-crossesCavityCount")) == n_cav
        assert n_les == 1 and n_cav == 1
        assert low >= 2  # forced low-ratio row + cavity flag
    finally:
        httpd.shutdown()


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(D0, "fidelity")),
    reason="D0 sidecars absent — owner-gated prep_fidelity --apply",
)
def test_d0_bank_reports_fidelity_status():
    man = os.path.join(D0, "manifest.json")
    if not os.path.isfile(man):
        pytest.skip("D0 manifest absent")
    httpd, service, base = _serve(man)
    try:
        if not service.banks:
            pytest.skip("D0 has no banks")
        bid = next(iter(service.banks))
        status, body, headers = _load(base, service, bid)
        assert status == 200
        st = _hdr(headers, "X-fidelityStatus")
        assert st in ("ok", "absent", "provenance-incomplete")
        if st == "ok":
            n = int(_hdr(headers, "X-nDisplayed"))
            off = int(_hdr(headers, "X-fidelityOffset"))
            assert len(body[off:]) == n * 9
        else:
            assert _hdr(headers, "X-lowSupportCount") is None
    finally:
        httpd.shutdown()


def _write_signed_sheet(case_root: Path, *, R: str, min_frac: str, sha: str,
                        who: str = "erion", date: str = "2026-09-01"):
    sheet = case_root / "docs" / "qc" / "OPERATING-POINT-fidelity.md"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(
        "# Fidelity operating point\n\n"
        f"R: {R}\nmin_frac: {min_frac}\napproved_by: {who}\ndate: {date}\n"
        f"sweep_csv_sha256: {sha}\n"
    )


def _fake_sweep(case_root: Path) -> str:
    csv = case_root / "fidelity" / "sweep.csv"
    csv.parent.mkdir(exist_ok=True)
    csv.write_bytes(b"bank_id,R,min_frac,marked_frac,n\n")
    return hashlib.sha256(csv.read_bytes()).hexdigest()


def test_signed_sheet_changes_served_operating_point(tmp_path):
    """The signature bites: headers + low-support count follow the SIGNED point,
    not the pilot defaults. R=0.45/min 0.9 differ from 0.30/0.70 on purpose."""
    man = _tiny_case(tmp_path, with_sidecar=True)
    sha = _fake_sweep(tmp_path)
    _write_signed_sheet(tmp_path, R="0.45", min_frac="0.9", sha=sha)
    httpd, service, base = _serve(man)
    try:
        assert service.operating_point.signed
        status, body, headers = _load(base, service)
        assert status == 200
        assert _hdr(headers, "X-fidelityStatus") == "ok"
        assert abs(float(_hdr(headers, "X-fidelityR")) - 0.45) < 1e-6
        assert abs(float(_hdr(headers, "X-fidelityMinFrac")) - 0.9) < 1e-6
        assert _hdr(headers, "X-fidelityOpSource") == "signed"
        assert _hdr(headers, "X-fidelityApprovedBy") == "erion"
        assert _hdr(headers, "X-fidelityOpDate") == "2026-09-01"
        # Low-support count is recomputed against the signed point on the
        # served frac column (R=0.45), not the pilot column.
        off = int(_hdr(headers, "X-fidelityOffset"))
        n = int(_hdr(headers, "X-nDisplayed"))
        block = body[off:]
        frac = np.frombuffer(block[: n * 4], dtype="<f4")
        flags = np.frombuffer(block[n * 8:], dtype=np.uint8)
        low = int(((frac < 0.9) | ((flags & FLAG_CROSSES_CAVITY) != 0)).sum())
        assert int(_hdr(headers, "X-lowSupportCount")) == low
    finally:
        httpd.shutdown()


def test_unsigned_sheet_serves_pilot_and_says_so(tmp_path):
    man = _tiny_case(tmp_path, with_sidecar=True)
    sha = _fake_sweep(tmp_path)
    _write_signed_sheet(tmp_path, R="0.45", min_frac="0.9", sha=sha, who="", date="")
    httpd, service, base = _serve(man)
    try:
        assert not service.operating_point.signed
        status, body, headers = _load(base, service)
        assert status == 200
        assert abs(float(_hdr(headers, "X-fidelityR")) - DEFAULT_R) < 1e-6
        assert _hdr(headers, "X-fidelityOpSource") == "pilot"
    finally:
        httpd.shutdown()


def test_signed_sheet_over_drifted_sweep_refuses_at_startup(tmp_path):
    """Fail closed where the owner sees it: serve() raises, no silent pilot."""
    man = _tiny_case(tmp_path, with_sidecar=True)
    _fake_sweep(tmp_path)
    _write_signed_sheet(tmp_path, R="0.45", min_frac="0.9", sha="0" * 64)
    with pytest.raises(FidelityRefusal, match="drifted"):
        serve(str(man), VIEWER, port=0)
