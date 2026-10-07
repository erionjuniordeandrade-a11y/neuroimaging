"""W1 delta tests: QC scope, layout isolation, and fidelity lineage selection."""
from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tractlab import atlas_prep, check_registration, delta_qc, fidelity_sweep
from tractlab import derivation as dv
from tractlab.fidelity import (
    DEFAULT_MIN_FRAC,
    DEFAULT_R,
    OperatingPoint,
    R_GRID,
    SIDECAR_SCHEMA,
    read_operating_point,
    save_sidecar,
)
import tractlab.serve as serve_mod
from tractlab.serve import TrackService, _fidelity_payload, _derivation_uncertainty
from tractlab.serve import make_handler


def _manifest(root: Path) -> dict:
    return {
        "case_id": "synthetic-w1",
        "case_root": str(root),
        "inputs": {"b0": {"path": "d1/b0.nii.gz"}},
    }


def _meta(path: str) -> dict:
    return {
        "path": path,
        "sha256_by_field": {"path": "aa" * 32},
        "bytes_by_field": {"path": 1},
    }


def _two_derivations(root: Path) -> dict:
    man = dv.migrate_manifest(_manifest(root), dv.KIND_RPE_PAIR)
    man["derivations"]["d2"] = {
        "kind": dv.KIND_UNCORRECTED,
        "inputs": {"b0": _meta("d2/b0.nii.gz")},
    }
    return man


def test_multiderivation_refuses_unscoped_qc(tmp_path):
    man = _two_derivations(tmp_path)
    man["atlas_prior_qc"] = {"approved_by": "owner"}
    with pytest.raises(dv.DerivationError, match="explicit.*derivation"):
        dv.validate(man)

    man["atlas_prior_qc"]["derivation"] = "d1"
    man["active_derivation"] = "d2"
    assert dv.qc_block_for(man, "atlas_prior_qc") is None


def test_derivation_writers_preserve_prior_lineage(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d1"
    (root / "manifest.json").write_text(json.dumps(man))
    stats = delta_qc.DeltaStats(1.0, 2.0, 3.0, 8, np.ones((2, 2, 2)))

    d1_block = delta_qc.write_sheet(root, stats)
    d1_sheet = root / "qc" / "delta_qc.png"
    assert d1_sheet.is_file()
    d1_bytes = d1_sheet.read_bytes()
    assert d1_block["sheet_path"] == "qc/delta_qc.png"

    man["active_derivation"] = "d2"
    (root / "manifest.json").write_text(json.dumps(man))
    d2_block = delta_qc.write_sheet(root, stats)
    assert d2_block["sheet_path"] == "derivations/d2/qc/delta_qc.png"
    assert (root / d2_block["sheet_path"]).is_file()
    assert d1_sheet.read_bytes() == d1_bytes


def test_fidelity_sidecar_and_operating_point_are_derivation_scoped(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["inputs"]["bank_test"] = {"path": "d1/bank.tck"}
    man["derivations"]["d2"]["inputs"]["bank_test"] = _meta("d2/bank.tck")
    (root / "manifest.json").write_text(json.dumps(man))

    # Schema-2 sidecars bind the CURRENT bank, FOD and per-derivation peak bytes.
    bank = root / "bank.tck"
    bank.write_bytes(b"synthetic bank")
    fod = root / "fod.nii.gz"
    fod.write_bytes(b"synthetic fod")
    bank_sha = hashlib.sha256(bank.read_bytes()).hexdigest()
    fod_sha = hashlib.sha256(fod.read_bytes()).hexdigest()
    peaks = {}
    for name, peak_dir in (("d1", root / "fidelity"),
                           ("d2", root / "derivations" / "d2" / "fidelity")):
        peak_dir.mkdir(parents=True, exist_ok=True)
        peak = peak_dir / "fod_peak.nii.gz"
        peak.write_bytes(f"synthetic peak {name}".encode())
        peaks[name] = hashlib.sha256(peak.read_bytes()).hexdigest()

    def sidecar(value: float, derivation: str) -> dict:
        return {
            "schema": SIDECAR_SCHEMA,
            "bank_sha256": bank_sha,
            "fod_sha256": fod_sha,
            "sh_load_sha256": fod_sha,
            "peak_sha256": peaks[derivation],
            "R_grid": R_GRID,
            "frac_ge": np.full((1, len(R_GRID)), value, dtype=np.float32),
            "p5_ratio": np.asarray([value], dtype=np.float32),
            "n_segments": np.asarray([1], dtype=np.uint16),
            "flags": np.asarray([0], dtype=np.uint8),
            "ratios_present": True,
        }

    save_sidecar(root / "fidelity" / "bank_test.fidelity.npz", sidecar(0.2, "d1"))
    d2_fid = root / "derivations" / "d2" / "fidelity"
    save_sidecar(d2_fid / "bank_test.fidelity.npz", sidecar(0.8, "d2"))
    spec = SimpleNamespace(path=str(bank), provenance=SimpleNamespace(
        bank_sha256=bank_sha, fod_sha256=fod_sha,
    ))
    op = OperatingPoint(DEFAULT_R, DEFAULT_MIN_FRAC, "pilot")

    man["active_derivation"] = "d1"
    d1_block, d1_headers = _fidelity_payload(man, "bank_test", spec, np.array([0]), op, fod_path=str(fod))
    man["active_derivation"] = "d2"
    d2_block, d2_headers = _fidelity_payload(man, "bank_test", spec, np.array([0]), op, fod_path=str(fod))
    assert d1_headers["fidelityStatus"] == d2_headers["fidelityStatus"] == "ok"
    assert d1_block != d2_block

    sweep = root / "fidelity" / "sweep.csv"
    sweep.write_text("bank_id,R,min_frac,marked_frac,n\n")
    sweep_sha = hashlib.sha256(sweep.read_bytes()).hexdigest()
    sheet = root / "docs" / "qc" / "OPERATING-POINT-fidelity.md"
    sheet.parent.mkdir(parents=True)
    sheet.write_text(
        f"approved_by: owner\ndate: 2026-09-14\nR: {DEFAULT_R}\n"
        f"min_frac: {DEFAULT_MIN_FRAC}\nsweep_csv_sha256: {sweep_sha}\n"
    )
    man["active_derivation"] = "d1"
    d1_op = read_operating_point(
        root, sheet_path=dv.operating_point_path(man), sweep_path=dv.fidelity_sweep_path(man)
    )
    man["active_derivation"] = "d2"
    d2_op = read_operating_point(
        root, sheet_path=dv.operating_point_path(man), sweep_path=dv.fidelity_sweep_path(man)
    )
    assert d1_op.signed
    assert d2_op.source == "pilot"


def test_fidelity_writer_refuses_symlinked_first_lineage_operating_point(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = dv.migrate_manifest(_manifest(root), dv.KIND_RPE_PAIR)
    (root / "docs").symlink_to(tmp_path / "outside", target_is_directory=False)
    (tmp_path / "outside").mkdir()
    (root / "manifest.json").write_text(json.dumps(man))

    with pytest.raises(dv.DerivationError, match="case_root"):
        fidelity_sweep.run_sweep(root)


def test_nonfirst_artifact_verifies_every_declared_path_field(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d2"
    base = root / "d2" / "b0.nii.gz"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"base")
    man["derivations"]["d2"]["inputs"]["b0"]["sha256_by_field"]["path"] = (
        hashlib.sha256(base.read_bytes()).hexdigest()
    )
    man["derivations"]["d2"]["inputs"]["b0"]["bytes_by_field"]["path"] = base.stat().st_size
    names = {
        "path": "d2/main.bin",
        "sift2": "d2/weights.txt",
        "lut_path": "shared/lut.txt",
        "and[0]": "d2/include.nii.gz",
        "not[0]": "d2/exclude.nii.gz",
    }
    entry = {
        "path": names["path"],
        "sift2": names["sift2"],
        "lut_path": names["lut_path"],
        "and": [names["and[0]"]],
        "not": [names["not[0]"]],
        "sha256_by_field": {},
        "bytes_by_field": {},
    }
    for field, name in names.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(field.encode())
        entry["sha256_by_field"][field] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["bytes_by_field"][field] = path.stat().st_size
    man["derivations"]["d2"]["inputs"]["all"] = entry
    dv.validate(man)
    dv.validate_active_artifact_metadata(man)

    entry["bytes_by_field"]["not[0]"] += 1
    with pytest.raises(dv.ArtifactMetadataError, match=r"not\[0\].*bytes mismatch"):
        dv.validate_active_artifact_metadata(man)


def test_add_derivation_refuses_artifact_without_sha256(tmp_path):
    man = dv.migrate_manifest(_manifest(tmp_path), dv.KIND_RPE_PAIR)
    with pytest.raises(dv.DerivationError, match="sha256"):
        dv.add_derivation(
            man, "d2", dv.KIND_UNCORRECTED,
            {"b0": {"path": "d2/b0.nii.gz", "bytes_by_field": {"path": 1}}},
        )
    assert "d2" not in man["derivations"]


def test_corrected_derivation_has_no_uncorrected_warning():
    man = dv.migrate_manifest(_manifest(Path("/synthetic")), dv.KIND_RPE_PAIR)
    warning = _derivation_uncertainty(man)
    assert "no reverse-PE" not in warning
    assert "reverse-PE corrected" in warning


def test_validate_rejects_missing_or_unknown_kind(tmp_path):
    missing = _two_derivations(tmp_path)
    missing["derivations"]["d1"].pop("kind")
    with pytest.raises(dv.DerivationError, match="d1.*kind"):
        dv.validate(missing)

    unknown = _two_derivations(tmp_path)
    unknown["derivations"]["d2"]["kind"] = "topup"
    with pytest.raises(dv.DerivationError, match="d2.*topup"):
        dv.validate(unknown)
    unknown["active_derivation"] = "d2"
    with pytest.raises(dv.DerivationError, match="kind"):
        dv.floor_label(unknown)


def test_add_derivation_refuses_unsafe_id_and_layout_escape(tmp_path):
    man = dv.migrate_manifest(_manifest(tmp_path), dv.KIND_RPE_PAIR)
    with pytest.raises(dv.DerivationError, match="safe path segment"):
        dv.add_derivation(
            man, "/tmp/escape", dv.KIND_UNCORRECTED,
            {"b0": _meta("d2/b0.nii.gz")},
        )

    malformed = _two_derivations(tmp_path)
    entry = malformed["derivations"].pop("d2")
    malformed["derivations"]["../escape"] = entry
    malformed["active_derivation"] = "../escape"
    with pytest.raises(dv.DerivationError, match="safe path segment"):
        dv.validate(malformed)
    with pytest.raises(dv.DerivationError, match="case_root|safe path segment"):
        dv.artifact_dir(malformed, "qc")


def test_d2_qc_write_preserves_d1_signature(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["inputs"].update({
        "t1": {"path": "d1/t1.nii.gz"},
        "mask": {"path": "d1/mask.nii.gz"},
    })
    man["derivations"]["d2"]["inputs"].update({
        "t1": _meta("d2/t1.nii.gz"),
        "mask": _meta("d2/mask.nii.gz"),
    })
    d1_qc = {
        "auto": {"ok": True},
        "approved_by": "owner",
        "date": "2026-09-14",
        "sheet_sha": "d1-sheet",
        "sheet_path": "qc/t1_b0_reg_qc.png",
        "derivation": "d1",
    }
    man["t1_qc"] = d1_qc.copy()
    man["active_derivation"] = "d2"
    man["derivations"]["d2"]["qc"] = {
        "t1_qc": {
            "auto": {"ok": True},
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "derivation": "d2",
        }
    }
    d2_sheet = root / "derivations" / "d2" / "qc" / "t1_b0_reg_qc.png"
    d2_sheet.parent.mkdir(parents=True)
    d2_sheet.write_bytes(b"d2-sheet")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(man))

    result = check_registration.approve_t1_qc(
        str(manifest_path), approved_by="second-owner"
    )
    saved = json.loads(manifest_path.read_text())
    assert saved["t1_qc"] == d1_qc
    assert saved["derivations"]["d2"]["qc"]["t1_qc"] == result
    summary = dv.derivation_summary(saved)
    assert summary["derivations"][0]["qc_signed"]["t1_qc"] is True
    assert summary["derivations"][1]["qc_signed"]["t1_qc"] is True


def test_each_derivation_writer_refuses_invalid_manifest_and_symlink_alias(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d2"
    man["derivations"]["d2"]["inputs"].update({
        "t1": _meta("d2/t1.nii.gz"),
        "mask": _meta("d2/mask.nii.gz"),
    })
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    stats = delta_qc.DeltaStats(1.0, 2.0, 3.0, 8, np.ones((2, 2, 2)))

    bad = json.loads(manifest_path.read_text())
    bad["derivations"]["d2"].pop("inputs")
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(dv.DerivationError, match="inputs"):
        delta_qc.write_sheet(root, stats)
    with pytest.raises(dv.DerivationError, match="inputs"):
        fidelity_sweep.run_sweep(root)
    with pytest.raises(dv.DerivationError, match="inputs"):
        atlas_prep.prepare(manifest_path=str(manifest_path), wanted={})
    with pytest.raises(dv.DerivationError, match="inputs"):
        check_registration.main(["--manifest", str(manifest_path)])

    manifest_path.write_text(json.dumps(man))
    outside = tmp_path / "outside"
    outside.mkdir()
    layout_root = root / "derivations" / "d2"
    layout_root.mkdir(parents=True)
    for kind in ("qc", "fidelity", "normative"):
        (outside / kind).mkdir()
        (layout_root / kind).symlink_to(outside / kind, target_is_directory=True)
    with pytest.raises(dv.DerivationError, match="case_root"):
        delta_qc.write_sheet(root, stats)
    with pytest.raises(dv.DerivationError, match="case_root"):
        fidelity_sweep.run_sweep(root)
    with pytest.raises(dv.DerivationError, match="case_root"):
        atlas_prep.prepare(manifest_path=str(manifest_path), wanted={})
    with pytest.raises(dv.DerivationError, match="case_root"):
        check_registration.main(["--manifest", str(manifest_path)])


def test_corrected_derivation_has_no_uncorrected_text_on_server_surfaces(tmp_path, monkeypatch):
    man = dv.migrate_manifest(_manifest(tmp_path), dv.KIND_RPE_PAIR)
    label = dv.floor_label(man)
    assert "no reverse-PE" not in label
    assert "no reverse-PE" not in f"FILTERED PREVIEW | {label} | not navigation"
    assert "no reverse-PE" not in f"{label}; rigid T1 only; aid not navigation"

    captured = {}
    monkeypatch.setattr(
        check_registration,
        "write_qc_sheet",
        lambda **kwargs: captured.update(kwargs) or "sheet-sha",
    )
    monkeypatch.setattr(
        check_registration,
        "auto_sanity",
        lambda **kwargs: check_registration.AutoSanity(
            ok=True, grid_id_match=True, shape_match=True,
            mask_overlap_frac=1.0, t1_nonzero_frac=1.0, notes=(),
        ),
    )
    check_registration.run_check(
        b0_path="b0", t1_path="t1", mask_path="mask", out_dir=tmp_path,
        floor_label_text=label,
    )
    assert "no reverse-PE" not in captured["floor_label_text"]
    assert "reverse-PE corrected" in captured["floor_label_text"]

    service = SimpleNamespace(
        manifest=man,
        _lock=threading.Lock(),
        _job_lock=threading.Lock(),
        _active_job_id=None,
        _active_started=None,
        case_id="synthetic-w1",
        grid_id="grid",
        volume_id="volume",
        grid_provenance={},
        space_id="space",
        recipe_hash="recipe",
        presets={},
        banks={},
        filter_bank_path=None,
        filter_bank_count=None,
        lesion_shell=None,
        fa_vol=None,
        dec_vol=None,
        t1_vol=None,
        priors={},
        parcellation=None,
        runtime_identity=SimpleNamespace(
            public_metadata=lambda: {},
            restart_payload=lambda port: None,
        ),
    )
    handler_type = make_handler(service, ".")
    handler = object.__new__(handler_type)
    payload = {}
    handler.path = "/api/derivation"
    handler._runtime_current = lambda: True
    handler._enforce_browser_origin_policy = lambda: None
    handler._json = lambda code, obj: payload.update(status=code, body=obj)
    handler.do_GET()
    assert "no reverse-PE" not in payload["body"]["floor_label"]

    health = {}
    health_handler = object.__new__(handler_type)
    health_handler.path = "/api/health"
    health_handler._runtime_current = lambda: True
    health_handler._enforce_browser_origin_policy = lambda: None
    health_handler._json = lambda code, obj: health.update(status=code, body=obj)
    health_handler.do_GET()
    assert health["status"] == 200
    assert "no reverse-PE" not in health["body"]["uncertainty"]
    assert "reverse-PE corrected" in health["body"]["uncertainty"]

    # The filter route binds the bank bytes before and after filtering.
    filter_bank = tmp_path / "synthetic-filter-bank.tck"
    filter_bank.write_bytes(b"synthetic filter bank")
    service.filter_bank_path = str(filter_bank)
    service.grid = object()
    service.analytic_sources = SimpleNamespace(
        commit_result=lambda *args, **kwargs: SimpleNamespace(response_headers=lambda: {})
    )
    monkeypatch.setattr(serve_mod, "layers_from_request", lambda req, require_seed: object())
    monkeypatch.setattr(serve_mod, "compile_roi_layers", lambda grid, layers: SimpleNamespace(
        seed_mask=np.ones((1, 1), dtype=np.uint8),
        and_masks=[], or_mask=None, not_mask=None,
    ))
    monkeypatch.setattr(serve_mod, "filter_bank_memory", lambda **kwargs: (
        [], {"engine": "synthetic", "n_kept": 0, "n_corpus": 0, "elapsed_s": 0.0, "label": "synthetic"}
    ))
    filter_handler = object.__new__(handler_type)
    filter_headers = {}
    filter_handler._pack_lines = lambda *args, **kwargs: (b"", {"warning": ""}, np.zeros(0, dtype=np.int64))
    filter_handler._send = lambda code, body=b"", ctype="application/octet-stream", extra=None: filter_headers.update(
        status=code, body=body, extra=extra or {}
    )
    filter_handler._handle_filter({"seed": [[0, 0, 0]], "params": {}})
    assert filter_headers["status"] == 200
    assert "no reverse-PE" not in filter_headers["extra"]["X-warning"]
    assert "reverse-PE corrected" in filter_headers["extra"]["X-warning"]

    # The same assertions are made through the actual HTTP route methods,
    # using a synthetic in-memory service so this test does not need case data.
    monkeypatch.setattr(
        handler_type,
        "_pack_lines",
        lambda self, *args, **kwargs: (
            b"", {"warning": ""}, np.zeros(0, dtype=np.int64)
        ),
    )
    try:
        httpd = serve_mod.ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    except PermissionError:
        pytest.skip("sandbox blocks loopback sockets; standalone gate caveat")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/health", timeout=5) as response:
            health_body = json.loads(response.read())
        assert "no reverse-PE" not in health_body["uncertainty"]
        request = urllib.request.Request(
            base + "/api/filter",
            data=json.dumps({
                "gridId": service.grid_id,
                "volumeId": service.volume_id,
                "seed": [[0, 0, 0]],
                "params": {},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            filter_warning = response.headers["X-warning"]
        assert "no reverse-PE" not in filter_warning
        assert "reverse-PE corrected" in filter_warning
    finally:
        httpd.shutdown()


def test_nonfirst_artifact_refuses_malformed_or_mismatched_metadata(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d2"
    target = root / "d2" / "b0.nii.gz"
    target.parent.mkdir()
    target.write_bytes(b"actual")
    entry = man["derivations"]["d2"]["inputs"]["b0"]
    entry["bytes_by_field"]["path"] = len(b"actual")
    entry["sha256_by_field"]["path"] = "not-a-sha"
    with pytest.raises(dv.DerivationError, match="64 lowercase hex"):
        dv.validate(man)

    entry["sha256_by_field"]["path"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(dv.ArtifactMetadataError, match="sha256 mismatch"):
        dv.validate_active_artifact_metadata(man)

    entry["sha256_by_field"]["path"] = hashlib.sha256(b"actual").hexdigest()
    entry["bytes_by_field"]["path"] = 0
    with pytest.raises(dv.DerivationError, match="positive"):
        dv.validate(man)


def test_fidelity_sweep_refuses_missing_or_nonobject_manifest(tmp_path):
    with pytest.raises(dv.DerivationError, match="manifest missing"):
        fidelity_sweep.run_sweep(tmp_path)
    (tmp_path / "manifest.json").write_text("[]")
    with pytest.raises(dv.DerivationError, match="JSON object"):
        fidelity_sweep.run_sweep(tmp_path)


def test_approve_t1_qc_refuses_missing_active_lineage_auto_qc(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = dv.migrate_manifest(_manifest(root), dv.KIND_RPE_PAIR)
    sheet = dv.artifact_path(man, "qc", "t1_b0_reg_qc.png")
    sheet.parent.mkdir(parents=True)
    sheet.write_bytes(b"sheet")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    with pytest.raises(dv.DerivationError, match="auto QC block"):
        check_registration.approve_t1_qc(str(manifest_path), approved_by="owner")


def test_interrupted_regeneration_cannot_leave_old_signature_live(tmp_path, monkeypatch):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d2"
    man["derivations"]["d2"]["inputs"].update({
        "t1": _meta("d2/t1.nii.gz"),
        "mask": _meta("d2/mask.nii.gz"),
    })
    man["derivations"]["d2"]["qc"] = {
        "t1_qc": {
            "auto": {"ok": True},
            "approved_by": "old-owner",
            "date": "2026-09-14",
            "sheet_sha": "old-sheet",
            "derivation": "d2",
        }
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    monkeypatch.setattr(
        check_registration,
        "auto_sanity",
        lambda **kwargs: check_registration.AutoSanity(
            ok=True, grid_id_match=True, shape_match=True,
            mask_overlap_frac=1.0, t1_nonzero_frac=1.0, notes=(),
        ),
    )

    def interrupted_write(**kwargs):
        saved = json.loads(manifest_path.read_text())
        assert saved["derivations"]["d2"]["qc"]["t1_qc"]["approved_by"] is None
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(check_registration, "write_qc_sheet", interrupted_write)
    assert check_registration.main(["--manifest", str(manifest_path)]) == 1
    saved = json.loads(manifest_path.read_text())
    assert saved["derivations"]["d2"]["qc"]["t1_qc"]["approved_by"] is None
    assert saved["derivations"]["d2"]["qc"]["t1_qc"]["sheet_sha"] is None


def test_track_service_refuses_nonfirst_metadata_mismatch_at_case_load(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    man = _two_derivations(root)
    man["active_derivation"] = "d2"
    target = root / "d2" / "b0.nii.gz"
    target.parent.mkdir()
    target.write_bytes(b"actual")
    entry = man["derivations"]["d2"]["inputs"]["b0"]
    entry["bytes_by_field"]["path"] = len(b"actual")
    entry["sha256_by_field"]["path"] = hashlib.sha256(b"different").hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(man))
    with pytest.raises(dv.ArtifactMetadataError, match="sha256 mismatch"):
        TrackService(str(manifest_path))


def test_nonfirst_seed_refuses_nonsequence_and_not_recipe_fields(tmp_path):
    """`and`/`not` must be lists; a bare string or dict is a typed refusal, never iterated."""
    man = _two_derivations(tmp_path)
    man["active_derivation"] = "d2"
    for bad in ("d2/include.nii.gz", {"d2/include.nii.gz": {}}):
        man["derivations"]["d2"]["inputs"]["seed_x"] = {**_meta("d2/seed.nii.gz"), "and": bad}
        with pytest.raises(dv.DerivationError, match="must be a list"):
            dv.validate(man)
    man["derivations"]["d2"]["inputs"]["seed_x"] = {**_meta("d2/seed.nii.gz"), "not": "d2/x.nii.gz"}
    with pytest.raises(dv.DerivationError, match="must be a list"):
        dv.validate(man)


def test_fidelity_sweep_uses_one_manifest_snapshot(tmp_path, monkeypatch):
    """run_sweep must read the manifest once; table and sheet describe one lineage."""
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(tmp_path)))
    real = fidelity_sweep._layout_manifest
    calls: list[str] = []

    def once(case_root):
        calls.append(str(case_root))
        if len(calls) > 1:
            raise AssertionError("manifest re-read inside run_sweep: lineage may switch mid-run")
        return real(case_root)

    monkeypatch.setattr(fidelity_sweep, "_layout_manifest", once)
    summary = fidelity_sweep.run_sweep(tmp_path)
    assert calls == [str(tmp_path)]
    assert summary["n_rows"] == 0
    assert Path(summary["csv"]).is_file() and Path(summary["sheet"]).is_file()
