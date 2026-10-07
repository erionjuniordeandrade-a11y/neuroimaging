"""Generated regressions for population identity and publication gates."""

from __future__ import annotations

import hashlib
import os
import threading
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import tractlab.bank as bank

import tractlab.serve as serve_module
import tractlab.sift2_util as sift2_util
import tractlab.track as track_module
from tractlab.bank import BankSpec
from tractlab.results import Result
from tractlab.serve import TrackService, make_handler
from tractlab.track import Outcome, TrackParams, run_tckgen


def _line(x: float = 0.0) -> np.ndarray:
    return np.array([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0]], dtype=np.float32)


def _handler(service, path: str):
    handler_type = make_handler(service, ".")
    handler = handler_type.__new__(handler_type)
    handler.path = path
    return handler


@pytest.mark.parametrize("line", [np.empty((0, 3)), np.zeros((1, 3))])
def test_result_rejects_non_exportable_streamlines(line):
    with pytest.raises(ValueError, match="at least 2 points"):
        Result.build(
            source_population="generated",
            derivation_id=None,
            grid_id="grid",
            volume_id="volume",
            lines=[line],
        )


def test_bank_load_rejects_degenerate_streamlines(monkeypatch):
    monkeypatch.setattr(
        bank,
        "load_tracks_cached",
        lambda _path: ([np.zeros((1, 3), dtype=np.float32)], np.array([0.0])),
    )
    with pytest.raises(ValueError, match="at least 2 points"):
        bank.load_prebuilt_bundle("generated.tck")


@pytest.mark.parametrize("stdout", ["count: -1\n", "count: 1.5\n", "count: 1\ncount: 2\n", "header only\n"])
def test_tckinfo_count_rejects_unverified_headers(monkeypatch, stdout):
    monkeypatch.setattr(
        track_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout),
    )
    assert track_module._tckinfo_count("generated.tck") is None


def test_tracking_refuses_tckinfo_decoded_count_mismatch(tmp_path, monkeypatch):
    line = _line()

    class FakeProcess:
        pid = os.getpid()
        returncode = 0

        def __init__(self, argv):
            self.argv = argv

        def communicate(self, timeout=None):
            del timeout
            nib.streamlines.save(
                nib.streamlines.Tractogram([line], affine_to_rasmm=np.eye(4)),
                self.argv[2],
            )
            return b"", b""

    monkeypatch.setattr(
        track_module.subprocess,
        "Popen",
        lambda argv, **kwargs: FakeProcess(argv),
    )
    monkeypatch.setattr(
        track_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="count: 2\n"),
    )

    result = run_tckgen(
        "fod.mif",
        "seed.nii.gz",
        "mask.nii.gz",
        str(tmp_path),
        TrackParams(seeds=8, select=2, nthreads=0),
    )

    assert result.outcome is Outcome.ENGINE_ERROR
    assert result.n_accepted is None
    assert "decoded" in (result.warning or "")


def test_sift2_requires_explicit_bank_and_weight_hashes(tmp_path):
    bank_bytes = b"generated-bank-bytes"
    bank_sha = hashlib.sha256(bank_bytes).hexdigest()
    weight_path = tmp_path / "bank.sift2.txt"
    weight_path.write_text("1.0\n2.0\n")
    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()

    assert sift2_util.load_sift2_weights(
        str(weight_path),
        n_expected=2,
        require_binding=True,
        expected_bank_sha256=bank_sha,
        expected_weight_sha256=weight_sha,
    ) is not None
    assert sift2_util.load_sift2_weights(
        str(weight_path), n_expected=2, require_binding=True,
    ) is None
    assert sift2_util.load_sift2_weights(
        str(weight_path),
        n_expected=2,
        require_binding=True,
        expected_bank_sha256=bank_sha,
        expected_weight_sha256="0" * 64,
    ) is None


@pytest.mark.parametrize("contents", ["-1.0\n2.0\n", "nan\n2.0\n", "inf\n2.0\n"])
def test_sift2_rejects_nonfinite_or_negative_values(tmp_path, contents):
    path = tmp_path / "bad.sift2.txt"
    path.write_text(contents)
    bank_sha = hashlib.sha256(b"bank").hexdigest()
    weight_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sift2_util.load_sift2_weights(
        str(path),
        n_expected=2,
        require_binding=True,
        expected_bank_sha256=bank_sha,
        expected_weight_sha256=weight_sha,
    ) is None


def test_bank_load_applies_sift2_only_with_manifest_receipts(tmp_path, monkeypatch):
    bank_path = tmp_path / "bank.tck"
    bank_path.write_bytes(b"bank-bytes")
    weight_path = tmp_path / "bank.sift2.txt"
    weight_path.write_text("1.0\n2.0\n")
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    spec = SimpleNamespace(
        path=str(bank_path),
        label="Generated",
        engine="generated",
        role="",
        sift2_path=str(weight_path),
        provenance=None,
    )
    service = SimpleNamespace(
        banks={"bank": spec},
        _lock=threading.Lock(),
        manifest={"inputs": {"bank": {"sha256": bank_sha, "sift2_sha256": weight_sha}}},
        case_root=str(tmp_path),
        fod=str(tmp_path / "fod.mif"),
        operating_point=None,
        grid=object(),
        geom_floor_mm=None,
        lesion_shell=np.empty((0, 3), dtype=np.float32),
    )
    handler = _handler(service, "/api/bank/load")
    captured = {}
    packed_weights = []
    published = []

    def fake_pack(lines, _engine, extra_hdr=None, **kwargs):
        del lines
        packed_weights.append(kwargs.get("weights"))
        return (
            b"",
            {"warning": "", "lineCount": 0, **(extra_hdr or {})},
            np.empty(0, dtype=np.int64),
        )

    handler._pack_lines = fake_pack
    handler._publish_result = lambda *args, **kwargs: published.append(True)
    handler._send = lambda code, body=b"", **kwargs: captured.update(
        code=code, extra=kwargs.get("extra", {})
    ) or captured
    monkeypatch.setattr(
        serve_module,
        "load_prebuilt_bundle",
        lambda *_args, **_kwargs: (
            [_line(0.0), _line(10.0)],
            {"n_kept": 2, "engine": "generated", "label": "generated"},
        ),
    )
    monkeypatch.setattr(
        serve_module,
        "_fidelity_payload",
        lambda *_args, **_kwargs: (b"", {"fidelityStatus": "absent"}),
    )

    handler._handle_bank_load({"bankId": "bank"})

    assert captured["code"] == 200
    assert packed_weights and packed_weights[0] is not None
    assert captured["extra"]["X-sift2Binding"] == "bound"
    assert captured["extra"]["X-sift2Applied"] == "1"

    # A count match alone cannot activate the same sidecar.
    service.manifest = {"inputs": {"bank": {"sha256": bank_sha}}}
    packed_weights.clear()
    published.clear()
    captured.clear()
    handler._handle_bank_load({"bankId": "bank"})
    assert captured["extra"]["X-sift2Binding"] == "unbound"
    assert captured["extra"]["X-sift2Applied"] == "0"
    assert packed_weights and packed_weights[0] is None


def test_connectotomy_cut_cache_refuses_same_count_reorder(monkeypatch):
    current = [_line(0.0), _line(10.0)]
    spec = SimpleNamespace(path="bank.tck", label="generated", engine="generated", role="")
    service = TrackService.__new__(TrackService)
    service._connectotomy_ready = False
    service._connectotomy_lock = threading.Lock()
    service._lesion_missing_error = None
    service._lesion_path = "lesion.nii.gz"
    service.lesion_vol = object()
    service.geom_floor_mm = 3.0
    service.grid = object()
    service.banks = {"bank": spec}
    service._connectotomy_error = None
    service.connectotomy_report = None
    service._connectotomy_cut_idx = {}
    # main validates named bank sources first; this test isolates the line digest.
    service.bank_sources = SimpleNamespace(validate=lambda _bank_id: "stub-source-hash")

    monkeypatch.setattr(serve_module, "load_lesion_bool", lambda *_: np.ones((2, 2, 2), bool))
    monkeypatch.setattr(
        serve_module,
        "load_prebuilt_bundle",
        lambda *_args, **_kwargs: (current, np.ones(len(current))),
    )
    monkeypatch.setattr(
        serve_module,
        "compute_connectotomy",
        lambda *_args, **_kwargs: (
            {"banks": [{"id": "bank", "n_cut": 1, "n_bank": 2, "fraction": 0.5}]},
            {"bank": np.array([0], dtype=np.int32)},
        ),
    )

    assert service.ensure_connectotomy() is not None
    assert "bank" in service._connectotomy_bank_digest

    current[:] = [current[1], current[0]]
    handler = _handler(service, "/api/connectotomy/bank/cut")
    captured = {}
    handler._json = lambda code, obj: captured.update(code=code, body=obj) or captured
    handler._send = lambda *args, **kwargs: captured.update(code=args[0]) or captured
    handler._pack_lines = lambda *args, **kwargs: pytest.fail("stale cut must not be packed")

    handler._handle_connectotomy_cut()

    assert captured["code"] == 422
    assert captured["body"]["code"] == "evidence_changed"


def test_filter_refuses_bank_change_between_read_and_publication(tmp_path, monkeypatch):
    bank_path = tmp_path / "filter.tck"
    bank_path.write_bytes(b"before")
    service = SimpleNamespace(
        filter_bank_path=str(bank_path),
        _lock=threading.Lock(),
        grid=object(),
    )
    handler = _handler(service, "/api/filter")
    captured = {}
    published = []
    handler._json = lambda code, obj: captured.update(code=code, body=obj) or captured
    handler._send = lambda *args, **kwargs: captured.update(code=args[0]) or captured
    handler._pack_lines = lambda *args, **kwargs: (b"", {"lineCount": 0}, np.empty(0, dtype=np.int64))
    handler._publish_result = lambda *args, **kwargs: published.append(True)

    compiled = SimpleNamespace(seed_mask=None, and_masks=[], or_mask=None, not_mask=None)
    monkeypatch.setattr(serve_module, "layers_from_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(serve_module, "compile_roi_layers", lambda *args, **kwargs: compiled)

    def changing_filter(*args, **kwargs):
        bank_path.write_bytes(b"after")
        return [_line()], {
            "engine": "generated",
            "label": "generated",
            "n_kept": 1,
            "n_corpus": 1,
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(serve_module, "filter_bank_memory", changing_filter)
    handler._handle_filter({"params": {"minlength": 1.0}})

    assert captured["code"] == 422
    assert captured["body"]["code"] == "evidence_changed"
    assert not published


def test_banks_manifest_count_is_explicitly_an_estimate():
    service = SimpleNamespace(
        banks={
            "generated": BankSpec(
                path="generated.tck",
                label="Generated",
                n_streamlines=123,
                engine="generated",
                note="synthetic",
            )
        },
        filter_bank_path=None,
        filter_bank_note="generated filter",
        manifest={},
    )
    handler = _handler(service, "/api/banks")
    captured = {}
    handler._json = lambda code, obj: captured.update(code=code, body=obj) or captured

    handler._do_GET_inner()

    assert captured["code"] == 200
    row = captured["body"]["banks"][0]
    assert row["nStreamlinesEstimate"] == 123
    assert row["nStreamlinesStatus"] == "estimate"
