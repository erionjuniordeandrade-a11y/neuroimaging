"""Local runtime identity contracts using an ephemeral synthetic case."""

from __future__ import annotations

import json
import importlib
import os
import runpy
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.serve import serve
from tractlab.runtime_identity import CaseSourceChanged, RuntimeIdentity


AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
RUN = Path(__file__).resolve().parents[1] / "run.py"
SERVE_SCRIPT = Path(__file__).resolve().parents[1] / "serve.sh"


def _mm_image(data):
    img = nib.Nifti1Image(data, AFFINE)
    img.header.set_xyzt_units("mm")
    return img


def _tiny_manifest(tmp_path: Path) -> Path:
    shape = (24, 24, 24)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:20, 4:20, 4:20] = 1
    b0 = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    fod = np.zeros(shape + (45,), dtype=np.float32)
    fod[..., 0] = 2.0
    nib.save(_mm_image(mask), str(tmp_path / "mask.nii.gz"))
    nib.save(_mm_image(b0), str(tmp_path / "b0.nii.gz"))
    nib.save(_mm_image(fod), str(tmp_path / "fod.nii.gz"))
    manifest = {
        "case_id": "runtime-test-case",
        "case_root": str(tmp_path),
        "inputs": {
            "fod": {"path": "fod.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "b0": {"path": "b0.nii.gz"},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def _post(url: str, payload: dict[str, object]) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def _bank_manifest(tmp_path: Path) -> Path:
    manifest_path = _tiny_manifest(tmp_path)
    bank_path = tmp_path / "cst.tck"
    tractogram = nib.streamlines.Tractogram(
        [np.array([[4.0, 4.0, 4.0], [12.0, 12.0, 12.0]], dtype=np.float32)],
        affine_to_rasmm=np.eye(4),
    )
    nib.streamlines.save(tractogram, str(bank_path))
    (tmp_path / "cst.sift2.txt").write_text("1.0\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["bank_cst"] = {
        "path": "cst.tck",
        "sift2": "cst.sift2.txt",
        "label": "CST fixture",
        "n_streamlines": 1,
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _case_source_manifest(tmp_path: Path) -> Path:
    manifest_path = _tiny_manifest(tmp_path)
    lesion = np.zeros((24, 24, 24), dtype=np.uint8)
    lesion[9:15, 9:15, 9:15] = 1
    nib.save(_mm_image(lesion), str(tmp_path / "lesion.nii.gz"))
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["lesion"] = {"path": "lesion.nii.gz"}
    (tmp_path / "dwi.bvec").write_text("0 0 0\n0 0 0\n1 1 1\n")
    (tmp_path / "dwi.bval").write_text("0 1000 1000\n")
    manifest["acquisition_evidence"] = {
        "gradients": {"path": "dwi.bvec"},
    }
    manifest_path.write_text(json.dumps(manifest))
    operating_point = tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md"
    operating_point.parent.mkdir(parents=True)
    operating_point.write_text("approved_by: \n")
    return manifest_path


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_process_exit(pid: int) -> bool:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_bootstrap_and_document_share_one_boot_fingerprint(tmp_path):
    """An old server must publish the exact build/protocol expected by its HTML."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text(
        "<html><head><title>fixture</title></head><body>fixture</body></html>"
    )
    manifest = _tiny_manifest(tmp_path)
    httpd, _service = serve(str(manifest), str(viewer), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, body, _headers = _get(base + "/api/bootstrap")
        assert status == 200, body.decode()
        bootstrap = json.loads(body)
        assert bootstrap["viewerProtocol"] == 1
        assert bootstrap["expectedViewerProtocol"] == 1
        assert bootstrap["buildId"] == bootstrap["expectedClientBuildId"]
        assert bootstrap["bootId"]
        assert bootstrap["bootedAt"]
        assert bootstrap["caseId"] == "runtime-test-case"
        assert len(bootstrap["caseSourceHash"]) == 64
        assert bootstrap["manifestId"] == RuntimeIdentity(
            viewer, manifest_path=manifest
        ).manifest_id

        status, body, _headers = _get(base + "/api/health")
        assert status == 200
        assert json.loads(body)["caseSourceHash"] == bootstrap["caseSourceHash"]

        status, body, _headers = _get(base + "/")
        assert status == 200
        document = body.decode()
        assert (
            f'<meta name="tractlab-build" content="{bootstrap["buildId"]}">' in document
        )
        assert '<meta name="tractlab-protocol" content="1">' in document
    finally:
        httpd.shutdown()


@pytest.mark.parametrize(
    "changed_name",
    (
        "lesion.nii.gz",
        "docs/qc/OPERATING-POINT-fidelity.md",
        "dwi.bvec",
        "dwi.bval",
        "preflight.json",
    ),
)
def test_changed_case_source_requires_restart_before_new_anatomy_is_served(tmp_path, changed_name):
    """Case anatomy and served evidence are boot-bound like served code."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _case_source_manifest(tmp_path)
    httpd, _service = serve(str(manifest), str(viewer), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, body, _headers = _get(base + "/api/bootstrap")
        assert status == 200
        bootstrap = json.loads(body)
        changed = tmp_path / changed_name
        if changed.exists():
            changed.write_bytes(changed.read_bytes() + b"\nchanged\n")
        else:
            changed.write_text("{\"schema\": 1}\n")

        status, body, _headers = _get(base + "/api/bootstrap")
        assert status == 409, body.decode()
        payload = json.loads(body)
        assert payload["code"] == "restart_required"
        assert payload["bootCaseSourceHash"] == bootstrap["caseSourceHash"]
        assert payload["currentCaseSourceHash"] != bootstrap["caseSourceHash"]
        assert payload["caseId"] == bootstrap["caseId"]
        assert payload["manifestId"] == bootstrap["manifestId"]
        assert str(tmp_path) not in body.decode()
    finally:
        httpd.shutdown()


def test_case_source_change_during_startup_refuses_the_boot(monkeypatch, tmp_path):
    """The boot hash must predate anatomy loading and be checked afterward."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _case_source_manifest(tmp_path)
    module = importlib.import_module("tractlab.serve")
    original_service = module.TrackService

    class SourceMutatingService(original_service):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            lesion = tmp_path / "lesion.nii.gz"
            lesion.write_bytes(lesion.read_bytes() + b"\nchanged-during-startup\n")

    monkeypatch.setattr(module, "TrackService", SourceMutatingService)
    with pytest.raises(CaseSourceChanged):
        module.serve(str(manifest), str(viewer), port=0)


def test_changed_client_source_requires_restart_instead_of_mixed_html(tmp_path):
    """A process that booted old Python must not serve a newly saved client."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    index = viewer / "index.html"
    index.write_text("<html><head></head><body>before</body></html>")
    httpd, _service = serve(str(_tiny_manifest(tmp_path)), str(viewer), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert _get(base + "/api/bootstrap")[0] == 200
        index.write_text("<html><head></head><body>after</body></html>")

        status, body, _headers = _get(base + "/api/bootstrap")
        assert status == 409, body.decode()
        payload = json.loads(body)
        assert payload["code"] == "restart_required"
        assert payload["restartRequired"] is True
        assert "./serve.sh" in payload["error"]

        status, body, _headers = _get(base + "/")
        assert status == 409
        assert b"Restart required" in body
    finally:
        httpd.shutdown()


def test_named_bank_load_and_export_share_the_source_hash(tmp_path):
    """A saved review can compare the source used for display with its export."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    httpd, service = serve(str(_bank_manifest(tmp_path)), str(viewer), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    request = {
        "gridId": service.grid_id,
        "volumeId": service.volume_id,
        "bankId": "bank_cst",
    }
    try:
        status, _body, headers = _post(base + "/api/bank/load", request)
        assert status == 200
        source_hash = headers["X-bankSourceHash"]
        assert len(source_hash) == 64

        status, _body, headers = _post(base + "/api/export/tck", request)
        assert status == 200
        assert headers["X-bankSourceHash"] == source_hash
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("changed_name", ("cst.tck", "cst.sift2.txt"))
def test_changed_named_bank_source_is_refused_for_display_and_export(tmp_path, changed_name):
    """Neither a changed tract file nor changed SIFT2 weights can reuse boot state."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    httpd, service = serve(str(_bank_manifest(tmp_path)), str(viewer), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    request = {
        "gridId": service.grid_id,
        "volumeId": service.volume_id,
        "bankId": "bank_cst",
    }
    changed = tmp_path / changed_name
    try:
        assert _post(base + "/api/bank/load", request)[0] == 200
        if changed.suffix == ".tck":
            changed.write_bytes(changed.read_bytes() + b"\nchanged\n")
        else:
            changed.write_text("2.0\n")

        status, body, _headers = _post(base + "/api/bank/load", request)
        assert status == 409, body.decode()
        assert json.loads(body)["code"] == "bank_source_changed"

        status, body, _headers = _post(base + "/api/export/tck", request)
        assert status == 409, body.decode()
        assert json.loads(body)["code"] == "bank_source_changed"
    finally:
        httpd.shutdown()


def test_run_uses_the_v1_preview_port_by_default(monkeypatch, tmp_path):
    """The documented one-command launch must target the v1 review preview."""
    observed = {}
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    monkeypatch.setenv("TRACTLAB_MANIFEST", str(_tiny_manifest(tmp_path)))
    monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))

    class FakeServer:
        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            observed["shutdown"] = True

    class FakeService:
        case_id = "fixture"
        grid_id = "grid-id"
        volume_id = "volume-id"

    def fake_serve(manifest, viewer, *, port, **_kwargs):
        observed.update({"manifest": manifest, "viewer": viewer, "port": port})
        return FakeServer(), FakeService()

    module = importlib.import_module("tractlab.serve")
    monkeypatch.setattr(module, "serve", fake_serve)
    def no_listener(*_args, **_kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))

    monkeypatch.setattr(urllib.request, "urlopen", no_listener)
    monkeypatch.setattr(sys, "argv", [str(RUN)])
    runpy.run_path(str(RUN), run_name="__main__")
    assert observed["port"] == 18993
    assert observed["shutdown"] is True


def test_run_reuses_a_matching_healthy_server(monkeypatch, tmp_path):
    """A second launch must leave an already-matching preview alone."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _tiny_manifest(tmp_path)
    identity = RuntimeIdentity(viewer, manifest_path=manifest)
    requests = []

    class BootstrapHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            payload = json.dumps({
                "viewerProtocol": 1,
                "buildId": identity.build_id,
                "caseId": identity.case_id,
                "manifestId": identity.manifest_id,
                "caseSourceHash": identity.case_source_hash,
                "runtimeStatus": "ok",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), BootstrapHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    module = importlib.import_module("tractlab.serve")
    try:
        def should_not_start(*_args, **_kwargs):
            raise AssertionError("a matching server must be reused")

        monkeypatch.setattr(module, "serve", should_not_start)
        monkeypatch.setenv("TRACTLAB_MANIFEST", str(manifest))
        monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
        monkeypatch.setattr(sys, "argv", [str(RUN), str(httpd.server_address[1])])
        runpy.run_path(str(RUN), run_name="__main__")
        assert requests == ["/api/bootstrap"]
    finally:
        httpd.shutdown()


def test_run_rejects_same_build_for_a_different_configured_case(monkeypatch, tmp_path, capsys):
    """A code-identical server for another manifest must not be reused."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    first_manifest = _tiny_manifest(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_manifest = _tiny_manifest(other_dir)
    other = json.loads(other_manifest.read_text())
    other["case_id"] = "other-runtime-test-case"
    other_manifest.write_text(json.dumps(other))
    first_identity = RuntimeIdentity(viewer, manifest_path=first_manifest)

    class BootstrapHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            payload = json.dumps({
                "viewerProtocol": 1,
                "buildId": first_identity.build_id,
                "caseId": first_identity.case_id,
                "manifestId": first_identity.manifest_id,
                "caseSourceHash": first_identity.case_source_hash,
                "runtimeStatus": "ok",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), BootstrapHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    module = importlib.import_module("tractlab.serve")
    try:
        def should_not_start(*_args, **_kwargs):
            raise AssertionError("a different configured case must not be reused")

        monkeypatch.setattr(module, "serve", should_not_start)
        monkeypatch.setenv("TRACTLAB_MANIFEST", str(other_manifest))
        monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
        monkeypatch.setattr(sys, "argv", [str(RUN), str(httpd.server_address[1])])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(RUN), run_name="__main__")
        assert stopped.value.code == 2
        assert "incompatible" in capsys.readouterr().err
    finally:
        httpd.shutdown()


def test_run_rejects_a_changed_case_source_with_the_same_manifest(monkeypatch, tmp_path, capsys):
    """Manifest identity alone cannot reuse a server after anatomy is replaced."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _case_source_manifest(tmp_path)
    original_identity = RuntimeIdentity(viewer, manifest_path=manifest)
    lesion = tmp_path / "lesion.nii.gz"
    lesion.write_bytes(lesion.read_bytes() + b"\nreplacement\n")
    replacement_identity = RuntimeIdentity(viewer, manifest_path=manifest)
    assert replacement_identity.manifest_id == original_identity.manifest_id
    assert replacement_identity.case_source_hash != original_identity.case_source_hash

    class BootstrapHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            payload = json.dumps({
                "viewerProtocol": 1,
                "buildId": original_identity.build_id,
                "caseId": original_identity.case_id,
                "manifestId": original_identity.manifest_id,
                "caseSourceHash": original_identity.case_source_hash,
                "runtimeStatus": "ok",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), BootstrapHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    module = importlib.import_module("tractlab.serve")
    try:
        def should_not_start(*_args, **_kwargs):
            raise AssertionError("changed case sources must not reuse an old server")

        monkeypatch.setattr(module, "serve", should_not_start)
        monkeypatch.setenv("TRACTLAB_MANIFEST", str(manifest))
        monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
        monkeypatch.setattr(sys, "argv", [str(RUN), str(httpd.server_address[1])])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(RUN), run_name="__main__")
        assert stopped.value.code == 2
        assert "incompatible" in capsys.readouterr().err
    finally:
        httpd.shutdown()


def test_run_refuses_an_occupied_incompatible_port(monkeypatch, tmp_path, capsys):
    """A foreign or stale listener must never be replaced or silently reused."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")

    class IncompatibleHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            payload = json.dumps({
                "viewerProtocol": 1,
                "buildId": "other-build",
                "runtimeStatus": "ok",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), IncompatibleHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    module = importlib.import_module("tractlab.serve")
    try:
        def should_not_start(*_args, **_kwargs):
            raise AssertionError("an occupied incompatible port must not be replaced")

        monkeypatch.setattr(module, "serve", should_not_start)
        monkeypatch.setenv("TRACTLAB_MANIFEST", str(_tiny_manifest(tmp_path)))
        monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
        monkeypatch.setattr(sys, "argv", [str(RUN), str(httpd.server_address[1])])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(RUN), run_name="__main__")
        assert stopped.value.code == 2
        message = capsys.readouterr().err
        assert "occupied" in message
        assert "incompatible" in message
        assert "stop it yourself" in message
    finally:
        httpd.shutdown()


def test_run_names_a_missing_required_input_without_a_filesystem_trace(monkeypatch, tmp_path, capsys):
    """A bad local case should explain the missing input before bind/startup."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    case_root = tmp_path / "case"
    case_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "case_id": "missing-input-fixture",
        "case_root": str(case_root),
        "inputs": {
            "b0": {"path": "missing-b0.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "fod": {"path": "fod.nii.gz"},
        },
    }))
    module = importlib.import_module("tractlab.serve")

    def should_not_start(*_args, **_kwargs):
        raise AssertionError("missing inputs must be rejected before server startup")

    def no_listener(*_args, **_kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))

    monkeypatch.setattr(module, "serve", should_not_start)
    monkeypatch.setattr(urllib.request, "urlopen", no_listener)
    monkeypatch.setenv("TRACTLAB_MANIFEST", str(manifest))
    monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
    monkeypatch.setattr(sys, "argv", [str(RUN), "18991"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(RUN), run_name="__main__")
    assert stopped.value.code == 2
    message = capsys.readouterr().err
    assert "required input unavailable: b0" in message
    assert str(case_root) not in message
    assert "Traceback" not in message


def test_run_hides_backend_file_paths_when_initialization_fails(monkeypatch, tmp_path, capsys):
    """A late local-file failure remains actionable without leaking a traceback."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    module = importlib.import_module("tractlab.serve")

    def missing_after_validation(*_args, **_kwargs):
        raise FileNotFoundError("/private/clinical-case/secret-source.nii.gz")

    def no_listener(*_args, **_kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))

    monkeypatch.setattr(module, "serve", missing_after_validation)
    monkeypatch.setattr(urllib.request, "urlopen", no_listener)
    monkeypatch.setenv("TRACTLAB_MANIFEST", str(_tiny_manifest(tmp_path)))
    monkeypatch.setenv("TRACTLAB_VIEWER", str(viewer))
    monkeypatch.setattr(sys, "argv", [str(RUN), "18991"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(RUN), run_name="__main__")
    assert stopped.value.code == 2
    message = capsys.readouterr().err
    assert "could not initialize required local inputs" in message
    assert "secret-source" not in message
    assert "Traceback" not in message


def test_serve_shell_uses_the_configured_interpreter(tmp_path):
    """The one-command entry point can use the documented local interpreter override."""
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf 'fake-python:%s\\n' \"$*\"\n")
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT), "18991"],
        cwd=SERVE_SCRIPT.parent,
        env={
            **os.environ,
            "TRACTLAB_PYTHON": str(fake_python),
            "TRACTLAB_MANIFEST": str(tmp_path / "missing-manifest.json"),
            "TRACTLAB_VIEWER": str(tmp_path / "missing-viewer"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fake-python:run.py 18991"


def test_serve_shell_forwards_the_background_flag(tmp_path):
    """The documented detached launch reaches run.py instead of being dropped."""
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf 'fake-python:%s\\n' \"$*\"\n")
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT), "18991", "--background"],
        cwd=SERVE_SCRIPT.parent,
        env={
            **os.environ,
            "TRACTLAB_PYTHON": str(fake_python),
            "TRACTLAB_MANIFEST": str(tmp_path / "missing-manifest.json"),
            "TRACTLAB_VIEWER": str(tmp_path / "missing-viewer"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fake-python:run.py 18991 --background"


def test_background_preview_reports_readiness_and_has_a_safe_stop_path(tmp_path):
    """Detached preview survives the launcher process and can stop only its own PID."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _tiny_manifest(tmp_path)
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    env = {
        **os.environ,
        "TRACTLAB_PYTHON": sys.executable,
        "TRACTLAB_MANIFEST": str(manifest),
        "TRACTLAB_VIEWER": str(viewer),
        "TRACTLAB_RUNTIME_DIR": str(runtime_dir),
    }
    started = subprocess.run(
        ["bash", str(SERVE_SCRIPT), str(port), "--background"],
        cwd=SERVE_SCRIPT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )
    state_path = runtime_dir / f"preview-{port}.json"
    try:
        assert started.returncode == 0, started.stderr
        assert "ready" in started.stdout
        assert state_path.is_file()
        state = json.loads(state_path.read_text())
        status, body, _headers = _get(f"http://127.0.0.1:{port}/api/bootstrap")
        assert status == 200
        assert json.loads(body)["bootId"] == state["bootId"]

        stopped = subprocess.run(
            ["bash", str(SERVE_SCRIPT), str(port), "--stop"],
            cwd=SERVE_SCRIPT.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert stopped.returncode == 0, stopped.stderr
        assert "stopped" in stopped.stdout
        assert not state_path.exists()
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bootstrap", timeout=1)
        assert _wait_for_process_exit(int(state["pid"]))
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bootstrap", timeout=1)
    finally:
        if state_path.exists():
            subprocess.run(
                ["bash", str(SERVE_SCRIPT), str(port), "--stop"],
                cwd=SERVE_SCRIPT.parent,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )


def test_background_preview_stop_accepts_its_own_restart_required_identity(tmp_path):
    """A changed source must not strand the background process that owns it."""
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html><head></head><body>fixture</body></html>")
    manifest = _case_source_manifest(tmp_path)
    runtime_dir = tmp_path / "runtime"
    port = _free_loopback_port()
    env = {
        **os.environ,
        "TRACTLAB_PYTHON": sys.executable,
        "TRACTLAB_MANIFEST": str(manifest),
        "TRACTLAB_VIEWER": str(viewer),
        "TRACTLAB_RUNTIME_DIR": str(runtime_dir),
    }
    state_path = runtime_dir / f"preview-{port}.json"
    started = subprocess.run(
        ["bash", str(SERVE_SCRIPT), str(port), "--background"],
        cwd=SERVE_SCRIPT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )
    try:
        assert started.returncode == 0, started.stderr
        state = json.loads(state_path.read_text())
        lesion = tmp_path / "lesion.nii.gz"
        lesion.write_bytes(lesion.read_bytes() + b"\nchanged-after-boot\n")
        status, body, _headers = _get(f"http://127.0.0.1:{port}/api/bootstrap")
        assert status == 409, body.decode()
        restart = json.loads(body)
        assert restart["bootId"] == state["bootId"]
        assert restart["bootBuildId"] == state["buildId"]
        assert restart["bootCaseSourceHash"] == state["caseSourceHash"]
        assert restart["caseId"] == state["caseId"]
        assert restart["manifestId"] == state["manifestId"]

        stopped = subprocess.run(
            ["bash", str(SERVE_SCRIPT), str(port), "--stop"],
            cwd=SERVE_SCRIPT.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert stopped.returncode == 0, stopped.stderr
        assert "stopped" in stopped.stdout
        assert not state_path.exists()
    finally:
        if state_path.exists():
            subprocess.run(
                ["bash", str(SERVE_SCRIPT), str(port), "--stop"],
                cwd=SERVE_SCRIPT.parent,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )


def test_background_stop_refuses_a_foreign_restart_identity(tmp_path):
    """A 409 alone cannot authorize signaling a recorded managed PID."""
    port = _free_loopback_port()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    state_path = runtime_dir / f"preview-{port}.json"
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    state = {
        "pid": sentinel.pid,
        "port": port,
        "bootId": "owned-boot",
        "buildId": "owned-build",
        "caseId": "owned-case",
        "manifestId": "owned-manifest",
        "caseSourceHash": "a" * 64,
    }
    state_path.write_text(json.dumps(state))

    class ForeignRestartHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({
                "code": "restart_required",
                "runtimeStatus": "restart-required",
                "bootId": "owned-boot",
                "bootBuildId": "owned-build",
                "bootCaseSourceHash": "b" * 64,
                "caseId": "owned-case",
                "manifestId": "owned-manifest",
            }).encode()
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), ForeignRestartHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "TRACTLAB_PYTHON": sys.executable,
        "TRACTLAB_RUNTIME_DIR": str(runtime_dir),
    }
    try:
        stopped = subprocess.run(
            ["bash", str(SERVE_SCRIPT), str(port), "--stop"],
            cwd=SERVE_SCRIPT.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert stopped.returncode == 2
        assert "does not match" in stopped.stderr
        assert sentinel.poll() is None
        assert state_path.exists()
    finally:
        httpd.shutdown()
        httpd.server_close()
        if sentinel.poll() is None:
            sentinel.terminate()
            sentinel.wait(timeout=10)
