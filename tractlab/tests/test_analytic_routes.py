"""Exercise real route handlers with explicit source identities and synthetic lines."""
from types import SimpleNamespace
import importlib

import numpy as np

from tractlab.analytic_source import AnalyticSourceRegistry

module = importlib.import_module("tractlab.serve")


def handler(tmp_path, monkeypatch):
    original = [np.array([[1., 2., 3.], [4., 5., 6.]])]
    wrong = [np.array([[91., 92., 93.], [94., 95., 96.]])]
    bank = tmp_path / "bank.tck"
    bank.write_bytes(b"original bank payload")
    service = SimpleNamespace(
        analytic_sources=AnalyticSourceRegistry(), last_lines=wrong, grid=object(),
        banks={"cst": SimpleNamespace(path=str(bank), n_streamlines=1)},
        _export_dir=str(tmp_path),
    )
    service.analytic_sources.commit_result(wrong, kind="recovery")
    monkeypatch.setattr(module, "load_prebuilt_bundle", lambda path, max_keep: (original, {}))
    route = object.__new__(module.make_handler(service, str(tmp_path)))
    route._send = lambda code, body, *args, **kwargs: (code, body, kwargs.get("extra", {}))
    route._json = lambda code, body: (code, body, {})
    return route, service, original


def test_envelope_uses_requested_bank_even_after_recovery(tmp_path, monkeypatch):
    route, _service, original = handler(tmp_path, monkeypatch)
    seen = []
    def build(lines, grid, margin_mm):
        seen.extend(lines)
        return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype="<f4"), np.array([[0, 1, 2]], dtype="<u4")
    monkeypatch.setattr(module, "build_margin_hull", build)
    status, _, headers = route._handle_margin({"bankId": "cst", "marginMm": 5})
    assert status == 200 and headers["X-bankId"] == "cst"
    np.testing.assert_array_equal(seen[0], original[0])
    assert route._handle_margin({"bankId": "unknown"})[0] == 400
    assert route._handle_margin({})[0] == 400


def test_export_is_bound_to_bank_or_current_result_and_refuses_stale(tmp_path, monkeypatch):
    route, service, original = handler(tmp_path, monkeypatch)
    assert route._handle_export({"bankId": "unknown"})[0] == 400
    assert route._handle_export({})[0] == 400
    status, body, headers = route._handle_export({"bankId": "cst"})
    assert status == 200 and body == b"original bank payload" and headers["X-bankId"] == "cst"
    result = service.analytic_sources.commit_result(original, kind="filter")
    status, body, headers = route._handle_export({"resultId": result.result_id})
    assert status == 200 and body.startswith(b"mrtrix tracks")
    assert headers["X-resultId"] == result.result_id
    service.analytic_sources.commit_result(original, kind="recovery")
    assert route._handle_export({"resultId": result.result_id})[0] == 409
