"""Workbench server behaviour on synthetic inputs only."""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _get(port: int, path: str, host: str | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers={"Host": host or f"127.0.0.1:{port}"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _poll_ready(port: int, path: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = _get(port, path)
        assert status == 200
        data = json.loads(body)
        if data["state"] in ("ready", "error"):
            return data
        time.sleep(0.3)
    raise AssertionError(f"{path} never became ready")


@pytest.fixture()
def server(tmp_path: Path):
    phantom = tmp_path / "phantom.capsule.html"
    phantom.write_text("<!doctype html><title>synthetic phantom stand-in</title>")
    capsules = tmp_path / "capsules"
    capsules.mkdir()
    (capsules / "teaching-one.capsule.html").write_text("<!doctype html><title>synthetic capsule</title>")
    proc = subprocess.Popen(
        [sys.executable, "-m", "neuro_workbench.server", "--port", "0", "--cache-dir", str(tmp_path / "cache"),
         "--archive-dir", str(tmp_path / "archive"), "--phantom", str(phantom), "--capsule-dir", str(capsules)],
        stdout=subprocess.PIPE, text=True)
    url = json.loads(proc.stdout.readline())["url"]
    port = int(url.rsplit(":", 1)[1].strip("/"))
    yield port
    proc.terminate()
    proc.wait(timeout=10)


def test_shell_has_the_four_views(server):
    status, body = _get(server, "/")
    assert status == 200
    html = body.decode()
    for view in ("case", "imaging", "tracts", "atlas"):
        assert f'data-view="{view}"' in html
    assert "not for clinical use" in html


def test_case_list_has_demo_and_explicit_capsules_only(server):
    status, body = _get(server, "/api/cases")
    cases = json.loads(body)["cases"]
    assert status == 200
    assert cases[0]["id"] == "demo-phantom" and cases[0]["synthetic"] is True
    assert cases[0]["views"] == {"case": True, "imaging": True, "tracts": True}
    assert [c["title"] for c in cases[1:]] == ["teaching-one"]
    assert cases[1]["views"]["tracts"] is False


def test_demo_imaging_serves_the_phantom_capsule(server):
    data = _poll_ready(server, "/api/case/demo-phantom/imaging")
    assert data["state"] == "ready"
    status, body = _get(server, data["url"])
    assert status == 200 and b"synthetic phantom stand-in" in body


def test_demo_tracts_start_a_tractlab_child(server):
    tracts = _poll_ready(server, "/api/case/demo-phantom/tracts")
    assert tracts["state"] == "ready", tracts
    child_port = int(tracts["url"].split(":")[2].split("/")[0])
    status, body = _get(child_port, "/?teaching=1")
    assert status == 200 and b"SYNTHETIC QA" in body


def test_atlas_is_case_free_and_cannot_reach_the_workstation(server):
    status, body = _get(server, "/api/atlas")
    assert status == 200 and json.loads(body) == {"state": "ready", "url": "/atlas/atlas.html"}
    assert _get(server, "/api/case/demo-phantom/atlas")[0] == 404
    status, page = _get(server, "/atlas/atlas.html")
    assert status == 200 and b"REFERENCE ATLAS" in page
    assert b"profile=clinical" not in page and b"Case reconstruction" not in page
    assert _get(server, "/atlas/atlas/manifest.json")[0] == 200
    assert _get(server, "/atlas/vendor/three.module.js")[0] == 200
    for blocked in ("/atlas/index.html", "/atlas/", "/atlas/app_home.html", "/atlas/../README.md",
                    "/atlas/atlas/../../../pyproject.toml", "/atlas/.git/config"):
        assert _get(server, blocked)[0] == 404, blocked


def test_rejects_foreign_host_and_unknown_routes(server):
    assert _get(server, "/api/cases", host="evil.example")[0] == 403
    assert _get(server, "/api/case/nope/imaging")[0] == 404
    assert _get(server, "/case/demo-phantom/../../etc/passwd")[0] == 404


def _post(port: int, path: str, body: bytes, headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=body, headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json",
                                                  **(headers or {})})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_annotations_round_trip_privately(server, tmp_path):
    path = "/api/case/demo-phantom/annotations"
    status, body = _get(server, path)
    assert status == 200 and json.loads(body) == {}
    doc = {"version": 1, "points": [{"name": "Point 1", "mm": [1.0, 2.0, 3.0]}],
           "trajectories": [{"name": "Trajectory 1", "target": [0, 0, 0], "entry": [10, 0, 0]}]}
    origin = {"Origin": f"http://127.0.0.1:{server}"}
    status, body = _post(server, path, json.dumps(doc).encode(), origin)
    assert status == 200 and "saved" in json.loads(body)
    status, body = _get(server, path)
    back = json.loads(body)
    assert back["points"] == doc["points"] and back["trajectories"] == doc["trajectories"]
    files = list((tmp_path / "cache" / "annotations").glob("*.json"))
    assert len(files) == 1 and "demo" not in files[0].name
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].parent.stat().st_mode & 0o777 == 0o700


def test_annotations_refuse_other_origins_unknown_cases_and_oversize(server):
    path = "/api/case/demo-phantom/annotations"
    assert _post(server, path, b"{}", {"Origin": "http://evil.example"})[0] == 403
    assert _post(server, path, b"{}", {"Content-Type": "text/plain"})[0] == 403
    assert _post(server, "/api/case/no-such-case/annotations", b"{}")[0] == 404
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    conn.putrequest("POST", path, skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{server}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str((64 << 20) + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()
