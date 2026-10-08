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
         "--phantom", str(phantom), "--capsule-dir", str(capsules)],
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
    assert cases[0]["views"] == {"case": True, "imaging": True, "tracts": True, "atlas": True}
    assert [c["title"] for c in cases[1:]] == ["teaching-one"]
    assert cases[1]["views"]["tracts"] is False


def test_demo_imaging_serves_the_phantom_capsule(server):
    data = _poll_ready(server, "/api/case/demo-phantom/imaging")
    assert data["state"] == "ready"
    status, body = _get(server, data["url"])
    assert status == 200 and b"synthetic phantom stand-in" in body


def test_demo_tracts_and_atlas_start_a_tractlab_child(server):
    tracts = _poll_ready(server, "/api/case/demo-phantom/tracts")
    assert tracts["state"] == "ready", tracts
    child_port = int(tracts["url"].split(":")[2].split("/")[0])
    status, body = _get(child_port, "/?teaching=1")
    assert status == 200 and b"SYNTHETIC QA" in body
    atlas = _poll_ready(server, "/api/case/demo-phantom/atlas")
    assert atlas["url"].endswith("/atlas.html?profile=teaching")
    assert _get(child_port, "/atlas.html?profile=teaching")[0] == 200


def test_rejects_foreign_host_and_unknown_routes(server):
    assert _get(server, "/api/cases", host="evil.example")[0] == 403
    assert _get(server, "/api/case/nope/imaging")[0] == 404
    assert _get(server, "/case/demo-phantom/../../etc/passwd")[0] == 404
