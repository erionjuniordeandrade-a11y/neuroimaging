"""Contract tests for the local TractLab launcher server."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tractlab import app_server


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def live_server(tmp_path: Path):
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    server = app_server.create_server(port=0, cases_root=cases_root, repo_root=REPO_ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server, cases_root
    finally:
        server.shutdown()
        thread.join(timeout=3)
        app_server.close_viewers(server)
        server.server_close()


@pytest.fixture
def ingest_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    _fake_commands(tmp_path, monkeypatch)
    server = app_server.create_server(port=0, cases_root=cases_root, repo_root=REPO_ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base_url, server, cases_root
    finally:
        server.shutdown()
        thread.join(timeout=3)
        app_server.close_viewers(server)
        server.server_close()


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        assert response.status == 200
        return json.loads(response.read())


def _post_json(url: str, value: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(value).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        assert response.status == 200
        return json.loads(response.read())


def _seed_case(cases_root: Path, case_id: str, *, label: str, state: str) -> Path:
    case_root = cases_root / case_id
    case_root.mkdir()
    (case_root / "manifest.json").write_text(
        json.dumps({"case_id": case_id, "case_root": "."}), encoding="utf-8"
    )
    (case_root / "case.json").write_text(
        json.dumps(
            {
                "schema": "tractlab.case/1",
                "case_id": case_id,
                "label": label,
                "created_utc": "2026-09-26T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (case_root / "status.json").write_text(
        json.dumps(
            {
                "schema": "tractlab.status/1",
                "state": state,
                "stage": "t1reg",
                "stages": [
                    {
                        "name": "ss3t",
                        "state": "done",
                        "started_utc": "2026-09-26T12:00:00Z",
                        "ended_utc": "2026-09-26T12:03:00Z",
                    },
                    {
                        "name": "t1reg",
                        "state": "running",
                        "started_utc": "2026-09-26T12:03:00Z",
                        "ended_utc": None,
                    },
                ],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return case_root


def _fake_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    marker = tmp_path / "pipeline-start.json"
    ingest_script = tmp_path / "fake_ingest.py"
    ingest_script.write_text(
        """import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dicom_dir')
parser.add_argument('--cases-root', required=True)
parser.add_argument('--label', default='')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
case_id = 'case-5eed1234'
case_root = Path(args.cases_root) / case_id
case_root.mkdir(parents=True)
(case_root / 'manifest.json').write_text(json.dumps({'case_id': case_id, 'case_root': '.'}))
(case_root / 'case.json').write_text(json.dumps({
    'schema': 'tractlab.case/1', 'case_id': case_id,
    'label': args.label, 'created_utc': '2026-09-26T12:00:00Z'
}))
(case_root / 'status.json').write_text(json.dumps({
    'schema': 'tractlab.status/1', 'state': 'running', 'stage': 'ss3t',
    'stages': [{'name': 'ss3t', 'state': 'running',
                'started_utc': '2026-09-26T12:00:00Z', 'ended_utc': None}],
    'error': None
}))
print(json.dumps({'ok': True, 'case_id': case_id, 'case_root': str(case_root),
                  'refusals': [], 'warnings': []}))
""",
        encoding="utf-8",
    )
    pipeline_script = tmp_path / "fake_pipeline.py"
    pipeline_script.write_text(
        """import json, os, sys
from pathlib import Path

case_root = Path(sys.argv[1])
Path(os.environ['FAKE_PIPELINE_MARKER']).write_text(json.dumps({
    'case_root': str(case_root), 'pid': os.getpid(), 'sid': os.getsid(0),
    'stdin_isatty': os.isatty(0)
}))
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "TRACTLAB_INGEST_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(ingest_script))}"
    )
    monkeypatch.setenv(
        "TRACTLAB_PIPELINE_CMD", f"{shlex.quote(sys.executable)} {shlex.quote(str(pipeline_script))}"
    )
    monkeypatch.setenv("FAKE_PIPELINE_MARKER", str(marker))
    return ingest_script, marker


def test_home_page_has_research_disclaimer_and_demo(live_server):
    base_url, _server, _cases_root = live_server
    with urlopen(base_url + "/", timeout=5) as response:
        page = response.read().decode("utf-8")
    assert response.status == 200
    assert "Research/preview only — not navigation." in page
    assert "Demo (CC0, Leipzig sub-010005)" in page


def test_cases_api_lists_demo_first_and_stage_status(live_server):
    base_url, _server, cases_root = live_server
    _seed_case(cases_root, "case-0123abcd", label="Synthetic control", state="running")

    cases = _get_json(base_url + "/api/cases")["cases"]
    assert [case["case_id"] for case in cases[:2]] == [
        "demo-leipzig-sub-010005",
        "case-0123abcd",
    ]
    demo = cases[0]
    assert demo["state"] == "done"
    assert demo["open_enabled"] is True
    fake = cases[1]
    assert fake["label"] == "Synthetic control"
    assert fake["state"] == "running"
    stages = {stage["name"]: stage for stage in fake["stages"]}
    assert stages["ss3t"]["state"] == "done"
    assert stages["ss3t"]["elapsed_seconds"] == 180
    assert stages["t1reg"]["state"] == "running"
    assert stages["t1reg"]["elapsed_seconds"] >= 0


def test_ingest_calls_cli_starts_detached_pipeline_and_does_not_store_source_path(
    ingest_server, tmp_path: Path
):
    base_url, _server, cases_root = ingest_server
    dicom_dir = str(tmp_path / "synthetic-dicom-source-do-not-store")

    result = _post_json(base_url + "/api/ingest", {"dicom_dir": dicom_dir, "label": "Synthetic"})
    assert result["ok"] is True
    assert result["case_id"] == "case-5eed1234"

    marker = Path(os.environ["FAKE_PIPELINE_MARKER"])
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists(), "detached pipeline command did not run"
    pipeline = json.loads(marker.read_text(encoding="utf-8"))
    assert pipeline["sid"] == pipeline["pid"]
    assert pipeline["stdin_isatty"] is False

    for path in cases_root.rglob("*"):
        assert dicom_dir not in str(path)
        if path.is_file():
            assert dicom_dir.encode("utf-8") not in path.read_bytes()


@pytest.mark.skipif(
    not app_server.DEMO_MANIFEST.is_file(),
    reason="the local demo case is not checked out (cases/ is excluded from the public repo)",
)
def test_open_demo_returns_live_viewer_url_and_child_is_stopped(live_server):
    base_url, server, _cases_root = live_server
    result = _post_json(base_url + "/api/open", {"case_id": "demo-leipzig-sub-010005"})
    with urlopen(result["url"], timeout=10) as response:
        assert response.status == 200
    child = server.viewer_processes["demo-leipzig-sub-010005"]
    assert child.poll() is None
    app_server.stop_viewer(server, "demo-leipzig-sub-010005")
    assert child.poll() is not None


def test_rejects_non_loopback_host(live_server):
    base_url, _server, _cases_root = live_server
    request = Request(base_url + "/", headers={"Host": "not-loopback.invalid"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert 400 <= error.value.code < 500
