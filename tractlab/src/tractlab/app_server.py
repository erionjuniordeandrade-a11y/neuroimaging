"""Loopback-only launcher server for the personal TractLab macOS app."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ID = "demo-leipzig-sub-010005"
DEMO_LABEL = "Demo (CC0, Leipzig sub-010005)"
DEMO_ROOT = REPO_ROOT / "cases" / DEMO_ID
DEMO_MANIFEST = DEMO_ROOT / "manifest.json"
VIEWER_DIR = (REPO_ROOT / "viewer").resolve()
STAGES = ("ss3t", "t1reg", "banks", "scalar_maps")
VALID_STAGE_STATES = {"pending", "running", "done", "failed"}
MAX_REQUEST_BYTES = 1024 * 1024


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _child_env() -> dict[str, str]:
    """Child processes import tractlab even when this server was started without PYTHONPATH."""
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH", "")
    if src not in current.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, current]))
    return env


def _parse_command(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        command = shlex.split(raw)
    except ValueError:
        raise ValueError(f"{name} must contain a valid command") from None
    if not command:
        raise ValueError(f"{name} must contain a valid command")
    return command


def _configured_cases_root(value: str | Path | None, repo_root: Path) -> Path:
    configured = value or os.environ.get("TRACTLAB_CASES_ROOT")
    if configured:
        cases_root = Path(configured).expanduser().resolve()
    else:
        cases_root = (
            Path.home()
            / "Library"
            / "Application Support"
            / "TractLab"
            / "cases"
        ).resolve()
    if _is_within(cases_root, repo_root):
        raise ValueError("cases root must be outside the repository")
    return cases_root


def _safe_file(path: Path, parent: Path) -> Path | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        resolved_parent = parent.resolve()
        resolved = path.resolve()
        if not _is_within(resolved, resolved_parent):
            return None
        return resolved
    except OSError:
        return None


def _read_json(path: Path, parent: Path) -> dict[str, Any] | None:
    safe_path = _safe_file(path, parent)
    if safe_path is None:
        return None
    try:
        with safe_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        return result if isinstance(result, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _elapsed_seconds(stage: dict[str, Any], now: datetime) -> int | None:
    started = _parse_timestamp(stage.get("started_utc"))
    if started is None:
        return None
    ended = _parse_timestamp(stage.get("ended_utc"))
    duration = ((ended or now) - started).total_seconds()
    return max(0, int(duration))


def _case_record(
    case_id: str,
    label: str,
    case_root: Path,
    manifest: Path,
    *,
    is_demo: bool,
) -> dict[str, Any]:
    case_data = _read_json(case_root / "case.json", case_root) or {}
    status_data = _read_json(case_root / "status.json", case_root) or {}
    created = case_data.get("created_utc")
    if not isinstance(created, str) or not created:
        created = None

    by_name: dict[str, dict[str, Any]] = {}
    raw_stages = status_data.get("stages")
    if isinstance(raw_stages, list):
        for raw_stage in raw_stages:
            if isinstance(raw_stage, dict) and raw_stage.get("name") in STAGES:
                by_name[raw_stage["name"]] = raw_stage
    now = datetime.now(timezone.utc)
    stages = []
    for name in STAGES:
        source = by_name.get(name, {})
        state = source.get("state")
        if state not in VALID_STAGE_STATES:
            # The demo was built by the demo scripts, which write no status.json.
            state = "done" if is_demo and not status_data else "pending"
        stages.append(
            {
                "name": name,
                "state": state,
                "elapsed_seconds": _elapsed_seconds(source, now),
            }
        )

    state = status_data.get("state")
    if state not in VALID_STAGE_STATES:
        state = "done" if is_demo and not status_data else "pending"
    return {
        "case_id": case_id,
        "label": label,
        "created_utc": created,
        "state": state,
        "stages": stages,
        "is_demo": is_demo,
        "open_enabled": is_demo or state == "done",
        "_case_root": case_root,
        "_manifest": manifest,
    }


def _collect_cases(server: "AppServer") -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        _case_record(
            DEMO_ID,
            DEMO_LABEL,
            DEMO_ROOT,
            DEMO_MANIFEST,
            is_demo=True,
        )
    ]
    root = server.cases_root
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name.casefold())
    except FileNotFoundError:
        entries = []
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_dir():
                continue
            case_root = entry.resolve()
            if case_root.parent != root:
                continue
            case_id = entry.name
            if not case_id or len(case_id) > 80 or any(
                not (char.isascii() and (char.isalnum() or char in "-_."))
                for char in case_id
            ):
                continue
            manifest = _safe_file(case_root / "manifest.json", case_root)
            if manifest is None:
                continue
            case_data = _read_json(case_root / "case.json", case_root) or {}
            stored_id = case_data.get("case_id")
            if stored_id is not None and stored_id != case_id:
                continue
            label = case_data.get("label")
            if not isinstance(label, str) or not label.strip():
                label = case_id
            cases.append(
                _case_record(case_id, label, case_root, manifest, is_demo=False)
            )
        except OSError:
            continue
    return cases


def _public_case(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _loopback_host(header: str | None) -> bool:
    if not header:
        return False
    try:
        parsed = urlsplit("//" + header)
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.path or parsed.query or parsed.fragment:
            return False
        host = parsed.hostname
        _ = parsed.port  # Validate the optional port syntax.
        if not host:
            return False
        if host.casefold().rstrip(".") == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        *,
        repo_root: Path,
        cases_root: Path,
        ingest_command: list[str],
        pipeline_command: list[str],
    ):
        self.repo_root = repo_root
        self.cases_root = cases_root
        self.ingest_command = ingest_command
        self.pipeline_command = pipeline_command
        self.viewer_processes: dict[str, subprocess.Popen[bytes]] = {}
        self.viewer_lock = threading.Lock()
        super().__init__(address, AppRequestHandler)


def create_server(
    *,
    port: int = 0,
    cases_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> AppServer:
    repo = Path(repo_root or REPO_ROOT).expanduser().resolve()
    cases = _configured_cases_root(cases_root, repo)
    cases.mkdir(mode=0o700, parents=True, exist_ok=True)
    ingest_command = _parse_command(
        "TRACTLAB_INGEST_CMD", [sys.executable, "-m", "tractlab.ingest"]
    )
    pipeline_command = _parse_command(
        "TRACTLAB_PIPELINE_CMD", [str(repo / "scripts" / "pipeline" / "run_case.sh")]
    )
    return AppServer(
        ("127.0.0.1", port),
        repo_root=repo,
        cases_root=cases,
        ingest_command=ingest_command,
        pipeline_command=pipeline_command,
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _viewer_url_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def stop_viewer(server: AppServer, case_id: str) -> None:
    with server.viewer_lock:
        process = server.viewer_processes.pop(case_id, None)
    if process is not None:
        _stop_process(process)


def close_viewers(server: AppServer) -> None:
    with server.viewer_lock:
        processes = list(server.viewer_processes.values())
        server.viewer_processes.clear()
    for process in processes:
        _stop_process(process)


def _open_viewer(server: AppServer, record: dict[str, Any]) -> str:
    case_id = record["case_id"]
    with server.viewer_lock:
        process = server.viewer_processes.get(case_id)
        if process is not None and process.poll() is None:
            port = getattr(process, "tractlab_port", None)
            if isinstance(port, int):
                existing_url = f"http://127.0.0.1:{port}/"
                if _viewer_url_is_ready(existing_url):
                    return existing_url
        manifest = Path(record["_manifest"]).resolve()
        viewer_dir = VIEWER_DIR.resolve()
        port = _free_loopback_port()
        command = [
            sys.executable,
            "-m",
            "tractlab.app_server",
            "--_viewer-child",
            "--manifest",
            str(manifest),
            "--viewer-dir",
            str(viewer_dir),
            "--port",
            str(port),
        ]
        process = subprocess.Popen(
            command,
            cwd=server.repo_root,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.tractlab_port = port  # type: ignore[attr-defined]
        server.viewer_processes[case_id] = process

    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if _viewer_url_is_ready(url):
            return url
        time.sleep(0.1)
    stop_viewer(server, case_id)
    raise RuntimeError("viewer failed to become ready")


def _tail_log(case_root: Path, status_data: dict[str, Any]) -> tuple[str, list[str]]:
    stage = status_data.get("stage")
    if stage not in STAGES:
        active = next(
            (
                item.get("name")
                for item in status_data.get("stages", [])
                if isinstance(item, dict) and item.get("state") in {"running", "failed"}
            ),
            None,
        )
        stage = active if active in STAGES else "ss3t"
    log_path = _safe_file(case_root / "work" / f"{stage}.log", case_root)
    if log_path is None:
        return stage, []
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - 1024 * 1024))
            content = handle.read().decode("utf-8", errors="replace")
        return stage, content.splitlines()[-200:]
    except OSError:
        return stage, []


class AppRequestHandler(BaseHTTPRequestHandler):
    server: AppServer
    server_version = "TractLab"
    sys_version = ""

    def log_message(self, _format: str, *args: Any) -> None:
        # Request bodies can contain the transient source DICOM path.
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _host_allowed(self) -> bool:
        if _loopback_host(self.headers.get("Host")):
            return True
        self._json(403, {"error": "Host must be loopback."})
        return False

    def _request_json(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError:
            self._json(400, {"error": "Invalid request length."})
            return None
        if length < 0:
            self._json(400, {"error": "Invalid request length."})
            return None
        if length > MAX_REQUEST_BYTES:
            self._json(413, {"error": "Request is too large."})
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeError, ValueError):
            self._json(400, {"error": "Request must contain JSON."})
            return None
        if not isinstance(value, dict):
            self._json(400, {"error": "Request must contain a JSON object."})
            return None
        return value

    def _case(self, case_id: str) -> dict[str, Any] | None:
        return next((item for item in _collect_cases(self.server) if item["case_id"] == case_id), None)

    def do_GET(self) -> None:
        if not self._host_allowed():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            page = self.server.repo_root / "viewer" / "app_home.html"
            safe_page = _safe_file(page, self.server.repo_root / "viewer")
            if safe_page is None:
                self._json(500, {"error": "App home page is unavailable."})
                return
            try:
                self._send(200, safe_page.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._json(500, {"error": "App home page is unavailable."})
            return
        if parsed.path == "/api/cases":
            try:
                cases = [_public_case(item) for item in _collect_cases(self.server)]
            except PermissionError:
                self._json(403, {"error": "Cases folder permission denied."})
                return
            self._json(200, {"cases": cases})
            return
        if parsed.path == "/api/log":
            query = parse_qs(parsed.query, keep_blank_values=True)
            case_id = query.get("case_id", [""])[0]
            record = self._case(case_id)
            if record is None:
                self._json(404, {"error": "Case not found."})
                return
            status_data = _read_json(Path(record["_case_root"]) / "status.json", Path(record["_case_root"])) or {}
            stage, lines = _tail_log(Path(record["_case_root"]), status_data)
            self._json(200, {"case_id": case_id, "stage": stage, "lines": lines})
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self._host_allowed():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/ingest":
            self._post_ingest()
            return
        if parsed.path == "/api/open":
            self._post_open()
            return
        self._json(404, {"error": "Not found."})

    def _post_ingest(self) -> None:
        value = self._request_json()
        if value is None:
            return
        dicom_dir = value.get("dicom_dir")
        label = value.get("label", "")
        if (
            not isinstance(dicom_dir, str)
            or not dicom_dir
            or "\x00" in dicom_dir
            or not isinstance(label, str)
            or len(label) > 160
        ):
            self._json(400, {"error": "A DICOM folder and label are required."})
            return

        command = [
            *self.server.ingest_command,
            dicom_dir,
            "--cases-root",
            str(self.server.cases_root),
            "--label",
            label,
            "--json",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.server.repo_root,
                env=_child_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except (OSError, UnicodeError):
            self._json(502, {"ok": False, "error": "Ingest command could not be started."})
            return
        try:
            ingest_result = json.loads(result.stdout)
        except (TypeError, ValueError):
            self._json(502, {"ok": False, "error": "Ingest command returned invalid JSON."})
            return
        if not isinstance(ingest_result, dict):
            self._json(502, {"ok": False, "error": "Ingest command returned invalid JSON."})
            return
        if result.returncode != 0 or not ingest_result.get("ok"):
            self._json(422, ingest_result)
            return

        case_id = ingest_result.get("case_id")
        case_root_value = ingest_result.get("case_root")
        if not isinstance(case_id, str) or not isinstance(case_root_value, str):
            self._json(502, {"ok": False, "error": "Ingest command returned an invalid case."})
            return
        case_root = Path(case_root_value).expanduser().resolve()
        if (
            not _is_within(case_root, self.server.cases_root)
            or case_root.parent != self.server.cases_root
            or case_root.name != case_id
        ):
            self._json(502, {"ok": False, "error": "Ingest command returned an invalid case."})
            return

        try:
            subprocess.Popen(
                [*self.server.pipeline_command, str(case_root)],
                cwd=self.server.repo_root,
                env=_child_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            self._json(
                502,
                {
                    "ok": False,
                    "case_id": case_id,
                    "error": "Case was ingested, but the pipeline could not be started.",
                },
            )
            return
        self._json(200, ingest_result)

    def _post_open(self) -> None:
        value = self._request_json()
        if value is None:
            return
        case_id = value.get("case_id")
        if not isinstance(case_id, str):
            self._json(400, {"error": "A case id is required."})
            return
        record = self._case(case_id)
        if record is None:
            self._json(404, {"error": "Case not found."})
            return
        if not record["open_enabled"]:
            self._json(409, {"error": "Case is not complete yet."})
            return
        manifest = Path(record["_manifest"])
        if _safe_file(manifest, Path(record["_case_root"])) is None:
            self._json(404, {"error": "Case manifest is unavailable."})
            return
        try:
            url = _open_viewer(self.server, record)
        except (OSError, RuntimeError):
            self._json(502, {"error": "Viewer failed to start."})
            return
        self._json(200, {"url": url})


def _run_viewer_child(manifest: str, viewer_dir: str, port: int) -> None:
    from tractlab.serve import serve

    httpd, _service = serve(manifest, viewer_dir, host="127.0.0.1", port=port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--cases-root")
    parser.add_argument("--_viewer-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", help=argparse.SUPPRESS)
    parser.add_argument("--viewer-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._viewer_child:
        if not args.manifest or not args.viewer_dir:
            parser.error("viewer child requires a manifest and viewer directory")
        _run_viewer_child(args.manifest, str(Path(args.viewer_dir).resolve()), args.port)
        return 0

    try:
        server = create_server(port=args.port, cases_root=args.cases_root)
    except (OSError, ValueError) as exc:
        print(f"Unable to start TractLab app server: {exc}", file=sys.stderr, flush=True)
        return 2

    print(f"LISTENING {server.server_address[1]}", flush=True)

    def stop_on_sigterm(_signum: int, _frame: Any) -> None:
        raise SystemExit(0)

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        close_viewers(server)
        server.server_close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
