"""Eidos workbench: one local window for Capsule imaging and TractLab tract evidence.

The workbench serves a single page on a loopback port. The page holds a case
list and three views per case:

* Case      — one shared viewer: every volume (CT, MR, CTA), mask and tract
              of the case in the same slices and 3D view (scene.html).
* Capsule   — the Capsule viewer2 file with its guided tour.
* Tracts    — the TractLab workstation, served by a TractLab child process.
* Atlas     — the TractLab reference atlas from the same child.

Research and teaching only, not for clinical use.

Case sources are explicit. The built-in demo is a synthetic phantom capsule
plus generated tract curves; they are not one subject. Patient studies come
from the Eidos archive, which holds only what was imported into it (Import
DICOM in the window, or --import). Capsules and TractLab manifests appear when
given with --capsule-dir or --manifest. The workbench never scans any other
folder for patient data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from neuro_workbench.archive import Archive, default_archive_dir

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_REPO = Path(__file__).resolve().parents[4]
DEMO_ID = "demo-phantom"
NIIVUE_JS = "capsule/vendor/niivue-0.69.0.umd.js"
FILE_TYPES = {".tck": "application/octet-stream", ".gz": "application/gzip", ".nii": "application/octet-stream"}
CHILD_START_TIMEOUT_S = 90.0


def default_cache_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "neuroimaging-workbench"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "neuroimaging-workbench"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _short_id(prefix: str, path: Path) -> str:
    return f"{prefix}-{hashlib.sha256(str(path).encode()).hexdigest()[:10]}"


class Job:
    """One lazily started background task with a JSON-visible state."""

    def __init__(self, start):
        self._start = start
        self._lock = threading.Lock()
        self.state = "idle"
        self.url: str | None = None
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None

    def ensure(self) -> dict:
        with self._lock:
            if self.state in ("idle", "error"):
                self.state, self.error = "starting", None
                threading.Thread(target=self._run, daemon=True).start()
            return self.status()

    def status(self) -> dict:
        out = {"state": self.state}
        if self.url:
            out["url"] = self.url
        if self.error:
            out["error"] = self.error
        return out

    def _run(self):
        try:
            self.url = self._start(self)
            self.state = "ready"
        except Exception as exc:  # reported to the page; never a traceback
            self.error = str(exc)[:300] or exc.__class__.__name__
            self.state = "error"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Workbench:
    def __init__(self, repo: Path, cache_dir: Path, phantom: Path | None,
                 capsule_dirs: list[Path], manifests: list[Path], archive: Archive | None = None):
        self.repo = repo
        self.archive = archive
        self.importer = {"state": "idle"}
        self._import_lock = threading.Lock()
        self.cache_dir = cache_dir
        self.phantom = phantom
        self.cases: dict[str, dict] = {}
        self.imaging: dict[str, Job] = {}
        self.tracts: dict[str, Job] = {}
        self.scenes: dict[str, Job] = {}
        self._phantom_lock = threading.Lock()

        self._add_case(DEMO_ID, {
            "title": "Demo case",
            "subtitle": "Phantom CT/MR + synthetic tracts",
            "scope": "Synthetic phantom imaging and generated tract curves. "
                     "They are not one subject and show no real anatomy.",
            "synthetic": True,
        }, imaging=Job(self._build_phantom), tracts=Job(self._start_synthetic_tracts),
            scene=Job(lambda job: self._scene(job, DEMO_ID, [("capsule", self._phantom_file())])))

        for d in capsule_dirs:
            for f in sorted(d.glob("*.capsule.html")):
                path = f.resolve()
                self._add_case(_short_id("capsule", path), {
                    "title": f.name.removesuffix(".capsule.html"),
                    "subtitle": "Capsule file",
                    "scope": "Imaging only. No TractLab manifest is linked to this capsule.",
                    "synthetic": False,
                }, imaging=Job(lambda job, p=path: self._existing_capsule(job, p)),
                    scene=Job(lambda job, p=path, c=_short_id("capsule", path): self._scene(job, c, [("capsule", p)])))

        for m in manifests:
            path = m.resolve()
            self._add_case(_short_id("tracts", path), {
                "title": path.parent.name,
                "subtitle": "TractLab case",
                "scope": "MR inputs and tract banks from one TractLab manifest. No capsule is linked to it.",
                "synthetic": False,
            }, tracts=Job(lambda job, p=path: self._start_manifest_viewer(job, p)),
                scene=Job(lambda job, p=path, c=_short_id("tracts", path): self._scene(job, c, [("tractlab", p)])))

    # Archive --------------------------------------------------------------
    def refresh_archive(self) -> None:
        """Add archive studies imported since the last call to the case list."""
        if not self.archive:
            return
        for st in self.archive.studies():
            if st["id"] in self.cases:
                continue
            date = st["date"]
            when = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date or "no date"
            self._add_case(st["id"], {
                "title": st["patient"] or "Unnamed patient",
                "subtitle": f"{when} · {' '.join(st['modalities']) or 'DICOM'} · {st['description'] or 'Study'}",
                "scope": "Patient study from the Eidos archive. Identifiable data: it stays on this Mac.",
                "synthetic": False, "archive": True,
            }, scene=Job(lambda job, c=st["id"]: self._scene(job, c, [("archive", c)])))

    def case_known(self, case_id: str) -> bool:
        if case_id not in self.cases:
            self.refresh_archive()
        return case_id in self.cases

    def start_import(self, source: str) -> dict:
        if not self.archive:
            raise RuntimeError("the archive is turned off")
        with self._import_lock:
            if self.importer.get("state") == "running":
                return self.importer
            self.importer = {"state": "running", "done": 0, "total": 0}

        def progress(done: int, total: int) -> None:
            self.importer.update(done=done, total=total)

        def run():
            try:
                report = self.archive.import_path(Path(source).expanduser(), progress)
                self.refresh_archive()
                self.importer = {"state": "done", "report": report}
            except Exception as exc:  # the page shows the message; no traceback, no path
                self.importer = {"state": "error", "error": str(exc)[:200] or exc.__class__.__name__}
        threading.Thread(target=run, daemon=True).start()
        return self.importer

    def _add_case(self, case_id, meta, imaging: Job | None = None, tracts: Job | None = None,
                  scene: Job | None = None):
        self.cases[case_id] = {"id": case_id, **meta,
                               "views": {"case": scene is not None, "imaging": imaging is not None,
                                         "tracts": tracts is not None, "atlas": True}}
        if scene:
            self.scenes[case_id] = scene
        if imaging:
            self.imaging[case_id] = imaging
        if tracts:
            self.tracts[case_id] = tracts

    # Shared scene ---------------------------------------------------------
    def _scene(self, job: Job, case_id: str, sources) -> str:
        from neuro_workbench.scene import capsule_scene, merge_scenes, tractlab_scene

        parts = []
        for kind, path in sources:
            if kind == "archive":
                parts.append(self.archive.study_scene(self.archive.study_uid(path)))
                continue
            path = path() if callable(path) else path
            if not Path(path).is_file():
                raise FileNotFoundError(f"the {kind} source file is missing")
            parts.append(capsule_scene(Path(path), self.cache_dir / "scenes") if kind == "capsule"
                         else tractlab_scene(Path(path)))
        scene, files = merge_scenes(parts, self.cases[case_id]["title"])
        for group in ("volumes", "labels", "tracts"):
            for layer in scene[group]:
                layer["url"] = f"/case/{case_id}/file/{layer['id']}"
        job.scene, job.files = scene, files
        return f"/case/{case_id}/scene.html"

    def scene_json(self, case_id: str) -> dict | None:
        job = self.scenes.get(case_id)
        return getattr(job, "scene", None) if job and job.state == "ready" else None

    def scene_file(self, case_id: str, key: str) -> Path | None:
        job = self.scenes.get(case_id)
        files = getattr(job, "files", None) if job and job.state == "ready" else None
        return files.get(key) if files else None

    # Imaging --------------------------------------------------------------
    def capsule_path(self, case_id: str) -> Path | None:
        job = self.imaging.get(case_id)
        return getattr(job, "path", None) if job and job.state == "ready" else None

    def _existing_capsule(self, job: Job, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError("capsule file is missing")
        job.path = path
        return f"/case/{_short_id('capsule', path)}/imaging.html"

    def _build_phantom(self, job: Job) -> str:
        job.path = self._phantom_file()
        return f"/case/{DEMO_ID}/imaging.html"

    def _phantom_file(self) -> Path:
        if self.phantom:
            if not self.phantom.is_file():
                raise FileNotFoundError("the --phantom file is missing")
            return self.phantom
        with self._phantom_lock:
            return self._phantom_build()

    def _phantom_build(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / "phantom.viewer2.capsule.html"
        if not target.is_file():
            tmp = self.cache_dir / f"phantom.{os.getpid()}.{int(time.time())}.capsule.html"
            fixture = self.repo / "capsule" / "viewer2" / "dev_fixture.py"
            proc = subprocess.run([sys.executable, str(fixture), "-o", str(tmp)], cwd=self.repo,
                                  capture_output=True, text=True)
            if proc.returncode != 0 or not tmp.is_file():
                tmp.unlink(missing_ok=True)
                raise RuntimeError("phantom build failed: " + (proc.stderr.strip().splitlines() or ["no output"])[-1])
            os.replace(tmp, target)
        return target

    # Tracts ---------------------------------------------------------------
    def _start_synthetic_tracts(self, job: Job) -> str:
        script = self.repo / "tractlab" / "scripts" / "serve_synthetic.py"
        job.proc = subprocess.Popen([sys.executable, str(script), "--port", "0"], cwd=self.repo / "tractlab",
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        line = job.proc.stdout.readline()
        if not line:
            raise RuntimeError("the synthetic TractLab server exited before it started")
        url = json.loads(line)["url"]
        return url

    def _start_manifest_viewer(self, job: Job, manifest: Path) -> str:
        if not manifest.is_file():
            raise FileNotFoundError("manifest file is missing")
        port = _free_port()
        env = {**os.environ, "PYTHONPATH": str(self.repo / "tractlab" / "src")}
        job.proc = subprocess.Popen(
            [sys.executable, "-m", "tractlab.app_server", "--_viewer-child", "--manifest", str(manifest),
             "--viewer-dir", str(self.repo / "tractlab" / "viewer"), "--port", str(port)],
            cwd=self.repo / "tractlab", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + CHILD_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if job.proc.poll() is not None:
                raise RuntimeError("the TractLab viewer exited before it started")
            if _port_open(port):
                return f"http://127.0.0.1:{port}/"
            time.sleep(0.25)
        raise TimeoutError("the TractLab viewer did not start in time")

    def atlas(self, case_id: str) -> dict:
        source = case_id if case_id in self.tracts else DEMO_ID
        status = self.tracts[source].ensure()
        if status.get("url"):
            parts = urlsplit(status["url"])
            status["url"] = f"{parts.scheme}://{parts.netloc}/atlas.html?profile=teaching"
        return status

    def stop(self):
        for job in self.tracts.values():
            job.stop()


def make_handler(bench: Workbench):
    class Handler(BaseHTTPRequestHandler):
        server_version = "neuro-workbench"

        def log_message(self, fmt, *args):  # keep paths and case names out of logs
            return

        def _host_ok(self) -> bool:
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_POST(self):
            # Same-origin JSON only: a page on another site cannot send this without a preflight.
            port = self.server.server_address[1]
            origin = self.headers.get("Origin")
            if not self._host_ok() or (origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")) \
                    or not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"error": "forbidden"})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 65536)) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            path = urlsplit(self.path).path
            if path == "/api/archive/import":
                source = str(body.get("path") or "").strip()
                if not source:
                    return self._json(400, {"error": "no folder given"})
                try:
                    return self._json(200, bench.start_import(source))
                except RuntimeError as exc:
                    return self._json(409, {"error": str(exc)})
            if path == "/api/archive/choose":
                return self._json(200, choose_folder("zip" if body.get("kind") == "zip" else "folder"))
            return self._json(404, {"error": "not found"})

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            url = urlsplit(self.path)
            path = url.path
            if path == "/api/archive":
                if not bench.archive:
                    return self._json(200, {"enabled": False, "patients": [], "import": bench.importer})
                q = (parse_qs(url.query).get("q") or [""])[0]
                return self._json(200, {"enabled": True, "patients": bench.archive.patients(q),
                                        "import": bench.importer})
            if path == "/api/archive/import":
                return self._json(200, bench.importer)
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/vendor/niivue.js":
                return self._send(200, (bench.repo / NIIVUE_JS).read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/cases":
                bench.refresh_archive()
                return self._json(200, {"cases": list(bench.cases.values()), "archive": bench.archive is not None})
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "case"] and bench.case_known(parts[2]):
                case_id, view = parts[2], parts[3]
                if view == "case" and case_id in bench.scenes:
                    return self._json(200, bench.scenes[case_id].ensure())
                if view == "imaging" and case_id in bench.imaging:
                    return self._json(200, bench.imaging[case_id].ensure())
                if view == "tracts" and case_id in bench.tracts:
                    return self._json(200, bench.tracts[case_id].ensure())
                if view == "atlas":
                    return self._json(200, bench.atlas(case_id))
            if len(parts) == 3 and parts[0] == "case" and parts[2] == "scene.html" and bench.scene_json(parts[1]):
                return self._send(200, (STATIC_DIR / "scene.html").read_bytes(), "text/html; charset=utf-8")
            if len(parts) == 3 and parts[0] == "case" and parts[2] == "scene.json":
                scene = bench.scene_json(parts[1])
                if scene:
                    return self._json(200, scene)
            if len(parts) == 4 and parts[0] == "case" and parts[2] == "file":
                file = bench.scene_file(parts[1], parts[3])
                if file and file.is_file():
                    ctype = FILE_TYPES.get(file.suffix, "application/octet-stream")
                    return self._send(200, file.read_bytes(), ctype)
            if len(parts) == 3 and parts[0] == "case" and parts[2] == "imaging.html":
                file = bench.capsule_path(parts[1])
                if file:
                    return self._send(200, file.read_bytes(), "text/html; charset=utf-8")
            return self._json(404, {"error": "not found"})

    return Handler


def choose_folder(kind: str = "folder") -> dict:
    """Ask for a folder or a .zip with the native macOS dialog. Other systems type the path in the page."""
    if sys.platform != "darwin":
        return {"state": "unsupported"}
    script = ('POSIX path of (choose file with prompt "Import DICOM: choose a .zip" of type {"public.zip-archive"})'
              if kind == "zip" else 'POSIX path of (choose folder with prompt "Import DICOM: choose a folder or a CD")')
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"state": "cancelled"}
    return {"state": "chosen", "path": proc.stdout.strip()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("NEURO_REPO", DEFAULT_REPO)))
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--phantom", type=Path, help="use this prebuilt phantom capsule instead of building one")
    parser.add_argument("--capsule-dir", type=Path, action="append", default=[],
                        help="folder of *.capsule.html files to list (repeatable)")
    parser.add_argument("--manifest", type=Path, action="append", default=[],
                        help="TractLab manifest.json to list (repeatable)")
    parser.add_argument("--archive-dir", type=Path, default=Path(os.environ.get("EIDOS_ARCHIVE", default_archive_dir())),
                        help="patient archive folder (default: the Eidos folder in Application Support)")
    parser.add_argument("--no-archive", action="store_true", help="run without the patient archive")
    parser.add_argument("--import", dest="import_path", type=Path,
                        help="import the DICOM files under this folder into the archive, print counts, and exit")
    args = parser.parse_args(argv)

    archive = None if args.no_archive else Archive(args.archive_dir.expanduser())
    if args.import_path:
        if not archive:
            parser.error("--import needs the archive")
        try:
            print(json.dumps(archive.import_path(args.import_path)), flush=True)
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"eidos: import failed: {exc}")
        return
    bench = Workbench(args.repo.resolve(), args.cache_dir, args.phantom, args.capsule_dir, args.manifest, archive)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bench))
    httpd.daemon_threads = True
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(json.dumps({"url": f"http://127.0.0.1:{httpd.server_address[1]}/",
                      "scope": "research and teaching only; not for clinical use"}), flush=True)
    try:
        stop.wait()
    finally:
        httpd.shutdown()
        httpd.server_close()
        bench.stop()


if __name__ == "__main__":
    main()
