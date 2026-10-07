#!/usr/bin/env python3
"""Launch the TractLab loopback server. Usage: python run.py [port] [--background|--stop]"""

import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tractlab import derivation as dv  # noqa: E402
from tractlab.serve import serve  # noqa: E402
from tractlab.runtime_identity import CaseSourceChanged, RuntimeIdentity  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "cases" / "local-case" / "manifest.json"
DEFAULT_VIEWER = ROOT / "viewer"
READY_TIMEOUT_S = 20.0
READY_POLL_S = 0.1


def _missing_launch_input(manifest_path: str, viewer_dir: str) -> str | None:
    """Return a concise safe startup error without echoing local data paths."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    viewer_index = Path(viewer_dir).expanduser().resolve() / "index.html"
    if not manifest_file.is_file():
        return "required manifest unavailable"
    if not viewer_index.is_file():
        return "required viewer input unavailable: index.html"
    try:
        manifest = json.loads(manifest_file.read_bytes())
    except (OSError, json.JSONDecodeError):
        return "manifest is unreadable"
    if not isinstance(manifest, dict):
        return "manifest is invalid"
    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        return "manifest case identity is invalid"
    root = manifest.get("case_root")
    if not isinstance(root, str) or not os.path.isdir(root):
        return "required case root unavailable"
    try:
        inputs = dv.active_inputs(manifest)
    except dv.DerivationError:
        return "manifest inputs are invalid"
    root_real = os.path.realpath(root)
    for name in ("b0", "mask", "fod"):
        entry = inputs.get(name)
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(rel, str) or not rel or os.path.isabs(rel):
            return f"manifest input is invalid: {name}"
        candidate = os.path.realpath(os.path.join(root_real, rel))
        if not candidate.startswith(root_real + os.sep) or not os.path.isfile(candidate):
            return f"required input unavailable: {name}"
    return None


def _fetch_bootstrap(port: int) -> tuple[str, dict[str, object] | None]:
    """Return response, unavailable, or occupied without disturbing a listener."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/bootstrap", timeout=1.0
        ) as response:
            if response.status != 200:
                return "occupied", None
            payload = json.loads(response.read())
    except urllib.error.HTTPError:
        return "occupied", None
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
            return "unavailable", None
        return "occupied", None
    except (OSError, ValueError):
        return "occupied", None
    return ("response", payload) if isinstance(payload, dict) else ("occupied", None)


def _fetch_stop_identity(port: int) -> tuple[str, dict[str, object] | None]:
    """Read the boot identity from an owned stale server without making it reusable."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/bootstrap", timeout=1.0
        ) as response:
            if response.status != 200:
                return "occupied", None
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code != 409:
            return "occupied", None
        try:
            payload = json.loads(error.read())
        except (OSError, ValueError):
            return "occupied", None
        return ("restart-required", payload) if isinstance(payload, dict) else ("occupied", None)
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
            return "unavailable", None
        return "occupied", None
    except (OSError, ValueError):
        return "occupied", None
    return ("response", payload) if isinstance(payload, dict) else ("occupied", None)


def _matches_identity(payload: dict[str, object], identity: RuntimeIdentity) -> bool:
    return (
        payload.get("runtimeStatus") == "ok"
        and payload.get("viewerProtocol") == 1
        and payload.get("buildId") == identity.build_id
        and payload.get("caseId") == identity.case_id
        and payload.get("manifestId") == identity.manifest_id
        and payload.get("caseSourceHash") == identity.case_source_hash
    )


def _server_state(port: int, identity: RuntimeIdentity) -> str:
    """Return available, matching, or occupied without touching another process."""
    status, payload = _fetch_bootstrap(port)
    if status == "unavailable":
        return "available"
    if status == "response" and payload is not None and _matches_identity(payload, identity):
        return "matching"
    return "occupied"


def _runtime_dir() -> Path:
    configured = os.environ.get("TRACTLAB_RUNTIME_DIR")
    base = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "tractlab-runtime"
    )
    return base.resolve()


def _runtime_paths(port: int) -> tuple[Path, Path]:
    runtime_dir = _runtime_dir()
    return (
        runtime_dir / f"preview-{port}.json",
        runtime_dir / f"preview-{port}.log",
    )


def _prepare_runtime_dir() -> Path | None:
    runtime_dir = _runtime_dir()
    try:
        runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return None
    return runtime_dir


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_state(path: Path, service, port: int) -> dict[str, object]:
    metadata = service.runtime_identity.public_metadata()
    payload = {
        "pid": os.getpid(),
        "port": port,
        "bootId": metadata["bootId"],
        "buildId": metadata["buildId"],
        "caseId": service.case_id,
        "manifestId": metadata["manifestId"],
        "caseSourceHash": metadata["caseSourceHash"],
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return payload


def _remove_owned_state(path: Path, pid: int, boot_id: object) -> None:
    state = _read_state(path)
    if state is None or state.get("pid") != pid or state.get("bootId") != boot_id:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _startup_failure(port: int, error: BaseException) -> int:
    if isinstance(error, OSError) and error.errno == errno.EADDRINUSE:
        print(
            f"TractLab cannot use port {port}: it became occupied; choose an unused port or stop it yourself.",
            file=sys.stderr,
        )
    else:
        print(
            "TractLab cannot start: could not initialize required local inputs; "
            "check the manifest and input availability.",
            file=sys.stderr,
        )
    return 2


def _serve_foreground(
    port: int,
    manifest: str,
    viewer: str,
    state_path: Path | None = None,
) -> int:
    try:
        httpd, service = serve(manifest, viewer, port=port)
    except (CaseSourceChanged, FileNotFoundError, OSError, ValueError, KeyError) as error:
        return _startup_failure(port, error)

    state = None
    previous_sigterm = None
    try:
        if state_path is not None:
            try:
                state = _write_state(state_path, service, port)
            except OSError:
                httpd.server_close()
                print("TractLab cannot start background preview: runtime state is unavailable.", file=sys.stderr)
                return 2

            def request_shutdown(_signum, _frame):
                threading.Thread(target=httpd.shutdown, daemon=True).start()

            previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)

        print(f"TractLab on http://127.0.0.1:{port}  case={service.case_id}")
        print(f"  gridId={service.grid_id[:12]}  volumeId={service.volume_id[:12]}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
    finally:
        close = getattr(httpd, "server_close", None)
        if close is not None:
            close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if state is not None:
            _remove_owned_state(state_path, os.getpid(), state["bootId"])
    return 0


def _matches_state(payload: dict[str, object], state: dict[str, object]) -> bool:
    return all(
        payload.get(key) == state.get(key)
        for key in ("bootId", "buildId", "caseId", "manifestId", "caseSourceHash")
    )


def _matches_restart_state(payload: dict[str, object], state: dict[str, object]) -> bool:
    """Only the recorded boot may authorize stopping a restart-required server."""
    return (
        payload.get("code") == "restart_required"
        and payload.get("runtimeStatus") == "restart-required"
        and payload.get("bootId") == state.get("bootId")
        and payload.get("bootBuildId") == state.get("buildId")
        and payload.get("bootCaseSourceHash") == state.get("caseSourceHash")
        and payload.get("caseId") == state.get("caseId")
        and payload.get("manifestId") == state.get("manifestId")
    )


def _wait_for_background_ready(port: int, identity: RuntimeIdentity, process, state_path: Path) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        status, payload = _fetch_bootstrap(port)
        state = _read_state(state_path)
        if (
            status == "response"
            and payload is not None
            and state is not None
            and state.get("pid") == process.pid
            and _matches_identity(payload, identity)
            and _matches_state(payload, state)
        ):
            return True
        if process.poll() is not None:
            return False
        time.sleep(READY_POLL_S)
    return False


def _stop_owned_process(process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _start_background(port: int, identity: RuntimeIdentity) -> int:
    runtime_dir = _prepare_runtime_dir()
    if runtime_dir is None:
        print("TractLab cannot start background preview: runtime state is unavailable.", file=sys.stderr)
        return 2
    state_path, log_path = _runtime_paths(port)
    child_env = os.environ.copy()
    child_env["TRACTLAB_RUNTIME_DIR"] = str(runtime_dir)
    command = [sys.executable, str(ROOT / "run.py"), str(port), "--serve-child"]
    try:
        with log_path.open("ab") as log:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError:
        print("TractLab cannot start background preview: process launch failed.", file=sys.stderr)
        return 2
    if _wait_for_background_ready(port, identity, process, state_path):
        print(f"TractLab ready at http://127.0.0.1:{port} (background)")
        return 0
    _stop_owned_process(process)
    state = _read_state(state_path)
    _remove_owned_state(state_path, process.pid, state.get("bootId") if state else None)
    print("TractLab background preview did not become ready; inspect the runtime log.", file=sys.stderr)
    return 2


def _stop_background(port: int) -> int:
    state_path, _log_path = _runtime_paths(port)
    state = _read_state(state_path)
    if state is None:
        print(f"TractLab has no managed background preview recorded for port {port}.", file=sys.stderr)
        return 2
    pid = state.get("pid")
    if state.get("port") != port or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        print("TractLab will not stop: managed preview state is invalid.", file=sys.stderr)
        return 2
    status, payload = _fetch_stop_identity(port)
    if status == "unavailable":
        _remove_owned_state(state_path, pid, state.get("bootId"))
        print(f"TractLab background preview already stopped at port {port}.")
        return 0
    matches_owned_listener = (
        payload is not None
        and (
            (status == "response" and _matches_state(payload, state))
            or (status == "restart-required" and _matches_restart_state(payload, state))
        )
    )
    if not matches_owned_listener:
        print(
            f"TractLab will not stop port {port}: the listener does not match this managed preview.",
            file=sys.stderr,
        )
        return 2
    try:
        os.kill(pid, 0)
    except OSError:
        print(
            f"TractLab will not stop port {port}: the recorded process is unavailable while the listener remains active.",
            file=sys.stderr,
        )
        return 2
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        print(f"TractLab could not stop its managed preview on port {port}.", file=sys.stderr)
        return 2
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        status, _payload = _fetch_bootstrap(port)
        if status == "unavailable":
            _remove_owned_state(state_path, pid, state.get("bootId"))
            print(f"TractLab background preview stopped at port {port}.")
            return 0
        time.sleep(READY_POLL_S)
    print(f"TractLab could not stop its managed preview on port {port}.", file=sys.stderr)
    return 2


def _parse_arguments(args: list[str]) -> tuple[int | None, str | None, str | None]:
    port_text = None
    mode = "foreground"
    for argument in args:
        if argument in ("--background", "--stop", "--serve-child"):
            requested_mode = {
                "--background": "background",
                "--stop": "stop",
                "--serve-child": "serve-child",
            }[argument]
            if mode != "foreground":
                return None, None, "choose one launch mode"
            mode = requested_mode
        elif argument.startswith("-"):
            return None, None, "unknown launch option"
        elif port_text is None:
            port_text = argument
        else:
            return None, None, "expected one port"
    try:
        port = 18993 if port_text is None else int(port_text)
    except ValueError:
        return None, None, "port must be a number"
    if not 1 <= port <= 65535:
        return None, None, "port must be between 1 and 65535"
    return port, mode, None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    port, mode, argument_error = _parse_arguments(args)
    if argument_error is not None:
        print(f"TractLab cannot start: {argument_error}.", file=sys.stderr)
        return 2
    assert port is not None and mode is not None
    if mode == "stop":
        return _stop_background(port)

    manifest = os.environ.get("TRACTLAB_MANIFEST", str(DEFAULT_MANIFEST))
    viewer = os.environ.get("TRACTLAB_VIEWER", str(DEFAULT_VIEWER))
    missing = _missing_launch_input(manifest, viewer)
    if missing is not None:
        print(f"TractLab cannot start: {missing}", file=sys.stderr)
        return 2
    try:
        identity = RuntimeIdentity(viewer, manifest_path=manifest)
    except (OSError, ValueError):
        print("TractLab cannot start: manifest is unreadable", file=sys.stderr)
        return 2

    if mode == "serve-child":
        state_path, _log_path = _runtime_paths(port)
        return _serve_foreground(port, manifest, viewer, state_path)

    state = _server_state(port, identity)
    if state == "matching":
        print(f"TractLab already ready at http://127.0.0.1:{port}  build={identity.build_id[:12]}")
        return 0
    if state == "occupied":
        print(
            f"TractLab cannot use port {port}: it is occupied by an incompatible or stale service. "
            "Choose an unused port or stop it yourself; this launcher will not kill it.",
            file=sys.stderr,
        )
        return 2
    if mode == "background":
        return _start_background(port, identity)
    return _serve_foreground(port, manifest, viewer)


if __name__ == "__main__":
    result = main()
    if result:
        raise SystemExit(result)
