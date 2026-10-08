"""Click-to-segment with nnInteractive, run on one volume of an open case.

The model runs in its own Python (a separate virtual environment with torch),
through the Capsule runner script, so the app itself never imports torch.
Exchange files live in a private temporary folder that is removed afterwards.
The result is a mask on that volume's voxel grid; the page adds it as a new,
unreviewed structure. Messages never carry paths or patient text.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import nibabel as nib
import numpy as np

MAX_POINTS = 32
TIMEOUT_S = 600
_lock = threading.Lock()


def python_path() -> Path:
    value = os.environ.get("CASE_CAPSULE_NNINTERACTIVE_PYTHON")
    return Path(value).expanduser() if value else Path.home() / ".local/share/case-capsule/nninteractive-venv/bin/python"


def runner_path() -> Path | None:
    spec = importlib.util.find_spec("capsule")
    if not spec or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "nninteractive_runner.py"
    return path if path.is_file() else None


def status() -> dict:
    if not python_path().is_file():
        return {"available": False, "reason": "the nnInteractive environment is not installed"}
    if runner_path() is None:
        return {"available": False, "reason": "the Capsule nnInteractive runner is missing"}
    return {"available": True, "device": "mps" if sys.platform == "darwin" else "cpu"}


def _points(volume: nib.Nifti1Image, points: list) -> list[dict]:
    if not isinstance(points, list) or not points:
        raise ValueError("add at least one point")
    if len(points) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points")
    inverse = np.linalg.inv(volume.affine)
    shape = volume.shape[:3]
    out = []
    for n, p in enumerate(points, 1):
        try:
            ras = np.asarray(p["ras"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"point {n} has no RAS position") from None
        if ras.shape != (3,) or not np.isfinite(ras).all():
            raise ValueError(f"point {n} has no RAS position")
        ijk = np.floor((inverse @ np.append(ras, 1.0))[:3] + 0.5).astype(int)
        if any(not 0 <= ijk[a] < shape[a] for a in range(3)):
            raise ValueError(f"point {n} is outside the image")
        # The runner indexes the array it is given; the array here is in NIfTI (i, j, k) order.
        out.append({"kji": [int(v) for v in ijk], "positive": bool(p.get("positive", True))})
    if not any(p["positive"] for p in out):
        raise ValueError("add at least one point inside the structure")
    return out


def segment(volume_file: Path, points: list, run=subprocess.run) -> dict:
    """Run nnInteractive once. Returns the mask's voxel indices (NIfTI order) and provenance."""
    st = status()
    if not st["available"]:
        raise RuntimeError(st["reason"])
    volume = nib.load(str(volume_file))
    prompts = _points(volume, points)
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another AI segmentation is running; wait for it to finish")
    try:
        with tempfile.TemporaryDirectory(prefix="eidos-segment-") as work:
            work = Path(work)
            os.chmod(work, 0o700)
            np.savez(work / "input.npz", image=np.asarray(volume.dataobj, dtype=np.float32))
            (work / "request.json").write_text(json.dumps({"prompts": [
                {"id": "p1", "name": "structure", "image_key": "image", "points_kji": prompts}]}))
            (work / "results").mkdir()
            cmd = [str(python_path()), str(runner_path()), "--input", str(work / "input.npz"),
                   "--request", str(work / "request.json"), "--output-dir", str(work / "results"),
                   "--result", str(work / "result.json"), "--device", st["device"]]
            try:
                done = run(cmd, capture_output=True, text=True, check=False, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                raise RuntimeError("nnInteractive did not finish in time") from None
            if done.returncode:
                tail = [ln for ln in (done.stderr or "").splitlines() if ln.strip()]
                kind = tail[-1].split(":")[0][:80] if tail else "no output"
                raise RuntimeError(f"nnInteractive stopped (exit {done.returncode}, {kind})")
            info = json.loads((work / "result.json").read_text())
            with np.load(work / "results" / "mask_0000.npz", allow_pickle=False) as data:
                mask = data["mask"]
    finally:
        _lock.release()
    if mask.shape != volume.shape[:3]:
        raise RuntimeError("nnInteractive returned the wrong grid shape")
    # Flat index i + nx*(j + ny*k), the order NiiVue stores a volume in.
    flat = np.flatnonzero(mask.transpose(2, 1, 0).ravel()).astype(np.uint32)
    voxel_ml = float(abs(np.linalg.det(volume.affine[:3, :3]))) / 1000.0
    return {
        "dims": [int(v) for v in volume.shape[:3]], "affine": volume.affine.tolist(),
        "count": int(flat.size), "volume_ml": round(flat.size * voxel_ml, 3),
        "indices": base64.b64encode(flat.tobytes()).decode(),
        "points": len(prompts),
        "source": f"nnInteractive {info.get('version', '?')} ({info.get('model_id', '?')})",
        "licence": f"nnInteractive code Apache-2.0; weights {info.get('weights_licence', '?')} (non-commercial)",
    }
