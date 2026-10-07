"""Run nnInteractive point-prompt segmentation for pending capsule prompts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata

import numpy as np

from .pack import PLACEHOLDER, _BLOB, _MANIFEST, read_capsule, read_capsule_tract_blobs, write_capsule


DEFAULT_NNINTERACTIVE_PYTHON = os.path.expanduser("~/.local/share/case-capsule/nninteractive-venv/bin/python")
RUNNER = Path(__file__).with_name("nninteractive_runner.py")


def _safe_output(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise FileExistsError(f"output must not be the input (never overwritten): {output_path}")
    if output_path.exists():
        raise FileExistsError(f"output already exists (never overwritten): {output_path}")


def _grid_point(ras: list[float], inverse_affine: np.ndarray, dims: tuple[int, int, int],
               name: str, point_n: int) -> tuple[int, int, int]:
    try:
        xyz = np.asarray(ras, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prompt {name!r} point {point_n} has invalid RAS coordinates") from exc
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"prompt {name!r} point {point_n} has invalid RAS coordinates")
    ijk_h = inverse_affine @ np.append(xyz, 1.0)
    if not np.isfinite(ijk_h).all() or abs(ijk_h[3]) < 1e-12:
        raise ValueError(f"prompt {name!r} point {point_n} has invalid RAS coordinates")
    ijk_float = ijk_h[:3] / ijk_h[3]
    ijk = np.floor(ijk_float + 0.5).astype(np.int64)
    nx, ny, nz = dims
    if not (0 <= ijk[0] < nx and 0 <= ijk[1] < ny and 0 <= ijk[2] < nz):
        raise ValueError(f"prompt {name!r} point {point_n} is outside the capsule grid")
    i, j, k = (int(v) for v in ijk)
    return k, j, i


def _slug(name: str, used_ids: set[str], used_blobs: set[str]) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    base = "seg_" + (re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or "structure")
    candidate, suffix = base, 2
    while candidate in used_ids or f"mask_{candidate}" in used_blobs:
        candidate, suffix = f"{base}-{suffix}", suffix + 1
    used_ids.add(candidate)
    used_blobs.add(f"mask_{candidate}")
    return candidate


def _original_template(input_path: Path, destination: Path) -> Path:
    html = input_path.read_text(encoding="utf-8")
    manifests = list(_MANIFEST.finditer(html))
    if len(manifests) != 1:
        raise ValueError("input capsule must contain exactly one manifest")
    manifest = manifests[0]
    payload_end = manifest.end()
    for blob in _BLOB.finditer(html):
        if blob.start() < payload_end or html[payload_end:blob.start()].strip():
            raise ValueError("input capsule payload is not a contiguous manifest and blob block")
        payload_end = blob.end()
    template = html[:manifest.start()] + PLACEHOLDER + html[payload_end:]
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("input capsule viewer template could not be recovered")
    destination.write_text(template, encoding="utf-8")
    return destination


def _runner_error(result: subprocess.CompletedProcess[str]) -> RuntimeError:
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > 1600:
        detail = detail[-1600:]
    return RuntimeError(f"nnInteractive runner failed with exit code {result.returncode}"
                        + (f": {detail}" if detail else ""))


def segment_capsule(input_path: str | Path, output_path: str | Path, *,
                    python_path: str | Path | None = None, device: str = "mps") -> tuple[Path, dict]:
    """Segment all pending prompts and write a new capsule without replacing either input or prior output."""
    source, destination = Path(input_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"input capsule not found: {source}")
    _safe_output(source, destination)
    if device not in {"mps", "cpu"}:
        raise ValueError("--device must be mps or cpu")

    manifest, arrays = read_capsule(source)
    for blob_id, tract_array in read_capsule_tract_blobs(source).items():
        arrays[blob_id] = tract_array
    prompts = manifest.get("seg_prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("manifest seg_prompts must be a list")
    if any(not isinstance(item, dict) for item in prompts):
        raise ValueError("manifest seg_prompts entries must be objects")
    pending = [item for item in prompts if item.get("status") == "pending"]
    if pending:
        runs = manifest.setdefault("segmentation_runs", [])
        if not isinstance(runs, list):
            raise ValueError("manifest segmentation_runs must be a list")
        next_version = int(manifest.get("version", 0)) + 1
    dims = tuple(int(v) for v in manifest["grid"]["dims"])
    if len(dims) != 3:
        raise ValueError("capsule grid must have three dimensions")
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=np.float64)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("capsule affine_ras must be a finite, invertible 4x4 matrix")
    inverse_affine = np.linalg.inv(affine)

    volume_by_id = {volume["id"]: volume for volume in manifest.get("volumes", [])}
    requests: list[dict] = []
    image_data: dict[str, np.ndarray] = {}
    image_keys: dict[str, str] = {}
    for prompt in pending:
        name = str(prompt.get("name", ""))
        if not name.strip():
            raise ValueError("pending segmentation prompt has no structure name")
        if not isinstance(prompt.get("color"), str):
            raise ValueError(f"prompt {name!r} has no color")
        prompt_points = prompt.get("points")
        if not isinstance(prompt_points, list) or not prompt_points:
            raise ValueError(f"prompt {name!r} has no points")
        volume_id = prompt.get("for_volume")
        volume = volume_by_id.get(volume_id)
        if volume is None:
            raise ValueError(f"prompt {name!r} references unknown volume {volume_id!r}")
        if volume_id not in image_keys:
            image_key = f"image_{len(image_keys)}"
            image_keys[volume_id] = image_key
            image_data[image_key] = arrays[volume["blob"]].astype(np.float32, copy=False)
        points = []
        for index, point in enumerate(prompt_points, start=1):
            if not isinstance(point, dict):
                raise ValueError(f"prompt {name!r} point {index} is invalid")
            positive = point.get("positive", True)
            if not isinstance(positive, bool):
                raise ValueError(f"prompt {name!r} point {index} has invalid polarity")
            kji = _grid_point(point.get("ras"), inverse_affine, dims, name, index)
            points.append({"kji": list(kji), "positive": positive})
        requests.append({"id": prompt.get("id"), "name": name, "image_key": image_keys[volume_id],
                         "points_kji": points})

    if pending:
        python_value = (python_path or os.environ.get("CASE_CAPSULE_NNINTERACTIVE_PYTHON")
                        or DEFAULT_NNINTERACTIVE_PYTHON)
        python = str(Path(python_value).expanduser())
        with tempfile.TemporaryDirectory(prefix="case-capsule-segment-") as work:
            workdir = Path(work)
            input_npz, request_json = workdir / "input.npz", workdir / "request.json"
            output_dir, result_json = workdir / "results", workdir / "result.json"
            output_dir.mkdir()
            np.savez(input_npz, **image_data)
            request_json.write_text(json.dumps({"prompts": requests}, separators=(",", ":")), encoding="utf-8")
            started = time.monotonic()
            result = subprocess.run(
                [python, str(RUNNER), "--input", str(input_npz), "--request", str(request_json),
                 "--output-dir", str(output_dir), "--result", str(result_json), "--device", device],
                capture_output=True, text=True, check=False,
            )
            seconds = time.monotonic() - started
            if result.returncode:
                raise _runner_error(result)
            try:
                runner_info = json.loads(result_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("nnInteractive runner did not return valid model provenance") from exc
            model_id = str(runner_info["model_id"])
            version = str(runner_info["version"])
            weights_licence = str(runner_info["weights_licence"])
            licence = f"nnInteractive code Apache-2.0; weights {weights_licence} (non-commercial)"

            used_ids = {mask["id"] for mask in manifest.get("masks", [])}
            used_blobs = {item["blob"] for kind in ("volumes", "masks", "anatomy", "tracts")
                          for item in manifest.get(kind, [])}
            added_ids = []
            voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
            for index, (prompt, request) in enumerate(zip(pending, requests)):
                mask_path = output_dir / f"mask_{index:04d}.npz"
                try:
                    with np.load(mask_path, allow_pickle=False) as result_data:
                        mask = np.asarray(result_data["mask"], dtype=np.uint8)
                except (OSError, KeyError, ValueError) as exc:
                    raise RuntimeError(f"nnInteractive runner did not return a mask for {request['name']!r}") from exc
                if mask.shape != tuple(reversed(dims)):
                    raise RuntimeError(f"nnInteractive returned the wrong grid shape for {request['name']!r}")
                count = int(np.count_nonzero(mask))
                prompt.pop("mask_id", None)
                if count == 0:
                    prompt["status"] = "empty"
                    continue
                mask_id = _slug(request["name"], used_ids, used_blobs)
                blob_id = f"mask_{mask_id}"
                arrays[blob_id] = mask
                positives = sum(point["positive"] for point in request["points_kji"])
                negatives = len(request["points_kji"]) - positives
                mask_meta = {
                    "id": mask_id, "blob": blob_id, "dtype": "uint8", "for_volume": prompt["for_volume"],
                    "volume_ml": round(count * voxel_ml, 2), "label": request["name"], "color": prompt["color"],
                    "source": "auto", "method": f"nninteractive point prompts ({positives}+, {negatives}−)",
                    "model": model_id, "licence": licence, "reviewed": False, "role": "structure",
                }
                manifest.setdefault("masks", []).append(mask_meta)
                prompt["status"], prompt["mask_id"] = "done", mask_id
                added_ids.append(mask_id)

            runs.append({"tool": "nnInteractive", "version": version, "model_id": model_id,
                         "licence": licence, "device": device, "seconds": round(seconds, 3)})
            manifest["version"] = next_version
    else:
        added_ids = []

    with tempfile.TemporaryDirectory(prefix="case-capsule-template-") as work:
        template = _original_template(source, Path(work) / "template.html")
        write_capsule(template, destination, manifest, arrays)
    return destination, {"masks_added": added_ids, "manifest": manifest}
