"""Build one shared layer list (volumes, label masks, tracts) for the unified viewer.

A scene is plain files in a cache folder plus a JSON description. The viewer
loads every layer into one NiiVue instance, so CT, MR, CTA, masks and tracts
share one set of slices and one 3D view in scanner RAS millimetres.

Sources:

* a Capsule file: its grid volumes and masks become NIfTI files, its tract
  blobs are already MRtrix .tck in world millimetres;
* a TractLab manifest: its NIfTI inputs and .tck bundle banks are served as
  they are, read only, by key.

Layers from two sources are only drawn together when the caller links them;
the viewer then states that their alignment is assumed, not verified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from capsule.pack import parse_capsule

SCENE_SCHEMA = "neuro-workbench-scene/1"
# TractLab manifest input keys that are images, in the order a reader expects them.
TRACTLAB_VOLUME_KEYS = ("t1", "T1", "t2", "flair", "b0", "fa", "FA", "md")
TRACTLAB_LABEL_KEYS = ("lesion", "tumour", "tumor", "mask")
MASK_ROLES_SKIPPED = ("render",)
# Render masks that are anatomy a reader needs (not the head/brain shells used for 3D cut-aways).
RENDER_MASKS_SHOWN = ("vessel",)
DEFAULT_COLOURS = ("#4FC3F7", "#7CFC00", "#FFB74D", "#BA68C8", "#F06292", "#4DB6AC", "#FFF176")


def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _window_for(kind: str, label: str) -> dict:
    text = f"{kind} {label}".upper()
    if "CTA" in text or "ANGIO" in text:
        return {"cal_min": 100.0, "cal_max": 600.0}
    if kind.upper() == "CT":
        return {"cal_min": 0.0, "cal_max": 80.0}
    return {}


def _volume_kind(meta: dict) -> str:
    text = f"{meta.get('kind', '')} {meta.get('label', '')}".upper()
    if "CTA" in text or "ANGIO" in text:
        return "CTA"
    return str(meta.get("kind") or "MR").upper()


def _anatomy_title(item_id: str, method: str) -> str:
    if item_id == "anat_mr_gyri":
        return "Cortical gyri (auto)"
    if item_id == "anat_mr":
        return "Brain anatomy (auto)"
    task = item_id.removeprefix("anat_ct_").replace("_", " ")
    return f"CT {task} (auto)" if item_id.startswith("anat_ct_") else f"{item_id} (auto)"


def capsule_scene(capsule: Path, cache_root: Path) -> tuple[dict, dict[str, Path]]:
    """Decode a capsule into NIfTI/.tck files once and describe them as layers."""
    capsule = capsule.resolve()
    out = cache_root / f"capsule-{_cache_key(capsule)}"
    described = out / "scene.json"
    if described.is_file():
        scene = json.loads(described.read_text())
        return scene, {k: out / v for k, v in scene.pop("_files").items()}

    manifest, blobs = parse_capsule(capsule)
    grid = manifest["grid"]
    nx, ny, nz = grid["dims"]
    affine = np.array(grid["affine_ras"], dtype=float)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    scene = {"schema": SCENE_SCHEMA, "frame": "scanner RAS mm", "source": "capsule",
             "title": (manifest.get("case") or {}).get("label") or capsule.name.removesuffix(".capsule.html"),
             "volumes": [], "labels": [], "tracts": []}

    def nifti(key: str, array: np.ndarray, slope: float = 1.0, inter: float = 0.0) -> None:
        image = nib.Nifti1Image(np.ascontiguousarray(array.transpose(2, 1, 0)), affine)
        image.header.set_xyzt_units("mm")
        if slope not in (0.0, 1.0) or inter != 0.0:
            image.header.set_slope_inter(slope, inter)
        name = f"{key}.nii.gz"
        tmp = out / f".{name}.{os.getpid()}.tmp.nii.gz"
        nib.save(image, str(tmp))
        os.replace(tmp, out / name)
        files[key] = name

    for meta in manifest.get("volumes", []):
        blob = blobs.get(meta["blob"])
        if blob is None:
            continue
        data = np.frombuffer(blob, dtype=np.dtype(meta.get("dtype", "int16"))).reshape((nz, ny, nx))
        key = f"vol-{meta['id']}"
        slope = float(meta.get("slope") or 1.0)
        nifti(key, data, slope, float(meta.get("intercept") or 0.0))
        kind = _volume_kind(meta)
        scene["volumes"].append({"id": key, "label": meta.get("label") or meta["id"], "kind": kind,
                                 "units": meta.get("units"), **_window_for(kind, meta.get("label", ""))})

    for meta in manifest.get("masks", []):
        shown = any(w in f"{meta.get('id', '')} {meta.get('label', '')}".lower() for w in RENDER_MASKS_SHOWN)
        if (meta.get("role") in MASK_ROLES_SKIPPED and not shown) or meta["blob"] not in blobs:
            continue
        data = np.frombuffer(blobs[meta["blob"]], dtype=np.uint8).reshape((nz, ny, nx))
        key = f"mask-{meta['id']}"
        nifti(key, (data > 0).astype(np.uint8))
        scene["labels"].append({"id": key, "label": meta.get("label") or meta["id"],
                                "color": meta.get("color") or "#E4572E",
                                "reviewed": bool(meta.get("reviewed")), "source": meta.get("source"),
                                "volume_ml": meta.get("volume_ml")})

    for meta in manifest.get("anatomy", []):
        blob = blobs.get(meta.get("blob"))
        labels = [lb for lb in meta.get("labels") or [] if 0 < int(lb.get("value", 0)) < 256]
        if blob is None or not labels:
            continue
        key = f"anat-{meta['id']}"
        nifti(key, np.frombuffer(blob, dtype=np.uint8).reshape((nz, ny, nx)))
        method = str(meta.get("method") or "automatic")
        scene["labels"].append({"id": key, "label": _anatomy_title(meta["id"], method), "multi": True,
                                "color": "#B0BEC5", "reviewed": False, "source": method,
                                "licence": meta.get("licence"),
                                "labels": [{"value": int(lb["value"]), "name": lb.get("name") or lb.get("key"),
                                            "color": lb.get("color") or "#B0BEC5",
                                            "volume_ml": lb.get("volume_ml")} for lb in labels]})

    for i, meta in enumerate(manifest.get("tracts", [])):
        blob = blobs.get(meta.get("blob"))
        if blob is None:
            continue
        key = f"tract-{meta['id']}"
        _atomic_write_bytes(out / f"{key}.tck", blob)
        files[key] = f"{key}.tck"
        scene["tracts"].append({"id": key, "label": meta.get("label") or meta["id"],
                                "color": meta.get("color") or DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)],
                                "n_streamlines": meta.get("n_streamlines"),
                                "reviewed": bool(meta.get("reviewed")), "source": meta.get("source")})

    _atomic_write_bytes(described, json.dumps({**scene, "_files": files}).encode())
    return scene, {k: out / v for k, v in files.items()}


def _resolve_input(manifest: dict, manifest_path: Path, rel: str) -> Path:
    root = Path(manifest.get("case_root") or manifest_path.parent)
    path = Path(rel)
    return (path if path.is_absolute() else root / path).resolve()


def tractlab_scene(manifest_path: Path) -> tuple[dict, dict[str, Path]]:
    """Describe a TractLab manifest's images and bundle banks as layers, by key only."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest.get("inputs") or {}
    files: dict[str, Path] = {}
    scene = {"schema": SCENE_SCHEMA, "frame": "scanner RAS mm", "source": "tractlab",
             "title": str(manifest.get("case_id") or manifest_path.parent.name),
             "volumes": [], "labels": [], "tracts": []}

    def entry_path(name: str) -> Path | None:
        item = inputs.get(name)
        rel = item.get("path") if isinstance(item, dict) else item if isinstance(item, str) else None
        if not rel:
            return None
        path = _resolve_input(manifest, manifest_path, rel)
        return path if path.is_file() else None

    for name in TRACTLAB_VOLUME_KEYS:
        path = entry_path(name)
        if path and path.name.endswith((".nii", ".nii.gz")):
            key = f"vol-{name.lower()}"
            files[key] = path
            scene["volumes"].append({"id": key, "label": name.upper() if len(name) <= 3 else name.title(),
                                     "kind": name.upper()})
    for name in TRACTLAB_LABEL_KEYS:
        path = entry_path(name)
        if path and path.name.endswith((".nii", ".nii.gz")):
            key = f"mask-{name}"
            files[key] = path
            scene["labels"].append({"id": key, "label": name.title(),
                                    "color": "#E4572E" if name != "mask" else "#9E9E9E",
                                    "reviewed": False, "source": "tractlab"})
    colour = 0
    for name, item in inputs.items():
        rel = item.get("path") if isinstance(item, dict) else None
        if not rel or not str(rel).endswith(".tck"):
            continue
        path = _resolve_input(manifest, manifest_path, rel)
        if not path.is_file():
            continue
        key = f"tract-{name}"
        files[key] = path
        scene["tracts"].append({"id": key, "label": item.get("label") or name,
                                "color": DEFAULT_COLOURS[colour % len(DEFAULT_COLOURS)],
                                "n_streamlines": item.get("n_streamlines"),
                                "reviewed": False, "source": item.get("engine") or "tractlab"})
        colour += 1
    return scene, files


def merge_scenes(parts: list[tuple[dict, dict[str, Path]]], title: str) -> tuple[dict, dict[str, Path]]:
    """Join scenes from several sources; layer ids are prefixed per source to stay unique."""
    if len(parts) == 1:
        return parts[0]
    scene = {"schema": SCENE_SCHEMA, "frame": "scanner RAS mm", "source": "linked", "title": title,
             "volumes": [], "labels": [], "tracts": [],
             "alignment": "assumed: sources are drawn in their own scanner coordinates without registration"}
    files: dict[str, Path] = {}
    for i, (part, part_files) in enumerate(parts):
        for group in ("volumes", "labels", "tracts"):
            for layer in part[group]:
                key = f"s{i}-{layer['id']}"
                scene[group].append({**layer, "id": key, "origin": part["source"]})
                files[key] = part_files[layer["id"]]
    return scene, files
