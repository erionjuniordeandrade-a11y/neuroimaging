"""Atlas prior discovery + fail-closed availability (ADR-0001 / ADR-0002)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

from . import derivation
from .grid import Grid, load_grid, voxel_to_world
from .surface import pack_mesh


@dataclass(frozen=True)
class PriorSpec:
    id: str  # e.g. norm_cst_r
    path: str
    label: str
    official_name: str
    atlas_index: int | None
    note: str


def atlas_prior_qc_ok(man: dict) -> bool:
    """True only when human-signed atlas_prior_qc is present and complete."""
    qc = derivation.qc_block_for(man, "atlas_prior_qc")
    if qc is None:
        return False
    if not qc.get("approved_by"):
        return False
    if not qc.get("date") or not qc.get("sheet_sha"):
        return False
    return True


def discover_priors(case_root: str, inputs: dict, man: dict) -> dict[str, PriorSpec]:
    """Return priors only when QC is signed AND every declared norm_* is on disk.

    Fail closed: any missing/invalid declaration → empty dict (panel absent).
    """
    if not atlas_prior_qc_ok(man):
        return {}
    root = os.path.realpath(case_root)
    declared: dict[str, PriorSpec] = {}
    for key, meta in (inputs or {}).items():
        if not key.startswith("norm_"):
            continue
        if not isinstance(meta, dict) or "path" not in meta:
            return {}  # corrupt declaration → fail closed entirely
        rel = meta["path"]
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            return {}
        abs_path = os.path.realpath(os.path.join(root, rel))
        if not abs_path.startswith(root + os.sep) or not os.path.isfile(abs_path):
            return {}
        idx = meta.get("atlas_index")
        declared[key] = PriorSpec(
            id=key,
            path=abs_path,
            label=str(meta.get("label") or key),
            official_name=str(meta.get("official_name") or meta.get("label") or key),
            atlas_index=int(idx) if isinstance(idx, (int, float)) else None,
            note=str(meta.get("note") or "population atlas prior"),
        )
    if not declared:
        return {}
    return declared


def prior_to_u8(path: str, grid: Grid, *, thr: float = 0.0) -> np.ndarray:
    """Load probability volume as u8 (0–255) on the tracking grid. Fail if grid mismatch."""
    g = load_grid(path)
    if g.shape != grid.shape or not np.allclose(g.affine, grid.affine, atol=1e-3):
        raise ValueError(f"prior grid mismatch: {path}")
    vol = np.asanyarray(nib.load(path).dataobj, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"prior must be 3-D: {path}")
    # XTRACT probs are often 0–100
    vmax = float(np.nanmax(vol)) if vol.size else 0.0
    if vmax > 1.5:
        vol = vol / 100.0
    vol = np.clip(vol, 0, 1)
    if thr > 0:
        vol = np.where(vol >= thr, vol, 0)
    return (vol * 255.0).astype(np.uint8)


def prior_mesh(
    path: str,
    grid: Grid,
    *,
    level: float = 0.25,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Marching-cubes hull at probability ``level`` (0–1). Empty if no isosurface."""
    g = load_grid(path)
    if g.shape != grid.shape or not np.allclose(g.affine, grid.affine, atol=1e-3):
        raise ValueError(f"prior grid mismatch: {path}")
    vol = np.asanyarray(nib.load(path).dataobj, dtype=np.float64)
    vmax = float(np.nanmax(vol)) if vol.size else 0.0
    if vmax > 1.5:
        vol = vol / 100.0
    vol = np.clip(vol, 0, 1)
    if float(vol.max()) < level:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    smooth = ndimage.gaussian_filter(vol.astype(np.float32), sigma=0.7)
    try:
        verts_vox, faces, _, _ = measure.marching_cubes(
            smooth, level=level, step_size=step,
        )
    except (ValueError, RuntimeError):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    verts_world = voxel_to_world(grid, verts_vox).astype("<f4")
    return verts_world, faces.astype("<u4")


def pack_prior_mesh(path: str, grid: Grid, *, level: float = 0.25) -> tuple[bytes, dict]:
    verts, faces = prior_mesh(path, grid, level=level)
    return pack_mesh(verts, faces)
