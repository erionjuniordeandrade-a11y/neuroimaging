"""Parcellation prior discovery + network meshes (A2). Fail-closed like tract priors."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

from . import derivation
from .atlas_fetch import YEO7_NETWORKS, load_schaefer_lut, label_to_network_map
from .grid import Grid, load_grid, voxel_to_world
from .surface import pack_mesh


@dataclass(frozen=True)
class ParcellationSpec:
    path: str
    lut_path: str
    label: str
    n_parcels: int
    n_networks: int
    note: str


def parcellation_qc_ok(man: dict) -> bool:
    qc = derivation.qc_block_for(man, "parcellation_qc")
    if qc is None:
        return False
    if not (qc.get("approved_by") and qc.get("date") and qc.get("sheet_sha")):
        return False
    return True


DEFAULT_PARCELLATION_KEY = "parc_schaefer200_yeo7"


def discover_parcellation(
    case_root: str, inputs: dict, man: dict, *, key: str = DEFAULT_PARCELLATION_KEY
) -> ParcellationSpec | None:
    """Fail-closed discovery for one declared parcellation input.

    ``key`` defaults to the sole production parcellation
    (``parc_schaefer200_yeo7``) so every existing caller is unaffected;
    callers that need to discover-by-arbitrary-key (e.g. connectome.py,
    which must use the exact key it was asked to build against, never a
    hardcoded one) pass ``key=`` explicitly.
    """
    if not parcellation_qc_ok(man):
        return None
    meta = (inputs or {}).get(key)
    if not isinstance(meta, dict) or "path" not in meta:
        return None
    root = os.path.realpath(case_root)
    rel = meta["path"]
    if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
        return None
    abs_path = os.path.realpath(os.path.join(root, rel))
    if not abs_path.startswith(root + os.sep) or not os.path.isfile(abs_path):
        return None
    lut_rel = meta.get("lut_path") or "normative/Schaefer2018_200Parcels_7Networks_order.txt"
    lut_path = os.path.realpath(os.path.join(root, lut_rel))
    if not lut_path.startswith(root + os.sep) or not os.path.isfile(lut_path):
        # fall back to cache LUT is not allowed for fail-closed serve — must be in case
        return None
    return ParcellationSpec(
        path=abs_path,
        lut_path=lut_path,
        label=str(meta.get("label") or "Schaefer-200 / Yeo-7"),
        n_parcels=int(meta.get("n_parcels") or 200),
        n_networks=int(meta.get("n_networks") or 7),
        note=str(meta.get("note") or "POPULATION parcellation prior"),
    )


def load_label_volume(path: str, grid: Grid) -> np.ndarray:
    g = load_grid(path)
    if g.shape != grid.shape or not np.allclose(g.affine, grid.affine, atol=1e-3):
        raise ValueError(f"parcellation grid mismatch: {path}")
    vol = np.asanyarray(nib.load(path).dataobj)
    if vol.ndim != 3:
        raise ValueError("parcellation must be 3-D labels")
    return vol.astype(np.int32)


def network_mask(labels: np.ndarray, lut_path: str, network_id: int) -> np.ndarray:
    lut = load_schaefer_lut(__import__("pathlib").Path(lut_path))
    mapping = label_to_network_map(lut)
    keep = {lab for lab, nid in mapping.items() if nid == network_id}
    return np.isin(labels, list(keep))


def network_mesh(
    path: str,
    grid: Grid,
    lut_path: str,
    network_id: int,
    *,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    labels = load_label_volume(path, grid)
    m = network_mask(labels, lut_path, network_id)
    if not m.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    m = ndimage.binary_fill_holes(m)
    smooth = ndimage.gaussian_filter(m.astype(np.float32), sigma=0.6)
    try:
        verts_vox, faces, _, _ = measure.marching_cubes(smooth, level=0.5, step_size=step)
    except (ValueError, RuntimeError):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    return voxel_to_world(grid, verts_vox).astype("<f4"), faces.astype("<u4")


def pack_network_mesh(path: str, grid: Grid, lut_path: str, network_id: int) -> tuple[bytes, dict]:
    v, f = network_mesh(path, grid, lut_path, network_id)
    return pack_mesh(v, f)


def labels_u8_for_mpr(labels: np.ndarray) -> np.ndarray:
    """Map 0–200 labels into u8 (0–200 fits)."""
    return np.clip(labels, 0, 255).astype(np.uint8)


# Desaturated Yeo-ish palette (sRGB 0–1) for wash / ghost — never direction RGB
YEO7_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.55, 0.45, 0.65),  # Visual — muted violet
    2: (0.55, 0.55, 0.45),  # Somatomotor — olive
    3: (0.45, 0.55, 0.50),  # Dorsal Attn — teal-gray
    4: (0.60, 0.48, 0.42),  # Ventral Attn — dusty rose
    5: (0.50, 0.55, 0.42),  # Limbic — sage
    6: (0.45, 0.50, 0.60),  # Control — steel
    7: (0.58, 0.48, 0.48),  # Default — mauve
}


def network_public_list() -> list[dict]:
    return [
        {"id": n, "name": YEO7_NETWORKS[n], "color": list(YEO7_COLORS[n])}
        for n in range(1, 8)
    ]


# --- FreeSurfer parcellation in diffusion space (crossed-FAT, archive crossed-fat-sdd) ---
#
# FreeSurfer parcellation in diffusion space: loaded, summarised, and gated.
#
# ⛔ The parcellation and the diffusion in this case come from TWO DIFFERENT SCAN
# SESSIONS (~27-28 mm translation plus ~6 deg rotation). A header-only transfer
# covers only 56.8% of the brain mask and leaves rh-precentral 48% outside the
# brain. The only supported path is a `bbregister --dti` transform applied with
# `mri_label2vol` (see docs/runbook-crossed-fat.md). ⛔ `flirt -applyxfm` is
# forbidden here: aparc+aseg is (L,I,A) det -1 against (L,P,S) det +1 diffusion,
# and flirt crosses that sign flip SILENTLY.

# FreeSurfer aparc+aseg label ids. Cortical: 1000 + DK index (lh), 2000 + (rh).
LABEL_IDS: dict[str, int] = {
    "lh_precentral": 1024,
    "rh_precentral": 2024,
    "lh_superiorfrontal": 1028,
    "rh_superiorfrontal": 2028,
    "lh_paracentral": 1017,
    "rh_paracentral": 2017,
    "brainstem": 16,
}


def load_parcellation(path: str) -> tuple[Grid, np.ndarray]:
    """Load a segmentation volume and its authoritative grid.

    Labels are returned as int32. The grid comes from ``load_grid`` so exactly
    one affine is ever used.
    """
    grid = load_grid(path)
    labels = np.asarray(nib.load(path).dataobj).astype(np.int32)
    if labels.shape != grid.shape:
        raise ValueError(f"{path}: labels {labels.shape} != grid {grid.shape}")
    return grid, labels


def label_centroids_world(
    grid: Grid, labels: np.ndarray, ids: dict[str, int]
) -> dict[str, np.ndarray | None]:
    """Centroid of each label in world mm, or None where the label is empty.

    None is an honest null: "this label did not survive resampling", which is
    materially different from "its centroid is at the origin".
    """
    out: dict[str, np.ndarray | None] = {}
    for name, lid in ids.items():
        ijk = np.argwhere(labels == lid)
        out[name] = None if ijk.shape[0] == 0 else voxel_to_world(grid, ijk).mean(axis=0)
    return out


@dataclass(frozen=True)
class ParcelQC:
    """Result of the geometric sanity gate. ``failures`` is empty iff ok."""

    failures: tuple[str, ...]
    centroids: dict[str, np.ndarray | None]

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0


def check_laterality_and_ap(centroids: dict[str, np.ndarray | None]) -> ParcelQC:
    """Assert right/left sidedness and precentral-posterior-to-superiorfrontal.

    World space is RAS+, so right-hemisphere centroids have x > 0 and posterior
    structures have smaller y. Both halves are required: a laterality-only gate
    passed a precentral ROI that was misplaced ANTERIORLY.
    """
    fails: list[str] = []
    for name in ("lh_precentral", "rh_precentral",
                 "lh_superiorfrontal", "rh_superiorfrontal"):
        if centroids.get(name) is None:
            fails.append(f"missing: {name} has zero voxels")

    for name, want_positive in (("rh_precentral", True), ("lh_precentral", False),
                                ("rh_superiorfrontal", True),
                                ("lh_superiorfrontal", False)):
        c = centroids.get(name)
        if c is None:
            continue
        if (c[0] > 0) != want_positive:
            side = "right (x>0)" if want_positive else "left (x<0)"
            fails.append(f"laterality: {name} centroid x={c[0]:.1f} is not {side}")

    for hemi in ("lh", "rh"):
        pre, sf = centroids.get(f"{hemi}_precentral"), centroids.get(f"{hemi}_superiorfrontal")
        if pre is None or sf is None:
            continue
        if not pre[1] < sf[1]:
            fails.append(
                f"anteroposterior: {hemi}-precentral y={pre[1]:.1f} is not posterior "
                f"to {hemi}-superiorfrontal y={sf[1]:.1f}"
            )

    return ParcelQC(failures=tuple(fails), centroids=centroids)
