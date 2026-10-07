"""Peri-lesional recovery — fibres near the lesion, not named-bundle proof.

Bank multi-ROI recipes use healthy cortical waypoints and systematically drop
streamlines that only survive in edematous / mass-effect space. Recovery
answers a different question:

  "What streamlines from this corpus pass within N mm of the lesion?"

It is intentionally NOT a true_cst / true_slf label. UI and export must stamp
RECOVERY · not named tract · research only.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from .bank import _hits_mask, load_tracks_cached
from .grid import Grid, load_grid


def voxel_spacing_mm(affine: np.ndarray) -> np.ndarray:
    a = np.asarray(affine, dtype=np.float64)
    return np.sqrt((a[:3, :3] ** 2).sum(axis=0))


def lesion_proximity_mask(
    lesion_bool: np.ndarray,
    affine: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Boolean mask: world-mm distance to lesion voxel centres.

    The old orthogonal-grid EDT used column norms as independent spacings,
    which is not a world-distance operation for oblique/sheared affines.  A
    KD-tree over lesion voxel centres and chunked world-coordinate queries
    preserves the same centre-shell convention while supporting anisotropic
    and oblique grids.
    """
    try:
        radius = float(radius_mm)
    except (TypeError, ValueError):
        raise ValueError("radius_mm must be a finite number >= 0") from None
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_mm must be >= 0")
    m = np.asarray(lesion_bool, dtype=bool)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    if radius == 0.0:
        return m.copy()
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    lesion_ijk = np.argwhere(m).astype(np.float64)
    lesion_world = lesion_ijk @ a[:3, :3].T + a[:3, 3]
    tree = cKDTree(lesion_world)
    result = np.zeros(m.shape, dtype=bool)
    nx = int(m.shape[0])
    # Query x-slabs to avoid materializing a whole large world grid at once.
    for x0 in range(0, nx, 32):
        x1 = min(nx, x0 + 32)
        ijk = np.indices((x1 - x0, m.shape[1], m.shape[2]), dtype=np.float64)
        ijk[0] += float(x0)
        coords = np.stack((ijk[0], ijk[1], ijk[2]), axis=-1).reshape(-1, 3)
        world = coords @ a[:3, :3].T + a[:3, 3]
        dist, _ = tree.query(world, k=1)
        result[x0:x1] = (dist <= radius).reshape(x1 - x0, m.shape[1], m.shape[2])
    return result


def load_lesion_bool(path: str, grid: Grid) -> np.ndarray:
    """Load lesion mask; must match tracking grid."""
    g = load_grid(path)
    if g.shape != grid.shape or not np.allclose(g.affine, grid.affine, atol=1e-3):
        raise ValueError(f"lesion grid mismatch: {path}")
    return np.asanyarray(nib.load(path).dataobj) > 0


def filter_near_lesion_memory(
    *,
    bank_path: str,
    grid: Grid,
    lesion_zone: np.ndarray,
    minlength: float = 10.0,
    maxlength: float = 250.0,
    hit_stride: int = 1,
    max_keep: int | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Keep streamlines whose continuous path hits the lesion proximity zone.

    ``lesion_zone`` should already be dilated (see lesion_proximity_mask).
    The default exact supercover predicate is insertion-invariant; an explicit
    ``hit_stride`` is retained only as a diagnostic undercount escape hatch.
    Returns full match list (analytic); caller applies display subsample.
    """
    t0 = time.time()
    if not np.isfinite(minlength) or not np.isfinite(maxlength) or not (0 <= minlength <= maxlength <= 1000):
        raise ValueError("recovery lengths must be finite and satisfy 0 <= min <= max <= 1000 mm")
    if max_keep is not None and (isinstance(max_keep, bool) or not isinstance(max_keep, (int, np.integer)) or max_keep < 1):
        raise ValueError("recovery max_keep must be a positive integer")
    zone = np.asarray(lesion_zone, dtype=bool)
    if zone.shape != grid.shape:
        raise ValueError("lesion_zone shape != tracking grid")
    if not zone.any():
        return [], {
            "engine": "RECOVERY | peri-lesional",
            "label": "PERI-LESIONAL RECOVERY (not named tract)",
            "n_corpus": 0,
            "n_kept": 0,
            "n_loaded": 0,
            "rejected": {"empty_zone": 1},
            "elapsed_s": 0.0,
            "bank": os.path.basename(bank_path),
            "role": "recovery_perilesional",
            "note": "empty lesion proximity zone",
        }

    tracks, lengths = load_tracks_cached(bank_path)
    inv = np.linalg.inv(np.asarray(grid.affine, dtype=np.float64))
    kept: list[np.ndarray] = []
    rej = {"length": 0, "miss_zone": 0}
    for tr, L in zip(tracks, lengths):
        if L < minlength or L > maxlength:
            rej["length"] += 1
            continue
        if not _hits_mask(tr, zone, inv, hit_stride):
            rej["miss_zone"] += 1
            continue
        kept.append(tr)

    n_kept = len(kept)
    if max_keep is not None and n_kept > int(max_keep):
        # Prefer longer among recovery hits for stability in tests only
        order = np.argsort([
            -float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
            for t in kept
        ])
        kept = [kept[int(i)] for i in order[: int(max_keep)]]

    meta: dict[str, Any] = {
        "engine": "RECOVERY | peri-lesional filter",
        "label": "PERI-LESIONAL RECOVERY (not named tract)",
        "n_corpus": len(tracks),
        "n_kept": n_kept,
        "n_loaded": len(kept),
        "n_display": len(kept),
        "rejected": rej,
        "elapsed_s": round(time.time() - t0, 3),
        "bank": os.path.basename(bank_path),
        "role": "recovery_perilesional",
        "note": (
            "Fibres within radius of lesion from corpus — not multi-ROI named bundle; "
            "not navigation; research only"
        ),
    }
    return kept, meta
