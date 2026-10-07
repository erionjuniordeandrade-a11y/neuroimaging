"""Named seed presets — server-owned ROI masks from the case manifest.

Only keys declared under the active manifest inputs with the ``seed_`` prefix
are eligible. The client may name a preset id; it never supplies a filesystem path.
Resolved paths must stay under ``case_root`` (path-traversal refuse).

Presets may declare a multi-ROI *recipe* (optional AND / NOT masks).

Live Commit uses interactive ``wmfod_norm``. On that FOD, hand-knob seed +
pons ``-include`` yields ~0 streamlines (measured: 3 / 1.9M seeds). The same
recipe on bank FOD ``wmfod_lmax8_norm`` recovers full multi-ROI CST (~2000).
The CST candidate for this 3T case is therefore the Explore ACT bank extract
(``bank_cst_r``), not live hand-knob Commit — a recipe-selected candidate,
not verified anatomy. Seed-only pons blobs look "like
CST" but stay local (~15 mm span).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import nibabel as nib

from .grid import Grid, load_grid, voxel_to_world


class PresetError(ValueError):
    """Unknown / missing / mismatched preset — map to HTTP 400/404."""


@dataclass(frozen=True)
class SeedPreset:
    id: str                 # e.g. "seed_cst_r"
    label: str              # e.g. "CST-R hand-knob"
    path: str               # absolute seed path (MPR overlay)
    n_voxels: int
    centroid_mm: tuple[float, float, float]
    and_paths: tuple[str, ...] = ()
    not_paths: tuple[str, ...] = ()
    note: str = ""
    # optional denser tracking + post-hoc geometry filter (server-owned)
    seeds: int | None = None
    select: int | None = None
    minlength: float | None = None
    z_max: float | None = None   # post-hoc: keep streamlines with min_z < z_max
    post_cap: int | None = None


def _label_from_id(preset_id: str) -> str:
    raw = preset_id[5:] if preset_id.startswith("seed_") else preset_id
    return raw.replace("_", "-").upper()


def _centroid_mm(mask: np.ndarray, grid: Grid) -> tuple[float, float, float]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return (0.0, 0.0, 0.0)
    c = ijk.mean(axis=0)
    w = voxel_to_world(grid, c.reshape(1, 3))[0]
    return (float(w[0]), float(w[1]), float(w[2]))


def _resolve_case_path(case_root: str, rel: str, *, what: str) -> str:
    root = os.path.realpath(case_root)
    if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
        raise PresetError(f"{what}: path must be relative with no '..'")
    abs_path = os.path.realpath(os.path.join(root, rel))
    if abs_path != root and not abs_path.startswith(root + os.sep):
        raise PresetError(f"{what}: path escapes case_root")
    if not os.path.isfile(abs_path):
        raise PresetError(f"{what}: file missing: {rel}")
    return abs_path


def _load_mask_on_grid(path: str, reference_grid: Grid, *, what: str) -> np.ndarray:
    img = nib.load(path)
    mask = np.asarray(img.dataobj) > 0
    g = load_grid(path)
    if g.shape != reference_grid.shape:
        raise PresetError(f"{what}: shape {g.shape} != tracking grid {reference_grid.shape}")
    if not np.allclose(g.affine, reference_grid.affine, atol=1e-3):
        raise PresetError(f"{what}: affine mismatch vs tracking grid")
    if int(mask.sum()) == 0:
        raise PresetError(f"{what}: empty mask")
    return mask


def load_preset_catalog(
    case_root: str,
    inputs: dict,
    reference_grid: Grid,
) -> dict[str, SeedPreset]:
    """Build the id→SeedPreset map from manifest inputs.

    Optional recipe fields on each seed_* entry:
      and:  ["relative/path.nii.gz", ...]  → extra -include each
      not:  ["relative/path.nii.gz", ...]  → unioned into one -exclude
      label / note: human strings
    """
    catalog: dict[str, SeedPreset] = {}
    for key, meta in (inputs or {}).items():
        if not key.startswith("seed_"):
            continue
        if not isinstance(meta, dict) or "path" not in meta:
            continue
        try:
            abs_path = _resolve_case_path(case_root, meta["path"], what=key)
        except PresetError as e:
            # path-escape / bad relative path = hard fail; missing file = skip
            msg = str(e).lower()
            if "missing" in msg:
                continue
            raise
        try:
            mask = _load_mask_on_grid(abs_path, reference_grid, what=key)
        except PresetError as e:
            if "empty" in str(e).lower() or "shape" in str(e).lower() or "affine" in str(e).lower():
                continue
            raise

        and_paths: list[str] = []
        for i, rel in enumerate(meta.get("and") or []):
            try:
                and_paths.append(
                    _resolve_case_path(case_root, rel, what=f"{key}.and[{i}]")
                )
                _load_mask_on_grid(and_paths[-1], reference_grid, what=f"{key}.and[{i}]")
            except PresetError:
                and_paths = []  # drop incomplete recipe
                break

        not_paths: list[str] = []
        for i, rel in enumerate(meta.get("not") or []):
            try:
                not_paths.append(
                    _resolve_case_path(case_root, rel, what=f"{key}.not[{i}]")
                )
                _load_mask_on_grid(not_paths[-1], reference_grid, what=f"{key}.not[{i}]")
            except PresetError:
                not_paths = []
                break

        label = meta.get("label") if isinstance(meta.get("label"), str) else _label_from_id(key)
        note = meta.get("note") if isinstance(meta.get("note"), str) else ""
        track = meta.get("track") if isinstance(meta.get("track"), dict) else {}

        def _opt_int(k: str) -> int | None:
            v = track.get(k)
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        def _opt_float(k: str) -> float | None:
            v = track.get(k)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        g = load_grid(abs_path)
        catalog[key] = SeedPreset(
            id=key,
            label=label,
            path=abs_path,
            n_voxels=int(mask.sum()),
            centroid_mm=_centroid_mm(mask, g),
            and_paths=tuple(and_paths),
            not_paths=tuple(not_paths),
            note=note,
            seeds=_opt_int("seeds"),
            select=_opt_int("select"),
            minlength=_opt_float("minlength"),
            z_max=_opt_float("z_max"),
            post_cap=_opt_int("post_cap"),
        )
    return catalog


def catalog_public(catalog: dict[str, SeedPreset]) -> list[dict]:
    """JSON-safe list for GET /api/presets (no absolute paths)."""
    return [
        {
            "id": p.id,
            "label": p.label,
            "nVoxels": p.n_voxels,
            "centroid_mm": list(p.centroid_mm),
            "nAnd": len(p.and_paths),
            "nNot": len(p.not_paths),
            "note": p.note,
            "dense": bool(p.seeds and p.seeds > 50_000),
            "postFilter": p.z_max is not None,
        }
        for p in sorted(catalog.values(), key=lambda x: x.label)
    ]


def load_preset_mask(catalog: dict[str, SeedPreset], preset_id: str) -> np.ndarray:
    """Return bool seed mask for a known preset id."""
    if preset_id not in catalog:
        raise PresetError(f"unknown preset: {preset_id!r}")
    img = nib.load(catalog[preset_id].path)
    return np.asarray(img.dataobj) > 0


def load_preset_recipe_masks(
    catalog: dict[str, SeedPreset],
    preset_id: str,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Return (seed_mask, and_masks, not_mask_or_None) for a preset recipe."""
    if preset_id not in catalog:
        raise PresetError(f"unknown preset: {preset_id!r}")
    p = catalog[preset_id]
    seed = load_preset_mask(catalog, preset_id)
    and_masks = [np.asarray(nib.load(path).dataobj) > 0 for path in p.and_paths]
    not_mask = None
    if p.not_paths:
        not_mask = np.zeros(seed.shape, dtype=bool)
        for path in p.not_paths:
            not_mask |= np.asarray(nib.load(path).dataobj) > 0
        if int(not_mask.sum()) == 0:
            not_mask = None
    return seed, and_masks, not_mask
