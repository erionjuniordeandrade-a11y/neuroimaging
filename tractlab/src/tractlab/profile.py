"""Along-tract scalar profiles + per-node lesion-distance track.

Descriptive geometry on a named analytic bank. Not a normative score, not a
clearance p5, not a resection margin.

Resampling is numpy arc-length (pack.resample_polyline). Tests compare a
tiny synthetic .tck against ``tckresample -num_points``. Scalar sampling is
trilinear in world mm via the nibabel affine.

Orientation is fail-closed: the rule is derived from the bank-id family and
recorded. Unknown families raise rather than guessing an axis.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from neuro_core.hashing import sha256_file

from .bank import BankSpec, load_prebuilt_bundle
from .casepath import resolve_case_path
from .clearance import lesion_surface_points, per_vertex_distance_mm
from .grid import load_grid, world_to_voxel
from .pack import resample_polyline
from . import derivation as dv

CLAIM = "descriptive profile; not a normative abnormality score"
N_POINTS_DEFAULT = 100
HISTOGRAM_BINS = 20
# Display threshold only: flag a node when (n_zero + n_nan) / n_streamlines
# exceeds this fraction. An exact zero is reported, not interpreted as mask.
ZERO_OR_MISSING_THRESHOLD = 0.20
CACHE_SCHEMA = "zero-or-missing-v2"

# Manifest keys, in lookup order. inputs.fa / inputs.md are the optional
# grid-matched underlays (GET /api/volume/fa). Demo Leipzig maps live under
# profile_fa / profile_md because they are on the native DWI grid.
SCALAR_KEYS: dict[str, tuple[str, ...]] = {
    "fa": ("fa", "profile_fa"),
    "md": ("md", "profile_md"),
}

# Longest-prefix first (slf3 before a hypothetical slf).
_KNOWN_FAMILIES: tuple[str, ...] = (
    "slf1", "slf2", "slf3", "ifof", "cing", "cst", "fat", "uf", "or",
)

# First include from scripts/rebuild_strict_banks.py recipes.
_FIRST_ROI_STEM: dict[str, str] = {
    "fat": "{family}_{side}_sfg",
    "slf1": "{family}_{side}_front",
    "slf2": "{family}_{side}_front",
    "slf3": "{family}_{side}_smg",
    "ifof": "{family}_{side}_occ",
    "uf": "{family}_{side}_temp",
    "cing": "{family}_{side}_ant",
    "or": "{family}_{side}_thal",
}

_ROI_OR_ANTERIOR = frozenset(_FIRST_ROI_STEM)


class ProfileError(ValueError):
    """Typed profile failure the route maps to a structured HTTP body."""

    http_status = 400
    code = "profile_error"


class UnknownBank(ProfileError):
    http_status = 404
    code = "unknown_bank"


class ScalarMissing(ProfileError):
    http_status = 409
    code = "scalar_missing"


class ScalarUnsupported(ProfileError):
    http_status = 409
    code = "scalar_unsupported"


class UnknownFamily(ProfileError):
    http_status = 409
    code = "orientation_unknown_family"


class InvalidBankId(ProfileError):
    http_status = 409
    code = "invalid_bank_id"


class HashMismatch(ProfileError):
    http_status = 409
    code = "hash_mismatch"


class LesionInvalid(ProfileError):
    http_status = 409
    code = "lesion_invalid"




def parse_bank_id(bank_id: str) -> tuple[str, str, bool]:
    """Return (family, side, soft) from a ``bank_*`` id.

    Tokens are split on ``_``. Family must be exactly one known token; side
    must be exactly ``l`` or ``r``. Prefix matches like ``bank_fat_right``
    are refusals, not laterality.
    """
    if not isinstance(bank_id, str):
        raise UnknownFamily(f"not a bank id: {bank_id!r}")
    parts = bank_id.split("_")
    if len(parts) < 3 or parts[0] != "bank":
        raise UnknownFamily(f"not a bank id: {bank_id!r}")
    family = parts[1]
    if family not in _KNOWN_FAMILIES:
        raise UnknownFamily(
            f"no orientation rule for {bank_id!r} "
            f"(known families: {', '.join(_KNOWN_FAMILIES)})"
        )
    rest = parts[2:]
    soft = bool(rest) and rest[-1] == "soft"
    if soft:
        rest = rest[:-1]
    if not rest or rest[0] not in ("l", "r"):
        raise InvalidBankId(
            f"side token must be exactly 'l' or 'r', got {bank_id!r}"
        )
    return family, rest[0], soft


def resample(streamlines: Iterable[np.ndarray], n_points: int = N_POINTS_DEFAULT) -> list[np.ndarray]:
    """Arc-length resample every streamline to ``n_points`` vertices."""
    if int(n_points) < 2:
        raise ValueError("n_points must be >= 2")
    return [resample_polyline(s, int(n_points)) for s in streamlines]


def sample_scalar(
    streamlines_resampled: Iterable[np.ndarray],
    scalar_nifti_path: str,
) -> np.ndarray:
    """Trilinear sample a scalar NIfTI at each vertex (world mm).

    Out-of-bounds vertices are NaN (never clamped onto the volume face).
    Returns shape (n_streamlines, n_points).
    """
    img = nib.load(scalar_nifti_path)
    data = np.asarray(img.dataobj, dtype=np.float64)
    if data.ndim != 3:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"{scalar_nifti_path}: scalar map must be 3-D, got {data.shape}")
    grid = load_grid(scalar_nifti_path)
    stacked = _stack_lines(streamlines_resampled)
    if stacked.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    vox = world_to_voxel(grid, stacked.reshape(-1, 3))
    shape = np.asarray(data.shape, dtype=np.float64)
    valid = np.all((vox >= 0.0) & (vox <= (shape - 1.0)), axis=1)
    samples = np.full(vox.shape[0], np.nan, dtype=np.float64)
    if valid.any():
        samples[valid] = map_coordinates(
            data, vox[valid].T, order=1, mode="constant", cval=np.nan,
        )
    n_line, n_pt = stacked.shape[0], stacked.shape[1]
    return samples.reshape(n_line, n_pt)


def roi_centroid_mm(nifti_path: str) -> np.ndarray | None:
    img = nib.load(nifti_path)
    mask = np.asarray(img.dataobj) > 0
    if not mask.any():
        return None
    ijk = np.argwhere(mask).astype(np.float64)
    from .grid import voxel_to_world
    return voxel_to_world(load_grid(nifti_path), ijk).mean(axis=0)


def _stack_lines(streamlines: Iterable[np.ndarray]) -> np.ndarray:
    lines = [np.asarray(s, dtype=np.float64) for s in streamlines]
    if not lines:
        return np.empty((0, 0, 3), dtype=np.float64)
    n_pt = lines[0].shape[0]
    out = np.empty((len(lines), n_pt, 3), dtype=np.float64)
    for i, arr in enumerate(lines):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"streamline {i} is not (n,3)")
        if arr.shape[0] != n_pt:
            raise ValueError("resampled streamlines must share n_points")
        out[i] = arr
    return out


def _endpoints(streamlines: Iterable[np.ndarray]) -> np.ndarray:
    ends: list[np.ndarray] = []
    for s in streamlines:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
            continue
        ends.append(arr[0])
        ends.append(arr[-1])
    if not ends:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(ends, axis=0)


def _pole_anchor(streamlines: Iterable[np.ndarray], axis: int, which: str) -> np.ndarray:
    ends = _endpoints(streamlines)
    if ends.shape[0] == 0:
        raise ValueError("no endpoints to derive an orientation anchor")
    col = ends[:, int(axis)]
    idx = int(np.argmin(col) if which == "min" else np.argmax(col))
    return ends[idx]


def _first_roi_path(case_root: str, family: str, side: str | None, soft: bool) -> str | None:
    if side is None or family not in _FIRST_ROI_STEM:
        return None
    stem = _FIRST_ROI_STEM[family].format(family=family, side=side)
    names = [f"{stem}_dil2.nii.gz", f"{stem}_dil1.nii.gz"] if soft else [
        f"{stem}_dil1.nii.gz", f"{stem}_dil2.nii.gz",
    ]
    roi_dir = os.path.join(os.path.realpath(case_root), "tracts", "roi")
    for name in names:
        path = os.path.join(roi_dir, name)
        if os.path.isfile(path):
            return path
    return None


def orient_streamlines(
    streamlines: Iterable[np.ndarray],
    anchor_mm: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """Flip each line so node 0 is the endpoint closer to ``anchor_mm``."""
    anchor = np.asarray(anchor_mm, dtype=np.float64).reshape(3)
    out: list[np.ndarray] = []
    n_flipped = 0
    for s in streamlines:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            out.append(arr)
            continue
        d0 = float(np.linalg.norm(arr[0] - anchor))
        d1 = float(np.linalg.norm(arr[-1] - anchor))
        if d1 < d0:
            arr = np.ascontiguousarray(arr[::-1])
            n_flipped += 1
        out.append(arr)
    return out, n_flipped


def resolve_orientation(
    bank_id: str,
    streamlines: Iterable[np.ndarray],
    case_root: str,
) -> dict[str, Any]:
    """Declare the orientation rule and the world-mm anchor it uses."""
    family, side, soft = parse_bank_id(bank_id)
    lines = [np.asarray(s, dtype=np.float64) for s in streamlines]
    if family == "cst":
        anchor = _pole_anchor(lines, axis=2, which="min")
        rule = "inferior"
        extra: dict[str, Any] = {}
    elif family in _ROI_OR_ANTERIOR:
        roi_path = _first_roi_path(case_root, family, side, soft)
        centroid = roi_centroid_mm(roi_path) if roi_path else None
        if centroid is not None:
            anchor = centroid
            rule = "roi_centroid"
            rel = os.path.relpath(roi_path, os.path.realpath(case_root))
            extra = {
                "roi_path": rel.replace("\\", "/"),
                "roi_sha256": sha256_file(roi_path),
            }
        else:
            anchor = _pole_anchor(lines, axis=1, which="max")
            rule = "anterior"
            extra = {
                "fallback_reason": (
                    "first ROI missing or empty; anterior (max y) end"
                ),
            }
    else:
        raise UnknownFamily(f"no orientation rule for family {family!r}")
    oriented, n_flipped = orient_streamlines(lines, anchor)
    record = {
        "rule": rule,
        "anchor_mm": [float(x) for x in np.asarray(anchor, dtype=np.float64).reshape(3)],
        "family": family,
        "side": side,
        "n_flipped": int(n_flipped),
        **extra,
    }
    return {"streamlines": oriented, "record": record}


def aggregate_nodes(samples: np.ndarray) -> dict[str, list]:
    """Per-node stats plus zero/missing counts. NaNs are dropped from the median.

    ``n_valid`` = finite samples (also mirrored as ``n``). ``n_zero`` = finite
    samples that are exactly 0. ``n_nan`` = non-finite samples. The
    ``zero_or_missing_flag`` is a display threshold on (n_zero + n_nan) /
    n_streamlines; it does not claim a cause for an exact zero.
    """
    empty = {
        "n": [], "n_valid": [], "n_zero": [], "n_nan": [],
        "zero_or_missing_flag": [], "median": [], "p25": [], "p75": [], "mean": [],
    }
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return empty
    n_lines = int(arr.shape[0])
    n_pt = arr.shape[1]
    n_list: list[int] = []
    n_zero: list[int] = []
    n_nan: list[int] = []
    flags: list[bool] = []
    med: list[float | None] = []
    p25: list[float | None] = []
    p75: list[float | None] = []
    mean: list[float | None] = []
    thresh = ZERO_OR_MISSING_THRESHOLD * n_lines
    for k in range(n_pt):
        col = arr[:, k]
        finite = col[np.isfinite(col)]
        n_valid = int(finite.size)
        n_z = int(np.count_nonzero(finite == 0.0))
        n_n = int(n_lines - n_valid)
        n_list.append(n_valid)
        n_zero.append(n_z)
        n_nan.append(n_n)
        flags.append(bool((n_z + n_n) > thresh))
        if n_valid == 0:
            med.append(None)
            p25.append(None)
            p75.append(None)
            mean.append(None)
            continue
        med.append(float(np.median(finite)))
        p25.append(float(np.percentile(finite, 25)))
        p75.append(float(np.percentile(finite, 75)))
        mean.append(float(np.mean(finite)))
    return {
        "n": n_list,
        "n_valid": list(n_list),
        "n_zero": n_zero,
        "n_nan": n_nan,
        "zero_or_missing_flag": flags,
        "median": med,
        "p25": p25,
        "p75": p75,
        "mean": mean,
    }


def zero_or_missing_warning(nodes: Mapping[str, list]) -> str | None:
    """List nodes over the display threshold, or null when none are."""
    flags = nodes.get("zero_or_missing_flag") or []
    flagged = [str(i) for i, on in enumerate(flags) if on]
    if not flagged:
        return None
    listed = ",".join(flagged)
    return (
        f"nodes {listed}: n_zero+n_nan exceeds "
        f"{ZERO_OR_MISSING_THRESHOLD:.0%} of streamlines "
        "(exact zeros or missing samples; not a tissue FA claim)"
    )


def streamline_mean_histogram(samples: np.ndarray, n_bins: int = HISTOGRAM_BINS) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"counts": [], "edges": [], "n": 0}
    means = np.nanmean(arr, axis=1)
    finite = means[np.isfinite(means)]
    if finite.size == 0:
        return {"counts": [], "edges": [], "n": 0}
    counts, edges = np.histogram(finite, bins=int(n_bins))
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
    }


def lesion_distance_track(
    streamlines_resampled: Iterable[np.ndarray],
    lesion_path: str | None,
) -> dict[str, Any]:
    """Per-node min distance (mm) from any streamline's node-k to the lesion surface.

    Undeclared / absent path → null track with reason ``no lesion mask``.
    A *declared* lesion is resolved by :func:`resolve_declared_lesion` first.
    """
    if not lesion_path or not os.path.isfile(lesion_path):
        return {"track_mm": None, "reason": "no lesion mask"}
    shell = lesion_surface_points(lesion_path)
    if shell.shape[0] == 0:
        return {"track_mm": None, "reason": "no lesion mask"}
    stacked = _stack_lines(streamlines_resampled)
    if stacked.size == 0:
        return {"track_mm": None, "reason": "no lesion mask"}
    dist = per_vertex_distance_mm(stacked.astype(np.float64), shell)
    if dist is None:
        return {"track_mm": None, "reason": "no lesion mask"}
    track = np.min(np.asarray(dist, dtype=np.float64), axis=0)
    return {"track_mm": [float(x) for x in track], "reason": None}


def _declared_lesion_rel(manifest: Mapping[str, Any]) -> str | None:
    inputs = dv.active_inputs(manifest)
    entry = inputs.get("lesion") if isinstance(inputs, dict) else None
    if not isinstance(entry, dict):
        return None
    rel = entry.get("path")
    if isinstance(rel, str) and rel.strip():
        return rel
    return None


def resolve_declared_lesion(case_root: str, manifest: Mapping[str, Any]) -> tuple[str, str]:
    """Return (abs_path, rel_path) for a manifest-declared lesion, or raise.

    Distinct reasons: missing, escapes case_root, empty. Undeclared is not
    this function's job — the caller returns the honest-null track instead.
    """
    rel = _declared_lesion_rel(manifest)
    if rel is None:
        raise LesionInvalid("declared lesion missing")
    try:
        path = resolve_case_path(case_root, rel, what="lesion")
    except ValueError as exc:
        raise LesionInvalid("declared lesion escapes case_root") from exc
    if not os.path.isfile(path):
        raise LesionInvalid("declared lesion missing")
    shell = lesion_surface_points(path)
    if shell.shape[0] == 0:
        raise LesionInvalid("declared lesion empty")
    return path, rel


def resolve_scalar_entry(manifest: Mapping[str, Any], scalar: str) -> tuple[str, dict]:
    name = str(scalar).strip().lower()
    if name not in SCALAR_KEYS:
        raise ScalarUnsupported(
            f"unsupported scalar {scalar!r}; want one of {sorted(SCALAR_KEYS)}"
        )
    inputs = dv.active_inputs(manifest)
    for key in SCALAR_KEYS[name]:
        entry = inputs.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]:
            return key, entry
    raise ScalarMissing(
        f"no {name} map in manifest (looked for {', '.join(SCALAR_KEYS[name])})"
    )


def _cache_path(case_root: str, bank_id: str, scalar: str) -> str:
    return os.path.join(os.path.realpath(case_root), "work", "profiles", f"{bank_id}.{scalar}.json")


def _load_cache(path: str, key: Mapping[str, Any]) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    stored = payload.get("cache") if isinstance(payload, dict) else None
    if not isinstance(stored, dict):
        return None
    for field, value in key.items():
        if stored.get(field) != value:
            return None
    return payload


def _save_cache(path: str, payload: dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", dir=folder, suffix=".tmp", prefix=".profile-", delete=False,
    )
    tmp = handle.name
    try:
        with handle:
            json.dump(payload, handle, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _require_recorded_sha(entry: Mapping[str, Any], computed: str, what: str) -> str:
    recorded = entry.get("sha256")
    if not isinstance(recorded, str) or not recorded:
        raise HashMismatch(f"{what} has no recorded sha256 in the manifest")
    if recorded != computed:
        raise HashMismatch(
            f"{what} sha256 mismatch: file {computed} != manifest {recorded}"
        )
    return computed


def _roi_cache_fields(case_root: str, bank_id: str) -> tuple[str | None, str | None]:
    family, side, soft = parse_bank_id(bank_id)
    if family not in _ROI_OR_ANTERIOR:
        return None, None
    roi_path = _first_roi_path(case_root, family, side, soft)
    if not roi_path:
        return None, None
    rel = os.path.relpath(roi_path, os.path.realpath(case_root)).replace("\\", "/")
    return rel, sha256_file(roi_path)


def _bank_path_of(spec: BankSpec | Mapping[str, Any] | Any) -> str:
    path = getattr(spec, "path", None)
    if path is None and isinstance(spec, Mapping):
        path = spec.get("path")
    if not isinstance(path, str) or not path:
        raise UnknownBank("bank spec has no path")
    return path


def compute_bank_profile(
    *,
    case_root: str,
    manifest: Mapping[str, Any],
    banks: Mapping[str, Any],
    bank_id: str,
    scalar: str = "fa",
    lesion_path: str | None = None,
    n_points: int = N_POINTS_DEFAULT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build (or reload) the along-tract profile for one manifest bank."""
    if not isinstance(bank_id, str) or not bank_id.startswith("bank_"):
        raise UnknownBank(f"unknown bank: {bank_id}")
    inputs = dv.active_inputs(manifest)
    if bank_id not in inputs or bank_id not in banks:
        raise UnknownBank(f"unknown bank: {bank_id}")
    parse_bank_id(bank_id)
    scalar_name = str(scalar).strip().lower()
    key, scalar_entry = resolve_scalar_entry(manifest, scalar_name)
    bank_entry = inputs[bank_id]
    if not isinstance(bank_entry, dict) or "path" not in bank_entry:
        raise UnknownBank(f"unknown bank: {bank_id}")

    bank_rel = str(bank_entry["path"])
    bank_path = _bank_path_of(banks[bank_id])
    if not os.path.isfile(bank_path):
        raise UnknownBank(f"unknown bank: {bank_id}")
    bank_sha = _require_recorded_sha(bank_entry, sha256_file(bank_path), "bank")

    scalar_rel = str(scalar_entry["path"])
    scalar_path = resolve_case_path(case_root, scalar_rel, what=scalar_name)
    if not os.path.isfile(scalar_path):
        raise ScalarMissing(f"{scalar_name} map missing: {scalar_rel}")
    scalar_sha = _require_recorded_sha(
        scalar_entry, sha256_file(scalar_path), f"{scalar_name} map",
    )

    declared = _declared_lesion_rel(manifest)
    lesion_used = None
    lesion_rel = None
    lesion_sha = None
    if declared is not None:
        lesion_used, lesion_rel = resolve_declared_lesion(case_root, manifest)
        lesion_sha = sha256_file(lesion_used)
    elif lesion_path:
        # A path the caller passed is still not a declaration.
        lesion_used = None

    active = manifest.get("active_derivation")
    if active is not None and not isinstance(active, str):
        active = str(active)
    roi_rel, roi_sha = _roi_cache_fields(case_root, bank_id)

    cache_key = {
        "schema": CACHE_SCHEMA,
        "bank_sha256": bank_sha,
        "scalar_sha256": scalar_sha,
        "lesion_sha256": lesion_sha,
        "roi_sha256": roi_sha,
        "n_points": int(n_points),
        "active_derivation": active,
    }
    cache_file = _cache_path(case_root, bank_id, scalar_name)
    if use_cache:
        cached = _load_cache(cache_file, cache_key)
        if cached is not None:
            return cached

    t0 = time.time()
    lines, _meta = load_prebuilt_bundle(bank_path, max_keep=None)
    if declared is not None and not lines:
        raise LesionInvalid("declared lesion has no streamline samples")
    oriented = resolve_orientation(bank_id, lines, case_root)
    resampled = resample(oriented["streamlines"], int(n_points))
    if declared is not None and not resampled:
        raise LesionInvalid("declared lesion has no streamline samples")
    samples = sample_scalar(resampled, scalar_path)
    nodes = aggregate_nodes(samples)
    les_track = lesion_distance_track(resampled, lesion_used)
    if lesion_used is not None:
        les_track = {
            **les_track,
            "lesion_path": lesion_rel,
            "lesion_sha256": lesion_sha,
        }
    payload: dict[str, Any] = {
        "bank_id": bank_id,
        "bank_path": bank_rel,
        "bank_sha256": bank_sha,
        "scalar": scalar_name,
        "scalar_key": key,
        "scalar_path": scalar_rel,
        "scalar_sha256": scalar_sha,
        "n_points": int(n_points),
        "n_streamlines": int(samples.shape[0]),
        "orientation": oriented["record"],
        "active_derivation": active,
        "mrtrix_version": None,
        "resample_method": "numpy_arc_length",
        "claim": CLAIM,
        "nodes": nodes,
        "zero_or_missing_threshold": ZERO_OR_MISSING_THRESHOLD,
        "zero_or_missing_warning": zero_or_missing_warning(nodes),
        "streamline_mean_histogram": streamline_mean_histogram(samples),
        "lesion_distance": les_track,
        "lesion_path": lesion_rel,
        "lesion_sha256": lesion_sha,
        "elapsed_s": round(time.time() - t0, 3),
        "cache": dict(cache_key),
    }
    if use_cache:
        _save_cache(cache_file, payload)
    return payload
