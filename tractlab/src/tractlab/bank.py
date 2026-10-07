"""Explore-mode: filter a precomputed dense streamline bank.

Research-quality path that this FOD *can* support without reverse-PE:

  whole-brain ACT iFOD2 corpus (already on disk) → tckedit / in-memory
  multi-ROI filter → display subsample.

Labels are honest: FILTERED PREVIEW, not a re-track; ACT bank provenance is
surfaced; no reverse-PE → not navigation.

In-memory filter is fast; ``tckedit`` remains the CLI oracle for tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from types import MappingProxyType
from pathlib import Path

import numpy as np
import nibabel as nib

from .grid import Grid
from .traversal import segment_voxel_hits_affine

TCKEDIT = os.path.expanduser("~/mrtrix3/bin/tckedit")
TCKINFO = os.path.expanduser("~/mrtrix3/bin/tckinfo")

_CACHE: dict = {"path": None, "mtime": None, "tracks": None, "lengths": None}


def _validate_publishable_tracks(tracks) -> list[np.ndarray]:
    """Validate the geometry that can enter a published bank population.

    MRtrix/Nibabel can represent an empty or one-vertex row, while the TCK
    writer drops those rows.  Such a bank would therefore have a different
    identity after export.  Reject it at the population boundary and keep
    direct low-level intersection helpers free to reason about synthetic
    degenerate rows used by their own unit tests.
    """
    out: list[np.ndarray] = []
    for i, track in enumerate(tracks):
        arr = np.asarray(track, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"bank streamline {i} must be a finite (N, 3) array"
            )
        if arr.shape[0] < 2:
            raise ValueError(
                f"bank streamline {i} must have at least 2 points"
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"bank streamline {i} contains nonfinite geometry"
            )
        out.append(arr)
    return out


@dataclass(frozen=True)
class BankProvenance:
    """E0 — how a bank was built. Absent fields are None with a reason in
    ``sources`` (``"recorded"`` | ``"recipe-script@<commit>"`` | ``"absent:<reason>"``)
    — honest absence, never a silent default."""

    bank_sha256: str | None = None
    fod_sha256: str | None = None
    cutoff: float | None = None
    step_mm: float | None = None
    downsample: int | None = None
    algorithm: str | None = None
    act: bool | None = None
    mrtrix_version: str | None = None
    # Required by provenance_from_manifest (the only supported constructor
    # path); read-only mapping, field name → recorded|recipe-script@…|absent:…
    sources: "MappingProxyType[str, str] | dict[str, str] | None" = None


_PROVENANCE_KEYS = frozenset(f.name for f in fields(BankProvenance))

# Exact runtime types per field (bool is checked FIRST — it subclasses int).
_PROVENANCE_TYPES: dict[str, tuple[type, ...]] = {
    "bank_sha256": (str,),
    "fod_sha256": (str,),
    "cutoff": (int, float),
    "step_mm": (int, float),
    "downsample": (int,),
    "algorithm": (str,),
    "act": (bool,),
    "mrtrix_version": (str,),
}


def _valid_source(value: object) -> bool:
    return isinstance(value, str) and (
        value == "recorded"
        or (value.startswith("recipe-script@") and len(value) > len("recipe-script@"))
        or (value.startswith("absent:") and len(value) > len("absent:"))
    )


def provenance_from_manifest(entry: dict) -> BankProvenance | None:
    """Parse a manifest bank entry's ``provenance`` dict. None ONLY when the
    key is absent; explicit ``null``, unknown keys, wrong types, or a missing/
    malformed ``sources`` block all raise ValueError — malformed provenance
    must never silently become absent provenance."""
    if "provenance" not in entry:
        return None
    raw = entry["provenance"]
    if raw is None:
        raise ValueError("provenance is explicit null — remove the key or fill it")
    if not isinstance(raw, dict):
        raise ValueError(f"provenance must be a dict, got {type(raw).__name__}")
    unknown = set(raw) - _PROVENANCE_KEYS
    if unknown:
        raise ValueError(f"unknown provenance keys: {sorted(unknown)}")

    clean: dict[str, object] = {}
    for name, allowed in _PROVENANCE_TYPES.items():
        if name not in raw or raw[name] is None:
            clean[name] = None
            continue
        v = raw[name]
        if isinstance(v, bool) and bool not in allowed:
            raise ValueError(f"provenance field {name!r} must be {allowed}, got bool")
        if not isinstance(v, allowed):
            raise ValueError(
                f"provenance field {name!r} must be {allowed}, got {type(v).__name__}"
            )
        clean[name] = float(v) if allowed == (int, float) else v

    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("provenance requires a non-empty 'sources' dict")
    bad_keys = set(sources) - (_PROVENANCE_KEYS - {"sources"})
    if bad_keys:
        raise ValueError(f"sources refers to unknown fields: {sorted(bad_keys)}")
    for k, v in sources.items():
        if not _valid_source(v):
            raise ValueError(
                f"sources[{k!r}] must be 'recorded', 'recipe-script@<commit>' "
                f"or 'absent:<reason>' (non-empty suffix), got {v!r}"
            )
        # Value/source coherence where a source is declared: a None field must
        # not claim recorded/recipe provenance, a populated field must not be
        # declared absent. (A populated field MAY lack a source entry — the
        # plan's canonical fixture does; unknown origin is not a lie.)
        if clean[k] is None and not v.startswith("absent:"):
            raise ValueError(f"sources[{k!r}] claims {v!r} but the field is absent")
        if clean[k] is not None and v.startswith("absent:"):
            raise ValueError(f"sources[{k!r}] says absent but the field is populated")
    # Private copy behind a read-only view: the caller's dict cannot mutate us.
    clean["sources"] = MappingProxyType(dict(sources))
    return BankProvenance(**clean)


@dataclass(frozen=True)
class BankSpec:
    """Server-owned bank path + provenance labels."""

    path: str
    label: str
    n_streamlines: int | None
    engine: str  # e.g. "ACT iFOD2 whole-brain bank"
    note: str
    default: bool = False  # auto-load on case open (true CST when available)
    role: str = ""  # e.g. "true_cst" — product primary bundle
    sift2_path: str | None = None  # optional per-streamline SIFT2 weights
    provenance: BankProvenance | None = None  # E0; None = manifest has none


def load_tracks_cached(tck_path: str) -> tuple[list[np.ndarray], np.ndarray]:
    from .evidence_identity import file_signature
    signature = file_signature(tck_path)
    key = os.path.realpath(tck_path)
    if (
        _CACHE["path"] == key
        and _CACHE["mtime"] == signature
        and _CACHE["tracks"] is not None
    ):
        return _validate_publishable_tracks(_CACHE["tracks"]), _CACHE["lengths"]
    tck = nib.streamlines.load(tck_path)
    tracks = _validate_publishable_tracks(tck.streamlines)
    lengths = np.empty(len(tracks), dtype=np.float64)
    for i, t in enumerate(tracks):
        lengths[i] = float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
    if file_signature(tck_path) != signature:
        raise ValueError("bank changed while loading; retry with stable inputs")
    _CACHE.update({"path": key, "mtime": signature, "tracks": tracks, "lengths": lengths})
    return tracks, lengths


def _hits_mask(
    track: np.ndarray,
    mask: np.ndarray,
    inv_affine: np.ndarray,
    stride: int = 1,
) -> bool:
    """Return whether a track's continuous path intersects a mask.

    The default ``stride=1`` uses the exact voxel-cell supercover shared by
    traversal and connectotomy.  A larger stride is retained as an explicit
    diagnostic escape hatch: it selects a reduced polyline first and can
    therefore undercount, while never clamping out-of-bounds coordinates.
    """
    pts = np.asarray(track, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"track must have shape (N, 3), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("non-finite vertex in streamline")
    if len(pts) == 0:
        return False
    if int(stride) > 1 and len(pts) > 1:
        stride_i = int(stride)
        idx = np.unique(np.concatenate([
            np.array([0, len(pts) - 1], dtype=np.int64),
            np.arange(0, len(pts), stride_i, dtype=np.int64),
        ]))
        pts = pts[idx]
    return segment_voxel_hits_affine(pts, np.asarray(mask), inv_affine)


def filter_bank_memory(
    *,
    bank_path: str,
    grid: Grid,
    seed: np.ndarray | None,
    and_masks: list[np.ndarray],
    or_mask: np.ndarray | None,
    not_mask: np.ndarray | None,
    minlength: float = 20.0,
    maxlength: float = 250.0,
    max_keep: int | None = None,
    hit_stride: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """In-memory Boolean filter. OOB never clamped.

    Returns the full filter match list (analytic population). Pass ``max_keep``
    only for tests; production display capping is the caller's job so clearance
    is never computed on a length-ranked cosmetic subsample.

    ``hit_stride`` remains an explicit diagnostic/performance escape hatch, but
    the production default uses the exact continuous segment supercover.
    Stride-thinning the hit test can miss a narrow ROI and is not equivalent to
    the tckedit oracle.
    """
    t0 = time.time()
    tracks, lengths = load_tracks_cached(bank_path)
    tracks = _validate_publishable_tracks(tracks)
    if len(lengths) != len(tracks):
        raise ValueError("bank geometry and length counts do not match")
    inv = np.linalg.inv(np.asarray(grid.affine, dtype=np.float64))
    kept: list[np.ndarray] = []
    rej = {"length": 0, "seed": 0, "and": 0, "or": 0, "not": 0}
    for tr, L in zip(tracks, lengths):
        if L < minlength or L > maxlength:
            rej["length"] += 1
            continue
        if seed is not None and seed.any():
            if not _hits_mask(tr, seed, inv, hit_stride):
                rej["seed"] += 1
                continue
        ok = True
        for m in and_masks:
            if m is None or not m.any():
                continue
            if not _hits_mask(tr, m, inv, hit_stride):
                ok = False
                break
        if not ok:
            rej["and"] += 1
            continue
        if or_mask is not None and or_mask.any():
            if not _hits_mask(tr, or_mask, inv, hit_stride):
                rej["or"] += 1
                continue
        if not_mask is not None and not_mask.any():
            if _hits_mask(tr, not_mask, inv, hit_stride):
                rej["not"] += 1
                continue
        kept.append(tr)

    n_kept = len(kept)
    if max_keep is not None and n_kept > int(max_keep):
        order = np.argsort([
            -float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
            for t in kept
        ])
        kept = [kept[int(i)] for i in order[: int(max_keep)]]

    meta = {
        "engine": "FILTER | corpus",
        "label": "FILTERED PREVIEW (not re-track)",
        "n_corpus": len(tracks),
        "n_kept": n_kept,
        "n_loaded": len(kept),
        "n_display": len(kept),
        "rejected": rej,
        "elapsed_s": round(time.time() - t0, 3),
        "bank": os.path.basename(bank_path),
    }
    return kept, meta


def tckedit_filter(
    *,
    bank_path: str,
    out_tck: str,
    include_paths: list[str],
    exclude_path: str | None,
    minlength: float,
    maxlength: float = 250.0,
) -> int:
    """Authoritative CLI filter (oracle for tests). Returns streamline count."""
    argv = [
        TCKEDIT, bank_path, out_tck,
        "-minlength", str(minlength),
        "-maxlength", str(maxlength),
        "-force", "-quiet",
    ]
    for p in include_paths:
        argv.extend(["-include", p])
    if exclude_path:
        argv.extend(["-exclude", exclude_path])
    subprocess.run(argv, check=True, timeout=600)
    out = subprocess.run([TCKINFO, out_tck], capture_output=True, text=True, timeout=60)
    for line in out.stdout.splitlines():
        if line.strip().startswith("count:"):
            return int(line.split(":", 1)[1])
    return 0


def load_prebuilt_bundle(
    tck_path: str,
    max_keep: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Load a pre-filtered bank file.

    By default returns **all** streamlines in the file (analytic population).
    Pass ``max_keep`` only for tests or explicit display-only loads — never use
    a length-ranked display subsample as the clearance population.
    """
    t0 = time.time()
    tracks, lengths = load_tracks_cached(tck_path)
    tracks = _validate_publishable_tracks(tracks)
    if len(lengths) != len(tracks):
        raise ValueError("bank geometry and length counts do not match")
    n_full = len(tracks)
    if max_keep is not None and n_full > int(max_keep):
        order = np.argsort(-lengths)
        tracks = [tracks[int(i)] for i in order[: int(max_keep)]]
    meta = {
        "engine": "BANK | prebuilt",
        "label": "PREBUILT FILTER (ACT bank -> multi-ROI)",
        "n_corpus": n_full,
        "n_kept": n_full,
        "n_loaded": len(tracks),
        "n_display": len(tracks),  # caller may subsample further for draw
        "elapsed_s": round(time.time() - t0, 3),
        "bank": os.path.basename(tck_path),
    }
    return tracks, meta


def discover_banks(case_root: str, inputs: dict) -> dict[str, BankSpec]:
    """Banks declared under manifest inputs with prefix bank_ or bank path keys."""
    root = os.path.realpath(case_root)
    out: dict[str, BankSpec] = {}
    for key, meta in (inputs or {}).items():
        if not key.startswith("bank_"):
            continue
        if not isinstance(meta, dict) or "path" not in meta:
            continue
        rel = meta["path"]
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            continue
        abs_path = os.path.realpath(os.path.join(root, rel))
        if not abs_path.startswith(root + os.sep) or not os.path.isfile(abs_path):
            continue
        n = meta.get("n_streamlines")
        sift2 = None
        sift_rel = meta.get("sift2")
        if isinstance(sift_rel, str) and sift_rel and ".." not in sift_rel.split("/"):
            sp = os.path.realpath(os.path.join(root, sift_rel))
            if sp.startswith(root + os.sep) and os.path.isfile(sp):
                sift2 = sp
        if sift2 is None:
            # auto-discover sibling <stem>.sift2.txt
            cand = os.path.splitext(abs_path)[0] + ".sift2.txt"
            if os.path.isfile(cand):
                sift2 = cand
        try:
            prov = provenance_from_manifest(meta)
        except ValueError as e:
            # Scoped refusal: this bank is NOT served (serving it with
            # provenance=None would silently downgrade malformed → absent),
            # but one bad entry must not take down every other bank.
            print(f"REFUSED {key}: malformed provenance — {e}", file=sys.stderr)
            continue
        out[key] = BankSpec(
            path=abs_path,
            label=str(meta.get("label") or key),
            n_streamlines=int(n) if isinstance(n, (int, float)) else None,
            engine=str(meta.get("engine") or "bank"),
            note=str(meta.get("note") or ""),
            default=bool(meta.get("default")),
            role=str(meta.get("role") or ""),
            sift2_path=sift2,
            provenance=prov,
        )
    return out
