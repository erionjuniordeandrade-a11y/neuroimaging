"""Loopback track server — WHITE-BOX (owner reads every line).

Endpoints:
  GET  /api/health          -> {status, gridId, volumeId, caseId}
  GET  /api/derivation      -> active lineage + all lineage kinds/QC status
  GET  /api/preflight       -> {stored, verified, drift} (read-only; never writes)
  GET  /api/volume/b0       -> binary u8 volume (i-fastest) + metadata headers
  GET  /api/volume/fa       -> FA grayscale u8 underlay (optional)
  GET  /api/volume/dec      -> FOD DEC RGB planes (R|G|B) underlay (optional)
  GET  /api/presets         -> named seed ROI catalog (ids/labels only)
  GET  /api/presets/<id>/volume -> binary u8 mask for MPR overlay
  GET  /api/banks           -> prebuilt ACT-bank bundles (Explore)
  GET  /api/connectotomy    -> C1 lesion ∩ banks (404 if no lesion)
  GET  /api/connectotomy/<id>/cut -> cut-subset tubes (bank styling)
  GET  /api/profile/<bank_id>?scalar=fa|md -> along-tract profile JSON
  GET  /api/connectome      -> C1b matrix + node labels + provenance (404 if not built)
  GET  /api/connectome/edge/<a>/<b> -> edge counts + tck path + cavity hits (ASSIGNED, not cortex — ADR-0003)
  GET  /api/connectome/edge/<a>/<b>/tubes -> edge streamlines as binary tubes (same
       response builder as /api/connectotomy/<id>/cut; one extraction serves both the
       tubes and X-lesionHits/X-lesionTotal/X-lesionReason, so Show needs one request,
       never two; X-edgeSourceHash [tck DATA-segment sha256, stable across repeated
       extractions] + X-corpusSourceHash + X-parcellationSourceHash + X-generationId
       replace X-bankSourceHash — an edge is not bank-backed; 409 on the same staleness
       check as the other two connectome routes)
  POST /api/bank/load       -> load a prebuilt bank bundle (FILTER quality)
  POST /api/filter          -> filter live WB bank by SEED/AND/OR/NOT paint
  POST /api/track           -> binary float32le tracts + metadata headers (Commit)
  GET  /  (and static)      -> the viewer files under viewer/

Track body (world-mm points / preset ids only — never paths/flags):
  seed: {points_mm, radius_mm}            paint SEED → -seed_image
  seed_preset: "seed_fa_r"                named ROI (manifest) → -seed_image
  and:  [{points_mm, radius_mm}, ...]     optional → one -include each
  or:   [{points_mm, radius_mm}, ...]     optional → one -include (union)
  not:  [{points_mm, radius_mm}, ...]     optional → one -exclude (union)
  Exactly one of paint seed or seed_preset is required.

Live Commit params (commercial knobs; never paths/flags):
  params.cutoff     FOD amplitude 0.02–0.20 (NOT tensor FA; default 0.05)
  params.angle      degrees 15–90 (default 45)
  params.minlength  mm (default 20)
  params.density    "sparse"|"normal"|"dense" → server-owned seeds/select

Hard rules (security boundary):
  * Bind 127.0.0.1 ONLY. ``serve()`` refuses any other host.
  * The client supplies numbers + world-mm paint points + preset *ids* ONLY —
    never a path, flag, algorithm, or output name. Paths come from the manifest.
  * gridId/volumeId mismatch → 409 code=stale_volume (reload required).
  * Concurrent track while busy → 409 code=busy (distinct from stale).
  * POST /api/cancel kills the active tckgen process group (best-effort).
  * Empty SEED without seed_preset → 400 (fail loud). Optional AND/OR/NOT ok.
  * Single-flight: one active tckgen at a time; each job in its own temp dir.
  * /api/health carries recipeHash of preset/bank definitions for provenance.
  * Live params mint liveHash in response headers (cutoff change → new hash).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import nibabel as nib

from .casepath import resolve_case_path as _resolve_case_path  # noqa: F401 — re-export; tests import this name
from .grid import (
    grid_id,
    load_grid,
    unknown_units_assumption_from_manifest,
    world_to_voxel,
)
from .fidelity import (
    OperatingPoint,
    FLAG_CROSSES_CAVITY,
    FLAG_CROSSES_LESION,
    FidelityRefusal,
    R_GRID,
    read_operating_point,
    load_sidecar,
    rows_for,
)
from .volume import (
    build_u8_volume,
    build_binary_u8_volume,
    build_rgb_u8_volume,
    assert_grid_match,
)
from .seed import write_seed_nifti
from .track import (
    run_tckgen,
    TrackParams,
    Outcome,
    DENSITY_PRESETS,
    CUTOFF_DEFAULT,
    CUTOFF_MIN,
    CUTOFF_MAX,
    ANGLE_DEFAULT,
    MINLENGTH_DEFAULT,
)


def header_ascii(v) -> str:
    """HTTP headers must be latin-1/ascii; map known unicode (no bare '?')."""
    s = str(v if v is not None else "")
    # Manifest and provenance strings must never change HTTP framing.
    s = "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in s)
    for a, b in (
        ("→", "->"), ("←", "<-"), ("↔", "<->"),
        ("·", " | "), ("•", " | "), ("≈", "~"),
        ("≥", ">="), ("≤", "<="), ("×", "x"),
        ("—", "-"), ("–", "-"), ("…", "..."),
    ):
        s = s.replace(a, b)
    return s.encode("ascii", "replace").decode("ascii")
from .pack import pack_streamlines_with_row_map, unpack_streamlines
from .surface import (
    CORTICAL_AID_LABEL, FS_CORTEX_LABEL, build_brain_hull, build_cortical_relief,
    build_freesurfer_cortex, find_freesurfer_cortex, pack_mesh,
)
from .analytic_source import AnalyticSourceRegistry, AnalyticSourceError, UnknownResultId
from .runtime_identity import (
    BankSourceChanged,
    BankSourceRegistry,
    FileHashCache,
    RuntimeIdentity,
)
from .clearance import (
    lesion_surface_points,
    seed_surface_points,
    clearance_report,
    clearance_report_weighted,
    per_vertex_distance_mm,
    analytic_subset,
    geom_floor_mm,
    ANALYTIC_CAP,
)
from .margin import build_margin_hull
from .export_tck import write_tck
from .sift2_util import is_sha256, load_sift2_weights
from .priors import discover_priors, prior_to_u8, pack_prior_mesh, atlas_prior_qc_ok
from .parcellation import (
    discover_parcellation,
    load_label_volume,
    labels_u8_for_mpr,
    pack_network_mesh,
    network_public_list,
    parcellation_qc_ok,
)
from .roi_boolean import (
    RoiCompileError,
    layers_from_request,
    compile_roi_layers,
    build_tckgen_roi_args,
)
from .presets import (
    PresetError,
    load_preset_catalog,
    catalog_public,
    load_preset_recipe_masks,
)
from .bank import (
    discover_banks,
    load_prebuilt_bundle,
    filter_bank_memory,
)
from .profile import (
    compute_bank_profile,
    ProfileError,
    UnknownBank,
)
from .recovery import (
    filter_near_lesion_memory,
    lesion_proximity_mask,
    load_lesion_bool,
)
from .connectotomy import (
    CavityInvalid,
    connectotomy_note,
    CavityRoleError,
    compute_connectotomy,
    reject_cavity_in_request,
)
from . import connectome as connectome_mod
from . import derivation as dv

from .evidence_identity import sha256_file
from .results import (
    ResultTooLarge,
    ResultUnavailable,
    digest_lines,
)
from . import http_policy
from .http_policy import PolicyRejected


# ── Derivation-honest served strings (audit S-1/S-2/S-4) ─────────────────────
# Every human-readable uncertainty string is built from derivation.floor_label
# so an uncorrected-case warning can never be served on a corrected derivation
# (CLAUDE.md). No literal "no reverse-PE" or "3 mm" lives on these paths.

def health_uncertainty(manifest: dict) -> str:
    return f"{dv.floor_label(manifest or {})}; rigid T1 only; aid not navigation"


def banks_note(manifest: dict) -> str:
    return (
        "Candidate identity, not verified anatomy — recipe-selected ACT 10M "
        "multi-ROI banks (this 3T). Live Commit is exploratory on interactive "
        f"FOD. FILTERED PREVIEW | {dv.floor_label(manifest or {})} | not navigation"
    )


def t1_header_label(manifest: dict) -> str:
    return f"T1 (QC-approved; {dv.floor_label(manifest or {})}; aid only)"


MAX_BODY = 4 * 1024 * 1024      # 4 MB request cap
DISPLAY_K = 64                  # points per streamline in the payload
DISPLAY_CAP = 1500              # max streamlines drawn (== default select)
FILTER_MINLENGTH_MAX = 250.0    # matches the corpus filter's fixed maxlength
RECOVERY_RADIUS_MIN = 3.0
RECOVERY_RADIUS_MAX = 20.0
RECOVERY_RADIUS_DEFAULT = 8.0


class _PopulationEvidenceChanged(ValueError):
    """A bank changed while an identity-sensitive population was computed."""


def _line_lengths_mm(lines: list) -> np.ndarray:
    n = len(lines)
    lengths = np.empty(n, dtype=np.float64)
    for i, s in enumerate(lines):
        arr = np.asarray(s)
        if len(arr) < 2:
            lengths[i] = 0.0
        else:
            lengths[i] = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
    return lengths


def _min_dist_to_shell_mm(
    lines: list,
    shell_xyz: np.ndarray,
    *,
    point_stride: int = 4,
    shell_cap: int = 800,
) -> np.ndarray:
    """Per-streamline min distance (mm) to lesion shell points. Inf if no shell."""
    n = len(lines)
    out = np.full(n, np.inf, dtype=np.float64)
    if shell_xyz is None:
        return out
    shell = np.asarray(shell_xyz, dtype=np.float64).reshape(-1, 3)
    if shell.shape[0] == 0:
        return out
    if shell.shape[0] > shell_cap:
        rng = np.random.default_rng(0)
        shell = shell[rng.choice(shell.shape[0], size=shell_cap, replace=False)]
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return out
    tree = cKDTree(shell)
    stride = max(1, int(point_stride))
    for i, s in enumerate(lines):
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        pts = arr[::stride]
        # Each query has only a few dozen points. Spawning a full CPU thread
        # pool for every streamline dominates bank loading; the serial query
        # returns the same exact nearest distances and source selection.
        d, _ = tree.query(pts, k=1)
        out[i] = float(np.min(d))
    return out


def _display_subsample(
    lines: list,
    weights: np.ndarray | None = None,
    *,
    cap: int = DISPLAY_CAP,
    seed: int = 0,
    lesion_shell: np.ndarray | None = None,
    near_lesion_frac: float = 0.0,
    near_radius_mm: float = 12.0,
) -> tuple[list, np.ndarray]:
    """Display-only sample — never use for clearance p5.

    Returns (shown_lines, source_ordinals) so fidelity sidecars join by
    source index, never by length.

    Prefer SIFT2-weighted top-k when weights align (shows dense core, not
    long outliers). Otherwise sample from the mid-length band (drop short
    stubs and the longest 15% tails) so every bank is less visually noisy
    than pure length-rank.

    Optional near-lesion bias: reserve a fraction of the display cap for
    streamlines with min distance to lesion shell ≤ near_radius_mm so
    peri-lesional anatomy is not erased by mid-length sampling alone.
    """
    n = len(lines)
    if n <= cap:
        return list(lines), np.arange(n, dtype=np.int64)

    near_frac = float(near_lesion_frac or 0.0)
    near_frac = min(0.9, max(0.0, near_frac))
    use_near = (
        near_frac > 0
        and lesion_shell is not None
        and getattr(lesion_shell, "shape", (0,))[0] > 0
    )

    # SIFT2 path (optionally blend with near-lesion quota)
    if weights is not None and not use_near:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) == n and float(np.nansum(w)) > 0:
            order = np.argsort(-w)
            picked = [int(i) for i in order[:cap]]
            return [lines[i] for i in picked], np.asarray(picked, dtype=np.int64)

    lengths = _line_lengths_mm(lines)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    remaining = np.arange(n, dtype=np.int64)

    if use_near:
        dmin = _min_dist_to_shell_mm(lines, lesion_shell)
        near_mask = np.isfinite(dmin) & (dmin <= float(near_radius_mm))
        near_idx = np.flatnonzero(near_mask)
        n_near_quota = int(round(cap * near_frac))
        n_near_quota = min(n_near_quota, cap, int(near_idx.size))
        if n_near_quota > 0:
            # Prefer nearer to lesion among the near set
            order_near = near_idx[np.argsort(dmin[near_idx])]
            if weights is not None:
                w = np.asarray(weights, dtype=np.float64).ravel()
                if len(w) == n:
                    # among near: high SIFT2 first, then closer
                    order_near = near_idx[
                        np.lexsort((dmin[near_idx], -w[near_idx]))
                    ]
            pick_near = order_near[:n_near_quota]
            picked.extend(int(i) for i in pick_near)
            remaining = np.setdiff1d(remaining, pick_near, assume_unique=False)

    rest_cap = cap - len(picked)
    if rest_cap <= 0:
        picked = picked[:cap]
        return [lines[i] for i in picked], np.asarray(picked, dtype=np.int64)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) == n and float(np.nansum(w)) > 0:
            order = remaining[np.argsort(-w[remaining])]
            picked.extend(int(i) for i in order[:rest_cap])
            picked = picked[:cap]
            return [lines[i] for i in picked], np.asarray(picked, dtype=np.int64)

    # mid-length band among remaining
    if remaining.size == 0:
        picked = picked[:cap]
        return [lines[i] for i in picked], np.asarray(picked, dtype=np.int64)
    rem_lengths = lengths[remaining]
    order_local = np.argsort(rem_lengths)
    lo = int(0.15 * len(order_local))
    hi = max(lo + 1, int(0.85 * len(order_local)))
    core_local = order_local[lo:hi]
    core = remaining[core_local]
    if len(core) >= rest_cap:
        pick = rng.choice(core, size=rest_cap, replace=False)
        picked.extend(int(i) for i in pick)
    else:
        picked.extend(int(i) for i in core)
        need = rest_cap - len(core)
        rest = np.setdiff1d(remaining, core, assume_unique=False)
        if need > 0 and len(rest) > 0:
            extra = rng.choice(rest, size=min(need, len(rest)), replace=False)
            picked.extend(int(i) for i in extra)
    picked = picked[:cap]
    return [lines[i] for i in picked], np.asarray(picked, dtype=np.int64)


def _append_source_ordinal_block(
    buf: bytes,
    headers: dict,
    ordinals: np.ndarray,
    *,
    source_population: str,
) -> tuple[bytes, dict]:
    """Append the display-row -> source-row lookup used by viewer picking.

    Ordinals are zero-based rows in the source population named by
    ``source_population``. They are transport metadata only: the geometry and
    fidelity blocks remain unchanged, and downstream offsets are calculated
    after this block is appended.
    """
    rows = np.asarray(ordinals, dtype=np.int64).reshape(-1)
    try:
        line_count = int(headers["lineCount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("packed headers require an integer lineCount") from exc
    if len(rows) != line_count:
        raise ValueError("source ordinal count must equal packed lineCount")
    if np.any(rows < 0) or np.any(rows > np.iinfo(np.uint32).max):
        raise ValueError("source ordinal is outside uint32 range")
    offset = len(buf)
    block = rows.astype("<u4", copy=False).tobytes(order="C")
    out_headers = dict(headers)
    out_headers.update({
        "sourceOrdinalOffset": str(offset),
        "sourceOrdinalCount": str(len(rows)),
        "sourceOrdinalEncoding": "uint32le",
        "sourceOrdinalBase": "0",
        "sourcePopulation": str(source_population),
    })
    return buf + block, out_headers


def _pack_display_streamlines(
    streamlines,
    source_ordinals: np.ndarray,
    k: int,
    space_id: str,
    *,
    minlength_mm: float | None = None,
) -> tuple[bytes, dict, np.ndarray]:
    """Pack geometry and keep its per-line metadata on the identical rows."""
    source_rows = np.asarray(source_ordinals)
    if source_rows.ndim != 1 or len(source_rows) != len(streamlines):
        raise ValueError("source ordinals must contain one row per input streamline")
    buf, headers, retained_input_rows = pack_streamlines_with_row_map(
        streamlines,
        k,
        space_id,
        minlength_mm=minlength_mm,
    )
    retained_source_rows = source_rows[retained_input_rows].astype(np.int64, copy=False)
    if len(retained_source_rows) != int(headers["lineCount"]):
        raise RuntimeError("packed geometry and source ordinals lost row alignment")
    return buf, headers, retained_source_rows


def _derivation_uncertainty(manifest: dict) -> str:
    """Viewer-facing uncertainty text derived from the active lineage floor."""
    return dv.floor_label(manifest)


def _fidelity_payload(
    manifest: dict, bank_id: str, spec, ordinals: np.ndarray, op: OperatingPoint,
    *, fod_path: str,
) -> tuple[bytes, dict]:
    """Append-ready fidelity block + headers for one bank load.

    Missing sidecar → absent (untested, never a 0 count). Provenance-incomplete
    sidecars keep that status and send no block. SHA/length mismatch raises
    FidelityRefusal — the handler maps that to HTTP 422. ``op`` is the
    service's operating point (signed sheet or pilot fallback); the headers
    name its source so the viewer can label the chip honestly.

    S-05: the sidecar is checked against the CURRENT bytes of the bank file
    actually being served (``spec.path``) and the CURRENT FOD file
    (``fod_path``) — never against the manifest's *declared* provenance
    strings alone. A same-count permutation or any other byte-for-byte
    change to either file therefore fails this join even if the manifest's
    declaration was never updated to match. The declared provenance is only
    consulted to decide whether this bank has fidelity data at all.
    """
    path = dv.artifact_path(manifest, "fidelity", f"{bank_id}.fidelity.npz")
    if not path.is_file():
        return b"", {"fidelityStatus": "absent"}
    prov = getattr(spec, "provenance", None)
    declared_bank_sha = getattr(prov, "bank_sha256", None) if prov is not None else None
    declared_fod_sha = getattr(prov, "fod_sha256", None) if prov is not None else None
    if not declared_bank_sha or not declared_fod_sha:
        return b"", {"fidelityStatus": "provenance-incomplete"}
    actual_bank_sha = sha256_file(spec.path)
    actual_fod_sha = sha256_file(fod_path)
    peak_path = dv.artifact_path(manifest, "fidelity", "fod_peak.nii.gz")
    if not peak_path.is_file():
        raise FidelityRefusal("schema-2 support requires its recorded fidelity/fod_peak.nii.gz")
    sc = load_sidecar(
        path, expected_bank_sha=actual_bank_sha, expected_fod_sha=actual_fod_sha,
        expected_peak_sha=sha256_file(peak_path),
        expected_sh_load_sha=actual_fod_sha if Path(fod_path).suffix.lower() != ".mif" else None,
    )
    if not sc.ratios_present:
        return b"", {"fidelityStatus": "provenance-incomplete"}
    rows = rows_for(sc, ordinals)
    r_hits = np.where(np.isclose(R_GRID, op.R))[0]
    if r_hits.size != 1:
        raise FidelityRefusal(f"operating point R {op.R} is not on R_GRID")
    frac = np.asarray(rows.frac_ge[:, int(r_hits[0])], dtype="<f4")
    p5 = np.asarray(rows.p5_ratio, dtype="<f4")
    flags = np.asarray(rows.flags, dtype=np.uint8)
    n = int(flags.shape[0])
    if frac.shape != (n,) or p5.shape != (n,):
        raise FidelityRefusal("fidelity row length mismatch after join")
    provenance_ok = (flags & 4) == 0
    measurable = np.isfinite(frac) & np.isfinite(p5) & (frac >= 0) & (frac <= 1) & provenance_ok
    low = int(
        ((measurable & (frac < op.min_frac))
         | (((flags & FLAG_CROSSES_CAVITY) != 0) & provenance_ok)).sum()
    )
    unmeasurable = int((~measurable).sum())
    n_les = int(((flags & FLAG_CROSSES_LESION) != 0).sum())
    n_cav = int(((flags & FLAG_CROSSES_CAVITY) != 0).sum())
    block = frac.tobytes(order="C") + p5.tobytes(order="C") + flags.tobytes(order="C")
    return block, {
        "fidelityStatus": "ok",
        "fidelityR": f"{op.R:.2f}",
        "fidelityMinFrac": f"{op.min_frac:.2f}",
        "fidelityOpSource": op.source,
        "fidelityApprovedBy": op.approved_by or "",
        "fidelityOpDate": op.date or "",
        "lowSupportCount": str(low),
        "unmeasurableCount": str(unmeasurable),
        "crossesLesionCount": str(n_les),
        "crossesCavityCount": str(n_cav),
    }


def _clearance_headers(clr, pop_meta: dict, floor_error: str | None = None) -> dict:
    """ASCII-safe clearance + population provenance for HTTP headers.

    floor_error: a floorless case must NAME its refusal — blank headers would
    make a broken manifest indistinguishable from a legitimate no-lesion null.
    """
    hdr = {
        "clearanceP5": (
            "" if clr is None or clr.p05_mm is None else f"{clr.p05_mm:.2f}"
        ),
        "clearanceP5Display": "" if clr is None else clr.format_p05(),
        "clearanceP50": (
            "" if clr is None or clr.p50_mm is None else f"{clr.p50_mm:.2f}"
        ),
        "clearanceEligible": "" if clr is None else str(clr.n_eligible),
        "clearanceSeedOnly": "" if clr is None else str(clr.n_seed_only),
        "clearanceFloorMm": "" if clr is None else str(clr.geom_floor_mm),
        "clearanceNumericalBoundMm": "" if clr is None else str(clr.distance_error_bound_mm),
        "clearanceRefusal": (floor_error or "").encode("ascii", "replace").decode(),
        "clearanceMethod": "" if clr is None else clr.method,
        "clearanceSummary": "" if clr is None else clr.summary_line(),
        "clearanceMm": "",
        "clearancePopulation": pop_meta.get("clearancePopulation", ""),
        "nAnalytic": str(pop_meta.get("nAnalytic", "")),
        "nAnalyticFull": str(pop_meta.get("nAnalyticFull", "")),
        "clearanceSampleNote": pop_meta.get("clearanceSampleNote", ""),
        "analyticCap": str(ANALYTIC_CAP),
    }
    return hdr


def _unpack_packed_streamlines(buf: bytes, header: dict) -> np.ndarray:
    """Decode using the K advertised by the packer, never a stale constant."""
    return unpack_streamlines(
        buf,
        int(header["lineCount"]),
        int(header["pointsPerLine"]),
    )


def _validate_filter_minlength(value) -> float:
    """Validate the corpus filter's finite, bounded minimum length."""
    try:
        minlength = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError("filter minlength must be a number") from e
    if not np.isfinite(minlength) or not (0.0 < minlength <= FILTER_MINLENGTH_MAX):
        raise ValueError(
            f"filter minlength must be finite and in (0, {FILTER_MINLENGTH_MAX:g}]"
        )
    return minlength


def _sift2_manifest_receipt(service, bank_id: str, spec) -> tuple[object, object]:
    """Read the explicit bank/weight receipts for one manifest bank."""
    manifest = getattr(service, "manifest", None)
    inputs = manifest.get("inputs", {}) if isinstance(manifest, dict) else {}
    meta = inputs.get(bank_id, {}) if isinstance(inputs, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    bank_sha = meta.get("sha256", meta.get("bank_sha256"))
    weight_sha = meta.get("sift2_sha256", meta.get("sift2Sha256"))
    sift_meta = meta.get("sift2")
    if isinstance(sift_meta, dict):
        weight_sha = sift_meta.get(
            "sha256", sift_meta.get("sift2_sha256", weight_sha)
        )
    prov = getattr(spec, "provenance", None)
    if bank_sha is None and prov is not None:
        bank_sha = getattr(prov, "bank_sha256", None)
    return bank_sha, weight_sha


class TrackService:
    """Holds the resolved case + the prebuilt u8 volume. One per server."""

    def __init__(self, manifest_path: str, file_hashes: FileHashCache | None = None):
        with open(manifest_path) as f:
            man = json.load(f)
        dv.validate(man)
        dv.validate_active_artifact_metadata(man)
        self.manifest = man
        # S-06: validate the derivation lineage BEFORE any artifact below is
        # opened — a malformed or multiply-mixed lineage must refuse at
        # startup, not after tracking has already read from it.
        dv.validate(man)
        self.case_id = man["case_id"]
        root = os.path.realpath(man["case_root"])
        self.inputs = dv.active_inputs(man)
        assume_unknown_units_mm = unknown_units_assumption_from_manifest(man)
        self.assume_unknown_spatial_units_mm = assume_unknown_units_mm
        self.fod = _resolve_case_path(root, self.inputs["fod"]["path"], what="fod")
        self.mask = _resolve_case_path(root, self.inputs["mask"]["path"], what="mask")
        b0 = _resolve_case_path(root, self.inputs["b0"]["path"], what="b0")
        for p in (self.fod, self.mask, b0):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        # Establish the b0 grid before any mask-derived volume, hull, lesion,
        # or seed can be created.  The FOD may be native MIF; assert_grid_match
        # dispatches to its explicit transform/spacing/stride reconstruction.
        reference_grid = load_grid(
            b0,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_units_mm,
            require_authoritative_affine=True,
        )
        assert_grid_match(
            self.mask,
            reference_grid,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_units_mm,
            require_authoritative_affine=True,
        )
        assert_grid_match(
            self.fod,
            reference_grid,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_units_mm,
            require_authoritative_affine=True,
        )
        self.volume = build_u8_volume(
            b0,
            self.mask,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_units_mm,
            require_authoritative_affine=True,
        )
        self.grid = self.volume.grid
        self.grid_id = grid_id(self.grid)
        assert_grid_match(self.mask, self.grid)
        self.mask_vol = build_binary_u8_volume(self.mask)
        self.volume_id = self.volume.volume_id
        self.grid_provenance = {
            **self.grid.unit_provenance,
            "assume_unknown_spatial_units_mm": assume_unknown_units_mm,
        }
        self.space_id = f"dwi-{self.case_id}-v1"
        self.case_root = root
        # Fidelity operating point: owner-signed sheet if present, else the
        # unsigned pilot fallback. A signed sheet that cannot be honoured
        # (off-grid, stale sweep) refuses HERE, at startup, so the owner sees
        # it — never silently downgraded to pilot under a signed name.
        self.operating_point = read_operating_point(
            root,
            sheet_path=dv.operating_point_path(man),
            sweep_path=dv.fidelity_sweep_path(man),
        )
        self._lock = threading.Lock()   # single-flight
        # Follow-up tools resolve a named bank or an explicit result token.
        self.analytic_sources = AnalyticSourceRegistry()
        self._export_dir = tempfile.mkdtemp(prefix=f"tractlab-export-{self.case_id}-")
        # brain hull for 3D anatomical context (built once)
        verts, faces = build_brain_hull(self.mask)
        self._hull_body, self._hull_hdr = pack_mesh(verts, faces)
        cen = verts.reshape(-1, 3).mean(axis=0)
        rad = float(np.linalg.norm(verts.reshape(-1, 3) - cen, axis=1).max())
        self._hull_hdr["center"] = ",".join(f"{x:.3f}" for x in cen)
        self._hull_hdr["radius"] = f"{rad:.3f}"

        self._cortex_body = self._cortex_hdr = None

        # optional T1 anatomy — fail-closed on human-signed t1_qc (Slice A0)
        self.t1_vol = None
        self.t1_qc_approved = dv.t1_qc_ok(man)
        t1_rel = self.inputs.get("t1", {}).get("path")
        if t1_rel and self.t1_qc_approved:
            t1_path = _resolve_case_path(root, t1_rel, what="t1")
            if os.path.exists(t1_path):
                assert_grid_match(
                    t1_path,
                    self.grid,
                    require_mm=True,
                    assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                    require_authoritative_affine=True,
                )
                self.t1_vol = build_u8_volume(
                    t1_path,
                    self.mask,
                    require_mm=True,
                    assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                    require_authoritative_affine=True,
                )
                # The case's own FreeSurfer pial when recon-all + bbregister exist;
                # otherwise the T1 intensity aid. Either way context only.
                fs_cortex = find_freesurfer_cortex(root)
                if fs_cortex:
                    try:
                        cv, cf, cs = build_freesurfer_cortex(fs_cortex["subject"], fs_cortex["transform"])
                        self._cortex_body, self._cortex_hdr = pack_mesh(cv, cf, cs)
                        self._cortex_hdr["Source"] = "freesurfer-pial"
                    except (OSError, ValueError, RuntimeError):
                        self._cortex_body = self._cortex_hdr = None
                if self._cortex_body is None:
                    try:
                        cv, cf = build_cortical_relief(t1_path, self.mask)
                        self._cortex_body, self._cortex_hdr = pack_mesh(cv, cf)
                    except (OSError, ValueError, RuntimeError):
                        self._cortex_body = self._cortex_hdr = None

        # optional FA / DEC underlays for ROI placement (grid-matched, fail closed)
        self.fa_vol = None
        fa_rel = self.inputs.get("fa", {}).get("path")
        if fa_rel:
            fa_path = _resolve_case_path(root, fa_rel, what="fa")
            if os.path.exists(fa_path):
                assert_grid_match(
                    fa_path,
                    self.grid,
                    require_mm=True,
                    assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                    require_authoritative_affine=True,
                )
                # FA is [0,1]-ish — window at masked percentiles like b0
                self.fa_vol = build_u8_volume(
                    fa_path,
                    self.mask,
                    lo_pct=1.0,
                    hi_pct=99.0,
                    require_mm=True,
                    assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                    require_authoritative_affine=True,
                )

        self.dec_vol = None
        dec_rel = self.inputs.get("dec", {}).get("path")
        if dec_rel:
            dec_path = _resolve_case_path(root, dec_rel, what="dec")
            if os.path.exists(dec_path):
                self.dec_vol = build_rgb_u8_volume(
                    dec_path,
                    self.grid,
                    require_mm=True,
                    assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                    require_authoritative_affine=True,
                )

        # optional lesion: binary volume (MPR overlay) + surface mesh + shell (clearance)
        self.lesion_vol = None
        self._lesion_body = self._lesion_hdr = None
        self.lesion_shell = np.empty((0, 3))
        les_rel = self.inputs.get("lesion", {}).get("path")
        les_path = _resolve_case_path(root, les_rel, what="lesion") if les_rel else None
        if les_path and os.path.exists(les_path):
            assert_grid_match(
                les_path,
                self.grid,
                require_mm=True,
                assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                require_authoritative_affine=True,
            )
            self.lesion_vol = build_binary_u8_volume(
                les_path,
                self.grid,
                require_mm=True,
                assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                require_authoritative_affine=True,
            )
            lv, lf = build_brain_hull(les_path, step=1)
            self._lesion_body, self._lesion_hdr = pack_mesh(lv, lf)
            lc = lv.reshape(-1, 3).mean(axis=0)
            self._lesion_hdr["center"] = ",".join(f"{x:.3f}" for x in lc)
            self.lesion_shell = lesion_surface_points(les_path)

        # named seed presets (manifest seed_* keys only — no client paths)
        self.presets = load_preset_catalog(root, self.inputs, self.grid)
        self._preset_vols: dict[str, object] = {}
        for pid, p in self.presets.items():
            self._preset_vols[pid] = build_binary_u8_volume(
                p.path,
                require_mm=True,
                assume_unknown_spatial_units_mm=assume_unknown_units_mm,
                require_authoritative_affine=True,
            )

        # Explore banks: prebuilt multi-ROI extracts from dense ACT whole-brain
        self.banks = discover_banks(root, self.inputs)
        self.bank_sources = BankSourceRegistry(file_hashes)
        for bank_id, spec in self.banks.items():
            self.bank_sources.capture(bank_id, spec.path, spec.sift2_path)
        # C1: lesion ∩ analytic banks. Cached once (first GET), never the 10M corpus.
        self._lesion_path = les_path if (les_path and os.path.exists(les_path)) else None
        # Manifest DECLARED a lesion that is not on disk: invalid, never 404-absent.
        self._lesion_missing_error: str | None = (
            f"manifest declares a lesion but the file is missing ({les_path})"
            if (les_path and not os.path.exists(les_path)) else None
        )
        # Per-case geometric floor (Task 16): resolved from the manifest,
        # fail-soft at boot into an explicit error — floor-dependent surfaces
        # refuse; nothing inherits another case's floor.
        try:
            self.geom_floor_mm: float | None = geom_floor_mm(man)
            self._geom_floor_error: str | None = None
        except (ValueError, OverflowError) as e:
            self.geom_floor_mm = None
            self._geom_floor_error = str(e)
        self.connectotomy_report = None
        self._connectotomy_cut_idx: dict[str, np.ndarray] = {}
        self._connectotomy_bank_digest: dict[str, str] = {}
        self._connectotomy_ready = False
        # Invalid evidence ≠ absent evidence: a failed compute stores its
        # reason here and surfaces as 422, never as the 404 "no cavity".
        self._connectotomy_error: str | None = None
        self._connectotomy_lock = threading.Lock()
        # Atlas priors — fail-closed (ADR-0002): empty unless QC signed + all present
        self.priors = discover_priors(root, self.inputs, man)
        self._prior_mesh_cache: dict[str, tuple[bytes, dict]] = {}
        self._prior_vol_cache: dict[str, bytes] = {}
        # Parcellation prior — fail-closed on parcellation_qc (A2)
        self.parcellation = discover_parcellation(root, self.inputs, man)
        self._parcel_vol_cache: bytes | None = None
        self._parcel_net_cache: bytes | None = None
        self._parcel_mesh_cache: dict[int, tuple[bytes, dict]] = {}
        self._parcel_labels: np.ndarray | None = None
        # live-filter corpus (100k ACT bank — small enough for in-memory filter)
        self.filter_bank_path = None
        self.filter_bank_count = None
        fb = self.inputs.get("filter_bank", {})
        if isinstance(fb, dict) and fb.get("path"):
            fbp = _resolve_case_path(root, fb["path"], what="filter_bank")
            if os.path.isfile(fbp):
                self.filter_bank_path = fbp
        self.filter_bank_note = banks_note(man)
        # Job control for cancel
        self._job_lock = threading.Lock()
        self._active_proc = None
        self._active_job_id = None
        self._active_started = None
        # Provenance: hash of preset/bank recipes (not imaging PHI)
        self.recipe_hash = self._compute_recipe_hash(self.inputs)

    def close(self) -> None:
        """Release service-owned temporary resources at server shutdown."""
        shutil.rmtree(self._export_dir, ignore_errors=True)

    def derivation_id(self) -> str | None:
        return dv.active_derivation(self.manifest or {})

    @staticmethod
    def _compute_recipe_hash(inputs: dict) -> str:
        """Stable fingerprint of active seed/bank/filter definitions."""
        keys = sorted(
            k for k in (inputs or {})
            if k.startswith("seed_") or k.startswith("bank_") or k == "filter_bank"
        )
        payload = {k: inputs[k] for k in keys}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def bank_source_hash(self, bank_id: str) -> str:
        return self.bank_sources.validate(bank_id)

    def cancel_active(self, job_id: str | None = None) -> dict:
        """Best-effort kill of the active tckgen process group.

        SEC-02: cancellation must name the active job. ``job_id=None`` keeps
        this usable for non-browser/local callers (direct Python access is
        outside the browser-origin boundary this contract protects); the
        HTTP ``/api/cancel`` handler always supplies the caller's jobId, so a
        request that does not name the running job can never cancel it.
        """
        with self._job_lock:
            proc = self._active_proc
            jid = self._active_job_id
            if proc is None:
                return {"cancelled": False, "reason": "idle"}
            if job_id is not None and job_id != jid:
                return {"cancelled": False, "reason": "job_id_mismatch", "jobId": jid}
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError) as e:
                return {"cancelled": False, "reason": str(e), "jobId": jid}
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                proc.wait()
            return {"cancelled": True, "jobId": jid}

    def _register_proc(self, proc, job_id: str) -> None:
        with self._job_lock:
            self._active_proc = proc
            self._active_job_id = job_id
            self._active_started = time.time()

    def _connectotomy_population_digests(self) -> dict[str, str]:
        """Fingerprint each bank population in the order used by cuts."""
        return {
            bid: digest_lines(
                load_prebuilt_bundle(spec.path, max_keep=None)[0]
            )
            for bid, spec in self.banks.items()
        }

    def ensure_connectotomy(self):
        """Compute C1 once per case. None when there is no/empty lesion (404).

        Thread-safe: the ready flag flips only AFTER the report is assigned
        (Task 15 — flag-before-compute let a concurrent first request read the
        unset report as a false 404). A ValueError (e.g. lesion grid mismatch)
        is stored in _connectotomy_error — invalid evidence must never be
        indistinguishable from absent evidence.
        """
        if self._connectotomy_ready:
            return self.connectotomy_report
        with self._connectotomy_lock:
            if self._connectotomy_ready:
                return self.connectotomy_report
            report = None
            cut_idx: dict[str, np.ndarray] = {}
            error: str | None = None
            if self._lesion_missing_error is not None:
                error = self._lesion_missing_error
            elif self._lesion_path and self.lesion_vol is not None:
                if self.geom_floor_mm is None:
                    error = self._geom_floor_error
                else:
                    try:
                        try:
                            cavity = load_lesion_bool(self._lesion_path, self.grid)
                        except ValueError as e:
                            # loader validation = cavity fault by definition
                            raise CavityInvalid(str(e)) from e
                        bank_digest_before = self._connectotomy_population_digests()
                        report, cut_idx = compute_connectotomy(
                            self.banks, cavity, self.grid,
                            floor_mm=self.geom_floor_mm,
                        )
                        bank_digest_after = self._connectotomy_population_digests()
                        if bank_digest_before != bank_digest_after:
                            raise _PopulationEvidenceChanged(
                                "bank population changed while computing connectotomy"
                            )
                    except CavityInvalid as e:
                        # ONLY typed cavity faults become 422; any other
                        # ValueError is a server bug and propagates loudly.
                        error = str(e)
                    except _PopulationEvidenceChanged as e:
                        error = str(e)
                        report = None
                        cut_idx = {}
                    else:
                        self._connectotomy_bank_digest = bank_digest_after
            self.connectotomy_report = report
            self._connectotomy_cut_idx = cut_idx
            self._connectotomy_error = error
            self._connectotomy_ready = True
        return self.connectotomy_report

    def _clear_proc(self) -> None:
        with self._job_lock:
            self._active_proc = None
            self._active_job_id = None
            self._active_started = None

    def track(self, req_body: dict, params: TrackParams):
        """Returns (buffer, header_dict) or raises RuntimeError('busy') / RoiCompileError."""
        reject_cavity_in_request(req_body)
        # SEC-02: the viewer mints its own UUID jobId before POSTing so it
        # never has to adopt a job id it read back from /api/health (which
        # could race with someone else's job). Non-browser callers may omit
        # it; the server then mints one, as before.
        raw_job_id = req_body.get("jobId")
        if raw_job_id is not None:
            if not isinstance(raw_job_id, str) or not raw_job_id:
                raise RoiCompileError("jobId must be a non-empty string")
            try:
                uuid.UUID(raw_job_id)
            except ValueError as e:
                raise RoiCompileError("jobId must be a UUID string") from e
            job_id = raw_job_id
        else:
            job_id = uuid.uuid4().hex
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("busy")
        work = None
        try:
            work = tempfile.mkdtemp(prefix=f"tractlab-{self.case_id}-")
            preset_id = req_body.get("seed_preset")
            seed_override = None
            recipe_and: list[np.ndarray] = []
            recipe_not: np.ndarray | None = None
            preset = None
            if preset_id is not None:
                if not isinstance(preset_id, str) or not preset_id:
                    raise RoiCompileError("seed_preset must be a non-empty string id")
                try:
                    seed_override, recipe_and, recipe_not = load_preset_recipe_masks(
                        self.presets, preset_id,
                    )
                    preset = self.presets[preset_id]
                except PresetError as e:
                    raise RoiCompileError(str(e)) from e

            # preset may raise seed budget / minlength (server-owned, never client)
            if preset is not None:
                params = TrackParams(
                    cutoff=params.cutoff,
                    angle=params.angle,
                    minlength=float(preset.minlength) if preset.minlength is not None
                    else params.minlength,
                    maxlength=params.maxlength,
                    seeds=int(preset.seeds) if preset.seeds is not None else params.seeds,
                    select=int(preset.select) if preset.select is not None else params.select,
                    nthreads=params.nthreads,
                    rng_seed=params.rng_seed,
                    density=params.density,
                ).validated()

            layers = layers_from_request(req_body, require_seed=(seed_override is None))
            compiled = compile_roi_layers(
                self.grid, layers, seed_mask_override=seed_override,
            )

            seed_path = os.path.join(work, "seed.nii.gz")
            write_seed_nifti(compiled.seed_mask, self.mask, seed_path)

            and_paths: list[str] = []
            for i, m in enumerate(list(compiled.and_masks) + recipe_and):
                if int(m.sum()) == 0:
                    continue
                p = os.path.join(work, f"and_{i}.nii.gz")
                write_seed_nifti(m, self.mask, p)
                and_paths.append(p)

            or_path = None
            if compiled.or_mask is not None:
                or_path = os.path.join(work, "or.nii.gz")
                write_seed_nifti(compiled.or_mask, self.mask, or_path)

            # paint NOT ∪ recipe NOT
            not_mask = compiled.not_mask
            if recipe_not is not None:
                not_mask = recipe_not if not_mask is None else (not_mask | recipe_not)
            not_path = None
            if not_mask is not None and int(not_mask.sum()) > 0:
                not_path = os.path.join(work, "not.nii.gz")
                write_seed_nifti(not_mask, self.mask, not_path)

            # argv fragment is the single source for include/exclude order:
            # AND regions first (each -include), then optional OR union, then NOT.
            roi_argv = build_tckgen_roi_args(
                seed_path=seed_path,
                and_paths=and_paths,
                or_path=or_path,
                not_path=not_path,
            )
            include_paths = [roi_argv[i + 1] for i, a in enumerate(roi_argv) if a == "-include"]
            excl = None
            if "-exclude" in roi_argv:
                excl = roi_argv[roi_argv.index("-exclude") + 1]

            # Cap wall time hard — dense hang is a product failure (review: 6 min no abort)
            timeout_s = 45.0 if (preset is not None and preset.seeds and preset.seeds >= 100_000) else 30.0
            res = run_tckgen(
                self.fod, seed_path, self.mask, work, params, timeout_s=timeout_s,
                include_paths=include_paths or None,
                exclude_path=excl,
                on_proc=lambda p: self._register_proc(p, job_id),
            )
            self._clear_proc()
            if res.outcome is not Outcome.OK:
                return b"", {"outcome": res.outcome.value, "nAccepted": None,
                             "nReturned": None,
                             "lineCount": 0, "warning": res.warning,
                             "nAnd": len(and_paths),
                             "hasOr": "1" if or_path else "0",
                             "hasNot": "1" if not_path else "0",
                             "seedPreset": preset_id or "",
                             "seedsBudget": params.seeds,
                             "selectCap": params.select,
                             "rngSeed": params.rng_seed,
                             "nthreads": params.nthreads,
                             "jobId": job_id,
                             "recipeHash": self.recipe_hash}

            import nibabel as nib
            lines = list(nib.streamlines.load(res.tck_path).streamlines)
            n_file = len(lines)  # independently decoded population count
            if res.n_accepted != n_file:
                return b"", {
                    "outcome": Outcome.ENGINE_ERROR.value,
                    "nAccepted": None,
                    "nReturned": None,
                    "lineCount": 0,
                    "nFile": n_file,
                    "warning": (
                        f"tracking receipt count {res.n_accepted} does not match "
                        f"decoded TCK count {n_file}"
                    ),
                }
            # ⛔ Do NOT relabel a weak tail as "CST" via z_max post-filter.
            # Brainstem-unreconstructible live iFOD2 → use Explore bank instead.

            # Clearance on analytic population (not display subsample)
            hit_select_cap = (res.n_accepted is not None
                              and res.n_accepted >= params.select)
            analytic, pop_meta, _idx = analytic_subset(lines)
            seed_pts = seed_surface_points(compiled.seed_mask, self.grid)
            if self.geom_floor_mm is None:
                clr = None  # floor unrecorded -> no clearance claim (honest null)
            else:
                clr = clearance_report(
                    analytic, self.lesion_shell, seed_pts=seed_pts,
                    geom_floor_mm=self.geom_floor_mm,
                )

            # Display subsample: length-ranked cosmetic set only
            shown, display_ordinals = _display_subsample(lines)
            buf, hdr, display_ordinals = _pack_display_streamlines(
                shown, display_ordinals, DISPLAY_K, self.space_id,
                minlength_mm=float(params.minlength),
            )

            dist_off = ""
            if self.lesion_shell.shape[0] and hdr["lineCount"]:
                verts = _unpack_packed_streamlines(buf, hdr)
                dist = per_vertex_distance_mm(verts, self.lesion_shell)
                if dist is not None:
                    dist_off = str(len(buf))
                    buf = buf + dist.tobytes(order="C")
            buf, hdr = _append_source_ordinal_block(
                buf,
                hdr,
                display_ordinals,
                source_population="live-track",
            )

            accept_ratio = ""
            if res.n_accepted is not None and params.seeds > 0:
                accept_ratio = f"{res.n_accepted / params.seeds:.4f}"

            warn = res.warning or ""
            if hit_select_cap:
                warn = (warn + " | " if warn else "") + (
                    f"select CAP filled ({params.select}); count is a quota not strength"
                )
            if n_file < params.select and res.n_accepted is not None:
                warn = (warn + " | " if warn else "") + (
                    f"under-select: only {res.n_accepted} of {params.select} "
                    f"after {params.seeds} seeds (budget exhausted or rare pathway)"
                )
            if pop_meta.get("clearanceSampleNote"):
                warn = (warn + " | " if warn else "") + pop_meta["clearanceSampleNote"]

            hdr.update({
                "outcome": "ok",
                "nAccepted": n_file,
                "nReturned": n_file,
                "nDisplayed": hdr["lineCount"],
                "nFile": n_file,
                "seedsBudget": params.seeds,
                "selectCap": params.select,
                "hitSelectCap": "1" if hit_select_cap else "0",
                "acceptRatio": accept_ratio,
                "rngSeed": params.rng_seed,
                "nthreads": params.nthreads,
                "deterministic": "1" if params.nthreads == 0 else "0",
                "wall_s": round(res.wall_s, 3),
                "engine": (
                    f"iFOD2 | nthreads={params.nthreads} | "
                    + ("deterministic" if params.nthreads == 0
                       else f"stochastic RNG={params.rng_seed}")
                ),
                **_clearance_headers(clr, pop_meta, self._geom_floor_error),
                "distanceOffset": dist_off,
                "nAnd": len(and_paths),
                "hasOr": "1" if or_path else "0",
                "hasNot": "1" if not_path else "0",
                "seedPreset": preset_id or "",
                "seedVoxels": int(compiled.seed_mask.sum()),
                "jobId": job_id,
                "recipeHash": self.recipe_hash,
                "liveHash": params.live_hash(seed_preset=preset_id),
                "cutoff": f"{params.cutoff:.4f}",
                "cutoffKind": "FOD amplitude (not tensor FA)",
                "density": params.density,
                "minlengthMm": params.minlength,
                "angleDeg": params.angle,
                "warning": warn or None,
            })
            hdr.update(self.analytic_sources.commit_result(lines, kind="live").response_headers())
            return buf, hdr
        finally:
            self._clear_proc()
            if work:
                shutil.rmtree(work, ignore_errors=True)
            self._lock.release()


def make_handler(service: TrackService, viewer_dir: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # A dead or stalled peer must fail loud instead of blocking a handler
        # thread indefinitely (applies to rfile reads and wfile sendall alike).
        timeout = 30

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body=b"", ctype="application/octet-stream", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:
                self.send_header("Connection", "close")
            for k, v in (extra or {}).items():
                self.send_header(k, header_ascii(v))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _checked_bank_source_hash(self, bank_id):
            checker = getattr(service, "bank_source_hash", None)
            if checker is None:  # lightweight route tests supply a partial service
                return ""
            try:
                return checker(bank_id)
            except BankSourceChanged:
                self._json(409, {
                    "error": (
                        "named bank source changed; restart with "
                        f"./serve.sh {self.server.server_address[1]}"
                    ),
                    "code": "bank_source_changed",
                    "restartRequired": True,
                    "bankId": bank_id,
                })
                return None

        def _runtime_current(self):
            payload = service.runtime_identity.restart_payload(self.server.server_address[1])
            if payload is None:
                return True
            if self.path.split("?", 1)[0] in ("/", "/index.html"):
                message = str(payload["error"])
                body = (
                    "<!doctype html><title>TractLab restart required</title>"
                    f"<main><h1>Restart required</h1><p>{message}</p></main>"
                ).encode("utf-8")
                self._send(409, body, "text/html")
            else:
                self._json(409, payload)
            return False
        def _enforce_browser_origin_policy(self):
            """SEC-01: same-origin/Fetch-Metadata checks for every request."""
            port = self.server.server_address[1]
            if len(self.headers.get_all("Host", [])) != 1 or len(self.headers.get_all("Origin", [])) > 1:
                raise PolicyRejected("ambiguous Host or Origin", code="bad_host")
            http_policy.enforce_browser_origin_policy(
                host_header=self.headers.get("Host"),
                origin_header=self.headers.get("Origin"),
                sec_fetch_site=self.headers.get("Sec-Fetch-Site"),
                sec_fetch_mode=self.headers.get("Sec-Fetch-Mode"),
                sec_fetch_dest=self.headers.get("Sec-Fetch-Dest"),
                method=self.command,
                expected_port=port,
            )

        def _reject_policy(self, exc: PolicyRejected):
            self.close_connection = True
            status = 403 if exc.code in ("bad_host", "bad_origin", "cross_site") else 400
            return self._json(status, {"error": str(exc), "code": exc.code})

        def do_GET(self):
            try:
                self._enforce_browser_origin_policy()
            except PolicyRejected as e:
                return self._reject_policy(e)
            if not self._runtime_current():
                return
            try:
                return self._do_GET_inner()
            except (BrokenPipeError, ConnectionResetError):
                return None
            except ResultTooLarge as e:
                return self._json(422, {"error": str(e), "code": "result_too_large"})
            except ResultUnavailable as e:
                return self._json(404, {"error": str(e), "code": "result_unavailable"})
            except Exception:
                # S-08: an unexpected failure is a typed unavailable response,
                # never a closed socket and never a traceback (which could
                # name this case's on-disk paths) reaching the client.
                return self._json(503, {"error": "internal error", "code": "service_unavailable"})

        def _do_GET_inner(self):
            if self.path == "/api/preflight":
                from .preflight import compute_drift, load_receipt, run_preflight

                try:
                    stored = load_receipt(service.case_root)
                except ValueError:
                    return self._json(409, {
                        "error": "preflight receipt requires re-review",
                        "code": "preflight_receipt_incompatible",
                    })
                except OSError:
                    return self._json(503, {
                        "error": "preflight receipt unavailable",
                        "code": "preflight_unavailable",
                    })
                try:
                    verified = run_preflight(service.case_root, manifest=service.manifest).to_json()
                    active = (service.manifest or {}).get("active_derivation")
                    if not isinstance(active, str):
                        active = None
                    drift = compute_drift(stored, verified, active_derivation=active)
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    return self._json(503, {
                        "error": "preflight unavailable",
                        "code": "preflight_unavailable",
                        "reason": str(e),
                    })
                return self._json(200, {
                    "stored": stored,
                    "verified": verified,
                    "drift": drift,
                })
            if self.path in ("/api/health", "/api/bootstrap"):
                busy = service._lock.locked()
                with service._job_lock:
                    jid = service._active_job_id
                    jstart = service._active_started
                payload = {
                    "status": "ok", "caseId": service.case_id,
                    "gridId": service.grid_id,
                    "gridProvenance": service.grid_provenance,
                    "volumeId": service.volume_id,
                    "spaceId": service.space_id,
                    "recipeHash": service.recipe_hash,
                    "presetCount": len(service.presets),
                    "bankCount": len(service.banks),
                    "hasFilterBank": service.filter_bank_path is not None,
                    "filterBankCount": service.filter_bank_count,
                    "hasLesion": service.lesion_shell is not None and getattr(service.lesion_shell, "shape", (0,))[0] > 0,
                    "sift2BankCount": sum(
                        1 for b in service.banks.values() if b.sift2_path
                    ),
                    "recovery": {
                        "perilesional": True,
                        "radiusMinMm": RECOVERY_RADIUS_MIN,
                        "radiusMaxMm": RECOVERY_RADIUS_MAX,
                        "radiusDefaultMm": RECOVERY_RADIUS_DEFAULT,
                        "note": "Fibres within radius of lesion — not named-bundle proof",
                    },
                    "displayBias": {
                        "nearLesionFracDefault": 0.5,
                        "nearRadiusMmDefault": 12.0,
                        "note": "Display-only sample bias toward peri-lesional streamlines",
                    },
                    "hasFa": service.fa_vol is not None,
                    "hasDec": service.dec_vol is not None,
                    "hasT1": service.t1_vol is not None,
                    "hasPriors": len(service.priors) > 0,
                    "priorCount": len(service.priors),
                    "hasParcellation": service.parcellation is not None,
                    "busy": busy,
                    "activeJobId": jid,
                    "activeJobAgeS": (
                        None if jstart is None
                        else round(time.time() - jstart, 2)
                    ),
                    "uncertainty": health_uncertainty(service.manifest),
                    # commercial live knobs (Commit only; bank Explore ignores these)
                    "liveDefaults": {
                        "cutoff": CUTOFF_DEFAULT,
                        "cutoffMin": CUTOFF_MIN,
                        "cutoffMax": CUTOFF_MAX,
                        "cutoffKind": "FOD amplitude (not tensor FA)",
                        "angle": ANGLE_DEFAULT,
                        "minlength": MINLENGTH_DEFAULT,
                        "density": "normal",
                        "densities": {
                            k: {"seeds": v[0], "select": v[1]}
                            for k, v in DENSITY_PRESETS.items()
                        },
                    },
                }
                payload.update(service.runtime_identity.public_metadata())
                return self._json(200, payload)
            if self.path == "/api/derivation":
                manifest = service.manifest or {}
                return self._json(200, dv.derivation_summary(manifest))
            if self.path == "/api/presets":
                return self._json(200, {"presets": catalog_public(service.presets)})
            if self.path == "/api/banks":
                # default / true anatomy banks first, then label
                _role_rank = {
                    "true_cst": 0, "true_fat": 1,
                    "true_slf1": 2, "true_slf2": 3, "true_slf3": 4,
                    "soft_fat": 5, "soft_slf1": 6, "soft_slf2": 7, "soft_slf3": 8,
                    "true_ifof": 9, "true_uf": 10, "true_cing": 11,
                    "true_or": 12, "true_cc": 13,
                }
                ordered = sorted(
                    service.banks.items(),
                    key=lambda kv: (0 if kv[1].default else 1,
                                    _role_rank.get(kv[1].role, 9),
                                    kv[1].label),
                )
                banks = [
                    {
                        "id": k,
                        "label": b.label,
                        "nStreamlines": b.n_streamlines,
                        "nStreamlinesEstimate": b.n_streamlines,
                        "nStreamlinesStatus": "estimate",
                        "engine": b.engine,
                        "note": b.note,
                        "default": b.default,
                        "role": b.role,
                        "hasSift2": bool(b.sift2_path),
                    }
                    for k, b in ordered
                ]
                default_id = next((k for k, b in ordered if b.default), None)
                return self._json(200, {
                    "banks": banks,
                    "defaultBankId": default_id,
                    "filterBank": bool(service.filter_bank_path),
                    "disclaimer": (
                        f"{banks_note(service.manifest)} | manifest streamline counts "
                        "are estimates; load responses report actual decoded counts"
                    ),
                })
            if self.path == "/api/connectotomy":
                report = service.ensure_connectotomy()
                if service._connectotomy_error is not None:
                    return self._json(422, {
                        "error": "connectotomy-invalid",
                        "reason": service._connectotomy_error,
                    })
                if report is None:
                    return self._send(404, b"no lesion cavity", "text/plain")
                return self._json(200, report)
            if self.path.startswith("/api/connectotomy/") and self.path.endswith("/cut"):
                return self._handle_connectotomy_cut()
            if self.path.split("?", 1)[0].startswith("/api/profile/"):
                return self._handle_profile()
            if self.path == "/api/connectome":
                return self._handle_connectome()
            if self.path.startswith("/api/connectome/edge/") and self.path.endswith("/tubes"):
                return self._handle_connectome_edge_tubes()
            if self.path.startswith("/api/connectome/edge/"):
                return self._handle_connectome_edge()
            if self.path.startswith("/api/presets/") and self.path.endswith("/volume"):
                # /api/presets/<id>/volume
                parts = self.path.strip("/").split("/")
                if len(parts) != 4 or parts[0] != "api" or parts[1] != "presets":
                    return self._send(404, b"not found", "text/plain")
                pid = parts[2]
                vol = service._preset_vols.get(pid)
                if vol is None:
                    return self._json(404, {"error": f"unknown preset: {pid}"})
                extra = {"X-Shape": ",".join(map(str, vol.grid.shape)),
                         "X-Order": vol.grid.order, "X-Grid-Id": service.grid_id,
                         "X-Preset-Id": pid,
                         "X-N-Voxels": str(service.presets[pid].n_voxels)}
                return self._send(200, vol.to_bytes(), extra=extra)
            if self.path == "/api/volume/b0":
                buf = service.volume.to_bytes()
                extra = {
                    "X-Shape": ",".join(map(str, service.grid.shape)),
                    "X-Order": service.grid.order,
                    "X-Grid-Id": service.grid_id,
                    "X-Volume-Id": service.volume_id,
                    "X-Affine": ";".join(f"{x:.8f}" for x in service.grid.affine.ravel()),
                    "X-Window": ",".join(f"{x:.4f}" for x in service.volume.window),
                    "X-Channels": "1",
                    "X-Format": "gray",
                }
                return self._send(200, buf, extra=extra)
            if self.path == "/api/volume/mask":
                return self._send(200, service.mask_vol.to_bytes(), extra={
                    "X-Shape": ",".join(map(str, service.grid.shape)),
                    "X-Grid-Id": service.grid_id,
                    "X-Order": service.grid.order,
                    "X-Label": "Recorded DWI support mask",
                })
            if self.path == "/api/volume/fa":
                if service.fa_vol is None:
                    return self._send(404, b"FA underlay not available", "text/plain")
                extra = {
                    "X-Shape": ",".join(map(str, service.fa_vol.grid.shape)),
                    "X-Order": service.fa_vol.grid.order,
                    "X-Grid-Id": service.grid_id,
                    "X-Window": ",".join(f"{x:.4f}" for x in service.fa_vol.window),
                    "X-Channels": "1",
                    "X-Format": "gray",
                    "X-Label": "FA (tensor; not iFOD2 cutoff)",
                }
                return self._send(200, service.fa_vol.to_bytes(), extra=extra)
            if self.path == "/api/volume/dec":
                if service.dec_vol is None:
                    return self._send(404, b"DEC underlay not available", "text/plain")
                extra = {
                    "X-Shape": ",".join(map(str, service.dec_vol.grid.shape)),
                    "X-Order": service.dec_vol.grid.order,
                    "X-Grid-Id": service.grid_id,
                    "X-Channels": "3",
                    "X-Format": "rgb-planes",
                    "X-Label": "FOD DEC (RGB)",
                }
                return self._send(200, service.dec_vol.to_bytes(), extra=extra)
            if self.path == "/api/surface/brain":
                extra = {f"X-{k}": v for k, v in service._hull_hdr.items()}
                return self._send(200, service._hull_body, extra=extra)
            if self.path == "/api/surface/cortex":
                if service._cortex_body is None:
                    return self._send(404, b"cortical relief not available", "text/plain")
                extra = {f"X-{k}": v for k, v in service._cortex_hdr.items()}
                extra["X-Label"] = (FS_CORTEX_LABEL if service._cortex_hdr.get("Source") == "freesurfer-pial"
                                    else CORTICAL_AID_LABEL)
                return self._send(200, service._cortex_body, extra=extra)
            if self.path in ("/api/volume/t1", "/api/volume/lesion"):
                vol = service.t1_vol if self.path.endswith("t1") else service.lesion_vol
                if vol is None:
                    return self._send(404, b"volume not available", "text/plain")
                extra = {"X-Shape": ",".join(map(str, vol.grid.shape)),
                         "X-Order": vol.grid.order, "X-Grid-Id": service.grid_id,
                         "X-Window": ",".join(f"{x:.4f}" for x in vol.window),
                         "X-Channels": "1", "X-Format": "gray"}
                if self.path.endswith("t1"):
                    extra["X-Label"] = t1_header_label(service.manifest)
                return self._send(200, vol.to_bytes(), extra=extra)
            if self.path == "/api/surface/lesion":
                if service._lesion_body is None:
                    return self._send(404, b"no lesion", "text/plain")
                extra = {f"X-{k}": v for k, v in service._lesion_hdr.items()}
                return self._send(200, service._lesion_body, extra=extra)
            if self.path == "/api/priors":
                # Fail closed: empty list when QC unsigned or priors missing
                priors = [
                    {
                        "id": p.id,
                        "label": p.label,
                        "officialName": p.official_name,
                        "atlasIndex": p.atlas_index,
                        "note": p.note,
                    }
                    for p in sorted(service.priors.values(), key=lambda x: x.id)
                ]
                return self._json(200, {
                    "priors": priors,
                    "disclaimer": "POPULATION ATLAS — not patient anatomy; not navigation",
                    "available": len(priors) > 0,
                })
            if self.path.startswith("/api/priors/") and (
                self.path.endswith("/volume") or self.path.endswith("/mesh")
            ):
                parts = self.path.strip("/").split("/")
                # api priors <id> volume|mesh
                if len(parts) != 4 or parts[0] != "api" or parts[1] != "priors":
                    return self._send(404, b"not found", "text/plain")
                pid = parts[2]
                if pid not in service.priors:
                    return self._send(404, b"prior not available", "text/plain")
                spec = service.priors[pid]
                kind = parts[3]
                try:
                    if kind == "volume":
                        if pid not in service._prior_vol_cache:
                            u8 = prior_to_u8(spec.path, service.grid)
                            service._prior_vol_cache[pid] = np.asfortranarray(u8).tobytes(order="F")
                        body = service._prior_vol_cache[pid]
                        extra = {
                            "X-Shape": ",".join(map(str, service.grid.shape)),
                            "X-Order": service.grid.order,
                            "X-Grid-Id": service.grid_id,
                            "X-Prior-Id": pid,
                            "X-Channels": "1",
                            "X-Format": "gray",
                            "X-Label": "POPULATION ATLAS prior (u8 prob)",
                            "X-Provenance": "population-atlas",
                        }
                        return self._send(200, body, extra=extra)
                    # mesh
                    if pid not in service._prior_mesh_cache:
                        body, mh = pack_prior_mesh(spec.path, service.grid, level=0.25)
                        service._prior_mesh_cache[pid] = (body, mh)
                    body, mh = service._prior_mesh_cache[pid]
                    extra = {
                        "X-vertexCount": str(mh["vertexCount"]),
                        "X-faceCount": str(mh["faceCount"]),
                        "X-Prior-Id": pid,
                        "X-Level": "0.25",
                        "X-Provenance": "population-atlas",
                        "X-note": "ghost shell display only; not patient tract",
                    }
                    return self._send(200, body, extra=extra)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            if self.path == "/api/parcellation":
                if service.parcellation is None:
                    # Honest pending vs absent — UI should not look empty/broken
                    pqc = dv.qc_block_for(
                        service.manifest or {}, "parcellation_qc"
                    ) or {}
                    auto = pqc.get("auto") or {}
                    has_input = isinstance(
                        service.inputs.get("parc_schaefer200_yeo7"),
                        dict,
                    )
                    reason = "absent"
                    if has_input and not pqc.get("approved_by"):
                        reason = "awaiting_human_qc"
                    elif has_input and auto and not auto.get("ok", False):
                        reason = "auto_qc_failed"
                    return self._json(200, {
                        "available": False,
                        "reason": reason,
                        "autoOk": bool(auto.get("ok")),
                        "sheetPath": pqc.get("sheet_path"),
                        "networks": [],
                        "disclaimer": (
                            "POPULATION ATLAS — parcellation pending human QC"
                            if reason == "awaiting_human_qc"
                            else "POPULATION ATLAS — parcellation not available"
                        ),
                    })
                spec = service.parcellation
                return self._json(200, {
                    "available": True,
                    "reason": "ok",
                    "id": "parc_schaefer200_yeo7",
                    "label": spec.label,
                    "nParcels": spec.n_parcels,
                    "nNetworks": spec.n_networks,
                    "networks": network_public_list(),
                    "disclaimer": "POPULATION ATLAS — territory prior, not patient",
                    "note": spec.note,
                })
            if self.path == "/api/parcellation/lut":
                # Parcel id → name/network for MPR cursor probe (population labels only)
                if service.parcellation is None:
                    return self._json(404, {"error": "parcellation not available"})
                try:
                    from .atlas_fetch import load_schaefer_lut, sha256_file
                    from pathlib import Path as _P
                    lut_path = _P(service.parcellation.lut_path)
                    lut_rel = os.path.relpath(
                        os.path.realpath(lut_path), os.path.realpath(service.case_root),
                    )
                    if lut_rel == ".." or lut_rel.startswith(".." + os.sep):
                        return self._json(500, {"error": "parcellation LUT is outside case_root"})
                    lut = load_schaefer_lut(lut_path)
                    labels = {
                        str(lab): {
                            "id": int(lab),
                            "name": str(meta.get("name") or ""),
                            "networkId": int(meta.get("network_id") or 0),
                            "networkName": str(meta.get("network_name") or ""),
                            "hemi": str(meta.get("hemi") or "?"),
                        }
                        for lab, meta in lut.items()
                    }
                    return self._json(200, {
                        "provenance": "population-atlas",
                        "disclaimer": "POPULATION ATLAS — Schaefer labels, not patient cortex",
                        "nLabels": len(labels),
                        "labels": labels,
                        # So a viewer panel that also reads a connectome's
                        # provenance.parcellation.lut_sha256 (built via the
                        # same sha256_file) can fail closed on a mismatch
                        # instead of silently drawing from a different LUT.
                        "lutSha256": sha256_file(lut_path),
                        # Case-root-relative, same form as the connectome's
                        # provenance.parcellation.lut_path — never absolute.
                        "lutPath": lut_rel,
                    })
                except Exception as e:
                    return self._json(400, {"error": str(e)})
            if self.path in ("/api/parcellation/volume", "/api/parcellation/networks"):
                if service.parcellation is None:
                    return self._send(404, b"parcellation not available", "text/plain")
                try:
                    if service._parcel_labels is None:
                        service._parcel_labels = load_label_volume(
                            service.parcellation.path, service.grid,
                        )
                    if self.path.endswith("/volume"):
                        if service._parcel_vol_cache is None:
                            u8 = labels_u8_for_mpr(service._parcel_labels)
                            service._parcel_vol_cache = np.asfortranarray(u8).tobytes(order="F")
                        body = service._parcel_vol_cache
                        fmt, lab = "labels-u8", "Schaefer-200 labels"
                    else:
                        # network id volume 0–7 for MPR wash
                        if not hasattr(service, "_parcel_net_cache") or service._parcel_net_cache is None:
                            from .atlas_fetch import load_schaefer_lut, label_to_network_map
                            from pathlib import Path as _P
                            lut = load_schaefer_lut(_P(service.parcellation.lut_path))
                            mapping = label_to_network_map(lut)
                            net = np.zeros(service._parcel_labels.shape, dtype=np.uint8)
                            for lab_i, nid in mapping.items():
                                if 1 <= nid <= 7:
                                    net[service._parcel_labels == lab_i] = nid
                            service._parcel_net_cache = np.asfortranarray(net).tobytes(order="F")
                        body = service._parcel_net_cache
                        fmt, lab = "network-u8", "Yeo-7 network ids"
                    extra = {
                        "X-Shape": ",".join(map(str, service.grid.shape)),
                        "X-Order": service.grid.order,
                        "X-Grid-Id": service.grid_id,
                        "X-Format": fmt,
                        "X-Provenance": "population-atlas",
                        "X-Label": lab,
                    }
                    return self._send(200, body, extra=extra)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            if self.path.startswith("/api/parcellation/network/") and self.path.endswith("/mesh"):
                if service.parcellation is None:
                    return self._send(404, b"parcellation not available", "text/plain")
                parts = self.path.strip("/").split("/")
                # api parcellation network <n> mesh
                if len(parts) != 5:
                    return self._send(404, b"not found", "text/plain")
                try:
                    nid = int(parts[3])
                except ValueError:
                    return self._json(400, {"error": "network id must be 1..7"})
                if nid < 1 or nid > 7:
                    return self._json(400, {"error": "network id must be 1..7"})
                try:
                    if nid not in service._parcel_mesh_cache:
                        body, mh = pack_network_mesh(
                            service.parcellation.path,
                            service.grid,
                            service.parcellation.lut_path,
                            nid,
                        )
                        service._parcel_mesh_cache[nid] = (body, mh)
                    body, mh = service._parcel_mesh_cache[nid]
                    extra = {
                        "X-vertexCount": str(mh["vertexCount"]),
                        "X-faceCount": str(mh["faceCount"]),
                        "X-Network-Id": str(nid),
                        "X-Provenance": "population-atlas",
                        "X-note": "network territory ghost; not patient anatomy",
                    }
                    return self._send(200, body, extra=extra)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            return self._serve_static()


        def _serve_static(self):
            """SEC-01: canonical ancestry check, including symlinks.

            ``realpath`` resolves symlinks on both sides before comparing, so
            a symlink planted inside ``viewer_dir`` that points outside the
            tree cannot be used to read arbitrary files. The separator-
            qualified prefix compare also refuses the classic false-positive
            where a plain ``str.startswith`` would accept a sibling
            directory that merely shares the same string prefix (e.g.
            ``viewer_dir + "_evil"``).
            """
            from urllib.parse import unquote, urlsplit

            rel = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
            viewer_root = os.path.realpath(viewer_dir)
            full = os.path.realpath(os.path.join(viewer_root, rel))
            if (
                (full != viewer_root and not full.startswith(viewer_root + os.sep))
                or not os.path.isfile(full)
            ):
                return self._send(404, b"not found", "text/plain")
            ctype = ("text/html" if full.endswith(".html")
                     else "text/css" if full.endswith(".css")
                     else "application/javascript" if full.endswith(".js")
                     else "text/css" if full.endswith(".css")
                     else "image/svg+xml" if full.endswith(".svg")
                     else "application/wasm" if full.endswith(".wasm")
                     else "application/json" if full.endswith(".json")
                     else "text/plain" if full.endswith((".txt", ".md"))
                     else "application/octet-stream")
            with open(full, "rb") as f:
                body = f.read()
            if not self._runtime_current():
                return
            if os.path.basename(full) == "index.html":
                body = service.runtime_identity.inject_document_meta(body)
            return self._send(200, body, ctype)

        def do_POST(self):
            try:
                self._enforce_browser_origin_policy()
            except PolicyRejected as e:
                return self._reject_policy(e)
            if not self._runtime_current():
                return
            try:
                return self._do_POST_inner()
            except (BrokenPipeError, ConnectionResetError):
                return None
            except PolicyRejected as e:
                return self._reject_policy(e)
            except ResultTooLarge as e:
                return self._json(422, {"error": str(e), "code": "result_too_large"})
            except ResultUnavailable as e:
                return self._json(404, {"error": str(e), "code": "result_unavailable"})
            except Exception:
                return self._json(503, {"error": "internal error", "code": "service_unavailable"})

        def _do_POST_inner(self):
            if self.path not in (
                "/api/track", "/api/bank/load", "/api/filter", "/api/cancel",
                "/api/export/tck", "/api/margin", "/api/recovery/perilesional",
            ):
                self.close_connection = True
                return self._send(404, b"not found", "text/plain")
            if self.headers.get_all("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) > 1:
                raise PolicyRejected("ambiguous request framing", code="bad_content_length")
            # cancel may send an empty body; every other endpoint requires one.
            n = http_policy.check_content_length(
                self.headers.get("Content-Length"),
                max_body=MAX_BODY,
                allow_zero=(self.path == "/api/cancel"),
            )
            raw = self.rfile.read(n) if n > 0 else b""
            if len(raw) != n:
                raise PolicyRejected("incomplete request body", code="bad_content_length")
            req = {}
            if n > 0:
                http_policy.check_json_content_type(self.headers.get("Content-Type"))
                try:
                    req = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return self._json(400, {"error": "invalid JSON", "code": "bad_json"})
                if not isinstance(req, dict):
                    return self._json(400, {
                        "error": "request body must be a JSON object",
                        "code": "bad_json_shape",
                    })

            if self.path == "/api/cancel":
                job_id = req.get("jobId")
                if job_id is not None and not isinstance(job_id, str):
                    return self._json(400, {"error": "jobId must be a string", "code": "bad_request"})
                if job_id is None:
                    # SEC-02: no ambient "cancel whatever is active" fallback
                    # over HTTP — an absent jobId reports status only.
                    with service._job_lock:
                        active_jid = service._active_job_id
                    if active_jid is None:
                        return self._json(200, {"cancelled": False, "reason": "idle"})
                    return self._json(200, {
                        "cancelled": False, "reason": "job_id_required", "jobId": active_jid,
                    })
                result = service.cancel_active(job_id=job_id)
                return self._json(200, result)

            # The result already binds its grid/volume. These consumers need
            # only that explicit immutable population ID, never ambient state.
            if self.path == "/api/export/tck":
                return self._handle_export(req)
            if self.path == "/api/margin":
                return self._handle_margin(req)
            if req.get("gridId") != service.grid_id or req.get("volumeId") != service.volume_id:
                return self._json(409, {
                    "error": "stale grid/volume — reload the volume",
                    "code": "stale_volume",
                })

            if self.path == "/api/bank/load":
                return self._handle_bank_load(req)
            if self.path == "/api/filter":
                return self._handle_filter(req)
            if self.path == "/api/recovery/perilesional":
                return self._handle_recovery_perilesional(req)

            p = req.get("params", {})
            if not isinstance(p, dict):
                return self._json(400, {"error": "params must be an object", "code": "bad_request"})
            try:
                density = str(p.get("density", "normal")).lower().strip()
                if density not in DENSITY_PRESETS:
                    return self._json(400, {
                        "error": f"density must be one of {sorted(DENSITY_PRESETS)}",
                    })
                seeds, select = DENSITY_PRESETS[density]
                # cutoff = commercial "FA threshold" analogue; FOD amplitude only
                cutoff = float(p.get("cutoff", CUTOFF_DEFAULT))
                params = TrackParams(
                    cutoff=cutoff,
                    angle=float(p.get("angle", ANGLE_DEFAULT)),
                    minlength=float(p.get("minlength", MINLENGTH_DEFAULT)),
                    seeds=seeds,
                    select=select,
                    density=density,
                    # nthreads/rng_seed remain server-owned
                ).validated()
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": f"bad params: {e}"})

            try:
                buf, hdr = service.track(req, params)
            except (RoiCompileError, PresetError, CavityRoleError) as e:
                return self._json(400, {"error": str(e)})
            except RuntimeError as e:
                if str(e) == "busy":
                    return self._json(409, {
                        "error": "a track job is already running",
                        "code": "busy",
                        "activeJobId": service._active_job_id,
                    })
                raise
            extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            return self._send(200, buf, extra=extra)

        def _pack_lines(
            self,
            lines,
            engine_label,
            extra_hdr=None,
            seed_mask=None,
            weights=None,
            *,
            source_population: str,
            near_lesion_frac: float = 0.0,
            near_radius_mm: float = 12.0,
        ):
            """Pack display tubes; clearance on analytic population (not length-rank).

            Does NOT commit a result id; callers commit only after every
            endpoint-specific refusal gate has passed.
            """
            if not lines:
                hdr = {"outcome": "ok", "nAccepted": 0, "nReturned": 0, "lineCount": 0,
                       "engine": engine_label, "warning": "no streamlines after filter",
                       "sourcePopulation": source_population, "nAnalyticFull": "0"}
                if extra_hdr:
                    hdr.update(extra_hdr)
                return b"", hdr, np.zeros(0, dtype=np.int64)
            n_full = len(lines)
            # Analytic first (full or random sample) — never the display set
            analytic, pop_meta, aidx = analytic_subset(lines)
            seed_pts = None
            if seed_mask is not None:
                seed_pts = seed_surface_points(seed_mask, service.grid)
            w_an = None
            if weights is not None:
                ww = np.asarray(weights, dtype=np.float64).ravel()
                if len(ww) == n_full:
                    w_an = ww[aidx]
            if service.geom_floor_mm is None:
                clr = None  # floor unrecorded -> no clearance claim
            elif w_an is not None:
                clr = clearance_report_weighted(
                    analytic, w_an, service.lesion_shell, seed_pts=seed_pts,
                    geom_floor_mm=service.geom_floor_mm,
                )
                pop_meta = dict(pop_meta)
                pop_meta["clearanceSampleNote"] = (
                    (pop_meta.get("clearanceSampleNote") or "")
                    + " | SIFT2-weighted p5"
                ).strip(" |")
            else:
                clr = clearance_report(
                    analytic, service.lesion_shell, seed_pts=seed_pts,
                    geom_floor_mm=service.geom_floor_mm,
                )

            # Stable seed from n_full so the same bank redraws consistently
            shell = service.lesion_shell if (
                service.lesion_shell is not None
                and getattr(service.lesion_shell, "shape", (0,))[0] > 0
            ) else None
            near_frac = float(near_lesion_frac or 0.0)
            shown, ordinals = _display_subsample(
                lines,
                weights=weights if weights is not None else None,
                seed=(n_full * 2654435761) & 0xFFFFFFFF,
                lesion_shell=shell if near_frac > 0 else None,
                near_lesion_frac=near_frac,
                near_radius_mm=float(near_radius_mm or 12.0),
            )
            min_L = None
            if extra_hdr and extra_hdr.get("minlengthMm") not in (None, ""):
                try:
                    min_L = float(extra_hdr["minlengthMm"])
                except (TypeError, ValueError):
                    min_L = None
            buf, hdr, ordinals = _pack_display_streamlines(
                shown, ordinals, DISPLAY_K, service.space_id, minlength_mm=min_L,
            )
            dist_off = ""
            if service.lesion_shell.shape[0] and hdr["lineCount"]:
                verts = _unpack_packed_streamlines(buf, hdr)
                dist = per_vertex_distance_mm(verts, service.lesion_shell)
                if dist is not None:
                    dist_off = str(len(buf))
                    buf = buf + dist.tobytes(order="C")
            buf, hdr = _append_source_ordinal_block(
                buf,
                hdr,
                ordinals,
                source_population=source_population,
            )
            # Keep warning short: full disclaimer lives in the page banner.
            # Do not re-paste filter_bank_note on every chip click.
            warn_bits = []
            if pop_meta.get("clearanceSampleNote"):
                warn_bits.append(str(pop_meta["clearanceSampleNote"]))
            if weights is not None and near_frac <= 0:
                warn_bits.append("display: SIFT2 top-k")
            elif weights is not None and near_frac > 0:
                warn_bits.append(
                    f"display: SIFT2 + near-lesion {int(near_frac*100)}% "
                    f"(r≤{float(near_radius_mm):.0f}mm)"
                )
            elif near_frac > 0 and n_full > DISPLAY_CAP:
                warn_bits.append(
                    f"display: near-lesion bias {int(near_frac*100)}% "
                    f"(r≤{float(near_radius_mm):.0f}mm) + mid-length"
                )
            elif n_full > DISPLAY_CAP:
                warn_bits.append("display: mid-length sample (not longest)")
            warn = " | ".join(warn_bits)
            hdr.update({
                "outcome": "ok",
                "nAccepted": n_full,
                "nReturned": n_full,
                "nDisplayed": hdr["lineCount"],
                "hitSelectCap": "0",
                "engine": engine_label,
                **_clearance_headers(clr, pop_meta, service._geom_floor_error),
                "distanceOffset": dist_off,
                "warning": warn,
            })
            if extra_hdr:
                hdr.update(extra_hdr)
            if w_an is not None:
                hdr["hasSift2"] = "1"
            return buf, hdr, ordinals

        def _parse_display_bias(self, req) -> tuple[float, float]:
            """Return (near_lesion_frac, near_radius_mm) for display-only sampling."""
            frac = 0.0
            radius = 12.0
            raw = req.get("nearLesionFrac", req.get("displayNearLesionFrac"))
            if raw is not None:
                try:
                    frac = float(raw)
                except (TypeError, ValueError):
                    frac = 0.0
            if req.get("displayBias") in ("near_lesion", "near-lesion", "peri"):
                frac = max(frac, 0.5)
            bias = str(req.get("displayBias") or "").lower()
            if bias in ("near_lesion", "near-lesion", "peri"):
                frac = max(frac, 0.5)
            if req.get("nearLesionDisplay") is True or req.get("nearLesionDisplay") == 1:
                frac = max(frac, 0.5)
            frac = min(0.9, max(0.0, frac))
            try:
                radius = float(req.get("nearRadiusMm", radius))
            except (TypeError, ValueError):
                radius = 12.0
            radius = min(30.0, max(3.0, radius))
            return frac, radius

        def _handle_profile(self):
            """Along-tract scalar profile for a named manifest bank."""
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "api" or parts[1] != "profile" or not parts[2]:
                return self._send(404, b"not found", "text/plain")
            bank_id = parts[2]
            qs = parse_qs(parsed.query, keep_blank_values=True)
            if "scalar" not in qs:
                scalar = "fa"
            else:
                values = qs.get("scalar") or []
                if len(values) != 1:
                    return self._json(409, {
                        "error": "duplicate scalar parameter",
                        "code": "scalar_unsupported",
                        "bankId": bank_id,
                        "scalar": values,
                    })
                scalar = values[0].strip().lower()
                if not scalar:
                    return self._json(409, {
                        "error": "blank scalar parameter",
                        "code": "scalar_unsupported",
                        "bankId": bank_id,
                        "scalar": "",
                    })
            # The single-flight lock guards the track job, and /api/bank/load
            # acquires it non-blocking: holding it across the profile compute
            # made a bank load clicked meanwhile answer 409 busy. The compute
            # below reads the .tck and the scalar map from disk on its own and
            # mutates no service state, so the lock is held only long enough to
            # read the immutable bank catalog and confirm the source is intact.
            lock = getattr(service, "_lock", None)
            banks_snapshot = getattr(service, "banks", {})
            if lock is not None:
                if not lock.acquire(blocking=True, timeout=25):
                    return self._json(409, {"error": "a track job is already running"})
                try:
                    known = bank_id in banks_snapshot
                    spec = banks_snapshot.get(bank_id) if known else None
                finally:
                    lock.release()
            else:
                known = bank_id in banks_snapshot
                spec = banks_snapshot.get(bank_id) if known else None
            if known and self._checked_bank_source_hash(bank_id) is None:
                return
            try:
                payload = compute_bank_profile(
                    case_root=service.case_root,
                    manifest=service.manifest,
                    banks={bank_id: spec} if known else banks_snapshot,
                    bank_id=bank_id,
                    scalar=scalar,
                    lesion_path=getattr(service, "_lesion_path", None),
                )
            except UnknownBank as exc:
                return self._json(404, {
                    "error": str(exc),
                    "code": exc.code,
                    "bankId": bank_id,
                })
            except ProfileError as exc:
                return self._json(exc.http_status, {
                    "error": str(exc),
                    "code": exc.code,
                    "bankId": bank_id,
                    "scalar": scalar,
                })
            # Re-check after the compute: a bank file replaced mid-read must
            # never be served as this tract's profile.
            if known and self._checked_bank_source_hash(bank_id) is None:
                return
            return self._json(200, payload)

        def _handle_connectotomy_cut(self):
            """Pack the cut-subset of one bank as bank-styled tubes (not a new identity)."""
            if service.ensure_connectotomy() is None:
                if service._connectotomy_error is not None:
                    return self._json(422, {
                        "error": "connectotomy-invalid",
                        "reason": service._connectotomy_error,
                    })
                return self._send(404, b"no lesion cavity", "text/plain")
            parts = self.path.strip("/").split("/")
            if len(parts) != 4 or parts[0] != "api" or parts[1] != "connectotomy":
                return self._send(404, b"not found", "text/plain")
            bid = parts[2]
            if bid not in service.banks:
                return self._json(404, {"error": f"unknown bank: {bid}"})
            spec = service.banks[bid]
            # A cut subset is still that bank's streamlines, so it carries the
            # same X-bankSourceHash contract as /api/bank/load. Without it a
            # saved review could not tell whether the bank changed underneath.
            source_hash = self._checked_bank_source_hash(bid)
            if source_hash is None:
                return
            lines, _meta = load_prebuilt_bundle(spec.path, max_keep=None)
            if self._checked_bank_source_hash(bid) is None:
                return
            expected_digest = getattr(service, "_connectotomy_bank_digest", {}).get(bid)
            actual_digest = digest_lines(lines)
            if expected_digest is None or actual_digest != expected_digest:
                return self._json(422, {
                    "error": "connectotomy bank population changed; reload connectotomy",
                    "code": "evidence_changed",
                    "bankId": bid,
                })
            idx = service._connectotomy_cut_idx.get(bid)
            if idx is None or len(idx) == 0:
                cut_lines = []
            else:
                try:
                    ordinals = []
                    for raw_idx in idx:
                        numeric_idx = float(raw_idx)
                        if not np.isfinite(numeric_idx) or numeric_idx != int(numeric_idx):
                            raise ValueError("nonintegral cut index")
                        ordinals.append(int(numeric_idx))
                except (TypeError, ValueError, OverflowError):
                    return self._json(422, {
                        "error": "connectotomy cut indices are invalid",
                        "code": "evidence_changed",
                        "bankId": bid,
                    })
                if any(i < 0 or i >= len(lines) for i in ordinals):
                    return self._json(422, {
                        "error": "connectotomy cut indices do not match bank population",
                        "code": "evidence_changed",
                        "bankId": bid,
                    })
                cut_lines = [lines[i] for i in ordinals]
            n_bank = len(lines)
            n_cut = len(cut_lines)
            buf, hdr, _ord = self._pack_lines(
                cut_lines,
                f"BANK | cut subset | {spec.engine}",
                {
                    "bankId": bid,
                    "label": spec.label,
                    "role": spec.role or "",
                    "cavity": "lesion",
                    "nCut": str(n_cut),
                    "nBank": str(n_bank),
                    "connectotomy": "cut-subset",
                    "bankSourceHash": source_hash,
                    "honesty": connectotomy_note(service.geom_floor_mm),
                    "bankDigest": expected_digest,
                },
                source_population=f"cut-subset:{bid}",
            )
            if digest_lines(load_prebuilt_bundle(spec.path, max_keep=None)[0]) != expected_digest:
                return self._json(422, {
                    "error": "connectotomy bank population changed; reload connectotomy",
                    "code": "evidence_changed",
                    "bankId": bid,
                })
            extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            return self._send(200, buf, extra=extra)

        def _handle_connectome(self):
            """C1b matrix — read-only. Building is script-owned, never a GET job."""
            out_dir = connectome_mod.connectome_out_dir(service.case_root)
            if not connectome_mod.is_built(out_dir):
                return self._send(404, b"connectome not built", "text/plain")
            try:
                # Pin `current`'s realpath once so load_provenance/check_fresh/
                # artifact_paths cannot land on three different generations if a
                # rebuild republishes `current` mid-request (same reasoning as
                # edge_streamlines' generation pinning).
                gen_dir = connectome_mod.pin_generation(out_dir)
                provenance = connectome_mod.load_provenance(out_dir, generation_dir=gen_dir)
                connectome_mod.check_fresh(
                    provenance, out_dir, service_manifest=service.manifest, generation_dir=gen_dir,
                )
                paths = connectome_mod.artifact_paths(out_dir, generation_dir=gen_dir)
                matrix = connectome_mod.read_matrix(paths["matrix"])
            except connectome_mod.ConnectomeInvalid as e:
                return self._json(409, {"error": "connectome-stale", "reason": str(e)})
            node_labels = [n["name"] for n in provenance.get("node_order", [])]
            # Strip absolute-path/argv internals — the client gets identity, not a filesystem map.
            public_prov = {k: v for k, v in provenance.items() if k not in ("case_root", "argv")}
            return self._json(200, {
                "matrix": matrix,
                "nodeLabels": node_labels,
                "provenance": public_prov,
            })

        def _edge_lesion_info(self, edge):
            """Cavity hits for one already-extracted edge, or an honest reason.

            Shared by the JSON route and the tubes route so lesion lookup is
            computed identically in both places and, in the tubes route, from
            the SAME extraction (edge["tck_path"]) that produced the tubes —
            never a second connectome2tck run.
            """
            lesion_meta = dv.active_inputs(service.manifest or {}).get("lesion")
            lesion = None
            lesion_reason = "manifest has no inputs.lesion"
            if isinstance(lesion_meta, dict) and lesion_meta.get("path"):
                lesion_path = None
                try:
                    lesion_path = _resolve_case_path(service.case_root, lesion_meta["path"], what="lesion")
                except ValueError as e:
                    lesion_reason = str(e)
                if lesion_path and os.path.isfile(lesion_path):
                    try:
                        hits = connectome_mod.edge_cavity_hits(
                            edge["tck_path"], lesion_path, service.case_root, manifest=service.manifest
                        )
                        lesion = {
                            "hits": hits["hits"],
                            "total": hits["total"],
                            "path": os.path.relpath(lesion_path, service.case_root),
                            "sha256": hits["lesion_sha256"],
                        }
                        lesion_reason = None
                    except CavityInvalid as e:
                        lesion_reason = str(e)
                elif lesion_path:
                    lesion_reason = f"manifest declares a lesion but the file is missing ({lesion_path})"
            return lesion, lesion_reason

        def _handle_connectome_edge(self):
            """GET /api/connectome/edge/<a>/<b> — traceable streamlines for one edge.

            Two independently-computed counts (matrix cell, assignment-row
            count) are cross-checked inside connectome.edge_streamlines; a
            disagreement is a server bug and surfaces as 500, never silently
            reconciled — same for an unavailable independent tckinfo count.
            Cavity hits reuse the C1 predicate on this edge's extracted
            streamlines only when manifest.inputs.lesion exists and its grid
            matches the corpus reference grid; otherwise lesion is null with
            an honest reason, never a silent zero.
            """
            parts = self.path.strip("/").split("/")
            if len(parts) != 5 or parts[:3] != ["api", "connectome", "edge"]:
                return self._send(404, b"not found", "text/plain")
            try:
                a = int(parts[3])
                b = int(parts[4])
            except ValueError:
                return self._json(400, {"error": f"node ids must be integers: {parts[3]!r},{parts[4]!r}"})
            out_dir = connectome_mod.connectome_out_dir(service.case_root)
            if not connectome_mod.is_built(out_dir):
                return self._send(404, b"connectome not built", "text/plain")
            try:
                edge = connectome_mod.edge_streamlines(out_dir, a, b, service_manifest=service.manifest)
            except connectome_mod.ConnectomeInvalid as e:
                return self._json(409, {"error": "connectome-stale", "reason": str(e)})
            except connectome_mod.ConnectomeEngineError as e:
                return self._json(500, {"error": "connectome-engine-error", "reason": str(e)})
            except connectome_mod.ConnectomeCountMismatch as e:
                return self._json(500, {"error": "connectome-count-mismatch", "reason": str(e)})

            lesion, lesion_reason = self._edge_lesion_info(edge)
            try:
                # Last step before responding: `current` must still be the
                # generation edge_streamlines pinned (a rebuild may have
                # published while lesion hits were computed).
                connectome_mod.ensure_still_pinned(
                    out_dir, edge["generation_dir"], what=f"edge {a}-{b}",
                )
            except connectome_mod.ConnectomeInvalid as e:
                return self._json(409, {"error": "connectome-stale", "reason": str(e)})
            return self._json(200, {
                "a": a, "b": b,
                "matrixCount": edge["matrix_count"],
                "assignmentRowCount": edge["assignment_row_count"],
                "extractedCount": edge["extracted_count"],
                "tckPath": edge["tck_path_rel"],
                "tckSha256": edge["tck_sha256"],
                "lesion": lesion,
                "lesionReason": lesion_reason,
                "labelNote": "ASSIGNED — parcel tag on a patient streamline, not cortex identity (ADR-0003)",
            })

        def _handle_connectome_edge_tubes(self):
            """GET /api/connectome/edge/<a>/<b>/tubes — this edge's exact
            extracted streamlines as the SAME binary tube response
            /api/connectotomy/<id>/cut uses (self._pack_lines) — no second
            response builder. edge_streamlines() re-runs the identical
            freshness check /api/connectome and /api/connectome/edge/<a>/<b>
            use (check_fresh, called before anything is extracted), so this
            route refuses 409 on the same staleness, never a default 200.

            One extraction serves everything: edge_streamlines() is called
            exactly once, and its own single internal provenance read is
            reused for every identity/path header below (never a second
            connectome_mod.load_provenance() call here — that would race a
            concurrent rebuild and could report a generation id from AFTER
            the extraction alongside bytes from before it). Lesion hits reuse
            that same extraction via _edge_lesion_info, so a single GET to
            this route is enough for the Show action: it never needs a
            second request to /api/connectome/edge/<a>/<b>.

            An edge is not bank-backed, so there is no X-bankSourceHash to
            report. Instead, ALL of the following are mandatory (a missing
            one refuses the response rather than shipping an empty header):
            X-edgeSourceHash (the extracted edge tck's DATA-segment sha256 —
            see connectome._tck_content_sha256; stable across repeated
            extractions, unlike a whole-file hash), X-corpusSourceHash,
            X-parcellationSourceHash, X-generationId, X-corpusPath,
            X-parcellationPath, X-edgeTckPath, X-matrixPath, X-matrixSha256,
            X-assignmentsPath, X-assignmentsSha256, X-assignmentRadiusMm (the
            radial-search radius of THIS extraction's own generation — the
            viewer's ASSIGNED label must read this header, never the
            panel's earlier /api/connectome fetch, so the label can never
            show one generation's radius next to another's tubes). Lesion
            path/sha ride along too (X-lesionPath/X-lesionSha256) when a
            lesion was used.

            edge_streamlines() is handed service.manifest so its freshness
            check also refuses a stale in-memory manifest (see
            connectome.check_fresh) — a derivation rebuild must never let
            this route pair fresh tubes with a lesion/grid resolved from a
            manifest copy the service loaded before the rebuild.
            """
            parts = self.path.strip("/").split("/")
            if len(parts) != 6 or parts[:3] != ["api", "connectome", "edge"] or parts[5] != "tubes":
                return self._send(404, b"not found", "text/plain")
            try:
                a = int(parts[3])
                b = int(parts[4])
            except ValueError:
                return self._json(400, {"error": f"node ids must be integers: {parts[3]!r},{parts[4]!r}"})
            out_dir = connectome_mod.connectome_out_dir(service.case_root)
            if not connectome_mod.is_built(out_dir):
                return self._send(404, b"connectome not built", "text/plain")
            try:
                edge = connectome_mod.edge_streamlines(out_dir, a, b, service_manifest=service.manifest)
            except connectome_mod.ConnectomeInvalid as e:
                return self._json(409, {"error": "connectome-stale", "reason": str(e)})
            except connectome_mod.ConnectomeEngineError as e:
                return self._json(500, {"error": "connectome-engine-error", "reason": str(e)})
            except connectome_mod.ConnectomeCountMismatch as e:
                return self._json(500, {"error": "connectome-count-mismatch", "reason": str(e)})

            required = (
                "content_sha256", "corpus_sha256", "parcellation_sha256", "gen_id",
                "corpus_path_rel", "parcellation_path_rel", "tck_path_rel",
                "matrix_path_rel", "matrix_sha256", "assignments_path_rel", "assignments_sha256",
            )
            missing = [k for k in required if not edge.get(k)]
            if missing or edge.get("radius_mm") is None:
                if edge.get("radius_mm") is None:
                    missing = [*missing, "radius_mm"]
                return self._json(500, {
                    "error": "connectome-engine-error",
                    "reason": f"edge {a}-{b}: extraction did not report {', '.join(missing)}",
                })

            lesion, lesion_reason = self._edge_lesion_info(edge)
            lines, _meta = load_prebuilt_bundle(edge["tck_path"], max_keep=None)
            hdr_extra = {
                "a": str(a),
                "b": str(b),
                "matrixCount": str(edge["matrix_count"]),
                "assignmentRowCount": str(edge["assignment_row_count"]),
                "extractedCount": str(edge["extracted_count"]),
                "edgeSourceHash": edge["content_sha256"],
                "corpusSourceHash": edge["corpus_sha256"],
                "parcellationSourceHash": edge["parcellation_sha256"],
                "generationId": edge["gen_id"],
                "assignmentRadiusMm": str(edge["radius_mm"]),
                "corpusPath": edge["corpus_path_rel"],
                "parcellationPath": edge["parcellation_path_rel"],
                "edgeTckPath": edge["tck_path_rel"],
                "matrixPath": edge["matrix_path_rel"],
                "matrixSha256": edge["matrix_sha256"],
                "assignmentsPath": edge["assignments_path_rel"],
                "assignmentsSha256": edge["assignments_sha256"],
                "lesionHits": "" if lesion is None else str(lesion["hits"]),
                "lesionTotal": "" if lesion is None else str(lesion["total"]),
                "lesionReason": lesion_reason or "",
                "lesionPath": "" if lesion is None else lesion["path"],
                "lesionSha256": "" if lesion is None else lesion["sha256"],
                "labelNote": "ASSIGNED — parcel tag on a patient streamline, not cortex identity (ADR-0003)",
            }
            buf, hdr, _ord = self._pack_lines(
                lines,
                f"CONNECTOME | edge {a}-{b}",
                hdr_extra,
                source_population=f"edge:{a}-{b}",
            )
            try:
                # The packed tubes come from the pinned generation's directory;
                # refuse if `current` moved between edge_streamlines' own final
                # pin check and the end of packing, never ship them as current.
                connectome_mod.ensure_still_pinned(
                    out_dir, edge["generation_dir"], what=f"edge {a}-{b} tubes",
                )
            except connectome_mod.ConnectomeInvalid as e:
                return self._json(409, {"error": "connectome-stale", "reason": str(e)})
            extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            return self._send(200, buf, extra=extra)

        def _handle_bank_load(self, req):
            bid = req.get("bankId")
            if not isinstance(bid, str) or bid not in service.banks:
                return self._json(400, {"error": f"unknown bankId: {bid!r}"})
            # Bank load is a prebuilt-file read. Multi-select fires overlapping
            # POSTs (CST-L while CST-R/boot is packing); 409-on-busy eats the
            # second chip. Wait for the live-track lock instead of failing.
            if not service._lock.acquire(blocking=True, timeout=25):
                return self._json(409, {"error": "a track job is already running"})
            try:
                spec = service.banks[bid]
                source_hash = self._checked_bank_source_hash(bid)
                if source_hash is None:
                    return
                before_bank_sha = sha256_file(spec.path)
                # Full bank file = analytic population; display cap applied in _pack_lines
                lines, meta = load_prebuilt_bundle(spec.path, max_keep=None)
                weights = None
                declared_bank_sha, declared_weight_sha = _sift2_manifest_receipt(
                    service, bid, spec,
                )
                sift2_binding = "absent"
                sift2_warning = ""
                if spec.sift2_path:
                    if not is_sha256(declared_bank_sha) or not is_sha256(declared_weight_sha):
                        sift2_binding = "unbound"
                    elif before_bank_sha.lower() != declared_bank_sha.lower():
                        sift2_binding = "bank-mismatch"
                    else:
                        weights = load_sift2_weights(
                            spec.sift2_path,
                            n_expected=len(lines),
                            require_binding=True,
                            expected_bank_sha256=declared_bank_sha,
                            expected_weight_sha256=declared_weight_sha,
                        )
                        sift2_binding = "bound" if weights is not None else "invalid"
                    if sift2_binding != "bound":
                        sift2_warning = (
                            "SIFT2 not applied: explicit bank and weight SHA256 "
                            f"binding is {sift2_binding}"
                        )
                honesty = (
                    "multi-ROI recipe match — may omit peri-lesional fibres "
                    "(edema/mass effect); not a completeness claim near lesion"
                )
                near_frac, near_r = self._parse_display_bias(req)
                buf, hdr, ordinals = self._pack_lines(
                    lines,
                    # ASCII-only separators (headers cannot carry unicode arrows/dots)
                    f"{meta['engine']} | {spec.engine}",
                    {
                        "bankId": bid,
                        "bankSourceHash": source_hash,
                        "nKept": meta["n_kept"],
                        "label": spec.label,
                        "role": spec.role or "",
                        "honesty": honesty,
                        "recipeNote": honesty,
                        "hasSift2File": "1" if spec.sift2_path else "0",
                        "sift2Applied": "1" if weights is not None else "0",
                        "sift2Binding": sift2_binding,
                        "nearLesionFrac": f"{near_frac:.2f}",
                        "nearRadiusMm": f"{near_r:.1f}",
                    },
                    weights=weights,
                    near_lesion_frac=near_frac,
                    near_radius_mm=near_r,
                    source_population=f"bank:{bid}",
                )
                try:
                    block, fid_hdr = _fidelity_payload(
                        service.manifest, bid, spec, ordinals,
                        service.operating_point, fod_path=service.fod,
                    )
                except FidelityRefusal as e:
                    return self._json(422, {
                        "error": "fidelity-refuse",
                        "reason": str(e),
                    })
                if block:
                    hdr["fidelityOffset"] = str(len(buf))
                    buf = buf + block
                hdr.update(fid_hdr)
                if self._checked_bank_source_hash(bid) is None:
                    return
                if sha256_file(spec.path) != before_bank_sha:
                    return self._json(422, {"error": "bank changed while loading", "code": "evidence_changed"})
                # Surface honesty in warning (HTTP-safe ASCII)
                prev = hdr.get("warning") or ""
                warnings = [prev, honesty, sift2_warning]
                hdr["warning"] = " | ".join(x for x in warnings if x)
                extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            finally:
                service._lock.release()
            # Send after release: a slow or vanished reader stalls only its own
            # connection, never the single-flight lock.
            return self._send(200, buf, extra=extra)

        def _handle_recovery_perilesional(self, req):
            """Fibres within radius of lesion — recovery, not named-bundle proof."""
            if service.lesion_shell is None or service.lesion_shell.shape[0] == 0:
                # also allow mask-only cases
                pass
            les_meta = service.inputs.get("lesion") or {}
            les_rel = les_meta.get("path") if isinstance(les_meta, dict) else None
            if not les_rel:
                return self._json(400, {"error": "no lesion in case — recovery needs a lesion mask"})
            try:
                radius = float(req.get("radiusMm", RECOVERY_RADIUS_DEFAULT))
            except (TypeError, ValueError):
                return self._json(400, {"error": "radiusMm must be a number"})
            if not (RECOVERY_RADIUS_MIN <= radius <= RECOVERY_RADIUS_MAX):
                return self._json(400, {
                    "error": f"radiusMm must be in [{RECOVERY_RADIUS_MIN}, {RECOVERY_RADIUS_MAX}]",
                })
            # Source: filter_bank (default) or a named multi-ROI bank file
            source = str(req.get("source") or "filter_bank").strip().lower()
            bank_path = None
            source_label = ""
            if source in ("filter_bank", "corpus", "wb"):
                if not service.filter_bank_path:
                    return self._json(404, {"error": "no filter_bank in manifest"})
                bank_path = service.filter_bank_path
                source_label = "filter_bank"
            elif source in ("bank", "bank_id"):
                bid = req.get("bankId")
                if not isinstance(bid, str) or bid not in service.banks:
                    return self._json(400, {"error": f"unknown bankId for recovery: {bid!r}"})
                bank_path = service.banks[bid].path
                source_label = bid
            else:
                return self._json(400, {
                    "error": "source must be filter_bank or bank (with bankId)",
                })

            if not service._lock.acquire(blocking=False):
                return self._json(409, {
                    "error": "a track job is already running",
                    "code": "busy",
                })
            try:
                les_path = _resolve_case_path(
                    service.case_root, les_rel, what="lesion",
                )
                les_bool = load_lesion_bool(les_path, service.grid)
                zone = lesion_proximity_mask(
                    les_bool, service.grid.affine, radius,
                )
                lines, meta = filter_near_lesion_memory(
                    bank_path=bank_path,
                    grid=service.grid,
                    lesion_zone=zone,
                    minlength=float(req.get("minlength", 10.0)),
                    maxlength=float(req.get("maxlength", 250.0)),
                )
                stamp = (
                    f"RECOVERY peri-lesional r={radius:.1f}mm from {source_label} "
                    f"— not multi-ROI named tract — research only — not navigation"
                )
                buf, hdr, _ord = self._pack_lines(
                    lines,
                    meta["engine"],
                    {
                        "role": "recovery_perilesional",
                        "label": meta["label"],
                        "radiusMm": f"{radius:.2f}",
                        "recoverySource": source_label,
                        "nKept": meta["n_kept"],
                        "nCorpus": meta["n_corpus"],
                        "honesty": stamp,
                        "provenance": "recovery-not-named-bundle",
                    },
                    source_population=f"recovery:{source_label}",
                )
                prev = hdr.get("warning") or ""
                hdr["warning"] = (prev + " | " + stamp).strip(" |")
                hdr.update(service.analytic_sources.commit_result(lines, kind="recovery").response_headers())
                extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            except ResultTooLarge:
                raise  # typed 422 at the do_POST boundary, not a generic 400
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            finally:
                service._lock.release()
            # Send after release: see _handle_bank_load.
            return self._send(200, buf, extra=extra)

        def _handle_export(self, req):
            """Export the explicit named bank or identified volatile result."""
            requested_bank = req.get("bankId")
            source_hash = None
            if isinstance(requested_bank, str) and requested_bank in service.banks:
                source_hash = self._checked_bank_source_hash(requested_bank)
                if source_hash is None:
                    return
            try:
                source = self._resolve_analytic_source(req)
            except AnalyticSourceError as exc:
                return self._json(409 if isinstance(exc, UnknownResultId) else 400, {"error": str(exc)})
            bid = source.bank_id
            if bid is not None:
                # Prefer original bank file on disk (full extract)
                source_hash = source_hash or self._checked_bank_source_hash(bid)
                if source_hash is None:
                    return
                path = service.banks[bid].path
                with open(path, "rb") as f:
                    data = f.read()
                if self._checked_bank_source_hash(bid) is None:
                    return
                name = Path(path).name
                return self._send(
                    200, data, "application/octet-stream",
                    extra={
                        "Content-Disposition": f'attachment; filename="{name}"',
                        "X-exportSource": "bank_file",
                        "X-bankId": bid,
                        "X-bankSourceHash": source_hash,
                        "X-nStreamlines": str(service.banks[bid].n_streamlines or ""),
                    },
                )
            if not source.lines:
                return self._json(400, {"error": "no tract loaded to export"})
            # A per-result file prevents concurrent exports overwriting one another.
            with tempfile.TemporaryDirectory(prefix="request-", dir=service._export_dir) as folder:
                out = os.path.join(folder, "tractlab_export.tck")
                n = write_tck(source.lines, out)
                with open(out, "rb") as f:
                    data = f.read()
            return self._send(
                200, data, "application/octet-stream",
                extra={
                    "Content-Disposition": 'attachment; filename="tractlab_export.tck"',
                    "X-exportSource": source.kind,
                    "X-resultId": source.result_id,
                    "X-nStreamlines": str(n),
                },
            )

        def _resolve_analytic_source(self, req):
            def load_bank(bid):
                spec = service.banks.get(bid)
                return None if spec is None else load_prebuilt_bundle(spec.path, max_keep=None)[0]
            return service.analytic_sources.resolve(req, load_bank=load_bank)

        def _handle_margin(self, req):
            """Build dilated streamline envelope mesh for the named source."""
            from .margin import MARGIN_MESH_ERROR_BOUND_MM
            try:
                source = self._resolve_analytic_source(req)
            except AnalyticSourceError as exc:
                return self._json(409 if isinstance(exc, UnknownResultId) else 400, {"error": str(exc)})
            if not source.lines:
                return self._json(400, {"error": "no tract loaded for margin"})
            try:
                mm = float(req.get("marginMm", 5.0))
            except (TypeError, ValueError):
                return self._json(400, {"error": "marginMm must be a number", "code": "bad_request"})
            try:
                verts, faces = build_margin_hull(
                    source.lines, service.grid, margin_mm=mm,
                )
            except ValueError as e:
                return self._json(400, {"error": str(e), "code": "bad_request"})
            if verts.shape[0] == 0:
                return self._json(400, {"error": "margin hull empty", "code": "bad_request"})
            body, mh = pack_mesh(verts, faces)
            extra = {
                "X-vertexCount": str(mh["vertexCount"]),
                "X-faceCount": str(mh["faceCount"]),
                "X-marginMm": f"{mm:.2f}",
                "X-marginNumericalBoundMm": f"{MARGIN_MESH_ERROR_BOUND_MM:.4f}",
                "X-marginSourceCount": str(min(len(source.lines), 1500)),
                "X-marginSampling": "longest 1500 paths at most; cosmetic subset",
                "X-bankId": source.bank_id or "",
                "X-resultId": source.result_id or "",
                "X-note": "display envelope only; not navigation / not resection margin",
            }
            return self._send(200, body, extra=extra)

        def _handle_filter(self, req):
            if not service.filter_bank_path:
                return self._json(404, {"error": "no filter_bank in manifest"})
            if not service._lock.acquire(blocking=False):
                return self._json(409, {"error": "a track job is already running"})
            try:
                before_bank_sha = sha256_file(service.filter_bank_path)
                try:
                    layers = layers_from_request(req, require_seed=True)
                    compiled = compile_roi_layers(service.grid, layers)
                except (RoiCompileError, CavityRoleError) as e:
                    return self._json(400, {"error": str(e)})
                p = req.get("params", {})
                if not isinstance(p, dict):
                    return self._json(400, {"error": "params must be an object", "code": "bad_request"})
                try:
                    minlength = _validate_filter_minlength(
                        p.get("minlength", MINLENGTH_DEFAULT)
                    )
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                lines, meta = filter_bank_memory(
                    bank_path=service.filter_bank_path,
                    grid=service.grid,
                    seed=compiled.seed_mask,
                    and_masks=list(compiled.and_masks),
                    or_mask=compiled.or_mask,
                    not_mask=compiled.not_mask,
                    minlength=minlength,
                    max_keep=None,  # full filter match; display cap in _pack_lines
                )
                after_bank_sha = sha256_file(service.filter_bank_path)
                if after_bank_sha != before_bank_sha:
                    return self._json(422, {
                        "error": "bank changed while filtering",
                        "code": "evidence_changed",
                    })
                buf, hdr, _ord = self._pack_lines(
                    lines,
                    meta["engine"],
                    {
                        "nKept": meta["n_kept"],
                        "nCorpus": meta["n_corpus"],
                        "filterMs": int(meta["elapsed_s"] * 1000),
                        "label": meta["label"],
                        "bankSha256": before_bank_sha,
                    },
                    source_population="filtered-corpus",
                )
                if sha256_file(service.filter_bank_path) != before_bank_sha:
                    return self._json(422, {
                        "error": "bank changed while filtering",
                        "code": "evidence_changed",
                    })
                floor = _derivation_uncertainty(service.manifest)
                hdr["warning"] = " | ".join(
                    part for part in (hdr.get("warning"), floor) if part
                )
                hdr.update(service.analytic_sources.commit_result(lines, kind="filter").response_headers())
                extra = {f"X-{k}": ("" if v is None else v) for k, v in hdr.items()}
            finally:
                service._lock.release()
            # Send after release: see _handle_bank_load.
            return self._send(200, buf, extra=extra)

    return Handler


class _TractlabHTTPServer(ThreadingHTTPServer):
    """``shutdown()`` also releases the service's owned temporary resources
    (S-08) — every existing caller already calls ``httpd.shutdown()``, so
    this needs no separate opt-in cleanup call."""

    _tractlab_service = None

    def shutdown(self):
        super().shutdown()
        if self._tractlab_service is not None:
            self._tractlab_service.close()


def serve(manifest_path: str, viewer_dir: str, host: str = "127.0.0.1", port: int = 8770):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(f"refusing non-loopback bind: {host!r}")
    file_hashes = FileHashCache()
    runtime_identity = RuntimeIdentity(
        viewer_dir,
        manifest_path=manifest_path,
        file_hashes=file_hashes,
    )
    service = TrackService(manifest_path, file_hashes=file_hashes)
    runtime_identity.verify_case_sources()
    service.runtime_identity = runtime_identity
    httpd = _TractlabHTTPServer((host, port), make_handler(service, viewer_dir))
    httpd._tractlab_service = service
    return httpd, service
