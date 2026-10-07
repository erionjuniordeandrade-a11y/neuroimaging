"""Honesty pack for crossed-FAT — funnel, reject layers, distance distribution.

Uses the path-anchored gate in ``crossedfat.validate_crossed_fat`` (not the old
endpoint-only shortcut). Built from an already-tracked raw .tck — no new tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clearance import min_clearance_mm, per_vertex_distance_mm
from .crossedfat import validate_crossed_fat
from .pack import pack_streamlines, unpack_streamlines

DISPLAY_K = 64
DISPLAY_CAP = 1500
HIST_BINS = 20
HIST_MAX_MM = 25.0


@dataclass(frozen=True)
class HonestyPack:
    """Precomputed binary layers + funnel + per-validated min-distance hist."""

    funnel: dict
    hist: dict
    layers: dict[str, tuple[bytes, dict]]
    report: str

    def meta_json(self) -> dict:
        return {
            "outcome": "ok",
            "funnel": self.funnel,
            "hist": self.hist,
            "report": self.report,
            "layers": {
                name: {
                    "lineCount": hdr.get("lineCount", 0),
                    "nSource": hdr.get("nSource", 0),
                    "pointsPerLine": hdr.get("pointsPerLine", DISPLAY_K),
                }
                for name, (_b, hdr) in self.layers.items()
            },
            "caveat": (
                "anatomical aid only, not navigation; ~few-mm geom uncertainty; "
                "SMA = posterior 1/3 superiorfrontal (stated choice, not atlas label); "
                "gate is endpoint- AND path-anchored (midline standoff + single "
                "callosal crossing)"
            ),
        }


def _subsample(lines: list, cap: int) -> list:
    if len(lines) <= cap:
        return lines
    idx = np.linspace(0, len(lines) - 1, cap).astype(int)
    return [lines[i] for i in idx]


def _pack_layer(
    lines: list,
    space_id: str,
    lesion_shell: np.ndarray,
    *,
    n_source: int,
    label: str,
) -> tuple[bytes, dict]:
    shown = _subsample(lines, DISPLAY_CAP)
    buf, hdr = pack_streamlines(shown, DISPLAY_K, space_id)
    dist_off = ""
    if lesion_shell.shape[0] and hdr["lineCount"]:
        verts = unpack_streamlines(buf, hdr["lineCount"], DISPLAY_K)
        dist = per_vertex_distance_mm(verts, lesion_shell)
        if dist is not None:
            dist_off = str(len(buf))
            buf = buf + dist.tobytes(order="C")
    clr = min_clearance_mm(lines, lesion_shell) if lines else None
    hdr.update({
        "outcome": "ok",
        "nSource": n_source,
        "nDisplayed": len(shown),
        "layer": label,
        "clearanceMm": "" if clr is None else f"{clr:.1f}",
        "distanceOffset": dist_off,
        "engine": f"honesty-{label}",
    })
    return buf, hdr


def _hist_min_dist(lines: list, lesion_shell: np.ndarray) -> dict:
    empty = {
        "edges": [], "counts": [], "n": 0,
        "min": None, "p50": None, "p90": None, "max": None,
    }
    if not lines or lesion_shell.shape[0] == 0:
        return empty
    from scipy.spatial import cKDTree
    tree = cKDTree(lesion_shell)
    mins = []
    for s in lines:
        d, _ = tree.query(np.asarray(s), k=1)
        mins.append(float(np.min(d)))
    arr = np.asarray(mins, dtype=np.float64)
    edges = np.linspace(0.0, HIST_MAX_MM, HIST_BINS + 1)
    counts, _ = np.histogram(np.clip(arr, 0, HIST_MAX_MM), bins=edges)
    return {
        "edges": [round(float(e), 2) for e in edges],
        "counts": [int(c) for c in counts],
        "n": int(arr.size),
        "min": round(float(arr.min()), 2),
        "p50": round(float(np.median(arr)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "max": round(float(arr.max()), 2),
    }


def build_honesty_pack(
    raw_tck: str,
    aparc_path: str,
    lesion_path: str,
    lesion_shell: np.ndarray,
    space_id: str,
    *,
    n_attempts: int | None = None,
) -> HonestyPack:
    """Classify raw crossed-FAT tractogram into honesty layers via path-anchored gate."""
    import nibabel as nib

    qc = validate_crossed_fat(raw_tck, aparc_path, lesion_path)
    lines = list(nib.streamlines.load(raw_tck).streamlines)
    n_selected = len(lines)
    if n_selected != qc.n_total:
        raise ValueError(
            f"tck has {n_selected} streamlines but gate saw {qc.n_total}"
        )

    validated_set = set(qc.validated_indices)
    validated = [lines[i] for i in qc.validated_indices]
    # lesion-term: any streamline with lesion endpoint (from qc report)
    # Recompute quickly: validated are never lesion-term under the gate.
    # For lesion layer, sample those rejected with REASON_LESION if possible;
    # otherwise re-scan via endpoints_in_mask is heavy — use rejection reason.
    # CrossedFatQC doesn't expose per-streamline lesion mask, only counts.
    # Read streamlines that failed REASON_LESION: not in validated, and...
    # Simplest honest approach: pack "rejected" as all non-validated subsample,
    # and "lesion" as empty if we can't recover indices — but we need lesion indices.
    # validate_crossed_fat doesn't return them. Re-run endpoint check only for lesion layer.
    from .fivett import endpoints_in_mask, read_streamline_endpoints
    ep = read_streamline_endpoints(raw_tck)
    lesion_m = endpoints_in_mask(ep, lesion_path).any(axis=1)
    lesion_lines = [lines[i] for i in range(n_selected) if lesion_m[i]]
    rejected = [lines[i] for i in range(n_selected) if i not in validated_set]

    funnel = {
        "nAttempts": n_attempts,
        "nSelected": n_selected,
        "nBihemispheric": qc.n_bihemispheric,
        "nLesionTerminating": qc.n_lesion_terminating,
        "nImpossible": qc.n_impossible,
        "nComplete": qc.n_complete,
        "nValidated": qc.n_validated,
        "pctValidated": round(100.0 * qc.n_validated / n_selected, 2) if n_selected else 0.0,
        "rejections": dict(qc.rejections),
        "xMidMm": None if qc.x_mid_mm is None else round(qc.x_mid_mm, 2),
        "standoffMm": qc.standoff_mm,
        "ccTolMm": qc.cc_tol_mm,
    }

    layers = {
        "validated": _pack_layer(
            validated, space_id, lesion_shell,
            n_source=len(validated), label="validated",
        ),
        "lesion": _pack_layer(
            lesion_lines, space_id, lesion_shell,
            n_source=len(lesion_lines), label="lesion",
        ),
        "rejected": _pack_layer(
            rejected, space_id, lesion_shell,
            n_source=len(rejected), label="rejected",
        ),
    }

    return HonestyPack(
        funnel=funnel,
        hist=_hist_min_dist(validated, lesion_shell),
        layers=layers,
        report=qc.report,
    )
