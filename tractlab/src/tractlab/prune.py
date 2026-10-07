"""Outlier pruning for multi-ROI bank extracts (research, not navigation).

Three sequential gates (each optional):

1. **Length band** — keep streamlines whose arc length sits between
   ``length_lo_pct`` and ``length_hi_pct`` of the *input* distribution.
2. **Spatial core** — among survivors, keep midpoints within
   ``spatial_mad_k`` × MAD of the midpoint cloud (or a percentile cap).
3. **SIFT2 floor** — if weights are provided and align 1:1, drop weights
   below ``sift2_rel_floor × median`` (near-zero junk only, not a fixed %).

Clears visual false positives without re-tracking. Does not replace reverse-PE
or claim navigation safety. Analytic clearance should use the pruned file as
the new bank (honest n, not a hidden display filter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PruneParams:
    length_lo_pct: float = 5.0
    length_hi_pct: float = 95.0
    spatial_mad_k: float = 3.0
    """Keep midpoints with dist ≤ median + k · 1.4826 · MAD."""
    spatial_max_pct: float = 99.0
    """Hard cap on midpoint distance (only extreme tails)."""
    sift2_rel_floor: float = 0.05
    """Drop weight < this × median(weight) among length+spatial survivors (0 = off)."""
    sift2_drop_pct: float = 0.0
    """Legacy percentile floor; 0 disables. Prefer sift2_rel_floor."""
    min_keep: int = 32
    """Refuse to prune below this count (return original + flag)."""


@dataclass(frozen=True)
class PruneReport:
    n_in: int
    n_out: int
    n_length: int
    n_spatial: int
    n_sift2: int
    length_lo_mm: float | None
    length_hi_mm: float | None
    spatial_cut_mm: float | None
    sift2_floor: float | None
    kept_indices: np.ndarray  # int64 indices into input
    refused: bool = False
    note: str = ""


def polyline_length_mm(s: np.ndarray) -> float:
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def midpoint_mm(s: np.ndarray) -> np.ndarray:
    arr = np.asarray(s, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    return arr.mean(axis=0)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def prune_streamlines(
    streamlines: list[np.ndarray],
    weights: np.ndarray | None = None,
    params: PruneParams | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None, PruneReport]:
    """Return (kept_lines, kept_weights_or_None, report).

    Indices in ``report.kept_indices`` map into the input list.
    """
    p = params or PruneParams()
    n = len(streamlines)
    if n == 0:
        return [], None, PruneReport(
            n_in=0, n_out=0, n_length=0, n_spatial=0, n_sift2=0,
            length_lo_mm=None, length_hi_mm=None, spatial_cut_mm=None,
            sift2_floor=None, kept_indices=np.zeros(0, dtype=np.int64),
            note="empty input",
        )

    lengths = np.array([polyline_length_mm(s) for s in streamlines], dtype=np.float64)
    lo = float(np.percentile(lengths, p.length_lo_pct))
    hi = float(np.percentile(lengths, p.length_hi_pct))
    keep = (lengths >= lo) & (lengths <= hi) & (lengths > 0)
    n_length = int(keep.sum())

    # Spatial core on length survivors
    spatial_cut = None
    idx_len = np.flatnonzero(keep)
    if len(idx_len) > 0:
        mids = np.vstack([midpoint_mm(streamlines[int(i)]) for i in idx_len])
        center = np.median(mids, axis=0)
        dist = np.linalg.norm(mids - center, axis=1)
        mad = _mad(dist)
        # robust sigma ≈ 1.4826 * MAD; fall back to percentile if MAD ~ 0
        if mad > 1e-6:
            cut_mad = float(np.median(dist) + p.spatial_mad_k * 1.4826 * mad)
        else:
            cut_mad = float(np.percentile(dist, p.spatial_max_pct))
        cut_pct = float(np.percentile(dist, p.spatial_max_pct))
        spatial_cut = min(cut_mad, cut_pct)
        keep_sp = dist <= spatial_cut + 1e-9
        # map back
        keep2 = np.zeros(n, dtype=bool)
        keep2[idx_len[keep_sp]] = True
        keep = keep2
    n_spatial = int(keep.sum())

    sift_floor = None
    w_in = None
    if weights is not None:
        w_in = np.asarray(weights, dtype=np.float64).ravel()
        if len(w_in) != n:
            w_in = None

    if w_in is not None and n_spatial > 0:
        w_sub = w_in[keep]
        med_w = float(np.median(w_sub)) if len(w_sub) else 0.0
        if p.sift2_rel_floor > 0 and med_w > 0:
            sift_floor = p.sift2_rel_floor * med_w
            keep = keep & (w_in >= sift_floor)
        elif p.sift2_drop_pct > 0:
            sift_floor = float(np.percentile(w_sub, p.sift2_drop_pct))
            keep = keep & (w_in >= sift_floor)
    n_sift2 = int(keep.sum())

    kept_idx = np.flatnonzero(keep).astype(np.int64)
    if len(kept_idx) < p.min_keep and n >= p.min_keep:
        # refuse — too aggressive for this bundle
        all_idx = np.arange(n, dtype=np.int64)
        return (
            list(streamlines),
            w_in.copy() if w_in is not None else None,
            PruneReport(
                n_in=n, n_out=n, n_length=n_length, n_spatial=n_spatial,
                n_sift2=n_sift2, length_lo_mm=lo, length_hi_mm=hi,
                spatial_cut_mm=spatial_cut, sift2_floor=sift_floor,
                kept_indices=all_idx, refused=True,
                note=f"refused: would keep {len(kept_idx)} < min_keep {p.min_keep}",
            ),
        )

    kept_lines = [streamlines[int(i)] for i in kept_idx]
    kept_w = w_in[kept_idx] if w_in is not None else None
    return kept_lines, kept_w, PruneReport(
        n_in=n, n_out=len(kept_lines), n_length=n_length, n_spatial=n_spatial,
        n_sift2=n_sift2, length_lo_mm=lo, length_hi_mm=hi,
        spatial_cut_mm=spatial_cut, sift2_floor=sift_floor,
        kept_indices=kept_idx, refused=False,
        note=(
            f"length P{p.length_lo_pct:g}–P{p.length_hi_pct:g}; "
            f"spatial MAD×{p.spatial_mad_k:g}; "
            f"SIFT2 rel×{p.sift2_rel_floor:g}"
        ),
    )


def prune_tck_file(
    in_path: str,
    out_path: str,
    weights_path: str | None = None,
    out_weights_path: str | None = None,
    params: PruneParams | None = None,
) -> PruneReport:
    """Load .tck (+ optional .sift2.txt), prune, write outputs."""
    import nibabel as nib
    from .export_tck import write_tck
    from .sift2_util import load_sift2_weights

    tck = nib.streamlines.load(in_path)
    lines = [np.asarray(s, dtype=np.float32) for s in tck.streamlines]
    w = None
    if weights_path:
        w = load_sift2_weights(weights_path, n_expected=len(lines))
    kept, kept_w, rep = prune_streamlines(lines, weights=w, params=params)
    if rep.refused or rep.n_out == 0:
        # still write a copy of input for pipeline predictability? no — raise
        if rep.n_out == 0:
            raise ValueError(f"prune emptied {in_path}: {rep.note}")
        # refused: write original
        write_tck(lines, out_path)
        if out_weights_path and w is not None:
            np.savetxt(out_weights_path, w, fmt="%.8g")
        return rep
    write_tck(kept, out_path)
    if out_weights_path and kept_w is not None:
        np.savetxt(out_weights_path, kept_w, fmt="%.8g")
    return rep
