"""Post-hoc streamline filters for interactive tractography.

Hard ``-include`` on a tiny pons ROI yields ~0 streamlines on this FOD.
The proven alternative (hand-knob seed + anterior exclude) still returns mostly
short U-fibers. Ranking / filtering by **inferior reach** (min world-z) after a
dense seed run recovers the long, descending subset without a hard include.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InferiorReachFilter:
    """Keep streamlines whose lowest vertex is below ``z_max`` (world mm).

    Optional: keep only the top ``cap`` by descending z-span (max_z − min_z),
    so the display prefers long vertical fascicles over short deep stubs.
    """

    z_max: float = 10.0
    cap: int | None = 1500
    min_length_mm: float = 0.0


def streamline_length_mm(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    if len(s) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))


def filter_inferior_reach(
    streamlines: list[np.ndarray],
    filt: InferiorReachFilter,
) -> tuple[list[np.ndarray], dict]:
    """Return (kept, stats). Never fabricates tracks; empty is honest."""
    kept: list[tuple[float, np.ndarray]] = []  # (zspan, streamline)
    n_len = 0
    for s in streamlines:
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 3:
            continue
        if filt.min_length_mm > 0 and streamline_length_mm(arr) < filt.min_length_mm:
            n_len += 1
            continue
        zmin = float(arr[:, 2].min())
        zmax = float(arr[:, 2].max())
        if zmin >= filt.z_max:
            continue
        kept.append((zmax - zmin, arr))

    kept.sort(key=lambda t: t[0], reverse=True)
    n_pass = len(kept)
    if filt.cap is not None and len(kept) > filt.cap:
        kept = kept[: filt.cap]
    out = [s for _, s in kept]
    stats = {
        "n_in": len(streamlines),
        "n_pass_z": n_pass,
        "n_drop_length": n_len,
        "n_out": len(out),
        "z_max": filt.z_max,
        "cap": filt.cap,
    }
    return out, stats
