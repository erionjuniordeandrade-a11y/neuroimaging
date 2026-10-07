"""Fixed-K binary tract payload for the tube renderer.

The tube geometry core needs a CONSTANT number of points K per streamline. The
interactive tracker returns variable-length streamlines, so we resample each
displayed track to exactly K points and emit a flat float32le buffer. Binary,
not nested JSON — Sol measured JSON packing at 0.1-0.4s and 5.8-6.6 MB, and the
browser parse is on top of that. See DESIGN v2 #9.

Layout (little-endian float32): K*L*3 values, streamline-major, then point, then
xyz. Header (lineCount L, pointsPerLine K, spaceId) travels as JSON alongside.
"""

from __future__ import annotations

import numpy as np


def resample_polyline(points: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline (n,3) to exactly k points by arc-length.

    Degenerate inputs (n<2 or zero length) are tiled to k points so K stays
    constant rather than raising and dropping the streamline silently.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (n,3)")
    n = points.shape[0]
    if n == 0:
        raise ValueError("empty streamline")
    if n == 1:
        return np.repeat(points, k, axis=0)

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return np.repeat(points[:1], k, axis=0)
    targets = np.linspace(0.0, total, k)
    out = np.empty((k, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cum, points[:, axis])
    return out


def polyline_length_mm(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def pack_streamlines_with_row_map(
    streamlines: list[np.ndarray],
    k: int,
    space_id: str,
    *,
    minlength_mm: float | None = None,
) -> tuple[bytes, dict, np.ndarray]:
    """Resample streamlines and return the retained input-row map.

    Rejects any streamline containing NaN/Inf (never let it reach Three.js).
    If ``minlength_mm`` is set, streamlines whose *source* arc length is below
    it are dropped (tckgen already applied this; this is a second guard). After
    resampling, if the chordal display length undershoots the source minlength
    by more than 2%, re-resample with denser K (up to 256) so the UI polyline
    does not appear to violate the user's length constraint.

    The third return value contains zero-based rows in ``streamlines`` for the
    geometry that was actually packed. Metadata producers must compose their
    own source ordinals through this map so a post-sample length drop cannot
    shift lineage or per-streamline evidence onto another displayed line.

    Returns (buffer_bytes, header_dict, retained_input_rows).
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    kept_src: list[np.ndarray] = []
    retained_input_rows: list[int] = []
    source_lengths: list[float] = []
    n_drop_short = 0
    for input_row, s in enumerate(streamlines):
        s = np.asarray(s, dtype=np.float64)
        if not np.isfinite(s).all():
            raise ValueError("non-finite coordinate in streamline (NaN/Inf)")
        src_L = polyline_length_mm(s)
        if minlength_mm is not None and src_L + 1e-6 < float(minlength_mm):
            n_drop_short += 1
            continue
        kept_src.append(s)
        retained_input_rows.append(input_row)
        source_lengths.append(src_L)

    # Choose uniform K: densify when chordal K would undershoot minlength
    k_final = k
    if minlength_mm is not None and kept_src:
        for s, src_L in zip(kept_src, source_lengths):
            kk = k
            disp_L = polyline_length_mm(resample_polyline(s, kk))
            target = max(float(minlength_mm), 0.98 * src_L)
            while disp_L + 1e-6 < target and kk < 256:
                kk = min(256, kk * 2)
                disp_L = polyline_length_mm(resample_polyline(s, kk))
            k_final = max(k_final, kk)

    rows = []
    display_lengths: list[float] = []
    for s in kept_src:
        rs = resample_polyline(s, k_final)
        rows.append(rs)
        display_lengths.append(polyline_length_mm(rs))

    if rows:
        arr = np.stack(rows, axis=0).astype("<f4")
    else:
        arr = np.empty((0, k_final, 3), dtype="<f4")

    n_len_under = 0
    if minlength_mm is not None and display_lengths:
        n_len_under = int(
            sum(1 for d in display_lengths if d + 1e-6 < float(minlength_mm))
        )

    header = {
        "encoding": "float32le",
        "lineCount": int(arr.shape[0]),
        "pointsPerLine": int(k_final),
        "layout": "line-major, then point, then xyz",
        "spaceId": space_id,
        "sourceMinLengthMm": (
            f"{min(source_lengths):.2f}" if source_lengths else ""
        ),
        "displayMinLengthMm": (
            f"{min(display_lengths):.2f}" if display_lengths else ""
        ),
        "nDropSourceShort": n_drop_short,
        "nDisplayLenUnderMin": n_len_under,
        "minlengthConstraint": (
            f"{float(minlength_mm):.2f}" if minlength_mm is not None else ""
        ),
    }
    return (
        arr.tobytes(order="C"),
        header,
        np.asarray(retained_input_rows, dtype=np.int64),
    )


def pack_streamlines(
    streamlines: list[np.ndarray],
    k: int,
    space_id: str,
    *,
    minlength_mm: float | None = None,
) -> tuple[bytes, dict]:
    """Backward-compatible two-value wrapper around the row-mapped packer."""
    buf, header, _retained_rows = pack_streamlines_with_row_map(
        streamlines,
        k,
        space_id,
        minlength_mm=minlength_mm,
    )
    return buf, header


def unpack_streamlines(buf: bytes, line_count: int, k: int) -> np.ndarray:
    """Inverse of pack (for tests): -> (line_count, k, 3) float array."""
    arr = np.frombuffer(buf, dtype="<f4")
    return arr.reshape(line_count, k, 3).astype(np.float64)
