"""Continuous segment traversal through a voxel mask.

The mask predicates in the bank, recovery, and connectotomy paths all answer
the same geometric question: does the *continuous polyline* touch a true
voxel?  Treating only vertices as samples makes the answer depend on how many
collinear vertices happen to be stored in a ``.tck`` file.

The default path below is an exact supercover traversal in fractional voxel
coordinates.  Each voxel owns the closed cell centred at its integer index
(``i - .5 <= x <= i + .5``); a segment is transformed by the full inverse
affine, then Amanatides-Woo stepping visits every cell whose closed cell it
intersects.  Thus anisotropic, rotated, and sheared affines use the same
world-space segment without an orthogonal-grid approximation.  Boundary ties
visit all touching cells, which is conservative for a binary mask.

``step_mm`` is retained only as an explicitly requested diagnostic sampling
mode for compatibility with the old performance escape hatch.  Its samples
are at most ``step_mm`` apart, so the nearest unsampled point is at most
``step_mm / 2`` from a sample along the segment.  Production callers leave it
unset and receive the exact cell supercover.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from .grid import Grid, world_to_voxel


def _default_step_mm(grid: Grid) -> float:
    """Compatibility value for the optional finite sampling mode."""
    vox = np.linalg.norm(np.asarray(grid.affine, dtype=np.float64)[:3, :3], axis=0)
    return float(vox.min()) / 2.0


def _densify(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Return samples with adjacent points no farther apart than ``spacing``."""
    segs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    n = np.maximum(1, np.ceil(seg_len / spacing).astype(np.int64))
    total = int(n.sum())
    if total == 0:
        return pts.copy()
    seg_id = np.repeat(np.arange(len(n)), n)
    local = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    frac = local / n[seg_id]
    dense = pts[seg_id] + segs[seg_id] * frac[:, None]
    return np.vstack([dense, pts[-1:]])


def _validate_mask_and_inverse(mask: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    inv = np.asarray(inv_affine, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {m.shape}")
    if inv.shape != (4, 4) or not np.isfinite(inv).all():
        raise ValueError("inverse affine must be a finite 4x4 matrix")
    return m


def _validate_line(line: np.ndarray) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"streamline must have shape (N, 3), got {pts.shape}")
    if not np.isfinite(pts).all():
        # Dropping a bad vertex would silently bridge the two valid pieces.
        raise ValueError("non-finite vertex in streamline")
    return pts


def _clip_segment_to_box(
    start: np.ndarray,
    end: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a voxel-coordinate segment to the closed volume box."""
    direction = end - start
    lo = np.full(3, -0.5, dtype=np.float64)
    hi = np.asarray(shape, dtype=np.float64) - 0.5
    t_enter = 0.0
    t_exit = 1.0
    eps = 1e-12
    for axis in range(3):
        d = float(direction[axis])
        p = float(start[axis])
        if abs(d) <= eps:
            if p < lo[axis] - eps or p > hi[axis] + eps:
                return None
            continue
        a = (lo[axis] - p) / d
        b = (hi[axis] - p) / d
        if a > b:
            a, b = b, a
        t_enter = max(t_enter, a)
        t_exit = min(t_exit, b)
        if t_enter > t_exit + eps:
            return None
    clipped_start = start + direction * max(0.0, t_enter)
    clipped_end = start + direction * min(1.0, t_exit)
    # Round only numerical spill from clipping; never clamp an actually OOB
    # segment onto a face before the intersection test above.
    clipped_start = np.minimum(np.maximum(clipped_start, lo), hi)
    clipped_end = np.minimum(np.maximum(clipped_end, lo), hi)
    return clipped_start, clipped_end


def _segment_cells(
    start: np.ndarray,
    end: np.ndarray,
    shape: tuple[int, int, int],
) -> Iterator[tuple[int, int, int]]:
    """Yield the closed-cell supercover of one fractional-voxel segment."""
    clipped = _clip_segment_to_box(start, end, shape)
    if clipped is None:
        return
    a, b = clipped
    delta = b - a
    # A cell is selected by nearest-centre geometry, with half-way boundaries.
    cell = np.floor(a + 0.5).astype(np.int64)
    end_cell = np.floor(b + 0.5).astype(np.int64)
    step = np.sign(delta).astype(np.int64)
    inf = float("inf")
    t_max = np.full(3, inf, dtype=np.float64)
    t_delta = np.full(3, inf, dtype=np.float64)
    for axis in range(3):
        d = float(delta[axis])
        if d > 1e-14:
            t_max[axis] = (float(cell[axis]) + 0.5 - float(a[axis])) / d
            t_delta[axis] = 1.0 / d
        elif d < -1e-14:
            t_max[axis] = (float(cell[axis]) - 0.5 - float(a[axis])) / d
            t_delta[axis] = -1.0 / d

    seen: set[tuple[int, int, int]] = set()

    def emit(candidate: np.ndarray) -> Iterator[tuple[int, int, int]]:
        key = (int(candidate[0]), int(candidate[1]), int(candidate[2]))
        if key not in seen:
            seen.add(key)
            yield key

    # Include the starting cell before handling a boundary tie.  The loop is
    # bounded by the number of crossed cells; every iteration advances one or
    # more axes, including exact corner/edge ties.
    yield from emit(cell)
    # ``floor(x + .5)`` chooses the cell on the positive side of an exact
    # half-way boundary.  A segment moving positively starts by touching the
    # cell on the other side too, so include that closed-cell contact at t=0.
    start_tied = [
        axis
        for axis in range(3)
        if step[axis] > 0
        and abs(float(a[axis]) - (float(cell[axis]) - 0.5)) <= 1e-12
    ]
    for bits in range(1, 1 << len(start_tied)):
        candidate = cell.copy()
        for j, axis in enumerate(start_tied):
            if bits & (1 << j):
                candidate[axis] -= 1
        yield from emit(candidate)
    max_iter = int(np.prod(np.asarray(shape, dtype=np.int64)) + 4)
    for _ in range(max_iter):
        if np.array_equal(cell, end_cell):
            break
        min_t = float(np.min(t_max))
        if not np.isfinite(min_t):
            break
        tied = np.flatnonzero(t_max <= min_t + 1e-12)
        if tied.size == 0:
            break
        old = cell.copy()
        # At an edge/corner, the segment touches the cells reached by every
        # non-empty subset of the tied axes.  Emit those side cells as well as
        # the diagonal cell so a boundary contact cannot be missed.
        for bits in range(1, 1 << int(tied.size)):
            candidate = old.copy()
            for j, axis in enumerate(tied):
                if bits & (1 << j):
                    candidate[axis] += step[axis]
            yield from emit(candidate)
        cell[tied] += step[tied]
        t_max[tied] += t_delta[tied]
    # Symmetric endpoint contact for a segment moving negatively into an exact
    # half-way boundary: floor chooses the lower cell, while the upper cell is
    # also touched at t=1.
    end_tied = [
        axis
        for axis in range(3)
        if step[axis] < 0
        and abs(float(b[axis]) - (float(end_cell[axis]) + 0.5)) <= 1e-12
    ]
    for bits in range(1, 1 << len(end_tied)):
        candidate = end_cell.copy()
        for j, axis in enumerate(end_tied):
            if bits & (1 << j):
                candidate[axis] += 1
        yield from emit(candidate)


def _exact_voxel_hits(pts_voxel: np.ndarray, mask: np.ndarray) -> bool:
    """Check all polyline segments against ``mask`` in voxel coordinates."""
    shape = tuple(int(s) for s in mask.shape)
    if pts_voxel.shape[0] == 1:
        ijk = np.floor(pts_voxel[0] + 0.5).astype(np.int64)
        good = np.all((ijk >= 0) & (ijk < np.asarray(shape)))
        return bool(good and mask[tuple(ijk)])
    for start, end in zip(pts_voxel[:-1], pts_voxel[1:]):
        for i, j, k in _segment_cells(start, end, shape):
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                if bool(mask[i, j, k]):
                    return True
    return False


def segment_voxel_hits_affine(
    line: np.ndarray,
    mask: np.ndarray,
    inv_affine: np.ndarray,
    *,
    step_mm: float | None = None,
) -> bool:
    """Shared mask predicate when the caller already has ``inv_affine``."""
    m = _validate_mask_and_inverse(mask, inv_affine)
    pts = _validate_line(line)
    if pts.shape[0] == 0:
        return False
    inv = np.asarray(inv_affine, dtype=np.float64)
    vox = pts @ inv[:3, :3].T + inv[:3, 3]
    if step_mm is None:
        return _exact_voxel_hits(vox, m)
    spacing = float(step_mm)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"sampling step must be finite and > 0, got {spacing!r}")
    sampled_world = _densify(pts, spacing)
    sampled_vox = sampled_world @ inv[:3, :3].T + inv[:3, 3]
    ijk = np.rint(sampled_vox).astype(np.int64)
    shape = np.asarray(m.shape, dtype=np.int64)
    good = np.all((ijk >= 0) & (ijk < shape), axis=1)
    if not bool(good.any()):
        return False
    ijk = ijk[good]
    return bool(m[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any())


def segment_voxel_hits(
    line: np.ndarray,
    mask: np.ndarray,
    grid: Grid,
    step_mm: float | None = None,
) -> bool:
    """True iff the continuous polyline intersects any true mask voxel.

    With the default ``step_mm=None`` this is an exact voxel-cell supercover.
    Supplying ``step_mm`` opts into the bounded legacy sampling diagnostic;
    that mode may undercount a path by at most the requested half-spacing in
    arclength and is never used by production filters.
    """
    m = np.asarray(mask)
    if m.shape != tuple(grid.shape):
        raise ValueError(f"mask shape {m.shape} != grid shape {tuple(grid.shape)}")
    pts = _validate_line(line)
    if pts.shape[0] == 0:
        return False
    try:
        inv = np.linalg.inv(np.asarray(grid.affine, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("grid affine must be invertible") from exc
    # Keep the authoritative grid transform in this call path.  It also makes
    # malformed coordinate shape failures explicit before mask indexing.
    _ = world_to_voxel(grid, pts[:1])
    return segment_voxel_hits_affine(pts, m, inv, step_mm=step_mm)


def hit_count(
    lines: Iterable[np.ndarray],
    mask: np.ndarray,
    grid: Grid,
    step_mm: float | None = None,
) -> int:
    """Number of streamlines whose continuous path touches the mask."""
    return sum(
        1 for line in lines if segment_voxel_hits(line, mask, grid, step_mm=step_mm)
    )
