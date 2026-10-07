"""Streamline endpoint accounting against a mask, and .tck completeness.

Endpoints, not passages. A waypoint test says where a streamline WENT; only an
endpoint test says where it STOPPED, and stopping inside a tumour is the failure
this module exists to count.

Two traps are guarded here, each in exactly one place.

⛔ Out-of-bounds endpoints are DISCARDED, never clamped. Clamping snaps an
exiting point onto the volume face and fabricates a hit. The rule lives in
``endpoints_in_bool_mask`` alone; ``endpoints_in_mask`` and
``tractlab.crossedfat`` both route through it, because two hand-kept copies of a
rule are two chances to get it wrong and no test that they still agree.

⛔ A ``.tck`` with a plausible SIZE is not a FINISHED ``.tck``. This module
previously had no completeness notion at all, and its callers gated on
``getsize(path) >= 50_000`` — which a tckgen killed after 50 kB passes, so a
partial tractogram was read as whole and produced quietly wrong counts.
``tck_completeness`` replaces the heuristic with three real signals: the file
must parse, the streamlines read must match the header's declared ``count``, and
that count must match the ``-select`` target recorded in ``command_history``
unless the seed budget was exhausted. What it cannot detect is stated on the
function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import nibabel as nib

from .grid import Grid, in_bounds, load_grid, world_to_voxel


def read_streamlines(tck_path: str) -> list[np.ndarray]:
    """Every vertex of every streamline, in world mm; one ``(M, 3)`` array each.

    The path gate in ``tractlab.crossedfat`` needs interior vertices — how many
    times a streamline crosses the midline, and where — which endpoints cannot
    express. Read once and derive the endpoints with
    ``endpoints_from_streamlines`` rather than parsing the file twice.
    """
    tractogram = nib.streamlines.load(tck_path).tractogram
    return [np.asarray(s, dtype=np.float64) for s in tractogram.streamlines]


def endpoints_from_streamlines(streamlines: list[np.ndarray]) -> np.ndarray:
    """``(N, 2, 3)`` first and last vertex of each streamline, in world mm."""
    if not streamlines:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.array([[s[0], s[-1]] for s in streamlines], dtype=np.float64)


def read_streamline_endpoints(tck_path: str) -> np.ndarray:
    """First and last vertex of every streamline, in world mm.

    Returns ``(N, 2, 3)``. Reads via nibabel so the .tck header's own
    affine_to_rasmm is honoured; the file SHA is deliberately not checked
    because a .tck header embeds a timestamp and is not stable.
    """
    return endpoints_from_streamlines(read_streamlines(tck_path))


def endpoints_in_bool_mask(
    endpoints: np.ndarray, grid: Grid, mask: np.ndarray
) -> np.ndarray:
    """Boolean ``(N, 2)``: is each endpoint inside this in-memory mask?

    ⛔ The single implementation of the never-clamp rule. Nearest-voxel lookup by
    rounding (which matches how a binary ROI is authored), then out-of-bounds
    indices are dropped — never clipped into range.
    """
    endpoints = np.asarray(endpoints, dtype=np.float64)
    if endpoints.size == 0:
        return np.empty((0, 2), dtype=bool)
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 3):
        raise ValueError(f"endpoints must be (N, 2, 3), got {endpoints.shape}")
    if tuple(mask.shape) != tuple(grid.shape):
        raise ValueError(f"mask {mask.shape} does not match grid {grid.shape}")

    flat = endpoints.reshape(-1, 3)
    ijk = np.round(world_to_voxel(grid, flat)).astype(np.int64)
    ok = in_bounds(grid, ijk)

    hit = np.zeros(flat.shape[0], dtype=bool)
    sel = ijk[ok]
    hit[ok] = np.asarray(mask, dtype=bool)[sel[:, 0], sel[:, 1], sel[:, 2]]
    return hit.reshape(endpoints.shape[0], 2)


def endpoints_in_mask(endpoints: np.ndarray, mask_path: str) -> np.ndarray:
    """Boolean ``(N, 2)``: is each endpoint inside the mask NIfTI at ``mask_path``?

    Thin wrapper: loads the grid and the mask, then defers to
    ``endpoints_in_bool_mask`` so the never-clamp rule has one owner.
    """
    if np.asarray(endpoints).size == 0:
        return np.empty((0, 2), dtype=bool)
    grid = load_grid(mask_path)
    mask = np.asarray(nib.load(mask_path).dataobj) > 0
    return endpoints_in_bool_mask(endpoints, grid, mask)


def count_endpoints_in_mask(tck_path: str, mask_path: str) -> int:
    """Total endpoints (out of 2N) that terminate inside the mask."""
    return int(endpoints_in_mask(read_streamline_endpoints(tck_path), mask_path).sum())


# --------------------------------------------------------------------------
# .tck completeness
# --------------------------------------------------------------------------

_SELECT_RE = re.compile(r"-select\s+(\d+)\s*([kKmM]?)")
_SUFFIX = {"": 1, "k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}


def parse_select_target(command_history: str | None) -> int | None:
    """The ``-select`` count tckgen was asked for, or None if not recoverable.

    None is an honest null: "this file does not record a target", which is
    different from "its target was zero". MRtrix accepts unit suffixes (``20k``),
    so a digits-only parse would silently read ``20k`` as 20.
    """
    if not command_history:
        return None
    m = _SELECT_RE.search(command_history)
    if m is None:
        return None
    return int(m.group(1)) * _SUFFIX[m.group(2)]


@dataclass(frozen=True)
class TckCompleteness:
    """Whether a .tck is a FINISHED tractogram, and how that was determined.

    Every count is ``None`` when it could not be read — never 0, which would be
    indistinguishable from "read, and genuinely empty".
    """

    is_complete: bool
    reason: str
    n_read: int | None = None
    declared_count: int | None = None
    select_target: int | None = None
    total_seeds_tried: int | None = None
    max_seeds_allowed: int | None = None


def _header_int(header: dict, key: str) -> int | None:
    try:
        return int(str(header[key]).strip())
    except (KeyError, TypeError, ValueError):
        return None


def tck_completeness(tck_path: str) -> TckCompleteness:
    """Decide whether ``tck_path`` holds a finished tractogram. Fails closed.

    Checks, in order:

    1. the file parses at all (a truncated .tck raises in nibabel);
    2. the streamlines actually read equal the header's declared ``count`` — a
       stale count means the writer never committed its last block;
    3. the declared count equals the ``-select`` target from ``command_history``,
       unless ``total_count`` shows the seed budget (``max_num_seeds``) was
       exhausted, which is a legitimate early stop rather than a kill.

    ⛔ What this CANNOT detect: a tckgen killed at exactly the moment its header
    commit landed AND whose declared count happens to equal ``-select``; a file
    hand-edited to be self-consistent; and, when ``command_history`` is absent
    (any .tck not produced by tckgen), whether the intended yield was reached at
    all — in that case only checks 1 and 2 apply and the reason says so.
    Anything unknown is reported incomplete, not complete.
    """
    try:
        tractogram_file = nib.streamlines.load(tck_path)
        n_read = len(list(tractogram_file.streamlines))
    except Exception as exc:  # truncated / absent / not a .tck
        return TckCompleteness(
            is_complete=False,
            reason=f"unreadable or truncated .tck ({type(exc).__name__}: {exc})",
        )

    header = dict(tractogram_file.header)
    declared = _header_int(header, "count")
    select_target = parse_select_target(header.get("command_history"))
    total_seeds = _header_int(header, "total_count")
    max_seeds = _header_int(header, "max_num_seeds")
    common = dict(
        n_read=n_read,
        declared_count=declared,
        select_target=select_target,
        total_seeds_tried=total_seeds,
        max_seeds_allowed=max_seeds,
    )

    if declared is None:
        return TckCompleteness(
            is_complete=False,
            reason="header declares no usable 'count' — completeness unverifiable",
            **common,
        )
    if declared != n_read:
        return TckCompleteness(
            is_complete=False,
            reason=(
                f"declared count {declared} != {n_read} streamlines read — the "
                "writer did not commit its last block (truncated or mid-write)"
            ),
            **common,
        )
    if select_target is None:
        return TckCompleteness(
            is_complete=True,
            reason=(
                f"count == {n_read} streamlines read; no -select target in "
                "command_history, so the intended yield was NOT checked"
            ),
            **common,
        )
    if declared == select_target:
        return TckCompleteness(
            is_complete=True,
            reason=f"reached the -select target of {select_target}",
            **common,
        )
    if max_seeds is not None and total_seeds is not None and total_seeds >= max_seeds:
        return TckCompleteness(
            is_complete=True,
            reason=(
                f"stopped at {declared} of -select {select_target} after "
                f"exhausting the seed budget ({total_seeds} >= {max_seeds}) — a "
                "normal tckgen termination, not a kill"
            ),
            **common,
        )
    return TckCompleteness(
        is_complete=False,
        reason=(
            f"only {declared} of -select {select_target} streamlines, with "
            f"{total_seeds} of {max_seeds} seeds tried — tckgen did not finish"
        ),
        **common,
    )
