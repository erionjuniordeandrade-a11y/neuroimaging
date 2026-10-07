"""C1 connectotomy — lesion cavity ∩ named analytic banks.

A hit count, not a distance distribution (that is clearance.py). The case's
geometric floor is copy only: the cavity is never dilated. Out-of-bounds
vertices are discarded, never clamped (face-snap fabricates cuts).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .bank import BankSpec, load_tracks_cached
from .clearance import format_floor_mm  # noqa: F401 — re-exported; one floor formatter
from .grid import Grid
from .traversal import segment_voxel_hits

CAVITY_ROLE = "cavity"


def connectotomy_note(floor_mm: float) -> str:
    """Honesty copy with the CASE's recorded floor — never another case's."""
    return (
        f"Lesion cavity. Geometry uncertain to ~{format_floor_mm(floor_mm)} mm. "
        "Research/preview only — not a resection plan."
    )

# Payload keys that would smuggle C1b assignment / population identity into C1.
_FORBIDDEN_ROW_KEYS = frozenset({
    "assigned", "assignment", "network", "yeo", "schaefer", "parcel",
})


class CavityInvalid(ValueError):
    """The cavity itself is invalid evidence (grid mismatch, unreadable mask).

    ONLY this type may surface as the client-facing 422 connectotomy-invalid;
    any other ValueError is a server bug and must propagate loudly (Sol
    batch-3: a broad except mislabeled algorithmic errors as cavity faults).
    """


class CavityRoleError(ValueError):
    """Cavity-typed masks are never tracking seeds or include/exclude."""


def reject_cavity_role(*roles: str | None) -> None:
    for role in roles:
        if role is None:
            continue
        if str(role).strip().lower() == CAVITY_ROLE:
            raise CavityRoleError(
                "cavity-typed masks cannot be used as tracking seeds or include/exclude"
            )


def reject_cavity_in_request(req: Mapping[str, Any] | None) -> None:
    """Refuse a cavity key or any stroke tagged role=cavity."""
    if not isinstance(req, Mapping):
        return
    if "cavity" in req:
        raise CavityRoleError("cavity is not a tracking input")

    def _scan(obj: Any) -> None:
        if isinstance(obj, Mapping):
            reject_cavity_role(obj.get("role"))
            for v in obj.values():
                if isinstance(v, (Mapping, list, tuple)):
                    _scan(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _scan(item)

    for key in ("seed", "and", "or", "not", "include", "exclude"):
        if key in req:
            _scan(req[key])


def streamline_hits_cavity(
    line: np.ndarray,
    cavity: np.ndarray,
    grid: Grid,
) -> bool:
    """True iff the streamline PATH touches an in-bounds cavity voxel. No clamp.

    Delegates to traversal.segment_voxel_hits — the ONE predicate shared with
    fidelity labels (spec S6/F2), so C1 counts can never disagree with them.
    Segment-dense sampling: vertex rounding missed cavities thinner than the
    inter-vertex spacing.
    """
    return segment_voxel_hits(line, cavity, grid)


def count_bank_cuts(
    lines: Iterable[np.ndarray],
    cavity: np.ndarray,
    grid: Grid,
) -> tuple[int, int, np.ndarray]:
    """Return (n_cut, n_bank, cut_indices). Counts whatever list is passed."""
    mask = np.asarray(cavity, dtype=bool)
    if mask.shape != grid.shape:
        raise CavityInvalid("cavity shape != tracking grid")
    idx: list[int] = []
    n_bank = 0
    for i, line in enumerate(lines):
        n_bank += 1
        if streamline_hits_cavity(line, mask, grid):
            idx.append(i)
    return len(idx), n_bank, np.asarray(idx, dtype=np.int32)


def report_from_banks(
    banks: Mapping[str, Mapping[str, Any]],
    cavity: np.ndarray | None,
    grid: Grid,
    *,
    floor_mm: float,
) -> dict[str, Any] | None:
    """Build the C1 JSON payload, or None when the cavity is missing/empty."""
    if cavity is None:
        return None
    mask = np.asarray(cavity, dtype=bool)
    if mask.shape != grid.shape or not bool(mask.any()):
        return None
    rows: list[dict[str, Any]] = []
    cut_idx: dict[str, np.ndarray] = {}
    for bid, spec in banks.items():
        lines = spec.get("lines") or []
        n_cut, n_bank, idx = count_bank_cuts(lines, mask, grid)
        fraction = (float(n_cut) / float(n_bank)) if n_bank else 0.0
        row = {
            "id": bid,
            "n_cut": int(n_cut),
            "n_bank": int(n_bank),
            "fraction": fraction,
        }
        extra = set(row) - {"id", "n_cut", "n_bank", "fraction"}
        if extra & _FORBIDDEN_ROW_KEYS:
            raise RuntimeError(f"C1 row leaked assignment fields: {extra}")
        rows.append(row)
        cut_idx[bid] = idx
    return {
        "banks": rows,
        "cavity": "lesion",
        "floor_mm": float(floor_mm),
        "note": connectotomy_note(floor_mm),
        "_cut_idx": cut_idx,  # stripped before HTTP
    }


def compute_connectotomy(
    banks: Mapping[str, BankSpec],
    cavity: np.ndarray,
    grid: Grid,
    *,
    floor_mm: float,
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray]]:
    """Load each bank_* extract (never the 10M corpus) and count cuts."""
    mask = np.asarray(cavity, dtype=bool)
    if not bool(mask.any()):
        return None, {}
    loaded: dict[str, dict[str, Any]] = {}
    for bid, spec in banks.items():
        lines, _ = load_tracks_cached(spec.path)
        loaded[bid] = {"label": spec.label, "lines": lines}
    raw = report_from_banks(loaded, mask, grid, floor_mm=floor_mm)
    if raw is None:
        return None, {}
    cut_idx = raw.pop("_cut_idx")
    return raw, cut_idx
