"""Tract-to-lesion proximity — distribution, not an extreme order statistic.

⛔ The old metric was min(distance) over every vertex of every streamline.
On a probabilistic sample of N×K points that quantity collapses toward zero
as N grows and cannot distinguish ipsilateral from contralateral seeds. It also
measured seed-abutting ROI geometry, not tract–lesion relationship.

This module reports the **5th percentile of per-streamline minimum distances**
after excluding path portions near the seed mask, with an explicit geometric floor
for uncorrected EPI distortion. Honest null when there is no lesion or no
eligible streamlines. Never a resection margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nibabel as nib
from scipy import ndimage

from .grid import load_grid, voxel_to_world
from . import derivation as dv


# A nearest-distance function is 1-Lipschitz along arclength.  When the exact
# segment-to-lesion calculation is too large for a bounded vectorized query,
# this spacing therefore gives a reported minimum no more than half a spacing
# above the continuous-path minimum.  The bound is recorded in the report.
CLEARANCE_SAMPLE_SPACING_MM = 0.5
CLEARANCE_EXACT_PAIR_CAP = 2_000_000
# Seed exclusion is exact segment clipping.  Refuse an unbounded quadratic
# setup instead of falling back to samples that could miss a short eligible
# arc between two seed-boundary crossings.
CLEARANCE_SEED_CLIP_PAIR_CAP = 2_000_000

def format_floor_mm(floor_mm: float) -> str:
    """3.0 -> '3', 0.6033 -> '0.6', 1.5 -> '1.5' — one decimal at most, so a
    signed delta-QC median is never rounded UP to a whole millimetre (0.6 -> "1")
    nor printed with false sub-0.1 mm precision. Shared by the HUD clearance
    strings and the connectotomy honesty note."""
    return f"{float(floor_mm):.1f}".rstrip("0").rstrip(".")


def geom_floor_mm(manifest: dict) -> float:
    """Per-case geometric floor (Task 16, S-14).

    Fail-closed: a case that never recorded its geometric uncertainty gets a
    ValueError, never a silent 3 mm default — printing another case's floor
    is a false uncertainty statement.

    Derivation-aware (ADR-0004, audit S-14): ``acquisition.geom_floor_mm`` is
    an UNCORRECTED-lineage figure and can survive a migration to a reverse-PE
    corrected derivation. On a ``rpe_pair`` derivation the only admissible
    millimetre figure is the signed delta-QC median — the same number
    ``derivation.floor_label`` prints — so an inherited 3.0 can never be
    served on a corrected case. Unsigned delta QC refuses.
    """
    if dv.active_kind(manifest) == dv.KIND_RPE_PAIR:
        if not dv.delta_qc_signed(manifest):
            raise ValueError(
                "active derivation is reverse-PE corrected but delta QC is "
                "unsigned — refusing acquisition.geom_floor_mm (it may predate "
                "the correction, ADR-0004); sign the delta QC sheet"
            )
        v = manifest["delta_qc"]["auto"]["median_mm"]
        fv = float(v)
        if not (fv > 0.0 and np.isfinite(fv)):
            raise ValueError(
                f"signed delta QC median_mm must be a finite positive number "
                f"to serve as geom_floor_mm, got {v!r}"
            )
        return fv
    acq = manifest.get("acquisition")
    v = acq.get("geom_floor_mm") if isinstance(acq, dict) else None
    if v is None:
        raise ValueError(
            "case manifest has no acquisition.geom_floor_mm — record this "
            "case's geometric uncertainty; refusing a silent default"
        )
    try:
        fv = float(v) if not isinstance(v, bool) and isinstance(v, (int, float)) else None
    except OverflowError:
        fv = None
    if fv is None or not (fv > 0.0 and np.isfinite(fv)):
        raise ValueError(f"acquisition.geom_floor_mm must be a finite positive "
                         f"number, got {v!r}")
    return fv


# Path portions closer than this to the seed mask are excluded from clearance
# (otherwise seed-abutting ROIs force clearance ≈ 0 before tracking).
SEED_EXCLUDE_MM = 3.0
# Cap for clearance *analytic* population. Display may length-rank a smaller set;
# p5 must not be computed on that cosmetic subsample alone.
ANALYTIC_CAP = 20_000
ANALYTIC_RNG_SEED = 42


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite coordinate in {name}")
    return arr


def _simplify_collinear(line: np.ndarray) -> np.ndarray:
    """Remove duplicate/same-direction collinear insertions, preserving path."""
    arr = _validate_points(line, "streamline")
    if arr.shape[0] <= 2:
        return arr.copy()
    keep = [arr[0]]
    for point in arr[1:]:
        if np.array_equal(point, keep[-1]):
            continue
        keep.append(point)
        while len(keep) >= 3:
            a, b, c = keep[-3:]
            first = b - a
            second = c - b
            n_first = float(np.linalg.norm(first))
            n_second = float(np.linalg.norm(second))
            if n_first == 0.0 or n_second == 0.0:
                keep.pop(-2)
                continue
            # A reversal is a real cusp and must remain two segments.  Same
            # direction collinear points are the redundant representation that
            # made sparse and dense .tck paths produce different clearances.
            cross_norm = float(np.linalg.norm(np.cross(first, second)))
            if cross_norm <= 1e-10 * max(n_first * n_second, 1.0):
                if float(np.dot(first, second)) >= 0.0:
                    keep.pop(-2)
                    continue
            break
    return np.asarray(keep, dtype=np.float64)


def _sample_polyline(line: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Sample a polyline by arclength at no more than ``spacing_mm`` spacing."""
    if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
        raise ValueError("clearance sampling spacing must be finite and > 0")
    arr = _simplify_collinear(line)
    if arr.shape[0] <= 1:
        return arr
    seg = np.diff(arr, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total == 0.0:
        return arr[:1].copy()
    n = max(1, int(np.ceil(total / float(spacing_mm))))
    distances = np.linspace(0.0, total, n + 1)
    ids = np.searchsorted(cumulative, distances, side="right") - 1
    ids = np.clip(ids, 0, len(seg) - 1)
    denom = lengths[ids]
    frac = np.divide(
        distances - cumulative[ids],
        denom,
        out=np.zeros_like(distances),
        where=denom > 0.0,
    )
    return arr[ids] + frac[:, None] * seg[ids]


def _exact_line_to_points_distance(line: np.ndarray, targets: np.ndarray) -> float:
    """Exact distance from a continuous polyline to a finite point cloud."""
    arr = _simplify_collinear(line)
    pts = _validate_points(targets, "lesion points")
    if arr.shape[0] == 0 or pts.shape[0] == 0:
        return float("inf")
    best_sq = np.full(pts.shape[0], np.inf, dtype=np.float64)
    if arr.shape[0] == 1:
        delta = pts - arr[0]
        return float(np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta))))
    for start, end in zip(arr[:-1], arr[1:]):
        direction = end - start
        denom = float(np.dot(direction, direction))
        if denom == 0.0:
            delta = pts - start
            dist_sq = np.einsum("ij,ij->i", delta, delta)
        else:
            projection = np.einsum("ij,j->i", pts - start, direction) / denom
            projection = np.clip(projection, 0.0, 1.0)
            closest = start + projection[:, None] * direction
            delta = pts - closest
            dist_sq = np.einsum("ij,ij->i", delta, delta)
        best_sq = np.minimum(best_sq, dist_sq)
    return float(np.sqrt(np.min(best_sq)))


def _segment_seed_eligible_intervals(
    start: np.ndarray,
    end: np.ndarray,
    seed_pts: np.ndarray,
    seed_tree,
    radius_mm: float,
) -> list[tuple[float, float]]:
    """Return parameter intervals outside the union of seed exclusion balls.

    For ``p(t) = start + t * (end - start)``, each seed ball cuts one open
    quadratic interval from ``[0, 1]``.  Sorting and merging those intervals
    gives the exact complement, independent of how many collinear vertices
    were inserted into the original streamline.  Tangent and shared boundary
    points are retained when the KD-tree confirms ``distance >= radius``.
    """
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    denom = float(np.dot(direction, direction))
    threshold_sq = float(radius_mm) * float(radius_mm)
    boundary_candidates: list[float] = []

    if denom == 0.0:
        distance, _ = seed_tree.query(np.asarray(start, dtype=np.float64), k=1)
        return [(0.0, 0.0)] if float(distance) >= float(radius_mm) else []
    if not np.isfinite(denom) or not np.isfinite(threshold_sq):
        raise ValueError("seed exclusion clipping overflowed finite geometry")

    inside: list[tuple[float, float]] = []
    for seed in seed_pts:
        relative = np.asarray(start, dtype=np.float64) - seed
        b = float(np.dot(relative, direction))
        c = float(np.dot(relative, relative) - threshold_sq)
        discriminant = b * b - denom * c
        if not np.isfinite(b) or not np.isfinite(c) or not np.isfinite(discriminant):
            raise ValueError("seed exclusion clipping overflowed finite geometry")
        discriminant_tolerance = 1e-12 * max(abs(b * b), abs(denom * c), 1.0)
        if discriminant < -discriminant_tolerance:
            continue
        root = float(np.sqrt(max(discriminant, 0.0)))
        lo = (-b - root) / denom
        hi = (-b + root) / denom
        if lo > hi:
            lo, hi = hi, lo
        clipped_lo = max(0.0, lo)
        clipped_hi = min(1.0, hi)
        if clipped_hi < 0.0 or clipped_lo > 1.0:
            continue
        # The ball interior is open because points exactly on the exclusion
        # radius satisfy the public >= predicate.  Keep roots separately so a
        # tangent or touching boundary can be tested below.
        boundary_candidates.extend((clipped_lo, clipped_hi))
        if clipped_hi > clipped_lo:
            inside.append((clipped_lo, clipped_hi))

    eligible: list[tuple[float, float]] = []
    if inside:
        inside.sort(key=lambda interval: interval[0])
        merged: list[list[float]] = []
        for lo, hi in inside:
            if not merged or lo > merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
        cursor = 0.0
        for lo, hi in merged:
            if lo > cursor:
                eligible.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < 1.0:
            eligible.append((cursor, 1.0))
    else:
        eligible.append((0.0, 1.0))

    # Positive-length complements above already include ordinary roots as
    # endpoints.  Add isolated/touching roots only when they really satisfy
    # the >= rule against every seed ball; this also avoids treating a root of
    # one ball as eligible while it remains inside another overlapping ball.
    if boundary_candidates:
        candidates = np.unique(np.clip(np.asarray(boundary_candidates), 0.0, 1.0))
        points = np.asarray(start, dtype=np.float64)[None, :] + candidates[:, None] * direction[None, :]
        distances, _ = seed_tree.query(points, k=1)
        tolerance = 1e-12 * max(1.0, float(radius_mm))
        for parameter, distance in zip(candidates, np.asarray(distances).reshape(-1)):
            if float(distance) + tolerance >= float(radius_mm):
                eligible.append((float(parameter), float(parameter)))
    return eligible


def _line_clearance_distance(
    line: np.ndarray,
    lesion_pts: np.ndarray,
    lesion_tree,
    seed_tree,
    seed_pts: np.ndarray | None,
    seed_exclude_mm: float,
) -> tuple[float | None, bool, float]:
    """Return (distance, eligible, numerical error bound) for one path."""
    arr = _simplify_collinear(line)
    if arr.shape[0] == 0:
        return None, False, 0.0

    # Clip each physical segment against the union of seed exclusion balls.
    # Sampling cannot guarantee that a short eligible arc is observed, so a
    # large exact setup fails closed rather than silently undercounting it.
    if (
        seed_tree is not None
        and seed_pts is not None
        and seed_pts.shape[0]
        and seed_exclude_mm > 0.0
    ):
        segment_count = max(arr.shape[0] - 1, 1)
        clip_pairs = int(segment_count) * int(seed_pts.shape[0])
        if clip_pairs > CLEARANCE_SEED_CLIP_PAIR_CAP:
            raise ValueError(
                "exact seed exclusion clipping exceeds bounded pair budget "
                f"({clip_pairs} > {CLEARANCE_SEED_CLIP_PAIR_CAP}); refusing "
                "a sampled eligible-path approximation"
            )

        pieces: list[np.ndarray] = []
        for segment_index in range(segment_count):
            if arr.shape[0] == 1:
                start = end = arr[0]
            else:
                start, end = arr[segment_index], arr[segment_index + 1]
            intervals = _segment_seed_eligible_intervals(
                start,
                end,
                seed_pts,
                seed_tree,
                float(seed_exclude_mm),
            )
            for lo, hi in intervals:
                p0 = start + float(lo) * (end - start)
                p1 = start + float(hi) * (end - start)
                pieces.append(np.asarray((p0, p1), dtype=np.float64))
        if not pieces:
            return None, False, 0.0
        exact_pairs = int(len(pieces)) * int(lesion_pts.shape[0])
        if exact_pairs > CLEARANCE_EXACT_PAIR_CAP:
            raise ValueError(
                "exact seed-clipped clearance exceeds bounded lesion pair "
                f"budget ({exact_pairs} > {CLEARANCE_EXACT_PAIR_CAP}); "
                "refusing a sampled eligible-path approximation"
            )
        distance = min(
            _exact_line_to_points_distance(piece, lesion_pts) for piece in pieces
        )
        return float(distance), True, 0.0

    pair_count = int(max(arr.shape[0] - 1, 1) * lesion_pts.shape[0])
    if pair_count <= CLEARANCE_EXACT_PAIR_CAP:
        return _exact_line_to_points_distance(arr, lesion_pts), True, 0.0

    sampled = _sample_polyline(arr, CLEARANCE_SAMPLE_SPACING_MM)
    d_lesion, _ = lesion_tree.query(sampled, k=1)
    return (
        float(np.min(d_lesion)),
        True,
        CLEARANCE_SAMPLE_SPACING_MM / 2.0,
    )


def _collect_clearance_distances(
    streamlines: list[np.ndarray],
    lesion_pts: np.ndarray,
    *,
    seed_pts: np.ndarray | None,
    seed_exclude_mm: float,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Shared path population for unweighted and weighted clearance."""
    from scipy.spatial import cKDTree

    lesion = _validate_points(lesion_pts, "lesion points")
    if lesion.shape[0] == 0:
        return np.empty(0, dtype=np.float64), 0, 0.0, np.empty(0, dtype=np.int64)
    if not np.isfinite(seed_exclude_mm) or seed_exclude_mm < 0.0:
        raise ValueError("seed_exclude_mm must be finite and >= 0")
    seed = None
    if seed_pts is not None:
        seed = _validate_points(seed_pts, "seed points")
    lesion_tree = cKDTree(lesion)
    seed_tree = cKDTree(seed) if seed is not None and seed.shape[0] else None
    distances: list[float] = []
    eligible_indices: list[int] = []
    n_seed_only = 0
    error_bound = 0.0
    for index, line in enumerate(streamlines):
        arr = _validate_points(line, "streamline")
        if arr.shape[0] == 0:
            continue
        distance, eligible, bound = _line_clearance_distance(
            arr, lesion, lesion_tree, seed_tree, seed, float(seed_exclude_mm)
        )
        error_bound = max(error_bound, float(bound))
        if not eligible:
            n_seed_only += 1
            continue
        assert distance is not None
        distances.append(float(distance))
        eligible_indices.append(index)
    return (
        np.asarray(distances, dtype=np.float64),
        n_seed_only,
        error_bound,
        np.asarray(eligible_indices, dtype=np.int64),
    )


def analytic_subset(
    streamlines: list[np.ndarray],
    *,
    cap: int = ANALYTIC_CAP,
    rng_seed: int = ANALYTIC_RNG_SEED,
) -> tuple[list[np.ndarray], dict, np.ndarray]:
    """Select streamlines for clearance stats (never the length-ranked display set).

    Returns (subset, meta, indices) where meta is ASCII-safe for HTTP headers:
      clearancePopulation: full | random_sample
      nAnalytic: size used for p5
      nAnalyticFull: size of the input analytic population
    ``indices`` maps subset back into the original list (for SIFT2 weight align).
    """
    n = len(streamlines)
    if n <= cap:
        idx = np.arange(n, dtype=np.int64)
        return list(streamlines), {
            "clearancePopulation": "full",
            "nAnalytic": n,
            "nAnalyticFull": n,
        }, idx
    rng = np.random.default_rng(int(rng_seed))
    idx = np.sort(rng.choice(n, size=int(cap), replace=False))
    subset = [streamlines[int(i)] for i in idx]
    return subset, {
        "clearancePopulation": "random_sample",
        "nAnalytic": int(cap),
        "nAnalyticFull": n,
        "clearanceSampleNote": (
            f"p5 on random {cap} of {n} streamlines "
            f"(rng={rng_seed}); not length-ranked display"
        ),
    }, idx


def lesion_surface_points(lesion_path: str) -> np.ndarray:
    """World-mm coordinates of the lesion's surface voxels (boundary shell)."""
    img = nib.load(lesion_path)
    m = np.asarray(img.dataobj) > 0
    if m.sum() == 0:
        return np.empty((0, 3))
    eroded = ndimage.binary_erosion(m)
    shell = m & ~eroded
    ijk = np.argwhere(shell if shell.sum() else m)
    return voxel_to_world(load_grid(lesion_path), ijk)


def seed_surface_points(seed_mask: np.ndarray, grid) -> np.ndarray:
    """World-mm surface voxels of a boolean seed mask (same grid as tracking)."""
    if seed_mask is None or not np.any(seed_mask):
        return np.empty((0, 3))
    m = np.asarray(seed_mask, dtype=bool)
    eroded = ndimage.binary_erosion(m)
    shell = m & ~eroded
    ijk = np.argwhere(shell if shell.sum() else m)
    return voxel_to_world(grid, ijk)


@dataclass(frozen=True)
class ClearanceReport:
    """Honest proximity summary. All distances in mm world space."""

    p05_mm: float | None          # raw 5th percentile (may be < floor)
    p50_mm: float | None
    n_streamlines: int
    n_eligible: int               # streamlines with an eligible path portion
    n_seed_only: int              # dropped: never left seed neighbourhood
    geom_floor_mm: float
    seed_exclude_mm: float
    method: str
    # Zero means exact point-to-continuous-segment distance.  Positive values
    # are an upper bound on the sampling bias for the bounded fallback.
    distance_error_bound_mm: float = 0.0

    @property
    def display_p05_mm(self) -> float | None:
        """Raw p5 retained for display alongside the separately recorded floor."""
        return self.p05_mm

    def format_p05(self) -> str:
        """UI string — never claims a sub-floor measurement reaches the floor."""
        if self.p05_mm is None:
            return ""
        if self.p05_mm < self.geom_floor_mm:
            # ASCII only — this string may ride in HTTP headers (latin-1)
            return f"<{format_floor_mm(self.geom_floor_mm)}"  # below recorded floor
        # one decimal max, and only if clearly above floor
        if self.p05_mm >= 10:
            return f"{self.p05_mm:.0f}"
        return f"{self.p05_mm:.1f}"

    def summary_line(self) -> str:
        if self.p05_mm is None:
            return "clearance: n/a"
        floor_note = (
            f"below recorded floor {format_floor_mm(self.geom_floor_mm)} mm"
            if self.p05_mm < self.geom_floor_mm
            else f"floor {format_floor_mm(self.geom_floor_mm)} mm geom."
        )
        error_note = (
            f"; distance error <={self.distance_error_bound_mm:g} mm"
            if self.distance_error_bound_mm > 0.0
            else ""
        )
        return (
            f"p5 clearance {self.format_p05()} mm "
            f"(n={self.n_eligible}/{self.n_streamlines}; "
            f"{floor_note}{error_note}; not navigation)"
        )


def _clearance_method(percentile, seed_pts, seed_exclude_mm, *, weighted=False):
    prefix = "SIFT2-weighted " if weighted else ""
    seed_note = (
        f"seed-excluded >={seed_exclude_mm:g} mm"
        if seed_pts is not None and seed_pts.shape[0] > 0
        else "all vertices; no seed exclusion"
    )
    return f"{prefix}p{percentile:g} of per-streamline min; {seed_note}"


def clearance_report(
    streamlines: list[np.ndarray],
    lesion_pts: np.ndarray,
    *,
    seed_pts: np.ndarray | None = None,
    seed_exclude_mm: float = SEED_EXCLUDE_MM,
    geom_floor_mm: float,  # REQUIRED — per-case, see geom_floor_mm(manifest)
    percentile: float = 5.0,
) -> ClearanceReport | None:
    """Per-streamline min distance → percentile on continuous paths.

    Returns None only when there is no lesion (caller shows nothing).
    Empty streamlines → report with n=0 and null distances (honest).  Small
    populations use exact point-to-segment distances to the lesion voxel-center
    shell.  Larger populations use 0.5 mm arclength samples with a recorded
    <=0.25 mm distance bias bound.  Seed exclusion clips the continuous path
    against the exact union of seed exclusion balls rather than only stored
    vertices.
    """
    lesion = _validate_points(lesion_pts, "lesion points")
    if lesion.shape[0] == 0:
        return None
    method = _clearance_method(percentile, seed_pts, seed_exclude_mm)
    if not np.isfinite(geom_floor_mm) or geom_floor_mm <= 0.0:
        raise ValueError("geom_floor_mm must be finite and > 0")
    if not streamlines:
        return ClearanceReport(
            p05_mm=None, p50_mm=None, n_streamlines=0, n_eligible=0, n_seed_only=0,
            geom_floor_mm=geom_floor_mm, seed_exclude_mm=seed_exclude_mm,
            method=method,
        )

    per_line, n_seed_only, error_bound, _eligible_idx = _collect_clearance_distances(
        streamlines,
        lesion,
        seed_pts=seed_pts,
        seed_exclude_mm=seed_exclude_mm,
    )

    n = len(streamlines)
    if not per_line.size:
        return ClearanceReport(
            p05_mm=None, p50_mm=None, n_streamlines=n, n_eligible=0,
            n_seed_only=n_seed_only, geom_floor_mm=geom_floor_mm,
            seed_exclude_mm=seed_exclude_mm,
            method=method,
            distance_error_bound_mm=error_bound,
        )

    return ClearanceReport(
        p05_mm=float(np.percentile(per_line, percentile)),
        p50_mm=float(np.percentile(per_line, 50)),
        n_streamlines=n,
        n_eligible=len(per_line),
        n_seed_only=n_seed_only,
        geom_floor_mm=geom_floor_mm,
        seed_exclude_mm=seed_exclude_mm,
        method=method,
        distance_error_bound_mm=error_bound,
    )


def clearance_report_weighted(
    streamlines: list[np.ndarray],
    weights: np.ndarray,
    lesion_pts: np.ndarray,
    *,
    seed_pts: np.ndarray | None = None,
    seed_exclude_mm: float = SEED_EXCLUDE_MM,
    geom_floor_mm: float,  # REQUIRED — per-case, see geom_floor_mm(manifest)
    percentile: float = 5.0,
) -> ClearanceReport | None:
    """Like clearance_report but p5/p50 are SIFT2-weighted percentiles.

    ``weights`` must align 1:1 with ``streamlines``. Streamlines dropped as
    seed-only are omitted from the weighted distribution. The distance
    population is produced by the same continuous-path helper as the
    unweighted report, so weighting changes only percentile mass.
    """
    from .sift2_util import weighted_percentile

    lesion = _validate_points(lesion_pts, "lesion points")
    if lesion.shape[0] == 0:
        return None
    method = _clearance_method(percentile, seed_pts, seed_exclude_mm, weighted=True)
    if not np.isfinite(geom_floor_mm) or geom_floor_mm <= 0.0:
        raise ValueError("geom_floor_mm must be finite and > 0")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != len(streamlines):
        # fall back unweighted
        return clearance_report(
            streamlines, lesion, seed_pts=seed_pts,
            seed_exclude_mm=seed_exclude_mm, geom_floor_mm=geom_floor_mm,
            percentile=percentile,
        )

    dists, n_seed_only, error_bound, eligible_idx = _collect_clearance_distances(
        streamlines,
        lesion,
        seed_pts=seed_pts,
        seed_exclude_mm=seed_exclude_mm,
    )
    if not dists.size:
        return ClearanceReport(
            p05_mm=None,
            p50_mm=None,
            n_streamlines=len(streamlines),
            n_eligible=0,
            n_seed_only=n_seed_only,
            geom_floor_mm=geom_floor_mm,
            seed_exclude_mm=seed_exclude_mm,
            method=method,
            distance_error_bound_mm=error_bound,
        )

    # The collector returns original input indices, so empty/seed-only lines
    # cannot shift SIFT2 weights relative to the distance population.
    if eligible_idx.size != int(dists.size):
        raise ValueError("weighted clearance population lost streamline alignment")
    ww = np.clip(w[eligible_idx], 0.0, None)
    return ClearanceReport(
        p05_mm=weighted_percentile(dists, ww, percentile),
        p50_mm=weighted_percentile(dists, ww, 50.0),
        n_streamlines=len(streamlines),
        n_eligible=int(dists.size),
        n_seed_only=n_seed_only,
        geom_floor_mm=geom_floor_mm,
        seed_exclude_mm=seed_exclude_mm,
        method=method,
        distance_error_bound_mm=error_bound,
    )


def min_clearance_mm(streamlines: list[np.ndarray], lesion_pts: np.ndarray) -> float | None:
    """DEPRECATED path kept for tests — global min (unsafe as a clinical number).

    Prefer :func:`clearance_report`.  Even this compatibility helper measures
    continuous segments so sparse and densely inserted collinear vertices do
    not change the result. Returns None if no lesion / no lines.
    """
    lesion = _validate_points(lesion_pts, "lesion points")
    if lesion.shape[0] == 0 or not streamlines:
        return None
    from scipy.spatial import cKDTree

    tree = cKDTree(lesion)
    values: list[float] = []
    for line in streamlines:
        arr = _validate_points(line, "streamline")
        if arr.shape[0] == 0:
            continue
        pairs = int(max(arr.shape[0] - 1, 1) * lesion.shape[0])
        if pairs <= CLEARANCE_EXACT_PAIR_CAP:
            values.append(_exact_line_to_points_distance(arr, lesion))
        else:
            sampled = _sample_polyline(arr, CLEARANCE_SAMPLE_SPACING_MM)
            d, _ = tree.query(sampled, k=1)
            values.append(float(np.min(d)))
    return float(np.min(values)) if values else None


def per_vertex_distance_mm(verts_lk3: np.ndarray, lesion_pts: np.ndarray) -> np.ndarray | None:
    """Distance (mm) from each streamline vertex to the lesion surface.

    Used for heat colour only — not for the clearance statistic.
    """
    if lesion_pts.shape[0] == 0 or verts_lk3.size == 0:
        return None
    from scipy.spatial import cKDTree
    L, K = verts_lk3.shape[0], verts_lk3.shape[1]
    tree = cKDTree(lesion_pts)
    d, _ = tree.query(verts_lk3.reshape(-1, 3), k=1)
    return d.reshape(L, K).astype("<f4")
