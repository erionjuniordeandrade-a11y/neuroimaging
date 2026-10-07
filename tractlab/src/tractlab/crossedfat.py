"""Crossed frontal aslant tract — bilateral SMA/premotor, path-gated.

The gate here is INVERTED relative to every other bundle in this project. For a
CST or an SLF, a streamline crossing the midline is a red flag; a waypoint-gated
"CST" built that way was 0.81% real with 20.26% crossing the contralateral
peduncle. For this bundle, crossing IS the anatomy.

⛔ THE TRAP THIS MODULE NOW GUARDS. Inverting the veto removed the contralateral
check and left NO path condition at all: the gate constrained only the two
endpoints' parcel LABELS. On real data it validated 202 streamlines of which 8
were 10-19 mm hairpins that never crossed the midline, because the rh SMA
territory leaks 73 voxels into the left hemisphere (to x = -3.58 mm) after the
parcellation was resampled across a ~27-28 mm inter-session registration. Labels
inherit that registration error; the streamline's own vertices do not. So three
conjuncts are added ON TOP OF the endpoint anchor, each computed from the
streamline's own geometry, each yielding a per-streamline rejection reason and a
reportable count:

1. the lh-labelled endpoint lies at least ``standoff_mm`` LEFT of the measured
   midline and the rh-labelled endpoint at least that far RIGHT of it;
2. the streamline crosses the midline exactly once (odd is necessary, one is
   anatomical);
3. that crossing lies within ``cc_tol_mm`` of a corpus-callosum voxel.

Conjunct 3 is an impossibility veto, not a waypoint filter. ACT does not model
the falx, so an FOD peak can bridge the thin interhemispheric CSF gap between the
two abutting medial gyri; a "crossing" at the medial cortical surface is an
artifact rather than a callosal fibre. These remain legitimate under this
project's anti-waypoint ruling because each is a property of the streamline's own
geometry and is individually reported — unlike an ``-include`` ROI, which
constrains where a streamline PASSES and reports nothing.

⛔ THE MIDLINE IS NOT x = 0 FOR THIS SUBJECT. Labels 251-255 span world x
[-5.13, +0.93] with median -1.93 mm, so assuming zero carries ~2 mm of bias into
every side and crossing decision. ``x_mid`` is measured from those labels, and
their absence raises rather than falling back to zero.

⛔ DK-84 has NO SMA parcel. SMA falls inside ``superiorfrontal``. The territory
used here is the POSTERIOR THIRD OF THE AP EXTENT of ``superiorfrontal`` per
hemisphere. That is a stated choice and must be reported as such, never presented
as an anatomical label. HCP-MMP (6ma / 6mp / SCEF) would localise it properly,
but multiplying parcels worsens the ACT-off endpoint-assignment problem, so it is
not used yet.

⛔ NO THRESHOLD HERE MAY BE TUNED TO RAISE THE VALIDATED COUNT. The defaults are
argued from the stated geometric uncertainty budget, and ``report`` prints the
full sensitivity curve so a reader sees how the number moves instead of one
cherry-picked figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .fivett import (
    endpoints_from_streamlines,
    endpoints_in_bool_mask,
    endpoints_in_mask,
    read_streamlines,
    tck_completeness,
)
from .grid import Grid, voxel_to_world
from .parcellation import LABEL_IDS, load_parcellation

_POSTERIOR_FRACTION = 1.0 / 3.0

#: aparc+aseg corpus callosum labels, verified against FreeSurferColorLUT.txt:
#: 251 CC_Posterior, 252 CC_Mid_Posterior, 253 CC_Central, 254 CC_Mid_Anterior,
#: 255 CC_Anterior. 250 (Fornix) is deliberately NOT included.
CC_LABEL_IDS: tuple[int, ...] = (251, 252, 253, 254, 255)

#: Minimum |x - x_mid| for an endpoint to count as belonging to a hemisphere.
#: Argued, not tuned: the rh territory leaks to 3.58 mm past the midline on this
#: subject, and the report already declares a few-mm geometric uncertainty, so a
#: standoff inside that band would be accepting registration error as anatomy.
DEFAULT_STANDOFF_MM = 5.0

#: Maximum distance from the midline crossing to a callosal voxel centre.
#: Argued, not tuned: one voxel half-diagonal on this 1.3 mm grid is ~1.13 mm,
#: and the CC label carries the same few-mm inter-session registration residual
#: as every other parcel. A tolerance tighter than the declared uncertainty of
#: the label's own position would reject fibres for registration error rather
#: than for anatomy.
DEFAULT_CC_TOL_MM = 5.0

#: Tolerances reported alongside the chosen one, so the sensitivity is visible.
CC_TOL_SWEEP_MM: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0)
STANDOFF_SWEEP_MM: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)

REASON_NOT_ANCHORED = "endpoints not one in each SMA territory"
REASON_SAME_ENDPOINT = "one endpoint labelled in both territories (masks overlap)"
REASON_STANDOFF = "endpoints not on opposite sides of the midline beyond the standoff"
REASON_CROSSING_COUNT = "midline crossings != 1"
REASON_FOOTPRINT = "midline crossing outside the callosal footprint"
REASON_LESION = "an endpoint terminates inside the lesion"

_REASON_ORDER = (
    REASON_NOT_ANCHORED,
    REASON_SAME_ENDPOINT,
    REASON_STANDOFF,
    REASON_CROSSING_COUNT,
    REASON_FOOTPRINT,
    REASON_LESION,
)

_IMPOSSIBLE_REASONS = frozenset(
    {REASON_SAME_ENDPOINT, REASON_STANDOFF, REASON_CROSSING_COUNT, REASON_FOOTPRINT}
)


# --------------------------------------------------------------------------
# territory
# --------------------------------------------------------------------------

def _posterior_axis(axcodes: tuple[str, str, str]) -> tuple[int, bool]:
    """``(voxel axis index, posterior_is_high_index)`` for a volume's A/P axis.

    ⛔ Derived from the volume, never assumed. The earlier version hardcoded
    voxel axis ``j`` and "posterior == high j" on this case's ("L","P","S")
    axcodes. On an ("L","A","S") volume that silently returns the ANTERIOR third
    — still a third of the parcel, still hemisphere-specific, so no downstream
    check notices — which is the ``.WRONG_ANTERIOR.bak`` error the sibling tree
    already made once.
    """
    for axis, code in enumerate(axcodes):
        if code == "P":
            return axis, True
        if code == "A":
            return axis, False
    raise ValueError(
        f"axcodes {axcodes!r} has no anterior/posterior axis — cannot locate a "
        "posterior third; an anatomical volume must have one"
    )


def sma_territory_mask(
    labels: np.ndarray, hemi: str, axcodes: tuple[str, str, str]
) -> np.ndarray:
    """Posterior third of the AP extent of ``superiorfrontal``, one hemisphere.

    ``hemi`` is "lh" or "rh". ``axcodes`` is ``Grid.axcodes`` for the volume the
    labels came from; it is required, not defaulted, so no caller can inherit the
    ("L","P","S") assumption by omission.

    "Posterior third" means the third of the parcel's anteroposterior EXTENT
    (bounding box) nearest the posterior end — not a third of its voxel count,
    and not, as before, a third of the extent plus the boundary slice (which kept
    36.7% of a 30-slice parcel).

    ⛔ Raises when the parcel has zero voxels. Returning an all-False mask made a
    failed parcellation transfer report "validated 0 / 20000 (0.00%)", which
    reads as a negative result rather than as a broken input.

    Returns a boolean array shaped like ``labels``.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError(f"hemi must be 'lh' or 'rh', got {hemi!r}")
    name = f"{hemi}_superiorfrontal"
    parent = labels == LABEL_IDS[name]
    if not parent.any():
        raise ValueError(
            f"{name} has zero voxels in this volume — the parcellation did not "
            "transfer. Refusing to return an empty territory, which would be "
            "reported as a validated count of zero."
        )

    axis, posterior_is_high = _posterior_axis(axcodes)
    idx = np.argwhere(parent)[:, axis]
    lo, hi = int(idx.min()), int(idx.max())
    extent = hi - lo + 1
    depth = extent * _POSTERIOR_FRACTION

    positions = np.arange(labels.shape[axis])
    # strict "<" against the fractional depth keeps exact thirds: 30 slices -> 10,
    # 10 slices -> 3. Never the boundary slice twice.
    keep = (hi - positions) < depth if posterior_is_high else (positions - lo) < depth

    broadcast = [1, 1, 1]
    broadcast[axis] = labels.shape[axis]
    return parent & keep.reshape(broadcast)


# --------------------------------------------------------------------------
# midsagittal reference
# --------------------------------------------------------------------------

def callosal_voxels_world(grid: Grid, labels: np.ndarray) -> np.ndarray:
    """World-mm centres of every corpus-callosum voxel, ``(M, 3)``.

    ⛔ Raises when no CC label survives. A silent fallback to x = 0 would present
    an assumption as a measurement, and every side and crossing decision in this
    module depends on it.
    """
    ijk = np.argwhere(np.isin(labels, CC_LABEL_IDS))
    if ijk.shape[0] == 0:
        raise ValueError(
            "no corpus callosum labels (251-255) in this volume — the "
            "midsagittal plane cannot be measured. Refusing to assume x = 0."
        )
    return voxel_to_world(grid, ijk)


def midsagittal_x_mm(cc_world_xyz: np.ndarray) -> float:
    """Median world x of the callosal voxels: this subject's midsagittal plane."""
    cc = np.asarray(cc_world_xyz, dtype=np.float64)
    if cc.size == 0:
        raise ValueError("cannot measure a midsagittal plane from zero voxels")
    return float(np.median(cc[:, 0]))


def midline_crossings(vertices: np.ndarray, x_mid_mm: float) -> np.ndarray:
    """Points ``(K, 3)`` where a streamline's polyline crosses ``x = x_mid_mm``.

    Vertices lying exactly on the plane take the sign of the preceding side, so
    grazing the plane and returning is one side (zero crossings) while passing
    through it is one crossing. Crossing points are linearly interpolated along
    the straddling segment, not snapped to the nearer vertex.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 2:
        return np.empty((0, 3), dtype=np.float64)

    offset = vertices[:, 0] - x_mid_mm
    sign = np.sign(offset)
    nonzero = sign != 0
    if not nonzero.any():
        return np.empty((0, 3), dtype=np.float64)

    # carry the last non-zero sign forward over any exact zeros
    carry = np.where(nonzero, np.arange(sign.size), 0)
    np.maximum.accumulate(carry, out=carry)
    filled = sign[carry]
    first = int(np.argmax(nonzero))
    filled[:first] = sign[first]

    straddle = np.flatnonzero(filled[:-1] * filled[1:] < 0)
    if straddle.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    a, b = offset[straddle], offset[straddle + 1]
    t = (a / (a - b))[:, None]
    return vertices[straddle] + t * (vertices[straddle + 1] - vertices[straddle])


def _nearest_distance_mm(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest point of ``cloud``, in mm.

    Chunked brute force rather than a KD-tree so this module needs no extra
    dependency; the real-data test cross-checks it with ``scipy.spatial.cKDTree``,
    which makes the two implementations independent.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0,), dtype=np.float64)
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], 1024):
        block = points[start : start + 1024]
        d2 = ((block[:, None, :] - cloud[None, :, :]) ** 2).sum(axis=-1)
        out[start : start + block.shape[0]] = np.sqrt(d2.min(axis=1))
    return out


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossedFatQC:
    """Endpoint- AND path-anchored accounting. Never report a ``-select`` yield.

    ``n_impossible``, ``n_complete`` and ``n_validated`` are a genuine partition
    of ``n_total`` — the constructor refuses any other arithmetic.

    ⛔ ``n_lesion_terminating`` counts every streamline with an endpoint in the
    lesion, over the whole tractogram; it overlaps the candidate set only by
    ``n_candidate_lesion_terminating``. Subtracting it from the candidate count
    is wrong, which is why the report prints the overlap.
    """

    n_total: int
    n_bihemispheric: int
    n_lesion_terminating: int
    n_validated: int
    n_impossible: int
    n_complete: int
    n_candidate_lesion_terminating: int
    x_mid_mm: float | None
    standoff_mm: float
    cc_tol_mm: float
    rejections: Mapping[str, int] = field(default_factory=dict)
    conjunct_failures: Mapping[str, int] = field(default_factory=dict)
    validated_indices: tuple[int, ...] = ()
    callosal_distance_mm: tuple[float, ...] = ()
    validated_by_cc_tol_mm: Mapping[float, int] = field(default_factory=dict)
    validated_by_standoff_mm: Mapping[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.n_impossible + self.n_complete + self.n_validated
        if total != self.n_total:
            raise ValueError(
                f"not a partition: impossible {self.n_impossible} + complete "
                f"{self.n_complete} + validated {self.n_validated} = {total}, "
                f"but n_total is {self.n_total}"
            )

    @property
    def report(self) -> str:
        if self.n_total:
            headline = (
                f"validated {self.n_validated} / {self.n_total} "
                f"({100.0 * self.n_validated / self.n_total:.2f}%)"
            )
        else:
            headline = (
                f"validated {self.n_validated} / {self.n_total} "
                "(percentage unknown: no streamlines were read)"
            )
        x_mid = "unknown" if self.x_mid_mm is None else f"{self.x_mid_mm:.2f} mm"

        lines = [
            f"crossed FAT: {headline}",
            f"partition (impossible : complete : validated) = "
            f"{self.n_impossible} : {self.n_complete} : {self.n_validated}",
            f"endpoint-anchored candidates {self.n_bihemispheric} | "
            f"lesion-terminating {self.n_lesion_terminating} "
            f"(overlap with candidates: {self.n_candidate_lesion_terminating})",
            f"midsagittal plane x_mid = {x_mid}, measured as the median world x "
            "of aparc labels 251-255 (corpus callosum); NOT assumed to be x = 0",
            f"path conditions: each endpoint >= {self.standoff_mm:.1f} mm from "
            f"x_mid on its own side (standoff), exactly one crossing of x_mid, "
            f"crossing within {self.cc_tol_mm:.1f} mm of a callosal voxel",
        ]

        if self.rejections:
            lines.append("rejected by (FIRST matching reason, so later rows are "
                         "shadowed by earlier ones):")
            lines += [
                f"  {count:>7d}  {reason}"
                for reason, count in self.rejections.items()
                if count
            ]
        if self.conjunct_failures:
            lines.append(
                f"each conjunct's INDEPENDENT failure count among the "
                f"{self.n_bihemispheric} candidates (a 0 above may only mean the "
                "reason was shadowed, never that no such streamline exists):"
            )
            lines += [
                f"  {count:>7d}  {reason}"
                for reason, count in self.conjunct_failures.items()
            ]

        if self.validated_by_cc_tol_mm or self.validated_by_standoff_mm:
            lines.append(
                "sensitivity (thresholds are argued from the stated geometric "
                "uncertainty, NOT tuned to the yield):"
            )
        if self.validated_by_cc_tol_mm:
            lines.append(
                "  callosal tolerance   "
                + " | ".join(
                    f"{tol:.1f} mm -> {n}"
                    for tol, n in sorted(self.validated_by_cc_tol_mm.items())
                )
            )
        if self.validated_by_standoff_mm:
            lines.append(
                "  midline standoff     "
                + " | ".join(
                    f"{so:.1f} mm -> {n}"
                    for so, n in sorted(self.validated_by_standoff_mm.items())
                )
            )
        if self.callosal_distance_mm:
            d = np.asarray(self.callosal_distance_mm)
            within = " | ".join(
                f"<={tol:.0f} mm {100.0 * float((d <= tol).mean()):.1f}%"
                for tol in CC_TOL_SWEEP_MM
            )
            lines.append(
                f"  crossing-to-callosum distance (n={d.size}, median "
                f"{float(np.median(d)):.2f} mm): {within}"
            )

        lines += [
            "SMA territory = posterior third of the AP extent of superiorfrontal "
            "(stated choice, not an atlas label).",
            "Anatomical aid only, not navigation; ~few-mm geometric uncertainty "
            "(no topup/reverse-PE).",
            "'impossible' is assessed only for the endpoint-anchored candidates; "
            "other streamlines are counted complete without a geometric test.",
        ]
        return "\n".join(lines)


def _candidate_geometry(
    streamlines: Sequence[np.ndarray],
    endpoint_x: np.ndarray,
    in_lh: np.ndarray,
    in_rh: np.ndarray,
    x_mid_mm: float,
    cc_world_xyz: np.ndarray,
) -> dict[int, tuple[float, float, int, float]]:
    """Per-candidate ``(d_lh, d_rh, n_crossings, crossing_distance_mm)``.

    Computed once, independently of any threshold, so the sensitivity sweep is
    pure arithmetic and cannot accidentally become a second, differently-behaving
    gate.
    """
    lh_idx = np.argmax(in_lh, axis=1)
    rh_idx = np.argmax(in_rh, axis=1)
    anchored = np.flatnonzero((in_lh.sum(axis=1) == 1) & (in_rh.sum(axis=1) == 1))

    crossings: dict[int, np.ndarray] = {}
    for i in anchored:
        crossings[int(i)] = midline_crossings(streamlines[i], x_mid_mm)

    single = [i for i in crossings if crossings[i].shape[0] == 1]
    if single:
        points = np.array([crossings[i][0] for i in single])
        distances = _nearest_distance_mm(points, cc_world_xyz)
    else:
        distances = np.empty((0,), dtype=np.float64)
    distance_by_index = dict(zip(single, (float(d) for d in distances)))

    geometry: dict[int, tuple[float, float, int, float]] = {}
    for i in anchored:
        i = int(i)
        geometry[i] = (
            float(endpoint_x[i, lh_idx[i]] - x_mid_mm),
            float(endpoint_x[i, rh_idx[i]] - x_mid_mm),
            int(crossings[i].shape[0]),
            distance_by_index.get(i, float("nan")),
        )
    return geometry


def _bucket_reasons(
    n_total: int,
    geometry: Mapping[int, tuple[float, float, int, float]],
    same_endpoint: np.ndarray,
    lesion: np.ndarray,
    standoff_mm: float,
    cc_tol_mm: float,
) -> dict[int, str]:
    """Rejection reason per streamline index; absent index == validated."""
    reasons: dict[int, str] = {}
    for i in range(n_total):
        if same_endpoint[i]:
            reasons[i] = REASON_SAME_ENDPOINT
            continue
        if i not in geometry:
            reasons[i] = REASON_NOT_ANCHORED
            continue
        d_lh, d_rh, n_cross, cc_distance = geometry[i]
        if not (d_lh <= -standoff_mm and d_rh >= standoff_mm):
            reasons[i] = REASON_STANDOFF
        elif n_cross != 1:
            reasons[i] = REASON_CROSSING_COUNT
        elif not (cc_distance <= cc_tol_mm):
            reasons[i] = REASON_FOOTPRINT
        elif lesion[i]:
            reasons[i] = REASON_LESION
    return reasons


def classify_streamlines(
    in_lh: np.ndarray,
    in_rh: np.ndarray,
    in_lesion: np.ndarray,
    streamlines: Sequence[np.ndarray],
    *,
    x_mid_mm: float,
    cc_world_xyz: np.ndarray,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    cc_tol_mm: float = DEFAULT_CC_TOL_MM,
) -> CrossedFatQC:
    """Apply the endpoint anchor AND the path conditions. Pure; no file access.

    ``in_lh``, ``in_rh`` and ``in_lesion`` are ``(N, 2)`` booleans per endpoint;
    ``streamlines`` holds the matching ``N`` vertex arrays in world mm. The
    streamline geometry is required, not optional, so no caller can fall back to
    the endpoint-only gate that admitted non-transcallosal hairpins.

    Validated requires ALL of: exactly one endpoint in each SMA territory; the
    lh-labelled endpoint at least ``standoff_mm`` left of ``x_mid_mm`` and the
    rh-labelled endpoint at least that far right of it; exactly one crossing of
    ``x_mid_mm``; that crossing within ``cc_tol_mm`` of a callosal voxel; and
    neither endpoint inside the lesion.
    """
    in_lh = np.asarray(in_lh, dtype=bool)
    in_rh = np.asarray(in_rh, dtype=bool)
    in_lesion = np.asarray(in_lesion, dtype=bool)

    if not (in_lh.shape == in_rh.shape == in_lesion.shape):
        raise ValueError(
            f"endpoint membership arrays disagree in shape: in_lh {in_lh.shape}, "
            f"in_rh {in_rh.shape}, in_lesion {in_lesion.shape}"
        )
    if in_lh.ndim != 2 or in_lh.shape[1] != 2:
        raise ValueError(f"membership arrays must be (N, 2), got {in_lh.shape}")
    n_total = int(in_lh.shape[0])
    if len(streamlines) != n_total:
        raise ValueError(
            f"{len(streamlines)} streamlines but membership arrays have length "
            f"{n_total} — a mismatched N mis-aligns every row"
        )

    endpoints = endpoints_from_streamlines(list(streamlines))
    endpoint_x = (
        endpoints[:, :, 0] if n_total else np.empty((0, 2), dtype=np.float64)
    )
    anchored_mask = (in_lh.sum(axis=1) == 1) & (in_rh.sum(axis=1) == 1)
    same_endpoint = anchored_mask & (in_lh & in_rh).any(axis=1)
    lesion = in_lesion.any(axis=1)

    geometry = _candidate_geometry(
        streamlines, endpoint_x, in_lh & ~same_endpoint[:, None],
        in_rh & ~same_endpoint[:, None], x_mid_mm, cc_world_xyz,
    )

    reasons = _bucket_reasons(
        n_total, geometry, same_endpoint, lesion, standoff_mm, cc_tol_mm
    )
    validated_indices = tuple(i for i in range(n_total) if i not in reasons)
    n_impossible = sum(1 for r in reasons.values() if r in _IMPOSSIBLE_REASONS)
    n_complete = len(reasons) - n_impossible

    tally = {reason: 0 for reason in _REASON_ORDER}
    for reason in reasons.values():
        tally[reason] += 1

    def _validated_count(standoff: float, tol: float) -> int:
        rejected = _bucket_reasons(
            n_total, geometry, same_endpoint, lesion, standoff, tol
        )
        return n_total - len(rejected)

    distances = tuple(
        d for (_, _, n_cross, d) in geometry.values() if n_cross == 1 and d == d
    )

    # Independent, UNSHADOWED counts. _bucket_reasons stops at the first matching
    # reason, so a zero there can mean "shadowed", not "none exist" — reporting
    # only the first-match tally would have shown "midline crossings != 1: 0" on
    # real data while all 7 hairpins were present and rejected on the standoff.
    conjunct_failures = {
        REASON_STANDOFF: sum(
            1 for (d_lh, d_rh, _, _) in geometry.values()
            if not (d_lh <= -standoff_mm and d_rh >= standoff_mm)
        ),
        REASON_CROSSING_COUNT: sum(
            1 for (_, _, n_cross, _) in geometry.values() if n_cross != 1
        ),
        REASON_FOOTPRINT: sum(
            1 for (_, _, n_cross, d) in geometry.values()
            if n_cross == 1 and not (d <= cc_tol_mm)
        ),
        REASON_LESION: int((anchored_mask & lesion).sum()),
    }

    return CrossedFatQC(
        n_total=n_total,
        n_bihemispheric=int(anchored_mask.sum()),
        n_lesion_terminating=int(lesion.sum()),
        n_validated=len(validated_indices),
        n_impossible=n_impossible,
        n_complete=n_complete,
        n_candidate_lesion_terminating=int((anchored_mask & lesion).sum()),
        x_mid_mm=float(x_mid_mm),
        standoff_mm=float(standoff_mm),
        cc_tol_mm=float(cc_tol_mm),
        rejections=tally,
        conjunct_failures=conjunct_failures,
        validated_indices=validated_indices,
        callosal_distance_mm=distances,
        validated_by_cc_tol_mm={
            tol: _validated_count(standoff_mm, tol) for tol in CC_TOL_SWEEP_MM
        },
        validated_by_standoff_mm={
            so: _validated_count(so, cc_tol_mm) for so in STANDOFF_SWEEP_MM
        },
    )


def validate_crossed_fat(
    tck_path: str,
    aparc_path: str,
    lesion_path: str,
    *,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    cc_tol_mm: float = DEFAULT_CC_TOL_MM,
) -> CrossedFatQC:
    """Full validation of a candidate crossed-FAT tractogram.

    ⛔ Refuses an unfinished ``.tck``. Reading a partially written tractogram
    produces a smaller, quietly wrong denominator instead of an error.
    """
    completeness = tck_completeness(tck_path)
    if not completeness.is_complete:
        raise ValueError(f"{tck_path} is not a finished tractogram: {completeness.reason}")

    grid, labels = load_parcellation(aparc_path)
    cc_world_xyz = callosal_voxels_world(grid, labels)
    x_mid_mm = midsagittal_x_mm(cc_world_xyz)

    streamlines = read_streamlines(tck_path)
    endpoints = endpoints_from_streamlines(streamlines)
    in_lh = endpoints_in_bool_mask(
        endpoints, grid, sma_territory_mask(labels, "lh", grid.axcodes)
    )
    in_rh = endpoints_in_bool_mask(
        endpoints, grid, sma_territory_mask(labels, "rh", grid.axcodes)
    )
    in_lesion = endpoints_in_mask(endpoints, lesion_path)

    return classify_streamlines(
        in_lh, in_rh, in_lesion, streamlines,
        x_mid_mm=x_mid_mm,
        cc_world_xyz=cc_world_xyz,
        standoff_mm=standoff_mm,
        cc_tol_mm=cc_tol_mm,
    )
