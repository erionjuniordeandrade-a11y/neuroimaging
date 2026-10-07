"""Minimal MRtrix .tck streamline reader/writer.

Coordinates are world RAS+ millimetres (MRtrix scanner space) and are never
transformed here. The writer emits a minimal header only: source header lines
can carry file paths or identifiers and are never copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from neuro_core.tck import decode_tck, encode_tck, read_tck, write_tck  # noqa: F401  (re-exported)


TRUST_THRESHOLDS = {
    "tortuosity_ratio_max": 1.5,
    "median_tortuosity_max": 3.5,
    "cap_fraction_max": 0.10,
    "min_streamlines": 50,
    "rim_mm": 5.0,
    "rim_fraction_flag": 0.30,
    "yield_ratio_flag": 1 / 3,
}
TRUST_CALIBRATION = "pilot: 1 case, 10 contralateral pairs"
_TRUST_SAMPLE_MAX = 5000
_RIM_SAMPLE_STEP_MM = 2.0


@dataclass(frozen=True)
class LesionGrid:
    """A boolean lesion array and the RAS-mm affine for its [k, j, i] grid."""

    mask_kji: np.ndarray
    affine_ras: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask_kji, dtype=bool)
        affine = np.asarray(self.affine_ras, dtype=float)
        if mask.ndim != 3:
            raise ValueError("lesion mask must be a three-dimensional [k, j, i] array")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("lesion affine must be a finite 4x4 RAS transform")
        if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("lesion affine must have a homogeneous final row")
        if np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError("lesion affine must have an invertible spatial transform")
        object.__setattr__(self, "mask_kji", mask)
        object.__setattr__(self, "affine_ras", affine)


def subsample(streamlines: list[np.ndarray], max_n: int, seed: int = 0) -> list[np.ndarray]:
    """Deterministic uniform subset of at most max_n streamlines, source order kept."""
    if max_n < 1:
        raise ValueError("max_n must be at least 1")
    if len(streamlines) <= max_n:
        return list(streamlines)
    chosen = np.sort(np.random.default_rng(seed).choice(len(streamlines), size=max_n, replace=False))
    return [streamlines[index] for index in chosen]



def _resample(streamline: np.ndarray, n: int) -> np.ndarray:
    """n points equally spaced by arc length."""
    if len(streamline) < 2:
        return np.repeat(streamline[:1], n, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))])
    target = np.linspace(0.0, arc[-1], n)
    return np.stack([np.interp(target, arc, streamline[:, axis]) for axis in range(3)], axis=1)


def length_mm(streamline: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(streamline, axis=0), axis=1).sum()) if len(streamline) > 1 else 0.0


def _tortuosity(streamline: np.ndarray) -> float:
    points = np.asarray(streamline, dtype=float)
    if len(points) < 2:
        return 0.0
    end_to_end = float(np.linalg.norm(points[-1] - points[0]))
    return length_mm(points) / max(end_to_end, 1.0)


def _contralateral_label(label: str) -> str | None:
    if label.endswith("_l"):
        return f"{label[:-2]}_r"
    if label.endswith("_r"):
        return f"{label[:-2]}_l"
    return None


def _lesion_distance_grid(lesion: LesionGrid) -> np.ndarray | None:
    """Physical distance to the lesion surface at each lesion-grid voxel centre."""
    if not lesion.mask_kji.any():
        return None
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(lesion.mask_kji.astype(np.uint8))
    spacing = np.linalg.norm(lesion.affine_ras[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("lesion affine must have non-zero finite voxel spacing")
    image.SetSpacing(spacing.tolist())
    distance = sitk.SignedMaurerDistanceMap(image, insideIsPositive=False,
                                             squaredDistance=False, useImageSpacing=True)
    return sitk.GetArrayFromImage(distance).astype(float)


def _sample_grid_trilinear(values_kji: np.ndarray, affine_ras: np.ndarray, points_ras: np.ndarray) -> np.ndarray:
    """Sample a [k, j, i] scalar grid at RAS points; outside values are infinity."""
    points = np.asarray(points_ras, dtype=float)
    if not len(points):
        return np.empty(0, dtype=float)
    inverse = np.linalg.inv(affine_ras)
    ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
    dims = np.asarray(values_kji.shape[::-1], dtype=int)
    valid = np.all((ijk >= 0) & (ijk <= dims - 1), axis=1)
    lower = np.floor(ijk).astype(int)
    lower = np.clip(lower, 0, dims - 1)
    upper = np.minimum(lower + 1, dims - 1)
    fraction = ijk - lower
    sampled = np.zeros(len(points), dtype=float)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                weights = ((fraction[:, 0] if di else 1.0 - fraction[:, 0]) *
                           (fraction[:, 1] if dj else 1.0 - fraction[:, 1]) *
                           (fraction[:, 2] if dk else 1.0 - fraction[:, 2]))
                i = upper[:, 0] if di else lower[:, 0]
                j = upper[:, 1] if dj else lower[:, 1]
                k = upper[:, 2] if dk else lower[:, 2]
                sampled += weights * values_kji[k, j, i]
    sampled[~valid] = np.inf
    return sampled


def _points_inside_lesion(lesion: LesionGrid, points_ras: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ras, dtype=float)
    if not len(points):
        return np.zeros(0, dtype=bool)
    inverse = np.linalg.inv(lesion.affine_ras)
    ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
    nearest = np.floor(ijk + 0.5).astype(int)
    dims = np.asarray(lesion.mask_kji.shape[::-1], dtype=int)
    valid = np.all((nearest >= 0) & (nearest < dims), axis=1)
    inside = np.zeros(len(points), dtype=bool)
    valid_nearest = nearest[valid]
    inside[valid] = lesion.mask_kji[valid_nearest[:, 2], valid_nearest[:, 1], valid_nearest[:, 0]]
    return inside


def _rim_fraction(streamlines: Sequence[np.ndarray], lesion: LesionGrid) -> float:
    distance = _lesion_distance_grid(lesion)
    if distance is None:
        return 0.0
    fractions = []
    for streamline in streamlines:
        points = np.asarray(streamline, dtype=float)
        count = max(1, int(np.ceil(length_mm(points) / _RIM_SAMPLE_STEP_MM)) + 1)
        sampled = _resample(points, count)
        near_surface = _sample_grid_trilinear(distance, lesion.affine_ras, sampled) <= TRUST_THRESHOLDS["rim_mm"]
        fractions.append(float(np.mean(_points_inside_lesion(lesion, sampled) | near_surface)))
    return float(np.mean(fractions)) if fractions else 0.0


def compute_tract_trust(bundles: Mapping[str, Sequence[np.ndarray]], *, tract_max_length_mm: float = 250.0,
                        lesion: LesionGrid | None = None) -> dict[str, dict]:
    """Return manifest-ready trust blocks from full source TCK streamlines in world RAS mm."""
    if not math.isfinite(tract_max_length_mm) or tract_max_length_mm <= 0:
        raise ValueError("tract_max_length_mm must be finite and positive")
    source = {str(label): list(streamlines) for label, streamlines in bundles.items()}
    sampled = {label: subsample(streamlines, _TRUST_SAMPLE_MAX, seed=0)
               for label, streamlines in source.items()}
    summaries = {
        label: {
            "n_source": len(streamlines),
            "median_tortuosity": float(np.median([_tortuosity(item) for item in sampled[label]]))
            if sampled[label] else 0.0,
            "cap_fraction": float(np.mean([length_mm(item) >= tract_max_length_mm - 2
                                            for item in streamlines])) if streamlines else 0.0,
            "rim_fraction": _rim_fraction(sampled[label], lesion) if lesion is not None else None,
        }
        for label, streamlines in source.items()
    }
    reports = {}
    minimum = TRUST_THRESHOLDS["min_streamlines"]
    for label, metrics in summaries.items():
        partner = _contralateral_label(label)
        contralateral = partner if partner in summaries else None
        ratio = yield_ratio = None
        if contralateral is not None:
            other = summaries[contralateral]
            if metrics["n_source"] >= minimum and other["n_source"] >= minimum and other["median_tortuosity"]:
                ratio = float(metrics["median_tortuosity"] / other["median_tortuosity"])
            if other["n_source"] >= minimum:
                yield_ratio = float(metrics["n_source"] / other["n_source"])
        failing = []
        if metrics["n_source"] < minimum:
            verdict = "UNAVAILABLE"
        else:
            if ratio is not None and ratio > TRUST_THRESHOLDS["tortuosity_ratio_max"]:
                failing.append("tortuosity_ratio")
            if metrics["median_tortuosity"] > TRUST_THRESHOLDS["median_tortuosity_max"]:
                failing.append("median_tortuosity")
            if metrics["cap_fraction"] > TRUST_THRESHOLDS["cap_fraction_max"]:
                failing.append("cap_fraction")
            verdict = "FAIL" if failing else "PASS"
        reports[label] = {
            "verdict": verdict,
            "failing": failing,
            "rim_flag": bool(metrics["rim_fraction"] is not None
                             and metrics["rim_fraction"] > TRUST_THRESHOLDS["rim_fraction_flag"]),
            "yield_flag": bool(yield_ratio is not None and yield_ratio < TRUST_THRESHOLDS["yield_ratio_flag"]),
            "metrics": {
                "n_source": int(metrics["n_source"]),
                "median_tortuosity": float(metrics["median_tortuosity"]),
                "contralateral": contralateral,
                "tortuosity_ratio": ratio,
                "cap_fraction": float(metrics["cap_fraction"]),
                "rim_fraction": metrics["rim_fraction"],
                "yield_ratio": yield_ratio,
            },
            "thresholds": dict(TRUST_THRESHOLDS),
            "calibration": TRUST_CALIBRATION,
        }
    return reports


def outlier_mask(streamlines: list[np.ndarray], min_length_fraction: float = 0.5, mad_k: float = 3.0,
                 n_points: int = 16) -> np.ndarray:
    """True for streamlines kept: length >= min_length_fraction x median, and mean distance to the bundle's
    mean curve within median + mad_k x MAD (both ends flipped to one orientation first). Display cleanup only:
    the source file is never changed and the manifest records how many were dropped."""
    if len(streamlines) < 10:
        return np.ones(len(streamlines), dtype=bool)
    lengths = np.array([length_mm(s) for s in streamlines])
    keep = lengths >= min_length_fraction * np.median(lengths)
    curves = np.stack([_resample(s, n_points) for s in streamlines])
    reference = curves[int(np.argmax(keep))]
    flip = np.linalg.norm(curves - reference, axis=2).mean(1) > np.linalg.norm(curves[:, ::-1] - reference, axis=2).mean(1)
    curves[flip] = curves[flip, ::-1]
    mean_curve = curves[keep].mean(0)
    distance = np.linalg.norm(curves - mean_curve, axis=2).mean(1)
    median = np.median(distance[keep])
    mad = np.median(np.abs(distance[keep] - median)) or 1e-6
    return keep & (distance <= median + mad_k * mad)


def decimate_indices(streamline: np.ndarray, step_mm: float) -> np.ndarray:
    """Indices of source points kept roughly step_mm apart along the arc; both endpoints kept.

    A point is kept when it is the first to reach the next multiple of step_mm of
    cumulative arc length. step_mm <= 0 keeps every point.
    """
    count = len(streamline)
    if step_mm <= 0 or count <= 2:
        return np.arange(count)
    points = np.asarray(streamline, dtype=np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    crossings = np.flatnonzero(np.diff(np.floor(arc / step_mm)) > 0) + 1
    return np.unique(np.concatenate([[0], crossings, [count - 1]]))


def decimate(streamline: np.ndarray, step_mm: float) -> np.ndarray:
    """Select (never interpolate) points, so kept coordinates are bit-identical to the source."""
    return np.asarray(streamline)[decimate_indices(streamline, step_mm)]


def max_deviation_mm(source: np.ndarray, kept: np.ndarray) -> float:
    """Largest distance from any source point to the polyline through source[kept]."""
    source = np.asarray(source, dtype=np.float64)
    kept = np.asarray(kept)
    if len(kept) < 2:
        return float(np.max(np.linalg.norm(source - source[kept[0]], axis=1)))
    segment = np.clip(np.searchsorted(kept, np.arange(len(source)), side="right") - 1, 0, len(kept) - 2)
    a, b = source[kept[segment]], source[kept[segment + 1]]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", source - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return float(np.max(np.linalg.norm(source - (a + t[:, None] * ab), axis=1)))
