"""QuickShear-style face removal on the capsule's common grid."""

from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk

from .resample import affine_ras


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    ordered = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(ordered) < 3:
        raise ValueError("brain mask projection is too small to define a defacing plane")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    result = lower[:-1] + upper[:-1]
    if len(result) < 3:
        raise ValueError("brain mask projection is degenerate; cannot define a defacing plane")
    return result


def _projection_points(brain: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Project brain voxel centers to RAS anterior/superior, retaining the silhouette."""
    nz, ny, nx = brain.shape
    projected = brain.any(axis=2)
    padded = np.pad(projected, 1, constant_values=False)
    interior = np.ones(projected.shape, dtype=bool)
    for dz in range(3):
        for dy in range(3):
            if dz == 1 and dy == 1:
                continue
            interior &= padded[dz:dz + nz, dy:dy + ny]
    boundary = projected & ~interior
    kz, jy = np.nonzero(boundary)

    # For oblique grids, the left/right voxel axis also moves in the A-S plane.
    # Keep each occupied row's lateral endpoints; interior points lie between them.
    if abs(affine[1, 0]) + abs(affine[2, 0]) > 1e-7:
        present = brain.any(axis=2)
        first = brain.argmax(axis=2)
        last = nx - 1 - brain[:, :, ::-1].argmax(axis=2)
        kz, jy = np.nonzero(present)
        ii = np.concatenate([first[kz, jy], last[kz, jy]])
        kz = np.concatenate([kz, kz])
        jy = np.concatenate([jy, jy])
    else:
        ii = np.zeros(len(kz), dtype=int)

    anterior = affine[1, 0] * ii + affine[1, 1] * jy + affine[1, 2] * kz + affine[1, 3]
    superior = affine[2, 0] * ii + affine[2, 1] * jy + affine[2, 2] * kz + affine[2, 3]
    return np.column_stack((anterior, superior))


def face_removal_mask(brain: np.ndarray, grid: sitk.Image, buffer_mm: float = 5.0) -> np.ndarray:
    """Return voxels anterior and inferior to the brain's projected hull edge plus a buffer.

    The sagittal A-S projection's anterior-inferior hull edge supplies the plane. The
    plane is translated outward by ``buffer_mm`` so it stays away from the brain.
    Brain voxels are excluded explicitly as a final invariant.
    """
    brain = np.asarray(brain, dtype=bool)
    if brain.ndim != 3 or tuple(brain.shape) != tuple(reversed(grid.GetSize())):
        raise ValueError("brain mask must match the 3D common grid")
    if not brain.any():
        raise ValueError("brain mask is empty; cannot deface")
    if not math.isfinite(buffer_mm) or buffer_mm < 0:
        raise ValueError("deface buffer must be a finite non-negative number of millimetres")

    affine = np.asarray(affine_ras(grid), dtype=float)
    hull = _hull(_projection_points(brain, affine))
    target = np.asarray([1.0, -1.0]) / math.sqrt(2.0)  # anterior and inferior
    plane = None
    best_alignment = -math.inf
    for start, end in zip(hull, hull[1:] + hull[:1]):
        edge = np.asarray(end) - np.asarray(start)
        # The hull is counter-clockwise, so this is its outward unit normal.
        normal = np.asarray([edge[1], -edge[0]], dtype=float)
        length = float(np.linalg.norm(normal))
        if length == 0:
            continue
        normal /= length
        if normal[0] <= 0 or normal[1] >= 0:
            continue
        alignment = float(np.dot(normal, target))
        if alignment > best_alignment:
            best_alignment = alignment
            plane = normal, float(np.dot(normal, start))
    if plane is None:
        # Degenerate digital hulls may have only horizontal/vertical edges. Use
        # the outward support line in the requested direction as the limiting edge.
        normal = target
        support = max(float(np.dot(normal, point)) for point in hull)
    else:
        normal, support = plane
    threshold = support + float(buffer_mm)

    nz, ny, nx = brain.shape
    ci = normal[0] * affine[1, 0] + normal[1] * affine[2, 0]
    cj = normal[0] * affine[1, 1] + normal[1] * affine[2, 1]
    ck = normal[0] * affine[1, 2] + normal[1] * affine[2, 2]
    origin = normal[0] * affine[1, 3] + normal[1] * affine[2, 3]
    x = np.arange(nx, dtype=float)[None, :]
    y = np.arange(ny, dtype=float)[:, None]
    removal = np.empty(brain.shape, dtype=bool)
    for k in range(nz):
        removal[k] = ci * x + cj * y + ck * k + origin > threshold
    removal &= ~brain
    return removal


def apply_deface(values: np.ndarray, removal: np.ndarray, kind: str) -> np.ndarray:
    """Copy scalar values with the removed face set to the modality's air value."""
    values = np.asarray(values)
    if values.shape != removal.shape:
        raise ValueError("deface mask must match the volume grid")
    result = values.copy()
    air = float(np.min(values)) if kind.upper() == "MR" else -1000.0 if kind.upper() == "CT" else None
    if air is None:
        raise ValueError("deface supports MR and CT volumes")
    result[removal] = air
    return result
