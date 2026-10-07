"""Curvilinear-reformatting (CR) peel: the numerical core.

Research visualization only · shows *enhancing vessels* · not navigation, not a device.

Reproduces the VMTK-Neuro "vessels on gyri" view (Wu et al., IJCARS 2019, PMID 30343394)
from ONE unstripped post-contrast T1: the head is enveloped, the envelope is peeled inward
by d mm and the image is reformatted onto that offset surface. Contract:
``handoff/ARCHITECT-BRIEF.md`` (owner-ruled 2026-08-21). Decisions implemented here:

* one depth-0 cap mesh; every depth reuses its vertices/faces (decision 3). Each vertex
  travels along the Gaussian-smoothed (``direction_sigma_mm``) gradient of a signed
  Euclidean distance field and is converged, by a directional Newton step, onto the exact
  ``d`` mm level set of the RAW field. The depth is exact; the correspondence between
  depths follows the smoothed field (the raw EDT gradient is piecewise-constant per
  boundary-voxel Voronoi cell and folds the mesh within ~5 mm). This is a design delta
  from the brief's literal "−∇EDT" wording and awaits owner ratification.
  ⚠ No self-intersection is NOT guaranteed: ``folded_faces`` is a proxy (orientation flip
  ∨ edge stretch ∨ invalid column), not a triangle–triangle test. Closing sulci or two
  sheets sliding past each other can pass it. A zero count is necessary, never sufficient,
  for the owner's visual gate.
* absolute depth stack 0–30 mm @ 1 mm plus an auto cortex-depth marker (decision 4);
* two channels per vertex × depth: ``grey`` = single trilinear sample (the honest
  reformat), ``vmax`` = max over a ±1 mm slab along the local inward direction
  (decision 5) — ``vmax`` is only ever an overlay;
* vessel overlay = global threshold fixed at the marker depth AND a thin-bright top-hat
  criterion (decision 6). The label is "enhancing vessels", never "veins": a post-gad
  T1 cannot separate artery from vein.

Everything is array-in / array-out. **Grid contract:** the volume must be RAS-canonical
(axis 2 = superior) and isotropic Euclidean (affine Gram matrix ≈ vx²·I); the CLI
canonicalises orientation and REFUSES anything else (no silent resampling) and records
the affine in the receipt — this module never sees an affine. "Upper convexity" is
therefore voxel-axis-2, which on an oblique acquisition is a few degrees off world-S.
Manifest, CLI, PLY export and the Slicer script live elsewhere; nothing here reads or
writes files.

⛔ Hair trap (lesson, do not re-learn): the head threshold must sit ABOVE hair/cushion
(≈100 on the 1.5 T SPGR; scalp 350–1300, air ≈1). A threshold of 40 swallowed ~7 mm of
hair and shifted every depth by +7 mm. Column-profile the scalp on any new sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from skimage import measure


@dataclass(frozen=True)
class PeelParams:
    head_thr: float = 200.0            # intensity; see hair trap above
    head_blur_mm: float = 0.7          # pre-threshold blur (mm, so 1.5 T and 3 T grids see the same smoothing)
    env_sigma_mm: float = 2.8          # envelope smoothing (stands in for CR's Laplacian mesh)
    base_cut_below_centroid_mm: float = 25.0  # keep a calvarial cap: drop mesh faces below centroid − this
    convexity_above_centroid_mm: float = 20.0  # vertices above centroid + this define the marker
    depths_mm: tuple[float, ...] = tuple(float(d) for d in range(0, 31))
    slab_mm: float = 1.0               # vmax = max over ±slab along inward direction
    slab_samples: int = 5
    vessel_pct: float = 97.5           # global threshold, fixed at the marker depth (decision 6)
    tophat_radius_mm: float = 1.5      # ball radius: bright structures thicker than ~2×radius (≈3 mm) lose their core; their WALLS survive
    tophat_frac: float = 0.5           # top-hat response must explain ≥ this fraction of vessel contrast
    band_below_marker_mm: float = 2.0  # superficial band (owner ruling 2026-08-24): overlay only within
    band_above_marker_mm: float = 6.0  #   [marker − below, marker + above]; nothing tints outside it
    direction_sigma_mm: float = 8.0    # smoothing of the projection direction field (8 mm: 0 convexity folds to 30 mm on the 1.5 T case; 3 mm folded at ≥20 mm)
    smooth_iters: int = 20             # Taubin smoothing of the depth-0 mesh
    newton_iters: int = 12
    newton_tol_mm: float = 0.05
    max_residual_mm: float = 0.5       # fail closed: any depth whose projection misses by more
    max_edge_stretch: float = 3.0      # a peeled face may shrink, never grow 3×: that is a bridge across a split


@dataclass(frozen=True)
class PeelResult:
    faces: np.ndarray            # (F, 3) int32, shared by every depth
    verts_vox: np.ndarray        # (D, N, 3) float32, voxel index space
    grey: np.ndarray             # (D, N) float32 — the reformat
    vmax: np.ndarray             # (D, N) float32 — ±slab max, overlay only
    vessel: np.ndarray           # (D, N) bool — "enhancing vessels" overlay
    in_brain: np.ndarray         # (D, N) bool
    valid: np.ndarray            # (D, N) bool — False where the column left the grid (values there are 0, never sampled)
    depths_mm: tuple[float, ...]
    marker_depth_mm: float       # median depth at which convexity columns first enter the brain mask
    marker_p10_p90_mm: tuple[float, float]
    band_mm: tuple[float, float]  # superficial band actually applied, absolute depths
    vessel_thr: float            # intensity threshold on vmax (fixed at marker depth)
    tophat_thr: float            # threshold on the top-hat channel (fixed at marker depth)
    depth_residual_mm: np.ndarray  # (D,) max |EDT(v) − d| after projection
    folded_faces: np.ndarray     # (D,) faces flipped vs depth-0 orientation OR stretched >max_edge_stretch OR touching an invalid column (proxy)
    brain_outside_envelope: int  # brain-mask voxels the envelope failed to contain (0 required; reported for the receipt)
    folded_faces_convexity: np.ndarray  # (D,) same, restricted to the upper convexity (the surgical view)


# --------------------------------------------------------------------------- volume stages

def head_mask(vol: np.ndarray, thr: float, blur_vox: float = 1.0) -> np.ndarray:
    """Largest connected component of the (lightly smoothed) head above ``thr``."""
    head = ndi.gaussian_filter(np.asarray(vol, np.float32), blur_vox) > thr
    head = ndi.binary_opening(head, iterations=2)
    head = ndi.binary_fill_holes(head)
    lbl, n = ndi.label(head)
    if n == 0:
        raise ValueError(f"no head voxels above threshold {thr}")
    sizes = ndi.sum(head, lbl, range(1, n + 1))
    return lbl == (1 + int(np.argmax(sizes)))


def envelope(head: np.ndarray, vx: float, sigma_mm: float) -> np.ndarray:
    """Smoothed, hole-filled head envelope (whole head — the cap is cut on the MESH)."""
    env = ndi.gaussian_filter(head.astype(np.float32), sigma_mm / vx) > 0.5
    return ndi.binary_fill_holes(env)


def signed_depth_field(env: np.ndarray, vx: float) -> np.ndarray:
    """mm inward from the envelope surface: >0 inside, <0 outside, smooth across 0."""
    inside = ndi.distance_transform_edt(env)
    outside = ndi.distance_transform_edt(~env)
    return ((inside - outside) * vx).astype(np.float32)


# --------------------------------------------------------------------------- mesh stages

def base_mesh(depth: np.ndarray, zcut: float, smooth_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Depth-0 cap mesh: marching cubes on the signed field, Taubin-smoothed, faces below
    ``zcut`` (voxel index) dropped and vertices compacted.

    Cutting the mesh rather than the envelope keeps the depth field crease-free: a flat
    cut in the mask makes a convex corner whose inward rays cross (vertices from the cut
    plane fly through the offset of the curved part), which is exactly the fold that
    ``folded_face_count`` reports. An open cap peeled like an onion has no such corner.
    """
    verts, faces, _, _ = measure.marching_cubes(depth, 0.0, step_size=1)
    verts = _taubin(verts.astype(np.float64), faces, smooth_iters)
    keep = np.all(verts[faces][:, :, 2] >= zcut, axis=1)
    if keep.sum() < 100:
        raise ValueError("fewer than 100 faces above the base cut")
    faces = faces[keep]
    used, faces = np.unique(faces, return_inverse=True)
    verts, faces = verts[used], faces.reshape(-1, 3).astype(np.int32)
    n = len(verts)
    adj = sparse.coo_matrix((np.ones(3 * len(faces)), (faces[:, [0, 1, 2]].ravel(), faces[:, [1, 2, 0]].ravel())),
                            shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp != 1:
        sizes = np.bincount(labels)
        raise ValueError(f"envelope mesh has {n_comp} components (sizes {sorted(sizes.tolist(), reverse=True)[:4]}): "
                         "the head mask is fragmented — check head_thr against the scalp column profile")
    return verts, faces


def _taubin(verts: np.ndarray, faces: np.ndarray, iters: int,
            lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Shrink-free Laplacian smoothing (Taubin 1995) on a uniform-weight adjacency."""
    n = len(verts)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    lap = sparse.diags(1.0 / deg) @ adj - sparse.identity(n)
    v = verts.copy()
    for _ in range(iters):
        v = v + lam * (lap @ v)
        v = v + mu * (lap @ v)
    return v


def _sample_vec(grad: tuple[np.ndarray, ...], pts: np.ndarray) -> np.ndarray:
    return np.stack([ndi.map_coordinates(gi, pts.T, order=1, mode="nearest") for gi in grad], axis=1)


def project_to_depth(verts: np.ndarray, depth: np.ndarray, grad: tuple[np.ndarray, ...],
                     grad_dir: tuple[np.ndarray, ...], d_mm: float, iters: int,
                     tol_mm: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Newton-project vertices onto the exact ``d_mm`` level set of ``depth``.

    Vertices travel along the SMOOTHED direction field ``grad_dir`` (unit vectors of the
    Gaussian-smoothed depth gradient) and converge on the RAW field via the directional
    derivative. The raw EDT gradient is piecewise-constant per boundary-voxel Voronoi cell,
    so following it directly scatters neighbouring vertices laterally and folds the mesh
    within ~5 mm; the smoothed field keeps neighbours on near-parallel paths while the
    depth itself stays exact. ``grad`` / ``grad_dir`` are ``np.gradient`` outputs (mm per
    voxel). Returns (verts, max residual mm over in-grid vertices, valid mask): a vertex
    that leaves the grid is a failed column — it is flagged, never clipped onto the face
    (that would fabricate a point on the level set) and never sampled.
    """
    v = np.asarray(verts, np.float64).copy()
    shape_max = np.asarray(depth.shape, np.float64) - 1.0

    def _residual(pos: np.ndarray, ok: np.ndarray) -> np.ndarray:
        val = ndi.map_coordinates(depth, np.clip(pos, 0.0, shape_max).T, order=1)
        return np.where(ok, d_mm - val, 0.0)

    valid = np.ones(len(v), bool)
    for _ in range(iters):
        valid &= ~np.any((v < 0.0) | (v > shape_max), axis=1)
        err = _residual(v, valid)
        if float(np.abs(err).max()) < tol_mm:
            break
        vv = np.clip(v, 0.0, shape_max)     # sampling positions only; v itself is never clipped
        direction = inward_directions(vv, grad_dir)
        slope = np.einsum("ij,ij->i", _sample_vec(grad, vv), direction)
        slope = np.maximum(slope, 0.2)      # never step backwards or blow up on a ridge
        v += (err / slope)[:, None] * direction
    valid &= ~np.any((v < 0.0) | (v > shape_max), axis=1)
    # residual is measured on the FINAL positions (the last update is never trusted unmeasured)
    resid = float(np.abs(_residual(v, valid)).max())
    return v, resid, valid


def inward_directions(verts: np.ndarray, grad: tuple[np.ndarray, ...]) -> np.ndarray:
    """Unit vectors (voxel space) pointing toward increasing depth at each vertex."""
    g = _sample_vec(grad, verts)
    norm = np.maximum(np.linalg.norm(g, axis=1), 1e-6)
    return g / norm[:, None]


def face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return np.cross(b - a, c - a)


def edge_lengths(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 3) edge lengths of each face."""
    p = verts[faces]
    return np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                     np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                     np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1)


def folded_faces_mask(verts: np.ndarray, faces: np.ndarray, grad: tuple[np.ndarray, ...],
                      edge0: np.ndarray, max_stretch: float, sign: float) -> np.ndarray:
    """Faces that folded (orientation opposes the depth gradient at their centroid) or
    bridged (an edge grew more than ``max_stretch`` × its depth-0 length).

    A face on a level set has its normal along ±∇depth and marching cubes orients every
    face the same way, so a sign flip or zero area means the projection folded or
    collapsed it. A peeled surface only shrinks; an edge that grows 3× spans a gap where
    the level set split (e.g. a concave waist) — the mesh is bridging, not following.
    ponytail: proxies, not triangle–triangle intersection; upgrade to a BVH
    self-intersection check if a real case ever misbehaves without tripping either.
    """
    n = face_normals(verts, faces)
    g = _sample_vec(grad, verts[faces].mean(axis=1))
    dots = np.einsum("ij,ij->i", n, g)
    stretch = edge_lengths(verts, faces) / np.maximum(edge0, 0.25)
    return (sign * dots <= 0.0) | (stretch.max(axis=1) > max_stretch)


def orientation_sign(verts: np.ndarray, faces: np.ndarray, grad: tuple[np.ndarray, ...]) -> float:
    """Global face-orientation sign vs the depth gradient, fixed ONCE at depth 0 so a
    whole-mesh inversion at a deeper level cannot re-normalise itself into 'clean'."""
    n = face_normals(verts, faces)
    g = _sample_vec(grad, verts[faces].mean(axis=1))
    return 1.0 if np.median(np.einsum("ij,ij->i", n, g)) >= 0 else -1.0


# --------------------------------------------------------------------------- sampling

def sample_channels(vol: np.ndarray, verts: np.ndarray, inward: np.ndarray, vx: float,
                    slab_mm: float, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grey, vmax, in_grid): single trilinear sample and max over ±slab along ``inward``.

    A slab sample that leaves the grid is never nearest-filled (that fabricates a
    face-voxel intensity); the column is reported ``in_grid=False`` instead.
    """
    shape_max = np.asarray(vol.shape, np.float64) - 1.0
    offsets = np.linspace(-slab_mm, slab_mm, n_samples) / vx
    positions = [verts + inward * t for t in offsets]
    in_grid = np.all([np.all((p >= 0.0) & (p <= shape_max), axis=1) for p in positions], axis=0)
    in_grid &= np.all((verts >= 0.0) & (verts <= shape_max), axis=1)
    grey = ndi.map_coordinates(vol, verts.T, order=1, mode="nearest")
    stack = [ndi.map_coordinates(vol, p.T, order=1, mode="nearest") for p in positions]
    return grey.astype(np.float32), np.max(stack, axis=0).astype(np.float32), in_grid


def tophat_volume(vol: np.ndarray, vx: float, radius_mm: float) -> np.ndarray:
    """White top-hat with a ball footprint: keeps bright structures thinner than ~2·radius."""
    r = max(1, int(round(radius_mm / vx)))
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (xx * xx + yy * yy + zz * zz) <= r * r
    return ndi.white_tophat(np.asarray(vol, np.float32), footprint=ball)


# --------------------------------------------------------------------------- orchestration

def run_peel(vol: np.ndarray, brain: np.ndarray, vx: float,
             params: PeelParams = PeelParams()) -> PeelResult:
    """Full peel on an UNSTRIPPED post-contrast T1 (voxel space, isotropic ``vx`` mm)."""
    vol = np.asarray(vol, np.float32)
    brain = np.asarray(brain) > 0
    if vol.shape != brain.shape:
        raise ValueError(f"volume {vol.shape} and brain mask {brain.shape} differ")
    if not brain.any():
        raise ValueError("brain mask is empty")

    zc = float(np.nonzero(brain)[2].mean())
    zcut = max(0.0, zc - params.base_cut_below_centroid_mm / vx)
    head = head_mask(vol, params.head_thr, params.head_blur_mm / vx)
    env = envelope(head, vx, params.env_sigma_mm)
    brain_outside = int(np.count_nonzero(brain & ~env))
    if brain_outside:
        raise ValueError(f"{brain_outside} brain-mask voxels lie outside the head envelope: "
                         "head_thr ate the scalp there (check the scalp column profile)")
    depth = signed_depth_field(env, vx)
    grad = tuple(np.asarray(g, np.float32) for g in np.gradient(depth))
    grad_dir = tuple(np.asarray(g, np.float32) for g in
                     np.gradient(ndi.gaussian_filter(depth, params.direction_sigma_mm / vx)))
    verts0, faces = base_mesh(depth, zcut, params.smooth_iters)
    verts0, _, valid0 = project_to_depth(verts0, depth, grad, grad_dir, 0.0,
                                         params.newton_iters, params.newton_tol_mm)
    if not valid0.all():
        raise ValueError(f"{int((~valid0).sum())} depth-0 vertices lie outside the grid")

    convex = verts0[:, 2] > zc + params.convexity_above_centroid_mm / vx
    convex_faces = np.all(convex[faces], axis=1)
    if convex.sum() < 100:
        raise ValueError("too few upper-convexity vertices to measure the cortex-depth marker")

    edge0 = edge_lengths(verts0, faces)
    sign0 = orientation_sign(verts0, faces, grad_dir)
    tophat = tophat_volume(vol, vx, params.tophat_radius_mm)
    depths = tuple(float(d) for d in params.depths_mm)
    if list(depths) != sorted(depths):
        raise ValueError("depths_mm must be ascending")
    n_d, n_v = len(depths), len(verts0)
    out_v = np.empty((n_d, n_v, 3), np.float32)
    grey = np.empty((n_d, n_v), np.float32)
    vmax = np.empty((n_d, n_v), np.float32)
    thmax = np.empty((n_d, n_v), np.float32)
    inb = np.empty((n_d, n_v), bool)
    valid = np.empty((n_d, n_v), bool)
    resid = np.empty(n_d, np.float32)
    folded = np.empty(n_d, np.int32)
    folded_conv = np.empty(n_d, np.int32)
    for k, d in enumerate(depths):
        v, resid[k], ok = project_to_depth(verts0, depth, grad, grad_dir, d,
                                           params.newton_iters, params.newton_tol_mm)
        if not ok[convex].all():
            raise ValueError(f"{int((~ok[convex]).sum())} upper-convexity columns left the grid at {d} mm")
        vs = np.where(ok[:, None], v, verts0)          # sampling positions for failed columns are irrelevant
        inward = inward_directions(vs, grad_dir)
        grey[k], vmax[k], in_grid = sample_channels(vol, vs, inward, vx, params.slab_mm, params.slab_samples)
        _, thmax[k], _ = sample_channels(tophat, vs, inward, vx, params.slab_mm, params.slab_samples)
        ok = ok & in_grid                               # a slab that leaves the grid invalidates the column
        if not ok[convex].all():
            raise ValueError(f"{int((~ok[convex]).sum())} upper-convexity slabs left the grid at {d} mm")
        inb[k] = ndi.map_coordinates(brain.astype(np.uint8), vs.T, order=0, mode="nearest") > 0
        grey[k, ~ok] = 0.0; vmax[k, ~ok] = 0.0; thmax[k, ~ok] = 0.0; inb[k, ~ok] = False
        valid[k] = ok
        bad = folded_faces_mask(vs, faces, grad_dir, edge0, params.max_edge_stretch, sign0) | ~np.all(ok[faces], axis=1)
        folded[k] = int(bad.sum())
        folded_conv[k] = int((bad & convex_faces).sum())
        out_v[k] = v
    if float(resid.max()) > params.max_residual_mm:
        raise ValueError(f"projection residual {resid.max():.2f} mm > {params.max_residual_mm} mm at "
                         f"depth {depths[int(resid.argmax())]}")

    # cortex-depth marker: depth at which each convexity column FIRST enters the brain mask,
    # measured along the actual projection trajectory (not nearest-Euclidean distance)
    entered = inb[:, convex]                               # (D, n_convex)
    hit = entered.any(axis=0)
    if hit.sum() < 100:
        raise ValueError("fewer than 100 convexity columns reach the brain mask within the depth stack")
    first = np.asarray(depths)[np.argmax(entered[:, hit], axis=0)]
    p10, p50, p90 = (float(x) for x in np.percentile(first, (10, 50, 90)))
    marker_k = int(np.argmin([abs(d - p50) for d in depths]))
    ref = inb[marker_k]
    if ref.sum() < 100:
        raise ValueError("marker depth has too few in-brain vertices to fix the vessel threshold")
    vessel_thr = float(np.percentile(vmax[marker_k][ref], params.vessel_pct))
    contrast = vessel_thr - float(np.median(vmax[marker_k][ref]))
    tophat_thr = params.tophat_frac * max(contrast, 0.0)
    # strict '>' so a flat image labels nothing (honest null) instead of everything
    band = (p50 - params.band_below_marker_mm, p50 + params.band_above_marker_mm)
    in_band = np.array([band[0] <= d <= band[1] for d in depths])[:, None]
    vessel = inb & (vmax > vessel_thr) & (thmax > tophat_thr) & in_band

    return PeelResult(
        faces=faces, verts_vox=out_v, grey=grey, vmax=vmax, vessel=vessel, in_brain=inb, valid=valid,
        depths_mm=depths, marker_depth_mm=p50, marker_p10_p90_mm=(p10, p90),
        band_mm=(float(band[0]), float(band[1])), vessel_thr=vessel_thr, tophat_thr=tophat_thr,
        depth_residual_mm=resid, folded_faces=folded, folded_faces_convexity=folded_conv,
        brain_outside_envelope=brain_outside,
    )
