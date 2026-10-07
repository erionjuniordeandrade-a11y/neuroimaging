"""Support-faithful anatomical context meshes in the tracking grid.

The brain hull is a rendering of the supplied support mask, not a fitted
surface and never an authority for whether a streamline is anatomically
contained. It intentionally does not smooth, fill, dilate, or discard mask
components: those display changes can hide unsupported paths. Use
``tractlab.spatial_qc`` to measure containment against the original support.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from skimage import measure

from .grid import Grid, load_grid, voxel_to_world


CORTICAL_AID_LABEL = (
    "T1 intensity-derived cortical aid — visual context only; not a cortical "
    "surface, parcellation, navigation anatomy, or streamline-containment mask."
)

_AFFINE_TOLERANCE = 1e-8
_GRID_MATCH_ATOL = 1e-3


def validate_surface_grid(grid: Grid) -> Grid:
    """Fail closed on an invalid world grid before producing a mesh.

    A non-singular affine, including reflection or shear, maps marching-cubes
    vertices exactly into world space. Unlike rectangular MPR canvases, mesh
    coordinates can represent shear without distortion, so it is supported.
    """
    shape = tuple(int(value) for value in grid.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("surface grid shape must contain three positive dimensions")
    affine = np.asarray(grid.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("surface grid affine must be finite 4x4")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], atol=_AFFINE_TOLERANCE, rtol=0.0):
        raise ValueError("surface grid affine must have homogeneous final row [0,0,0,1]")
    axes = affine[:3, :3]
    scale = float(np.prod(np.linalg.norm(axes, axis=0)))
    determinant = float(np.linalg.det(axes))
    if not np.isfinite(scale) or scale <= 0.0 or abs(determinant) <= _AFFINE_TOLERANCE * scale:
        raise ValueError("surface grid affine is singular or degenerate")
    return grid


def _load_3d(path: str, *, label: str, dtype=None) -> tuple[np.ndarray, Grid]:
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=dtype)
    if data.ndim != 3:
        raise ValueError(f"{label} must be 3-D, got shape {data.shape}")
    grid = validate_surface_grid(load_grid(path))
    if tuple(data.shape) != tuple(grid.shape):
        raise ValueError(f"{label} data shape does not match its grid")
    return data, grid


def _require_same_grid(first: Grid, second: Grid, *, first_label: str, second_label: str) -> None:
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(f"{first_label}/{second_label} shape mismatch")
    if not np.allclose(first.affine, second.affine, atol=_GRID_MATCH_ATOL, rtol=0.0):
        raise ValueError(f"{first_label}/{second_label} affine mismatch")


def _mesh_from_support(support: np.ndarray, grid: Grid, *, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Mesh exact binary support boundaries, including image-edge boundaries."""
    if not isinstance(step, int) or step != 1:
        raise ValueError("surface step must be 1 to preserve source support")
    binary = np.asarray(support, dtype=bool)
    if binary.shape != tuple(grid.shape):
        raise ValueError("support shape does not match surface grid")
    if not bool(binary.any()):
        raise ValueError("support mask is empty")
    # Padding gives marching cubes a real zero exterior for masks that reach a
    # NIfTI face. Subtracting one restores the source voxel coordinates.
    padded = np.pad(binary.astype(np.float32), 1, mode="constant")
    verts_vox, faces, _, _ = measure.marching_cubes(
        padded, level=0.5, step_size=step, allow_degenerate=False,
    )
    verts_vox -= 1.0
    verts_world = voxel_to_world(grid, verts_vox).astype("<f4")
    return verts_world, faces.astype("<u4")


def _require_no_smoothing(smooth_sigma: float) -> None:
    sigma = float(smooth_sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("surface smoothing value must be finite and non-negative")
    if sigma != 0.0:
        raise ValueError("surface smoothing is refused because it changes source support")


def build_brain_hull(mask_path: str, step: int = 1, smooth_sigma: float = 0.0):
    """Return a world-mm mesh of the exact binary support-mask envelope.

    ``smooth_sigma`` is retained only to fail loudly at old call sites. A
    smoothed or coarsened hull may appear to contain a path it does not; a
    separate containment diagnostic is the accepted way to evaluate that.
    """
    _require_no_smoothing(smooth_sigma)
    mask, grid = _load_3d(mask_path, label="support mask")
    return _mesh_from_support(mask > 0, grid, step=step)


def build_cortical_relief(
    t1_path: str,
    mask_path: str,
    *,
    threshold_pct: float = 20.0,
    step: int = 1,
    smooth_sigma: float = 0.0,
):
    """Build a grid-matched T1 intensity aid inside the supplied support mask.

    This does not infer gyri or sulci. It is deliberately an intensity-derived
    visual aid, and must never be used to clip or certify streamlines.
    """
    _require_no_smoothing(smooth_sigma)
    threshold_pct = float(threshold_pct)
    if not np.isfinite(threshold_pct) or not 0.0 <= threshold_pct <= 100.0:
        raise ValueError("threshold_pct must be finite and within [0, 100]")
    t1, t1_grid = _load_3d(t1_path, label="T1", dtype=np.float32)
    mask, mask_grid = _load_3d(mask_path, label="support mask")
    _require_same_grid(t1_grid, mask_grid, first_label="T1", second_label="support mask")
    support = mask > 0
    positive = t1[support & np.isfinite(t1) & (t1 > 0)]
    if positive.size == 0:
        raise ValueError("T1 has no finite positive values within support mask")
    threshold = float(np.percentile(positive, threshold_pct))
    tissue = support & np.isfinite(t1) & (t1 >= threshold)
    return _mesh_from_support(tissue, mask_grid, step=step)


def pack_mesh(verts: np.ndarray, faces: np.ndarray, sulc: np.ndarray | None = None) -> tuple[bytes, dict]:
    """Pack a mesh as float32le verts followed by uint32le faces (then float32le sulc)."""
    body = verts.tobytes(order="C") + faces.tobytes(order="C")
    header = {"vertexCount": int(verts.shape[0]), "faceCount": int(faces.shape[0]),
              "encoding": "float32le-verts,then,uint32le-faces"}
    if sulc is not None:
        sulc = np.asarray(sulc, dtype="<f4")
        if sulc.shape != (verts.shape[0],):
            raise ValueError("sulc needs one value per vertex")
        body += sulc.tobytes(order="C")
        header["sulcCount"] = int(sulc.shape[0])
        header["encoding"] += ",then,float32le-sulc"
    return body, header


FS_CORTEX_LABEL = (
    "FreeSurfer pial surface of this case's T1, placed in DWI space by the "
    "pipeline's bbregister transform — visual context only; not navigation "
    "anatomy or a streamline-containment mask."
)


def find_freesurfer_cortex(case_root: str) -> dict | None:
    """Locate one recon-all subject and its b0->FS transform, or ``None``.

    Fails closed: zero or several candidate subjects or transforms, a missing
    surface/curvature file, or a transform older than ``orig.mgz`` (FreeSurfer
    re-run after registration) all return ``None``.
    """
    import glob
    import os

    work = os.path.join(case_root, "work")
    subjects = [
        d for d in glob.glob(os.path.join(work, "freesurfer", "*"))
        if os.path.isdir(d) and not os.path.islink(d)
        and os.path.isfile(os.path.join(d, "surf", "lh.pial"))
    ]
    transforms = sorted(set(
        glob.glob(os.path.join(work, "anat", "b0_to_fs.mrtrix"))
        + glob.glob(os.path.join(work, "*", "anat", "b0_to_fs.mrtrix"))
    ))
    if len(subjects) != 1 or len(transforms) != 1:
        return None
    subject, transform = subjects[0], transforms[0]
    orig = os.path.join(subject, "mri", "orig.mgz")
    needed = [orig] + [
        os.path.join(subject, "surf", f"{hemi}.{kind}")
        for hemi in ("lh", "rh") for kind in ("pial", "sulc")
    ]
    if not all(os.path.isfile(p) for p in needed):
        return None
    if os.path.getmtime(transform) < os.path.getmtime(orig):
        return None
    return {"subject": subject, "transform": transform}


def _read_mrtrix_linear(path: str) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape not in ((3, 4), (4, 4)):
        raise ValueError("MRtrix linear transform must be 3x4 or 4x4")
    out = np.eye(4)
    out[:3] = rows[:3]
    if not np.all(np.isfinite(out)):
        raise ValueError("MRtrix linear transform is not finite")
    return out


def build_freesurfer_cortex(subject_dir: str, transform_path: str):
    """Both pial hemispheres in DWI world mm, plus per-vertex ``sulc``.

    FreeSurfer surfaces are in tkregister RAS; ``orig.mgz`` maps them to the
    T1's scanner RAS. The pipeline's ``b0_to_fs.mrtrix`` uses MRtrix's reverse
    convention (``mrtransform -linear T -inverse`` moves FS images onto the
    DWI), so FS scanner points reach DWI scanner space by ``T`` applied
    forward. Verified on the demo case 2026-10-03: white-surface edge strength
    on the FLIRT-resampled T1 peaks at zero shift on all three axes.
    """
    import os
    from nibabel.freesurfer import read_geometry, read_morph_data

    orig = nib.load(os.path.join(subject_dir, "mri", "orig.mgz"))
    tkr_to_dwi = _read_mrtrix_linear(transform_path) @ orig.affine @ np.linalg.inv(
        orig.header.get_vox2ras_tkr())
    verts, faces, sulc, offset = [], [], [], 0
    for hemi in ("lh", "rh"):
        v, f = read_geometry(os.path.join(subject_dir, "surf", f"{hemi}.pial"))
        s = read_morph_data(os.path.join(subject_dir, "surf", f"{hemi}.sulc"))
        if s.shape[0] != v.shape[0]:
            raise ValueError(f"{hemi}.sulc does not match {hemi}.pial")
        verts.append(v @ tkr_to_dwi[:3, :3].T + tkr_to_dwi[:3, 3])
        faces.append(f + offset)
        sulc.append(s)
        offset += v.shape[0]
    verts = np.vstack(verts).astype("<f4")
    if not np.all(np.isfinite(verts)):
        raise ValueError("pial vertices are not finite")
    return verts, np.vstack(faces).astype("<u4"), np.concatenate(sulc).astype("<f4")
