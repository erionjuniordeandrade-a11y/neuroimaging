"""M-anatomy surface.py — brain hull in world mm."""

from __future__ import annotations

import os
import numpy as np
import pytest
import nibabel as nib

from tractlab.surface import (
    CORTICAL_AID_LABEL,
    build_brain_hull,
    build_cortical_relief,
    pack_mesh,
    validate_surface_grid,
)
from tractlab.grid import Grid, load_grid, voxel_to_world, world_to_voxel
from tractlab.spatial_qc import (
    diagnose_streamline_containment,
    diagnose_tck_containment,
)

MASK = os.path.expanduser("~/tractlab-data/cases/local-case/nifti/mask_up.nii.gz")
needs = pytest.mark.skipif(not os.path.exists(MASK), reason="mask absent")


@needs
def test_hull_has_geometry_and_is_in_world_space():
    verts, faces = build_brain_hull(MASK)
    assert verts.shape[0] > 1000 and faces.shape[0] > 1000
    assert verts.dtype == np.dtype("<f4") and faces.dtype == np.dtype("<u4")
    # vertices must lie within the volume's world bounding box (+ a small margin)
    g = load_grid(MASK)
    corners = np.array([[0, 0, 0], [184, 184, 108], [184, 0, 0], [0, 184, 0], [0, 0, 108]])
    wc = voxel_to_world(g, corners)
    lo, hi = wc.min(0) - 5, wc.max(0) + 5
    assert (verts.min(0) >= lo).all() and (verts.max(0) <= hi).all()


@needs
def test_pack_mesh_byte_math():
    verts, faces = build_brain_hull(MASK)
    body, hdr = pack_mesh(verts, faces)
    assert hdr["vertexCount"] == verts.shape[0]
    assert len(body) == verts.shape[0] * 3 * 4 + faces.shape[0] * 3 * 4


@needs
def test_cortical_relief_is_t1_derived_and_world_aligned():
    t1 = os.path.expanduser("~/tractlab-data/cases/local-case/nifti/t1c_brain_dwi.nii.gz")
    if not os.path.exists(t1):
        pytest.skip("T1 absent")
    verts, faces = build_cortical_relief(t1, MASK)
    assert verts.shape[0] > 1000 and faces.shape[0] > 1000
    assert verts.dtype == np.dtype("<f4") and faces.dtype == np.dtype("<u4")


def _write_nifti(path, data, affine):
    nib.save(nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=float)), str(path))


def _identity_grid(shape):
    return Grid(shape=tuple(shape), affine=np.eye(4), axcodes=("R", "A", "S"))


def test_surface_grid_rejects_degenerate_and_non_affine_matrices():
    # Grid validates its affine at construction, so a singular matrix never
    # reaches the surface grid check.
    with pytest.raises(ValueError, match="singular"):
        validate_surface_grid(Grid(
            shape=(4, 4, 4),
            affine=np.diag([1.0, 1.0, 0.0, 1.0]),
            axcodes=("R", "A", "S"),
        ))

    with pytest.raises(ValueError, match="homogeneous"):
        validate_surface_grid(Grid(
            shape=(4, 4, 4),
            affine=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .1, 1]], float),
            axcodes=("R", "A", "S"),
        ))


def test_hull_preserves_single_voxel_protrusion_and_reflected_world_geometry(tmp_path):
    mask = np.zeros((10, 9, 8), np.uint8)
    mask[2:7, 2:7, 2:6] = 1
    mask[7, 4, 4] = 1  # a connected, one-voxel support protrusion
    affine = np.array([
        [-2.0, 0.0, 0.0, 100.0],
        [0.0, 3.0, 0.0, -30.0],
        [0.0, 0.0, 4.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    path = tmp_path / "mask.nii.gz"
    _write_nifti(path, mask, affine)

    verts, faces = build_brain_hull(str(path))
    vox = world_to_voxel(load_grid(str(path)), verts)
    assert faces.shape[0] > 0
    assert float(vox[:, 0].max()) >= 7.49
    np.testing.assert_allclose(voxel_to_world(load_grid(str(path)), vox), verts, atol=1e-5)


def test_surface_smoothing_is_refused_so_support_is_not_shrunk(tmp_path):
    path = tmp_path / "mask.nii.gz"
    _write_nifti(path, np.pad(np.ones((3, 3, 3), np.uint8), 2), np.eye(4))
    with pytest.raises(ValueError, match="smoothing"):
        build_brain_hull(str(path), smooth_sigma=0.5)


def test_cortical_aid_requires_exact_t1_mask_grid_and_uses_honest_label(tmp_path):
    shape = (8, 8, 8)
    mask = np.zeros(shape, np.uint8)
    mask[1:7, 1:7, 1:7] = 1
    t1 = np.zeros(shape, np.float32)
    t1[mask > 0] = 100.0
    mask_path = tmp_path / "mask.nii.gz"
    t1_path = tmp_path / "t1.nii.gz"
    _write_nifti(mask_path, mask, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 0.5
    _write_nifti(t1_path, t1, shifted)

    with pytest.raises(ValueError, match="affine mismatch"):
        build_cortical_relief(str(t1_path), str(mask_path))
    assert "aid" in CORTICAL_AID_LABEL.lower()
    assert "not" in CORTICAL_AID_LABEL.lower()


def test_containment_diagnostic_samples_original_paths_without_mask_dilation(tmp_path):
    support = np.zeros((12, 12, 12), bool)
    support[2:10, 2:10, 2:10] = True
    support[6, 6, 6] = False  # an unsupported voxel crossed between vertices
    grid = _identity_grid(support.shape)
    crossing = np.array([[2.0, 6.0, 6.0], [9.0, 6.0, 6.0]])
    inside = np.array([[2.0, 4.0, 4.0], [5.0, 4.0, 4.0]])

    diagnostic = diagnose_streamline_containment([crossing, inside], support, grid)
    assert diagnostic.streamline_count == 2
    assert diagnostic.source_vertex_count == 4
    assert diagnostic.sample_count > diagnostic.source_vertex_count
    assert diagnostic.outside_streamline_count == 1
    assert diagnostic.outside_sample_count > 0
    assert diagnostic.max_nearest_support_center_mm > 0
    assert set(diagnostic.summary()) == {
        "streamline_count", "source_vertex_count", "sample_count",
        "outside_streamline_count", "outside_sample_count", "max_nearest_support_center_mm",
    }

    support_path = tmp_path / "support.nii.gz"
    tck_path = tmp_path / "original.tck"
    _write_nifti(support_path, support.astype(np.uint8), np.eye(4))
    tractogram = nib.streamlines.Tractogram([crossing], affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(tck_path))
    from_tck = diagnose_tck_containment(str(tck_path), str(support_path))
    assert from_tck.source_vertex_count == 2
    assert from_tck.outside_streamline_count == 1


def test_containment_diagnostic_refuses_mismatched_or_nonpositive_numeric_support():
    grid = _identity_grid((4, 4, 4))
    line = [np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])]
    with pytest.raises(ValueError, match="shape"):
        diagnose_streamline_containment(line, np.ones((3, 4, 4)), grid)
    with pytest.raises(ValueError, match="empty"):
        diagnose_streamline_containment(line, np.full((4, 4, 4), -1.0), grid)


def _fake_fs_case(root, *, shift=(0.0, 0.0, 0.0)):
    """Tiny synthetic recon-all subject + MRtrix transform under ``root/work``."""
    from nibabel.freesurfer import write_geometry, write_morph_data
    subj = root / "work" / "freesurfer" / "sub-x"
    (subj / "surf").mkdir(parents=True)
    (subj / "mri").mkdir()
    orig_aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    orig_aff[:3, 3] = [10.0, -20.0, 30.0]
    nib.save(nib.MGHImage(np.zeros((8, 8, 8), np.uint8), orig_aff), str(subj / "mri" / "orig.mgz"))
    tri = np.array([[0, 1, 2]], dtype=np.int32)
    for hemi, x in (("lh", -1.0), ("rh", 1.0)):
        write_geometry(str(subj / "surf" / f"{hemi}.pial"),
                       np.array([[x, 0, 0], [x, 1, 0], [x, 0, 1]], float), tri)
        write_morph_data(str(subj / "surf" / f"{hemi}.sulc"), np.array([-5.0, 0.0, 5.0]))
    anat = root / "work" / "banks" / "anat"
    anat.mkdir(parents=True)
    xfm = np.eye(4)[:3]
    xfm[:, 3] = shift
    np.savetxt(anat / "b0_to_fs.mrtrix", xfm, header="tracking_linear_transform", comments="#")
    return subj, anat / "b0_to_fs.mrtrix", orig_aff


def test_freesurfer_cortex_maps_tkr_through_orig_then_transform_forward(tmp_path):
    from tractlab.surface import build_freesurfer_cortex, find_freesurfer_cortex
    subj, xfm, orig_aff = _fake_fs_case(tmp_path, shift=(2.0, 3.0, -4.0))
    found = find_freesurfer_cortex(str(tmp_path))
    assert found == {"subject": str(subj), "transform": str(xfm)}
    verts, faces, sulc = build_freesurfer_cortex(found["subject"], found["transform"])
    orig = nib.load(str(subj / "mri" / "orig.mgz"))
    tkr_to_scan = orig_aff @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    expected = (tkr_to_scan @ np.array([-1.0, 0, 0, 1]))[:3] + [2.0, 3.0, -4.0]
    np.testing.assert_allclose(verts[0], expected, atol=1e-5)
    assert verts.dtype == np.dtype("<f4") and faces.dtype == np.dtype("<u4")
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5]]  # rh indices offset past lh
    assert sulc.tolist() == [-5.0, 0.0, 5.0, -5.0, 0.0, 5.0]


def test_freesurfer_cortex_fails_closed(tmp_path):
    import shutil
    import time
    from tractlab.surface import find_freesurfer_cortex
    subj, xfm, _ = _fake_fs_case(tmp_path)
    (subj / "surf" / "rh.sulc").unlink()
    assert find_freesurfer_cortex(str(tmp_path)) is None  # missing curvature
    shutil.rmtree(tmp_path / "work")
    subj, xfm, _ = _fake_fs_case(tmp_path)
    shutil.copytree(subj, subj.parent / "sub-y")
    assert find_freesurfer_cortex(str(tmp_path)) is None  # two subjects: ambiguous
    shutil.rmtree(subj.parent / "sub-y")
    later = time.time() + 60
    os.utime(subj / "mri" / "orig.mgz", (later, later))
    assert find_freesurfer_cortex(str(tmp_path)) is None  # FS re-run after registration
    assert find_freesurfer_cortex(str(tmp_path / "nowhere")) is None


def test_pack_mesh_appends_sulc_and_refuses_a_mismatch():
    verts = np.zeros((3, 3), "<f4")
    faces = np.array([[0, 1, 2]], "<u4")
    body, hdr = pack_mesh(verts, faces, np.array([1, 2, 3], "<f4"))
    assert hdr["sulcCount"] == 3 and len(body) == 36 + 12 + 12
    assert np.frombuffer(body[48:], "<f4").tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        pack_mesh(verts, faces, np.array([1, 2], "<f4"))
