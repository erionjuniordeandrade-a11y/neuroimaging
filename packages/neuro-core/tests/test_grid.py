import nibabel as nib
import numpy as np
import pytest

from neuro_core.grid import (Grid, LPS_TO_RAS, grid_id, in_bounds, load_grid, lps_to_ras_affine,
                             voxel_to_world, world_to_voxel)


def _grid() -> Grid:
    affine = np.array([[-2.0, 0, 0, 90], [0, -2.0, 0, 126], [0, 0, 2.0, -72], [0, 0, 0, 1]])
    return Grid(shape=(10, 12, 8), affine=affine, axcodes=tuple(nib.aff2axcodes(affine)))


def test_voxel_world_round_trip():
    grid = _grid()
    ijk = np.array([[0, 0, 0], [3.5, 2.0, 7.0]])
    assert np.allclose(world_to_voxel(grid, voxel_to_world(grid, ijk)), ijk)
    assert np.allclose(voxel_to_world(grid, [1, 0, 0]), [88, 126, -72])


def test_in_bounds_never_clamps():
    grid = _grid()
    assert in_bounds(grid, np.array([[0, 0, 0], [9, 11, 7], [10, 0, 0], [-1, 0, 0]])).tolist() == [
        True, True, False, False]


def test_grid_id_ignores_sub_micron_noise_only():
    grid = _grid()
    noisy = Grid(grid.shape, grid.affine + 1e-9, grid.axcodes)
    shifted = grid.affine.copy()
    shifted[0, 3] += 0.01
    moved = Grid(grid.shape, shifted, grid.axcodes)
    assert grid_id(grid) == grid_id(noisy)
    assert grid_id(grid) != grid_id(moved)


def test_load_grid_takes_spatial_shape(tmp_path):
    grid = _grid()
    path = tmp_path / "v.nii"
    nib.save(nib.Nifti1Image(np.zeros((10, 12, 8, 3), np.float32), grid.affine), path)
    loaded = load_grid(str(path))
    assert loaded.shape == (10, 12, 8)
    assert loaded.axcodes == ("L", "P", "S")
    assert np.allclose(loaded.affine, grid.affine)


def test_lps_to_ras_affine_flips_x_and_y():
    affine = lps_to_ras_affine((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.5, 0.5, 2.0), (10, 20, 30))
    expected = np.array([[-0.5, 0, 0, -10], [0, -0.5, 0, -20], [0, 0, 2.0, 30], [0, 0, 0, 1]])
    assert np.allclose(affine, expected)
    assert np.allclose(LPS_TO_RAS @ LPS_TO_RAS, np.eye(4))


def test_lps_to_ras_affine_rejects_bad_direction():
    with pytest.raises(ValueError):
        lps_to_ras_affine((1, 0, 0), (1, 1, 1), (0, 0, 0))
