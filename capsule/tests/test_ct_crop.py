"""CT head crop excludes a thin table/headrest shell and never loses skull."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from capsule.ingest import convert_series, scan_series
from capsule.phantom import write_phantom
from capsule.resample import common_grid, ct_head_mask


def _physical(image: sitk.Image, zyx: np.ndarray) -> np.ndarray:
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return (zyx[:, ::-1] * np.asarray(image.GetSpacing())) @ direction.T + np.asarray(image.GetOrigin())


def _inside(grid: sitk.Image, points: np.ndarray) -> np.ndarray:
    direction = np.asarray(grid.GetDirection()).reshape(3, 3)
    index = ((points - np.asarray(grid.GetOrigin())) @ direction) / np.asarray(grid.GetSpacing())
    return np.all((index >= -0.5) & (index <= np.asarray(grid.GetSize()) - 0.5), axis=1)


@pytest.fixture(scope="module")
def phantom_ct(tmp_path_factory):
    root = tmp_path_factory.mktemp("ct_crop")
    write_phantom(root / "dicom", seed=911)
    series = next(s for s in scan_series(root / "dicom")[0] if s.modality == "CT")
    work = root / "work"
    work.mkdir()
    return sitk.ReadImage(str(convert_series(series, work)))


def _with_table(reference: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    """Pad image y and x (high end) and add a 2-voxel carbon-like shell touching the head, wider than it."""
    array = sitk.GetArrayFromImage(reference)
    head_rows = np.flatnonzero((array > -500).any(axis=(0, 2)))
    padded = np.concatenate([array, np.full((array.shape[0], 40, array.shape[2]), -1000, array.dtype)], axis=1)
    padded = np.concatenate([padded, np.full((padded.shape[0], padded.shape[1], 40), -1000, array.dtype)], axis=2)
    table = np.zeros(padded.shape, dtype=bool)
    edge = head_rows.max()
    table[:, edge:edge + 2, :] = True                     # overlaps the head edge: one connected component
    table[:, edge + 25:edge + 27, :] = True               # a second, detached table sheet
    padded[table & (padded < 200)] = 200
    image = sitk.GetImageFromArray(padded)
    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())
    image.SetOrigin(reference.GetOrigin())                # padding at the high end keeps the origin
    return image, table


def test_table_shell_excluded_and_skull_kept(phantom_ct):
    clean_plan = common_grid(phantom_ct, "CT", 1.0)
    clean_grid = clean_plan.image
    tabled, table = _with_table(phantom_ct)
    grid = common_grid(tabled, "CT", 1.0).image
    array = sitk.GetArrayFromImage(tabled)
    bone = _physical(tabled, np.argwhere((array > 300) & ~table))
    shell = _physical(tabled, np.argwhere(table))
    lost = int((~_inside(grid, bone)).sum())
    print(f"phantom grid clean={clean_grid.GetSize()} with table={grid.GetSize()}; bone voxels lost={lost} "
          f"of {len(bone)}; table voxels inside grid={_inside(grid, shell).mean():.1%}")
    assert lost == 0
    # Where the shell touches the head a sliver survives the opening: allow <= 3 mm of growth.
    assert np.all(np.abs(np.subtract(grid.GetSize(), clean_grid.GetSize())) <= 3)
    np.testing.assert_allclose(grid.GetOrigin(), clean_grid.GetOrigin(), atol=3.0)
    # Only the 10 mm margin can reach the shell that touches the head; the detached sheet is out.
    assert _inside(grid, shell).mean() < 0.5

    # Control: without the opening the attached shell joins the head component and widens the box in x.
    def x_extent(mask):
        return np.flatnonzero(mask.any(axis=(0, 1))).max()
    clean_x = x_extent(ct_head_mask(phantom_ct))
    assert x_extent(ct_head_mask(tabled)) == clean_x
    assert x_extent(ct_head_mask(tabled, opening_mm=0)) > clean_x + 30
