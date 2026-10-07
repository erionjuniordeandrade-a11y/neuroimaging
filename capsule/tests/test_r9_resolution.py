"""Resolution, crop and native-grid metadata contracts for case-capsule/1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from capsule.pack import read_capsule


def _write_ct_and_mask(root: Path, shape=(24, 24, 24), spacing=(5.0, 5.0, 5.0)):
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = [-60.0, -60.0, -60.0]
    ct_data = np.full(shape, 40.0, dtype=np.float32)
    ct_path = root / "ct.nii.gz"
    nib.save(nib.Nifti1Image(ct_data, affine), ct_path)
    return ct_path, affine


def _write_mask(root: Path, affine, name="lesion"):
    mask = np.zeros((24, 24, 24), dtype=np.uint8)
    mask[9:15, 10:14, 11:16] = 1
    path = root / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), path)
    return path, mask


def _run(*args):
    return subprocess.run([sys.executable, "-m", "capsule.cli", *map(str, args)],
                          text=True, capture_output=True)


def test_voxel_budget_raise_and_native_spacing_are_recorded(tmp_path):
    ct, _ = _write_ct_and_mask(tmp_path)
    output = tmp_path / "raised.capsule.html"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--spacing", "0.4", "--max-voxels", "12000", "--viewer", "v1")
    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    grid = manifest["grid"]
    volume = manifest["volumes"][0]
    assert grid["requested_spacing_mm"] == 0.4 and grid["spacing_raised"] is True
    assert grid["spacing_mm"][0] > 0.4
    assert np.prod(grid["dims"]) <= 12000
    assert volume["source_spacing_mm"] == [5.0, 5.0, 5.0]
    assert volume["resampled"] is True
    assert "raised from 0.4" in run.stdout
    assert "voxel count" in run.stdout and "raw bytes" in run.stdout


def test_crop_around_nifti_mask_covers_mask_with_margin(tmp_path):
    ct, affine = _write_ct_and_mask(tmp_path, shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    mask_path, mask = _write_mask(tmp_path, affine)
    output = tmp_path / "cropped.capsule.html"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--spacing", "1.0", "--crop-around", "lesion", "--crop-mask", f"{mask_path}:lesion",
               "--margin-mm", "4", "--viewer", "v1")
    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    crop = manifest["grid"]["crop"]
    assert crop["source"] == "mask:lesion" and crop["margin_mm"] == 4.0
    ijk = np.argwhere(mask)
    ras = nib.affines.apply_affine(affine, ijk)
    expected = np.r_[ras.min(axis=0) - 4.0, ras.max(axis=0) + 4.0]
    np.testing.assert_allclose(crop["ras_mm"], expected, atol=1e-4)
    grid = manifest["grid"]
    grid_affine = np.asarray(grid["affine_ras"])
    dims = np.asarray(grid["dims"], dtype=int)
    corners = np.array([[i, j, k, 1.0]
                        for i in (0, dims[0] - 1)
                        for j in (0, dims[1] - 1)
                        for k in (0, dims[2] - 1)]) @ grid_affine.T
    np.testing.assert_allclose(corners[:, :3].min(axis=0), expected[:3], atol=1e-4)
    np.testing.assert_allclose(corners[:, :3].max(axis=0), expected[3:], atol=1e-4)
    assert max(manifest["grid"]["dims"]) < 30  # crop box, not the 24 mm head plus default margin
    assert np.prod(manifest["grid"]["dims"]) < 27000
    assert manifest["volumes"][0]["resampled"] is False


def test_over_budget_crop_fails_with_voxel_count_and_remedy(tmp_path):
    ct, affine = _write_ct_and_mask(tmp_path, shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    mask_path, _ = _write_mask(tmp_path, affine)
    output = tmp_path / "over-budget.capsule.html"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--spacing", "0.5", "--max-voxels", "100", "--crop-around", "lesion",
               "--crop-mask", f"{mask_path}:lesion", "--margin-mm", "4", "--viewer", "v1")
    assert run.returncode != 0
    assert "crop" in run.stderr and "voxels" in run.stderr and "--spacing" in run.stderr
    assert not output.exists()


def test_uncropped_budget_too_small_for_minimum_grid_fails(tmp_path):
    ct, _ = _write_ct_and_mask(tmp_path)
    output = tmp_path / "impossible-budget.capsule.html"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--max-voxels", "1", "--viewer", "v1")
    assert run.returncode != 0
    assert "at least 8 voxels are required" in run.stderr
    assert not output.exists()


def test_crop_ras_and_crop_around_are_mutually_exclusive(tmp_path):
    ct, _ = _write_ct_and_mask(tmp_path)
    output = tmp_path / "bad.capsule.html"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--crop-ras", "-40,-40,-40,40,40,40", "--crop-around", "lesion",
               "--viewer", "v1")
    assert run.returncode != 0 and "mutually exclusive" in run.stderr
    assert not output.exists()


def test_crop_ras_is_recorded_and_limits_grid_to_intersection(tmp_path):
    ct, _ = _write_ct_and_mask(tmp_path, shape=(24, 24, 24), spacing=(1.0, 1.0, 1.0))
    output = tmp_path / "ras-cropped.capsule.html"
    box = "-55,-54,-53,-42,-43,-41"
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--spacing", "1", "--crop-ras", box, "--viewer", "v1")
    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    assert manifest["grid"]["crop"] == {
        "ras_mm": [-55.0, -54.0, -53.0, -42.0, -43.0, -41.0],
        "source": "ras", "margin_mm": 0.0}
    assert max(manifest["grid"]["dims"]) < 24


def test_oblique_ras_crop_keeps_all_grid_samples_inside_requested_box(tmp_path):
    angle = np.deg2rad(45.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.0, 1.0, 1.0])
    affine[:3, 3] = [-20.0, -20.0, -10.0]
    ct = tmp_path / "oblique.nii.gz"
    nib.save(nib.Nifti1Image(np.full((40, 40, 24), 40.0, dtype=np.float32), affine), ct)
    output = tmp_path / "oblique-crop.capsule.html"
    box = [-8.0, -8.0, -5.0, 8.0, 8.0, 5.0]
    run = _run("build-nifti", "--volume", f"{ct}:CT:TC", "--label", "Synthetic", "-o", output,
               "--spacing", "1", "--crop-ras", ",".join(map(str, box)), "--viewer", "v1")
    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    grid = manifest["grid"]
    affine_ras = np.asarray(grid["affine_ras"])
    dims = np.asarray(grid["dims"], dtype=int)
    corners = np.array([[x, y, z, 1.0]
                        for x in (0, dims[0] - 1)
                        for y in (0, dims[1] - 1)
                        for z in (0, dims[2] - 1)]) @ affine_ras.T
    assert np.all(corners[:, :3] >= np.asarray(box[:3]) - 1e-5)
    assert np.all(corners[:, :3] <= np.asarray(box[3:]) + 1e-5)
    assert grid["crop"]["ras_mm"] == box
