"""Synthetic 0.5 mm CTA controls for vessels that touch skull-base bone."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from capsule import masks
from capsule.masks import cta_vessel_mask, cta_vessel_mask_r7
from capsule.cli import _cta_masks

SPACING = 0.5


def _grid(shape):
    image = sitk.Image([int(size) for size in shape[::-1]], sitk.sitkUInt8)
    image.SetSpacing([SPACING] * 3)
    return image


def _half_voxel_shift(array, interpolation):
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing([SPACING] * 3)
    transform = sitk.TranslationTransform(3, (-SPACING / 2.0, 0.0, 0.0))
    shifted = sitk.Resample(image, image, transform, interpolation, 40.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(shifted)


def _skull_base_pair():
    """Two long contrast tubes; the canal wall alternately touches the first tube."""
    shape = (160, 96, 112)  # z, y, x; 80 x 48 x 56 mm at 0.5 mm
    z, y, x = np.indices(shape)
    native = np.full(shape, 40.0, dtype=np.float32)
    canal_x, canal_y = 35, 48
    canal_r = np.hypot(x - canal_x, y - canal_y)
    tube_canal = (canal_r <= 2.5) & (z >= 8) & (z <= 151)

    # The canal wall touches the lumen at a 5 mm constriction. Wider segments
    # on either side are long enough to survive r7's component-size filter.
    narrow = ((z >= 76) & (z < 86) & (canal_r >= 3.0) & (canal_r <= 5.0))
    wide_region = ((z >= 8) & (z < 76)) | ((z >= 86) & (z <= 151))
    wide = wide_region & (canal_r >= 7.0) & (canal_r <= 8.0)
    native[narrow | wide] = 1000.0

    open_x, open_y = 78, 48
    open_r = np.hypot(x - open_x, y - open_y)
    tube_open = (open_r <= 5.5) & (z >= 8) & (z <= 151)
    contrast = native.copy()
    contrast[tube_canal | tube_open] = 300.0

    # Resample the contrast phase half a voxel (0.25 mm) off the native lattice.
    shifted_contrast = _half_voxel_shift(contrast, sitk.sitkLinear)
    shifted_canal = _half_voxel_shift(tube_canal, sitk.sitkLinear) > 0.5
    shifted_open = _half_voxel_shift(tube_open, sitk.sitkLinear) > 0.5
    head = np.ones(shape, dtype=bool)
    return native, shifted_contrast, head, shifted_canal, shifted_open


def _sinus_pair():
    shape = (128, 96, 112)
    z, y, x = np.indices(shape)
    native = np.full(shape, 40.0, dtype=np.float32)
    # A thin inner-table plate touches one side of a sinus-like tube.
    native[(y <= 47) & (z >= 8) & (z <= 119)] = 1000.0
    sinus = (np.abs(y - 51) <= 2) & (np.abs(x - 55) <= 2) & (z >= 8) & (z <= 119)
    contrast = native.copy()
    contrast[sinus] = 300.0
    return native, np.roll(contrast, 1, axis=1), np.ones(shape, bool), np.roll(sinus, 1, axis=1)


def _components(mask, grid):
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(
        sitk.GetImageFromArray(mask.astype(np.uint8))))
    sizes = np.bincount(labels.ravel())[1:]
    return int(len(sizes)), float(sizes.max() / mask.sum()) if mask.any() else 0.0


def _report(case, old, new, native, grid):
    voxel_ml = SPACING ** 3 / 1000.0
    bone = native > 200
    distance = sitk.GetArrayFromImage(sitk.DanielssonDistanceMap(
        sitk.GetImageFromArray(bone.astype(np.uint8)), inputIsBinary=True, squaredDistance=False))
    near = lambda mask: float(mask[distance * SPACING <= 2.0].sum() * voxel_ml)
    old_near, new_near = near(old), near(new)
    print(f"{case}: r7={old.sum()} vox/{old.sum()*voxel_ml:.3f} mL/{_components(old, grid)[0]} cc/"
          f"{old_near:.3f} mL <=2mm bone; new={new.sum()} vox/{new.sum()*voxel_ml:.3f} mL/"
          f"{_components(new, grid)[0]} cc/{new_near:.3f} mL <=2mm bone")
    return {"old_near_ml": old_near, "new_near_ml": new_near,
            "old_components": _components(old, grid)[0], "new_components": _components(new, grid)[0]}


def test_canal_and_open_tubes_survive_half_voxel_shift_and_r7_is_fragmented():
    native, contrast, head, canal, open_tube = _skull_base_pair()
    grid = _grid(native.shape)
    old, _ = cta_vessel_mask_r7(contrast, native, head, grid)
    new, record = cta_vessel_mask(contrast, native, head, grid)
    stats = _report("canal+open", old, new, native, grid)

    kept = new & canal
    old_kept = old & canal
    assert kept.sum() / canal.sum() >= 0.90
    assert _components(kept, grid)[0] == 1
    assert _components(old_kept, grid)[0] > 1
    assert (new & open_tube).sum() / open_tube.sum() >= 0.90
    assert record["components"] >= 2
    assert "minmax" in record["method"]
    print(f"canal voxels: r7 {old_kept.sum()}/{canal.sum()} in { _components(old_kept, grid)[0]} cc; "
          f"new {kept.sum()}/{canal.sum()} in {_components(kept, grid)[0]} cc")
    old_open, new_open = old & open_tube, new & open_tube
    print(f"open-tube voxels: r7 {old_open.sum()}/{open_tube.sum()} in {_components(old_open, grid)[0]} cc; "
          f"new {new_open.sum()}/{open_tube.sum()} in {_components(new_open, grid)[0]} cc")

    outside_tubes = new & ~(canal | open_tube)
    bone_distance = sitk.GetArrayFromImage(sitk.DanielssonDistanceMap(
        sitk.GetImageFromArray((native > 200).astype(np.uint8)), inputIsBinary=True,
        squaredDistance=False)) * SPACING
    bone_edge_false_positive = outside_tubes & (bone_distance <= 1.0)
    print(f"bone-edge false positives: {bone_edge_false_positive.sum()} vox/"
          f"{bone_edge_false_positive.sum()*SPACING**3/1000:.3f} mL; "
          f"ceiling 5% of {((canal | open_tube).sum()*SPACING**3/1000):.3f} mL tube volume")
    # The 5% ceiling bounds connected bone-edge residue while tolerating a few
    # partial-volume voxels from the half-voxel phase shift.
    assert bone_edge_false_positive.sum() * SPACING ** 3 < 0.05 * (canal | open_tube).sum() * SPACING ** 3
    assert record["ml_within_2mm_of_bone"] > 0
    assert stats["new_near_ml"] > stats["old_near_ml"]
    assert abs(record["ml_within_2mm_of_bone"] - stats["new_near_ml"]) <= 0.025


def test_swapped_roles_and_identical_scans_yield_nothing():
    native, contrast, head, _, _ = _skull_base_pair()
    grid = _grid(native.shape)
    old_swapped, _ = cta_vessel_mask_r7(native, contrast, head, grid)
    swapped, _ = cta_vessel_mask(native, contrast, head, grid)
    old_same, _ = cta_vessel_mask_r7(native, native.copy(), head, grid)
    same, _ = cta_vessel_mask(native, native.copy(), head, grid)
    swap_stats = _report("swap control", old_swapped, swapped, native, grid)
    same_stats = _report("identical control", old_same, same, native, grid)
    assert swapped.sum() == 0 and same.sum() == 0
    assert old_swapped.sum() == 0 and old_same.sum() == 0
    assert swap_stats["old_near_ml"] == swap_stats["new_near_ml"] == 0.0
    assert same_stats["old_near_ml"] == same_stats["new_near_ml"] == 0.0


def test_inner_table_sinus_remains_continuous():
    native, contrast, head, sinus = _sinus_pair()
    grid = _grid(native.shape)
    old, _ = cta_vessel_mask_r7(contrast, native, head, grid)
    new, record = cta_vessel_mask(contrast, native, head, grid)
    stats = _report("sinus", old, new, native, grid)
    kept = new & sinus
    assert kept.sum() / sinus.sum() >= 0.90
    assert _components(kept, grid)[0] == 1
    assert record["ml_within_2mm_of_bone"] > 0
    print(f"sinus voxels: r7 {(old & sinus).sum()}/{sinus.sum()} in {_components(old & sinus, grid)[0]} cc; "
          f"new {kept.sum()}/{sinus.sum()} in {_components(kept, grid)[0]} cc")
    assert stats["new_near_ml"] > stats["old_near_ml"]
    assert abs(record["ml_within_2mm_of_bone"] - stats["new_near_ml"]) <= 0.025


def test_cta_manifest_records_bone_adjacent_volume_metrics(capsys):
    native, contrast, head, _, _ = _skull_base_pair()
    grid = _grid(native.shape)
    manifest, arrays = {"masks": []}, {}
    _cta_masks(manifest, arrays, set(), grid, SPACING,
               [("angio", contrast, head), ("native", native, head)])
    assert len(manifest["masks"]) == 1
    mask = manifest["masks"][0]
    assert mask["volume_ml"] > 2.0
    assert mask["components"] >= 2
    assert "largest_component_fraction" in mask and "ml_within_2mm_of_bone" in mask
    assert mask["ml_within_2mm_of_bone"] > 0
    assert "within 2 mm of bone" in capsys.readouterr().out


def test_whole_voxel_shift_and_kernel_mismatch_give_no_swap_tree():
    """Regression: a real CTA pair (different kernels, sub-mm residual motion) made a 113 mL tree from the
    swapped phases with the seeded-hysteresis method. Bone edges moved a whole voxel, one phase sharper,
    plus noise, must still give nothing when swapped."""
    native, contrast, head, canal, _ = _skull_base_pair()
    grid = _grid(native.shape)
    rng = np.random.default_rng(9)
    def blur(values, sigma):
        image = sitk.GetImageFromArray(values.astype(np.float32)); image.SetSpacing([SPACING] * 3)
        return sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(image, sigma))
    contrast = blur(np.roll(contrast, 1, axis=2), 0.2) + rng.normal(0, 12, native.shape).astype(np.float32)
    native = blur(native, 0.6) + rng.normal(0, 12, native.shape).astype(np.float32)
    swapped, _ = cta_vessel_mask(native, contrast, head, grid)
    forward, _ = cta_vessel_mask(contrast, native, head, grid)
    print(f"kernel+shift: swap {swapped.sum()} vox, forward {forward.sum()} vox, canal {canal.sum()} vox")
    assert swapped.sum() < 0.01 * canal.sum()
    kept = forward & np.roll(canal, 1, axis=2)
    assert kept.sum() / canal.sum() >= 0.80


def test_component_count_uses_the_same_connectivity_as_the_size_filter() -> None:
    """The size filter joins voxels by 26-connectivity; the recorded count must too (else a tree reads as fragments)."""
    grid = sitk.Image([12, 12, 12], sitk.sitkUInt8)
    mask = np.zeros((12, 12, 12), bool)
    mask[2:5, 2:5, 2:5] = True
    mask[5:8, 5:8, 5:8] = True  # touches the first cube only at a corner
    stats = masks._cta_statistics(mask, np.zeros(mask.shape, np.float32), grid)
    assert stats["components"] == 1 and stats["largest_component_fraction"] == 1.0, stats
