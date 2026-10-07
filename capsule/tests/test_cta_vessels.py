"""Angio-CT vessels by bone-masked subtraction: synthetic skull + contrast tube, swapped-role and gate controls."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from capsule import masks as maskops
from capsule.cli import CTA_MIN_ML, _cta_masks, cta_plausible


def _grid(shape, spacing=1.0):
    image = sitk.Image([int(s) for s in shape[::-1]], sitk.sitkUInt8)
    image.SetSpacing([spacing] * 3)
    return image


def _pair(seed=0):
    """Non-contrast and contrast CT of a 'head': brain 35 HU inside a 1200 HU skull shell, same noise level.
    The angiogram adds a 300 HU tube inside the brain and a tube running along the inner skull surface."""
    shape = (64, 64, 64)
    z, y, x = np.indices(shape)
    r = np.sqrt((z - 32) ** 2 + (y - 32) ** 2 + (x - 32) ** 2)
    rng = np.random.default_rng(seed)
    native = np.where(r < 30, 35.0, -1000.0)
    native[(r >= 26) & (r < 30)] = 1200.0
    contrast = native.copy()
    tube = ((y - 32) ** 2 + (x - 28) ** 2 <= 2 ** 2) & (r < 22)
    contrast[tube] = 300.0
    native += rng.normal(0, 8, shape)
    contrast += rng.normal(0, 8, shape)
    head = r < 30
    return shape, native, contrast, tube, head, r


def test_contrast_tube_is_found_and_skull_is_not():
    shape, native, contrast, tube, head, r = _pair()
    vessels, record = maskops.cta_vessel_mask(contrast, native, head, _grid(shape))
    assert (vessels & tube).sum() / tube.sum() > 0.7
    assert (vessels & (r >= 25)).sum() == 0  # skull and its dilated rim stay out
    assert record["components"] == 1 and record["largest_component_fraction"] == 1.0


def test_bone_edge_misregistration_is_absorbed_by_the_bone_margin():
    shape, native, contrast, tube, head, r = _pair()
    shifted = np.roll(contrast, 1, axis=2)  # one-voxel residual misregistration: skull edges no longer cancel
    vessels, _ = maskops.cta_vessel_mask(shifted, native, head, _grid(shape))
    assert (vessels & (r >= 24)).sum() == 0


def test_swapped_roles_and_identical_scans_yield_nothing():
    shape, native, contrast, tube, head, _ = _pair()
    swapped, _ = maskops.cta_vessel_mask(native, contrast, head, _grid(shape))
    same, _ = maskops.cta_vessel_mask(native, native + np.random.default_rng(1).normal(0, 8, shape), head, _grid(shape))
    assert swapped.sum() == 0 and same.sum() == 0


def test_cta_gate_bounds():
    head = 4_000_000  # ~4 L of head at 1 mm
    assert not cta_plausible(0.5, 500, head)          # a failed pairing leaves specks
    assert cta_plausible(26.0, 26_000, head)          # the local arterial - non-contrast tree
    assert not cta_plausible(270.0, 270_000, head)    # unmasked bone-edge residue, ~8 % of the head
    assert CTA_MIN_ML >= 1.0


def test_pairing_keeps_the_contrast_volume_only(capsys):
    shape, native, contrast, tube, head, _ = _pair()
    grid = _grid(shape, 1.0)
    # Tube is ~0.6 mL at 1 mm; scale the voxel to 2 mm so the synthetic tree clears the 2 mL gate.
    grid.SetSpacing([2.0] * 3)
    manifest, arrays = {"masks": []}, {}
    _cta_masks(manifest, arrays, set(), grid, 2.0, [("ct_5", contrast, head), ("ct_3", native, head)])
    assert [(m["id"], m["for_volume"], m["subtracted"]) for m in manifest["masks"]] == [("vessels", "ct_5", "ct_3")]
    assert manifest["masks"][0]["role"] == "render"
    assert "ct_3: none" in capsys.readouterr().out


def test_single_ct_gets_no_vessel_mask():
    shape, _, contrast, _, head, _ = _pair()
    manifest = {"masks": []}
    _cta_masks(manifest, {}, set(), _grid(shape), 1.0, [("ct_5", contrast, head)])
    assert manifest["masks"] == []
