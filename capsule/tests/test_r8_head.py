from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from capsule import cli, masks


def test_render_head_contains_brain_and_records_preunion_fraction(monkeypatch):
    shape = (32, 32, 32)
    z, y, x = np.ogrid[:32, :32, :32]
    head = ((x - 16) / 6) ** 2 + ((y - 16) / 7) ** 2 + ((z - 16) / 7) ** 2 <= 1
    brain = ((x - 16) / 9) ** 2 + ((y - 16) / 9) ** 2 + ((z - 16) / 9) ** 2 <= 1
    values = np.zeros(shape, dtype=np.float32)
    values[head] = 50
    values[brain] = 100
    grid = sitk.Image([32, 32, 32], sitk.sitkFloat32)
    grid.SetSpacing((1, 1, 1))

    monkeypatch.setattr(masks, "mr_head", lambda *_: head.copy())
    monkeypatch.setattr(masks, "brain_mask", lambda *_: (brain.copy(), {"method": "synthetic", "seconds": 0}))
    monkeypatch.setattr(masks, "vessel_mask", lambda *_: (np.zeros(shape, dtype=bool), {"components": 0}))

    manifest = {"masks": [], "volumes": [{"id": "mr"}]}
    arrays = {}
    cli._render_masks(SimpleNamespace(brain_mask="bet"), manifest, arrays, set(), grid, 1.0,
                      [("mr", "MR", values, True)])

    head_meta = next(item for item in manifest["masks"] if item["id"] == "head")
    head_after = arrays[head_meta["blob"]].astype(bool)
    brain_after = arrays["mask_brain"].astype(bool)
    assert np.all(head_after[brain_after]), "render head must contain every brain-mask voxel"
    assert head_meta["brain_outside_head_fraction_before"] > 0.0
    assert manifest["volumes"][0]["skull_stripped_input"] is True


def test_small_head_mask_on_a_real_head_is_not_called_skull_stripped(monkeypatch):
    """Control: log-Otsu misses most of the head (as on the VS T2) but the scalp signal is there -> no flag."""
    shape = (32, 32, 32)
    z, y, x = np.ogrid[:32, :32, :32]
    brain = ((x - 16) / 8) ** 2 + ((y - 16) / 8) ** 2 + ((z - 16) / 8) ** 2 <= 1
    scalp = ((x - 16) / 14) ** 2 + ((y - 16) / 14) ** 2 + ((z - 16) / 14) ** 2 <= 1
    small_head = ((x - 16) / 5) ** 2 + ((y - 16) / 5) ** 2 + ((z - 16) / 5) ** 2 <= 1
    values = np.zeros(shape, dtype=np.float32)
    values[scalp] = 60
    values[brain] = 100
    grid = sitk.Image([32, 32, 32], sitk.sitkFloat32)
    grid.SetSpacing((1, 1, 1))

    monkeypatch.setattr(masks, "mr_head", lambda *_: small_head.copy())
    monkeypatch.setattr(masks, "brain_mask", lambda *_: (brain.copy(), {"method": "synthetic", "seconds": 0}))
    monkeypatch.setattr(masks, "vessel_mask", lambda *_: (np.zeros(shape, dtype=bool), {"components": 0}))

    manifest = {"masks": [], "volumes": [{"id": "mr"}]}
    cli._render_masks(SimpleNamespace(brain_mask="bet"), manifest, {}, set(), grid, 1.0, [("mr", "MR", values, True)])

    head_meta = next(item for item in manifest["masks"] if item["id"] == "head")
    assert head_meta["brain_outside_head_fraction_before"] > 0.5
    assert head_meta["signal_outside_brain_fraction"] > 0.05
    assert "skull_stripped_input" not in manifest["volumes"][0]
