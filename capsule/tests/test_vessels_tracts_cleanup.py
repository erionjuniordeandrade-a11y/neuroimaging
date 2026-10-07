"""Pack-time display cleanup: Hessian vessel mask (synthetic tube vs sheet) and tract outlier filter."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from capsule import masks as maskops
from capsule.cli import vessels_plausible
from capsule.tracts import outlier_mask


def _grid(shape, spacing=1.0):
    image = sitk.Image([int(s) for s in shape[::-1]], sitk.sitkUInt8)
    image.SetSpacing([spacing] * 3)
    return image


def _phantom():
    shape = (48, 48, 48)
    z, y, x = np.indices(shape)
    values = np.full(shape, 100.0) + np.random.default_rng(0).normal(0, 2, shape)
    brain = ((z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2) < 20 ** 2
    head = ((z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2) < 23 ** 2
    return shape, z, y, x, values, brain, head


def test_bright_tube_is_found_as_one_component():
    shape, z, y, x, values, brain, head = _phantom()
    tube = ((y - 24) ** 2 + (x - 20) ** 2 <= 1.5 ** 2) & brain
    values[tube] = 400.0
    vessels, record = maskops.vessel_mask(values, brain, head, _grid(shape))
    assert record["components"] == 1
    hit = (vessels & tube).sum() / tube.sum()
    assert hit > 0.6, hit
    assert (vessels & ~np.roll(np.roll(tube, 3, 1), 3, 2) & ~tube).sum() < 0.5 * vessels.sum()


def test_bright_sheet_is_not_a_vessel():
    shape, z, y, x, values, brain, head = _phantom()
    sheet = (np.abs(x - 24) <= 1) & brain  # plate-like, like a CSF-filled fissure
    values[sheet] = 400.0
    vessels, _ = maskops.vessel_mask(values, brain, head, _grid(shape))
    # The truncated rim of the disc is a bright edge and reads as tubular; the plate itself must not.
    interior = sheet & ((z - 24) ** 2 + (y - 24) ** 2 < 16 ** 2)
    assert (vessels & interior).sum() < 0.02 * interior.sum()


def test_flat_volume_yields_no_vessels():
    shape, z, y, x, values, brain, head = _phantom()
    vessels, record = maskops.vessel_mask(values, brain, head, _grid(shape))
    assert vessels.sum() == 0 and record["components"] == 0


def test_vessel_gate_rejects_tiny_and_csf_sized_masks():
    brain = 1_000_000
    assert not vessels_plausible(500, brain)          # a few specks: no tree worth a preset
    assert vessels_plausible(20_000, brain)           # contrast T1 range
    assert not vessels_plausible(240_000, brain)      # T2 sulcal CSF


def _bundle(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 40)[:, None]
    course = np.hstack([t * 80, np.sin(t * np.pi) * 20, np.zeros_like(t)])
    return [(course + rng.normal(0, 1.5, 3)).astype(np.float32) for _ in range(n)]


def test_outlier_mask_drops_short_fragments_and_strays_keeps_the_bundle():
    bundle = _bundle()
    reversed_ok = bundle[0][::-1].copy()                      # same course, opposite orientation
    fragment = bundle[1][:8].copy()                           # 1/5 of the length
    stray = bundle[2] + np.array([0, 0, 40], np.float32)      # right length, wrong place
    keep = outlier_mask(bundle + [reversed_ok, fragment, stray])
    assert keep[: len(bundle)].mean() > 0.95
    assert keep[len(bundle)]
    assert not keep[len(bundle) + 1] and not keep[len(bundle) + 2]


def test_outlier_mask_leaves_tiny_bundles_alone():
    assert outlier_mask(_bundle(n=5)).all()
