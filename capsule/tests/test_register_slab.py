"""A partial-coverage slab in the same frame of reference registers to ~identity."""

import numpy as np
import pytest
import SimpleITK as sitk

from capsule.register import register


def _head(shape=(100, 96, 96)):
    z, y, x = np.indices(shape, dtype=float)
    c = np.array(shape) / 2.0
    r = ((z - c[0]) / 34) ** 2 + ((y - c[1]) / 40) ** 2 + ((x - c[2]) / 36) ** 2
    img = np.where(r < 1, 300.0, 0.0) + np.where(r < 0.8, 200.0, 0.0)
    img += 400.0 * (((z - 30) ** 2 + (y - 40) ** 2 + (x - 58) ** 2) < 64)
    img += 250.0 * (((z - 26) ** 2 + (y - 60) ** 2 + (x - 36) ** 2) < 36)
    return img.astype(np.float32)


def test_offcentre_slab_registers_to_identity():
    head = _head()
    fixed = sitk.GetImageFromArray(head)
    fixed.SetSpacing((2.0, 2.0, 2.0))
    # Slab covering z 14..40 only, far from the head centre, same world geometry.
    # Geometry centring alone throws "All samples map outside" here (the
    # pre-fix failure on a real IAC T2 slab); the identity start recovers it.
    moving = sitk.GetImageFromArray(head[14:40])
    moving.SetSpacing((2.0, 2.0, 2.0))
    moving.SetOrigin(fixed.TransformIndexToPhysicalPoint((0, 0, 14)))

    transform, _metric, meta = register(fixed, moving)
    matrix = np.asarray(meta["moving_to_fixed_ras"])
    assert np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1) / 2, -1, 1))) < 1.0
    assert np.linalg.norm(matrix[:3, 3]) < 1.0


def test_too_thin_slab_fails_loudly_with_the_escape_named():
    head = _head()
    fixed = sitk.GetImageFromArray(head)
    fixed.SetSpacing((2.0, 2.0, 2.0))
    moving = sitk.GetImageFromArray(head[6:16])
    moving.SetSpacing((2.0, 2.0, 2.0))
    moving.SetOrigin(fixed.TransformIndexToPhysicalPoint((0, 0, 6)))
    with pytest.raises(ValueError, match="--no-register"):
        register(fixed, moving)
