#!/usr/bin/env python3
"""Generate the sh2amp golden fixture (Task 4, E1).

Creates a tiny synthetic SH image (8x8x8, lmax=8 -> 45 coef), a 60-direction
set from MRtrix ``dirgen`` (cartesian), and the golden amplitudes from
``sh2amp`` — the oracle that pins our numpy real-SH convention. Outputs are
converted to .nii.gz/.txt so the tests need no MRtrix at runtime.

Outputs land in tests/fixtures/generated/ (gitignored): the repo pre-commit
PHI guard refuses ALL volumetric formats with no override — synthetic or not
— so only this generator is committed and tests build the fixture lazily on
first run (deterministic seed → identical golden everywhere).

Run from the repo root:
    ~/fsl/bin/python tests/fixtures/make_sh_fixture.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

FIX = Path(__file__).resolve().parent / "generated"
MRTRIX = os.path.expanduser("~/mrtrix3/bin")
SHAPE = (8, 8, 8)
LMAX = 8
NCOEF = (LMAX + 1) * (LMAX + 2) // 2  # 45 for lmax=8 (even-l real SH)


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260817)  # fixed seed — golden is reproducible
    coeffs = rng.standard_normal(SHAPE + (NCOEF,)).astype(np.float64)
    # Smooth spatially and damp higher-order coefficients so amplitudes are
    # tame and interpolation is well-conditioned.
    scale = np.concatenate(
        [np.full((2 * l + 1,), 1.0 / (1.0 + l)) for l in range(0, LMAX + 1, 2)]
    )
    assert scale.shape[0] == NCOEF
    for c in range(NCOEF):
        coeffs[..., c] = gaussian_filter(coeffs[..., c], sigma=1.0) * scale[c]
    coeffs[..., 0] += 1.0  # positive mean lobe

    aff = np.diag([2.0, 2.0, 2.0, 1.0])  # 2 mm iso, origin at voxel (0,0,0)
    sh_nii = FIX / "fixture_sh.nii.gz"
    nib.save(nib.Nifti1Image(coeffs.astype(np.float32), aff), str(sh_nii))

    sh_mif = FIX / "fixture_sh.mif"
    dirs = FIX / "dirs60.txt"
    amp_mif = FIX / "golden_amp.mif"
    amp_nii = FIX / "golden_amp.nii.gz"

    run = lambda *argv: subprocess.run(list(argv), check=True)  # noqa: E731
    run(f"{MRTRIX}/mrconvert", str(sh_nii), str(sh_mif), "-force", "-quiet")
    run(f"{MRTRIX}/dirgen", "60", str(dirs), "-cartesian", "-force", "-quiet")
    run(f"{MRTRIX}/sh2amp", str(sh_mif), str(dirs), str(amp_mif), "-force", "-quiet")
    run(f"{MRTRIX}/mrconvert", str(amp_mif), str(amp_nii), "-force", "-quiet")
    sh_mif.unlink()
    amp_mif.unlink()
    print(f"fixture written: {sh_nii.name}, {dirs.name}, {amp_nii.name}")


if __name__ == "__main__":
    main()
