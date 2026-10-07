"""E1 — real-SH amplitude evaluation, pinned to the sh2amp golden fixture.

The golden test is the convention oracle: if our m-sign/order differs from
MRtrix, it fails. Tolerance 1e-4 is a contract — never loosen it.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.fidelity import amplitude_along, lmax_from_ncoef, real_sh_basis, unit_dirs

FIX = Path(__file__).resolve().parent / "fixtures" / "generated"

# Lazy build: the PHI pre-commit guard forbids committing volumetric files,
# so the golden fixture regenerates deterministically (fixed seed) on first
# run. Requires MRtrix (~/mrtrix3/bin) ONCE per checkout — fail loud, no skip:
# a skipped golden gate is a gate that cannot fail.
if not (FIX / "golden_amp.nii.gz").is_file():
    import subprocess
    import sys

    gen = Path(__file__).resolve().parent / "fixtures" / "make_sh_fixture.py"
    proc = subprocess.run([sys.executable, str(gen)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"golden fixture generation failed (MRtrix required once): {proc.stderr}"
        )


def test_lmax_from_ncoef():
    assert lmax_from_ncoef(45) == 8 and lmax_from_ncoef(28) == 6
    assert lmax_from_ncoef(1) == 0 and lmax_from_ncoef(6) == 2


def test_lmax_from_ncoef_rejects_non_sh_sizes():
    with pytest.raises(ValueError, match="44"):
        lmax_from_ncoef(44)


def test_basis_shape():
    dirs = unit_dirs(np.loadtxt(FIX / "dirs60.txt"))
    b = real_sh_basis(dirs, lmax=8)
    assert b.shape == (60, 45)


def test_numpy_matches_sh2amp_golden():
    sh = nib.load(str(FIX / "fixture_sh.nii.gz"))
    amp = nib.load(str(FIX / "golden_amp.nii.gz"))
    dirs = unit_dirs(np.loadtxt(FIX / "dirs60.txt"))
    coeffs = sh.get_fdata()
    golden = amp.get_fdata()
    for v in [(2, 3, 4), (5, 5, 5), (1, 6, 2), (0, 0, 0), (7, 7, 7)]:
        pts = np.repeat(
            [nib.affines.apply_affine(sh.affine, v)], len(dirs), axis=0
        )
        mine = amplitude_along(coeffs, sh.affine, pts, dirs)
        assert mine.shape == (len(dirs),)
        assert np.allclose(mine, golden[v], atol=1e-4), (
            v, float(np.abs(mine - golden[v]).max())
        )


def test_amplitude_between_voxels_is_interpolated():
    sh = nib.load(str(FIX / "fixture_sh.nii.gz"))
    dirs = unit_dirs(np.loadtxt(FIX / "dirs60.txt"))[:1]
    coeffs = sh.get_fdata()
    a = nib.affines.apply_affine(sh.affine, (2, 3, 4))
    b = nib.affines.apply_affine(sh.affine, (3, 3, 4))
    mid = (np.asarray(a) + np.asarray(b)) / 2.0
    amp_a = amplitude_along(coeffs, sh.affine, np.array([a]), dirs)[0]
    amp_b = amplitude_along(coeffs, sh.affine, np.array([b]), dirs)[0]
    amp_mid = amplitude_along(coeffs, sh.affine, np.array([mid]), dirs)[0]
    assert np.isclose(amp_mid, (amp_a + amp_b) / 2.0, atol=1e-6)
