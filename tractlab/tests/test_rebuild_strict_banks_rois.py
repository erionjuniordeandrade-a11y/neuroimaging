"""build_rois must derive the hemisphere cortex masks a freshly ingested case lacks.

A newly imported case has only nifti/aparc_dwi.nii.gz; the bank stage used to
assume tracts/roi/hemi_{l,r}_cortex.nii.gz already existed and failed with
MISSING ROI. Synthetic labels only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rebuild_strict_banks.py"


def _load():
    spec = importlib.util.spec_from_file_location("_rsb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(tmp_path: Path) -> tuple[Path, np.ndarray]:
    lab = np.zeros((12, 12, 12), np.int32)
    lab[1:3, 1:3, 1:3] = 1024  # lh precentral
    lab[1:3, 4:6, 1:3] = 1001  # lh first aparc label
    lab[1:3, 7:9, 1:3] = 1035  # lh last aparc label
    lab[9:11, 1:3, 1:3] = 2024  # rh precentral
    lab[9:11, 4:6, 1:3] = 2035
    lab[5:7, 5:7, 5:7] = 16  # brainstem
    lab[5:7, 1:3, 8:10] = 1000  # lh unknown: not cortex
    lab[5:7, 8:10, 8:10] = 3  # aseg cortex label: not in the reference masks
    (tmp_path / "nifti").mkdir()
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp_path / "nifti/aparc_dwi.nii.gz"))
    return tmp_path, lab


def _mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def test_fresh_case_gets_hemisphere_cortex_from_aparc(tmp_path):
    case, lab = _case(tmp_path)
    rois = _load().build_rois(case)
    for side, lo in (("l", 1001), ("r", 2001)):
        path = rois[f"hemi_{side}"]
        assert path.is_file(), f"hemi_{side} not written"
        expected = (lab >= lo) & (lab <= lo + 34)
        assert np.array_equal(_mask(path), expected)


def test_existing_hemisphere_masks_are_kept(tmp_path):
    case, lab = _case(tmp_path)
    roi = case / "tracts/roi"
    roi.mkdir(parents=True)
    curated = np.zeros(lab.shape, np.uint8)
    curated[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(curated, np.eye(4)), str(roi / "hemi_l_cortex.nii.gz"))
    rois = _load().build_rois(case)
    assert np.array_equal(_mask(rois["hemi_l"]), curated > 0)
    assert rois["hemi_r"].is_file()
