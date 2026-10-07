"""Gantry-tilted CT (CQ500 thin series, 8 degrees): the ingest picks dcm2niix's _Tilt_1 volume and its
geometry agrees with the raw DICOM pixels at their ImagePositionPatient/ImageOrientationPatient positions.
Control: the uncorrected sheared stack, sampled through the same code path, disagrees."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from capsule.ingest import convert_series, dcm2niix_executable, scan_series

THIN = Path(__file__).resolve().parent.parent / "data" / "public" / "CQ500" / "head-ct-thin"
pytestmark = pytest.mark.skipif(not THIN.is_dir(), reason="CQ500 thin head CT not present (scripts/fetch_cq500_thin.py)")


def _raw_samples(n_slices: int = 5, stride: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """LPS points and HU of raw DICOM pixels (soft tissue/bone interior, away from edges)."""
    files = sorted(THIN.glob("*.dcm"))
    points, values = [], []
    for path in [files[int(i)] for i in np.linspace(20, len(files) - 21, n_slices)]:
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        ipp = np.asarray(ds.ImagePositionPatient, float)
        row_dir, col_dir = np.asarray(ds.ImageOrientationPatient[:3], float), np.asarray(ds.ImageOrientationPatient[3:], float)
        dr, dc = (float(v) for v in ds.PixelSpacing)  # row spacing, column spacing
        for r in range(stride, hu.shape[0] - stride, stride):
            for c in range(stride, hu.shape[1] - stride, stride):
                patch = hu[r - 2:r + 3, c - 2:c + 3]
                if patch.min() > -200 and np.ptp(patch) < 150:  # homogeneous tissue: interpolation error is small
                    points.append(ipp + c * dc * row_dir + r * dr * col_dir)
                    values.append(hu[r, c])
    return np.asarray(points), np.asarray(values)


def _sample(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    interpolator = sitk.Resample(image, image, sitk.Transform(), sitk.sitkLinear)  # float copy keeps geometry
    array = sitk.GetArrayFromImage(interpolator).astype(float)
    out = np.full(len(points), np.nan)
    size = np.asarray(image.GetSize())
    for n, point in enumerate(points):
        index = np.asarray(image.TransformPhysicalPointToContinuousIndex(tuple(map(float, point))))
        if np.all(index >= 0) and np.all(index <= size - 1):
            i, j, k = np.rint(index).astype(int)
            out[n] = array[k, j, i]
    return out


def test_tilt_corrected_volume_matches_raw_pixels(tmp_path):
    series = scan_series(THIN)[0][0]
    corrected = sitk.ReadImage(str(convert_series(series, tmp_path)))
    points, values = _raw_samples()
    assert len(points) > 500

    # Control: the raw sheared stack written by the same dcm2niix call.
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    subprocess.run([dcm2niix_executable(), "-z", "n", "-f", "raw", "-o", str(control_dir), str(THIN)],
                   capture_output=True, check=True)
    raw = sitk.ReadImage(str(control_dir / "raw.nii"))

    ok = np.abs(_sample(corrected, points) - values)
    bad = np.abs(_sample(raw, points) - values)
    print(f"tilt: {len(points)} raw pixels; |HU diff| median corrected={np.nanmedian(ok):.1f} "
          f"p90={np.nanpercentile(ok, 90):.1f}; uncorrected control median={np.nanmedian(bad):.1f} "
          f"p90={np.nanpercentile(bad, 90):.1f}")
    assert np.mean(np.isfinite(ok)) > 0.95
    assert np.nanmedian(ok) < 10 and np.nanpercentile(ok, 90) < 40
    assert np.nanpercentile(bad, 90) > 3 * np.nanpercentile(ok, 90)
    shutil.rmtree(control_dir)
