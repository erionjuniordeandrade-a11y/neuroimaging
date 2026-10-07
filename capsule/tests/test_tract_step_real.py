"""--tract-step-mm on the real TractLab CC0 Leipzig CST (skipped when the demo data is absent)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from capsule.pack import read_capsule, read_capsule_tracts
from capsule.tracts import decimate_indices, max_deviation_mm, outlier_mask, read_tck, subsample


LEIPZIG = Path.home() / "tractlab" / "cases" / "demo-leipzig-sub-010005"
T1 = LEIPZIG / "nifti" / "t1_brain_dwi.nii.gz"
CST = LEIPZIG / "tracts" / "bank" / "cst_l_motor_pons.tck"
pytestmark = pytest.mark.skipif(not (T1.is_file() and CST.is_file()), reason="TractLab Leipzig demo not present")


def _foreground_fraction(path: Path) -> float:
    manifest, arrays = read_capsule(path)
    volume = manifest["volumes"][0]
    image = arrays[volume["blob"]].astype(float) * volume["slope"] + volume["intercept"]
    inverse = np.linalg.inv(np.asarray(manifest["grid"]["affine_ras"]))
    points = np.concatenate(read_capsule_tracts(path)["t01"]).astype(float)
    ijk = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
    inside = np.all((ijk >= 0) & (ijk < manifest["grid"]["dims"]), axis=1)
    foreground = np.zeros(len(points), dtype=bool)
    foreground[inside] = image[ijk[inside, 2], ijk[inside, 1], ijk[inside, 0]] > volume["slope"]
    return float(foreground.mean())


def test_real_cst_step_deviation_and_foreground(tmp_path):
    source = read_tck(CST)
    keep = outlier_mask(source)
    subset = subsample([s for s, ok in zip(source, keep) if ok], 1500, seed=1)
    deviations = [max_deviation_mm(s, decimate_indices(s, 1.0)) for s in subset]
    kept = [decimate_indices(s, 1.0) for s in subset]
    assert all(k[0] == 0 and k[-1] == len(s) - 1 for k, s in zip(kept, subset))
    capsules = {}
    for step in ("0", "1.0"):
        output = tmp_path / f"cst_step_{step}.capsule.html"
        subprocess.run([sys.executable, "-m", "capsule.cli", "build-nifti", "--volume", f"{T1}:MR:RM T1",
                        "--tract", f"{CST}:CST E", "--tract-step-mm", step, "--label", "Leipzig",
                        "-o", str(output)], check=True, capture_output=True, text=True)
        capsules[step] = output
    full, stepped = (_foreground_fraction(capsules[s]) for s in ("0", "1.0"))
    manifest, _ = read_capsule(capsules["1.0"])
    tract = manifest["tracts"][0]
    assert tract["n_streamlines_source"] == len(source) and tract["n_streamlines_outliers"] == int((~keep).sum()) > 0
    decoded = read_capsule_tracts(capsules["1.0"])["t01"]
    for streamline, source, indices in zip(decoded, subset, kept):
        np.testing.assert_array_equal(streamline, source[indices])
    print(f"CST step 1.0 mm: max deviation {max(deviations):.4f} mm (median {np.median(deviations):.4f}); "
          f"points {tract['n_points']}/{tract['n_points_before_step']}; foreground full={full:.4%} "
          f"stepped={stepped:.4%} delta={100 * (stepped - full):+.3f} pp; "
          f"sizes {capsules['0'].stat().st_size}/{capsules['1.0'].stat().st_size} bytes")
    assert max(deviations) < 0.5
    assert abs(stepped - full) < 0.005
