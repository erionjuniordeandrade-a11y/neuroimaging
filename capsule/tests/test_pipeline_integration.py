"""Real DICOM -> dcm2niix -> registered offline capsule checks."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest

from capsule.pack import read_capsule
from capsule.phantom import write_phantom


def _run(*args):
    return subprocess.run([sys.executable, "-m", "capsule.cli", *map(str, args)],
                          text=True, capture_output=True, check=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("capsule_pipeline")
    truth = write_phantom(root / "dicom", seed=3817)
    scan = _run("scan", root / "dicom")
    registered = root / "registered.capsule.html"
    control = root / "control.capsule.html"
    build = _run("build", root / "dicom", "--series", "1,2", "--label", "Caso 01", "-o", registered, "--viewer", "v1")
    _run("build", root / "dicom", "--series", "1,2", "--label", "Caso 01", "-o", control, "--no-register", "--viewer", "v1")
    return truth, scan.stdout, build.stdout, registered, control


def _volume(manifest, arrays, kind):
    entry = next(item for item in manifest["volumes"] if item["kind"] == kind)
    return entry, arrays[entry["blob"]].astype(float) * entry["slope"] + entry["intercept"]


def _ras_points(manifest, positions_zyx):
    xyz = positions_zyx[:, ::-1]
    affine = np.asarray(manifest["grid"]["affine_ras"])
    return xyz @ affine[:3, :3].T + affine[:3, 3]


def _centroid(manifest, array, low, high, near=None, radius=None):
    positions = np.argwhere((array >= low) & (array <= high))
    points = _ras_points(manifest, positions)
    if near is not None:
        points = points[np.linalg.norm(points - np.asarray(near), axis=1) < radius]
    assert len(points) > 10
    return points.mean(axis=0)


def test_scan_and_canary_deidentification(built):
    truth, scan, _, path, _ = built
    manifest, _ = read_capsule(path)
    raw = path.read_bytes()
    decoded = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    scan_bytes = scan.encode("utf-8")
    counts = {"file": 0, "manifest": 0, "scan": 0}
    for value in truth["canary_strings"]:
        token = value.encode("utf-8")
        counts["file"] += raw.count(token)
        counts["manifest"] += decoded.count(token)
        counts["scan"] += scan_bytes.count(token)
    print(f"canary counts: {counts}")
    assert counts == {"file": 0, "manifest": 0, "scan": 0}
    assert "1 | CT" in scan and "2 | MR" in scan
    assert "123×101×47" in scan
    assert manifest["case"]["anonymized"] is False
    assert manifest["case"]["deidentification"] == {
        "identifiers_removed": True, "face_removed": False, "method": None}


def test_geometry_and_hu(built):
    truth, _, _, path, _ = built
    manifest, arrays = read_capsule(path)
    _, ct = _volume(manifest, arrays, "CT")
    lesion = _centroid(manifest, ct, 55, 65, truth["lesion_center_ras_mm"], 11)
    marker = _centroid(manifest, ct, 1600, 2000)
    lesion_error = np.linalg.norm(lesion - truth["lesion_center_ras_mm"])
    assert marker[0] < 0
    assert lesion_error < 1.0
    lesion_points = _ras_points(manifest, np.argwhere(np.ones(ct.shape, dtype=bool)))
    eroded = np.linalg.norm(lesion_points - truth["lesion_center_ras_mm"], axis=1) <= 5
    lesion_mean = ct.reshape(-1)[eroded].mean()
    assert abs(lesion_mean - 60) <= 5
    assert abs(ct[0, 0, 0] - (-1000)) <= 10
    print(f"CT lesion centroid error={lesion_error:.3f} mm; marker RAS x={marker[0]:.3f} mm; lesion HU={lesion_mean:.2f}")


def test_registration_and_no_register_control(built):
    truth, _, _, path, control_path = built
    manifest, arrays = read_capsule(path)
    control_manifest, control_arrays = read_capsule(control_path)
    mr_entry, mr = _volume(manifest, arrays, "MR")
    _, ct = _volume(manifest, arrays, "CT")
    _, mr_control = _volume(control_manifest, control_arrays, "MR")
    ct_lesion = _centroid(manifest, ct, 55, 65, truth["lesion_center_ras_mm"], 11)
    mr_lesion = _centroid(manifest, mr, 200, 235, truth["lesion_center_ras_mm"], 14)
    unregistered = _centroid(control_manifest, mr_control, 200, 235, truth["lesion_center_ras_mm"], 20)
    registered_error = np.linalg.norm(mr_lesion - ct_lesion)
    control_error = np.linalg.norm(unregistered - ct_lesion)
    recovered = np.asarray(mr_entry["registration"]["moving_to_reference_ras"])
    known = np.asarray(truth["mr_to_ct_ras"])
    delta = recovered[:3, :3] @ known[:3, :3].T
    angle_error = math.degrees(math.acos(float(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    translation_error = np.linalg.norm(recovered[:3, 3] - known[:3, 3])
    print(f"registration centroid={registered_error:.3f} mm; no-register centroid={control_error:.3f} mm; "
          f"rotation error={angle_error:.3f} deg; translation error={translation_error:.3f} mm")
    print(f"recovered MR->CT RAS={np.round(recovered, 5).tolist()}; known={np.round(known, 5).tolist()}")
    assert registered_error < 1.0
    assert control_error > 3.0
    assert angle_error < 1.0
    assert translation_error < 1.0


def test_offline_text_and_blob_shape(built):
    _, _, _, path, _ = built
    text = path.read_text(encoding="utf-8")
    stripped = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    stripped = re.sub(r"<meta\b[^>]*http-equiv=[\"']Content-Security-Policy[\"'][^>]*>", "", stripped, flags=re.I)
    assert "http://" not in stripped and "https://" not in stripped
    manifest, arrays = read_capsule(path)
    expected_shape = tuple(reversed(manifest["grid"]["dims"]))
    assert set(arrays) == {volume["blob"] for volume in manifest["volumes"]}
    assert all(array.shape == expected_shape for array in arrays.values())
    print(f"capsule bytes={path.stat().st_size}")
