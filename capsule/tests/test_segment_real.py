from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from capsule.pack import read_capsule, read_capsule_tract_blobs, write_capsule
from capsule.segment import DEFAULT_NNINTERACTIVE_PYTHON, _original_template
from conftest import ROOT


NN_PYTHON = Path(os.environ.get("CASE_CAPSULE_NNINTERACTIVE_PYTHON", DEFAULT_NNINTERACTIVE_PYTHON)).expanduser()
NN_WEIGHTS = Path.home() / ".nninteractive" / "models" / "nnInteractive_v1.0"
HAS_WEIGHTS = NN_WEIGHTS.is_dir() and (NN_WEIGHTS / "LICENSE").is_file() and any(
    path.name != "LICENSE" for path in NN_WEIGHTS.iterdir()
)
REAL_ENABLED = os.environ.get("CASE_CAPSULE_REAL") == "1" and NN_PYTHON.is_file() and HAS_WEIGHTS


@pytest.mark.skipif(not REAL_ENABLED, reason="requires CASE_CAPSULE_REAL=1 and installed nnInteractive Python + weights")
def test_real_tumour_and_air_point_controls():
    candidates = list((ROOT / "out").glob("vs-seg-brain-mri.v2.*.capsule.html"))
    if not candidates:
        pytest.skip("no out/vs-seg-brain-mri.v2.*.capsule.html is present")
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    original_manifest, arrays = read_capsule(source)
    tracts = read_capsule_tract_blobs(source)
    tumour = next(mask for mask in original_manifest["masks"] if mask["id"] == "tumour")
    reference = arrays[tumour["blob"]] > 0
    kji = np.argwhere(reference)
    if not len(kji):
        raise AssertionError("dataset tumour mask is empty")
    centre_kji = kji.mean(axis=0)
    i, j, k = centre_kji[[2, 1, 0]]
    affine = np.asarray(original_manifest["grid"]["affine_ras"], dtype=np.float64)
    tumour_ras = (affine @ np.array([i, j, k, 1.0]))[:3].tolist()
    nx, ny, nz = original_manifest["grid"]["dims"]
    assert min(nx, ny, nz) > 5, "the requested air control voxel (5,5,5) must be inside the grid"
    air_ras = (affine @ np.array([5.0, 5.0, 5.0, 1.0]))[:3].tolist()
    volume_id = tumour["for_volume"]
    test_manifest = dict(original_manifest)
    test_manifest["seg_prompts"] = [
        {"id": "real-tumour-centroid", "name": "Tumour centroid control", "for_volume": volume_id,
         "color": "#E4572E", "points": [{"ras": tumour_ras, "positive": True}], "status": "pending"},
        {"id": "real-air-point", "name": "Air point control", "for_volume": volume_id,
         "color": "#4C9BE8", "points": [{"ras": air_ras, "positive": True}], "status": "pending"},
    ]
    arrays.update(tracts)

    with tempfile.TemporaryDirectory(prefix="case-capsule-real-segment-") as temp_dir:
        work = Path(temp_dir)
        template = _original_template(source, work / "template.html")
        test_input, output = work / "input.capsule.html", work / "segmented.capsule.html"
        write_capsule(template, test_input, test_manifest, arrays)
        result = subprocess.run(
            [sys.executable, "-m", "capsule.cli", "segment", str(test_input), "-o", str(output),
             "--python", str(NN_PYTHON), "--device", "mps"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        segmented_manifest, segmented_arrays = read_capsule(output)

    by_name = {item["label"]: item for item in segmented_manifest["masks"] if item.get("source") == "auto" and item.get("model")}
    prompts = {prompt["name"]: prompt for prompt in segmented_manifest["seg_prompts"]}
    tumour_seg = segmented_arrays[by_name["Tumour centroid control"]["blob"]] > 0
    dice = 2 * int(np.count_nonzero(tumour_seg & reference)) / (int(tumour_seg.sum()) + int(reference.sum()))
    assert dice >= 0.80, f"tumour centroid Dice {dice:.3f} < 0.80"
    air_prompt = prompts["Air point control"]
    air_ml = 0.0 if air_prompt["status"] == "empty" else by_name["Air point control"]["volume_ml"]
    assert air_ml < 0.05, f"air positive point produced {air_ml:.3f} mL"
