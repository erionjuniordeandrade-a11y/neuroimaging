"""Shared scene: synthetic Capsule and TractLab inputs become one layer list."""

from __future__ import annotations

import base64
import gzip
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from neuro_workbench.scene import capsule_scene, merge_scenes, tractlab_scene

from test_server import _get, _poll_ready

DIMS = (6, 5, 4)  # nx, ny, nz
AFFINE = [[1, 0, 0, -3], [0, 1, 0, -2], [0, 0, 1, -2], [0, 0, 0, 1]]


def _tck(path: Path, n: int = 3) -> None:
    curves = [np.array([[0, 0, z], [1, 1, z], [2, 0, z]], np.float32) for z in range(n)]
    nib.streamlines.save(nib.streamlines.Tractogram(curves, affine_to_rasmm=np.eye(4)), str(path))


def _blob(key: str, data: bytes) -> str:
    text = base64.b64encode(gzip.compress(data)).decode()
    return f'<script id="capsule-blob-{key}" type="application/octet-stream" data-encoding="gzip+base64">{text}</script>'


def synthetic_capsule(path: Path, tmp: Path, anatomy: bool = False) -> Path:
    nx, ny, nz = DIMS
    ct = np.arange(nx * ny * nz, dtype=np.int16).reshape((nz, ny, nx))
    lesion = np.zeros((nz, ny, nx), np.uint8)
    lesion[1:3, 1:3, 2:4] = 1
    _tck(tmp / "t.tck")
    manifest = {
        "schema": "case-capsule/1", "case": {"label": "SYNTHETIC"},
        "grid": {"dims": list(DIMS), "spacing_mm": [1, 1, 1], "affine_ras": AFFINE},
        "volumes": [{"id": "ct", "label": "CT", "kind": "CT", "dtype": "int16", "blob": "ct", "slope": 1}],
        "masks": [{"id": "lesion", "label": "Lesion", "blob": "lesion", "color": "#E4572E", "reviewed": True},
                  {"id": "head", "label": "Head", "blob": "lesion", "role": "render"}],
        "tracts": [{"id": "cst", "label": "CST", "blob": "cst", "color": "#4FC3F7", "n_streamlines": 3}],
    }
    extra = ""
    if anatomy:
        anat = np.zeros((nz, ny, nx), np.uint8)
        anat[0:2] = 1
        anat[2:4, 0:2] = 2
        manifest["masks"].append({"id": "vessels", "label": "Vessels", "blob": "lesion", "role": "render", "color": "#D32F2F"})
        manifest["anatomy"] = [{"id": "anat_mr", "blob": "anatomy_anat_mr", "method": "SynthSeg 2.0", "reviewed": False,
                                "licence": "FreeSurfer", "labels": [
                                    {"value": 1, "key": "lv_l", "name": "Ventrículo lateral esquerdo", "color": "#7851A9", "volume_ml": 0.06},
                                    {"value": 2, "key": "thal_l", "name": "Tálamo esquerdo", "color": "#00760E", "volume_ml": 0.024}]}]
        extra = _blob("anatomy_anat_mr", anat.tobytes())
    path.write_text("<!doctype html>"
                    f'<script id="capsule-manifest" type="application/json">{json.dumps(manifest)}</script>'
                    + _blob("ct", ct.tobytes()) + _blob("lesion", lesion.tobytes())
                    + _blob("cst", (tmp / "t.tck").read_bytes()) + extra)
    return path


def synthetic_manifest(case: Path) -> Path:
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    for name, data in [("b0", np.ones((4, 4, 4), np.float32)), ("lesion", np.eye(4, dtype=np.uint8)[:, :, None].repeat(4, 2))]:
        nib.save(nib.Nifti1Image(data, aff), str(case / f"{name}.nii.gz"))
    _tck(case / "bank_cst_r.tck")
    manifest = {"case_id": "SYNTHETIC-QA", "inputs": {
        "b0": {"path": "b0.nii.gz"}, "lesion": {"path": "lesion.nii.gz"},
        "fod": {"path": "missing-fod.nii.gz"},
        "bank_cst_r": {"path": "bank_cst_r.tck", "label": "CST right", "n_streamlines": 3}}}
    path = case / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_capsule_becomes_volumes_labels_and_tracts_in_scanner_space(tmp_path):
    capsule = synthetic_capsule(tmp_path / "x.capsule.html", tmp_path)
    scene, files = capsule_scene(capsule, tmp_path / "cache")
    assert [v["id"] for v in scene["volumes"]] == ["vol-ct"]
    assert scene["volumes"][0]["cal_min"] == 0.0 and scene["volumes"][0]["cal_max"] == 80.0
    assert [l["id"] for l in scene["labels"]] == ["mask-lesion"]  # render masks stay out
    assert [t["id"] for t in scene["tracts"]] == ["tract-cst"]
    ct = nib.load(str(files["vol-ct"]))
    assert ct.shape == DIMS and np.allclose(ct.affine, AFFINE)
    native = np.arange(np.prod(DIMS)).reshape(DIMS[::-1])
    assert np.asarray(ct.dataobj)[5, 4, 3] == native[3, 4, 5]
    assert np.asarray(nib.load(str(files["mask-lesion"])).dataobj).sum() == 8
    assert files["tract-cst"].read_bytes() == (tmp_path / "t.tck").read_bytes()
    again, _ = capsule_scene(capsule, tmp_path / "cache")  # second call reads the cache
    assert again == scene


def test_capsule_vessels_and_automatic_anatomy_become_layers(tmp_path):
    capsule = synthetic_capsule(tmp_path / "x.capsule.html", tmp_path, anatomy=True)
    scene, files = capsule_scene(capsule, tmp_path / "cache")
    assert [l["id"] for l in scene["labels"]] == ["mask-lesion", "mask-vessels", "anat-anat_mr"]
    anat = scene["labels"][2]
    assert anat["multi"] is True and anat["reviewed"] is False and anat["source"] == "SynthSeg 2.0"
    assert [(lb["value"], lb["name"]) for lb in anat["labels"]] == [(1, "Ventrículo lateral esquerdo"), (2, "Tálamo esquerdo")]
    data = np.asarray(nib.load(str(files["anat-anat_mr"])).dataobj)
    assert data.shape == DIMS and set(np.unique(data)) == {0, 1, 2} and (data == 2).sum() == 2 * 2 * 6


def test_tractlab_manifest_lists_images_lesion_and_banks_only(tmp_path):
    scene, files = tractlab_scene(synthetic_manifest(tmp_path))
    assert [v["id"] for v in scene["volumes"]] == ["vol-b0"]
    assert [l["id"] for l in scene["labels"]] == ["mask-lesion"]
    assert [t["label"] for t in scene["tracts"]] == ["CST right"]
    assert set(files) == {"vol-b0", "mask-lesion", "tract-bank_cst_r"}


def test_linked_scene_states_alignment_is_assumed(tmp_path):
    a = capsule_scene(synthetic_capsule(tmp_path / "x.capsule.html", tmp_path), tmp_path / "cache")
    b = tractlab_scene(synthetic_manifest(tmp_path))
    scene, files = merge_scenes([a, b], "linked")
    assert "assumed" in scene["alignment"]
    ids = [layer["id"] for g in ("volumes", "labels", "tracts") for layer in scene[g]]
    assert len(ids) == len(set(ids)) == len(files)


def test_server_serves_scene_and_only_listed_files(tmp_path):
    capsules = tmp_path / "capsules"
    capsules.mkdir()
    synthetic_capsule(capsules / "teach.capsule.html", tmp_path)
    phantom = synthetic_capsule(tmp_path / "phantom.capsule.html", tmp_path)
    proc = subprocess.Popen([sys.executable, "-m", "neuro_workbench.server", "--port", "0",
                             "--cache-dir", str(tmp_path / "cache"), "--phantom", str(phantom),
                             "--capsule-dir", str(capsules), "--manifest", str(synthetic_manifest(tmp_path))],
                            stdout=subprocess.PIPE, text=True)
    try:
        port = int(json.loads(proc.stdout.readline())["url"].rsplit(":", 1)[1].strip("/"))
        for case in json.loads(_get(port, "/api/cases")[1])["cases"]:
            data = _poll_ready(port, f"/api/case/{case['id']}/case")
            assert data["state"] == "ready", data
            status, page = _get(port, data["url"])
            assert status == 200 and b"/vendor/niivue.js" in page
            scene = json.loads(_get(port, f"/case/{case['id']}/scene.json")[1])
            for layer in scene["volumes"] + scene["labels"] + scene["tracts"]:
                assert _get(port, layer["url"])[0] == 200
            assert _get(port, f"/case/{case['id']}/file/..%2Fmanifest.json")[0] == 404
            assert _get(port, f"/case/{case['id']}/file/vol-fod")[0] == 404
        status, js = _get(port, "/vendor/niivue.js")
        assert status == 200 and len(js) > 100_000
    finally:
        proc.terminate()
        proc.wait(timeout=10)
