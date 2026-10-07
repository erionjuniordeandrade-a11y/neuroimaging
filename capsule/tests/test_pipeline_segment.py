from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from capsule.pack import read_capsule, write_capsule
from capsule.pack import read_capsule_tract_blobs
from capsule.segment import segment_capsule


AFFINE = np.array([
    [0.0, -2.0, 0.5, 30.0],
    [1.5, 0.2, 0.0, -10.0],
    [0.0, 0.0, -3.0, 5.0],
    [0.0, 0.0, 0.0, 1.0],
])
DIMS = [9, 7, 5]  # i, j, k; blobs are stored k, j, i.


def _ras(ijk: tuple[int, int, int]) -> list[float]:
    return (AFFINE @ np.array([*ijk, 1.0]))[:3].tolist()


def _capsule(tmp_path: Path, prompts: list[dict]) -> Path:
    template = tmp_path / "v2-template.html"
    template.write_text("<!doctype html><title>viewer-v2-test</title><!--CAPSULE_PAYLOAD-->", encoding="utf-8")
    source = tmp_path / "input.capsule.html"
    volume = np.arange(np.prod(DIMS), dtype=np.int16).reshape(tuple(reversed(DIMS)))
    tract = np.frombuffer(b"original tract bytes", dtype=np.uint8).copy()
    manifest = {
        "schema": "case-capsule/1", "version": 2, "generator": "case-capsule 0.1.0",
        "case": {"label": "test"},
        "grid": {"dims": DIMS, "spacing_mm": [1, 1, 1], "affine_ras": AFFINE.tolist()},
        "volumes": [{"id": "mr", "blob": "mr", "kind": "MR", "dtype": "int16", "slope": 1, "intercept": 0}],
        "masks": [], "anatomy": [],
        "tracts": [{"id": "t1", "blob": "tract_t1", "format": "tck"}],
        "annotations": [], "tour": [], "seg_prompts": prompts,
    }
    write_capsule(template, source, manifest, {"mr": volume, "tract_t1": tract})
    return source


def _template(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    manifest = re.search(r'<script id="capsule-manifest" type="application/json">.*?</script>', html, flags=re.S)
    assert manifest
    blobs = list(re.finditer(r'<script id="capsule-blob-[^"<>]+"[^>]*>.*?</script>', html, flags=re.S))
    payload_end = max([manifest.end(), *(blob.end() for blob in blobs)])
    return html[:manifest.start()] + "<!--CAPSULE_PAYLOAD-->" + html[payload_end:]


def _prompt(name="Nervo facial", ijk=(7, 2, 0), points=None):
    return {
        "id": "prompt-1", "name": name, "for_volume": "mr", "color": "#43B581",
        "points": points or [{"ras": _ras(ijk), "positive": True}, {"ras": _ras((2, 4, 0)), "positive": False}],
        "status": "pending",
    }


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-nninteractive-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "import numpy as np\n"
        "args = sys.argv[2:]\n"
        "opts = dict(zip(args[::2], args[1::2]))\n"
        "spec = json.loads(pathlib.Path(opts['--request']).read_text())\n"
        "with np.load(opts['--input'], allow_pickle=False) as data:\n"
        "    results = {}\n"
        "    for n, prompt in enumerate(spec['prompts']):\n"
        "        image = data[prompt['image_key']]\n"
        "        mask = np.zeros(image.shape, dtype=np.uint8)\n"
        "        if prompt['name'] != 'empty result':\n"
        "            for point in prompt['points_kji']:\n"
        "                if not point['positive']:\n"
        "                    continue\n"
        "                k, j, i = point['kji']\n"
        "                for kk in range(max(0, k-1), min(mask.shape[0], k+2)):\n"
        "                    for jj in range(max(0, j-1), min(mask.shape[1], j+2)):\n"
        "                        for ii in range(max(0, i-1), min(mask.shape[2], i+2)):\n"
        "                            mask[kk, jj, ii] = 1\n"
        "        pathlib.Path(opts['--output-dir']).mkdir(parents=True, exist_ok=True)\n"
        "        np.savez(pathlib.Path(opts['--output-dir']) / f'mask_{n:04d}.npz', mask=mask)\n"
        "pathlib.Path(opts['--result']).write_text(json.dumps({"
        "'model_id':'nnInteractive_v1.0','version':'2.6.0','weights_licence':'CC BY-NC-SA 4.0'}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_segment_converts_ras_to_kji_on_flipped_oblique_affine_and_records_mask(tmp_path):
    source = _capsule(tmp_path, [_prompt()])
    output = tmp_path / "segmented.capsule.html"
    _saved_path, result = segment_capsule(source, output, python_path=_fake_python(tmp_path), device="cpu")

    manifest, arrays = read_capsule(output)
    assert manifest["version"] == 3
    prompt = manifest["seg_prompts"][0]
    assert prompt["status"] == "done" and prompt["mask_id"] == "seg_nervo-facial"
    mask_meta = next(mask for mask in manifest["masks"] if mask["id"] == prompt["mask_id"])
    mask = arrays[mask_meta["blob"]]
    assert mask.dtype == np.uint8 and mask.shape == (5, 7, 9)
    expected_kji = tuple(np.floor(np.linalg.inv(AFFINE) @ np.array([*_ras((7, 2, 0)), 1.0]) + 0.5).astype(int)[[2, 1, 0]])
    assert expected_kji == (0, 2, 7)
    assert mask[expected_kji] == 1
    assert mask_meta["dtype"] == "uint8" and mask_meta["for_volume"] == "mr"
    assert mask_meta["label"] == "Nervo facial" and mask_meta["color"] == "#43B581"
    assert mask_meta["source"] == "auto" and mask_meta["reviewed"] is False and mask_meta["role"] == "structure"
    assert mask_meta["method"] == "nninteractive point prompts (1+, 1−)"
    assert mask_meta["model"] == "nnInteractive_v1.0"
    assert mask_meta["licence"] == "nnInteractive code Apache-2.0; weights CC BY-NC-SA 4.0 (non-commercial)"
    assert mask_meta["volume_ml"] == round(int(mask.sum()) * abs(np.linalg.det(AFFINE[:3, :3])) / 1000, 2)
    run = manifest["segmentation_runs"][-1]
    assert run["tool"] == "nnInteractive" and run["version"] == "2.6.0"
    assert run["model_id"] == "nnInteractive_v1.0" and run["device"] == "cpu"
    assert run["licence"] == mask_meta["licence"] and run["seconds"] >= 0
    assert result["masks_added"] == [mask_meta["id"]]
    assert _template(output) == _template(source)
    assert arrays["mr"].shape == (5, 7, 9)
    assert read_capsule_tract_blobs(output)["tract_t1"].tobytes() == b"original tract bytes"


def test_empty_result_marks_prompt_empty_without_zero_ml_mask(tmp_path):
    source = _capsule(tmp_path, [_prompt(name="empty result")])
    output = tmp_path / "empty.capsule.html"
    segment_capsule(source, output, python_path=_fake_python(tmp_path), device="cpu")

    manifest, arrays = read_capsule(output)
    prompt = manifest["seg_prompts"][0]
    assert prompt["status"] == "empty" and "mask_id" not in prompt
    assert manifest["masks"] == [] and set(arrays) == {"mr"}


def test_out_of_grid_point_names_prompt_and_does_not_run_or_write(tmp_path):
    source = _capsule(tmp_path, [_prompt(points=[{"ras": _ras((100, 4, 0)), "positive": True}])])
    output = tmp_path / "out-of-grid.capsule.html"

    with pytest.raises(ValueError, match="Nervo facial.*outside the capsule grid"):
        segment_capsule(source, output, python_path=_fake_python(tmp_path), device="cpu")
    assert not output.exists()


@pytest.mark.parametrize("destination", ["existing", "input"])
def test_segment_refuses_to_overwrite_existing_file_or_input(tmp_path, destination):
    source = _capsule(tmp_path, [_prompt()])
    output = source if destination == "input" else tmp_path / "existing.capsule.html"
    if destination == "existing":
        output.write_text("keep me", encoding="utf-8")
    before = output.read_bytes()

    with pytest.raises(FileExistsError, match="never overwritten"):
        segment_capsule(source, output, python_path=_fake_python(tmp_path), device="cpu")
    assert output.read_bytes() == before


def test_segment_cli_accepts_python_override_and_device(tmp_path, capsys):
    from capsule.cli import main

    source = _capsule(tmp_path, [_prompt()])
    output = tmp_path / "cli-segmented.capsule.html"
    assert main(["segment", str(source), "-o", str(output), "--python", str(_fake_python(tmp_path)), "--device", "cpu"]) == 0
    assert "mask seg_nervo-facial" in capsys.readouterr().out
    manifest, _ = read_capsule(output)
    assert manifest["seg_prompts"][0]["status"] == "done"
