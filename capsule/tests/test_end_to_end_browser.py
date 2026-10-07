"""Pipeline-built phantom capsule opened in the real viewer (Chromium, file://)."""

from __future__ import annotations

import subprocess
import sys
from urllib.parse import urlsplit

import numpy as np
import pytest

from capsule.pack import read_capsule
from capsule.phantom import write_phantom


pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def pipeline_capsule(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    truth = write_phantom(root / "dicom", seed=5171)
    output = root / "phantom.capsule.html"
    subprocess.run(
        [sys.executable, "-m", "capsule.cli", "build", str(root / "dicom"), "--series", "1,2",
         "--label", "Caso E2E", "-o", str(output), "--viewer", "v1"],
        text=True, capture_output=True, check=True,
    )
    return truth, output


def test_pipeline_capsule_in_real_viewer(chromium, pipeline_capsule, repo_root):
    truth, capsule = pipeline_capsule
    manifest, arrays = read_capsule(capsule)
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)

    context = chromium.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    requests, errors = [], []
    page.on("request", lambda r: requests.append(r.url)
            if urlsplit(r.url).scheme not in {"data", "blob"} and r.url != capsule.as_uri() else None)
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(capsule.as_uri(), wait_until="load", timeout=60000)
    assert page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status; }") == "ready"

    # The viewer's decode equals the pipeline's decode at voxel centres, for every volume.
    rng = np.random.default_rng(7)
    nx, ny, nz = manifest["grid"]["dims"]
    for volume in manifest["volumes"]:
        data = arrays[volume["blob"]].astype(float) * volume["slope"] + volume["intercept"]
        for _ in range(12):
            i, j, k = int(rng.integers(nx)), int(rng.integers(ny)), int(rng.integers(nz))
            x, y, z = (affine @ [i, j, k, 1.0])[:3]
            shown = page.evaluate(f"() => window.__capsule.api.valueAtRAS('{volume['id']}', {x}, {y}, {z})")
            assert abs(shown - data[k, j, i]) <= max(1.0, abs(volume["slope"])), (volume["id"], i, j, k)

    ct_id = next(v["id"] for v in manifest["volumes"] if v["kind"] == "CT")
    lx, ly, lz = truth["lesion_center_ras_mm"]
    page.evaluate(f"() => window.__capsule.api.setCrosshairRAS({lx}, {ly}, {lz})")
    assert abs(page.evaluate(f"() => window.__capsule.api.valueAtRAS('{ct_id}', {lx}, {ly}, {lz})") - 60.0) <= 5.0
    assert "HU" in page.locator("#readout").inner_text()

    # Patient-left marker (RAS x < 0) must render on the axial image's right half.
    mx, my, mz = truth["marker_center_ras_mm"]
    assert mx < 0
    page.locator("#level-number").fill("1400")
    page.locator("#width-number").fill("400")
    page.evaluate(f"() => window.__capsule.api.setCrosshairRAS({mx}, {my}, {mz})")
    pixels = page.evaluate(
        """() => {
          const canvas = document.querySelector('#axial-canvas');
          const {width, height} = canvas;
          const rgba = canvas.getContext('2d').getImageData(0, 0, width, height).data;
          let left = 0, right = 0;
          for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const p = (y * width + x) * 4;
            if (rgba[p] > 235 && rgba[p + 1] > 235 && rgba[p + 2] > 235) {
              if (x < width / 2) left++; else right++;
            }
          }
          return {left, right};
        }"""
    )
    assert pixels["right"] > 10 and pixels["right"] > pixels["left"] * 5, pixels

    out = repo_root / "out"
    out.mkdir(exist_ok=True)
    page.locator("#level-number").fill("40")
    page.locator("#width-number").fill("80")
    page.evaluate(f"() => window.__capsule.api.setCrosshairRAS({lx}, {ly}, {lz})")
    page.screenshot(path=str(out / "e2e-pipeline-capsule.png"), full_page=True)

    assert requests == []
    assert errors == []
    context.close()
