from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import parse_capsule


pytestmark = pytest.mark.browser


def _write_empty_tour_copy(source: Path, output: Path) -> Path:
    html = source.read_text(encoding="utf-8")
    manifest_script = re.compile(
        r'(<script\s+id="capsule-manifest"\s+type="application/json">)(.*?)(</script>)',
        flags=re.DOTALL,
    )
    match = manifest_script.search(html)
    assert match is not None
    manifest = json.loads(match.group(2))
    manifest["tour"] = []
    replacement = match.group(1) + json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + match.group(3)
    output.write_text(html[: match.start()] + replacement + html[match.end() :], encoding="utf-8")
    return output


def test_browser_viewer_acceptance(viewer_session, capsule_file: Path, repo_root: Path, tmp_path: Path) -> None:
    page, _context, evidence = viewer_session

    # Human review artifacts are emitted by this real-browser acceptance test.
    out = repo_root / "out"
    out.mkdir(parents=True, exist_ok=True)
    surgeon_screenshot = out / "viewer-surgeon-2x2.png"
    page.screenshot(path=str(surgeon_screenshot), full_page=True)
    assert surgeon_screenshot.is_file()

    webgl = page.evaluate(
        """() => {
          const canvas = document.querySelector('#three-canvas');
          const gl = canvas && canvas.getContext('webgl2');
          return {
            created: !!gl,
            sameContext: !!gl && window.__capsule.state.gl === gl,
            version: gl && gl.getParameter(gl.VERSION),
            renderer: gl && gl.getParameter(gl.RENDERER)
          };
        }"""
    )
    assert webgl["created"] is True
    assert webgl["sameContext"] is True
    assert "WebGL 2" in webgl["version"]

    page.evaluate("() => window.__capsule.api.setCrosshairRAS(10, 0, 0)")
    lesion_hu = page.evaluate("() => window.__capsule.api.valueAtRAS('ct', 10, 0, 0)")
    assert abs(lesion_hu - 60.0) <= 5.0
    assert "60.0 HU" in page.locator("#readout").inner_text()

    page.evaluate("() => window.__capsule.api.setCrosshairRAS(70, 70, 70)")
    air_hu = page.evaluate("() => window.__capsule.api.valueAtRAS('ct', 70, 70, 70)")
    assert abs(air_hu + 1000.0) <= 10.0
    assert "-1000.0 HU" in page.locator("#readout").inner_text()

    # Use a narrow marker window: 2000 HU renders white, while the 1000 HU
    # symmetric bone shell is black. RAS negative x must land in image-right.
    page.locator("#level-number").fill("1600")
    page.locator("#width-number").fill("400")
    page.evaluate("() => window.__capsule.api.setCrosshairRAS(-35, 0, 0)")
    marker_pixels = page.evaluate(
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
          return {left, right, width, height};
        }"""
    )
    assert marker_pixels["right"] > 20, marker_pixels
    assert marker_pixels["right"] > marker_pixels["left"] * 5, marker_pixels

    page.evaluate("() => window.__capsule.api.setCrosshairRAS(10, 0, 0)")
    grown = page.evaluate(
        "() => window.__capsule.api.growRegionRAS(10, 0, 0, {volumeId: 'ct', tolerance: 5, radiusMm: 30})"
    )
    expected_ml = (4.0 / 3.0) * 3.141592653589793 * 10.0**3 / 1000.0
    assert abs(grown["volume_ml"] - expected_ml) / expected_ml < 0.10

    annotation = page.evaluate(
        "() => window.__capsule.api.addAnnotation('distance', [[10, 0, 0], [20, 0, 0]], 'Distância sintética')"
    )
    assert annotation["type"] == "distance"
    assert abs(annotation["value"] - 10.0) < 1e-6

    with page.expect_download(timeout=30000) as download_info:
        page.locator("#save-button").click()
    download = download_info.value
    saved_file = tmp_path / download.suggested_filename
    download.save_as(saved_file)
    assert saved_file.name.endswith(".v2.capsule.html")

    saved_manifest, saved_blobs = parse_capsule(saved_file)
    assert saved_manifest["version"] == 2
    assert any(item["id"] == annotation["id"] for item in saved_manifest["annotations"])
    assert saved_manifest["synthetic_extension"] == {"preserve_on_save": True, "kind": "fixture"}
    original_manifest, _ = parse_capsule(capsule_file)
    original_mask_ids = {item["id"] for item in original_manifest["masks"]}
    saved_mask_ids = {item["id"] for item in saved_manifest["masks"]}
    assert original_mask_ids <= saved_mask_ids
    for mask in saved_manifest["masks"]:
        assert len(saved_blobs[mask["blob"]]) == 160 * 192 * 160

    # Reopen the actual download from file:// and verify the application state.
    evidence["allowed_file_urls"].add(saved_file.as_uri())
    page.goto(saved_file.as_uri(), wait_until="load", timeout=60000)
    assert page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status; }") == "ready"
    reopened = page.evaluate(
        """() => ({
          version: window.__capsule.state.manifest.version,
          annotationCount: window.__capsule.state.manifest.annotations.length,
          annotationLabel: window.__capsule.state.manifest.annotations[0]?.label,
          maskCount: window.__capsule.state.masks.size
        })"""
    )
    assert reopened == {
        "version": 2,
        "annotationCount": 1,
        "annotationLabel": "Distância sintética",
        "maskCount": len(saved_manifest["masks"]),
    }

    page.evaluate("() => window.__capsule.api.setMode('patient')")
    first_step = page.evaluate("() => window.__capsule.state.crosshair.slice()")
    assert first_step == [10.0, 0.0, 0.0]
    assert page.locator("#step-title").text_content() == "Lesão sintética"
    patient_screenshot = out / "viewer-patient.png"
    page.screenshot(path=str(patient_screenshot), full_page=True)
    assert patient_screenshot.is_file()

    page.locator("#next-step").click()
    next_step = page.evaluate("() => window.__capsule.state.crosshair.slice()")
    assert next_step == [-35.0, 0.0, 0.0]
    assert page.locator("#step-title").text_content() == "Marcador sintético"

    # The active tour explicitly requests both masks at this slice. Patient
    # mode must still draw only the surgeon-reviewed mask.
    patient_mask_ids = page.evaluate(
        """() => {
          const tourFilter = window.__capsule.state.tourMaskFilter.slice();
          window.__capsule.api.setCrosshairRAS(10, 0, 0);
          const canvas = document.querySelector('#axial-canvas');
          // Sample inside each mask away from the gold crosshair lines.
          const [lx, ly] = rasToPixel('axial', [14, 3, 0], canvas);
          const lesion = canvas.getContext('2d').getImageData(Math.round(lx), Math.round(ly), 1, 1).data;
          window.__capsule.api.setCrosshairRAS(-20, -20, 0);
          const [x, y] = rasToPixel('axial', [-18, -18, 0], canvas);
          const rgba = canvas.getContext('2d').getImageData(Math.round(x), Math.round(y), 1, 1).data;
          return {
            tourFilter,
            ids: visibleMasks().map(mask => mask.id),
            lesion: [lesion[0], lesion[1], lesion[2]],
            unreviewed: [rgba[0], rgba[1], rgba[2]]
          };
        }"""
    )
    assert patient_mask_ids["tourFilter"] == ["lesion", "unreviewed"]
    assert patient_mask_ids["ids"] == ["lesion"]
    assert patient_mask_ids["lesion"][0] > 150 and patient_mask_ids["lesion"][1] < 130, patient_mask_ids
    assert patient_mask_ids["unreviewed"][1] < 140, patient_mask_ids
    assert "Região sintética não revisada" not in page.locator("body").inner_text()

    empty_tour_file = _write_empty_tour_copy(capsule_file, tmp_path / "empty-tour.capsule.html")
    evidence["allowed_file_urls"].add(empty_tour_file.as_uri())
    page.goto(empty_tour_file.as_uri(), wait_until="load", timeout=60000)
    assert page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status; }") == "ready"
    page.evaluate("() => window.__capsule.api.setMode('patient')")
    assert "O cirurgião ainda não preparou a explicação" in page.locator("#step-text").inner_text()

    assert evidence["requests"] == []
    assert evidence["console_errors"] == []
    assert evidence["page_errors"] == []
