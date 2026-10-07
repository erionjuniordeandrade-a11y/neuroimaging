from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from capsule.pack import parse_capsule, write_capsule
from conftest import ROOT

sys.path.insert(0, str(ROOT / "viewer2"))
from dev_fixture import build_phantom  # noqa: E402

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def qc_capsule_factory(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("r15-parc-qc")
    base = build_phantom(tmp_path / "base.capsule.html")
    manifest, blobs = parse_capsule(base)
    arrays = {key: np.frombuffer(value, dtype=np.uint8).copy() for key, value in blobs.items()}
    dims = manifest["grid"]["dims"]
    labels = [
        {"value": index + 1, "key": f"parcel_{index + 1}", "name": f"Parcela {index + 1}",
         "group": "Córtex", "color": "#FFFFFF", "volume_ml": 12.3 + index,
         "soft_volume_ml": 13.0 + index}
        for index in range(3)
    ]
    data = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)
    data[1, 1, 1:4] = [1, 2, 3]
    arrays["anatomy_anat_mr_gyri"] = data

    def create(verdict: str | None, filename: str):
        item = {"id": "anat_mr_gyri", "blob": "anatomy_anat_mr_gyri", "for_volume": "mr",
                "dtype": "uint8", "method": "synthetic SynthSeg", "licence": "test",
                "labels": [dict(label) for label in labels]}
        if verdict:
            failing = ["parcel_1", "parcel_2", "parcel_3"] if verdict == "FAIL" else ["parcel_1", "parcel_2"]
            item["qc"] = {"method": "synthetic", "ratio_bounds": [0.5, 2.0], "max_failing": 2,
                          "n_failing": len(failing), "failing": failing, "verdict": verdict,
                          "calibration": "pilot"}
        output = dict(manifest)
        output["anatomy"] = [*manifest["anatomy"], item]
        capsule = tmp_path / filename
        write_capsule(ROOT / "viewer2" / "template.html", capsule, output, arrays)
        return capsule

    return create


def open_capsule(chromium, capsule: Path):
    context = chromium.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    evidence = {"console": [], "page": [], "requests": []}
    page.on("console", lambda message: evidence["console"].append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: evidence["page"].append(str(error)))
    page.on("request", lambda request: evidence["requests"].append(request.url)
            if not request.url.startswith(("file:", "data:", "blob:")) else None)
    page.goto(capsule.as_uri(), wait_until="load", timeout=120000)
    status = page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status }")
    assert status == "ready", status
    return page, context, evidence


def assert_clean(evidence):
    assert evidence == {"console": [], "page": [], "requests": []}, evidence


def test_fail_hides_parcels_and_requires_explicit_overlay_toggle(chromium, qc_capsule_factory):
    page, context, evidence = open_capsule(chromium, qc_capsule_factory("FAIL", "fail.capsule.html"))
    try:
        row = page.locator("#anatomy-list [data-item='anat_mr_gyri']")
        assert "3 de 68" in row.inner_text()
        assert row.locator(".anatomy-qc-row").count() == 1
        assert row.locator(".anatomy-label").count() == 0
        assert not re.search(r"\d[\d,.]*\s*mL", row.inner_text(), re.IGNORECASE), row.inner_text()
        assert page.evaluate("window.__capsule.api.visibleAnatomy().filter(id => id.startsWith('anat_mr_gyri:'))") == []
        row.locator("button.eye").click()
        assert "Parcelamento reprovado no QC" in row.inner_text()
        assert not re.search(r"\d[\d,.]*\s*mL", row.inner_text(), re.IGNORECASE), row.inner_text()
        assert page.evaluate("window.__capsule.api.visibleAnatomy().filter(id => id.startsWith('anat_mr_gyri:'))") == [
            "anat_mr_gyri:1", "anat_mr_gyri:2", "anat_mr_gyri:3"]
        readout = page.evaluate("""() => {
          const c=window.__capsule,p=affinePoint(1,1,1);c.api.setCrosshairRAS(...p);return document.getElementById('readout').textContent;
        }""")
        assert "Parcela 1" in readout and not re.search(r"\d[\d,.]*\s*mL", readout, re.IGNORECASE), readout
        for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 600}):
            page.set_viewport_size(viewport)
            overlap = page.locator(".anatomy-qc-row").evaluate("""row => {
              const boxes=[...row.children].map(el=>{const r=el.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]});
              return boxes.some((a,i)=>boxes.slice(i+1).some(b=>a[0]<b[2]&&a[2]>b[0]&&a[1]<b[3]&&a[3]>b[1]));
            }""")
            assert not overlap, viewport
        assert_clean(evidence)
    finally:
        context.close()


def test_pass_marks_failing_rows_and_keeps_other_volumes(chromium, qc_capsule_factory):
    page, context, evidence = open_capsule(chromium, qc_capsule_factory("PASS", "pass.capsule.html"))
    try:
        rows = page.locator("#anatomy-list [data-item='anat_mr_gyri'] .anatomy-label")
        assert "falhou QC" in rows.nth(0).inner_text()
        assert "falhou QC" in rows.nth(1).inner_text()
        assert "12,3 mL" not in rows.nth(0).inner_text()
        assert "13,3 mL" not in rows.nth(1).inner_text()
        assert "14,3 mL" in rows.nth(2).inner_text()
        assert_clean(evidence)
    finally:
        context.close()


@pytest.mark.parametrize(("verdict", "expected"), [("UNAVAILABLE", "QC indisponível"), (None, "sem QC")])
def test_missing_or_unavailable_qc_is_disclosed(chromium, qc_capsule_factory, verdict, expected):
    page, context, evidence = open_capsule(chromium, qc_capsule_factory(verdict, f"{verdict or 'legacy'}.capsule.html"))
    try:
        row = page.locator("#anatomy-list [data-item='anat_mr_gyri']")
        assert expected in row.inner_text()
        assert "12,3 mL" in row.inner_text()
        assert page.locator("#anatomy-list [data-item='anat_mr'] .anatomy-qc-note").count() == 0
        assert_clean(evidence)
    finally:
        context.close()
