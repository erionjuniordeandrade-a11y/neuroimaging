from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from capsule.pack import parse_capsule, write_capsule
from capsule.tracts import encode_tck
from conftest import ROOT

sys.path.insert(0, str(ROOT / "viewer2"))
from dev_fixture import build_phantom  # noqa: E402  (synthetic fixture only)

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def trust_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("r14-trust")
    base = build_phantom(work / "base.capsule.html")
    manifest, blobs = parse_capsule(base)
    tract_blobs = {t.get(key) for t in manifest["tracts"] for key in ("blob", "outlier_blob") if t.get(key)}
    arrays = {key: np.frombuffer(value, dtype=np.uint8).copy() for key, value in blobs.items() if key not in tract_blobs}

    def lines(shift: float) -> list[np.ndarray]:
        output = []
        for index in range(1500):
            x = shift + (index % 300) * 0.02
            y = ((index // 300) % 5) * 0.02
            output.append(np.array([[x, y, 0], [x, y + 0.5, 0], [x, y + 1.0, 0]], dtype=np.float32))
        return output

    new_tracts = []
    # Same shape as capsule/cli.py: opaque ids, XTRACT bundle key in the label.
    for tract_id, label, shift, source, color in (("t01", "fa_l", -3.0, 16000, "#4FC3F7"), ("t02", "fa_r", 3.0, 1600, "#7CFC00")):
        blob = "tract_" + tract_id
        arrays[blob] = np.frombuffer(encode_tck(lines(shift)), dtype=np.uint8).copy()
        new_tracts.append({"id": tract_id, "blob": blob, "label": label, "color": color, "format": "tck",
                           "n_streamlines": 1500, "n_streamlines_source": source, "source": "synthetic", "reviewed": True})
    manifest["tracts"] = new_tracts
    output = work / "trust.capsule.html"
    write_capsule(ROOT / "viewer2" / "template.html", output, manifest, arrays)
    return output


def test_density_and_trust_presentation(chromium, trust_capsule: Path, request) -> None:
    browser_context = chromium.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    request.addfinalizer(browser_context.close)
    page = browser_context.new_page()
    external_requests: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on("request", lambda request: external_requests.append(request.url) if not request.url.startswith(("file:", "data:", "blob:")) else None)
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(trust_capsule.as_uri(), wait_until="load", timeout=120000)
    assert page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status }") == "ready"
    page.evaluate("window.__capsule.api.setLayout('3d')")

    default = page.evaluate("""() => {
      const s=window.__capsule.state;
      return ['t01','t02'].map(id=>{const t=s.tracts.get(id);return {id,k2:t.mesh.offsetPt0.length-1,k3:t.mesh3d.offsetPt0.length-1,
        indexCount2d:t.mesh.indexCount,indexCount3d:t.mesh3d.indexCount,
        count:t.displayCount,indices:Array.from(t.displayIndices),sharedPoints:t.mesh.pts===t.mesh3d.pts,sharedOffsets:t.mesh.offsetPt0===t.mesh3d.offsetPt0,
        fiberLength:t.mesh.fiberLength,fiberDensityLength:(t.mesh3d.createFiberDensityMap(),t.mesh3d.fiberDensity?.length||0),
        pointsPerLine:t.streamlines[0].length/3,fiberRadius:t.mesh3d.fiberRadius,fiberSides:t.mesh3d.fiberSides}});
    }""")
    assert [(t["k2"], t["k3"], t["count"]) for t in default] == [(1500, 1500, 1500), (150, 150, 150)], default
    assert [t["indexCount2d"] for t in default] == [6000, 600], default
    assert [t["indexCount3d"] for t in default] == [
        t["count"] * ((t["pointsPerLine"] - 1) * t["fiberSides"] * 6 if t["fiberRadius"] > 0 else t["pointsPerLine"] + 1)
        for t in default
    ], default
    # Software GL disables tubes/occlusion, so NiiVue never builds the density map itself; build it here.
    # A stale cache would keep the full-bundle length instead of the displayed point count.
    assert [t["fiberDensityLength"] for t in default] == [4500, 450], default
    assert default[0]["sharedPoints"] and default[0]["sharedOffsets"] and default[0]["fiberLength"] == -1, default[0]
    assert default[1]["indices"][0] == 0 and default[1]["indices"][-1] == 1499 and len(set(default[1]["indices"])) == 150, default[1]
    row_names = page.locator("#tract-list .item .name").all_text_contents()
    assert row_names == ["Aslant frontal (FAT) E", "Aslant frontal (FAT) D"], row_names
    assert "1.500 fibras / 16.000" in page.locator("#tract-list").inner_text()
    warning = page.locator("#tract-list .tract-warning")
    assert warning.is_visible() and "Assimetria de contagem E/D (16.000 vs 1.600)." in warning.inner_text()

    page.evaluate("""() => {const s=window.__capsule.state;s.manifest.tracts[0].n_streamlines_source=1000;s.manifest.tracts[1].n_streamlines_source=900;window.__capsule.api.setTractVisible('t01',true)}""")
    assert page.locator("#tract-list .tract-warning").count() == 0
    page.evaluate("""() => {delete window.__capsule.state.manifest.tracts[0].n_streamlines_source;window.__capsule.api.setTractVisible('t01',true)}""")
    assert page.locator("#tract-list .tract-warning").count() == 0
    page.evaluate("""() => {const t=window.__capsule.state.manifest.tracts;t[0].n_streamlines_source=16000;t[1].n_streamlines_source=1600;window.__capsule.api.setTractVisible('t01',true)}""")

    page.evaluate("window.__capsule.api.setTractVisible('t01',false)")
    visible_denominator = page.evaluate("window.__capsule.state.tracts.get('t02').displayCount")
    assert visible_denominator == 1500, visible_denominator
    page.evaluate("window.__capsule.api.setTractVisible('t01',true)")
    restored_denominator = page.evaluate("window.__capsule.state.tracts.get('t02').displayCount")
    assert restored_denominator == 150, restored_denominator

    # In-page click, as the existing browser suite does: software-GL frames starve Playwright's stability wait.
    page.evaluate("document.getElementById('tract-density-toggle').click()")
    # fa_r's density map was built at 150 lines (450 points); a stale cache would stay at 450 after equalizing.
    density_after = page.evaluate("""() => {const m=window.__capsule.state.tracts.get('t02').mesh3d;m.createFiberDensityMap();return m.fiberDensity.length}""")
    assert density_after == 4500, density_after
    equalized = page.evaluate("""() => ['t01','t02'].map(id=>{const t=window.__capsule.state.tracts.get(id);return [t.mesh.offsetPt0.length-1,t.mesh3d.offsetPt0.length-1,t.mesh.indexCount,t.mesh3d.indexCount]})""")
    assert equalized == [[1500, 1500, 6000, default[0]["indexCount3d"]], [1500, 1500, 6000, default[0]["indexCount3d"]]], equalized
    assert page.locator("#density-warning").is_visible()
    assert page.locator("#density-warning").inner_text() == "Densidade igualada — não compare lados pela aparência"
    assert page.locator("#tract-density-toggle").get_attribute("aria-pressed") == "true"
    page.evaluate("""() => {window.__capsule.api.setTractVisible('t01',false);window.__capsule.api.setTractVisible('t02',false)}""")
    assert page.locator("#density-warning").is_visible()
    page.evaluate("""() => {window.__capsule.api.setTractVisible('t01',true);window.__capsule.api.setTractVisible('t02',true)}""")

    page.evaluate("""() => {const s=window.__capsule.state;s.manifest.tracts[1].label='Rótulo próprio';window.__capsule.api.setTractVisible('t02',true)}""")
    assert page.locator("#tract-list .item .name").all_text_contents() == ["Aslant frontal (FAT) E", "Rótulo próprio"]
    page.evaluate("""() => {const s=window.__capsule.state;s.manifest.tracts[1].label='fa_r';
      const mask=s.manifest.masks.find(m=>m.id==='lesion');mask.volume_ml=16.57;window.__capsule.api.setMaskVisible('lesion',true);
      const anatomy=s.manifest.anatomy[0],label=anatomy.labels[0];label.volume_ml=16.57;window.__capsule.api.setAnatomyVisible(anatomy.id,label.value,true)}""")
    assert "16,6 mL" in page.locator("#mask-list").inner_text()
    assert "16,6 mL" in page.locator("#anatomy-list").inner_text()
    assert page.evaluate("window.__capsule.state.manifest.masks.find(m=>m.id==='lesion').volume_ml") == 16.57
    assert page.evaluate("window.__capsule.state.manifest.anatomy[0].labels[0].volume_ml") == 16.57

    for mode in ("surgeon", "patient"):
        page.evaluate("mode => window.__capsule.api.setMode(mode)", mode)
        chip = page.locator("#intended-use-chip")
        assert chip.is_visible() and chip.inner_text() == "Apoio à discussão — não usar para navegação"
    page.evaluate("window.__capsule.api.setMode('surgeon')")

    page.set_viewport_size({"width": 390, "height": 844})
    overlap = page.evaluate("""() => {
      const ids=['.topbar .brand','#intended-use-chip','#layout-buttons','.topbar .actions'];
      const r=ids.map(id=>{const b=document.querySelector(id).getBoundingClientRect();return {id,left:b.left,right:b.right,top:b.top,bottom:b.bottom,visible:b.width>0&&b.height>0}}).filter(x=>x.visible);
      const pairs=[];for(let i=0;i<r.length;i++)for(let j=i+1;j<r.length;j++)if(r[i].left<r[j].right&&r[i].right>r[j].left&&r[i].top<r[j].bottom&&r[i].bottom>r[j].top)pairs.push([r[i].id,r[j].id]);
      return {r,pairs,chipVisible:document.getElementById('intended-use-chip').getBoundingClientRect().width>0};
    }""")
    assert overlap["chipVisible"] and not overlap["pairs"], overlap
    page.set_viewport_size({"width": 1440, "height": 600})
    assert page.locator("#intended-use-chip").is_visible()

    saved = page.evaluate("""async () => {
      const result=await window.__capsule.api.save(),html=await result.blob.text(),doc=new DOMParser().parseFromString(html,'text/html');
      return {manifest:JSON.parse(doc.getElementById('capsule-manifest').textContent),
        warning:doc.getElementById('density-warning')?.hidden,
        density:doc.getElementById('tract-density-toggle')?.getAttribute('aria-pressed'),
        intended:doc.getElementById('intended-use-chip')?.textContent};
    }""")
    assert saved["manifest"]["tracts"][0]["n_streamlines_source"] == 16000
    assert saved["manifest"]["masks"][0]["volume_ml"] == 16.57
    assert saved["manifest"]["anatomy"][0]["labels"][0]["volume_ml"] == 16.57
    assert saved["warning"] is True and saved["density"] == "false"
    assert saved["intended"] == "Apoio à discussão — não usar para navegação"
    assert external_requests == [] and console_errors == [] and page_errors == [], (external_requests, console_errors, page_errors)
