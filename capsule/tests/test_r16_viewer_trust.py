from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from capsule.pack import parse_capsule, write_capsule
from capsule.tracts import encode_tck
from conftest import ROOT

sys.path.insert(0, str(ROOT / "viewer2"))
from dev_fixture import build_phantom  # noqa: E402

pytestmark = pytest.mark.browser


def _trust(verdict: str, *, failing: list[str], ratio: float | None, median: float | None,
           rim_flag: bool, n_source: int = 1863, yield_ratio: float | None = None) -> dict:
    return {
        "verdict": verdict,
        "failing": failing,
        "rim_flag": rim_flag,
        "yield_flag": yield_ratio is not None and yield_ratio < 1 / 3,
        "metrics": {
            "n_source": n_source,
            "median_tortuosity": median,
            "contralateral": "slf1_r" if ratio is not None else None,
            "tortuosity_ratio": ratio,
            "cap_fraction": 0.0,
            "rim_fraction": 0.61 if rim_flag else None,
            "yield_ratio": yield_ratio,
        },
        "thresholds": {
            "tortuosity_ratio_max": 1.5,
            "median_tortuosity_max": 3.5,
            "cap_fraction_max": 0.10,
            "min_streamlines": 50,
            "rim_mm": 5.0,
            "rim_fraction_flag": 0.30,
            "yield_ratio_flag": 1 / 3,
        },
        "calibration": "pilot: 1 case, 10 contralateral pairs",
    }


@pytest.fixture(scope="module")
def r16_capsule_factory(tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("r16-viewer-trust")
    base = build_phantom(work / "base.capsule.html")
    base_manifest, blobs = parse_capsule(base)
    tract_blobs = {
        tract.get(key)
        for tract in base_manifest.get("tracts", [])
        for key in ("blob", "outlier_blob")
        if tract.get(key)
    }
    base_arrays = {
        key: np.frombuffer(value, dtype=np.uint8).copy()
        for key, value in blobs.items()
        if key not in tract_blobs
    }

    def lines(shift: float, count: int) -> list[np.ndarray]:
        return [
            np.array([[shift + index * 0.01, 0, 0], [shift + index * 0.01, 1, 0],
                      [shift + index * 0.01, 2, 0]], dtype=np.float32)
            for index in range(count)
        ]

    def create(tracts: list[dict], filename: str, brain_mask_qc: dict | None = None,
               skull_stripped_input: bool = False) -> Path:
        manifest = copy.deepcopy(base_manifest)
        arrays = {key: value.copy() for key, value in base_arrays.items()}
        if skull_stripped_input:
            mr = next(volume for volume in manifest["volumes"] if volume["kind"] == "MR")
            mr["skull_stripped_input"] = True
        manifest["tracts"] = []
        for index, spec in enumerate(tracts):
            tract_id = f"t{index + 1:02d}"
            blob = f"tract_{tract_id}"
            source_count = (spec.get("trust") or {}).get("metrics", {}).get("n_source", 1863)
            display_count = min(64, source_count)
            arrays[blob] = np.frombuffer(encode_tck(lines(index * 3.0, display_count)), dtype=np.uint8).copy()
            meta = {
                "id": tract_id,
                "blob": blob,
                "label": spec["label"],
                "color": "#4FC3F7" if index == 0 else "#7CFC00",
                "format": "tck",
                "n_streamlines": display_count,
                "n_streamlines_source": source_count,
                "source": "synthetic",
                "reviewed": True,
            }
            if spec.get("trust") is not None:
                meta["trust"] = copy.deepcopy(spec["trust"])
            manifest["tracts"].append(meta)

        if brain_mask_qc is not None:
            manifest["brain_mask_qc"] = copy.deepcopy(brain_mask_qc)
            if brain_mask_qc.get("verdict") == "FAIL":
                brain_mask_blobs = {
                    mask["blob"] for mask in manifest.get("masks", [])
                    if mask.get("role") == "render" and
                    (mask.get("id") == "brain" or mask.get("id", "").startswith("brain_"))
                }
                manifest["masks"] = [
                    mask for mask in manifest.get("masks", [])
                    if mask.get("blob") not in brain_mask_blobs
                ]
                for blob in brain_mask_blobs:
                    arrays.pop(blob, None)

        output = work / filename
        write_capsule(ROOT / "viewer2" / "template.html", output, manifest, arrays)
        return output

    return create


def _open_capsule(chromium, capsule: Path):
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


def _assert_clean(evidence: dict) -> None:
    assert evidence == {"console": [], "page": [], "requests": []}, evidence


def test_fail_tract_is_hidden_and_disclosed_when_toggled(chromium, r16_capsule_factory):
    trust = _trust("FAIL", failing=["tortuosity_ratio", "median_tortuosity"], ratio=2.48,
                   median=3.89, rim_flag=True)
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": trust}], "fail.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        row = page.locator("#tract-list .item").filter(has_text="Fascículo longitudinal superior I (SLF I) E")
        row_text = row.inner_text()
        assert "— reprovado no QC de confiabilidade (tortuosidade 2,5× a do lado oposto, trajeto excessivamente tortuoso)" in row_text
        assert not re.search(r"\d[\d,.]*\s*mm\b", row_text, re.IGNORECASE), row_text
        assert page.evaluate("window.__capsule.api.visibleTracts()") == []
        assert page.locator("#tract-trust-chip").count() == 0

        row.locator("button.eye").click()
        assert page.evaluate("window.__capsule.api.visibleTracts()") == ["t01"]
        chip = page.locator("#tract-trust-chip")
        assert chip.is_visible() and chip.inner_text() == "Feixe reprovado no QC"

        row.locator("button.eye").click()
        assert page.evaluate("window.__capsule.api.visibleTracts()") == []
        assert not chip.is_visible()
        _assert_clean(evidence)
    finally:
        context.close()


def test_pass_tract_with_rim_flag_stays_visible_and_is_noted(chromium, r16_capsule_factory):
    trust = _trust("PASS", failing=[], ratio=1.1, median=2.4, rim_flag=True)
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": trust}], "pass-rim.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        row = page.locator("#tract-list .item").filter(has_text="Fascículo longitudinal superior I (SLF I) E")
        assert "passa junto à lesão" in row.inner_text()
        assert page.evaluate("window.__capsule.api.visibleTracts()") == ["t01"]
        assert page.locator("#tract-trust-chip").count() == 0
        _assert_clean(evidence)
    finally:
        context.close()


@pytest.mark.parametrize(
    ("trust", "expected"),
    [
        (_trust("UNAVAILABLE", failing=[], ratio=None, median=None, rim_flag=False, n_source=12),
         "QC indisponível"),
        (None, "sem QC"),
    ],
    ids=["unavailable", "legacy-no-trust"],
)
def test_unavailable_and_legacy_trust_states_are_disclosed(chromium, r16_capsule_factory, trust, expected):
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": trust}], f"{expected}.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        row_text = page.locator("#tract-list .item").inner_text()
        assert expected in row_text
        assert not re.search(r"\d[\d,.]*\s*mm\b", row_text, re.IGNORECASE), row_text
        _assert_clean(evidence)
    finally:
        context.close()


def test_failed_brain_mask_qc_discloses_missing_render_and_removes_preset(chromium, r16_capsule_factory):
    brain_mask_qc = {
        "gate_ml": [900, 1800],
        "attempts": [{"method": "synthstrip", "volume_ml": 2900.0, "verdict": "FAIL"}],
        "verdict": "FAIL",
    }
    capsule = r16_capsule_factory([], "brain-mask-fail.capsule.html", brain_mask_qc=brain_mask_qc)
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        mr_id = page.evaluate("""() => {
          const id=window.__capsule.state.manifest.volumes.find(volume => volume.kind === 'MR').id;
          window.__capsule.api.setBase(id);
          return id;
        }""")
        assert page.evaluate("id => window.__capsule.api.renderMaskFor(id, 'brain')", mr_id) is None
        assert page.locator("#preset3d option[value='mr-brain']").count() == 0
        note = page.locator("#brain-mask-qc-note")
        assert note.is_visible()
        assert note.inner_text() == "Máscara cerebral reprovada no QC — render do cérebro indisponível"
        assert not re.search(r"\d[\d,.]*\s*mm\b", note.inner_text(), re.IGNORECASE)
        _assert_clean(evidence)
    finally:
        context.close()


def test_failed_brain_mask_qc_hides_3d_when_no_safe_mr_preset_exists(chromium, r16_capsule_factory):
    brain_mask_qc = {
        "gate_ml": [900, 1800],
        "attempts": [{"method": "synthstrip", "volume_ml": 2900.0, "verdict": "FAIL"}],
        "verdict": "FAIL",
    }
    capsule = r16_capsule_factory([], "brain-mask-fail-skull-stripped.capsule.html",
                                  brain_mask_qc=brain_mask_qc, skull_stripped_input=True)
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        mr_id = page.evaluate("""() => {
          const id=window.__capsule.state.manifest.volumes.find(volume => volume.kind === 'MR').id;
          window.__capsule.api.setBase(id);
          return id;
        }""")
        page.evaluate("async () => { await window.__capsule.api.renderReady() }")
        assert page.evaluate("id => window.__capsule.state.base === id && !window.__capsule.state.preset3d", mr_id)
        assert page.evaluate("window.__capsule.state.render3d === null")
        assert page.locator("#preset3d option").count() == 0
        assert not page.locator("#gl3d-wrap").is_visible()
        note = page.locator("#brain-mask-qc-note")
        assert note.is_visible()
        assert note.inner_text() == "Máscara cerebral reprovada no QC — render do cérebro indisponível"
        _assert_clean(evidence)

        page.evaluate("""() => {
          const id=window.__capsule.state.manifest.volumes.find(volume => volume.kind === 'CT').id;
          window.__capsule.api.setBase(id);
        }""")
        page.evaluate("async () => { await window.__capsule.api.renderReady() }")
        assert page.evaluate("window.__capsule.state.preset3d === 'ct-bone'")
        assert page.locator("#gl3d-wrap").is_visible()
        _assert_clean(evidence)
    finally:
        context.close()


@pytest.mark.parametrize(
    ("trust", "expected"),
    [
        (_trust("PASS", failing=[], ratio=1.16, median=1.6, rim_flag=True, n_source=279, yield_ratio=279 / 6497),
         "passa junto à lesão · poucas fibras: 23,3× menos que o lado oposto (não prova ausência do trato)"),
        (_trust("UNAVAILABLE", failing=[], ratio=None, median=None, rim_flag=False, n_source=5,
                yield_ratio=5 / 608),
         "QC indisponível · poucas fibras: 121,6× menos que o lado oposto (não prova ausência do trato)"),
    ],
    ids=["pass-low-yield", "unavailable-low-yield"],
)
def test_low_yield_tract_is_noted_and_stays_visible(chromium, r16_capsule_factory, trust, expected):
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": trust}], f"yield-{trust['verdict']}.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        row_text = page.locator("#tract-list .item").inner_text()
        assert expected in row_text
        assert not re.search(r"\d[\d,.]*\s*mm\b", row_text, re.IGNORECASE), row_text
        if trust["verdict"] == "PASS":
            assert page.evaluate("window.__capsule.api.visibleTracts()") == ["t01"]
        _assert_clean(evidence)
    finally:
        context.close()


def test_balanced_yield_adds_no_note(chromium, r16_capsule_factory):
    trust = _trust("PASS", failing=[], ratio=0.94, median=2.3, rim_flag=False, n_source=5088, yield_ratio=0.5)
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": trust}], "yield-balanced.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        assert "poucas fibras" not in page.locator("#tract-list .item").inner_text()
        _assert_clean(evidence)
    finally:
        context.close()


def test_yield_flag_replaces_the_r14_pair_warning(chromium, r16_capsule_factory):
    low = _trust("PASS", failing=[], ratio=1.16, median=1.6, rim_flag=False, n_source=279, yield_ratio=279 / 6497)
    high = _trust("PASS", failing=[], ratio=0.86, median=1.4, rim_flag=False, n_source=6497, yield_ratio=6497 / 279)
    capsule = r16_capsule_factory([{"label": "slf1_l", "trust": low}, {"label": "slf1_r", "trust": high}],
                                  "yield-pair.capsule.html")
    page, context, evidence = _open_capsule(chromium, capsule)
    try:
        assert "poucas fibras: 23,3×" in page.locator("#tract-list").inner_text()
        assert page.locator("#tract-list .tract-warning").count() == 0
        _assert_clean(evidence)
    finally:
        context.close()
