"""Browser acceptance for viewer2 (NiiVue 0.69 build) on the synthetic phantom capsule."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from conftest import ROOT, _cached_chromium_executable, parse_capsule

sys.path.insert(0, str(ROOT / "viewer2"))
from dev_fixture import read_tck  # noqa: E402  (independent reader, not the JS parser)

pytestmark = pytest.mark.browser

DTYPES = {"int16": np.int16, "uint16": np.uint16, "uint8": np.uint8}
LESION_RAS = (14.0, -9.0, 4.0)
MARKER_RAS = (-56.0, 8.0, 12.0)


@pytest.fixture(scope="session")
def v2_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("case-capsule-v2") / "fixture.capsule.html"
    result = subprocess.run(
        [sys.executable, str(ROOT / "viewer2" / "dev_fixture.py"), "-o", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AssertionError(f"viewer2 fixture failed ({result.returncode}): {result.stderr}")
    return output


@pytest.fixture(scope="session")
def v2_corridor_vessel_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("case-capsule-v2-corridor-vessel") / "fixture.capsule.html"
    result = subprocess.run(
        [sys.executable, str(ROOT / "viewer2" / "dev_fixture.py"), "-o", str(output), "--corridor-vessel"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AssertionError(f"viewer2 corridor vessel fixture failed ({result.returncode}): {result.stderr}")
    return output


@pytest.fixture(scope="session")
def v2_cropped_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("case-capsule-v2-cropped") / "fixture.capsule.html"
    result = subprocess.run(
        [sys.executable, str(ROOT / "viewer2" / "dev_fixture.py"), "-o", str(output), "--crop"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AssertionError(f"cropped viewer2 fixture failed ({result.returncode}): {result.stderr}")
    return output


def _open(chromium, capsule: Path):
    context = chromium.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    evidence: dict[str, list[str]] = {"requests": [], "console_errors": [], "page_errors": []}

    def record_request(request) -> None:
        scheme = urlsplit(request.url).scheme
        if scheme in {"data", "blob"} or request.url == capsule.as_uri():
            return
        evidence["requests"].append(request.url)

    page.on("request", record_request)
    page.on("console", lambda m: evidence["console_errors"].append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: evidence["page_errors"].append(str(e)))
    page.goto(capsule.as_uri(), wait_until="load", timeout=120000)
    status = page.evaluate("async () => { try { await window.__capsule.ready } catch (e) { return 'ERR ' + e.message } return window.__capsule.state.status }")
    assert status == "ready", status
    return page, context, evidence


@pytest.fixture
def v2(chromium, v2_capsule: Path):
    page, context, evidence = _open(chromium, v2_capsule)
    yield page, context, evidence
    context.close()


def _assert_clean(evidence) -> None:
    assert evidence["requests"] == [], evidence["requests"]
    assert evidence["console_errors"] == [], evidence["console_errors"]
    assert evidence["page_errors"] == [], evidence["page_errors"]


def _volume_arrays(manifest, blobs):
    nx, ny, nz = manifest["grid"]["dims"]
    for meta in manifest["volumes"]:
        data = np.frombuffer(blobs[meta["blob"]], dtype=DTYPES[meta["dtype"]]).reshape(nz, ny, nx)
        yield meta, data.astype(np.float64) * float(meta.get("slope", 1) or 1) + float(meta.get("intercept", 0) or 0)


def test_values_match_python_decode_and_lesion_hu(v2, v2_capsule: Path) -> None:
    page, _, evidence = v2
    manifest, blobs = parse_capsule(v2_capsule)
    affine = np.array(manifest["grid"]["affine_ras"], dtype=float)
    nx, ny, nz = manifest["grid"]["dims"]
    rng = np.random.default_rng(20260927)
    checked = 0
    for meta, values in _volume_arrays(manifest, blobs):
        ijk = np.stack([rng.integers(0, n, 12) for n in (nx, ny, nz)], axis=1)
        ras = (affine @ np.c_[ijk, np.ones(12)].T).T[:, :3]
        expected = [values[k, j, i] for i, j, k in ijk]
        got = page.evaluate(
            """([id, pts]) => pts.map(p => {
                 const js = window.__capsule.api.valueAtRAS(id, ...p);
                 const img = window.__capsule.state.volumes.get(id).nvimg, v = img.mm2vox(p);
                 return [js, img.getValue(v[0], v[1], v[2])];
               })""",
            [meta["id"], ras.tolist()],
        )
        for (js_value, nv_value), want in zip(got, expected):
            tol = 1e-3 * max(1.0, abs(want))
            assert abs(js_value - want) <= tol, (meta["id"], js_value, want)
            # Same voxel read back through NiiVue's own NVImage (the NIfTI built in JS): catches header/affine mistakes.
            assert abs(nv_value - want) <= tol, (meta["id"], "niivue", nv_value, want)
            checked += 1
    assert checked == 12 * len(manifest["volumes"])
    lesion = page.evaluate("p => window.__capsule.api.valueAtRAS('ct', ...p)", list(LESION_RAS))
    assert 55 <= lesion <= 65, lesion
    _assert_clean(evidence)


PIXELS_JS = """
(radiological) => {
  const c = window.__capsule, nv = c.state.nv, api = c.api;
  api.setMaskVisible('unreviewed', false);
  api.setLayout('axial');
  api.setWindow('ct', 1500, 200);             // only the 1800 HU marker is white
  api.setCrosshairRAS(...%s);
  nv.opts.isRadiologicalConvention = radiological;
  nv.opts.isOrientationTextVisible = false;
  nv.opts.crosshairWidth = 0;
  nv.drawScene();
  const gl = nv.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, b = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, b);
  let n = 0, sx = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4;
    if (b[o] > 200 && b[o + 1] > 200 && b[o + 2] > 200) { n++; sx += x; }
  }
  return {n, meanX: n ? sx / n : null, width: w};
}
""" % (list(MARKER_RAS),)


def test_patient_left_marker_is_on_screen_right(v2) -> None:
    page, _, evidence = v2
    radiological = page.evaluate(PIXELS_JS, True)
    assert radiological["n"] > 20, radiological
    assert radiological["meanX"] > radiological["width"] / 2, radiological
    # Known-positive control on the same path: neurological convention must move the marker to screen left.
    neurological = page.evaluate(PIXELS_JS, False)
    assert neurological["n"] > 20, neurological
    assert neurological["meanX"] < neurological["width"] / 2, neurological
    _assert_clean(evidence)


def test_hiding_a_tract_changes_3d_pixels(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """() => {
          const api = window.__capsule.api, nv = window.__capsule.state.nv3d;  // the 3D view is its own instance
          api.setLayout('3d'); api.setCrosshairRAS(%s); api.setClip('sagittal');
          const snap = () => { nv.drawScene(); const gl = nv.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight,
            b = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, b); return b; };
          const diff = (a, b) => { let n = 0; for (let i = 0; i < a.length; i += 4)
            if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 30) n++; return n; };
          const shown = snap(); api.setTractVisible('through', false); const hidden = snap();
          api.setTractVisible('through', true); const again = snap();
          return {changed: diff(shown, hidden), control: diff(shown, again),
                  meshes: nv.meshes.map(m => [m.name, m.visible])};
        }""" % ", ".join(str(v) for v in LESION_RAS)
    )
    assert result["changed"] > 500, result
    assert result["control"] < 50, result  # re-showing restores the frame: the change is the tract, not noise
    _assert_clean(evidence)


def test_eye_toggles_keep_their_button_across_rapid_clicks(v2) -> None:
    # Regression: each toggle rebuilt the list, so a held button went stale and quick successive clicks were lost.
    page, _, evidence = v2
    for list_id, key in (("tract-list", "tractVisible"), ("mask-list", "maskVisible")):
        eye = page.locator(f"#{list_id} button.eye").first.element_handle()
        for _ in range(3):
            eye.click(timeout=2000)
        state = page.evaluate(f"(el) => ({{connected: el.isConnected, off: el.classList.contains('off'),"
                              f" hidden: Object.values(window.__capsule.state.{key}).filter(v => v === false).length}})", eye)
        assert state == {"connected": True, "off": True, "hidden": 1}, (list_id, state)
        eye.click(timeout=2000)
        assert page.evaluate(f"() => Object.values(window.__capsule.state.{key}).every(v => v !== false)"), list_id
    _assert_clean(evidence)


def test_patient_mode_hides_unreviewed_tract_and_mask(v2) -> None:
    page, _, evidence = v2
    surgeon = page.evaluate("() => ({t: window.__capsule.api.visibleTracts(), m: window.__capsule.api.visibleMasks()})")
    # Surgeon mode shows the unreviewed dataset tumour; render masks are never listed.
    assert set(surgeon["t"]) == {"through", "left"} and set(surgeon["m"]) == {"lesion", "unreviewed", "tumour"}
    patient = page.evaluate(
        """async () => {
          const c = window.__capsule, nv = c.state.nv; await c.api.setMode('patient');
          return {t: c.api.visibleTracts(), m: c.api.visibleMasks(),
                  meshes: nv.meshes.filter(m => m.visible).length,
                  meshes3d: c.state.nv3d.meshes.map(m => m.name),
                  volumes: nv.volumes.map(v => v.name),
                  readout: getComputedStyle(document.getElementById('readout')).display,
                  tools: getComputedStyle(document.getElementById('surgeon-panel')).display,
                  save: getComputedStyle(document.getElementById('save-button')).display};
        }"""
    )
    # The fixture has legacy reviewed=true flags without review signatures; they are unsigned and hidden.
    assert patient["t"] == [] and patient["m"] == [], patient
    assert patient["meshes"] == 0, patient
    assert not any("unreviewed" in n or "tumour" in n or "left" in n for n in patient["meshes3d"]), patient
    assert not any("unreviewed" in name for name in patient["volumes"]), patient
    assert patient["readout"] == "none" and patient["tools"] == "none" and patient["save"] == "none", patient
    _assert_clean(evidence)


def test_mask_edit_round_trip_and_brush(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const c = window.__capsule, api = c.api, m = c.state.masks.get('lesion');
          const before = m.data.slice(), ml = m.meta.volume_ml;
          api.editMask('lesion'); syncDrawing();                      // native -> drawing -> native
          let same = before.length === m.data.length; for (let i = 0; same && i < before.length; i++) same = before[i] === m.data[i];
          api.editMask(null);
          const target = [30, 30, 20]; const idx = voxelToIndex(...rasToVoxel(...target));
          const was = c.state.masks.get('unreviewed').data[idx];
          await api.brushAtRAS(...target, false, 3);
          const last = c.state.manifest.masks.at(-1);
          return {same, ml, mlAfter: m.meta.volume_ml, was, now: c.state.masks.get(last.id).data[idx], last: last.id};
        }"""
    )
    assert result["same"], result
    assert result["ml"] == result["mlAfter"], result
    assert result["now"] == 1 and result["was"] == 0, result
    _assert_clean(evidence)


def test_save_round_trip_keeps_tracts_reviews_tour_and_unknown_fields(v2, chromium, v2_capsule: Path, tmp_path: Path) -> None:
    page, _, evidence = v2
    original_manifest, original_blobs = parse_capsule(v2_capsule)
    step = page.evaluate(
        """async () => {
          const api = window.__capsule.api;
          const reviewer = document.getElementById('reviewer-name'); reviewer.value = 'Revisor de teste';
          reviewer.dispatchEvent(new Event('input', {bubbles: true}));
          await api.setTractReviewed('left', true); await api.setMaskReviewed('unreviewed', true);
          api.setLayout('3d'); api.setCamera({azimuth: 200, elevation: 30, zoom: 1.2}); api.setCrosshairRAS(10, 0, 5); api.setClip('coronal');
          return api.addTourStep('Passo de teste', 'Texto do passo');
        }"""
    )
    assert step["view"]["camera"] == {"azimuth": 200, "elevation": 30, "zoom": 1.2}
    assert step["view"]["clip"] == "coronal" and step["view"]["layout"] == "3d"
    with page.expect_download(timeout=60000) as info:
        page.click("#save-button")
    download = info.value
    assert download.suggested_filename == "fixture.v2.capsule.html"
    saved = tmp_path / download.suggested_filename
    download.save_as(saved)
    _assert_clean(evidence)

    manifest, blobs = parse_capsule(saved)
    assert manifest["version"] == original_manifest["version"] + 1
    assert manifest["synthetic_extension"] == original_manifest["synthetic_extension"]
    assert manifest["tracts"][0].get("x_fixture_note") == original_manifest["tracts"][0]["x_fixture_note"]
    assert {t["id"]: t["reviewed"] for t in manifest["tracts"]} == {"through": True, "left": True}
    assert {m["id"]: m["reviewed"] for m in manifest["masks"]}["unreviewed"] is True
    # Render masks survive the save unchanged (contract fields and voxels); the tumour keeps its dataset source.
    render = {m["id"]: m for m in original_manifest["masks"] if m.get("role") == "render"}
    assert len(render) == 3
    for meta in manifest["masks"]:
        if meta["id"] in render:
            assert (meta["role"], meta["for_volume"], meta["source"]) == ("render", render[meta["id"]]["for_volume"], "auto")
            assert blobs[meta["blob"]] == original_blobs[render[meta["id"]]["blob"]]
    tumour = next(m for m in manifest["masks"] if m["id"] == "tumour")
    assert (tumour["source"], tumour["reviewed"], tumour["color"]) == ("dataset", False, "#E4572E")
    for meta in manifest["tracts"]:
        assert blobs[meta["blob"]] == original_blobs[meta["blob"]]
        assert len(read_tck(blobs[meta["blob"]])) == meta["n_streamlines"]
    assert manifest["tour"][-1]["title"] == "Passo de teste"
    assert manifest["tour"][-1]["view"]["camera"]["azimuth"] == 200

    # Reopen the downloaded file in a fresh context and check the restored state and the tour step.
    page2, context2, evidence2 = _open(chromium, saved)
    try:
        reopened = page2.evaluate(
            """async () => {
              const c = window.__capsule; await c.api.setMode('patient');
              const s = c.api.applyTourStep(c.state.manifest.tour.length - 1);
              return {version: c.state.manifest.version, tracts: c.api.visibleTracts(), masks: c.api.visibleMasks(),
                      camera: getCamera(), clip: c.state.clip, layout: c.state.layout, title: s.title,
                      badge: document.getElementById('version-badge').textContent};
            }"""
        )
    finally:
        context2.close()
    assert reopened["version"] == 2 and reopened["badge"] == "versão 2"
    assert set(reopened["tracts"]) == {"left"}  # the legacy through flag has no signature
    assert set(reopened["masks"]) == {"unreviewed"}  # the legacy lesion flag has no signature
    assert reopened["camera"] == {"azimuth": 200, "elevation": 30, "zoom": 1.2}
    assert reopened["clip"] == "coronal" and reopened["layout"] == "3d" and reopened["title"] == "Passo de teste"
    _assert_clean(evidence2)


# ---------------------------------------------------------------------------------------------------------
# Real public capsules (CPTAC-AML head CT; Leipzig CC0 T1 + 12 tracts). Built once per session into tmp from
# the read-only public inputs the demo script uses; skipped when those inputs are not on this machine.
PUBLIC = Path.home() / "case-capsule" / "data" / "public"
LEIPZIG = Path.home() / "tractlab" / "cases" / "demo-leipzig-sub-010005"
LEIPZIG_TRACTS = [
    ("cst_l_motor_pons", "#E4572E"), ("cst_r_motor_pons", "#F2A07B"), ("ifof_l_occ_front", "#4C9BE8"),
    ("ifof_r_occ_front", "#9CC8F3"), ("or_l_meyer", "#59C36A"), ("or_r_meyer", "#A5E0AE"),
    ("uf_l_temp_orb", "#C9A84C"), ("uf_r_temp_orb", "#E6D39E"), ("slf2_l_mfg_ipl", "#B06AD9"),
    ("slf2_r_mfg_ipl", "#D6AEEC"), ("cing_l_ant_post", "#3CC8C8"), ("cing_r_ant_post", "#97E3E3"),
]
SNAP_JS = """const nv = window.__capsule.state.nv3d;  // 3D render instance
  const snap = () => { nv.drawScene(); const gl = nv.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight,
    b = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, b); return b; };
  const diff = (a, b) => { let n = 0; for (let i = 0; i < a.length; i += 4)
    if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 30) n++; return n; };"""


def _cli(*args) -> None:
    result = subprocess.run([sys.executable, "-m", "capsule.cli", *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(f"capsule build failed ({result.returncode}): {result.stderr[-2000:]}")


@pytest.fixture(scope="session")
def real_ct_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = PUBLIC / "CPTAC-AML" / "head-ct"
    if not source.is_dir():
        pytest.skip("public CPTAC-AML head CT not present on this machine")
    output = tmp_path_factory.mktemp("real-ct") / "cptac-aml-head-ct.capsule.html"
    _cli("build", source, "--series", "2", "--label", "CPTAC-AML TC crânio", "--viewer", "v2", "-o", output)
    return output


@pytest.fixture(scope="session")
def real_tract_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bank = LEIPZIG / "tracts" / "bank"
    if not (LEIPZIG / "nifti" / "t1_brain_dwi.nii.gz").is_file() or not bank.is_dir():
        pytest.skip("Leipzig CC0 demo case not present on this machine")
    output = tmp_path_factory.mktemp("real-tracts") / "leipzig-t1-tracts.capsule.html"
    args: list = ["build-nifti", "--volume", f"{LEIPZIG / 'nifti' / 't1_brain_dwi.nii.gz'}:MR:RM T1"]
    for name, color in LEIPZIG_TRACTS:
        args += ["--tract", f"{bank / (name + '.tck')}:{name}:{color}"]
    _cli(*args, "--max-streamlines", "300", "--label", "Leipzig demo (CC0)", "--viewer", "v2", "-o", output)
    return output


def test_ct_3d_preset_drives_the_render_on_load_and_layout_switch(chromium, real_ct_capsule: Path) -> None:
    """Regression: the 3D layout showed 'TC Osso' but rendered with the 40/400 soft-tissue window.
    Round 2: the 3D view is its own instance with a bone ramp; the 2D tiles keep the grey reading window."""
    page, context, evidence = _open(chromium, real_ct_capsule)
    try:
        result = page.evaluate(
            """async () => {
              const api = window.__capsule.api, st = window.__capsule.state;
              %s
              await api.renderReady();
              const layer = () => [st.nv3d.volumes[0].cal_min, st.nv3d.volumes[0].cal_max, st.nv3d.volumes[0].colormap];
              const slice2d = () => [st.nv.volumes[0].cal_min, st.nv.volumes[0].cal_max, st.nv.volumes[0].colormap];
              api.setLayout('3d'); api.setCamera({azimuth: 110, elevation: 15, zoom: 1});
              const onLoad = {preset: st.preset3d, select: document.getElementById('preset3d').value, layer: layer(), slices: slice2d()};
              const bone = snap();
              api.setLayout('2x2'); const slices = slice2d(); api.setLayout('3d');
              const afterSwitch = layer(); const boneAgain = snap();
              await api.setPreset3d('ct-skin'); const skinLayer = layer(), skinSlices = slice2d(); const skin = snap();
              await api.setPreset3d('ct-bone');
              return {onLoad, slices, afterSwitch, skinLayer, skinSlices, boneVsSkin: diff(bone, skin), control: diff(bone, boneAgain),
                      lit: (() => { let n = 0; for (let i = 0; i < bone.length; i += 4) if (bone[i] + bone[i+1] + bone[i+2] > 60) n++; return n; })()};
            }""" % SNAP_JS
        )
    finally:
        context.close()
    assert result["onLoad"]["preset"] == "ct-bone" and result["onLoad"]["select"] == "ct-bone", result
    assert result["onLoad"]["layer"] == [150, 1650, "cc-bone"], result
    assert result["afterSwitch"] == [150, 1650, "cc-bone"], result
    # 2D tiles keep the base layer's grey reading window in every layout and preset (brief item 4).
    for key in ("slices", "skinSlices"):
        assert result[key][2] == "gray" and result[key][:2] == [-160, 240], result
    assert result["onLoad"]["slices"][2] == "gray", result
    assert result["skinLayer"] == [-500, 1500, "cc-skinbone"], result
    assert result["lit"] > 5000, result  # something is rendered
    assert result["boneVsSkin"] > 5000, result  # the preset changes the 3D image (skin vs bone)
    assert result["control"] < 50, result  # same preset twice renders the same frame
    _assert_clean(evidence)


def _foreground_centroid_ras(manifest, blobs):
    meta, data = next(_volume_arrays(manifest, blobs))
    cut = -300.0 if meta["kind"] == "CT" else 0.1 * float(meta["stats"]["p99"])
    k, j, i = np.nonzero(data > cut)
    voxel = np.array([i.mean(), j.mean(), k.mean(), 1.0])
    return (np.array(manifest["grid"]["affine_ras"]) @ voxel)[:3], cut, data, meta


@pytest.mark.parametrize("which", ["phantom", "ct", "tracts"])
def test_initial_crosshair_starts_inside_the_base_volume_foreground(which, request, chromium) -> None:
    capsule = request.getfixturevalue({"phantom": "v2_capsule", "ct": "real_ct_capsule", "tracts": "real_tract_capsule"}[which])
    manifest, blobs = parse_capsule(capsule)
    expected, cut, data, meta = _foreground_centroid_ras(manifest, blobs)
    page, context, evidence = _open(chromium, capsule)
    try:
        got = page.evaluate(
            """() => { const s = window.__capsule.state, c = s.crosshair;
                       return {c, value: window.__capsule.api.valueAtRAS(s.base, ...c),
                               corner: window.__capsule.api.valueAtRAS(s.base, ...s.manifest.grid.affine_ras.slice(0, 3).map(r => r[3]))}; }"""
        )
    finally:
        context.close()
    # Independent numpy centroid (all voxels, not the JS stride-2 sample) agrees within 3 mm.
    assert np.linalg.norm(np.array(got["c"]) - expected) < 3.0, (got, expected.tolist())
    assert got["value"] > cut, got  # the start voxel is anatomy, not air
    assert got["corner"] <= cut, got  # control: the grid corner is air, so the check can fail
    _assert_clean(evidence)


def test_tracts_show_through_the_brain_by_default_and_each_of_12_is_visible(chromium, real_tract_capsule: Path) -> None:
    page, context, evidence = _open(chromium, real_tract_capsule)
    try:
        result = page.evaluate(
            """() => {
              const api = window.__capsule.api, st = window.__capsule.state;
              %s
              api.setLayout('3d'); api.setCamera({azimuth: 110, elevation: 15, zoom: 1});
              const xray = nv.opts.meshXRay, slider = document.getElementById('tract-xray').value;
              const on = snap();
              const perTract = st.manifest.tracts.map(t => { api.setTractVisible(t.id, false); const h = snap();
                                                             api.setTractVisible(t.id, true); return diff(on, h); });
              const again = snap();
              api.setTractXray(0); const off = snap(); api.setTractXray(0.6);
              const radii = [...st.tracts.values()].map(t => [t.mesh.fiberRadius, t.mesh3d.fiberRadius]);
              return {xray, slider, meshes: nv.meshes.length, perTract, control: diff(on, again), xrayEffect: diff(on, off),
                      tubes: st.tractTubes, radii};
            }""" % SNAP_JS
        )
    finally:
        context.close()
    assert result["xray"] == pytest.approx(0.6) and result["slider"] == "60", result
    assert result["meshes"] == 12, result
    assert result["xrayEffect"] > 5000, result  # without x-ray the brain hides most of the tracts
    assert all(n > 100 for n in result["perTract"]), result  # each of the 12 contributes visible pixels
    assert result["control"] < 50, result
    # The suite renders with SwiftShader: software GL keeps 1 px lines in 3D (tubes stall it); 2D is always lines.
    assert result["tubes"] is False and all(r == [0, 0] for r in result["radii"]), result
    _assert_clean(evidence)


# ---------------------------------------------------------------------------------------------------------
# Round 2: render masks (display-only), grey 2D tiles in the 3D layout, dataset tumour in 2D and 3D.
from dev_fixture import TRUTH  # noqa: E402  (fixture ground truth for the synthetic table/fiducial/tumour)

PIX_JS = """
  const snapOf = nv => { nv.drawScene(); const gl = nv.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight,
    b = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, b); return b; };
  const diff = (a, b) => { let n = 0; for (let i = 0; i < a.length; i += 4)
    if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 30) n++; return n; };
  const count = (b, f) => { let n = 0; for (let i = 0; i < b.length; i += 4) if (f(b[i], b[i + 1], b[i + 2])) n++; return n; };
  const renderValue = p => { const img = st.nv3d.volumes[0], v = img.mm2vox(p); return img.getValue(v[0], v[1], v[2]); };
  // Control path: the same render code with the render mask withheld (as if the capsule had none).
  const withoutMask = async id => { const m = st.masks.get(id); st.masks.delete(id); renderCache.clear(); await updateRender();
    return async () => { st.masks.set(id, m); renderCache.clear(); await updateRender(); }; };
"""


def _probe_value(manifest, blobs, volume_id, ras):
    affine = np.array(manifest["grid"]["affine_ras"], dtype=float)
    i, j, k = np.rint(np.linalg.solve(affine, [*ras, 1.0])[:3]).astype(int)
    meta, values = next((m, v) for m, v in _volume_arrays(manifest, blobs) if m["id"] == volume_id)
    return float(values[k, j, i])


def test_ct_render_mask_removes_the_table_in_3d_only(v2, v2_capsule: Path) -> None:
    page, _, evidence = v2
    manifest, blobs = parse_capsule(v2_capsule)
    table_hu = _probe_value(manifest, blobs, "ct", TRUTH["table_probe_ras"])
    assert table_hu == pytest.approx(TRUTH["table_hu"]), table_hu  # the table is in the capsule data
    result = page.evaluate(
        """async ([table, inside]) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          await api.renderReady();
          api.setLayout('3d'); api.setCamera({azimuth: 180, elevation: 0, zoom: 1});
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          const masked = snapOf(st.nv3d), maskedAgain = snapOf(st.nv3d);
          const out = {preset: st.preset3d, mask: st.render3d.mask, readout: api.valueAtRAS('ct', ...table),
                       render: renderValue(table), renderInside: renderValue(inside)};
          const restore = await withoutMask('head');
          out.unmaskedMask = st.render3d.mask; out.unmaskedRender = renderValue(table);
          out.readoutUnmasked = api.valueAtRAS('ct', ...table);
          const unmasked = snapOf(st.nv3d); await restore();
          out.change = diff(masked, unmasked); out.control = diff(masked, maskedAgain);
          out.restored = diff(masked, snapOf(st.nv3d)); out.a2d = st.nv.volumes[0].colormap;
          return out;
        }""" % PIX_JS,
        [list(TRUTH["table_probe_ras"]), [0.0, 0.0, 0.0]],
    )
    assert result["preset"] == "ct-bone" and result["mask"] == "head", result
    assert result["readout"] == pytest.approx(table_hu), result  # 2D/readout keep the original HU
    assert result["readoutUnmasked"] == pytest.approx(table_hu), result
    # 3D copy: the table voxel is background (smoothing near the scalp may leak a few HU).
    assert result["render"] < -900, result
    assert result["renderInside"] > -1000, result  # inside the head the copy keeps tissue
    assert result["unmaskedMask"] is None and result["unmaskedRender"] > 250, result  # control: without the mask the table is bone-bright
    assert result["change"] > 300, result  # and it shows in the 3D image
    assert result["control"] < 50 and result["restored"] < 50, result
    assert result["a2d"] == "gray", result
    _assert_clean(evidence)


def test_mr_fiducial_plate_is_gone_from_both_mr_presets(v2, v2_capsule: Path) -> None:
    """Brief item 3. Real cause on the VS-SEG MR: Leksell N-localizer plates in the image data outside the head."""
    page, _, evidence = v2
    manifest, blobs = parse_capsule(v2_capsule)
    mr = next(m for m in manifest["volumes"] if m["kind"] == "MR")
    plate = _probe_value(manifest, blobs, mr["id"], TRUTH["fiducial_probe_ras"])
    result = page.evaluate(
        """async ([id, probe]) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          api.setBase(id); await api.renderReady();
          api.setLayout('3d'); api.setCamera({azimuth: 270, elevation: 0, zoom: 1});  // patient-left side, where the plate is
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          const out = {preset: st.preset3d, select: document.getElementById('preset3d').value, readout: api.valueAtRAS(id, ...probe)};
          const brain = snapOf(st.nv3d);
          out.brainMask = st.render3d.mask; out.brainRender = renderValue(probe);
          const restore = await withoutMask(st.render3d.mask);
          out.noMask = st.render3d.mask; out.noMaskRender = renderValue(probe); out.brainChange = diff(brain, snapOf(st.nv3d));
          await restore();
          await api.setPreset3d('mr-skin');
          const skin = snapOf(st.nv3d); out.skinMask = st.render3d.mask; out.skinRender = renderValue(probe);
          const restore2 = await withoutMask(st.render3d.mask);
          out.skinChange = diff(skin, snapOf(st.nv3d)); await restore2();
          out.control = diff(skin, snapOf(st.nv3d)); out.a2d = st.nv.volumes[0].colormap;
          return out;
        }""" % PIX_JS,
        [mr["id"], list(TRUTH["fiducial_probe_ras"])],
    )
    assert plate > 0.5 * float(mr["stats"]["p99"]), (plate, mr["stats"])  # the plate is bright in the data
    assert result["preset"] == "mr-brain" and result["select"] == "mr-brain", result  # default when a brain mask exists
    assert result["readout"] == pytest.approx(plate, rel=1e-3), result
    assert result["brainMask"] == f"brain_{mr['id']}" and result["skinMask"] == f"head_{mr['id']}", result
    assert result["brainRender"] == pytest.approx(0, abs=1e-3) and result["skinRender"] == pytest.approx(0, abs=1e-3), result
    # Control: the same render path without a render mask keeps the plate, and the 3D image changes.
    assert result["noMask"] is None and result["noMaskRender"] > 0.5 * plate, result
    assert result["brainChange"] > 300 and result["skinChange"] > 300, result
    assert result["control"] < 50, result
    assert result["a2d"] == "gray", result
    _assert_clean(evidence)


def test_mr_3d_window_sliders_peel_the_render_and_survive_a_tour_step(v2, v2_capsule: Path) -> None:
    """Owner fix (MR card): window the 3D render. r8: the no-mask bright-shell variant of 'RM Cérebro + vasos' is
    gone (the preset needs a pack-time vessel mask), so the sliders are exercised on 'RM Cérebro'."""
    page, _, evidence = v2
    manifest, _ = parse_capsule(v2_capsule)
    mr = next(m for m in manifest["volumes"] if m["kind"] == "MR")
    result = page.evaluate(
        """async (id) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          api.setBase(id); await api.renderReady(); api.setLayout('3d');
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          const out = {select: [...document.getElementById('preset3d').options].map(o => o.value)};
          await api.setPreset3d('mr-brain'); out.mask = st.render3d.mask;
          const img = () => st.nv3d.volumes[0], lo = img().cal_min, hi = img().cal_max; out.range = [lo, hi];
          const before = snapOf(st.nv3d), sliders = () => ['window3d-min', 'window3d-max'].map(x => document.getElementById(x).value);
          out.slidersBefore = sliders();
          out.set = api.setWindow3d(lo + 0.45 * (hi - lo), hi);
          out.cal = [img().cal_min, img().cal_max]; out.slidersAfter = sliders(); out.peel = diff(before, snapOf(st.nv3d));
          out.step = api.addTourStep('janela 3D', '').view.window3d;
          await api.setPreset3d('mr-skin'); await api.setPreset3d('mr-brain');
          out.reset = [st.window3d, img().cal_min, img().cal_max]; out.back = diff(before, snapOf(st.nv3d));
          api.applyTourStep(st.manifest.tour.length - 1); await api.renderReady(); out.replayed = [st.window3d, img().cal_min];
          return out;
        }""" % PIX_JS,
        mr["id"],
    )
    assert "mr-vessels" not in result["select"], result  # no vessel mask in the fixture
    assert result["mask"] == f"brain_{mr['id']}", result
    lo, hi = result["range"]
    assert result["cal"] == pytest.approx([lo + 0.45 * (hi - lo), hi]), result
    assert result["slidersAfter"][0] != result["slidersBefore"][0], result
    assert result["peel"] > 300, result  # raising the threshold changes the 3D image
    assert result["step"]["min"] == pytest.approx(lo + 0.45 * (hi - lo)), result
    assert result["reset"] == [None, pytest.approx(lo), pytest.approx(hi)] and result["back"] < 50, result
    assert result["replayed"][0] is not None and result["replayed"][1] == pytest.approx(lo + 0.45 * (hi - lo)), result
    _assert_clean(evidence)


def test_mr_vessels_preset_draws_a_pack_time_vessel_mask_as_a_surface(v2, v2_capsule: Path) -> None:
    """r6: with a vessel render mask the preset keeps the render to the brain and draws the vessels as a mesh;
    r8: without one the preset is not offered and the view falls back to 'RM Cérebro'."""
    page, _, evidence = v2
    manifest, _ = parse_capsule(v2_capsule)
    mr = next(m for m in manifest["volumes"] if m["kind"] == "MR")
    result = page.evaluate(
        """async (id) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          api.setBase(id); await api.renderReady(); api.setLayout('3d');
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          const names = () => st.nv3d.meshes.map(m => m.name).filter(n => n.startsWith('vessels-'));
          const out = {};
          const offered = () => presets3dFor(st.volumes.get(id).meta).map(([k]) => k);
          await api.setPreset3d('mr-brain'); out.shell = offered().includes('mr-vessels');
          const before = snapOf(st.nv3d);
          // Synthetic vessel: a 3-voxel-wide rod along x through the brain centre.
          const [nx, ny, nz] = st.manifest.grid.dims, brain = st.masks.get('brain_' + id).data, data = new Uint8Array(brain.length);
          const j0 = ny >> 1, k0 = nz >> 1;
          for (let i = 0; i < nx; i++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
            const n = ((k0 + dk) * ny + j0 + dj) * nx + i; if (brain[n]) data[n] = 1; }
          const meta = {id: 'vessels_' + id, blob: 'mask_vessels_' + id, role: 'render', for_volume: id, label: 'Vasos', color: '#3A60D6'};
          st.manifest.masks.push(meta); st.masks.set(meta.id, {meta, data, version: 0, nvimg: null});
          renderCache.clear(); await api.setPreset3d('mr-brain'); await api.setPreset3d('mr-vessels');
          out.mesh = [names(), st.nv3d.volumes[0].colormap]; out.diff = diff(before, snapOf(st.nv3d));
          st.manifest.masks.pop(); st.masks.delete(meta.id);
          renderCache.clear(); api.setBase(st.manifest.volumes.find(v => v.id !== id).id); api.setBase(id); await api.renderReady();
          out.after = [offered().includes('mr-vessels'), st.preset3d, names()];
          return out;
        }""" % PIX_JS,
        mr["id"],
    )
    assert result["shell"] is False, result  # r8: never offered without a vessel mask
    assert result["mesh"] == [[f"vessels-vessels_{mr['id']}"], "cc-brain"], result
    assert result["diff"] > 300, result  # the rod and brain-only render change the 3D image
    assert result["after"] == [False, "mr-brain", []], result
    _assert_clean(evidence)


def test_ct_angio_preset_exists_only_with_a_vessel_mask_and_shows_it_through_bone(v2) -> None:
    """r7: 'TC Angio' is offered, and becomes the CT default, only when the capsule carries a subtraction vessel
    mask; the vessels are a red surface seen through the bone render (mesh x-ray)."""
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          api.setBase('ct'); await api.renderReady(); api.setLayout('3d');
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          const opts = () => [...document.getElementById('preset3d').options].map(o => o.value);
          const names = () => st.nv3d.meshes.map(m => m.name).filter(n => n.startsWith('vessels-'));
          const out = {before: [opts(), default3d(st.volumes.get('ct').meta)]};
          const head = st.masks.get(renderMaskFor('ct', 'head').id).data, [nx, ny, nz] = st.manifest.grid.dims;
          const data = new Uint8Array(head.length), j0 = ny >> 1, k0 = nz >> 1;
          for (let i = 0; i < nx; i++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
            const n = ((k0 + dk) * ny + j0 + dj) * nx + i; if (head[n]) data[n] = 1; }
          const meta = {id: 'vessels', blob: 'mask_vessels', role: 'render', for_volume: 'ct', label: 'Vasos', color: '#D8433A'};
          st.manifest.masks.push(meta); st.masks.set(meta.id, {meta, data, version: 0, nvimg: null});
          api.setBase(st.manifest.volumes[1].id); api.setBase('ct'); await api.renderReady();
          out.defaultWith = st.preset3d;
          await api.setPreset3d('ct-bone'); const shot0 = snapOf(st.nv3d);
          await api.setPreset3d('ct-vessels');
          const mesh = st.nv3d.meshes.find(m => m.name.startsWith('vessels-'));
          out.with = [opts(), st.preset3d, names(), st.nv3d.volumes[0].colormap, st.nv3d.opts.meshXRay, mesh && [...mesh.rgba255].slice(0, 3)];
          out.diff = diff(shot0, snapOf(st.nv3d));
          await api.setPreset3d('ct-bone'); out.boneXray = [names(), st.nv3d.opts.meshXRay === st.tractXray];
          st.manifest.masks.pop(); st.masks.delete(meta.id); renderCache.clear();
          api.setBase(st.manifest.volumes[1].id); api.setBase('ct'); await api.renderReady();
          out.after = [opts(), st.preset3d];
          return out;
        }""" % PIX_JS
    )
    assert "ct-vessels" not in result["before"][0] and result["before"][1] == "ct-bone", result
    assert "ct-vessels" in result["with"][0], result
    assert result["with"][1:5] == ["ct-vessels", ["vessels-vessels"], "cc-bone", 0.75], result
    assert result["with"][5] == [216, 67, 58], result
    assert result["defaultWith"] == "ct-vessels", result
    assert result["diff"] > 300, result  # the red surface changes the 3D image
    assert result["boneXray"] == [[], True], result
    assert "ct-vessels" not in result["after"][0] and result["after"][1] == "ct-bone", result
    _assert_clean(evidence)


def test_2d_tiles_in_the_3d_layout_are_grey(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          await api.renderReady();
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of api.visibleMasks()) api.setMaskVisible(m, false);
          st.nv.opts.crosshairWidth = 0;
          const tinted = b => count(b, (r, g, bl) => Math.max(r, g, bl) - Math.min(r, g, bl) > 30);
          const out = {};
          for (const [base, preset] of [['ct', 'ct-bone'], ['ct', 'ct-skin'], [st.manifest.volumes[1].id, 'mr-brain']]) {
            api.setBase(base); await api.setPreset3d(preset); api.setLayout('3d');
            out[preset] = {cm: st.nv.volumes[0].colormap, tinted: tinted(snapOf(st.nv)), grey: count(snapOf(st.nv), (r, g, b) => r > 40 && r === g && g === b)};
          }
          // Known-positive control: a tinted colormap on the same 2D tiles is detected.
          const v = st.nv.volumes[0], lo = v.cal_min, hi = v.cal_max; v.colormap = 'winter'; v.cal_min = lo; v.cal_max = hi; st.nv.updateGLVolume();
          out.control = tinted(snapOf(st.nv));
          return out;
        }""" % PIX_JS
    )
    for preset in ("ct-bone", "ct-skin", "mr-brain"):
        assert result[preset]["cm"] == "gray", result
        assert result[preset]["grey"] > 5000, result
        assert result[preset]["tinted"] < 50, result
    assert result["control"] > 5000, result
    _assert_clean(evidence)


def test_dataset_tumour_is_solid_in_3d_and_filled_outlined_in_2d(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """async (center) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          await api.renderReady();
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          for (const m of ['lesion', 'unreviewed']) api.setMaskVisible(m, false);
          const orange = (r, g, b) => r > 170 && g > 50 && g < 130 && b < 90 && r - g > 80;
          const solid = (r, g, b) => Math.abs(r - 228) < 20 && Math.abs(g - 87) < 20 && Math.abs(b - 46) < 20;
          api.setCrosshairRAS(...center); api.setLayout('axial'); st.nv.opts.crosshairWidth = 0;
          const on2d = snapOf(st.nv);
          const out = {orange2d: count(on2d, orange), solid2d: count(on2d, solid)};
          api.setMaskVisible('tumour', false); const off2d = snapOf(st.nv);
          out.orange2dHidden = count(off2d, orange); out.solid2dHidden = count(off2d, solid);
          // Translucent fill: pixels the mask changed that are tinted toward its colour but not the pure colour
          // (anatomy still shows through). Opaque fill would make nearly every changed pixel "solid".
          let fill = 0, changed = 0;
          for (let i = 0; i < on2d.length; i += 4) {
            const r = on2d[i], g = on2d[i + 1], b = on2d[i + 2];
            if (Math.abs(r - off2d[i]) + Math.abs(g - off2d[i + 1]) + Math.abs(b - off2d[i + 2]) <= 30) continue;
            changed++; if (!solid(r, g, b) && r > g + 25 && r > b + 35) fill++;
          }
          out.fill2d = fill; out.changed2d = changed;
          api.setMaskVisible('tumour', true);
          api.setLayout('3d'); api.setCamera({azimuth: 270, elevation: 0, zoom: 1});
          const on3d = snapOf(st.nv3d), again = snapOf(st.nv3d);
          out.meshes = st.nv3d.meshes.map(m => m.name);
          out.orange3d = count(on3d, orange);
          api.setMaskVisible('tumour', false); const off3d = snapOf(st.nv3d);
          out.meshesHidden = st.nv3d.meshes.map(m => m.name);
          out.orange3dHidden = count(off3d, orange); out.change3d = diff(on3d, off3d); out.control = diff(on3d, again);
          api.setMaskVisible('tumour', true);
          const legend = () => [...document.querySelectorAll('#mask-legend .legend-row')].map(r => [r.textContent, !!r.querySelector('.unreviewed-badge')]);
          const row = () => document.querySelector('#mask-list [data-mask="tumour"]') || [...document.querySelectorAll('#mask-list > *')].find(e => e.textContent.includes('Schwannoma'));
          out.legend = legend(); out.listBadge = !!(row() && row().querySelector('.unreviewed-badge'));
          await api.setMode('patient');
          out.patient = {masks: api.visibleMasks(), meshes: st.nv3d.meshes.map(m => m.name), legend: legend()};
          return out;
        }""" % PIX_JS,
        list(TRUTH["tumour_center_ras"]),
    )
    # 2D: coloured fill plus a solid outline, both gone when hidden (control).
    assert result["changed2d"] > 150 and result["solid2d"] > 30, result
    assert result["fill2d"] > 100 and result["fill2d"] > result["solid2d"], result  # fill translucent, rim opaque
    assert result["orange2dHidden"] < 10 and result["solid2dHidden"] < 5, result
    # 3D: a tumour mesh, drawn in its colour through the head, removed when hidden.
    assert any("tumour" in n for n in result["meshes"]), result
    assert not any("tumour" in n for n in result["meshesHidden"]), result
    assert result["orange3d"] > 150 and result["orange3dHidden"] < 20, result
    assert result["change3d"] > 150 and result["control"] < 50, result
    # Surgeon: legend and mask list carry the "não revisado" badge; patient: no tumour anywhere.
    tumour_rows = [r for r in result["legend"] if "Schwannoma" in r[0]]
    assert tumour_rows and tumour_rows[0][1] and "não revisado" in tumour_rows[0][0], result
    assert result["listBadge"], result
    assert "tumour" not in result["patient"]["masks"], result
    assert not any("tumour" in n for n in result["patient"]["meshes"]), result
    assert not any("Schwannoma" in r[0] for r in result["patient"]["legend"]), result
    _assert_clean(evidence)


TILES_JS = """(naive) => {
  const c = window.__capsule, nv = c.state.nv, g = c.state.manifest.grid;
  c.api.setLayout('2x2');
  if (naive) nv.setCustomLayout([[nv.sliceTypeAxial, [0, 0, .5, .5]], [nv.sliceTypeCoronal, [.5, 0, .5, .5]],
                                 [nv.sliceTypeSagittal, [0, .5, .5, .5]]].map(([sliceType, position]) => ({sliceType, position})));
  nv.drawScene();
  const W = nv.gl.drawingBufferWidth, H = nv.gl.drawingBufferHeight, cells = [[0, 0], [.5, 0], [0, .5]];
  const ext = g.dims.map((d, i) => d * g.spacing_mm[i]), fov = [[ext[0], ext[1]], [ext[0], ext[2]], [ext[1], ext[2]]];
  return nv.screenSlices.map((s, i) => { const [l, t, w, h] = s.leftTopWidthHeight, [cx, cy] = cells[i];
    return {type: s.axCorSag, sx: w / fov[s.axCorSag][0], sy: h / fov[s.axCorSag][1],
            fovMM: Array.from(s.fovMM).map(Math.abs), grid: fov[s.axCorSag],
            margin: Math.min((l - cx * W) / W, (t - cy * H) / H, (cx * W + W / 2 - l - w) / W, (cy * H + H / 2 - t - h) / H)}; });
}"""


def _assert_tiles_fit(tiles) -> None:
    assert [t["type"] for t in tiles] == [0, 1, 2], tiles
    scales = [t["sx"] for t in tiles]
    for t in tiles:
        assert abs(t["sx"] - t["sy"]) / t["sx"] < 0.01, t                     # aspect kept: nothing stretched
        assert all(abs(a - b) < 0.5 for a, b in zip(t["fovMM"], t["grid"])), t  # the whole grid extent is on screen
        assert t["margin"] >= 0.015, t                                          # inside its cell, off the edges
    assert (max(scales) - min(scales)) / min(scales) < 0.01, scales            # one mm scale for all three tiles


def test_2x2_tiles_show_the_whole_slice_at_one_scale(v2) -> None:
    """Round 3: the fractional custom layout fitted each slice to its own cell, so the same head was drawn
    at different magnifications (CQ500: 1.68 / 2.29 / 2.17 px/mm) flush against the cell edges."""
    page, _, evidence = v2
    _assert_tiles_fit(page.evaluate(TILES_JS, False))
    # Known-positive control on the same measurement: the naive quarter-cell layout must fail the shared-scale check.
    naive = page.evaluate(TILES_JS, True)
    scales = [t["sx"] for t in naive]
    assert (max(scales) - min(scales)) / min(scales) > 0.05 or min(t["margin"] for t in naive) < 0.015, naive
    page.evaluate("() => window.__capsule.api.setLayout('2x2')")
    # Resize path: the cells are recomputed from the new canvas shape.
    page.set_viewport_size({"width": 1000, "height": 1100})
    page.wait_for_timeout(400)
    _assert_tiles_fit(page.evaluate(TILES_JS, False))
    _assert_clean(evidence)


# ---------------------------------------------------------------------------------------------------------
# Anatomy panel: synthetic automatic label map (dev_fixture.ANATOMY_LABELS).
def test_anatomy_panel_defaults_toggles_readout_patient_gate_and_save_round_trip(v2, chromium, v2_capsule: Path, tmp_path: Path) -> None:
    from dev_fixture import ANATOMY_LABELS

    page, _, evidence = v2
    original_manifest, original_blobs = parse_capsule(v2_capsule)
    item = original_manifest["anatomy"][0]
    nx, ny, nz = original_manifest["grid"]["dims"]
    labels = np.frombuffer(original_blobs[item["blob"]], dtype=np.uint8).reshape(nz, ny, nx)
    affine = np.array(original_manifest["grid"]["affine_ras"], dtype=float)
    # Probe the right lateral ventricle centre with an independent voxel lookup (Python), then ask the viewer.
    centre = next(c for v, *_rest, c, _s in ANATOMY_LABELS if v == 2)
    i, j, k = np.rint(np.linalg.solve(affine[:3, :3], np.array(centre) - affine[:3, 3])).astype(int)
    assert labels[k, j, i] == 2
    result = page.evaluate(
        """async ([x, y, z]) => {
          const c = window.__capsule, api = c.api, st = c.state, names = () => st.nv3d.meshes.map(m => m.name);
          const out = {visible: api.visibleAnatomy(), layers: st.nv.volumes.map(v => v.name), meshes: names(),
            groups: [...document.querySelectorAll('#anatomy-list .anatomy-grp')].map(g => [g.dataset.group, g.querySelectorAll('.anatomy-label').length]),
            badge: !!document.querySelector('#anatomy-list .unreviewed-badge')};
          api.setCrosshairRAS(x, y, z); out.readout = document.getElementById('readout').textContent;
          out.at = api.anatomyAtRAS(x, y, z).map(a => a.value);
          const e = st.anatomy.get('anat_mr'), n = voxelToIndex(...rasToVoxel(x, y, z)); out.fill = e.fill.img[n];
          api.setAnatomyVisible('anat_mr', 2, false); out.fillHidden = e.fill.img[n]; out.hidden = api.visibleAnatomy(); out.meshesHidden = names();
          document.querySelector('#anatomy-list [data-group="Vasos"] .anatomy-grp-head .eye').click();
          out.afterClick = api.visibleAnatomy(); out.meshesVessels = names();
          api.setAnatomyOpacity('Vasos', 0.3); out.opacity = st.nv3d.meshes.find(m => m.name === 'anatomy-anat_mr:4').opacity;
          await api.setMode('patient'); out.patient = api.visibleAnatomy(); out.patientLayers = st.nv.volumes.map(v => v.name); out.patientMeshes = names();
          await api.setMode('surgeon');
          const reviewer = document.getElementById('reviewer-name'); reviewer.value = 'Revisor de teste';
          reviewer.dispatchEvent(new Event('input', {bubbles: true}));
          await api.setAnatomyReviewed('anat_mr', true);
          out.badgeAfter = !!document.querySelector('#anatomy-list .unreviewed-badge');
          await api.setMode('patient'); out.patientReviewed = api.visibleAnatomy(); await api.setMode('surgeon');
          return out;
        }""",
        list(centre),
    )
    # Default: only the ventricles (both sides of the paired structure) are shown, in 2D and 3D.
    assert result["visible"] == ["anat_mr:1", "anat_mr:2"], result
    assert {"anatomy-anat_mr-fill.nii", "anatomy-anat_mr-rim.nii"} <= set(result["layers"]), result
    assert {"anatomy-anat_mr:1", "anatomy-anat_mr:2"} <= set(result["meshes"]) and "anatomy-anat_mr:3" not in result["meshes"], result
    assert result["groups"] == [["Ventrículos", 2], ["Tronco e cerebelo", 1], ["Vasos", 1], ["Nervos e órbita", 1]], result
    assert result["badge"] and not result["badgeAfter"], result
    assert result["at"] == [2] and "Anatomia: Ventrículo lateral direito" in result["readout"], result
    assert result["fill"] == 2 and result["fillHidden"] == 0, result
    assert result["hidden"] == ["anat_mr:1"] and "anatomy-anat_mr:2" not in result["meshesHidden"], result
    assert result["afterClick"] == ["anat_mr:1", "anat_mr:4"] and "anatomy-anat_mr:4" in result["meshesVessels"], result
    assert abs(result["opacity"] - 0.3) < 1e-9, result
    # Patient mode hides the unreviewed item entirely; once reviewed, it shows the surgeon's selection.
    assert result["patient"] == [] and not any("anatomy" in n for n in result["patientLayers"] + result["patientMeshes"]), result
    assert result["patientReviewed"] == ["anat_mr:1", "anat_mr:4"], result

    with page.expect_download(timeout=60000) as info:
        page.click("#save-button")
    saved = tmp_path / info.value.suggested_filename
    info.value.save_as(saved)
    _assert_clean(evidence)
    manifest, blobs = parse_capsule(saved)
    # Byte-for-byte except `reviewed`: every other field and the label blob are unchanged.
    signed = manifest["anatomy"][0].pop("review")
    assert signed["by"] == "Revisor de teste" and signed["blob_sha256"] == hashlib.sha256(original_blobs[item["blob"]]).hexdigest()
    expected = [dict(it, reviewed=True) for it in original_manifest["anatomy"]]
    for it in expected: it.pop("blob_sha256", None)
    for it in manifest["anatomy"]: it.pop("blob_sha256", None)
    assert manifest["anatomy"] == expected
    assert blobs[item["blob"]] == original_blobs[item["blob"]]
    # Literal bytes too: the gzip+base64 script text is carried unchanged (raw HTML compare, no decoding).
    pattern = re.compile(r'<script id="capsule-blob-' + item["blob"] + r'"[^>]*>([^<]*)</script>')
    raw = [pattern.findall(p.read_text(encoding="utf-8")) for p in (v2_capsule, saved)]
    assert len(raw[0]) == 1 and raw[0] == raw[1]

    page2, context2, evidence2 = _open(chromium, saved)
    try:
        reopened = page2.evaluate(
            """async () => { const api = window.__capsule.api; await api.setMode('patient');
              return {visible: api.visibleAnatomy(), reviewed: window.__capsule.state.manifest.anatomy[0].reviewed}; }"""
        )
    finally:
        context2.close()
    # Reviewed now, so the patient sees the default selection (visibility toggles are session state, not saved).
    assert reopened == {"visible": ["anat_mr:1", "anat_mr:2"], "reviewed": True}, reopened
    _assert_clean(evidence2)


def _corridor_truth(manifest, blobs, mask_id, entry, target, radius):
    """Independent Python corridor for one mask: crossing by a dense 0.01 mm walk, contact by brute force over voxels."""
    meta = next(m for m in manifest["masks"] if m["id"] == mask_id)
    nx, ny, nz = manifest["grid"]["dims"]
    mask = np.frombuffer(blobs[meta["blob"]], dtype=np.uint8).reshape(nz, ny, nx)
    affine = np.array(manifest["grid"]["affine_ras"], dtype=float)
    entry, target = np.array(entry, float), np.array(target, float)
    length = float(np.linalg.norm(target - entry))
    w = (target - entry) / length
    depths = np.arange(0.0, length + 1e-9, 0.01)
    ijk = np.rint(np.linalg.solve(affine[:3, :3], (entry + depths[:, None] * w - affine[:3, 3]).T).T).astype(int)
    ok = np.all((ijk >= 0) & (ijk < [nx, ny, nz]), axis=1)
    inside = np.zeros(len(depths), bool)
    inside[ok] = mask[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]] > 0
    crossing = float(depths[np.argmax(inside)]) if inside.any() else None
    k, j, i = np.nonzero(mask)
    centres = (affine[:3, :3] @ np.stack([i, j, k]).astype(float)).T + affine[:3, 3]
    half = 0.5 * np.linalg.norm(affine[:3, :3], axis=0).min()
    rel = centres - entry
    tu = rel @ w
    closest = rel - np.clip(tu, 0, length)[:, None] * w
    dist = np.linalg.norm(closest, axis=1)
    near = dist <= radius + half
    perp2 = np.maximum(0.0, (rel * rel).sum(1) - tu * tu)
    contact = np.clip(tu - np.sqrt(np.maximum(0.0, (radius + half) ** 2 - perp2)), 0, length)
    return {"crossing": crossing, "contact": float(contact[near].min()) if near.any() else None,
            "min_distance": float(max(0.0, dist.min() - half))}


def test_trajectory_corridor_lists_the_phantom_lesion_and_not_a_path_20mm_away(v2, v2_capsule: Path) -> None:
    page, _, evidence = v2
    manifest, blobs = parse_capsule(v2_capsule)
    lesion = next(m for m in manifest["masks"] if m["id"] == "lesion")
    x, y, z = LESION_RAS
    entry, target, control_entry, control_target = [x, y, 60.0], [x, y, z], [x + 20, y, 60.0], [x + 20, y, z]
    through = next(t for t in manifest["tracts"] if t["id"] == "through")
    line = read_tck(blobs[through["blob"]])[0]
    tract_point = np.asarray(line[len(line) // 2], dtype=float).tolist()
    tract_entry, tract_target = [tract_point[0], tract_point[1], 60.0], tract_point
    tract_control_entry, tract_control_target = [tract_point[0] + 30, tract_point[1], 60.0], [tract_point[0] + 30, tract_point[1], tract_point[2]]
    result = page.evaluate(
        """([e, t, ce, ct, te, tt, tce, tct]) => { const api = window.__capsule.api, out = {};
          out.ann = api.addTrajectory(e, t, {radius: 5}); out.report = api.corridorReport();
          out.panel = [...document.querySelectorAll('#corridor-report .corridor-row')].map(r => [r.dataset.id, r.textContent]);
          api.addTrajectory(ce, ct, {radius: 5}); out.control = api.corridorReport();
          out.controlPanel = document.querySelectorAll('#corridor-report .corridor-row').length;
          // Editing a mask on the same path must refresh the (cached) report.
          return api.brushAtRAS(ct[0], ct[1], ct[2], false, 3).then(() => { out.edited = api.corridorReport();
            return api.brushAtRAS(ct[0], ct[1], ct[2], true, 3); }).then(() => {
              out.erased = api.corridorReport(); api.addTrajectory(te, tt, {radius: 5}); out.tractReport = api.corridorReport();
              const tractRow = out.tractReport.find(r => r.id === 'through');
              out.tractPanel = {text: document.querySelector('#corridor-report [data-id="through"]')?.textContent || '',
                badge: !!document.querySelector('#corridor-report [data-id="through"] .unreviewed-badge')};
              api.addTrajectory(tce, tct, {radius: 5}); out.tractControl = api.corridorReport();
              api.addTrajectory(te, tt, {radius: 5}); api.setTractVisible('through', false); out.hiddenTractReport = api.corridorReport();
              api.setTractVisible('through', true);
              return out; }); }""",
        [entry, target, control_entry, control_target, tract_entry, tract_target, tract_control_entry, tract_control_target],
    )
    ann = result["ann"]
    assert ann["type"] == "trajectory" and ann["points_ras"] == [entry, target] and ann["corridor_radius_mm"] == 5, ann
    rows = {r["id"]: r for r in result["report"]}
    assert "lesion" in rows, result["report"]
    row = rows["lesion"]
    truth = _corridor_truth(manifest, blobs, "lesion", entry, target, 5)
    # The path enters the 10 mm phantom sphere 46 mm below the entry (60 - (4 + 10)); voxelised, within 1 mm.
    analytic_crossing = 60.0 - (z + 10.0)
    assert row["min_distance_mm"] == 0 and row["crossing_mm"] is not None, row
    assert abs(row["crossing_mm"] - analytic_crossing) <= 1.0, (row, analytic_crossing)
    assert abs(row["crossing_mm"] - truth["crossing"]) <= 0.51, (row, truth)
    assert abs(row["first_contact_mm"] - truth["contact"]) <= 0.05, (row, truth)
    assert row["first_contact_mm"] <= row["crossing_mm"], row
    # Report order is the first-contact depth; the panel shows the lesion in pt-BR.
    assert [r["first_contact_mm"] for r in result["report"]] == sorted(r["first_contact_mm"] for r in result["report"])
    panel = dict(result["panel"])
    assert lesion["label"] in panel["lesion"] and "atravessa a" in panel["lesion"], panel
    # Control: the same path 20 mm lateral stays > 5 mm from the lesion (Python agrees) and must not list it.
    control_truth = _corridor_truth(manifest, blobs, "lesion", control_entry, control_target, 5)
    assert control_truth["contact"] is None and control_truth["min_distance"] > 5, control_truth
    assert "lesion" not in {r["id"] for r in result["control"]}, result["control"]
    assert result["controlPanel"] == len(result["control"])
    control_ids = {r["id"] for r in result["control"]}
    painted = [r for r in result["edited"] if r["id"] not in control_ids or r["min_distance_mm"] == 0]
    assert any(r["min_distance_mm"] == 0 for r in painted), (result["control"], result["edited"])
    assert {r["id"]: r["min_distance_mm"] for r in result["erased"]} == {r["id"]: r["min_distance_mm"] for r in result["control"]}
    tract_rows = [r for r in result["tractReport"] if r["kind"] == "tract"]
    through_row = next((r for r in tract_rows if r["id"] == "through"), None)
    assert through_row is not None and through_row["min_distance_mm"] <= 2.5, result["tractReport"]
    assert through_row["reviewed"] is False and through_row["visible"] is True and through_row["group"] == "Tratos", through_row
    assert "Tratos" in result["tractPanel"]["text"] and result["tractPanel"]["badge"] is True, result["tractPanel"]
    assert not [r for r in result["tractControl"] if r["kind"] == "tract"], result["tractControl"]
    hidden = next((r for r in result["hiddenTractReport"] if r["id"] == "through"), None)
    assert hidden is not None and hidden["visible"] is False, result["hiddenTractReport"]
    row_keys = {"kind", "id", "name", "color", "group", "min_distance_mm", "first_contact_mm", "crossing_mm", "reviewed", "visible"}
    assert all(set(r) == row_keys for r in [*result["report"], *result["tractReport"], *result["tractControl"]]), result
    _assert_clean(evidence)


def test_trajectory_corridor_lists_a_synthetic_vessel_render_mask(chromium, v2_corridor_vessel_capsule: Path) -> None:
    page, context, evidence = _open(chromium, v2_corridor_vessel_capsule)
    try:
        result = page.evaluate(
            """async () => {
              const c = window.__capsule, api = c.api, st = c.state, mr = st.manifest.volumes.find(v => v.kind === 'MR').id;
              api.setBase(mr); await api.renderReady(); await api.setPreset3d('mr-vessels');
              const entry = [-30, 45, 20], target = [30, 45, 20];
              api.addTrajectory(entry, target, {radius: 5});
              const report = api.corridorReport(), row = report.find(r => r.id.startsWith('vessels'));
              return {row, keys: report.map(r => Object.keys(r).sort()), panel:
                {text: document.querySelector('#corridor-report [data-id^="vessels"]')?.textContent || '',
                 badge: !!document.querySelector('#corridor-report [data-id^="vessels"] .unreviewed-badge')}};
            }"""
        )
    finally:
        context.close()
    row = result["row"]
    assert row and row["kind"] == "vessels" and row["crossing_mm"] is not None, result
    assert row["reviewed"] is False and row["group"] == "Vasos", row
    assert "Vasos" in result["panel"]["text"] and result["panel"]["badge"] is True, result["panel"]
    expected = sorted({"kind", "id", "name", "color", "group", "min_distance_mm", "first_contact_mm", "crossing_mm", "reviewed", "visible"})
    assert all(keys == expected for keys in result["keys"]), result["keys"]
    _assert_clean(evidence)


def test_surgeons_eye_view_centres_the_target_and_looks_from_the_entry(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """async (target) => { const c = window.__capsule, api = c.api, st = c.state, nv = st.nv3d; api.setLayout('3d');
          for (const t of st.manifest.tracts) api.setTractVisible(t.id, false);
          api.setMaskVisible('lesion', false); api.setMaskVisible('tumour', false);
          // r8 orientation cube has a green face; hide it while counting the green marker.
          const green = () => { const gl = nv.gl, W = gl.canvas.width, H = gl.canvas.height, cube = nv.opts.isOrientCube;
            nv.opts.isOrientCube = false; nv.drawScene(); nv.opts.isOrientCube = cube;
            const px = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px); let n = 0, sx = 0, sy = 0;
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 4;
              if (px[i + 1] > 120 && px[i + 1] > px[i] + 60 && px[i + 1] > px[i + 2] + 60) { n++; sx += x; sy += H - 1 - y } }
            return {n, cx: n ? sx / n / W : null, cy: n ? sy / n / H : null}; };
          // NDC depth of a point (NiiVue draws with depthFunc LEQUAL: the smaller value is nearer the camera).
          const ndc = p => { const m = nv.calculateMvpMatrix(null, [0, 0, nv.gl.canvas.width, nv.gl.canvas.height], nv.scene.renderAzimuth, nv.scene.renderElevation)[0];
            return (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / (m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]); };
          const entry = [target[0] - 30, target[1] + 22, target[2] + 28], out = {};
          const a = api.addTrajectory(entry, target); out.view = api.surgeonView(true);
          out.active = document.getElementById('surgeon-view').classList.contains('active');
          out.centred = green(); out.depth = [ndc(entry), ndc(target)];
          const pos = nv.position; nv.position = null; out.uncentred = green(); nv.position = pos;
          // The pan survives a scene change (mesh set changes move NiiVue's pivot).
          api.setMaskVisible('lesion', true); api.setMaskVisible('lesion', false); out.afterMeshChange = green();
          // Reversed path: the camera must move to the other side.
          api.addTrajectory(target.map((v, i) => 2 * v - entry[i]), target); api.surgeonView(true);
          out.reversed = [ndc(entry), ndc(target)]; out.reversedCentred = green();
          api.setCamera({azimuth: 120, elevation: 15}); out.afterSetCamera = {on: st.surgeonView, position: nv.position};
          return out; }""",
        list(MARKER_RAS),
    )
    for key in ("centred", "afterMeshChange", "reversedCentred"):
        g = result[key]
        assert g["n"] > 50 and abs(g["cx"] - 0.5) < 0.03 and abs(g["cy"] - 0.5) < 0.03, (key, result)
    # Positive control: without the pan, the same camera angles leave the marker visibly off-centre.
    u = result["uncentred"]
    assert u["n"] > 50 and max(abs(u["cx"] - 0.5), abs(u["cy"] - 0.5)) > 0.05, result
    # The camera is beyond the entry: the entry is nearer than the target, and the reverse for the reversed path.
    assert result["depth"][0] < result["depth"][1] and result["reversed"][0] > result["reversed"][1], result
    assert result["active"] and result["afterSetCamera"] == {"on": False, "position": None}, result
    _assert_clean(evidence)


def test_trajectory_tool_clicks_draw_in_2d_and_3d_and_survive_save(v2, chromium, tmp_path: Path) -> None:
    page, _, evidence = v2
    x, y, z = LESION_RAS
    page.evaluate("([x, y, z]) => { const api = window.__capsule.api; api.setCrosshairRAS(x, y, z); api.setTool('trajectory'); }", [x, y, z])
    # Target the axial tile: canvas pixels for two in-plane RAS points, from NiiVue's own tile geometry.
    pts = page.evaluate(
        """([a, b]) => { const st = window.__capsule.state, s = st.nv.screenSlices.find(t => t.axCorSag === 0);
          const r = st.nv.canvas.getBoundingClientRect(), dpr = st.nv.canvas.width / r.width;
          return [a, b].map(p => { const q = sliceProject(s, p); return [r.left + q.x / dpr, r.top + q.y / dpr]; }); }""",
        [[x + 25, y + 30, z], [x, y, z]],
    )
    for px, py in pts:
        page.mouse.click(px, py)
    state = page.evaluate(
        """() => { const st = window.__capsule.state, ann = st.manifest.annotations.filter(a => a.type === 'trajectory');
          const c = document.getElementById('traj-overlay'), ctx = c.getContext('2d'), d = ctx.getImageData(0, 0, c.width, c.height).data;
          let gold = 0; for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 200 && Math.abs(d[i] - 0xC9) < 12 && Math.abs(d[i + 1] - 0xA8) < 12 && Math.abs(d[i + 2] - 0x4C) < 12) gold++;
          return {ann, gold, meshes: st.nv3d.meshes.map(m => m.name), rows: document.querySelectorAll('#annotation-list .trajectory-row').length,
            panel: !document.getElementById('trajectory-panel').hidden}; }"""
    )
    assert len(state["ann"]) == 1, state
    ann = state["ann"][0]
    # One canvas pixel is < 1 mm here; the plane coordinate is exact.
    for got, want in zip(ann["points_ras"], [[x + 25, y + 30, z], [x, y, z]]):
        assert np.allclose(got[:2], want[:2], atol=1.0) and abs(got[2] - z) < 0.6, (got, want)
    assert state["gold"] > 100 and state["rows"] == 1 and state["panel"], state
    assert f"trajectory-{ann['id']}" in state["meshes"], state
    hidden = page.evaluate(
        """async () => { const api = window.__capsule.api, st = window.__capsule.state; await api.setMode('patient');
          const c = document.getElementById('traj-overlay'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++;
          const out = {overlay: n, meshes: st.nv3d.meshes.map(m => m.name).filter(m => m.startsWith('trajectory'))}; await api.setMode('surgeon'); return out; }"""
    )
    assert hidden == {"overlay": 0, "meshes": []}, hidden

    with page.expect_download(timeout=60000) as info:
        page.click("#save-button")
    saved = tmp_path / info.value.suggested_filename
    info.value.save_as(saved)
    _assert_clean(evidence)
    manifest, _ = parse_capsule(saved)
    assert [a for a in manifest["annotations"] if a["type"] == "trajectory"] == [ann]

    page2, context2, evidence2 = _open(chromium, saved)
    try:
        reopened = page2.evaluate(
            """() => { const st = window.__capsule.state; window.__capsule.api.setLayout('2x2'); st.nv.drawScene();
              const c = document.getElementById('traj-overlay'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
              let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++;
              return {overlay: n, meshes: st.nv3d.meshes.map(m => m.name), report: window.__capsule.api.corridorReport().map(r => r.id)}; }"""
        )
    finally:
        context2.close()
    assert reopened["overlay"] > 100 and f"trajectory-{ann['id']}" in reopened["meshes"], reopened
    assert "lesion" in reopened["report"], reopened
    _assert_clean(evidence2)


def test_segmentation_prompt_clicks_round_trip_and_patient_export_hides_control(v2, chromium, tmp_path: Path) -> None:
    page, _, evidence = v2
    prompt_id = page.evaluate(
        """async () => { const c = window.__capsule, api = c.api; api.setCrosshairRAS(...%s);
          const prompt = api.addSegPrompt('Nervo facial'); await api.setTool('segprompt');
          return prompt.id; }""" % (list(LESION_RAS),)
    )
    screen_points = page.evaluate(
        """([x, y, z]) => { const st = window.__capsule.state, gl = document.getElementById('gl');
          const s = st.nv.screenSlices.find(t => t.axCorSag === 0), rect = gl.getBoundingClientRect(), dpr = gl.width / rect.width;
          return [[x, y, z], [x + 7, y + 6, z]].map(p => { const q = sliceProject(s, p);
            return [rect.left + q.x / dpr, rect.top + q.y / dpr]; }); }""",
        list(LESION_RAS),
    )
    page.mouse.click(*screen_points[0])
    page.keyboard.down("Shift")
    page.mouse.click(*screen_points[1])
    page.keyboard.up("Shift")
    expected = page.evaluate(
        """(id) => { const c = window.__capsule, prompt = c.api.segPrompts().find(p => p.id === id);
          const row = document.querySelector('#seg-prompt-list .seg-prompt-row'), meta = row.querySelector('.prompt-meta').textContent;
          const gl = document.getElementById('gl'), s = c.state.nv.screenSlices.find(t => t.axCorSag === 0);
          const colors = prompt.points.map(p => { const q = sliceProject(s, p.ras), d = document.getElementById('seg-prompt-overlay')
            .getContext('2d').getImageData(Math.round(q.x), Math.round(q.y), 1, 1).data; return [d[0], d[1], d[2]]; });
          return {prompt, meta, colors, volume: c.state.base}; }""",
        prompt_id,
    )
    assert expected["prompt"]["name"] == "Nervo facial" and expected["prompt"]["for_volume"] == expected["volume"], expected
    assert expected["prompt"]["status"] == "pending" and expected["meta"].startswith("2 ponto(s)"), expected
    assert [point["positive"] for point in expected["prompt"]["points"]] == [True, False], expected
    assert np.allclose(expected["prompt"]["points"][0]["ras"], LESION_RAS, atol=1.0), expected
    assert np.allclose(expected["prompt"]["points"][1]["ras"], [LESION_RAS[0] + 7, LESION_RAS[1] + 6, LESION_RAS[2]], atol=1.0), expected
    assert expected["colors"][0][1] > expected["colors"][0][0] and expected["colors"][0][1] > expected["colors"][0][2], expected
    assert expected["colors"][1][0] > expected["colors"][1][1] and expected["colors"][1][0] > expected["colors"][1][2], expected

    patient = page.evaluate(
        """async () => { const c = window.__capsule; await c.api.setMode('patient');
          return {mode: c.state.mode, panel: getComputedStyle(document.getElementById('surgeon-panel')).display,
            control: document.getElementById('segmentation-panel').offsetParent === null}; }"""
    )
    assert patient == {"mode": "patient", "panel": "none", "control": True}, patient
    page.evaluate("() => window.__capsule.api.setMode('surgeon')")

    with page.expect_download(timeout=60000) as info:
        page.click("#save-button")
    saved = tmp_path / info.value.suggested_filename
    info.value.save_as(saved)
    _assert_clean(evidence)
    manifest, _ = parse_capsule(saved)
    assert manifest["seg_prompts"] == [expected["prompt"]]

    page2, context2, evidence2 = _open(chromium, saved)
    try:
        reopened = page2.evaluate(
            """() => ({prompts: window.__capsule.api.segPrompts(), hidden: document.getElementById('segmentation-panel').hidden})"""
        )
    finally:
        context2.close()
    assert reopened == {"prompts": [expected["prompt"]], "hidden": True}, reopened
    _assert_clean(evidence2)


def test_mask_legend_returns_to_corner_in_single_slice_layout(v2) -> None:
    page, _, evidence = v2
    pos = page.evaluate(
        """async () => { const api = window.__capsule.api; api.setLayout('2x2'); api.setLayout('axial'); api.render();
          await new Promise(r => setTimeout(r, 300)); const s = document.getElementById('mask-legend').style;
          return [s.left, s.top]; }"""
    )
    assert pos == ["12px", "12px"], pos
    _assert_clean(evidence)


# ---------------------------------------------------------------- r8 honesty
def _sign_in(page, name: str) -> None:
    page.evaluate("""(name) => { const r = document.getElementById('reviewer-name'); r.value = name;
      r.dispatchEvent(new Event('input', {bubbles: true})); }""", name)


def test_r8_mr_vessels_needs_a_mask_and_in_patient_mode_a_signed_review(v2, v2_capsule: Path) -> None:
    """r8 item 2: 'RM Cérebro + vasos' is not offered without a vessel mask; the patient sees it only after a
    signed review of that mask, and a legacy unsigned reviewed:true does not count."""
    page, _, evidence = v2
    manifest, _ = parse_capsule(v2_capsule)
    mr = next(m for m in manifest["volumes"] if m["kind"] == "MR")
    result = page.evaluate(
        """async (id) => {
          const st = window.__capsule.state, api = window.__capsule.api;
          api.setBase(id); await api.renderReady(); api.setLayout('3d');
          const offered = () => presets3dFor(st.volumes.get(id).meta).map(([k]) => k);
          const opts = () => [...document.getElementById('preset3d').options].map(o => o.value);
          const out = {none: [offered(), opts(), default3d(st.volumes.get(id).meta)]};
          const [nx, ny, nz] = st.manifest.grid.dims, brain = st.masks.get('brain_' + id).data, data = new Uint8Array(brain.length);
          const j0 = ny >> 1, k0 = nz >> 1;
          for (let i = 0; i < nx; i++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
            const n = ((k0 + dk) * ny + j0 + dj) * nx + i; if (brain[n]) data[n] = 1; }
          const meta = {id: 'vessels_' + id, blob: 'mask_vessels_' + id, role: 'render', for_volume: id, label: 'Vasos', color: '#3A60D6'};
          st.manifest.masks.push(meta); st.masks.set(meta.id, {meta, data, version: 0, nvimg: null});
          api.setBase(st.manifest.volumes.find(v => v.id !== id).id); api.setBase(id); await api.renderReady();
          out.surgeon = [offered(), opts()];
          await api.setPreset3d('mr-vessels');
          await api.setMode('patient'); out.patientUnreviewed = [offered(), presets3dFor(st.volumes.get(st.base).meta).map(([k]) => k).includes(st.preset3d)]; await api.setMode('surgeon'); api.setBase(id); await api.renderReady();
          out.boxDisabledNoReviewer = document.getElementById('vessel-review-checkbox').disabled;
          try { await api.setMaskReviewed(meta.id, true); out.noReviewer = 'accepted'; } catch (e) { out.noReviewer = e.message; }
          const r = document.getElementById('reviewer-name'); r.value = 'Revisor de teste'; r.dispatchEvent(new Event('input', {bubbles: true}));
          out.boxDisabledWithReviewer = document.getElementById('vessel-review-checkbox').disabled;
          out.signed = await api.setMaskReviewed(meta.id, true);
          out.review = meta.review && [meta.review.by, meta.review.blob_sha256 === meta.blob_sha256, /^\\d{4}-\\d\\d-\\d\\dT/.test(meta.review.at_utc)];
          await api.setMode('patient'); out.patientSigned = offered(); await api.setMode('surgeon');
          delete meta.review; meta.reviewed = true;
          out.legacyStatus = reviewStatus(meta);
          await api.setMode('patient'); out.patientLegacy = offered(); await api.setMode('surgeon');
          meta.review = {by: 'x', at_utc: new Date().toISOString(), blob_sha256: '0'.repeat(64)};
          out.tampered = isValidReview(meta);
          st.manifest.masks.pop(); st.masks.delete(meta.id); renderCache.clear();
          return out;
        }""",
        mr["id"],
    )
    assert "mr-vessels" not in result["none"][0] and "mr-vessels" not in result["none"][1], result
    assert result["none"][2] in result["none"][0], result  # default is always an offered preset
    assert "mr-vessels" in result["surgeon"][0] and "mr-vessels" in result["surgeon"][1], result
    assert "mr-vessels" not in result["patientUnreviewed"][0], result
    assert result["patientUnreviewed"][1] is True, result  # the current preset is always one that is offered
    assert result["boxDisabledNoReviewer"] is True and result["noReviewer"] != "accepted", result
    assert result["boxDisabledWithReviewer"] is False and result["signed"] is True, result
    assert result["review"] == ["Revisor de teste", True, True], result
    assert "mr-vessels" in result["patientSigned"], result
    assert result["legacyStatus"] == "unsigned" and "mr-vessels" not in result["patientLegacy"], result
    assert result["tampered"] is False, result
    _assert_clean(evidence)


def test_r8_orientation_cube_is_drawn_in_3d_and_in_the_2x2_render_tile(v2) -> None:
    """r8 item 3: the 3D instance draws NiiVue's orientation cube; the 2x2 layout's render tile is the same instance."""
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const st = window.__capsule.state, api = window.__capsule.api;
          %s
          const out = {};
          for (const layout of ['3d', '2x2']) {
            api.setLayout(layout); await api.renderReady();
            const on = st.nv3d.opts.isOrientCube, a = snapOf(st.nv3d);
            st.nv3d.opts.isOrientCube = false; const b = snapOf(st.nv3d); st.nv3d.opts.isOrientCube = on; st.nv3d.drawScene();
            out[layout] = [on, diff(a, b)];
          }
          return out;
        }""" % PIX_JS
    )
    for layout in ("3d", "2x2"):
        assert result[layout][0] is True and result[layout][1] > 50, result
    _assert_clean(evidence)


def test_r8_badge_says_pseudonymised_when_the_face_was_not_removed(v2) -> None:
    """r8 item 4: a capsule without deidentification.face_removed never claims 'Anonimizado'."""
    page, _, evidence = v2
    badge = page.evaluate("() => { const b = document.getElementById('anonymous'); return [b.textContent, b.title]; }")
    assert badge == ["Pseudonimizado", "Identificadores removidos; o rosto continua visível na reconstrução 3D"], badge
    _assert_clean(evidence)


def test_r8_filtered_fibres_are_labelled_and_switchable_in_surgeon_mode_only(v2) -> None:
    """r8 item 5: the row shows the filtered share with a pt-BR decimal comma; the toggle adds the outlier meshes
    (off by default) and patient mode shows neither."""
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const st = window.__capsule.state, api = window.__capsule.api;
          api.setLayout('3d'); await api.renderReady();
          const names = () => st.nv3d.meshes.map(m => m.name).filter(n => n.includes('outlier'));
          const rowText = () => [...document.querySelectorAll('#tract-list .meta')].map(e => e.textContent).join(' | ');
          const meta = st.manifest.tracts.find(t => t.id === 'left');
          const out = {meta: [meta.n_streamlines, meta.n_streamlines_outliers, meta.n_streamlines_source, !!meta.outlier_blob],
                       text: rowText(), off: names(), box: document.getElementById('show-outliers').checked,
                       slab: document.querySelector('[data-t="slab"]').textContent};
          api.setShowOutliers(true); await api.renderReady(); out.on = names();
          await api.setMode('patient'); out.patient = [names(), rowText(),
            getComputedStyle(document.getElementById('show-outliers')).display === 'none' || !document.getElementById('show-outliers').offsetParent];
          await api.setMode('surgeon'); out.back = names();
          api.setShowOutliers(false); out.offAgain = names();
          return out;
        }"""
    )
    kept, outliers, source, has_blob = result["meta"]
    assert has_blob and kept + outliers == source, result
    share = f"{100 * outliers / source:.1f}".replace(".", ",")
    assert f"−{share}% filtradas" in result["text"], result
    assert result["off"] == [] and result["box"] is False, result
    assert result["on"] == ["left-outlier.tck"], result
    assert result["patient"][0] == [] and "filtradas" not in result["patient"][1] and result["patient"][2], result
    assert result["back"] == ["left-outlier.tck"] and result["offAgain"] == [], result
    assert result["slab"] == "Espessura no 2D (apenas visual)", result
    _assert_clean(evidence)


def test_r8_tract_review_is_signed_bound_to_the_blob_and_legacy_flags_are_unsigned(v2, v2_capsule: Path) -> None:
    """r8 item 6: legacy reviewed:true shows 'revisado (sem assinatura)'; checkboxes stay disabled until a Revisor
    is named; a signature carries the sha256 of the tract bytes."""
    page, _, evidence = v2
    manifest, blobs = parse_capsule(v2_capsule)
    expected = hashlib.sha256(blobs["tract_left"]).hexdigest()
    result = page.evaluate(
        """async () => {
          const st = window.__capsule.state, api = window.__capsule.api;
          const rows = () => [...document.querySelectorAll('#tract-list .item-row, #tract-list > *')].map(r => r.textContent);
          const boxes = () => [...document.querySelectorAll('#tract-list .reviewed input')].map(b => b.disabled);
          const out = {through: reviewStatus(st.manifest.tracts.find(t => t.id === 'through')), rows: rows(), boxes: boxes()};
          try { await api.setTractReviewed('left', true); out.noReviewer = 'accepted'; } catch (e) { out.noReviewer = e.message; }
          const r = document.getElementById('reviewer-name'); r.value = 'Revisor de teste'; r.dispatchEvent(new Event('input', {bubbles: true}));
          out.boxesAfter = boxes();
          out.signed = await api.setTractReviewed('left', true);
          const meta = st.manifest.tracts.find(t => t.id === 'left');
          out.review = [meta.review.by, meta.review.blob_sha256, meta.blob_sha256];
          return out;
        }"""
    )
    assert result["through"] == "unsigned", result
    assert any("revisado (sem assinatura)" in row for row in result["rows"]), result
    assert all(result["boxes"]) and result["noReviewer"] != "accepted", result
    assert not any(result["boxesAfter"]) and result["signed"] is True, result
    assert result["review"] == ["Revisor de teste", expected, expected], result
    _assert_clean(evidence)


def test_crop_grid_uses_surfaces_default_and_keeps_uncropped_defaults(v2, chromium, v2_cropped_capsule: Path) -> None:
    page, _, evidence = v2
    uncropped = page.evaluate(
        """async () => {
          const {state, api} = window.__capsule, defaults = {};
          for (const volume of state.manifest.volumes) {
            await api.setBase(volume.id); await api.renderReady(); defaults[volume.kind] = state.preset3d;
          }
          return defaults;
        }"""
    )
    assert uncropped == {"CT": "ct-bone", "MR": "mr-brain"}, uncropped

    crop_page, crop_context, crop_evidence = _open(chromium, v2_cropped_capsule)
    try:
        cropped = crop_page.evaluate(
            """async () => {
              const {state, api} = window.__capsule, out = {hasCrop: !!state.manifest.grid.crop, kinds: {}};
              for (const volume of state.manifest.volumes) {
                await api.setBase(volume.id); await api.renderReady();
                out.kinds[volume.kind] = {preset: state.preset3d,
                  offered: [...document.getElementById('preset3d').options].map(option => option.value),
                  volumeOpacity: state.nv3d.volumes[0]?.opacity};
              }
              await api.setMode('patient');
              out.patient = {preset: state.preset3d, tracts: api.visibleTracts(), masks: api.visibleMasks()};
              return out;
            }"""
        )
    finally:
        crop_context.close()

    assert cropped["hasCrop"] is True, cropped
    assert cropped["kinds"]["CT"]["preset"] == "surfaces", cropped
    assert "surfaces" in cropped["kinds"]["CT"]["offered"] and "ct-skin" not in cropped["kinds"]["CT"]["offered"], cropped
    assert cropped["kinds"]["CT"]["volumeOpacity"] == 0, cropped
    assert cropped["kinds"]["MR"]["preset"] == "surfaces", cropped
    assert "surfaces" in cropped["kinds"]["MR"]["offered"], cropped
    assert "mr-brain" in cropped["kinds"]["MR"]["offered"] and "mr-skin" not in cropped["kinds"]["MR"]["offered"], cropped
    assert cropped["kinds"]["MR"]["volumeOpacity"] == 0, cropped
    assert cropped["patient"]["preset"] == "surfaces", cropped
    assert cropped["patient"]["tracts"] == [] and cropped["patient"]["masks"] == [], cropped
    _assert_clean(evidence)
    _assert_clean(crop_evidence)


def test_tract_import_note_is_compact_and_surgeon_only(v2) -> None:
    page, _, evidence = v2
    result = page.evaluate(
        """async () => {
          const {state, api} = window.__capsule, out = {};
          state.manifest.tracts[0].provenance = {recorded: true, software: 'MRtrix', algorithm: 'iFOD2', b_values: '1000'};
          await api.setMode('surgeon');
          out.surgeon = document.querySelector('#tract-list .tract-import-note')?.textContent ?? null;
          await api.setMode('patient');
          out.patient = document.querySelector('#tract-list .tract-import-note')?.textContent ?? null;
          out.patientRowTitles = [...document.querySelectorAll('#tract-list .item')].map(row => row.title).filter(Boolean);
          await api.setMode('surgeon'); state.manifest.tracts = []; renderTractList();
          out.empty = {note: document.querySelector('#tract-list .tract-import-note')?.textContent ?? null,
                       message: document.querySelector('#tract-list .empty-note')?.textContent ?? null};
          return out;
        }"""
    )
    assert result["surgeon"] == "Tratos importados (.tck): calculados fora da cápsula · MRtrix · iFOD2 · b=1000", result
    assert result["patient"] is None and result["patientRowTitles"] == [], result
    assert result["empty"]["note"] is None and result["empty"]["message"] == "Nenhum trato nesta cápsula.", result
    _assert_clean(evidence)


def test_r9_readout_names_native_and_grid_spacing_in_surgeon_mode_only(chromium, real_ct_capsule: Path) -> None:
    """r9 item A: a resampled volume says so in the surgeon readout; patient mode shows no readout text."""
    manifest, _ = parse_capsule(real_ct_capsule)
    volume = manifest["volumes"][0]
    assert volume["resampled"] is True and len(volume["source_spacing_mm"]) == 3, volume
    fmt = lambda v: f"{round(float(v), 2):g}".replace(".", ",")  # pt-BR, at most 2 decimals
    expected = (f"reamostrado de {'×'.join(fmt(v) for v in volume['source_spacing_mm'])} mm "
                f"para {fmt(manifest['grid']['spacing_mm'][0])} mm")
    page, context, evidence = _open(chromium, real_ct_capsule)
    try:
        result = page.evaluate(
            """async () => {
              const c = window.__capsule, el = document.getElementById('readout'), out = {};
              c.api.setCrosshairRAS(...c.state.crosshair); out.surgeon = el.textContent;
              await c.api.setMode('patient'); out.patient = el.textContent; return out;
            }"""
        )
    finally:
        context.close()
    assert expected in result["surgeon"], (expected, result)
    assert result["patient"] == "", result
    _assert_clean(evidence)


@pytest.fixture(scope="session")
def v2_large_capsule(tmp_path_factory: pytest.TempPathFactory, v2_corridor_vessel_capsule: Path) -> Path:
    """Synthetic 340x400x320 CT + MR capsule (head, brain and vessel render masks on the CT); no patient data."""
    from dev_fixture import LARGE_DIMS, build_large

    output = tmp_path_factory.mktemp("case-capsule-v2-large") / "large.capsule.html"
    build_large(output, v2_corridor_vessel_capsule)
    assert tuple(parse_capsule(output)[0]["grid"]["dims"]) == LARGE_DIMS
    return output


# Measured post-GC JS heap on this grid (Apple Silicon, headless Chromium): before the fix 576 MB at ready growing
# to 750 MB after two 3D presets (each volume held twice, each 3D preset cached beside the last); after it 402 MB
# flat. The budget is the 500 MB target for real 340x401x321 capsules; the old code fails it by 250 MB.
LARGE_HEAP_BUDGET_MB = 500


def test_large_grid_js_heap_stays_within_budget_across_3d_presets_and_patient_mode(chromium, v2_large_capsule: Path) -> None:
    browser = chromium.browser_type.launch(
        executable_path=str(_cached_chromium_executable()), headless=True,
        args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist",
              "--enable-unsafe-swiftshader", "--enable-precise-memory-info", "--js-flags=--expose-gc"],
    )
    heap: dict[str, int] = {}
    cache_sizes: list[int] = []
    try:
        page, context, evidence = _open(browser, v2_large_capsule)
        mark = """async () => { gc(); gc(); await new Promise(r => setTimeout(r, 200));
                               return Math.round(performance.memory.usedJSHeapSize / 1e6); }"""
        try:
            heap["ready"] = page.evaluate(mark)
            page.evaluate("() => window.__capsule.api.setLayout('3d')")
            page.evaluate("async () => window.__capsule.api.renderReady()")
            heap["3d"] = page.evaluate(mark)
            presets = page.evaluate("() => [...document.getElementById('preset3d').options].map(o => o.value)")
            assert {"ct-bone", "ct-skin"} <= set(presets), presets
            for preset in presets:
                page.evaluate("async (p) => { await window.__capsule.api.setPreset3d(p); await window.__capsule.api.renderReady(); }", preset)
                heap[f"preset {preset}"] = page.evaluate(mark)
                cache_sizes.append(page.evaluate("() => window.__capsule.api.renderCacheSize()"))
            page.evaluate("() => window.__capsule.api.setLayout('2x2')")
            page.evaluate("async () => window.__capsule.api.renderReady()")
            heap["2x2"] = page.evaluate(mark)
            page.evaluate("async () => { await window.__capsule.api.setMode('patient'); await window.__capsule.api.renderReady(); }")
            heap["patient"] = page.evaluate(mark)
        finally:
            context.close()
    finally:
        browser.close()
    print("post-GC JS heap (MB):", heap)
    # A large grid keeps one render copy: switching presets rebuilds instead of caching several int16 grids.
    assert max(cache_sizes) == 1, cache_sizes
    assert max(heap.values()) <= LARGE_HEAP_BUDGET_MB, heap
    _assert_clean(evidence)
