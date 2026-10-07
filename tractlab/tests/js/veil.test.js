import test from "node:test";
import assert from "node:assert/strict";

import {
  hullVeilParams, ghostMaterialParams, GHOST_OPACITY, GHOST_COLOR,
  spotlightLowSupportParams, SPOTLIGHT_OPACITY, SPOTLIGHT_EMISSIVE_INTENSITY,
  pialShade, PIAL_MID_OPACITY,
} from "../../viewer/veil.js";

test("glass hull is translucent context without additive points", () => {
  const p = hullVeilParams("ghost");
  assert.equal(p.opacity, 0.14, "single layer: silhouette alpha; face-on is opacity × HULL_RIM_FLOOR");
  assert.equal(p.side, "front", "nearest layer only; double-sided folds stacked into milk");
  assert.equal(p.depthWrite, false);
  assert.equal(p.points, false);
  assert.equal(p.transparent, true);
});

test("hull veil mid state commits opacity, drops points, keeps depthWrite off", () => {
  const p = hullVeilParams("mid");
  assert.equal(p.opacity, 0.28);
  assert.equal(p.side, "front");
  assert.equal(p.depthWrite, false);
  assert.equal(p.points, false);
});

test("solid hull writes depth as an opaque surface", () => {
  const p = hullVeilParams("solid");
  assert.equal(p.opacity, 1);
  assert.equal(p.transparent, false);
  assert.equal(p.side, "front");
  assert.equal(p.depthWrite, true);
  assert.equal(p.points, false);
});

test("hull veil opacity strictly increases ghost < mid < solid", () => {
  const g = hullVeilParams("ghost").opacity;
  const m = hullVeilParams("mid").opacity;
  const s = hullVeilParams("solid").opacity;
  assert.ok(g < m && m < s);
});

test("hull veil rejects an unknown state", () => {
  assert.throws(() => hullVeilParams("bogus"));
});

test("hull veil params are independent copies (no shared mutable object)", () => {
  const a = hullVeilParams("ghost");
  a.opacity = 999;
  const b = hullVeilParams("ghost");
  assert.equal(b.opacity, 0.14);
});

test("ghost material params: transparent, no depth write, colourless (no vertex colours)", () => {
  const g = ghostMaterialParams();
  assert.equal(g.vertexColors, false, "colour is reserved for evidence — the ghost must be shape only");
  assert.equal(g.color, "#6e747e", "ghost colour is the --mute token, not a brand/DEC colour");
  assert.equal(g.transparent, true);
  assert.equal(g.depthWrite, false);
  assert.equal(g.emissive, 0x000000);
});

test("ghost colour is pinned to the --mute token value", () => {
  assert.equal(GHOST_COLOR, "#6e747e");
  assert.equal(GHOST_COLOR, ghostMaterialParams().color);
});

test("ghost opacity is pinned, low, and non-zero", () => {
  // 0.06 (first landing) and 0.03 both piled up above the ~90/255 dense-core
  // target under overdraw accumulation (measured: 170 and 108.9); 0.02 is
  // the floor the owner set (shape must stay visible at the periphery) and
  // measured closest to target (dense core 97.2, still >90 but the floor
  // wins — see the capture report for the follow-up commit).
  assert.equal(GHOST_OPACITY, ghostMaterialParams().opacity);
  assert.equal(GHOST_OPACITY, 0.02);
  assert.ok(GHOST_OPACITY > 0, "zero opacity would make the ghost invisible");
  assert.ok(GHOST_OPACITY >= 0.02, "must not go below 0.02 — shape must stay visible at the periphery");
  assert.ok(GHOST_OPACITY <= 0.10, "ghost must stay faint per the design kill criterion");
});

test("spotlight params: near-solid opacity, boosted emissive, above the default dim weight", () => {
  const s = spotlightLowSupportParams();
  assert.equal(s.opacity, SPOTLIGHT_OPACITY);
  assert.equal(s.emissiveIntensity, SPOTLIGHT_EMISSIVE_INTENSITY);
  // buildTubeMesh's default low-support weight is opacity 0.35 / emissive
  // 0.12 — the spotlight must clearly exceed both, or "Only low-support"
  // gains nothing over the untouched default.
  assert.ok(s.opacity > 0.35, "spotlight opacity must exceed the default dim low-support weight");
  assert.ok(s.emissiveIntensity > 0.12, "spotlight emissive must exceed the default dim low-support weight");
  assert.ok(s.opacity <= 1, "opacity is a [0,1] material property");
  assert.ok(s.emissiveIntensity <= 0.8, "owner ceiling — do not raise emissive past 0.8");
});

test('pial shade: gyral crowns light, sulcal fundi dark, neutral grey only', () => {
  const crown = pialShade(-6), fundus = pialShade(8), mid = pialShade(1);
  assert.ok(crown[0] > mid[0] && mid[0] > fundus[0]);
  for (const c of [crown, fundus, mid]) assert.ok(Math.max(...c) - Math.min(...c) < 0.1, 'no hue');
  assert.deepEqual(pialShade(NaN), pialShade(1));
  assert.ok(PIAL_MID_OPACITY > 0 && PIAL_MID_OPACITY < 1);
});
