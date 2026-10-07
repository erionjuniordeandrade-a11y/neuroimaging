/**
 * tract_colour.js — pure colour-helper suite (rgbToHex).
 * Run: npm test   or   node --test tests/js/
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { rgbToHex, solidColorForBundle } from "../../viewer/tract_colour.js";

test("rgbToHex: pure channels", () => {
  assert.equal(rgbToHex([1, 0, 0]), "#ff0000");
  assert.equal(rgbToHex([0, 1, 0]), "#00ff00");
  assert.equal(rgbToHex([0, 0, 1]), "#0000ff");
  assert.equal(rgbToHex([0, 0, 0]), "#000000");
  assert.equal(rgbToHex([1, 1, 1]), "#ffffff");
});

test("rgbToHex: rounds and clamps out-of-range floats", () => {
  assert.equal(rgbToHex([1.4, -0.2, 0.5]), "#ff0080");
});

test("rgbToHex: round-trips a real bundle accent to a 6-digit hex", () => {
  const hex = rgbToHex(solidColorForBundle("true_cst", "CST-L"));
  assert.match(hex, /^#[0-9a-f]{6}$/);
});

// --- OR hue-move regression -------------------------------------------
//
// Owner ruling (docs/DESIGN-viewer-brainstorm-2026-09-01.md, "Owner rulings"
// item 2): CST keeps gold; optic radiation (OR) moves off amber-orange onto
// the open fuchsia-magenta slot, because amber-OR and gold-CST read as
// near-identical chips in the sidebar. This test proves the new OR hue is
// perceptually distinct (CIE76 ΔE ≥ 25) from every neighbouring family hue,
// and that the separation from CST specifically at least doubled.

/** sRGB companding (gamma decode): 0..1 channel → linear light. */
function decompand(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** [r,g,b] 0..1 → D65 CIE XYZ. */
function rgbToXyz([rs, gs, bs]) {
  const r = decompand(rs);
  const g = decompand(gs);
  const b = decompand(bs);
  return [
    r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
    r * 0.2126729 + g * 0.7151522 + b * 0.072175,
    r * 0.0193339 + g * 0.119192 + b * 0.9503041,
  ];
}

/** D65 XYZ → CIE Lab. */
function xyzToLab([X, Y, Z]) {
  const Xn = 0.95047;
  const Yn = 1.0;
  const Zn = 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X / Xn);
  const fy = f(Y / Yn);
  const fz = f(Z / Zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

const rgbToLab = (rgb) => xyzToLab(rgbToXyz(rgb));

/** CIE76 ΔE: Euclidean distance in Lab space. */
function deltaE76(labA, labB) {
  return Math.sqrt(
    (labA[0] - labB[0]) ** 2 + (labA[1] - labB[1]) ** 2 + (labA[2] - labB[2]) ** 2
  );
}

test("solid colour: OR hue is CIE76 ΔE >= 25 from every neighbouring family, and >= 2x the old amber-orange separation from CST", () => {
  const cst = solidColorForBundle("true_cst", "CST-L");
  const or = solidColorForBundle("true_or", "OR-L");
  const uf = solidColorForBundle("uncinate", "UF-L");
  const cc = solidColorForBundle("callosum", "CC");
  const slf = solidColorForBundle("slf", "SLF-L");

  const labOr = rgbToLab(or);
  const deltas = {
    "OR-CST": deltaE76(labOr, rgbToLab(cst)),
    "OR-UF": deltaE76(labOr, rgbToLab(uf)),
    "OR-CC": deltaE76(labOr, rgbToLab(cc)),
    "OR-SLF": deltaE76(labOr, rgbToLab(slf)),
  };

  // Old OR colour, pre-ruling — documents why the hue moved.
  const oldOr = [0.95, 0.55, 0.2]; // amber-orange
  const oldDeltaOrCst = deltaE76(rgbToLab(oldOr), rgbToLab(cst));

  console.log("ΔE(OR, CST) =", deltas["OR-CST"].toFixed(2));
  console.log("ΔE(OR, UF)  =", deltas["OR-UF"].toFixed(2));
  console.log("ΔE(OR, CC)  =", deltas["OR-CC"].toFixed(2));
  console.log("ΔE(OR, SLF) =", deltas["OR-SLF"].toFixed(2));
  console.log("ΔE(old amber-orange OR, CST) =", oldDeltaOrCst.toFixed(2));

  for (const [pair, d] of Object.entries(deltas)) {
    assert.ok(d >= 25, `${pair} ΔE ${d.toFixed(2)} should be >= 25`);
  }
  assert.ok(
    deltas["OR-CST"] >= oldDeltaOrCst * 2,
    `new ΔE(OR,CST) ${deltas["OR-CST"].toFixed(2)} should be >= 2x old ${oldDeltaOrCst.toFixed(2)}`
  );
});
