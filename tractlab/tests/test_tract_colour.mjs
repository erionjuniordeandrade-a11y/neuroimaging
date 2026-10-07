/**
 * Run: node --test tests/test_tract_colour.mjs
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DIST_MAX,
  heatColor,
  dirColor,
  solidColorForBundle,
} from "../viewer/tract_colour.js";

test("dirColor: pure L-R is red-dominant", () => {
  const [r, g, b] = dirColor(1, 0, 0);
  assert.ok(r > g && r > b);
  assert.ok(r > 0.9);
});

test("dirColor: pure S-I is blue-dominant", () => {
  const [r, g, b] = dirColor(0, 0, 1);
  assert.ok(b > r && b > g);
});

test("heatColor: near lesion is warmer than far", () => {
  const near = heatColor(0);
  const far = heatColor(DIST_MAX);
  // near: high R; far: high B (cyan end)
  assert.ok(near[0] > far[0]);
  assert.ok(far[2] > near[2]);
});

test("solidColorForBundle laterality-safe roles", () => {
  const cst = solidColorForBundle("true_cst", "True CST-R");
  const fat = solidColorForBundle("true_fat", "FAT-L");
  const slf = solidColorForBundle("true_slf3", "SLF-III-L (SMG↔frontal)");
  assert.ok(cst[0] > 0.8); // gold-ish
  assert.ok(fat[1] > 0.7); // teal-ish
  assert.ok(slf[2] > 0.8); // violet-ish
});
