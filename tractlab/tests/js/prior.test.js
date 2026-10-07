import { test } from "node:test";
import assert from "node:assert/strict";
import { describePrior, PRIOR_SHELL_COLOR } from "../../viewer/prior.js";

test("describePrior laterality from id token only", () => {
  const r = describePrior({ id: "norm_cst_r", label: "FAKE L" });
  assert.equal(r.side, "R");
  assert.equal(r.source, "atlas");
  assert.equal(r.provenance, "POPULATION ATLAS");
  assert.match(r.short, /CST-R/);
});

test("describePrior unknown side is ?", () => {
  const r = describePrior({ id: "norm_fmi" });
  assert.equal(r.side, "?");
  assert.equal(r.source, "atlas");
});

test("prior shell colour is monochrome constant (not RGB direction)", () => {
  assert.equal(typeof PRIOR_SHELL_COLOR, "number");
});
