import { test } from "node:test";
import assert from "node:assert/strict";
import {
  formatCutLine,
  bankShortName,
  panelCopyIsClean,
} from "../../viewer/connectotomy.js";

test("formatCutLine uses the locked verb", () => {
  assert.equal(
    formatCutLine({ n_cut: 12, n_bank: 400 }),
    "cuts 12 of 400 bank streamlines",
  );
});

test("C1 copy fails closed on forbidden words", () => {
  assert.equal(
    panelCopyIsClean("Lesion cavity. Geometry uncertain to ~0.6 mm. Research/preview only — not a resection plan."),
    true,
  );
  assert.equal(panelCopyIsClean("cuts 3 of 10 bank streamlines"), true);
  assert.equal(panelCopyIsClean("resection margin"), false);
  assert.equal(panelCopyIsClean("fibres at risk"), false);
  assert.equal(panelCopyIsClean("ASSIGNED edge"), false);
  assert.equal(panelCopyIsClean("Yeo-7 network"), false);
  assert.equal(panelCopyIsClean("Schaefer-200"), false);
});

test("bankShortName strips the bank_ prefix", () => {
  assert.equal(bankShortName({ id: "bank_cst_r" }), "cst-r");
  assert.equal(bankShortName({ label: "CST-R", id: "bank_cst_r" }), "CST-R");
});
