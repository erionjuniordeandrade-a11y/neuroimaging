/**
 * Derivation copy — pure builder tests (ADR-0007). No DOM; index.html wiring
 * is exercised via tests/perf/derivation_copy_smoke.mjs on a live server.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  KIND_UNCORRECTED,
  KIND_RPE_PAIR,
  reversePeNote,
  researchChipTitle,
  t1AnatomyTitle,
  recoveryFloorNote,
} from "../../viewer/derivation_copy.js";

test("reversePeNote: rpe_pair reads corrected, uncorrected reads no reverse-PE", () => {
  assert.equal(reversePeNote(KIND_RPE_PAIR), "reverse-PE corrected");
  assert.equal(reversePeNote(KIND_UNCORRECTED), "no reverse-PE");
});

test("unknown_derivation_kind_returns_unavailable_never_uncorrected", () => {
  // A null/unrecognized kind (API not loaded yet, fetch failed, unexpected
  // value) is an ABSENT answer, not evidence of an uncorrected case — it
  // must never silently read as "no reverse-PE".
  for (const kind of [null, undefined, "some-unknown-kind", "", 0, {}]) {
    const note = reversePeNote(kind);
    assert.equal(note, "derivation status unavailable", `kind=${JSON.stringify(kind)}`);
    assert.notEqual(note, "no reverse-PE", `kind=${JSON.stringify(kind)} must not read as uncorrected`);
  }
});

test("researchChipTitle never says no reverse-PE for a corrected derivation, and never guesses on unknown", () => {
  assert.equal(
    researchChipTitle(KIND_RPE_PAIR),
    "Research preview only — geometric uncertainty, reverse-PE corrected",
  );
  assert.doesNotMatch(researchChipTitle(KIND_RPE_PAIR), /no reverse-PE/);
  assert.equal(
    researchChipTitle(KIND_UNCORRECTED),
    "Research preview only — geometric uncertainty, no reverse-PE",
  );
  assert.equal(
    researchChipTitle(null),
    "Research preview only — geometric uncertainty, derivation status unavailable",
  );
  assert.doesNotMatch(researchChipTitle(null), /no reverse-PE/);
});

test("t1AnatomyTitle: signed rpe_pair states few-mm uncertainty", () => {
  assert.equal(
    t1AnatomyTitle(KIND_RPE_PAIR, true),
    "T1 anatomy (QC-approved · reverse-PE corrected · few-mm uncertainty)",
  );
});

test("t1AnatomyTitle: unsigned rpe_pair never claims few-mm uncertainty, states correction status only", () => {
  for (const signed of [false, undefined, null]) {
    const title = t1AnatomyTitle(KIND_RPE_PAIR, signed);
    assert.equal(title, "T1 anatomy (QC-approved · reverse-PE corrected · residual uncertainty unquantified)");
    assert.doesNotMatch(title, /few-mm uncertainty/);
    assert.doesNotMatch(title, /no reverse-PE/);
  }
});

test("t1AnatomyTitle: uncorrected always states its fixed few-mm floor regardless of signed flag", () => {
  assert.equal(
    t1AnatomyTitle(KIND_UNCORRECTED, true),
    "T1 anatomy (QC-approved · no reverse-PE · few-mm uncertainty)",
  );
  assert.equal(
    t1AnatomyTitle(KIND_UNCORRECTED, false),
    "T1 anatomy (QC-approved · no reverse-PE · few-mm uncertainty)",
  );
});

test("t1AnatomyTitle: unknown kind carries no uncertainty claim at all", () => {
  const title = t1AnatomyTitle(null, true);
  assert.equal(title, "T1 anatomy (QC-approved · derivation status unavailable)");
  assert.doesNotMatch(title, /few-mm uncertainty/);
  assert.doesNotMatch(title, /no reverse-PE/);
});

test("recoveryFloorNote: uncorrected floor label carries the recorded mm figure", () => {
  assert.equal(
    recoveryFloorNote("no reverse-PE — ~3 mm geometric floor"),
    "(from lesion surface · geom floor ~3 mm)",
  );
});

test("recoveryFloorNote: corrected (signed or not) never borrows the uncorrected floor wording", () => {
  assert.equal(
    recoveryFloorNote("reverse-PE corrected — median shift 0.6 mm (signed delta QC)"),
    "(from lesion surface · reverse-PE corrected)",
  );
  assert.equal(
    recoveryFloorNote("reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"),
    "(from lesion surface · reverse-PE corrected)",
  );
  assert.doesNotMatch(
    recoveryFloorNote("reverse-PE corrected — median shift 0.6 mm (signed delta QC)"),
    /~3 mm/,
  );
});

test("recoveryFloorNote: unavailable/unloaded floor label never invents a number", () => {
  assert.equal(recoveryFloorNote(""), "(from lesion surface · floor unavailable)");
  assert.equal(recoveryFloorNote(undefined), "(from lesion surface · floor unavailable)");
  assert.equal(recoveryFloorNote("Derivation unavailable"), "(from lesion surface · floor unavailable)");
});
