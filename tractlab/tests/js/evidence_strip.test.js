/**
 * Per-bundle evidence strip — pure row-model tests (unit 2C).
 * DOM wiring in index.html is checked by the browser pass, not here.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  buildEvidenceRows, rowStrings, PROVENANCE_REFUSALS, STRIP_FRAMING,
  CLASS_TRACT, CLASS_PRIOR,
} from "../../viewer/evidence_strip.js";

const HASH = "a".repeat(64);

function profile(trackMm, { reason = null, sourceHash = "src1" } = {}) {
  return {
    scalar: "fa",
    sourceHash,
    payload: { lesion_distance: { track_mm: trackMm, reason } },
  };
}
function track(minMm, minNode, n = 100) {
  return Array.from({ length: n }, (_, i) => (i === minNode ? minMm : minMm + 5 + i));
}
const SIGNED = { status: "ok", opSource: "signed", approvedBy: "reviewer", opDate: "2026-09-01" };
const PILOT = { status: "ok", opSource: "pilot", approvedBy: "", opDate: "" };

test("Tract row: served closest approach and signed operating point", () => {
  const [row] = buildEvidenceRows({ tracts: [{
    bankId: "bank_cst_r", label: "CST-R", fidelity: SIGNED,
    profile: profile(track(0.6, 75)), layerSourceHash: "src1",
  }] });
  assert.equal(row.evidenceClass, CLASS_TRACT);
  assert.equal(row.displayed, true);
  assert.equal(row.lesion.state, "served");
  assert.equal(row.lesion.text, "closest 0.6 mm, node 75");
  assert.equal(row.fidelity.state, "served");
  assert.match(row.fidelity.text, /signed by reviewer 2026-09-01/);
  assert.deepEqual([...row.change], ["no open refusal served for this row"]);
});

test("Tract row with pilot operating point names the missing signature", () => {
  const [row] = buildEvidenceRows({ tracts: [{
    bankId: "bank_fat_r", label: "FAT-R", fidelity: PILOT,
    profile: profile(track(12, 3)), layerSourceHash: "src1",
  }] });
  assert.match(row.fidelity.text, /pilot, unsigned/);
  assert.match(row.change[0], /owner-signed operating-point sheet/);
  assert.equal(row.lesion.text, "closest 12 mm, node 3");
});

test("Atlas-prior row: never a Tract, no distance, no fidelity", () => {
  const [row] = buildEvidenceRows({ priors: [{ id: "norm_cst_r", label: "CST-R prior" }] });
  assert.equal(row.evidenceClass, CLASS_PRIOR);
  assert.equal(row.lesion.state, "not-applicable");
  assert.equal(row.fidelity.state, "not-applicable");
  assert.match(row.classNote, /not this patient/);
  assert.match(row.change[0], /never becomes a Tract \(ADR-0001\)/);
});

test("refused row lists the served refusal reason verbatim as what would change it", () => {
  const reason = "sidecar bank sha mismatch: 0123456789ab… vs expected ba9876543210…";
  const rows = buildEvidenceRows({ refused: [{ bankId: "bank_slf3_r", label: "SLF-III-R", reason }] });
  assert.equal(rows.length, 1);
  const [row] = rows;
  assert.equal(row.displayed, false);
  assert.equal(row.evidenceClass, CLASS_TRACT);
  assert.equal(row.fidelity.state, "missing");
  assert.deepEqual([...row.change], [reason]);
});

test("refused row without a served reason says so, never a guessed reason", () => {
  const [row] = buildEvidenceRows({ refused: [{ bankId: "bank_x_r", label: "X", reason: "" }] });
  assert.deepEqual([...row.change], ["refusal reason not served"]);
});

test("provenance-incomplete quotes both FidelityRefusal texts from fidelity.py", () => {
  const src = readFileSync(new URL("../../src/tractlab/fidelity.py", import.meta.url), "utf8");
  for (const text of PROVENANCE_REFUSALS) assert.ok(src.includes(text), `not in fidelity.py: ${text}`);
  const [row] = buildEvidenceRows({ tracts: [{ bankId: "bank_cst_l", label: "CST-L",
    fidelity: { status: "provenance-incomplete" }, profile: null }] });
  assert.deepEqual([...row.change], [...PROVENANCE_REFUSALS]);
  assert.equal(row.fidelity.state, "missing");
});

test("missing profile: every absence says why, never a default number", () => {
  const cases = [
    [null, /profile not requested yet/],
    [{ error: "scalar_missing", scalar: "fa" }, /profile unavailable: FA map not available/],
    [{ error: "request_failed", scalar: "fa" }, /profile unavailable: The profile request did not complete/],
    [profile(null, { reason: "no lesion mask" }), /no lesion distance: no lesion mask/],
    [profile(null), /no lesion distance: reason not served/],
    [profile(track(0.6, 75), { sourceHash: "old" }), /tract file changed on disk/],
  ];
  for (const [p, re] of cases) {
    const [row] = buildEvidenceRows({ tracts: [{ bankId: "bank_cst_r", label: "CST-R",
      fidelity: { status: "absent" }, profile: p, layerSourceHash: "src1" }] });
    assert.equal(row.lesion.state, "missing");
    assert.match(row.lesion.text, re);
    assert.doesNotMatch(row.lesion.text, /\d+(\.\d+)?\s*mm/, "a missing distance must carry no number");
  }
  const [absent] = buildEvidenceRows({ tracts: [{ bankId: "bank_a_r", fidelity: { status: "absent" } }] });
  assert.match(absent.fidelity.text, /untested/);
  const [none] = buildEvidenceRows({ tracts: [{ bankId: "bank_a_r" }] });
  assert.match(none.fidelity.text, /fidelity status not served/);
});

test("a refused bank that is also displayed is not listed twice", () => {
  const rows = buildEvidenceRows({
    tracts: [{ bankId: "bank_cst_r", label: "CST-R", fidelity: SIGNED }],
    refused: [{ bankId: "bank_cst_r", label: "CST-R", reason: "x" }],
  });
  assert.equal(rows.length, 1);
});

// ── negative control ─────────────────────────────────────────────────────────

const RISK_WORDS = /\b(safe|safer|safety|unsafe|risk|risks|risky|danger|dangerous|hazard|hazardous|threat|alarm|score)\b/i;

function servedRefusalTexts() {
  // Every literal FidelityRefusal message in the production sources, so the
  // control tracks the real strings rather than a hand-picked sample.
  const out = [];
  for (const file of ["fidelity.py", "serve.py"]) {
    const src = readFileSync(new URL(`../../src/tractlab/${file}`, import.meta.url), "utf8");
    for (const m of src.matchAll(/FidelityRefusal\(\s*((?:f?"[^"]*"\s*)+)\)/g)) {
      out.push(m[1].replace(/f?"([^"]*)"\s*/g, "$1"));
    }
  }
  return out;
}

test("negative control: the refusal harvester finds the real strings", () => {
  const texts = servedRefusalTexts();
  assert.ok(texts.length >= 10, `only ${texts.length} refusal texts harvested`);
  assert.ok(texts.some((t) => t.includes("cannot pin the FOD")));
  // known-positive for the word check itself
  assert.match("this tract is safe", RISK_WORDS);
  assert.match("risk score 0.3", RISK_WORDS);
});

test("negative control: no row ever emits a risk word, and order is never by distance", () => {
  const refusals = servedRefusalTexts();
  const statuses = [SIGNED, PILOT, { status: "absent" }, { status: "provenance-incomplete" },
    { status: "error" }, undefined];
  const mm = [30, 0.4, 12, 3, 0.9, 55];
  const tracts = mm.map((d, i) => ({
    bankId: `bank_t${i}_r`, label: `T${i}`, fidelity: statuses[i],
    profile: profile(track(d, i * 7)), layerSourceHash: "src1",
  }));
  const rows = buildEvidenceRows({
    tracts,
    refused: refusals.map((reason, i) => ({ bankId: `bank_r${i}_l`, label: `R${i}`, reason })),
    priors: [{ id: "norm_cst_r", label: "CST-R prior" }, { id: "norm_af_l", label: "AF-L prior" }],
  });
  for (const row of rows) {
    for (const s of rowStrings(row)) assert.doesNotMatch(s, RISK_WORDS, `risk word in: ${s}`);
  }
  assert.doesNotMatch(STRIP_FRAMING, RISK_WORDS);
  // Order is the caller's, whatever the distances are.
  assert.deepEqual(rows.slice(0, mm.length).map((r) => r.label), mm.map((_, i) => `T${i}`));
  const reversed = buildEvidenceRows({ tracts: [...tracts].reverse() });
  assert.deepEqual(reversed.map((r) => r.label), mm.map((_, i) => `T${i}`).reverse());
  // No aggregate: every row is about one object; the model carries no totals.
  for (const row of rows) {
    assert.deepEqual(Object.keys(row).sort(),
      ["change", "classNote", "displayed", "evidenceClass", "fidelity", "key", "label", "lesion"]);
  }
  const src = readFileSync(new URL("../../viewer/evidence_strip.js", import.meta.url), "utf8");
  assert.doesNotMatch(src, /\.sort\(|\.reduce\(|Math\.(min|max)\(/, "the strip must not rank or aggregate");
  assert.doesNotMatch(src, /#(?:[0-9a-f]{3}){1,2}\b|rgb\(|\b(?:red|amber|green)\b/i, "the strip carries no traffic-light colour");
});
