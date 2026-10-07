/**
 * Provenance strip — pure builder tests (owner ruling 8, DESIGN doc D+E).
 * No DOM; index.html wiring is exercised via the browser smoke capture.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { provenanceChips, preflightState } from "../../viewer/provenance.js";

const SIGNED_SHIFT = "reverse-PE corrected — median shift 0.6 mm (signed delta QC)";
const UNSIGNED_CORRECTED = "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)";
const UNCORRECTED = "no reverse-PE — ~3 mm geometric floor";

function chips(overrides = {}) {
  return provenanceChips({
    space: UNCORRECTED,
    preflightState: "absent",
    fidelitySummary: { state: "", who: "", chip: "" },
    ...overrides,
  });
}

test("space chip: signed delta-QC shift uses the served mm figure, never invents one", () => {
  const [space] = chips({ space: SIGNED_SHIFT });
  assert.equal(space.text, "DWI space, 0.6 mm shift");
  assert.equal(space.state, "signed");
});

test("space chip: uncorrected case is the geometric floor, no invented number", () => {
  const [space] = chips({ space: UNCORRECTED });
  assert.equal(space.text, "DWI space, recorded floor 3 mm");
  assert.equal(space.state, "caution");
});

test("space chip: corrected but unsigned never borrows the uncorrected floor wording", () => {
  const [space] = chips({ space: UNSIGNED_CORRECTED });
  assert.equal(space.text, "DWI space, corrected, unsigned");
  assert.equal(space.state, "caution");
});

test("preflight chip: signed / unsigned / absent", () => {
  assert.deepEqual(chips({ preflightState: "signed" })[1], { text: "Preflight signed", state: "signed" });
  assert.deepEqual(chips({ preflightState: "unsigned" })[1], { text: "Preflight unsigned", state: "caution" });
  assert.deepEqual(chips({ preflightState: "absent" })[1], { text: "Preflight absent", state: "mute" });
});

test('preflight distinguishes untestable evidence, unavailable responses, and an absent stored record',()=>{
  const criteria=[{id:'gradients',verdict:'untestable'}];
  assert.equal(preflightState({stored:null,verified:{criteria},drift:[]}), 'untestable');
  assert.deepEqual(chips({preflightState:'untestable'})[1],{text:'Preflight untestable',state:'caution'});
  assert.equal(preflightState(null),'unavailable');
  assert.equal(preflightState({}),'unavailable');
  assert.equal(preflightState({stored:null,verified:{criteria:[{id:'gradients',verdict:'pass'}]},drift:[]}), 'absent');
  assert.deepEqual(chips({preflightState:'unavailable'})[1],{text:'Preflight unavailable',state:'caution'});
});

test('a stored signature with any verified drift cannot become a signed preflight chip',()=>{
  const payload={stored:{approved_by:'reviewer'},verified:{criteria:[{id:'gradients',verdict:'pass'}]},drift:[]};
  assert.equal(preflightState(payload),'signed');
  assert.equal(preflightState({...payload,drift:['gradients']}),'unsigned');
});

test("operating-point chip: signed carries who, styled gold", () => {
  const [, , op] = chips({ fidelitySummary: { state: "signed", who: "signed erion", chip: "R=0.45 · min 0.9 · signed erion" } });
  assert.equal(op.text, "Operating point: signed erion");
  assert.equal(op.state, "signed");
});

test("operating-point chip: pilot, unsigned is caution", () => {
  const [, , op] = chips({ fidelitySummary: { state: "pilot", who: "pilot · unsigned", chip: "R=0.3 · min 0.7 · pilot · unsigned" } });
  assert.equal(op.text, "Operating point: pilot, unsigned");
  assert.equal(op.state, "caution");
});

test("operating-point chip: mixed provenance is caution + literal 'mixed', never signed", () => {
  const [, , op] = chips({
    fidelitySummary: { state: "pilot", who: "mixed operating points · unsigned", chip: "operating point: mixed" },
  });
  assert.equal(op.text, "Operating point: mixed");
  assert.equal(op.state, "caution");
});

test("operating-point chip: absent (nothing loaded) is mute", () => {
  const [, , op] = chips({ fidelitySummary: { state: "", who: "", chip: "" } });
  assert.equal(op.text, "Operating point: absent");
  assert.equal(op.state, "mute");
});

test("research chip is static and always present, fourth in order", () => {
  const all = chips();
  assert.equal(all.length, 4);
  assert.deepEqual(all[3], { text: "Research only. Not for navigation.", state: "research" });
});

test("chip order is fixed: space, preflight, operating point, research", () => {
  const all = chips({
    space: SIGNED_SHIFT,
    preflightState: "signed",
    fidelitySummary: { state: "signed", who: "signed erion", chip: "R=0.45 · min 0.9 · signed erion" },
  });
  assert.deepEqual(all.map((c) => c.text), [
    "DWI space, 0.6 mm shift",
    "Preflight signed",
    "Operating point: signed erion",
    "Research only. Not for navigation.",
  ]);
});
