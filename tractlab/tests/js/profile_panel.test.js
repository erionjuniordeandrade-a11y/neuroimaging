import test from "node:test";
import assert from "node:assert/strict";
import {
  bandSegments,
  closestApproach,
  closestApproachLabel,
  endLabels,
  flaggedNodes,
  intervalPointMask,
  isSha256,
  isStreamlineFlipped,
  normalizeInterval,
  pointRangeForInterval,
  familyFromBankId,
  profileErrorLine,
  provenanceRows,
  renderProfilePanel,
  sideFromBankId,
  SERVED_CLAIM,
  roiShortName,
  scalarAxis,
  shortHash,
  validateProfilePayload,
  xToNode,
  CHART,
  NEAR_LESION_MM,
  TRACK_MAX_MM,
} from "../../viewer/profile_panel.js";
import { DIST_MAX, FAR_FADE_START_MM } from "../../viewer/tract_colour.js";

// ---- axis labels come from the served orientation rule ----

test("cst orientation labels the inferior anchor end first", () => {
  assert.deepEqual(
    endLabels({ rule: "inferior", family: "cst", side: "r" }),
    { start: "inferior", end: "superior" },
  );
});

test("roi_centroid names the served ROI file, not the family", () => {
  const orientation = {
    rule: "roi_centroid", family: "fat", side: "r",
    roi_path: "tracts/roi/fat_r_sfg_dil1.nii.gz",
  };
  assert.equal(roiShortName(orientation), "sfg");
  assert.deepEqual(endLabels(orientation), { start: "ROI: sfg", end: "far end" });
  assert.equal(
    roiShortName({ rule: "roi_centroid", family: "cing", side: "l", roi_path: "tracts/roi/cing_l_ant_dil2.nii.gz" }),
    "ant",
  );
});

test("an roi_centroid rule with no served ROI path does not invent one", () => {
  assert.deepEqual(
    endLabels({ rule: "roi_centroid", family: "fat", side: "r" }),
    { start: "ROI", end: "far end" },
  );
});

test("the anterior fallback rule labels both ends", () => {
  assert.deepEqual(
    endLabels({ rule: "anterior", family: "fat", side: "r" }),
    { start: "anterior", end: "posterior" },
  );
});

test("an unrecorded rule falls back to node numbers rather than a guess", () => {
  assert.deepEqual(endLabels({}), { start: "node 0", end: "node 99" });
});

// ---- band gaps come from the served flags ----

test("flagged nodes break the band instead of being smoothed over", () => {
  const nodes = {
    median: [0.1, 0.2, 0.3, 0.4, 0.5],
    p25: [0.0, 0.1, 0.2, 0.3, 0.4],
    p75: [0.2, 0.3, 0.4, 0.5, 0.6],
    zero_or_missing_flag: [false, false, true, false, false],
  };
  assert.deepEqual(bandSegments(nodes), [{ from: 0, to: 1 }, { from: 3, to: 4 }]);
  assert.deepEqual(flaggedNodes(nodes), [2]);
});

test("a node with no usable value is a gap even when it carries no flag", () => {
  const nodes = {
    median: [0.1, null, 0.3],
    p25: [0.0, null, 0.2],
    p75: [0.2, null, 0.4],
    zero_or_missing_flag: [false, false, false],
  };
  assert.deepEqual(bandSegments(nodes), [{ from: 0, to: 0 }, { from: 2, to: 2 }]);
  assert.deepEqual(flaggedNodes(nodes), []);
});

test("an all-usable profile is one segment of every node", () => {
  const n = 100;
  const nodes = {
    median: Array.from({ length: n }, (_, i) => i / n),
    p25: Array.from({ length: n }, (_, i) => i / n - 0.05),
    p75: Array.from({ length: n }, (_, i) => i / n + 0.05),
    zero_or_missing_flag: Array.from({ length: n }, () => false),
  };
  assert.deepEqual(bandSegments(nodes), [{ from: 0, to: 99 }]);
});

// ---- interval -> vertex mapping ----

test("a node interval maps onto the tube points that cover it", () => {
  // 100 nodes over 64 tube points: node i sits at point i * 63 / 99.
  assert.deepEqual(pointRangeForInterval(64, 100, [0, 99]), { lo: 0, hi: 63 });
  assert.deepEqual(pointRangeForInterval(64, 100, [0, 0]), { lo: 0, hi: 0 });
  assert.deepEqual(pointRangeForInterval(64, 100, [99, 99]), { lo: 63, hi: 63 });
  // 50..60 -> 31.81..38.18 -> points 32..38
  assert.deepEqual(pointRangeForInterval(64, 100, [50, 60]), { lo: 32, hi: 38 });
});

test("an interval narrower than one tube step still selects its nearest point", () => {
  const range = pointRangeForInterval(64, 100, [50, 50]);
  assert.ok(range.lo <= range.hi, "a selection is never empty");
  assert.equal(range.lo, Math.round((50 / 99) * 63));
});

test("a reversed drag and out-of-range nodes are clamped, not refused", () => {
  assert.deepEqual(normalizeInterval(80, 20, 100), [20, 80]);
  assert.deepEqual(normalizeInterval(-5, 400, 100), [0, 99]);
  assert.equal(pointRangeForInterval(1, 100, [0, 10]), null);
  assert.equal(pointRangeForInterval(64, 100, [0]), null);
});

test("a streamline whose last point is nearer the anchor is flipped", () => {
  // k = 3, two lines. Line 0 runs away from the anchor at the origin,
  // line 1 runs toward it.
  const k = 3;
  const pts = Float32Array.from([
    0, 0, 0, 0, 0, 5, 0, 0, 10,
    0, 0, 10, 0, 0, 5, 0, 0, 0,
  ]);
  assert.equal(isStreamlineFlipped(pts, 0, k, [0, 0, 0]), false);
  assert.equal(isStreamlineFlipped(pts, 1, k, [0, 0, 0]), true);
});

test("the interval mask follows each streamline's own orientation", () => {
  const k = 3;
  const nPoints = 3;
  const pts = Float32Array.from([
    0, 0, 0, 0, 0, 5, 0, 0, 10,   // line 0: node 0 is packed point 0
    0, 0, 10, 0, 0, 5, 0, 0, 0,   // line 1: node 0 is packed point 2
  ]);
  const mask = intervalPointMask({
    pts, lineCount: 2, k, nPoints, interval: [0, 0], anchorMm: [0, 0, 0],
  });
  assert.equal(mask.length, 6);
  assert.deepEqual(Array.from(mask.subarray(0, 3)), [1, 0, 0]);
  assert.deepEqual(Array.from(mask.subarray(3, 6)), [0, 0, 1]);
});

test("the whole-profile interval keeps every tube point", () => {
  const k = 4;
  const pts = Float32Array.from(Array.from({ length: 2 * k * 3 }, (_, i) => i));
  const mask = intervalPointMask({
    pts, lineCount: 2, k, nPoints: 100, interval: [0, 99], anchorMm: [0, 0, 0],
  });
  assert.equal(mask.reduce((sum, value) => sum + value, 0), 2 * k);
});

test("an absent anchor leaves the mask unflipped rather than guessing", () => {
  const k = 3;
  const pts = Float32Array.from([0, 0, 10, 0, 0, 5, 0, 0, 0]);
  const mask = intervalPointMask({
    pts, lineCount: 1, k, nPoints: 3, interval: [0, 0], anchorMm: null,
  });
  assert.deepEqual(Array.from(mask), [1, 0, 0]);
});

// ---- error lines ----

test("each served refusal code maps to one honest line", () => {
  assert.equal(profileErrorLine("scalar_missing", "md"), "MD map not available for this case");
  assert.equal(profileErrorLine("scalar_missing", "fa"), "FA map not available for this case");
  assert.equal(profileErrorLine("orientation_unknown_family"), "No orientation rule for this tract family");
  assert.equal(profileErrorLine("unknown_bank"), "This tract is not a named bank in this case");
  assert.equal(profileErrorLine("bank_source_changed"), "The tract file changed on disk, reload the case");
  assert.equal(profileErrorLine("hash_mismatch"), "The tract file changed on disk, reload the case");
  assert.equal(profileErrorLine("http_500"), "Profile unavailable for this tract");
});

test("no error line claims a score, a risk or a margin", () => {
  const forbidden = /\b(margin|navigation|at[\s-]risk|abnormal|validated)\b/i;
  for (const code of [
    "scalar_missing", "scalar_unsupported", "orientation_unknown_family", "invalid_bank_id",
    "unknown_bank", "bank_source_changed", "hash_mismatch", "lesion_invalid", "http_500",
  ]) {
    assert.equal(forbidden.test(profileErrorLine(code)), false, code);
  }
});

// ---- axes, thresholds, provenance ----

test("the lesion track reuses the proximity ramp's own thresholds", () => {
  assert.equal(NEAR_LESION_MM, FAR_FADE_START_MM);
  assert.equal(TRACK_MAX_MM, DIST_MAX);
});

test("FA and MD axes declare their own units", () => {
  assert.equal(scalarAxis("fa").label, "FA");
  assert.equal(scalarAxis("fa").max, 1);
  assert.equal(scalarAxis("fa").scale, 1);
  const md = scalarAxis("md");
  assert.match(md.label, /10⁻³ mm²\/s/);
  assert.equal(md.scale, 1000);
});

const BANK_SHA = "8bf1f924960580acf7f8f388144148b2cf3a35074a0608ec4fa21d6128c604a6";
const SCALAR_SHA = "7cbc153abc815d153ae3563411c589fba563f8f4d4f26a7677acb248d66fec52";
const ROI_SHA = "63be707987e9f1045f4a468da7f9cc479282b9b562093e65139648e12de2a699";
const LESION_SHA = "a".repeat(64);

function servedPayload(overrides = {}) {
  const n = 100;
  const column = (value) => Array.from({ length: n }, () => value);
  return {
    bank_id: "bank_cst_r",
    bank_path: "tracts/bank/cst_r_motor_pons.tck",
    bank_sha256: BANK_SHA,
    scalar: "fa",
    scalar_path: "work/profiles/fa_casemask.nii.gz",
    scalar_sha256: SCALAR_SHA,
    n_points: n,
    n_streamlines: 8303,
    orientation: { rule: "inferior", anchor_mm: [-3.1, -20.8, -73.0], family: "cst", side: "r", n_flipped: 6194 },
    active_derivation: "d1",
    resample_method: "numpy_arc_length",
    claim: "descriptive profile; not a normative abnormality score",
    nodes: {
      median: column(0.4), p25: column(0.3), p75: column(0.5), mean: column(0.4),
      n: column(8303), n_valid: column(8303), n_zero: column(0), n_nan: column(0),
      zero_or_missing_flag: column(false),
    },
    zero_or_missing_threshold: 0.2,
    zero_or_missing_warning: null,
    lesion_distance: { track_mm: null, reason: "no lesion mask" },
    lesion_path: null,
    lesion_sha256: null,
    ...overrides,
  };
}

const SOURCE_A = "1".repeat(64);
const SOURCE_B = "2".repeat(64);
const asked = {
  bankId: "bank_cst_r", scalar: "fa", activeDerivation: "d1",
  sourceHash: { expected: SOURCE_A, current: SOURCE_A },
};

test("profile_response_refuses_invalid_or_mismatched_payload", () => {
  assert.equal(validateProfilePayload(servedPayload(), asked), null);

  assert.equal(validateProfilePayload(null, asked), "payload_invalid");
  assert.equal(validateProfilePayload("ok", asked), "payload_invalid");
  assert.equal(validateProfilePayload([], asked), "payload_invalid");
  assert.equal(validateProfilePayload(servedPayload({ bank_id: "bank_fat_r" }), asked), "bank_mismatch");
  assert.equal(validateProfilePayload(servedPayload({ scalar: "md" }), asked), "scalar_mismatch");
  assert.equal(validateProfilePayload(servedPayload({ active_derivation: "d2" }), asked), "derivation_mismatch");

  // Missing or short digests are refused, never rendered as "unrecorded".
  assert.equal(validateProfilePayload(servedPayload({ bank_sha256: null }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ bank_sha256: BANK_SHA.slice(0, 32) }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ scalar_sha256: "" }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ bank_path: "" }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ scalar_path: null }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ active_derivation: null }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({ orientation: { rule: "inferior" } }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({
    orientation: { rule: "roi_centroid", anchor_mm: [0, 0, 0], roi_path: "tracts/roi/fat_r_sfg_dil1.nii.gz" },
  }), asked), "provenance_missing");
  assert.equal(validateProfilePayload(servedPayload({
    lesion_path: "nifti/lesion.nii.gz", lesion_sha256: null,
  }), asked), "provenance_missing");

  // Node arrays must be exactly n_points long, and so must the lesion track.
  const short = servedPayload();
  short.nodes.p25 = short.nodes.p25.slice(0, 40);
  assert.equal(validateProfilePayload(short, asked), "nodes_malformed");
  assert.equal(validateProfilePayload(servedPayload({ nodes: null }), asked), "nodes_malformed");
  assert.equal(validateProfilePayload(servedPayload({ n_points: 1 }), asked), "nodes_malformed");
  assert.equal(validateProfilePayload(servedPayload({
    lesion_distance: { track_mm: [1, 2, 3], reason: null },
  }), asked), "nodes_malformed");

  // The bank file behind the layer changed since the profile was read.
  assert.equal(validateProfilePayload(servedPayload(), {
    ...asked, sourceHash: { expected: SOURCE_A, current: SOURCE_B },
  }), "bank_source_changed");
});

test("profile_response_started_on_source_A_refuses_after_reload_B", () => {
  // The digest captured when the request left is the one that must still be
  // on the layer when the answer arrives; a reload in between is a refusal,
  // not a profile drawn over freshly reloaded geometry.
  assert.equal(validateProfilePayload(servedPayload(), {
    ...asked, sourceHash: { expected: SOURCE_A, current: SOURCE_A },
  }), null);
  assert.equal(validateProfilePayload(servedPayload(), {
    ...asked, sourceHash: { expected: SOURCE_A, current: SOURCE_B },
  }), "bank_source_changed");
  // The layer disappearing mid-flight is also a mismatch, never a pass.
  assert.equal(validateProfilePayload(servedPayload(), {
    ...asked, sourceHash: { expected: SOURCE_A, current: null },
  }), "bank_source_changed");
  assert.equal(profileErrorLine("bank_source_changed"), "The tract file changed on disk, reload the case");
});

test("profile_response_requires_layer_source_hash", () => {
  for (const sourceHash of [undefined, null, {}, { current: SOURCE_A },
    { expected: null, current: SOURCE_A }, { expected: "", current: SOURCE_A }]) {
    assert.equal(
      validateProfilePayload(servedPayload(), { ...asked, sourceHash }),
      "bank_source_unknown",
      JSON.stringify(sourceHash ?? null),
    );
  }
  assert.equal(profileErrorLine("bank_source_unknown"), "This tract carries no source digest to check");
});

test("profile_response_refuses_when_active_derivation_unknown", () => {
  // /api/derivation failed, so the viewer does not know its own lineage. The
  // comparison must not be skipped: with no authority, nothing is accepted.
  for (const activeDerivation of [null, undefined, ""]) {
    assert.equal(
      validateProfilePayload(servedPayload(), { ...asked, activeDerivation }),
      "derivation_unknown",
      String(activeDerivation),
    );
  }
  // Even a payload that is perfect in every other way.
  assert.equal(validateProfilePayload(servedPayload({ active_derivation: "d1" }),
    { ...asked, activeDerivation: null }), "derivation_unknown");
  assert.equal(profileErrorLine("derivation_unknown"), "Derivation status unavailable");
});

test("profile_response_refuses_orientation_identity_mismatch", () => {
  const orient = (extra) => servedPayload({
    orientation: { rule: "inferior", anchor_mm: [0, 0, 0], family: "cst", side: "r", n_flipped: 1, ...extra },
  });
  assert.equal(validateProfilePayload(orient(), asked), null);
  // Family and side must be the ones the structured id declares.
  assert.equal(validateProfilePayload(orient({ family: "fat" }), asked), "orientation_mismatch");
  assert.equal(validateProfilePayload(orient({ side: "l" }), asked), "orientation_mismatch");
  assert.equal(validateProfilePayload(orient({ family: "" }), asked), "orientation_mismatch");
  assert.equal(validateProfilePayload(orient({ side: null }), asked), "orientation_mismatch");
  // The rule must be one profile.py can actually declare.
  assert.equal(validateProfilePayload(orient({ rule: "superior" }), asked), "orientation_mismatch");
  assert.equal(validateProfilePayload(orient({ rule: "roi_centroid", roi_path: "tracts/roi/cst_r_pons_dil1.nii.gz",
    roi_sha256: ROI_SHA }), asked), null);

  // The viewer's own token rule decides the side; an id with no usable side
  // token, or a conflicting one, cannot be verified and is refused.
  assert.equal(familyFromBankId("bank_slf3_l"), "slf3");
  assert.equal(familyFromBankId("bank_nope_l"), null);
  assert.equal(familyFromBankId("cst_r"), null);
  assert.equal(sideFromBankId("bank_cst_r"), "r");
  assert.equal(sideFromBankId("bank_fat_r_soft"), "r");
  assert.equal(sideFromBankId("bank_cst_l_r"), null, "a conflicting token is never a hemisphere");
  assert.equal(
    validateProfilePayload(servedPayload({ bank_id: "bank_cc_forceps_minor" }),
      { ...asked, bankId: "bank_cc_forceps_minor" }),
    "orientation_mismatch",
  );
  assert.equal(profileErrorLine("orientation_mismatch"),
    "The profile orientation does not match this tract's identity");
});

test("profile_response_refuses_nonfinite_or_null_measurements_and_missing_or_forbidden_claim", () => {
  const withNodes = (mutate) => {
    const body = servedPayload();
    mutate(body.nodes);
    return body;
  };
  // A node with no usable sample is null; anything else must be a real number.
  // JSON.parse turns 1e999 into Infinity, so this is reachable over the wire.
  assert.equal(validateProfilePayload(withNodes((n) => { n.median[3] = null; }), asked), null,
    "a genuinely empty node stays null and is drawn as a gap");
  for (const bad of [Infinity, -Infinity, NaN, "0.4", true, []]) {
    assert.equal(validateProfilePayload(withNodes((n) => { n.median[3] = bad; }), asked),
      "measurements_invalid", String(bad));
  }
  assert.equal(validateProfilePayload(withNodes((n) => { n.p75[0] = {}; }), asked), "measurements_invalid");
  assert.equal(validateProfilePayload(withNodes((n) => { n.mean[99] = "x"; }), asked), "measurements_invalid");
  // Counts are whole and non-negative, and never null.
  for (const bad of [-1, 1.5, null, "8303", Infinity]) {
    assert.equal(validateProfilePayload(withNodes((n) => { n.n_valid[10] = bad; }), asked),
      "measurements_invalid", String(bad));
  }
  assert.equal(validateProfilePayload(withNodes((n) => { n.n_zero[0] = -0.0001; }), asked), "measurements_invalid");
  // Flags are booleans, not truthy stand-ins.
  assert.equal(validateProfilePayload(withNodes((n) => { n.zero_or_missing_flag[5] = 1; }), asked),
    "measurements_invalid");

  // The lesion track is all-or-nothing, finite, and never negative.
  const track = (values) => servedPayload({ lesion_distance: { track_mm: values, reason: null } });
  const good = Array.from({ length: 100 }, (_, i) => i * 0.25);
  assert.equal(validateProfilePayload(track(good), asked), null);
  assert.equal(validateProfilePayload(servedPayload(), asked), null, "a null track is allowed");
  for (const bad of [null, -0.5, Infinity, NaN, "1.0"]) {
    const values = [...good];
    values[42] = bad;
    assert.equal(validateProfilePayload(track(values), asked), "measurements_invalid", String(bad));
  }

  // The claim is contract text, and pilot language is owner law.
  assert.equal(validateProfilePayload(servedPayload({ claim: undefined }), asked), "claim_invalid");
  assert.equal(validateProfilePayload(servedPayload({ claim: "" }), asked), "claim_invalid");
  assert.equal(validateProfilePayload(servedPayload({ claim: "descriptive profile" }), asked), "claim_invalid");
  assert.equal(validateProfilePayload(servedPayload({ claim: SERVED_CLAIM.toUpperCase() }), asked), "claim_invalid");
  for (const claim of [
    "clinically validated abnormality score",
    "descriptive profile; validated against a normative cohort",
    "VALIDATED profile",
  ]) {
    assert.equal(validateProfilePayload(servedPayload({ claim }), asked), "claim_forbidden", claim);
  }
  assert.equal(profileErrorLine("measurements_invalid"), "This profile arrived with unusable measurements");
  assert.equal(profileErrorLine("claim_invalid"), "This profile did not carry the served claim");
  assert.equal(profileErrorLine("claim_forbidden"),
    "This profile claims validation, which this instrument never asserts");
});

test("profile_panel_refuses_missing_provenance_and_displays_all_paths", () => {
  const rows = new Map(provenanceRows(servedPayload({
    orientation: {
      rule: "roi_centroid", anchor_mm: [1, 2, 3], family: "fat", side: "r", n_flipped: 16398,
      roi_path: "tracts/roi/fat_r_sfg_dil1.nii.gz", roi_sha256: ROI_SHA,
    },
    lesion_path: "nifti/lesion.nii.gz",
    lesion_sha256: LESION_SHA,
    lesion_distance: { track_mm: Array.from({ length: 100 }, (_, i) => i * 0.3), reason: null },
  })));
  assert.equal(rows.get("bank file"), "tracts/bank/cst_r_motor_pons.tck");
  assert.equal(rows.get("bank sha256"), "8bf1f924");
  assert.equal(rows.get("scalar file"), "work/profiles/fa_casemask.nii.gz");
  assert.equal(rows.get("scalar sha256"), "7cbc153a");
  assert.equal(rows.get("orientation ROI"), "tracts/roi/fat_r_sfg_dil1.nii.gz");
  assert.equal(rows.get("ROI sha256"), ROI_SHA.slice(0, 8));
  assert.equal(rows.get("lesion mask"), "nifti/lesion.nii.gz");
  assert.equal(rows.get("lesion sha256"), LESION_SHA.slice(0, 8));
  assert.equal(rows.get("near band"), `under ${NEAR_LESION_MM} mm, track axis to ${TRACK_MAX_MM} mm`);
  assert.equal(rows.get("derivation"), "d1");

  // Nothing is ever printed as "unrecorded": a payload missing a field is
  // refused upstream, and a row with no served value is simply absent.
  const lean = new Map(provenanceRows(servedPayload()));
  assert.equal(lean.has("orientation ROI"), false);
  assert.equal(lean.has("lesion mask"), false);
  for (const [, value] of provenanceRows(servedPayload())) {
    assert.notEqual(value, "unrecorded");
    assert.ok(value !== null && value !== "");
  }

  // A digest must be a full sha256 to be shortened at all.
  assert.equal(isSha256(BANK_SHA), true);
  assert.equal(isSha256(BANK_SHA.slice(0, 63)), false);
  assert.equal(isSha256(`${BANK_SHA}0`), false);
  assert.equal(isSha256("zz"), false);
  assert.equal(shortHash(null), null);
  assert.equal(shortHash("8bf1f924"), null);
  assert.equal(shortHash(BANK_SHA), "8bf1f924");
});

test("the closest approach is read off the served track, not recomputed", () => {
  const track = Array.from({ length: 100 }, (_, i) => 20 - Math.abs(i - 75) * 0.2);
  track[75] = 0.6;
  const payload = servedPayload({ lesion_distance: { track_mm: track, reason: null } });
  assert.deepEqual(closestApproach(payload), { node: 75, mm: 0.6 });
  assert.equal(closestApproachLabel(payload), "closest 0.6 mm, node 75");
  assert.equal(closestApproach(servedPayload()), null);
  assert.equal(closestApproachLabel(servedPayload()), null);
});

test("x maps back to the node it was drawn from", () => {
  assert.equal(xToNode(CHART.left, 100, CHART), 0);
  assert.equal(xToNode(CHART.right, 100, CHART), 99);
  assert.equal(xToNode(CHART.left - 50, 100, CHART), 0);
  assert.equal(xToNode(CHART.right + 50, 100, CHART), 99);
});

// ---- panel states, against a minimal fake DOM ----
// The module's DOM surface is small and deliberate (create, append, attributes,
// dataset, hidden); a fake keeps these states testable in plain node.

function fakeDom() {
  const make = (tagName) => {
    const node = {
      tagName, children: [], attributes: {}, dataset: {},
      _text: "", hidden: false, id: "", className: "", type: "",
      classList: { add(name) { node.className = `${node.className} ${name}`.trim(); } },
      setAttribute(name, value) {
        node.attributes[name] = String(value);
        // The real DOM reflects these two onto properties; the fake must too,
        // or an SVG node created with setAttribute("id") would be unfindable.
        if (name === "id") node.id = String(value);
        if (name === "class") node.className = String(value);
      },
      getAttribute(name) { return Object.hasOwn(node.attributes, name) ? node.attributes[name] : null; },
      append(...kids) { node.children.push(...kids); },
      replaceChildren(...kids) { node.children = [...kids]; },
      get textContent() {
        return node.children.length
          ? node.children.map((kid) => kid.textContent).join("")
          : node._text;
      },
      set textContent(value) { node._text = String(value); node.children = []; },
    };
    return node;
  };
  return {
    createElement: make,
    createElementNS: (_ns, tag) => make(tag),
    createTextNode: (value) => ({ tagName: "#text", children: [], textContent: String(value) }),
  };
}

function withFakeDom(run) {
  const previous = globalThis.document;
  globalThis.document = fakeDom();
  try { return run(globalThis.document); } finally { globalThis.document = previous; }
}

const findById = (node, id) => {
  if (node.id === id) return node;
  for (const kid of node.children || []) {
    const hit = findById(kid, id);
    if (hit) return hit;
  }
  return null;
};

test("profile_panel_hides_after_last_unsupported_layer_removed", () => {
  withFakeDom((doc) => {
    const host = doc.createElement("div");

    // A focused layer with no named-bank identity gets one short line.
    const unsupported = renderProfilePanel(host, { status: "unsupported" });
    assert.ok(unsupported, "an unsupported layer still shows the panel");
    assert.equal(host.children.length, 1);
    assert.equal(unsupported.textContent, "Profile needs a named tract");
    assert.equal(findById(unsupported, "profileChart"), null, "no chart is drawn");

    // Removing that last layer leaves nothing focused: the panel goes away
    // rather than lingering with stale copy.
    const absent = renderProfilePanel(host, { status: "absent" });
    assert.equal(absent, null, "no panel is built when nothing is focused");
    assert.equal(host.children.length, 0, "the host is emptied");
  });
});

test("the panel never prints its own title (the dock summary owns it)", () => {
  withFakeDom((doc) => {
    const host = doc.createElement("div");
    const panel = renderProfilePanel(host, {
      status: "ready", scalar: "fa", hue: [0.9, 0.8, 0.3], payload: servedPayload(),
    });
    const titles = [];
    (function walk(node) {
      if (/section-title/.test(node.className || "")) titles.push(node);
      for (const kid of node.children || []) walk(kid);
    })(panel);
    assert.deepEqual(titles, [], "the panel adds no heading of its own");
    assert.equal(findById(panel, "profileScalarName").textContent, "FA", "the scalar names the y axis");
    assert.equal(findById(panel, "profileHint").textContent, "Drag to highlight");
    assert.equal(findById(panel, "profileClaim").textContent,
      "descriptive profile; not a normative abnormality score");
    assert.ok(findById(panel, "profileChipBankHash").textContent.startsWith("bank "));
    // No lesion in this payload: one line, and no closest-approach marker.
    assert.equal(findById(panel, "profileClosestMarker"), null);
  });
});

test("a loading panel shows a skeleton and an error panel shows one line", () => {
  withFakeDom((doc) => {
    const loading = renderProfilePanel(doc.createElement("div"), { status: "loading", scalar: "fa" });
    assert.ok(findById(loading, "profileSkeleton"), "a skeleton stands in while loading");
    assert.equal(findById(loading, "profileChart"), null);

    const failed = renderProfilePanel(doc.createElement("div"), {
      status: "error", code: "scalar_missing", scalar: "md",
    });
    assert.equal(findById(failed, "profileError").textContent, "MD map not available for this case");
    assert.equal(findById(failed, "profileProvCard"), null, "a refused payload shows no provenance");
  });
});
