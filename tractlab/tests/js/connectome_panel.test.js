import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ADR0003_NOTE,
  nodeNetworkIds,
  missingNetworkNodes,
  lutMatchesProvenance,
  aggregateNetworkMatrix,
  matrixTotal,
  topPairsInCell,
  logAlpha,
  cellFill,
  validateEdgeTubesHeaders,
} from "../../viewer/connectome_panel.js";

// A fake Headers-like object — everything validateEdgeTubesHeaders needs
// (.get(name)) without a real fetch Response, per a=10,b=13 with a complete,
// valid identity tuple.
const GEN_ID = "20260915T000000Z-abcd1234";
function fakeHeaders(overrides = {}) {
  const base = {
    "X-a": "10",
    "X-b": "13",
    "X-sourcePopulation": "edge:10-13",
    "X-edgeSourceHash": "e".repeat(64),
    "X-corpusSourceHash": "c".repeat(64),
    "X-parcellationSourceHash": "9".repeat(64),
    "X-generationId": GEN_ID,
    "X-assignmentRadiusMm": "4",
    "X-matrixPath": "matrix.csv",
    "X-matrixSha256": "1".repeat(64),
    "X-assignmentsPath": "assignments.txt",
    "X-assignmentsSha256": "2".repeat(64),
  };
  const merged = { ...base, ...overrides };
  return { get: (name) => (Object.hasOwn(merged, name) ? merged[name] : null) };
}

// A tiny synthetic 4-node matrix: nodes 1,2 -> network 1; nodes 3,4 -> network 2.
// Symmetric, zero diagonal, like a real tck2connectome output.
const MATRIX = [
  [0, 5, 2, 0],
  [5, 0, 0, 1],
  [2, 0, 0, 3],
  [0, 1, 3, 0],
];
const LABELS = ["Parcel1", "Parcel2", "Parcel3", "Parcel4"];
const LUT = {
  1: { name: "Parcel1", networkId: 1 },
  2: { name: "Parcel2", networkId: 1 },
  3: { name: "Parcel3", networkId: 2 },
  4: { name: "Parcel4", networkId: 2 },
};

test("nodeNetworkIds reads only the served LUT, never a hand-typed table", () => {
  assert.deepEqual(nodeNetworkIds(4, LUT), [1, 1, 2, 2]);
  assert.deepEqual(nodeNetworkIds(4, null), [0, 0, 0, 0], "no LUT means no invented network ids");
});

test("nodeNetworkIds ignores an out-of-range network id from a malformed LUT row", () => {
  const bad = { 1: { networkId: 9 }, 2: { networkId: 1 }, 3: { networkId: 2 }, 4: { networkId: 2 } };
  assert.deepEqual(nodeNetworkIds(4, bad), [9, 1, 2, 2]);
});

// panel_refuses_missing_malformed_or_wrong_hash_lut — the panel must fail
// closed (one honest refusal line, never a partial heatmap) on any of:
// a missing/failed LUT fetch, a node with no membership, or a served LUT
// whose hash disagrees with the connectome's own build-time record of it.
test("missingNetworkNodes: null LUT means every node is missing (fail closed, not silently zero)", () => {
  assert.deepEqual(missingNetworkNodes(4, null), [1, 2, 3, 4]);
});

test("missingNetworkNodes: a fully-covered LUT reports nothing missing", () => {
  assert.deepEqual(missingNetworkNodes(4, LUT), []);
});

test("missingNetworkNodes: one node with no membership is named, not silently dropped", () => {
  const partial = { 1: { networkId: 1 }, 2: { networkId: 1 }, 3: {}, 4: { networkId: 2 } };
  assert.deepEqual(missingNetworkNodes(4, partial), [3]);
});

test("missingNetworkNodes: an out-of-range network id (9) counts as missing", () => {
  const bad = { 1: { networkId: 9 }, 2: { networkId: 1 }, 3: { networkId: 2 }, 4: { networkId: 2 } };
  assert.deepEqual(missingNetworkNodes(4, bad), [1]);
});

const PROVENANCE = { parcellation: { lut_sha256: "a".repeat(64) } };

test("lutMatchesProvenance: exact match passes", () => {
  assert.equal(lutMatchesProvenance("a".repeat(64), PROVENANCE), true);
});

test("lutMatchesProvenance: a different served hash refuses (malformed or stale LUT)", () => {
  assert.equal(lutMatchesProvenance("b".repeat(64), PROVENANCE), false);
});

test("lutMatchesProvenance: a missing served hash (failed fetch) refuses", () => {
  assert.equal(lutMatchesProvenance(null, PROVENANCE), false);
  assert.equal(lutMatchesProvenance(undefined, PROVENANCE), false);
  assert.equal(lutMatchesProvenance("", PROVENANCE), false);
});

test("lutMatchesProvenance: a connectome provenance with no recorded LUT hash never passes", () => {
  assert.equal(lutMatchesProvenance("a".repeat(64), { parcellation: {} }), false);
  assert.equal(lutMatchesProvenance("a".repeat(64), null), false);
});

test("aggregateNetworkMatrix sums raw counts into a 7x7 grid, diagonal included", () => {
  const netIds = nodeNetworkIds(4, LUT);
  const cells = aggregateNetworkMatrix(MATRIX, netIds);
  assert.equal(cells.length, 7);
  assert.ok(cells.every((row) => row.length === 7));
  // network1<->network1 (diagonal cell): node1-node2 edge counted twice (i->j and j->i), value 5 each way.
  assert.equal(cells[0][0], 10);
  // network2<->network2: node3-node4 edge, 3 each way.
  assert.equal(cells[1][1], 6);
  // network1->network2 sums only i in network1, j in network2: (1,3)=2, (2,4)=1 -> 3;
  // the mirror cell (network2->network1) sums the same edges the other way and is equal.
  assert.equal(cells[0][1], 3);
  assert.equal(cells[1][0], 3);
});

test("aggregateNetworkMatrix accounts for the whole matrix when every node has a valid network id", () => {
  const netIds = nodeNetworkIds(4, LUT);
  const cells = aggregateNetworkMatrix(MATRIX, netIds);
  const cellSum = cells.flat().reduce((a, b) => a + b, 0);
  assert.equal(cellSum, matrixTotal(MATRIX));
});

test("aggregateNetworkMatrix drops nodes with no network id (0) rather than mis-bucketing them", () => {
  const netIds = [1, 1, 0, 2]; // node 3 unassigned in the LUT
  const cells = aggregateNetworkMatrix(MATRIX, netIds);
  const cellSum = cells.flat().reduce((a, b) => a + b, 0);
  assert.ok(cellSum < matrixTotal(MATRIX), "edges touching an unassigned node are not silently counted");
});

test("matrixTotal sums every cell including zeros", () => {
  assert.equal(matrixTotal(MATRIX), 2 * (5 + 2 + 1 + 3));
  assert.equal(matrixTotal([[0]]), 0);
});

test("topPairsInCell lists each unordered pair once, sorted by count, capped at k", () => {
  const netIds = nodeNetworkIds(4, LUT);
  const withinNet1 = topPairsInCell(MATRIX, LABELS, netIds, 1, 1, 12);
  assert.equal(withinNet1.length, 1, "node1-node2 is the only within-network-1 pair");
  assert.equal(withinNet1[0].count, 5);
  assert.equal(withinNet1[0].a, 1);
  assert.equal(withinNet1[0].b, 2);

  const acrossNets = topPairsInCell(MATRIX, LABELS, netIds, 1, 2, 12);
  assert.equal(acrossNets.length, 2);
  assert.deepEqual(acrossNets.map((p) => p.count), [2, 1], "sorted descending by count");
  assert.ok(acrossNets.every((p) => p.count > 0), "zero-count pairs are dropped, never listed");
});

test("topPairsInCell caps at k even when more pairs exist", () => {
  const n = 5;
  const matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 0 : i + j + 1)));
  const netIds = new Array(n).fill(1);
  const labels = matrix.map((_, i) => `P${i + 1}`);
  const pairs = topPairsInCell(matrix, labels, netIds, 1, 1, 3);
  assert.equal(pairs.length, 3);
  assert.ok(pairs[0].count >= pairs[1].count && pairs[1].count >= pairs[2].count);
});

test("topPairsInCell returns nothing for an empty cell", () => {
  const netIds = nodeNetworkIds(4, LUT);
  assert.deepEqual(topPairsInCell(MATRIX, LABELS, netIds, 1, 5, 12), []);
});

test("logAlpha is 0 with no evidence and monotone increasing with count", () => {
  assert.equal(logAlpha(0, 100), 0);
  assert.equal(logAlpha(5, 0), 0);
  const low = logAlpha(1, 100);
  const high = logAlpha(50, 100);
  assert.ok(low > 0 && low < high && high <= 1);
  assert.equal(logAlpha(100, 100), 1);
});

test("cellFill is a distinct, transparent colour at zero evidence", () => {
  assert.equal(cellFill(0, 100), "rgba(79,140,255,0.04)");
  assert.notEqual(cellFill(50, 100), cellFill(0, 100));
  assert.match(cellFill(50, 100), /^rgba\(79,140,255,0\.\d+\)$/);
});

test("the ADR-0003 note is exported for reuse, not restated ad hoc", () => {
  assert.match(ADR0003_NOTE, /ASSIGNED/);
  assert.match(ADR0003_NOTE, /ADR-0003/);
});

// ---------------------------------------------------------------------------
// W8 final round, item 1 — pre-render identity guard completeness.
// validateEdgeTubesHeaders is the single source of truth showTrackResponse
// calls before upserting an edge layer.
// ---------------------------------------------------------------------------

test("validateEdgeTubesHeaders: a complete, matching response passes", () => {
  const result = validateEdgeTubesHeaders(fakeHeaders(), { a: 10, b: 13, expectedGenerationId: GEN_ID });
  assert.equal(result.ok, true);
  assert.equal(result.reason, null);
  assert.equal(result.radiusMm, 4);
});

test("load_edge_tubes_refuses_missing_radius_or_count_provenance_before_upsert", () => {
  // Missing radius header entirely (Number("") -> NaN) must refuse, never
  // silently default to 0 and render an unlabeled/zero-radius ASSIGNED chip.
  const missingRadius = validateEdgeTubesHeaders(fakeHeaders({ "X-assignmentRadiusMm": null }), { a: 10, b: 13 });
  assert.equal(missingRadius.ok, false);
  assert.equal(missingRadius.radiusMm, null);

  // A literal "0" must also refuse — radius is never a valid 0.
  const zeroRadius = validateEdgeTubesHeaders(fakeHeaders({ "X-assignmentRadiusMm": "0" }), { a: 10, b: 13 });
  assert.equal(zeroRadius.ok, false);

  const negativeRadius = validateEdgeTubesHeaders(fakeHeaders({ "X-assignmentRadiusMm": "-4" }), { a: 10, b: 13 });
  assert.equal(negativeRadius.ok, false);

  for (const field of ["X-matrixPath", "X-matrixSha256", "X-assignmentsPath", "X-assignmentsSha256"]) {
    const missing = validateEdgeTubesHeaders(fakeHeaders({ [field]: null }), { a: 10, b: 13 });
    assert.equal(missing.ok, false, `${field} missing must refuse`);
    const blank = validateEdgeTubesHeaders(fakeHeaders({ [field]: "" }), { a: 10, b: 13 });
    assert.equal(blank.ok, false, `${field} blank must refuse`);
  }
  // A non-hex or short "sha256" must refuse too (a placeholder value must not pass).
  const badMatrixSha = validateEdgeTubesHeaders(fakeHeaders({ "X-matrixSha256": "not-a-real-hash" }), { a: 10, b: 13 });
  assert.equal(badMatrixSha.ok, false);
});

// edge_zero_line_response_refuses_incomplete_identity — a zero-streamline
// response is still just a response: validateEdgeTubesHeaders knows nothing
// about line count, so an incomplete identity is refused exactly the same
// whether or not there happen to be any streamlines (the browser harness
// separately proves the lc===0 path in showTrackResponse actually calls this
// guard before its own early return).
test("edge_zero_line_response_refuses_incomplete_identity", () => {
  const incomplete = validateEdgeTubesHeaders(
    fakeHeaders({ "X-lineCount": "0", "X-edgeSourceHash": "" }),
    { a: 10, b: 13 },
  );
  assert.equal(incomplete.ok, false);
  assert.match(incomplete.reason, /does not match the requested pair or reported incomplete identity/);

  const completeButEmpty = validateEdgeTubesHeaders(fakeHeaders({ "X-lineCount": "0" }), { a: 10, b: 13 });
  assert.equal(completeButEmpty.ok, true, "a complete identity with zero streamlines is not itself a refusal reason");
});

test("load_edge_tubes_refuses_panel_generation_mismatch", () => {
  const mismatch = validateEdgeTubesHeaders(fakeHeaders(), { a: 10, b: 13, expectedGenerationId: "a-different-generation" });
  assert.equal(mismatch.ok, false);
  assert.equal(mismatch.reason, "connectome rebuilt, reopen the panel");

  // No expectedGenerationId given (e.g. the panel never successfully
  // fetched one) -> the generation check is skipped, not a false refusal.
  const noExpectation = validateEdgeTubesHeaders(fakeHeaders(), { a: 10, b: 13, expectedGenerationId: null });
  assert.equal(noExpectation.ok, true);
});

test("validateEdgeTubesHeaders refuses a mismatched pair even with otherwise-complete identity", () => {
  const wrongPair = validateEdgeTubesHeaders(fakeHeaders({ "X-a": "99" }), { a: 10, b: 13 });
  assert.equal(wrongPair.ok, false);
  const wrongSourcePopulation = validateEdgeTubesHeaders(
    fakeHeaders({ "X-sourcePopulation": "edge:99-100" }), { a: 10, b: 13 },
  );
  assert.equal(wrongSourcePopulation.ok, false);
});
