import test from "node:test";
import assert from "node:assert/strict";

import {
  decodeSourceOrdinals,
  streamlineIndexFromSegmentEntry,
  summarizeStreamline,
} from "../../viewer/streamline_inspector.js";

function headers(entries) {
  const values = new Map(
    Object.entries(entries).map(([key, value]) => [key.toLowerCase(), String(value)]),
  );
  return { get: (key) => values.get(String(key).toLowerCase()) ?? null };
}

test("source ordinal decoder copies the exact displayed-to-source mapping", () => {
  const buffer = new ArrayBuffer(40);
  new Uint32Array(buffer, 12, 3).set([17, 2, 91]);

  const got = decodeSourceOrdinals(
    buffer,
    headers({
      "X-sourceOrdinalOffset": 12,
      "X-sourceOrdinalCount": 3,
      "X-sourceOrdinalEncoding": "uint32le",
    }),
    3,
  );

  assert.deepEqual([...got], [17, 2, 91]);
  new Uint32Array(buffer, 12, 1)[0] = 999;
  assert.deepEqual([...got], [17, 2, 91], "decoder must not retain the response buffer");
});

test("source ordinal decoder fails closed on a short or mismatched block", () => {
  const short = new ArrayBuffer(16);
  assert.equal(
    decodeSourceOrdinals(
      short,
      headers({
        "X-sourceOrdinalOffset": 8,
        "X-sourceOrdinalCount": 3,
        "X-sourceOrdinalEncoding": "uint32le",
      }),
      3,
    ),
    null,
  );
  assert.equal(
    decodeSourceOrdinals(
      new ArrayBuffer(32),
      headers({
        "X-sourceOrdinalOffset": 8,
        "X-sourceOrdinalCount": 2,
        "X-sourceOrdinalEncoding": "uint32le",
      }),
      3,
    ),
    null,
  );
});

test("line-segment raycast indices map back to the displayed streamline", () => {
  // k=4 means three segments per streamline and two index entries per segment.
  assert.equal(streamlineIndexFromSegmentEntry(0, 4, 3), 0);
  assert.equal(streamlineIndexFromSegmentEntry(4, 4, 3), 0);
  assert.equal(streamlineIndexFromSegmentEntry(6, 4, 3), 1);
  assert.equal(streamlineIndexFromSegmentEntry(16, 4, 3), 2);
  assert.equal(streamlineIndexFromSegmentEntry(18, 4, 3), null);
  assert.equal(streamlineIndexFromSegmentEntry(null, 4, 3), null);
});

test("picked-streamline summary preserves lineage and floors sub-resolution distance", () => {
  const summary = summarizeStreamline({
    displayIndex: 1,
    sourceOrdinals: new Uint32Array([17, 2]),
    sourcePopulation: "bank:bank_cst_r",
    lineCount: 2,
    pointsPerLine: 3,
    distances: new Float32Array([8, 7, 6, 2.4, 2.1, 2.8]),
    fidelity: {
      status: "ok",
      fracGeR: new Float32Array([0.9, 0.5]),
      p5: new Float32Array([0.7, 0.21]),
      flags: new Uint8Array([0, 2]),
      R: 0.3,
      minFrac: 0.7,
    },
    tract: { short: "CST-R" },
    geomFloorMm: 3,
  });

  assert.equal(summary.displayIndex, 1);
  assert.equal(summary.displayOrdinal, 2);
  assert.equal(summary.sourceIndex, 2);
  assert.equal(summary.sourceOrdinal, 3);
  assert.equal(summary.sourcePopulation, "bank:bank_cst_r");
  assert.equal(summary.tractLabel, "CST-R");
  assert.equal(summary.clearanceText, "below 3 mm geometric floor");
  assert.equal(summary.lowSupport, true);
  assert.equal(summary.crossesCavity, true);
  assert.equal(summary.crossesLesion, false);
  assert.match(summary.fidelityText, /support.*50%/i);
  assert.doesNotMatch(summary.fidelityText, /confidence/i);
});

test("missing fidelity remains explicitly untested", () => {
  const summary = summarizeStreamline({
    displayIndex: 0,
    sourceOrdinals: null,
    sourcePopulation: "live-track",
    lineCount: 1,
    pointsPerLine: 2,
    distances: null,
    fidelity: { status: "absent" },
    tract: { short: "live" },
    geomFloorMm: 3,
  });

  assert.equal(summary.sourceIndex, null);
  assert.equal(summary.clearanceText, "distance unavailable");
  assert.equal(summary.fidelityText, "fidelity: untested");
  assert.equal(summary.lowSupport, null);
});

test("distance remains unreportable when the geometric floor is unrecorded", () => {
  const summary = summarizeStreamline({
    displayIndex: 0,
    sourceOrdinals: new Uint32Array([4]),
    sourcePopulation: "live-track",
    lineCount: 1,
    pointsPerLine: 3,
    distances: new Float32Array([2.4, 2.1, 2.8]),
    fidelity: { status: "absent" },
    tract: { short: "live" },
    geomFloorMm: null,
  });

  assert.ok(Math.abs(summary.minDistanceMm - 2.1) < 1e-5);
  assert.equal(
    summary.clearanceText,
    "geometric floor unrecorded — distance not reportable",
  );
  assert.doesNotMatch(summary.clearanceText, /2\.1 mm/);
});
