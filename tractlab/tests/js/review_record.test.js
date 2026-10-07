import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  REVIEW_LIMITS,
  REVIEW_SCHEMA_VERSION,
  assertReviewContext,
  assertReviewSources,
  createReviewRecord,
  decodeVoxelRuns,
  encodeVoxelRuns,
  generatedRoute,
  gridAffineDigest,
  matchGeneratedIdentity,
  parseReviewRecord,
  requestDigest,
  restoreStatusLines,
  reviewSummary,
  textDigest,
  upgradeReviewRecord,
  EDGE_TUBES_ROUTE,
  OUTCOME_EFFECTS,
  OUTCOME_LABELS,
  OUTCOME_QUESTION,
} from "../../viewer/review_record.js";

const SOURCE_HASH = "a".repeat(64);
const CASE_SOURCE_HASH = "c".repeat(64);
const EDGE_HASH = "e".repeat(64);
const CORPUS_HASH2 = "9".repeat(64);
const PARC_HASH2 = "8".repeat(64);
const GEN_ID = "20260914T101709Z-6daf6e78";

/** A fully-valid edge-backed generated entry input for layer 10-13. */
function edgeGeneratedInput(overrides = {}) {
  return generatedInput({
    layerKey: "edge:10-13",
    request: {
      route: EDGE_TUBES_ROUTE, bankId: null, roiRef: null, params: { a: 10, b: 13, radiusMm: 4 },
      digest: requestDigest({
        route: EDGE_TUBES_ROUTE, bankId: null, seedPresetId: null, body: { a: 10, b: 13, radiusMm: 4 },
      }),
      ...(overrides.request || {}),
    },
    identity: {
      sourcePopulation: "edge:10-13", engineDigest: textDigest("CONNECTOME | edge 10-13"),
      nReturned: 253, lineCount: 253, bankSourceHash: null,
      edgeSourceHash: EDGE_HASH, corpusSourceHash: CORPUS_HASH2,
      parcellationSourceHash: PARC_HASH2, generationId: GEN_ID,
      ...(overrides.identity || {}),
    },
    ...Object.fromEntries(Object.entries(overrides).filter(([key]) => !["request", "identity"].includes(key))),
  });
}

/** v2Input wired so an edge:10-13 layer is a valid focus (no bank/envelope/pin to conflict with). */
function edgeV2Input(overrides = {}) {
  return v2Input({
    banks: [],
    envelope: null,
    view: { ...reviewInput().view, focusLayerKey: "edge:10-13", pin: null },
    ...overrides,
  });
}

function reviewInput(overrides = {}) {
  const bank = {
    bankId: "cst-r",
    sourcePopulation: "bank:cst-r",
    sourceHash: SOURCE_HASH,
    tractLabel: "CST-R",
    displayedCount: 822,
    analyticCount: 1500,
    fullCount: 2204,
    p5: 7.25,
    floorMm: 3,
    clearanceMethod: "surface-distance",
    clearancePopulation: "analytic bank sample",
    weighting: "SIFT2",
    fidelityStatus: "ok",
    operatingPoint: "R=0.30 · min 0.70 · signed",
  };
  return {
    context: {
      caseId: "demo-leipzig-sub-010005",
      gridId: "grid-983ab",
      volumeId: "volume-b0-9c22",
      recipeHash: "8fca7793abc12345",
      caseSourceHash: CASE_SOURCE_HASH,
      buildId: "build-20260910-a1",
    },
    banks: [bank],
    view: {
      focusLayerKey: "bank:cst-r",
      colour: "distance",
      camera: {
        position: [120, -60, 24],
        target: [0, 0, 0],
        up: [0, 0, 1],
      },
      slices: { ax: 54, cor: 48, sag: 52 },
      underlay: "b0",
      radius: 0.24,
      displayNearLesion: true,
      nearLesionFrac: 0.55,
      nearRadiusMm: 12,
      hideLowSupport: false,
      onlyLowSupport: false,
      hullVeil: "ghost",
      hullVisible: true,
      lesionVisible: true,
      slicePlane: "off",
      layout: { slicesOpen: true, dockFraction: 0.35, focusedSlice: null },
      pin: { layerKey: "bank:cst-r", sourceIndex: 12 },
    },
    evidence: {
      preflightState: "signed",
      preflightText: "gradients · pass",
      spaceLabel: "DWI space · corrected, unsigned",
    },
    unsupported: [],
    createdAt: "2026-09-10T12:00:00.000Z",
    ...overrides,
  };
}

function asJson(value) {
  return JSON.parse(JSON.stringify(value));
}

test("a named-bank record round-trips its bounded state with a readable historical summary", () => {
  const record = createReviewRecord(reviewInput());

  assert.equal(record.schemaVersion, 3);
  assert.equal(record.schemaVersion, REVIEW_SCHEMA_VERSION);
  assert.deepEqual(record.generated, []);
  assert.equal(record.paint, null);
  assert.deepEqual(record.overlays, { priors: [], parcelNetwork: null });
  assert.equal(record.envelope, null);
  assert.deepEqual(record.view.camera.offset, [0, 0]);
  assert.equal(record.banks[0].sourcePopulation, "bank:cst-r");
  assert.equal(record.banks[0].sourceHash, SOURCE_HASH);
  assert.equal(record.context.caseSourceHash, CASE_SOURCE_HASH);
  assert.match(record.summary, /Historical annotations/i);
  assert.match(record.summary, /source hash: a{64}/i);
  assert.match(record.summary, /case source hash: c{64}/i);
  assert.match(record.summary, /recorded floor.*display limit.*not measured spatial accuracy/i);
  assert.match(record.summary, /p5 describes reconstructed streamlines/i);
  assert.equal(reviewSummary(record), record.summary);
  assert.deepEqual(parseReviewRecord(JSON.stringify(record)), record);
});

test("case source fingerprint is required and must match before restore", () => {
  const missing = reviewInput();
  delete missing.context.caseSourceHash;
  assert.throws(() => createReviewRecord(missing), /case source hash.*required/i);

  const record = createReviewRecord(reviewInput());
  assert.throws(
    () => assertReviewContext(record, { ...record.context, caseSourceHash: "d".repeat(64) }),
    /case source/i,
  );
});

test("overview fraction defaults only at creation and is explicit in a saved record", () => {
  const record = createReviewRecord(reviewInput());
  assert.equal(record.view.layout.overviewFraction, record.view.layout.dockFraction);

  const missing = asJson(record);
  delete missing.view.layout.overviewFraction;
  assert.throws(() => parseReviewRecord(JSON.stringify(missing)), /overview fraction.*required/i);
});

test("named-bank reviews accept the 32-bank catalogue and preserve plain-text comparison methods", () => {
  const input = reviewInput();
  input.banks = Array.from({ length: 32 }, (_, index) => ({
    ...input.banks[0],
    bankId: `bank-${index}`,
    sourcePopulation: `bank:bank-${index}`,
    clearanceMethod: `p5 >= recorded floor ${"x".repeat(300)}`,
  }));
  input.view.focusLayerKey = "bank:bank-0";
  input.view.pin = { layerKey: "bank:bank-0", sourceIndex: 12 };

  const record = createReviewRecord(input);
  assert.equal(record.banks.length, 32);
  assert.match(record.banks[0].clearanceMethod, />=/);
});

test("parsing requires an explicit camera offset even when creation defaulted it", () => {
  const record = asJson(createReviewRecord(reviewInput()));
  delete record.view.camera.offset;

  assert.throws(() => parseReviewRecord(JSON.stringify(record)), /offset.*required/i);
});

test("unknown historical measurements remain explicit nulls instead of invented values", () => {
  const input = reviewInput();
  input.banks[0] = {
    ...input.banks[0],
    displayedCount: null,
    analyticCount: null,
    fullCount: null,
    p5: null,
    floorMm: null,
    clearanceMethod: null,
    clearancePopulation: null,
    weighting: null,
    fidelityStatus: null,
  };
  delete input.banks[0].operatingPoint;

  const record = createReviewRecord(input);
  assert.equal(record.banks[0].p5, null);
  assert.equal(record.banks[0].operatingPoint, null);
  assert.match(record.summary, /p5: unknown/i);
  assert.match(record.summary, /floor: unknown/i);
});

test("context and named source checks fail closed on a different case, recipe, or source hash", () => {
  const record = createReviewRecord(reviewInput());

  assert.equal(assertReviewContext(record, record.context), record);
  assert.equal(assertReviewSources(record, record.banks), record);
  assert.throws(
    () => assertReviewContext(record, { ...record.context, caseId: "other-case" }),
    /case/i,
  );
  assert.throws(
    () => assertReviewContext(record, { ...record.context, recipeHash: "1".repeat(16) }),
    /recipe/i,
  );
  assert.throws(
    () => assertReviewSources(record, [{ ...record.banks[0], sourceHash: "b".repeat(64) }]),
    /source hash/i,
  );
});

test("the banks array still holds named banks only, and unsupported state still refuses", () => {
  assert.throws(
    () => createReviewRecord(reviewInput({ unsupported: ["painted ROI workflow is active"] })),
    /unsupported/i,
  );
  assert.throws(
    () => createReviewRecord({
      ...reviewInput(),
      banks: [{ ...reviewInput().banks[0], sourcePopulation: "cut-subset:cst-r" }],
    }),
    /named bank/i,
  );
});

test("unsafe file content cannot add paths, markup, URLs, or unrecognised data", () => {
  const record = createReviewRecord(reviewInput());

  const withPath = asJson(record);
  withPath.path = "/private/case/geometry.tck";
  assert.throws(() => parseReviewRecord(JSON.stringify(withPath)), /unexpected field/i);

  const withMarkup = asJson(record);
  withMarkup.evidence.preflightText = "<img src=x onerror=alert(1)>";
  assert.throws(() => parseReviewRecord(JSON.stringify(withMarkup)), /plain text/i);

  const withUrlHash = asJson(record);
  withUrlHash.banks[0].sourceHash = "file:///private/case/bank.tck";
  assert.throws(() => parseReviewRecord(JSON.stringify(withUrlHash)), /source hash/i);

  const withGeometry = asJson(record);
  withGeometry.geometry = [1, 2, 3];
  assert.throws(() => parseReviewRecord(JSON.stringify(withGeometry)), /unexpected field/i);

  const withPunctuatedUrl = reviewInput();
  withPunctuatedUrl.evidence.preflightText = "See (https://evil.example)";
  assert.throws(() => createReviewRecord(withPunctuatedUrl), /URL/i);

  const withPunctuatedPath = reviewInput();
  withPunctuatedPath.evidence.preflightText = "See (/private/case.tck)";
  assert.throws(() => createReviewRecord(withPunctuatedPath), /filesystem path/i);

  for (const text of [
    "/tmp",
    "\\\\server\\share\\case.tck",
    "ftp://example.test/case",
    "www.example.test/case",
  ]) {
    const escaped = reviewInput();
    escaped.evidence.preflightText = text;
    assert.throws(() => createReviewRecord(escaped), /URL or filesystem path/i, text);
  }
});

test("view state rejects bad camera geometry, unsafe offsets, duplicate banks, and invalid pins", () => {
  const degenerate = reviewInput();
  degenerate.view.camera.position = [0, 0, 0];
  assert.throws(() => createReviewRecord(degenerate), /camera/i);

  const offset = reviewInput();
  offset.view.camera.offset = [2.01, 0];
  assert.throws(() => createReviewRecord(offset), /offset/i);

  const duplicate = reviewInput();
  duplicate.banks.push({ ...duplicate.banks[0] });
  assert.throws(() => createReviewRecord(duplicate), /duplicate/i);

  const badPin = reviewInput();
  badPin.view.pin = { layerKey: "bank:cst-r", sourceIndex: -1 };
  assert.throws(() => createReviewRecord(badPin), /source index/i);
});

// ---------------------------------------------------------------------------
// Schema 2: generated layers, painted ROIs, overlays, envelope.
// ---------------------------------------------------------------------------

const V1_FIXTURE = new URL("./fixtures/review-record-v1.json", import.meta.url);

function paintInput(overrides = {}) {
  return {
    grid: { gridId: "grid-983ab", dims: [8, 8, 8], affineDigest: gridAffineDigest(AFFINE) },
    paintMode: true,
    role: "seed",
    andIndex: 0,
    brushRadiusMm: 4,
    seedPresetId: "seed-handknob-r",
    rois: {
      seed: encodeVoxelRuns([10, 11, 12, 40]),
      and: [encodeVoxelRuns([100, 101])],
      or: [],
      not: encodeVoxelRuns([200]),
    },
    ...overrides,
  };
}

const AFFINE = [2, 0, 0, -90, 0, 2, 0, -126, 0, 0, 2, -72, 0, 0, 0, 1];

function generatedInput(overrides = {}) {
  const request = {
    route: "/api/filter",
    bankId: null,
    seedPresetId: null,
    roiRef: "paint",
    params: { params: { minlength: 20 } },
    digest: "0".repeat(16),
    ...(overrides.request || {}),
  };
  return {
    layerKey: "__live__",
    tractLabel: "Filtered corpus",
    request,
    identity: {
      sourcePopulation: "filtered-corpus",
      engineDigest: textDigest("FILTER | corpus"),
      nReturned: 1204,
      lineCount: 900,
      bankSourceHash: null,
      edgeSourceHash: null,
      corpusSourceHash: null,
      parcellationSourceHash: null,
      generationId: null,
      ...(overrides.identity || {}),
    },
    ...Object.fromEntries(Object.entries(overrides).filter(([key]) => !["request", "identity"].includes(key))),
  };
}

function v2Input(overrides = {}) {
  return {
    ...reviewInput(),
    paint: paintInput(),
    generated: [generatedInput()],
    overlays: { priors: ["norm_cst_r"], parcelNetwork: 3 },
    envelope: { layerKey: "bank:cst-r", marginMm: 5 },
    ...overrides,
  };
}

test("a schema 1 record is accepted unchanged and upgraded purely to the current schema", () => {
  const text = readFileSync(V1_FIXTURE, "utf8");
  const saved = JSON.parse(text);
  assert.equal(saved.schemaVersion, 1);
  assert.equal(saved.view.focusBankId, "cst-r");

  const upgraded = parseReviewRecord(text);
  assert.equal(upgraded.schemaVersion, 3);
  // The v1 objects survive byte for byte.
  assert.deepEqual(upgraded.banks, saved.banks);
  assert.deepEqual(upgraded.context, saved.context);
  assert.deepEqual(upgraded.evidence, saved.evidence);
  assert.equal(upgraded.createdAt, saved.createdAt);
  // The focus and pin move to layer keys; nothing is invented for the rest.
  assert.equal(upgraded.view.focusLayerKey, "bank:cst-r");
  assert.deepEqual(upgraded.view.pin, { layerKey: "bank:cst-r", sourceIndex: 12 });
  assert.equal(upgraded.view.focusBankId, undefined);
  assert.deepEqual(upgraded.generated, []);
  assert.equal(upgraded.paint, null);
  assert.deepEqual(upgraded.overlays, { priors: [], parcelNetwork: null });
  assert.equal(upgraded.envelope, null);
  assert.deepEqual(upgraded.outcome, { effect: "not-recorded", note: "" });

  // Pure: same input, same output, and the result re-reads as the current schema.
  assert.deepEqual(upgradeReviewRecord(saved), upgraded);
  assert.deepEqual(parseReviewRecord(JSON.stringify(upgraded)), upgraded);
  assert.match(upgraded.summary, /Schema: 3/);
});

test("a schema 1 document may not smuggle schema 2 objects past the v1 reader", () => {
  const saved = JSON.parse(readFileSync(V1_FIXTURE, "utf8"));
  saved.paint = paintInput();
  assert.throws(() => upgradeReviewRecord(saved), /schema 1 record cannot carry "paint"/i);

  const wrongVersion = JSON.parse(readFileSync(V1_FIXTURE, "utf8"));
  wrongVersion.schemaVersion = 4;
  assert.throws(() => upgradeReviewRecord(wrongVersion), /schemaVersion must be 1, 2 or 3/i);

  const smuggled = JSON.parse(readFileSync(V1_FIXTURE, "utf8"));
  smuggled.outcome = { effect: "approach", note: "" };
  assert.throws(() => upgradeReviewRecord(smuggled), /schema 1 record cannot carry "outcome"/i);
});

test("voxel runs round-trip as a set, dropping stroke order and duplicates", () => {
  const runs = encodeVoxelRuns([12, 10, 11, 10, 40]);
  assert.deepEqual(runs, [10, 3, 40, 1]);
  assert.deepEqual(decodeVoxelRuns(runs), [10, 11, 12, 40]);
  assert.deepEqual(encodeVoxelRuns([]), []);
});

test("a schema 2 record round-trips every new object", () => {
  const record = createReviewRecord(v2Input());

  assert.equal(record.schemaVersion, 3);
  assert.equal(record.generated.length, 1);
  assert.equal(record.generated[0].layerKey, "__live__");
  assert.equal(record.generated[0].request.route, "/api/filter");
  assert.equal(record.generated[0].request.roiRef, "paint");
  assert.deepEqual(record.generated[0].request.params, { params: { minlength: 20 } });
  assert.equal(record.generated[0].identity.sourcePopulation, "filtered-corpus");
  assert.equal(record.generated[0].identity.engineDigest, textDigest("FILTER | corpus"));
  assert.deepEqual(record.paint.rois.seed, [10, 3, 40, 1]);
  assert.equal(record.paint.grid.affineDigest, gridAffineDigest(AFFINE));
  assert.deepEqual(record.overlays, { priors: ["norm_cst_r"], parcelNetwork: 3 });
  assert.deepEqual(record.envelope, { layerKey: "bank:cst-r", marginMm: 5 });

  assert.match(record.summary, /Generated and subset layers/);
  assert.match(record.summary, /Painted and preset ROI state/);
  assert.match(record.summary, /Atlas priors: norm_cst_r/);
  assert.match(record.summary, /display only, not clearance and not a resection margin/);
  // No streamline geometry is ever written into a record.
  assert.equal(/points_mm|Float32|streamline geometry saved/i.test(JSON.stringify(record)), false);

  assert.deepEqual(parseReviewRecord(JSON.stringify(record)), record);
  assert.equal(reviewSummary(record), record.summary);
});

test("a saved schema 2 record must state every object explicitly", () => {
  const record = asJson(createReviewRecord(v2Input()));
  for (const field of ["generated", "paint", "overlays", "envelope"]) {
    const missing = asJson(record);
    delete missing[field];
    assert.throws(() => parseReviewRecord(JSON.stringify(missing)), new RegExp(`${field} is required`, "i"), field);
  }
});

test("painted ROI state is refused above the byte cap, with a reason to act on", () => {
  // 12000 non-adjacent voxels: every one its own run, so no run-length win.
  const wide = Array.from({ length: 12_000 }, (_, index) => index * 2);
  assert.throws(
    () => createReviewRecord(v2Input({
      paint: paintInput({
        grid: { gridId: "grid-983ab", dims: [64, 64, 64], affineDigest: gridAffineDigest(AFFINE) },
        rois: { seed: encodeVoxelRuns(wide), and: [], or: [], not: [] },
      }),
      generated: [],
    })),
    /at most 8192 \[start, length\] voxel runs/i,
  );

  const many = Array.from({ length: 6_000 }, (_, index) => index * 2);
  assert.throws(
    () => createReviewRecord(v2Input({
      paint: paintInput({
        grid: { gridId: "grid-983ab", dims: [64, 64, 64], affineDigest: gridAffineDigest(AFFINE) },
        rois: {
          seed: encodeVoxelRuns(many),
          and: [encodeVoxelRuns(many.map((value) => value + 30_000))],
          or: [],
          not: [],
        },
      }),
      generated: [],
    })),
    /over the 32768-character limit.*clear some paint/i,
  );
});

test("painted ROI state is refused on a different grid", () => {
  const otherGrid = v2Input({ paint: paintInput({ grid: { gridId: "grid-other", dims: [8, 8, 8], affineDigest: gridAffineDigest(AFFINE) } }) });
  assert.throws(() => createReviewRecord(otherGrid), /paint.grid.gridId must match/i);

  const outside = v2Input({
    paint: paintInput({ rois: { seed: encodeVoxelRuns([511, 512]), and: [], or: [], not: [] } }),
  });
  assert.throws(() => createReviewRecord(outside), /outside paint.grid.dims/i);

  // A changed affine on the same named grid is a different digest, so the
  // restore side refuses by comparison rather than by guessing.
  const moved = [...AFFINE];
  moved[3] += 1;
  assert.notEqual(gridAffineDigest(moved), gridAffineDigest(AFFINE));
  assert.equal(gridAffineDigest([...AFFINE]), gridAffineDigest(AFFINE));
});

/** The request a saved layer rebuilds to; its digest is what the record stores. */
function rebuiltFor(saved) {
  return { rebuilt: saved.request.digest };
}

test("a generated layer is marked not restored when the returned identity differs", () => {
  const record = createReviewRecord(v2Input({
    generated: [generatedInput({ request: { digest: requestDigest({ rebuilt: "x" }) } })],
  }));
  const saved = record.generated[0];
  const rebuiltRequest = { rebuilt: "x" };
  const served = {
    sourcePopulation: "filtered-corpus",
    engineDigest: textDigest("FILTER | corpus"),
    nReturned: 1204,
    lineCount: 900,
    bankSourceHash: null,
  };

  assert.deepEqual(
    matchGeneratedIdentity(saved, served, { rebuiltRequest }),
    { layerKey: "__live__", label: "Filtered corpus", restored: true, reason: null },
  );

  const fewer = matchGeneratedIdentity(saved, { ...served, nReturned: 1198 }, { rebuiltRequest });
  assert.equal(fewer.restored, false);
  assert.match(fewer.reason, /streamline count changed from 1204 to 1198/);

  const other = matchGeneratedIdentity(saved, { ...served, sourcePopulation: "live-track" }, { rebuiltRequest });
  assert.equal(other.restored, false);
  assert.match(other.reason, /different source population/);

  const rebuilt = matchGeneratedIdentity(saved, served, { rebuiltRequest: { rebuilt: "changed" } });
  assert.equal(rebuilt.restored, false);
  assert.match(rebuilt.reason, /rebuilt request differs/);
});

test("generated_identity_refuses_engine_mismatch", () => {
  const record = createReviewRecord(v2Input({
    generated: [generatedInput({ request: { digest: requestDigest({ rebuilt: "x" }) } })],
  }));
  const rebuiltRequest = { rebuilt: "x" };
  const served = {
    sourcePopulation: "filtered-corpus",
    engineDigest: textDigest("FILTER | corpus"),
    nReturned: 1204,
    lineCount: 900,
    bankSourceHash: null,
  };
  const swapped = matchGeneratedIdentity(
    record.generated[0],
    { ...served, engineDigest: textDigest("RECOVERY | peri-lesional") },
    { rebuiltRequest },
  );
  assert.equal(swapped.restored, false);
  assert.match(swapped.reason, /different engine/);
  // The record carries the digest, never the served free-text engine label.
  assert.equal(JSON.stringify(record).includes("FILTER | corpus"), false);
});

test("generated_identity_requires_complete_identity", () => {
  // Every identity field is mandatory in the record itself.
  for (const field of ["sourcePopulation", "engineDigest", "nReturned", "lineCount", "bankSourceHash"]) {
    const identity = { [field]: undefined };
    const input = v2Input({ generated: [generatedInput()] });
    delete input.generated[0].identity[field];
    void identity;
    assert.throws(() => createReviewRecord(input), new RegExp(`identity.${field} is required`, "i"), field);
  }
  // Nulls are refused where a value must exist.
  for (const field of ["engineDigest", "nReturned", "lineCount"]) {
    const input = v2Input({ generated: [generatedInput()] });
    input.generated[0].identity[field] = null;
    assert.throws(() => createReviewRecord(input), new RegExp(field, "i"), field);
  }
  // A missing served value is refused at comparison time, never treated as a match.
  const record = createReviewRecord(v2Input({
    generated: [generatedInput({ request: { digest: requestDigest({ rebuilt: "x" }) } })],
  }));
  const rebuiltRequest = { rebuilt: "x" };
  const served = {
    sourcePopulation: "filtered-corpus",
    engineDigest: textDigest("FILTER | corpus"),
    nReturned: 1204,
    lineCount: 900,
    bankSourceHash: null,
  };
  const noCount = matchGeneratedIdentity(record.generated[0], { ...served, lineCount: null }, { rebuiltRequest });
  assert.equal(noCount.restored, false);
  assert.match(noCount.reason, /did not report a displayed count/);
  const strayHash = matchGeneratedIdentity(
    record.generated[0],
    { ...served, bankSourceHash: SOURCE_HASH },
    { rebuiltRequest },
  );
  assert.equal(strayHash.restored, false);
  assert.match(strayHash.reason, /bank source for a layer that has none/);
});

test("identity_match_requires_rebuilt_request", () => {
  const record = createReviewRecord(v2Input());
  const served = {
    sourcePopulation: "filtered-corpus",
    engineDigest: textDigest("FILTER | corpus"),
    nReturned: 1204,
    lineCount: 900,
    bankSourceHash: null,
  };
  assert.throws(
    () => matchGeneratedIdentity(record.generated[0], served),
    /rebuilt request is required/i,
  );
  assert.throws(
    () => matchGeneratedIdentity(record.generated[0], served, {}),
    /rebuilt request is required/i,
  );
  // Even a route that is never re-issued must be handed the rebuilt request.
  const stochastic = createReviewRecord(v2Input({
    generated: [generatedInput({
      request: { route: "/api/track", roiRef: "paint", params: { params: { cutoff: 0.08 } } },
      identity: { sourcePopulation: "live-track" },
    })],
  }));
  assert.throws(
    () => matchGeneratedIdentity(stochastic.generated[0], null),
    /rebuilt request is required/i,
  );
});

test("review_connectotomy_restore_refuses_changed_bank_source", () => {
  // A bank-backed subset without a source hash cannot be saved at all.
  assert.throws(
    () => createReviewRecord(v2Input({
      banks: [],
      envelope: null,
      generated: [generatedInput({
        layerKey: "bank:cst-r",
        request: { route: "/api/connectotomy/cut", bankId: "cst-r", roiRef: null, params: {} },
        identity: { sourcePopulation: "cut-subset:cst-r", bankSourceHash: null },
      })],
    })),
    /bankSourceHash is required for a bank-backed layer/i,
  );

  const record = createReviewRecord(v2Input({
    banks: [],
    envelope: null,
    generated: [generatedInput({
      layerKey: "bank:cst-r",
      request: {
        route: "/api/connectotomy/cut", bankId: "cst-r", roiRef: null, params: {},
        digest: requestDigest({ rebuilt: "cut" }),
      },
      identity: { sourcePopulation: "cut-subset:cst-r", bankSourceHash: SOURCE_HASH },
    })],
  }));
  const rebuiltRequest = { rebuilt: "cut" };
  const served = {
    sourcePopulation: "cut-subset:cst-r",
    engineDigest: textDigest("FILTER | corpus"),
    nReturned: 1204,
    lineCount: 900,
  };
  assert.equal(
    matchGeneratedIdentity(record.generated[0], { ...served, bankSourceHash: SOURCE_HASH }, { rebuiltRequest }).restored,
    true,
  );
  const changed = matchGeneratedIdentity(
    record.generated[0], { ...served, bankSourceHash: "b".repeat(64) }, { rebuiltRequest },
  );
  assert.equal(changed.restored, false);
  assert.match(changed.reason, /bank source hash changed/);
  // A server that stops reporting the hash is refused, never quietly restored.
  const absent = matchGeneratedIdentity(
    record.generated[0], { ...served, bankSourceHash: null }, { rebuiltRequest },
  );
  assert.equal(absent.restored, false);
  assert.match(absent.reason, /bank source hash changed/);
});

test("connectome edge tubes route is registered as a reproducible GET with an {a,b,radiusMm} param schema", () => {
  const route = generatedRoute(EDGE_TUBES_ROUTE);
  assert.equal(EDGE_TUBES_ROUTE, "/api/connectome/edge/tubes");
  assert.equal(route.method, "GET");
  assert.equal(route.reproducible, true);

  const record = createReviewRecord(edgeV2Input({ generated: [edgeGeneratedInput()] }));
  assert.equal(record.generated[0].request.params.a, 10);
  assert.equal(record.generated[0].request.params.b, 13);
  assert.equal(record.generated[0].request.params.radiusMm, 4);
  assert.equal(record.generated[0].identity.edgeSourceHash, EDGE_HASH);
  assert.equal(record.generated[0].identity.corpusSourceHash, CORPUS_HASH2);
  assert.equal(record.generated[0].identity.parcellationSourceHash, PARC_HASH2);
  assert.equal(record.generated[0].identity.generationId, GEN_ID);
  assert.equal(record.generated[0].identity.bankSourceHash, null);
});

test("connectome edge tubes params refuse an undeclared field", () => {
  assert.throws(
    () => createReviewRecord(edgeV2Input({
      generated: [edgeGeneratedInput({ request: { params: { a: 10, b: 13, c: 1 } } })],
    })),
    /unexpected request field "c"/i,
  );
});

// edge_layer_requires_key_params_and_source_population_agreement — layerKey,
// request.params {a,b}, and identity.sourcePopulation are three
// independently-writable places the same edge could drift apart in a
// hand-edited or corrupted file; each disagreement must refuse the entry.
test("edge_layer_requires_key_params_and_source_population_agreement", () => {
  assert.throws(
    () => createReviewRecord(edgeV2Input({
      generated: [edgeGeneratedInput({ layerKey: "edge:99-100" })], // params still say {a:10,b:13}
    })),
    /layerKey \("edge:99-100"\) must match its request\.params/i,
  );
  assert.throws(
    () => createReviewRecord(edgeV2Input({
      generated: [edgeGeneratedInput({ identity: { sourcePopulation: "edge:99-100" } })],
    })),
    /identity\.sourcePopulation \("edge:99-100"\) must match .*layerKey \("edge:10-13"\)/i,
  );
  assert.throws(
    () => createReviewRecord(edgeV2Input({
      generated: [edgeGeneratedInput({ request: { params: { a: 10 } } })], // b missing
    })),
    /request\.params must carry \{a,b\}/i,
  );
  // The agreeing case (edgeGeneratedInput's own default) must NOT throw.
  assert.doesNotThrow(() => createReviewRecord(edgeV2Input({ generated: [edgeGeneratedInput()] })));
});

// Item 0 (root cause fix): edge identity is a MANDATORY joint tuple — edge
// content hash + corpus hash + parcellation hash + generation id — every
// field required for an edge-backed layer, every field compared on restore.
test("review_connectome_edge_restore_refuses_changed_edge_source", () => {
  for (const field of ["edgeSourceHash", "corpusSourceHash", "parcellationSourceHash", "generationId"]) {
    assert.throws(
      () => createReviewRecord(edgeV2Input({
        generated: [edgeGeneratedInput({ identity: { [field]: null } })],
      })),
      new RegExp(`${field} is required for an edge tube layer`, "i"),
      field,
    );
  }

  const record = createReviewRecord(edgeV2Input({
    generated: [edgeGeneratedInput({ request: { digest: requestDigest({ rebuilt: "edge" }) } })],
  }));
  const rebuiltRequest = { rebuilt: "edge" };
  const served = {
    sourcePopulation: "edge:10-13",
    engineDigest: textDigest("CONNECTOME | edge 10-13"),
    nReturned: 253,
    lineCount: 253,
    bankSourceHash: null,
    edgeSourceHash: EDGE_HASH,
    corpusSourceHash: CORPUS_HASH2,
    parcellationSourceHash: PARC_HASH2,
    generationId: GEN_ID,
  };
  assert.equal(matchGeneratedIdentity(record.generated[0], served, { rebuiltRequest }).restored, true);

  const cases = [
    ["edgeSourceHash", "f".repeat(64), /edge source hash changed/],
    ["edgeSourceHash", null, /edge source hash changed/],
    ["corpusSourceHash", "7".repeat(64), /corpus source hash changed/],
    ["corpusSourceHash", null, /corpus source hash changed/],
    ["parcellationSourceHash", "6".repeat(64), /parcellation source hash changed/],
    ["parcellationSourceHash", null, /parcellation source hash changed/],
  ];
  for (const [field, value, pattern] of cases) {
    const result = matchGeneratedIdentity(record.generated[0], { ...served, [field]: value }, { rebuiltRequest });
    assert.equal(result.restored, false, `${field}=${value}`);
    assert.match(result.reason, pattern, `${field}=${value}`);
  }
});

// restore_refuses_same_corpus_changed_assignment_lineage — a rebuild that
// republishes a new connectome generation from the IDENTICAL corpus bytes
// (e.g. a different radius, or simply a re-run) must still be caught: the
// corpus hash alone cannot see it, but generationId does.
test("restore_refuses_same_corpus_changed_assignment_lineage", () => {
  const record = createReviewRecord(edgeV2Input({
    generated: [edgeGeneratedInput({ request: { digest: requestDigest({ rebuilt: "edge" }) } })],
  }));
  const rebuiltRequest = { rebuilt: "edge" };
  const served = {
    sourcePopulation: "edge:10-13",
    engineDigest: textDigest("CONNECTOME | edge 10-13"),
    nReturned: 253,
    lineCount: 253,
    bankSourceHash: null,
    edgeSourceHash: EDGE_HASH,
    corpusSourceHash: CORPUS_HASH2, // same corpus bytes as the saved layer
    parcellationSourceHash: PARC_HASH2,
    generationId: "20260915T000000Z-newgen1", // a different published generation
  };
  const result = matchGeneratedIdentity(record.generated[0], served, { rebuiltRequest });
  assert.equal(result.restored, false, "same corpus must not be enough to restore across a generation swap");
  assert.match(result.reason, /connectome generation changed/);
});

test("a generated identity's new source fields are mandatory KEYS on every layer, not just edge ones", () => {
  // Removing the identity.generationId key from a non-edge (filter) layer
  // must still fail: every identity field is a required key on every entry,
  // its VALUE null unless the route needs it (same pattern as bankSourceHash
  // from schema 2's original release).
  const input = v2Input({ generated: [generatedInput()] });
  delete input.generated[0].identity.generationId;
  assert.throws(() => createReviewRecord(input), /identity\.generationId is required/i);
});

test("v1_upgrade_preserves_64k_input_cap", () => {
  const saved = JSON.parse(readFileSync(V1_FIXTURE, "utf8"));
  // Pad a schema 1 document past the frozen v1 cap without changing its record.
  const padded = `${JSON.stringify(saved)}${" ".repeat(64 * 1024)}`;
  assert.ok(padded.length > 64 * 1024 && padded.length < 256 * 1024);
  assert.throws(
    () => parseReviewRecord(padded),
    /schema 1 document must be no larger than 65536 bytes/i,
  );
  // A schema 2 document of the same size is fine: the wider cap is v2's own.
  const v2 = createReviewRecord(v2Input());
  const paddedV2 = `${JSON.stringify(v2)}${" ".repeat(64 * 1024)}`;
  assert.equal(parseReviewRecord(paddedV2).schemaVersion, 3);
});

test("a stochastic route is never redrawn from its saved request", () => {
  assert.equal(generatedRoute("/api/track").reproducible, false);
  assert.equal(generatedRoute("/api/filter").reproducible, true);
  assert.equal(generatedRoute("/api/connectotomy/cut").method, "GET");
  assert.throws(() => generatedRoute("/api/evil"), /not a route/i);

  const record = createReviewRecord(v2Input({
    generated: [generatedInput({
      request: { route: "/api/track", roiRef: "paint", params: { params: { cutoff: 0.08 } } },
      identity: { sourcePopulation: "live-track" },
    })],
  }));
  const status = matchGeneratedIdentity(record.generated[0], null, {
    rebuiltRequest: { rebuilt: "anything" },
  });
  assert.equal(status.restored, false);
  assert.match(status.reason, /live tracking is stochastic/);
  assert.match(status.reason, /kept as provenance/);
});

test("a generated layer's bank id must match its layer key, and a paint reference must exist", () => {
  assert.throws(
    () => createReviewRecord(v2Input({
      generated: [generatedInput({ layerKey: "bank:cst-r", request: { bankId: "other", route: "/api/connectotomy/cut", roiRef: null, params: {} } })],
    })),
    /bankId must match/i,
  );
  assert.throws(
    () => createReviewRecord(v2Input({ paint: null })),
    /references painted ROIs that are not in the record/i,
  );
  assert.throws(
    () => createReviewRecord(v2Input({ generated: [generatedInput({ request: { route: "/api/evil" } })] })),
    /not a route that produces a restorable layer/i,
  );
  assert.throws(
    () => createReviewRecord(v2Input({ envelope: { layerKey: "__recovery__", marginMm: 5 } })),
    /envelope.layerKey must name a saved layer/i,
  );
});

test("request_params_reject_unknown_keys_and_phi_text", () => {
  const withParams = (params, route = "/api/filter") => v2Input({
    generated: [generatedInput({ request: { route, params } })],
  });

  // Only fields a route declares can reach a record, and the refusal says which.
  assert.throws(
    () => createReviewRecord(withParams({ params: { minlength: 20 }, note: "pt c/ glioma frontal D" })),
    /unexpected request field "note"/i,
  );
  assert.throws(
    () => createReviewRecord(withParams({ params: { minlength: 20, comment: "Maria, 54a" } })),
    /unexpected request field "comment"/i,
  );
  assert.throws(
    () => createReviewRecord(withParams({ params: { minlength: 20 }, patient: { name: "x" } })),
    /unexpected request field "patient"/i,
  );

  // A declared field still refuses free text: only short machine tokens pass.
  assert.throws(
    () => createReviewRecord(withParams(
      { radiusMm: 8, source: "filter bank for Maria's left frontal case" },
      "/api/recovery/perilesional",
    )),
    /request.params.source must be a short token, not free text/i,
  );
  assert.throws(
    () => createReviewRecord(withParams({ params: { minlength: "twenty" } })),
    /request.params.params.minlength must be a finite number/i,
  );
  assert.throws(
    () => createReviewRecord(withParams({ params: { minlength: 20, density: "NORMAL, per Dr. A" } }, "/api/track")),
    /density must be a short token/i,
  );

  // Declared, well-typed values are kept as sent.
  const ok = createReviewRecord(withParams({ radiusMm: 8, source: "filter_bank" }, "/api/recovery/perilesional"));
  assert.deepEqual(ok.generated[0].request.params, { radiusMm: 8, source: "filter_bank" });

  // The digest is order-insensitive so a rebuilt body cannot fail on key order.
  assert.equal(requestDigest({ a: 1, b: 2 }), requestDigest({ b: 2, a: 1 }));
  assert.notEqual(requestDigest({ a: 1 }), requestDigest({ a: 2 }));
});

test("schema_rejects_bank_generated_layer_key_collision", () => {
  // The saved bank is "cst-r", so a generated layer may not claim bank:cst-r.
  assert.throws(
    () => createReviewRecord(v2Input({
      generated: [generatedInput({
        layerKey: "bank:cst-r",
        request: { route: "/api/connectotomy/cut", bankId: "cst-r", roiRef: null, params: {} },
        identity: { sourcePopulation: "cut-subset:cst-r", bankSourceHash: SOURCE_HASH },
      })],
    })),
    /names both a saved bank and a generated layer/i,
  );

  // Twice in one list is refused on either side.
  assert.throws(
    () => createReviewRecord(v2Input({
      generated: [generatedInput(), generatedInput()],
    })),
    /duplicate layerKey/i,
  );
  const twoBanks = v2Input();
  twoBanks.banks = [twoBanks.banks[0], { ...twoBanks.banks[0] }];
  assert.throws(() => createReviewRecord(twoBanks), /duplicate bankId/i);

  // A parsed document is held to the same rule, not just a created one.
  const record = asJson(createReviewRecord(v2Input()));
  record.generated[0].layerKey = "bank:cst-r";
  record.generated[0].request.bankId = "cst-r";
  record.generated[0].identity.bankSourceHash = SOURCE_HASH;
  assert.throws(
    () => parseReviewRecord(JSON.stringify(record)),
    /names both a saved bank and a generated layer/i,
  );
});

test("created_record_always_fits_reopen_utf8_byte_cap", () => {
  const record = createReviewRecord(v2Input());
  const text = JSON.stringify(record);
  assert.ok(Buffer.byteLength(text, "utf8") <= REVIEW_LIMITS.maxRecordBytes);
  assert.equal(parseReviewRecord(text).schemaVersion, 3);

  // The cap is UTF-8 bytes, not UTF-16 code units: a document short enough by
  // .length but too long in bytes is refused, and says so in bytes.
  const padded = `${text}${"\u2265".repeat(REVIEW_LIMITS.maxRecordBytes / 2)}`;
  assert.ok(padded.length < REVIEW_LIMITS.maxRecordBytes, "under the cap by code units");
  assert.ok(Buffer.byteLength(padded, "utf8") > REVIEW_LIMITS.maxRecordBytes, "over the cap in bytes");
  assert.throws(() => parseReviewRecord(padded), /no larger than 262144 bytes/i);

  // Anything creation accepts, reopen must accept. Build the largest record the
  // per-object caps allow and hold both halves to the one limit.
  const wide = v2Input({ generated: [], envelope: null });
  wide.banks = Array.from({ length: 32 }, (_, index) => ({
    ...wide.banks[0],
    bankId: `bank-${index}`,
    sourcePopulation: `bank:bank-${index}`,
    clearanceMethod: "x".repeat(600),
    clearancePopulation: "y".repeat(240),
    operatingPoint: "z".repeat(240),
  }));
  wide.view.focusLayerKey = "bank:bank-0";
  wide.view.pin = { layerKey: "bank:bank-0", sourceIndex: 12 };
  wide.paint = paintInput({
    grid: { gridId: "grid-983ab", dims: [64, 64, 64], affineDigest: gridAffineDigest(AFFINE) },
    rois: { seed: encodeVoxelRuns(Array.from({ length: 3_000 }, (_, i) => i * 2)), and: [], or: [], not: [] },
  });
  wide.evidence = { ...wide.evidence, preflightText: "\u2265".repeat(1_100) };
  wide.outcome = { effect: "limit", note: "\u2265".repeat(500) };
  const created = createReviewRecord(wide);
  const size = Buffer.byteLength(JSON.stringify(created), "utf8");
  assert.ok(size <= REVIEW_LIMITS.maxRecordBytes, `the largest constructible record is ${size} bytes`);
  assert.equal(parseReviewRecord(JSON.stringify(created)).schemaVersion, 3,
    "a record this viewer saved always reopens in this viewer");
});

test("save_preserves_empty_active_and_region", () => {
  // "New AND region" opens an empty region and makes it active. The record
  // keeps it so andIndex still names the region the user is working in.
  const withEmpty = v2Input({
    generated: [],
    paint: paintInput({
      andIndex: 1,
      rois: { seed: encodeVoxelRuns([10, 11]), and: [encodeVoxelRuns([100, 101]), []], or: [], not: [] },
    }),
  });
  const record = createReviewRecord(withEmpty);
  assert.equal(record.paint.rois.and.length, 2, "the empty region survives the save");
  assert.deepEqual(record.paint.rois.and[1], []);
  assert.equal(record.paint.andIndex, 1, "the active index still names it");
  assert.deepEqual(parseReviewRecord(JSON.stringify(record)).paint, record.paint);
  assert.match(record.summary, /AND regions: 2, 0 voxels/);

  // An index past the regions is still refused.
  assert.throws(
    () => createReviewRecord(v2Input({
      generated: [],
      paint: paintInput({ andIndex: 2, rois: { seed: [], and: [[], []], or: [], not: [] } }),
    })),
    /andIndex is outside/i,
  );
  assert.throws(
    () => createReviewRecord(v2Input({
      generated: [],
      paint: paintInput({ andIndex: 1, rois: { seed: encodeVoxelRuns([1]), and: [], or: [], not: [] } }),
    })),
    /andIndex must be 0 when there are no AND regions/i,
  );
});

test("an unknown prior id is carried but the overlay list stays a bounded id list", () => {
  const record = createReviewRecord(v2Input({ overlays: { priors: ["norm_cst_r", "norm_af_l"], parcelNetwork: null } }));
  assert.deepEqual(record.overlays.priors, ["norm_cst_r", "norm_af_l"]);
  assert.equal(record.overlays.parcelNetwork, null);

  assert.throws(
    () => createReviewRecord(v2Input({ overlays: { priors: ["norm_cst_r", "norm_cst_r"], parcelNetwork: null } })),
    /duplicate id/i,
  );
  // Prior volumes are population data on the server; a record carries ids only.
  assert.throws(
    () => createReviewRecord(v2Input({ overlays: { priors: [{ id: "norm_cst_r", volume: [1, 2] }], parcelNetwork: null } })),
    /plain text/i,
  );

  // The restore side reports an absent id rather than drawing anything.
  assert.deepEqual(
    restoreStatusLines([
      { label: "CST-R", restored: true, reason: null },
      { label: "Atlas prior norm_af_l", restored: false, reason: "not restored, this prior is not served for this case" },
    ]),
    [
      "CST-R: restored",
      "Atlas prior norm_af_l: not restored, this prior is not served for this case",
    ],
  );
});

// ---- view.profileInterval (display state, schema 2) ----

test("a saved profile interval round-trips through the schema", () => {
  const record = createReviewRecord(reviewInput({
    view: {
      ...reviewInput().view,
      profileInterval: { layerKey: "bank:cst-r", scalar: "fa", from: 20, to: 44 },
    },
  }));
  const parsed = parseReviewRecord(JSON.stringify(record));
  assert.deepEqual(parsed.view.profileInterval, {
    layerKey: "bank:cst-r", scalar: "fa", from: 20, to: 44,
  });
});

test("a record with no profile interval is accepted and reads as null", () => {
  const record = createReviewRecord(reviewInput());
  assert.equal(record.view.profileInterval, null);
  const { profileInterval, ...view } = record.view;
  assert.equal(profileInterval, null);
  // A schema-2 file written before the profile panel existed omits the key.
  const older = parseReviewRecord(JSON.stringify({ ...record, view }));
  assert.equal(older.view.profileInterval, null);
});

test("an upgraded schema-1 record carries no profile interval and is not refused", () => {
  const upgraded = upgradeReviewRecord(JSON.parse(readFileSync(V1_FIXTURE, "utf8")));
  assert.equal(upgraded.schemaVersion, 3);
  assert.equal(upgraded.view.profileInterval, null);
});

test("restore_profile_interval_refuses_nonbank_or_out_of_bounds_state", () => {
  const base = reviewInput().view;
  const bad = (profileInterval) => () => createReviewRecord(reviewInput({
    view: { ...base, profileInterval },
  }));
  assert.throws(bad({ layerKey: "bank:fat-r", scalar: "fa", from: 0, to: 5 }), /must name a saved layer/);
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "adc", from: 0, to: 5 }), /scalar/);
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "fa", from: 9, to: 2 }), /must not precede/);
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "fa", from: 0, to: 5, extra: 1 }), /profileInterval/);
  // Node indices live in the served profile's 0..99 range; anything else was
  // not produced by this instrument and is refused at parse, not clamped.
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "fa", from: 0, to: 100 }), /view\.profileInterval\.to/);
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "fa", from: -1, to: 5 }), /view\.profileInterval\.from/);
  assert.throws(bad({ layerKey: "bank:cst-r", scalar: "fa", from: 0.5, to: 5 }), /view\.profileInterval\.from/);
  assert.doesNotThrow(bad({ layerKey: "bank:cst-r", scalar: "fa", from: 0, to: 99 }));
});

test("a profile interval may only name a named bank layer", () => {
  const withGenerated = v2Input();
  const generatedKey = withGenerated.generated[0].layerKey;
  assert.ok(!generatedKey.startsWith("bank:"), "the fixture's generated layer is not a bank");
  assert.throws(
    () => createReviewRecord({
      ...withGenerated,
      view: {
        ...withGenerated.view,
        profileInterval: { layerKey: generatedKey, scalar: "fa", from: 0, to: 9 },
      },
    }),
    /named bank layer/,
  );
});

// ---- outcome (schema 3, reviewer self-report) ----

// Written by the schema-2 writer at the commit before schema 3 (synthetic demo
// inputs only: v2Input() with a painted ROI, a generated layer, overlays and an
// envelope). It must keep restoring after the bump.
const V2_FIXTURE = new URL("./fixtures/review-record-v2.json", import.meta.url);

test("outcome effects are a closed set with the reviewer wording", () => {
  assert.deepEqual([...OUTCOME_EFFECTS], ["approach", "limit", "none", "not-recorded"]);
  assert.equal(OUTCOME_QUESTION, "Did the evidence wall change the plan?");
  assert.deepEqual({ ...OUTCOME_LABELS }, {
    approach: "Changed approach",
    limit: "Set a limit",
    none: "No change",
    "not-recorded": "Not recorded",
  });
});

test("a schema 2 record restores unchanged and reads its outcome as not recorded", () => {
  const text = readFileSync(V2_FIXTURE, "utf8");
  const saved = JSON.parse(text);
  assert.equal(saved.schemaVersion, 2);
  assert.equal(Object.hasOwn(saved, "outcome"), false);
  // The schema-2 summary is still the one this reader would build for it.
  assert.equal(reviewSummary(saved), saved.summary);

  const upgraded = parseReviewRecord(text);
  assert.equal(upgraded.schemaVersion, 3);
  // Every schema-2 field survives exactly; only the version and summary move.
  for (const key of Object.keys(saved)) {
    if (key === "schemaVersion" || key === "summary") continue;
    assert.deepEqual(upgraded[key], saved[key], key);
  }
  assert.deepEqual(
    Object.keys(upgraded).sort(),
    [...Object.keys(saved), "outcome"].sort(),
    "the upgrade adds outcome and nothing else",
  );
  assert.deepEqual(upgraded.outcome, { effect: "not-recorded", note: "" });
  assert.match(upgraded.summary, /Schema: 3/);
  assert.match(upgraded.summary, /Did the evidence wall change the plan\? Not recorded/);
  assert.match(upgraded.summary, /Outcome note: none/);

  // Pure and idempotent: re-serialising the upgraded record re-reads identically.
  assert.deepEqual(upgradeReviewRecord(saved), upgraded);
  assert.deepEqual(parseReviewRecord(JSON.stringify(upgraded)), upgraded);
  assert.deepEqual(parseReviewRecord(JSON.stringify(upgraded, null, 2)), upgraded);
  assert.equal(reviewSummary(upgraded), upgraded.summary);
  // It still passes the restore gates against the same served sources.
  assertReviewContext(upgraded, saved.context);
  assertReviewSources(upgraded, [{ bankId: "cst-r", sourcePopulation: "bank:cst-r", sourceHash: SOURCE_HASH }]);
});

test("a schema 2 restore fails closed on tampering, a smuggled outcome, or a missing object", () => {
  const fresh = () => JSON.parse(readFileSync(V2_FIXTURE, "utf8"));

  const tamperedSummary = fresh();
  tamperedSummary.summary = tamperedSummary.summary.replace("Schema: 2", "Schema: 3");
  assert.throws(() => parseReviewRecord(JSON.stringify(tamperedSummary)), /summary does not match/i);

  const tamperedField = fresh();
  tamperedField.banks[0].p5 = 9.5;
  assert.throws(() => parseReviewRecord(JSON.stringify(tamperedField)), /summary does not match/i);

  const smuggled = fresh();
  smuggled.outcome = { effect: "approach", note: "" };
  assert.throws(() => parseReviewRecord(JSON.stringify(smuggled)), /schema 2 record cannot carry "outcome"/i);

  const missing = fresh();
  delete missing.generated;
  assert.throws(() => parseReviewRecord(JSON.stringify(missing)), /generated is required in a saved schema 2 record/i);
});

test("each outcome effect round-trips with its note and shows in the summary", () => {
  for (const effect of OUTCOME_EFFECTS) {
    const record = createReviewRecord(v2Input({ outcome: { effect, note: "Reviewed with the fellow >= 2 views" } }));
    assert.equal(record.schemaVersion, 3);
    assert.deepEqual(record.outcome, { effect, note: "Reviewed with the fellow >= 2 views" });
    assert.match(record.summary, new RegExp(`change the plan\\? ${OUTCOME_LABELS[effect]}$`, "m"));
    assert.match(record.summary, /Outcome note: Reviewed with the fellow >= 2 views/);
    assert.match(record.summary, /reviewer self-report, not a measurement/);
    assert.deepEqual(parseReviewRecord(JSON.stringify(record)), record);
    assert.equal(reviewSummary(record), record.summary);
  }
});

test("an absent outcome reads as not recorded, at creation and in a saved schema 3 record", () => {
  const created = createReviewRecord(reviewInput());
  assert.deepEqual(created.outcome, { effect: "not-recorded", note: "" });
  assert.deepEqual(createReviewRecord(reviewInput({ outcome: null })).outcome, created.outcome);
  assert.deepEqual(createReviewRecord(reviewInput({ outcome: { effect: "none" } })).outcome, { effect: "none", note: "" });

  const withoutKey = asJson(created);
  delete withoutKey.outcome;
  const reread = parseReviewRecord(JSON.stringify(withoutKey));
  assert.deepEqual(reread.outcome, { effect: "not-recorded", note: "" });
  assert.deepEqual(reread, created);
});

test("outcome fails closed on an unknown effect, extra field, or unsafe note", () => {
  const create = (outcome) => () => createReviewRecord(reviewInput({ outcome }));
  assert.throws(create({ effect: "changed", note: "" }), /outcome\.effect has an unsupported value/);
  assert.throws(create({ effect: "Approach", note: "" }), /outcome\.effect has an unsupported value/);
  assert.throws(create({ effect: "", note: "" }), /outcome\.effect has an unsupported value/);
  assert.throws(create({ effect: 1, note: "" }), /outcome\.effect has an unsupported value/);
  assert.throws(create({ note: "no effect" }), /outcome\.effect is required/);
  assert.throws(create({ effect: "none", note: "", approvedBy: "someone" }), /outcome has unexpected field "approvedBy"/);
  assert.throws(create("approach"), /outcome must be an object/);
  assert.throws(create({ effect: "none", note: "x".repeat(501) }), /at most 500 characters/);
  assert.throws(create({ effect: "none", note: "see https://example.org" }), /must not contain a URL or filesystem path/);
  assert.throws(create({ effect: "none", note: "copied from ~/cases/x" }), /must not contain a URL or filesystem path/);
  assert.throws(create({ effect: "none", note: "<b>bold</b>" }), /not HTML markup/);
  assert.throws(create({ effect: "none", note: 5 }), /outcome\.note must be plain text/);
  assert.deepEqual(createReviewRecord(reviewInput({ outcome: { effect: "none", note: "x".repeat(500) } })).outcome.note.length, 500);

  // A saved schema 3 file edited by hand: an unknown effect is refused before
  // anything else is read, and a valid effect swapped after saving no longer
  // matches its own summary.
  const saved = asJson(createReviewRecord(reviewInput({ outcome: { effect: "approach", note: "" } })));
  assert.throws(
    () => parseReviewRecord(JSON.stringify({ ...saved, outcome: { effect: "maybe", note: "" } })),
    /outcome\.effect has an unsupported value/,
  );
  assert.throws(
    () => parseReviewRecord(JSON.stringify({ ...saved, outcome: { effect: "none", note: "" } })),
    /summary does not match/,
  );
});
