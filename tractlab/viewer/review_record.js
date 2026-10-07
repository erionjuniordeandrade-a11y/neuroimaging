/**
 * Versioned, local-only review records.
 *
 * This module deliberately has no DOM, fetch, file, or storage access. It
 * accepts only the bounded state needed to reopen a review against served,
 * matching sources. Historical measurements in a record are never treated as
 * current measurements.
 *
 * Schema 2 adds the objects schema 1 refused to save: generated/subset tract
 * layers (as the request that produced them plus the identity the server
 * returned, never their geometry), painted/preset ROI state (as voxel runs on
 * a named grid), population overlay toggles (ids only), and the display
 * envelope (parameters only). Schema 1 documents are still accepted and are
 * upgraded purely, with those objects empty.
 *
 * Schema 3 adds one optional reviewer-entered object, outcome: whether the
 * evidence wall changed the plan, as a closed effect value plus a short plain
 * note. It is a self-report about the review, never a measurement or a clinical
 * claim. Absent reads as "not-recorded"; schema 1 and 2 documents upgrade to it.
 */

const SCHEMA_VERSION = 3;
const SUPPORTED_SCHEMA_VERSIONS = new Set([1, 2, 3]);
const MAX_INPUT_CHARS = 256 * 1024;
const MAX_V1_INPUT_CHARS = 64 * 1024;
// The painted-ROI object is the only unbounded-by-nature part of a record.
// Its own cap is refused at save time with a reason the user can act on.
const MAX_PAINT_CHARS = 32 * 1024;
const MAX_BANKS = 32;
const MAX_GENERATED = 16;
const MAX_PRIORS = 64;
const MAX_AND_REGIONS = 16;
const MAX_RUN_PAIRS = 8192;
const MAX_UNSUPPORTED = 16;
const MAX_IDENTIFIER_LENGTH = 160;
const MAX_TEXT_LENGTH = 1_200;
const MAX_OPERATING_POINT_LENGTH = 240;
const MAX_COORDINATE = 1_000_000;
const MAX_COUNT = 100_000_000;
const MAX_SLICE = 1_000_000;
const MAX_SOURCE_INDEX = 0xffff_ffff;
const MAX_VOXEL_INDEX = 0x3fff_ffff;
const MAX_DIMENSION = 4_096;

const AXES = new Set(["ax", "cor", "sag"]);
const COLOURS = new Set(["distance", "direction", "solid"]);
const UNDERLAYS = new Set(["b0", "t1", "fa", "dec"]);
const HULL_VEILS = new Set(["ghost", "mid", "solid"]);
const PREFLIGHT_STATES = new Set([
  "signed", "unsigned", "untestable", "unavailable", "loading", "absent", "unknown",
]);
const FIDELITY_STATES = new Set(["ok", "absent", "error", "untestable", "unknown"]);
const HASH_RE = /^(?:sha256:)?[a-f0-9]{64}$/i;
const IDENTIFIER_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/;
const HTML_TAG_RE = /<\s*\/?\s*[A-Za-z][^>]*>/;
const FORBIDDEN_URL_RE = /(?:https?|ftp|file):\/+|(?:^|[^A-Za-z0-9_])(?:data|javascript|vbscript):|(?:^|[^A-Za-z0-9._-])www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:\/[^\s]*)?/i;
const FILESYSTEM_PATH_RE = /(?:^|[\s([{"'])(?:~\/|\/(?:[^\s/]+\/)+[^\s/)\]}>,;:]+|\.\.?(?:\/|\\)|[A-Za-z]:\\)/;
const ABSOLUTE_PATH_RE = /(?:^|[\s([{"'])\/[^\s/)\]}>,;:]+(?:\/[^\s/)\]}>,;:]*)*/;
const UNC_PATH_RE = /(?:^|[\s([{"'])\\\\[^\\/\s]+\\[^\\/\s]+(?:\\[^\\/\s]+)*/;

const V2_OBJECT_FIELDS = ["generated", "paint", "overlays", "envelope"];
const V3_OBJECT_FIELDS = ["outcome"];
const CREATE_FIELDS = new Set([
  "context", "banks", "view", "evidence", "unsupported", "createdAt", ...V2_OBJECT_FIELDS, ...V3_OBJECT_FIELDS,
]);
const RECORD_FIELDS_V1 = new Set([
  "schemaVersion", "createdAt", "context", "banks", "view", "evidence", "unsupported", "summary",
]);
const RECORD_FIELDS_V2 = new Set([...RECORD_FIELDS_V1, ...V2_OBJECT_FIELDS]);
const RECORD_FIELDS_V3 = new Set([...RECORD_FIELDS_V2, ...V3_OBJECT_FIELDS]);
const OUTCOME_FIELDS = new Set(["effect", "note"]);
/** Closed set, in display order. "not-recorded" is the default and the absent value. */
export const OUTCOME_EFFECTS = Object.freeze(["approach", "limit", "none", "not-recorded"]);
const OUTCOME_EFFECT_SET = new Set(OUTCOME_EFFECTS);
/** Reviewer-facing wording; the question and options make no clinical claim. */
export const OUTCOME_QUESTION = "Did the evidence wall change the plan?";
export const OUTCOME_LABELS = Object.freeze({
  approach: "Changed approach",
  limit: "Set a limit",
  none: "No change",
  "not-recorded": "Not recorded",
});
const MAX_OUTCOME_NOTE_LENGTH = 500;
const CONTEXT_FIELDS = new Set(["caseId", "gridId", "volumeId", "recipeHash", "caseSourceHash", "buildId"]);
const BANK_FIELDS = new Set([
  "bankId", "sourcePopulation", "sourceHash", "tractLabel", "displayedCount", "analyticCount",
  "fullCount", "p5", "floorMm", "clearanceMethod", "clearancePopulation", "weighting",
  "fidelityStatus", "operatingPoint",
]);
const VIEW_COMMON_FIELDS = [
  "colour", "camera", "slices", "underlay", "radius", "displayNearLesion",
  "nearLesionFrac", "nearRadiusMm", "hideLowSupport", "onlyLowSupport", "hullVeil",
  "hullVisible", "lesionVisible", "slicePlane", "layout", "pin",
];
// v1 names the focused layer by bank id; v2 names it by layer key so a
// generated or subset layer can be the focus without inventing a bank id.
const VIEW_FIELDS_V1 = new Set(["focusBankId", ...VIEW_COMMON_FIELDS]);
// `profileInterval` is optional on purpose: it is display state a v2 record
// written before the profile panel existed simply does not carry, and a
// record must never be refused for the absence (or the presence) of it.
const VIEW_FIELDS_V2 = new Set(["focusLayerKey", "profileInterval", ...VIEW_COMMON_FIELDS]);
const PROFILE_INTERVAL_FIELDS = new Set(["layerKey", "scalar", "from", "to"]);
const PROFILE_SCALARS = new Set(["fa", "md"]);
// The served profile is 100 nodes (profile.py N_POINTS_DEFAULT). A saved index
// outside 0..99 never came from this instrument.
const MAX_PROFILE_NODE = 99;
const CAMERA_FIELDS = new Set(["position", "target", "up", "offset"]);
const SLICES_FIELDS = new Set(["ax", "cor", "sag"]);
const LAYOUT_FIELDS = new Set(["slicesOpen", "dockFraction", "overviewFraction", "focusedSlice"]);
const PIN_FIELDS_V1 = new Set(["bankId", "sourceIndex"]);
const PIN_FIELDS_V2 = new Set(["layerKey", "sourceIndex"]);
const EVIDENCE_FIELDS = new Set(["preflightState", "preflightText", "spaceLabel"]);

const GENERATED_FIELDS = new Set(["layerKey", "tractLabel", "request", "identity"]);
const GENERATED_REQUEST_FIELDS = new Set([
  "route", "bankId", "seedPresetId", "roiRef", "params", "digest",
]);
const GENERATED_IDENTITY_FIELDS = new Set([
  "sourcePopulation", "engineDigest", "nReturned", "lineCount", "bankSourceHash",
  "edgeSourceHash", "corpusSourceHash", "parcellationSourceHash", "generationId",
]);
const PAINT_FIELDS = new Set([
  "grid", "paintMode", "role", "andIndex", "brushRadiusMm", "seedPresetId", "rois",
]);
const PAINT_GRID_FIELDS = new Set(["gridId", "dims", "affineDigest"]);
const ROI_FIELDS = new Set(["seed", "and", "or", "not"]);
const OVERLAYS_FIELDS = new Set(["priors", "parcelNetwork"]);
const ENVELOPE_FIELDS = new Set(["layerKey", "marginMm"]);

const ROI_ROLES = new Set(["seed", "and", "or", "not"]);
const PARAM_TOKEN_RE = /^[a-z0-9_:.-]{1,64}$/;
const LAYER_KEY_RE = /^(?:bank:[A-Za-z0-9][A-Za-z0-9._-]*|edge:\d{1,10}-\d{1,10}|__live__|__recovery__)$/;
const DIGEST_RE = /^[a-f0-9]{16}$/;

/**
 * What each route is allowed to carry in `request.params`, by key and by type.
 *
 * A record is a file a clinician may share. Nothing reaches it unless a route
 * declares it here, so an unrecognised key or a free-text value refuses the
 * save by name rather than copying whatever the viewer happened to send.
 * Types: "number" (finite, bounded), "boolean", "token" (PARAM_TOKEN_RE).
 */
const GENERATED_PARAM_SCHEMAS = new Map([
  ["/api/track", { params: { cutoff: "number", angle: "number", minlength: "number", density: "token" } }],
  ["/api/filter", { params: { minlength: "number" } }],
  ["/api/recovery/perilesional", { radiusMm: "number", source: "token" }],
  ["/api/connectotomy/cut", {}],
  ["/api/connectome/edge/tubes", { a: "number", b: "number", radiusMm: "number" }],
]);

/**
 * Routes that can produce a generated or subset layer. `reproducible` says
 * whether re-issuing the identical request can return the same streamlines:
 * live iFOD2 tracking is stochastic, so its request is kept for provenance and
 * the layer is never redrawn from it.
 */
const GENERATED_ROUTES = new Map([
  ["/api/track", { method: "POST", reproducible: false, why: "live tracking is stochastic", label: "live track" }],
  ["/api/filter", { method: "POST", reproducible: true, why: null, label: "bank filter" }],
  ["/api/recovery/perilesional", { method: "POST", reproducible: true, why: null, label: "peri-lesional recovery" }],
  ["/api/connectotomy/cut", { method: "GET", reproducible: true, why: null, label: "connectotomy cut subset" }],
  ["/api/connectome/edge/tubes", { method: "GET", reproducible: true, why: null, label: "connectome edge tubes" }],
]);

/** The one GET route whose identity is an edge source hash, not a bank source hash. */
export const EDGE_TUBES_ROUTE = "/api/connectome/edge/tubes";

function fail(message) {
  throw new Error(`Review record: ${message}`);
}

const TEXT_ENCODER = new TextEncoder();

/** Size of a document in the units the cap is written in: UTF-8 bytes. */
function utf8Bytes(text) {
  return TEXT_ENCODER.encode(text).length;
}

function isPlainObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function assertPlainObject(value, name) {
  if (!isPlainObject(value)) fail(`${name} must be an object`);
  return value;
}

function assertOnlyKeys(value, allowed, name) {
  assertPlainObject(value, name);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) fail(`${name} has unexpected field "${key}"`);
  }
}

function requireOwn(value, key, name) {
  if (!Object.hasOwn(value, key)) fail(`${name}.${key} is required`);
  return value[key];
}

function normalizeText(value, name, { max = MAX_TEXT_LENGTH, allowEmpty = false } = {}) {
  if (typeof value !== "string") fail(`${name} must be plain text`);
  const text = value.replace(/\r\n?/g, "\n");
  if ((!allowEmpty && text.trim() === "") || text.length > max) {
    fail(`${name} must be ${allowEmpty ? "at most" : "a non-empty value of at most"} ${max} characters`);
  }
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(text)) {
    fail(`${name} contains unsafe control characters`);
  }
  // Text reaches the UI through textContent. Keep comparison notation such as
  // >= and <=, but refuse tag-shaped markup so imported text cannot become a
  // tempting HTML payload in a later caller.
  if (HTML_TAG_RE.test(text)) fail(`${name} must be plain text, not HTML markup`);
  if (FORBIDDEN_URL_RE.test(text) || FILESYSTEM_PATH_RE.test(text)
      || ABSOLUTE_PATH_RE.test(text) || UNC_PATH_RE.test(text)) {
    fail(`${name} must not contain a URL or filesystem path`);
  }
  return text;
}

function normalizeIdentifier(value, name) {
  const identifier = normalizeText(value, name, { max: MAX_IDENTIFIER_LENGTH });
  if (!IDENTIFIER_RE.test(identifier) || FORBIDDEN_URL_RE.test(identifier)) {
    fail(`${name} must be a safe identifier`);
  }
  return identifier;
}

function normalizeHash(value, name) {
  if (typeof value !== "string" || !HASH_RE.test(value)) {
    fail(`${name} must be a 64-character SHA-256 source hash`);
  }
  return value.toLowerCase();
}

function normalizeLayerKey(value, name) {
  if (typeof value !== "string" || !LAYER_KEY_RE.test(value)) {
    fail(`${name} must be a known layer key`);
  }
  return value;
}

function normalizeDigest(value, name) {
  if (typeof value !== "string" || !DIGEST_RE.test(value)) {
    fail(`${name} must be a 16-character request digest`);
  }
  return value;
}

/**
 * FNV-1a 64-bit over a canonical string, as 16 lowercase hex characters.
 *
 * Not a cryptographic hash and never used as one: it exists so the capture and
 * restore sides can agree, byte for byte, on a grid affine or a rebuilt request
 * body without either side carrying the payload twice.
 */
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

function fnv1a64(text) {
  let hash = FNV_OFFSET;
  for (let index = 0; index < text.length; index += 1) {
    const unit = text.charCodeAt(index);
    // Feed both bytes of every UTF-16 code unit; no platform global needed.
    hash = ((hash ^ BigInt(unit & 0xff)) * FNV_PRIME) & FNV_MASK;
    hash = ((hash ^ BigInt(unit >>> 8)) * FNV_PRIME) & FNV_MASK;
  }
  return hash.toString(16).padStart(16, "0");
}

/** Canonical JSON: object keys sorted, so key order cannot change a digest. */
export function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value ?? null);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
}

/** Stable digest of any JSON-serializable request body. Pure. */
export function requestDigest(body) {
  return fnv1a64(canonicalJson(body === undefined ? null : body));
}

/**
 * Stable digest of a served free-text label, such as an engine string.
 * The record stores the digest so the label can be compared exactly on
 * restore without free text entering the file.
 */
export function textDigest(value) {
  if (typeof value !== "string" || value.length === 0 || value.length > MAX_TEXT_LENGTH) {
    fail("a digestible label must be a non-empty string");
  }
  return fnv1a64(value);
}

/** Stable digest of a row-major 4x4 grid affine. Pure. */
export function gridAffineDigest(affine) {
  if (!Array.isArray(affine) || affine.length !== 16
      || !affine.every((value) => typeof value === "number" && Number.isFinite(value))) {
    fail("grid affine must be 16 finite numbers");
  }
  // Round to picometre-scale so a float round-trip through JSON cannot change
  // the digest while a genuinely different grid still does.
  return fnv1a64(affine.map((value) => (value === 0 ? 0 : Number(value.toPrecision(12)))).join(","));
}

/**
 * Encode ascending voxel indices as [start, runLength, ...] pairs.
 * Duplicates and order are dropped: an ROI is a set of voxels, not a stroke.
 */
export function encodeVoxelRuns(indices) {
  const sorted = [...new Set(indices)].sort((a, b) => a - b);
  const runs = [];
  for (const index of sorted) {
    if (!Number.isSafeInteger(index) || index < 0 || index > MAX_VOXEL_INDEX) {
      fail("voxel index is outside the addressable grid");
    }
    const last = runs.length - 2;
    if (last >= 0 && runs[last] + runs[last + 1] === index) runs[last + 1] += 1;
    else runs.push(index, 1);
  }
  return runs;
}

/** Expand [start, runLength, ...] pairs back to ascending voxel indices. */
export function decodeVoxelRuns(runs) {
  const indices = [];
  for (let pair = 0; pair < runs.length; pair += 2) {
    for (let offset = 0; offset < runs[pair + 1]; offset += 1) indices.push(runs[pair] + offset);
  }
  return indices;
}

function normalizeRuns(value, name) {
  if (!Array.isArray(value) || value.length % 2 !== 0 || value.length > MAX_RUN_PAIRS * 2) {
    fail(`${name} must be at most ${MAX_RUN_PAIRS} [start, length] voxel runs`);
  }
  let previousEnd = -1;
  const runs = [];
  for (let pair = 0; pair < value.length; pair += 2) {
    const start = integer(value[pair], `${name}[${pair}]`, { min: 0, max: MAX_VOXEL_INDEX });
    const length = integer(value[pair + 1], `${name}[${pair + 1}]`, { min: 1, max: MAX_VOXEL_INDEX });
    if (start <= previousEnd) fail(`${name} runs must be ascending and non-overlapping`);
    previousEnd = start + length - 1;
    if (previousEnd > MAX_VOXEL_INDEX) fail(`${name} run leaves the addressable grid`);
    runs.push(start, length);
  }
  return runs;
}

function finiteNumber(value, name, { min = -Infinity, max = Infinity } = {}) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    fail(`${name} must be a finite number in [${min}, ${max}]`);
  }
  return value;
}

function integer(value, name, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const number = finiteNumber(value, name, { min, max });
  if (!Number.isSafeInteger(number)) fail(`${name} must be a safe integer`);
  return number;
}

function boolean(value, name) {
  if (typeof value !== "boolean") fail(`${name} must be true or false`);
  return value;
}

function enumValue(value, name, values) {
  if (typeof value !== "string" || !values.has(value)) {
    fail(`${name} has an unsupported value`);
  }
  return value;
}

function vector(value, name, length, { min = -MAX_COORDINATE, max = MAX_COORDINATE } = {}) {
  if (!Array.isArray(value) || value.length !== length) fail(`${name} must contain ${length} values`);
  return value.map((item, index) => finiteNumber(item, `${name}[${index}]`, { min, max }));
}

function nullable(value, normalize) {
  return value == null ? null : normalize(value);
}

function normalizeCreatedAt(value) {
  if (typeof value !== "string" || value.length > 40 || Number.isNaN(Date.parse(value))) {
    fail("createdAt must be an ISO timestamp");
  }
  if (new Date(value).toISOString() !== value) fail("createdAt must be a canonical UTC ISO timestamp");
  return value;
}

function normalizeContext(value, name, { strict = true } = {}) {
  if (strict) assertOnlyKeys(value, CONTEXT_FIELDS, name);
  else assertPlainObject(value, name);
  if (!Object.hasOwn(value, "caseSourceHash")) fail(`${name}.case source hash is required`);
  const context = {
    caseId: normalizeIdentifier(requireOwn(value, "caseId", name), `${name}.caseId`),
    gridId: normalizeIdentifier(requireOwn(value, "gridId", name), `${name}.gridId`),
    volumeId: normalizeIdentifier(requireOwn(value, "volumeId", name), `${name}.volumeId`),
    recipeHash: normalizeIdentifier(requireOwn(value, "recipeHash", name), `${name}.recipeHash`),
    caseSourceHash: normalizeHash(value.caseSourceHash, `${name}.case source hash`),
  };
  if (Object.hasOwn(value, "buildId") && value.buildId != null) {
    context.buildId = normalizeIdentifier(value.buildId, `${name}.buildId`);
  }
  return context;
}

function nullableCount(value, name) {
  return nullable(value, (number) => integer(number, name, { min: 0, max: MAX_COUNT }));
}

function nullableMillimetres(value, name) {
  return nullable(value, (number) => finiteNumber(number, name, { min: 0, max: MAX_COORDINATE }));
}

function optionalField(value, key) {
  return Object.hasOwn(value, key) ? value[key] : null;
}

function normalizeBank(value, index) {
  const name = `banks[${index}]`;
  assertOnlyKeys(value, BANK_FIELDS, name);
  const bankId = normalizeIdentifier(requireOwn(value, "bankId", name), `${name}.bankId`);
  const sourcePopulation = normalizeIdentifier(
    requireOwn(value, "sourcePopulation", name),
    `${name}.sourcePopulation`,
  );
  if (sourcePopulation !== `bank:${bankId}`) {
    fail(`${name}.sourcePopulation must be exactly "bank:${bankId}"; only named bank sources are restorable`);
  }
  const bank = {
    bankId,
    sourcePopulation,
    sourceHash: normalizeHash(requireOwn(value, "sourceHash", name), `${name}.sourceHash`),
    tractLabel: normalizeText(requireOwn(value, "tractLabel", name), `${name}.tractLabel`, { max: 160 }),
    displayedCount: nullableCount(optionalField(value, "displayedCount"), `${name}.displayedCount`),
    analyticCount: nullableCount(optionalField(value, "analyticCount"), `${name}.analyticCount`),
    fullCount: nullableCount(optionalField(value, "fullCount"), `${name}.fullCount`),
    p5: nullableMillimetres(optionalField(value, "p5"), `${name}.p5`),
    floorMm: nullableMillimetres(optionalField(value, "floorMm"), `${name}.floorMm`),
    clearanceMethod: nullable(
      optionalField(value, "clearanceMethod"),
      (text) => normalizeText(text, `${name}.clearanceMethod`, { max: 600 }),
    ),
    clearancePopulation: nullable(
      optionalField(value, "clearancePopulation"),
      (text) => normalizeText(text, `${name}.clearancePopulation`, { max: 240 }),
    ),
    weighting: nullable(
      optionalField(value, "weighting"),
      (text) => normalizeText(text, `${name}.weighting`, { max: 160 }),
    ),
    fidelityStatus: nullable(
      optionalField(value, "fidelityStatus"),
      (status) => enumValue(status, `${name}.fidelityStatus`, FIDELITY_STATES),
    ),
    operatingPoint: nullable(
      optionalField(value, "operatingPoint"),
      (text) => normalizeText(text, `${name}.operatingPoint`, { max: MAX_OPERATING_POINT_LENGTH }),
    ),
  };
  for (const key of ["displayedCount", "analyticCount"]) {
    if (bank.fullCount != null && bank[key] != null && bank[key] > bank.fullCount) {
      fail(`${name}.${key} cannot exceed ${name}.fullCount`);
    }
  }
  return bank;
}

function normalizeBanks(value, { minimum = 1 } = {}) {
  if (!Array.isArray(value) || value.length < minimum || value.length > MAX_BANKS) {
    fail(`banks must contain between ${minimum} and ${MAX_BANKS} named banks`);
  }
  const bankIds = new Set();
  const banks = value.map((bank, index) => {
    const normalized = normalizeBank(bank, index);
    if (bankIds.has(normalized.bankId)) fail(`banks contains duplicate bankId "${normalized.bankId}"`);
    bankIds.add(normalized.bankId);
    return normalized;
  });
  return { banks, bankIds };
}

function normalizeCamera(value, { requireOffset = false } = {}) {
  assertOnlyKeys(value, CAMERA_FIELDS, "view.camera");
  const position = vector(requireOwn(value, "position", "view.camera"), "view.camera.position", 3);
  const target = vector(requireOwn(value, "target", "view.camera"), "view.camera.target", 3);
  const up = vector(requireOwn(value, "up", "view.camera"), "view.camera.up", 3);
  let offset;
  if (Object.hasOwn(value, "offset")) {
    offset = vector(value.offset, "view.camera.offset", 2, { min: -2, max: 2 });
  } else if (requireOffset) {
    fail("view.camera.offset is required in a saved review record");
  } else {
    offset = [0, 0];
  }
  const direction = position.map((coordinate, index) => coordinate - target[index]);
  const directionLength = Math.hypot(...direction);
  const upLength = Math.hypot(...up);
  const crossLength = Math.hypot(
    direction[1] * up[2] - direction[2] * up[1],
    direction[2] * up[0] - direction[0] * up[2],
    direction[0] * up[1] - direction[1] * up[0],
  );
  if (directionLength < 1e-6 || upLength < 1e-6 || crossLength / (directionLength * upLength) < 1e-6) {
    fail("view.camera is degenerate");
  }
  return { position, target, up, offset };
}

function normalizeSlices(value) {
  assertOnlyKeys(value, SLICES_FIELDS, "view.slices");
  return {
    ax: integer(requireOwn(value, "ax", "view.slices"), "view.slices.ax", { min: 0, max: MAX_SLICE }),
    cor: integer(requireOwn(value, "cor", "view.slices"), "view.slices.cor", { min: 0, max: MAX_SLICE }),
    sag: integer(requireOwn(value, "sag", "view.slices"), "view.slices.sag", { min: 0, max: MAX_SLICE }),
  };
}

function normalizeLayout(value, { requireOverviewFraction = false } = {}) {
  assertOnlyKeys(value, LAYOUT_FIELDS, "view.layout");
  const focusedSlice = requireOwn(value, "focusedSlice", "view.layout");
  if (focusedSlice !== null && !AXES.has(focusedSlice)) {
    fail("view.layout.focusedSlice must be null, ax, cor, or sag");
  }
  const dockFraction = finiteNumber(
    requireOwn(value, "dockFraction", "view.layout"),
    "view.layout.dockFraction",
    { min: Number.MIN_VALUE, max: 1 },
  );
  let overviewFraction;
  if (Object.hasOwn(value, "overviewFraction")) {
    overviewFraction = finiteNumber(value.overviewFraction, "view.layout.overviewFraction", {
      min: Number.MIN_VALUE,
      max: 1,
    });
  } else if (requireOverviewFraction) {
    fail("view.layout.overview fraction is required in a saved review record");
  } else {
    overviewFraction = dockFraction;
  }
  return {
    slicesOpen: boolean(requireOwn(value, "slicesOpen", "view.layout"), "view.layout.slicesOpen"),
    dockFraction,
    overviewFraction,
    focusedSlice,
  };
}

function normalizePin(value, layerKeys, version) {
  if (value == null) return null;
  const sourceIndex = (raw) => integer(raw, "view.pin.source index", { min: 0, max: MAX_SOURCE_INDEX });
  if (version === 1) {
    assertOnlyKeys(value, PIN_FIELDS_V1, "view.pin");
    const bankId = normalizeIdentifier(requireOwn(value, "bankId", "view.pin"), "view.pin.bankId");
    if (!layerKeys.has(`bank:${bankId}`)) fail("view.pin.bankId must name a saved bank");
    return { bankId, sourceIndex: sourceIndex(requireOwn(value, "sourceIndex", "view.pin")) };
  }
  assertOnlyKeys(value, PIN_FIELDS_V2, "view.pin");
  const layerKey = normalizeLayerKey(requireOwn(value, "layerKey", "view.pin"), "view.pin.layerKey");
  if (!layerKeys.has(layerKey)) fail("view.pin.layerKey must name a saved layer");
  return { layerKey, sourceIndex: sourceIndex(requireOwn(value, "sourceIndex", "view.pin")) };
}

/**
 * The along-tract node interval the profile panel was showing.
 *
 * Display state only. It names the layer it belongs to so a restore can check
 * the layer is still there; `from`/`to` are node indices in the served profile
 * (n_points), not measurements. Absent or null means no selection.
 */
function normalizeProfileInterval(value, layerKeys, version) {
  if (version === 1 || value == null) return null;
  assertPlainObject(value, "view.profileInterval");
  assertOnlyKeys(value, PROFILE_INTERVAL_FIELDS, "view.profileInterval");
  const layerKey = normalizeLayerKey(
    requireOwn(value, "layerKey", "view.profileInterval"), "view.profileInterval.layerKey",
  );
  if (!layerKeys.has(layerKey)) fail("view.profileInterval.layerKey must name a saved layer");
  // A profile exists only for a named bank; a generated or subset layer key
  // here would be a record claiming evidence the panel never showed.
  if (!layerKey.startsWith("bank:")) fail("view.profileInterval.layerKey must name a named bank layer");
  const scalar = enumValue(
    requireOwn(value, "scalar", "view.profileInterval"), "view.profileInterval.scalar", PROFILE_SCALARS,
  );
  const bound = { min: 0, max: MAX_PROFILE_NODE };
  const from = integer(requireOwn(value, "from", "view.profileInterval"), "view.profileInterval.from", bound);
  const to = integer(requireOwn(value, "to", "view.profileInterval"), "view.profileInterval.to", bound);
  if (to < from) fail("view.profileInterval.to must not precede view.profileInterval.from");
  return { layerKey, scalar, from, to };
}

function normalizeView(value, layerKeys, {
  requireCameraOffset = false,
  requireOverviewFraction = false,
  version = SCHEMA_VERSION,
} = {}) {
  assertOnlyKeys(value, version === 1 ? VIEW_FIELDS_V1 : VIEW_FIELDS_V2, "view");
  let focus;
  if (version === 1) {
    const focusBankId = normalizeIdentifier(requireOwn(value, "focusBankId", "view"), "view.focusBankId");
    if (!layerKeys.has(`bank:${focusBankId}`)) fail("view.focusBankId must name a saved bank");
    focus = { focusBankId };
  } else {
    const focusLayerKey = normalizeLayerKey(requireOwn(value, "focusLayerKey", "view"), "view.focusLayerKey");
    if (!layerKeys.has(focusLayerKey)) fail("view.focusLayerKey must name a saved layer");
    focus = { focusLayerKey };
  }
  const slicePlane = enumValue(requireOwn(value, "slicePlane", "view"), "view.slicePlane", new Set(["off", ...AXES]));
  return {
    ...focus,
    colour: enumValue(requireOwn(value, "colour", "view"), "view.colour", COLOURS),
    camera: normalizeCamera(requireOwn(value, "camera", "view"), { requireOffset: requireCameraOffset }),
    slices: normalizeSlices(requireOwn(value, "slices", "view")),
    underlay: enumValue(requireOwn(value, "underlay", "view"), "view.underlay", UNDERLAYS),
    radius: finiteNumber(requireOwn(value, "radius", "view"), "view.radius", { min: 0.01, max: 10 }),
    displayNearLesion: boolean(requireOwn(value, "displayNearLesion", "view"), "view.displayNearLesion"),
    nearLesionFrac: finiteNumber(requireOwn(value, "nearLesionFrac", "view"), "view.nearLesionFrac", { min: 0, max: 1 }),
    nearRadiusMm: finiteNumber(requireOwn(value, "nearRadiusMm", "view"), "view.nearRadiusMm", { min: 0, max: 500 }),
    hideLowSupport: boolean(requireOwn(value, "hideLowSupport", "view"), "view.hideLowSupport"),
    onlyLowSupport: boolean(requireOwn(value, "onlyLowSupport", "view"), "view.onlyLowSupport"),
    hullVeil: enumValue(requireOwn(value, "hullVeil", "view"), "view.hullVeil", HULL_VEILS),
    hullVisible: boolean(requireOwn(value, "hullVisible", "view"), "view.hullVisible"),
    lesionVisible: boolean(requireOwn(value, "lesionVisible", "view"), "view.lesionVisible"),
    slicePlane,
    layout: normalizeLayout(requireOwn(value, "layout", "view"), { requireOverviewFraction }),
    pin: normalizePin(requireOwn(value, "pin", "view"), layerKeys, version),
    profileInterval: normalizeProfileInterval(value.profileInterval ?? null, layerKeys, version),
  };
}

function normalizeParamValue(value, type, name) {
  if (type === "number") return finiteNumber(value, name, { min: -MAX_COORDINATE, max: MAX_COORDINATE });
  if (type === "boolean") return boolean(value, name);
  // "token": a short machine value. Free text is refused by name, so prose a
  // clinician typed elsewhere can never ride along inside a request record.
  if (typeof value !== "string" || !PARAM_TOKEN_RE.test(value)) {
    fail(`${name} must be a short token, not free text`);
  }
  return value;
}

/**
 * A request's `params` block, checked against its route's allowlist.
 *
 * Every key must be declared for the route and every value must match its
 * declared type. An unknown key refuses the save and names itself, so a new
 * request field has to be declared here before it can ever be written to a file.
 */
function normalizeParams(value, schema, name) {
  assertPlainObject(value, name);
  for (const key of Object.keys(value)) {
    if (!Object.hasOwn(schema, key)) {
      fail(`${name} has unexpected request field "${key}"; only declared request fields can be saved`);
    }
  }
  const out = {};
  for (const key of Object.keys(schema).sort()) {
    if (!Object.hasOwn(value, key)) continue;
    const type = schema[key];
    out[key] = isPlainObject(type)
      ? normalizeParams(value[key], type, `${name}.${key}`)
      : normalizeParamValue(value[key], type, `${name}.${key}`);
  }
  return out;
}

function normalizeGenerated(value, index) {
  const name = `generated[${index}]`;
  assertOnlyKeys(value, GENERATED_FIELDS, name);
  const layerKey = normalizeLayerKey(requireOwn(value, "layerKey", name), `${name}.layerKey`);

  const rawRequest = requireOwn(value, "request", name);
  assertOnlyKeys(rawRequest, GENERATED_REQUEST_FIELDS, `${name}.request`);
  const route = requireOwn(rawRequest, "route", `${name}.request`);
  if (typeof route !== "string" || !GENERATED_ROUTES.has(route)) {
    fail(`${name}.request.route is not a route that produces a restorable layer`);
  }
  const bankId = nullable(optionalField(rawRequest, "bankId"), (id) => normalizeIdentifier(id, `${name}.request.bankId`));
  if (layerKey.startsWith("bank:") && layerKey !== `bank:${bankId}`) {
    fail(`${name}.request.bankId must match ${name}.layerKey`);
  }
  const roiRef = optionalField(rawRequest, "roiRef");
  if (roiRef !== null && roiRef !== "paint") fail(`${name}.request.roiRef must be null or "paint"`);
  const request = {
    route,
    bankId,
    seedPresetId: nullable(
      optionalField(rawRequest, "seedPresetId"),
      (id) => normalizeIdentifier(id, `${name}.request.seedPresetId`),
    ),
    roiRef,
    params: normalizeParams(
      Object.hasOwn(rawRequest, "params") ? rawRequest.params ?? {} : {},
      GENERATED_PARAM_SCHEMAS.get(route),
      `${name}.request.params`,
    ),
    digest: normalizeDigest(requireOwn(rawRequest, "digest", `${name}.request`), `${name}.request.digest`),
  };
  if (!isPlainObject(request.params)) fail(`${name}.request.params must be an object`);

  const edgeBacked = route === EDGE_TUBES_ROUTE;
  // An assigned edge's layerKey, request.params {a,b}, and identity.sourcePopulation
  // are three independently-writable places the same (a,b) pair could drift apart in
  // a hand-edited or corrupted file; refuse the whole entry rather than restore a
  // layer whose displayed key disagrees with the edge it actually loads.
  if (edgeBacked) {
    const a = request.params?.a, b = request.params?.b;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      fail(`${name}.request.params must carry {a,b} for an edge tube layer`);
    }
    const expectedKey = `edge:${a}-${b}`;
    if (layerKey !== expectedKey) {
      fail(`${name}.layerKey ("${layerKey}") must match its request.params {a:${a},b:${b}} ("${expectedKey}")`);
    }
  }

  // Every identity field is mandatory and non-null: restore compares all of
  // them for equality, and a null would silently compare nothing.
  const rawIdentity = requireOwn(value, "identity", name);
  assertOnlyKeys(rawIdentity, GENERATED_IDENTITY_FIELDS, `${name}.identity`);
  const bankBacked = bankId !== null;
  const bankSourceHash = requireOwn(rawIdentity, "bankSourceHash", `${name}.identity`);
  if (bankBacked && bankSourceHash == null) {
    fail(`${name}.identity.bankSourceHash is required for a bank-backed layer`);
  }
  if (!bankBacked && bankSourceHash !== null) {
    fail(`${name}.identity.bankSourceHash must be null for a layer that is not bank-backed`);
  }
  // An edge tube layer is not bank-backed (bankId stays null: an assigned
  // edge is not a named bank), so its identity is a joint tuple instead of a
  // single source hash: the edge tck's own DATA-segment sha256 (what
  // bankSourceHash would mean for a bank), the corpus sha256, the
  // parcellation sha256, and the connectome generation id — all four
  // mandatory and all four compared on restore, so a rebuild that keeps the
  // same corpus but republishes a new generation (different assignment
  // lineage) is caught even though the corpus hash alone would not catch it.
  const edgeSourceHash = requireOwn(rawIdentity, "edgeSourceHash", `${name}.identity`);
  const corpusSourceHash = requireOwn(rawIdentity, "corpusSourceHash", `${name}.identity`);
  const parcellationSourceHash = requireOwn(rawIdentity, "parcellationSourceHash", `${name}.identity`);
  const generationId = requireOwn(rawIdentity, "generationId", `${name}.identity`);
  for (const [field, raw] of [
    ["edgeSourceHash", edgeSourceHash], ["corpusSourceHash", corpusSourceHash],
    ["parcellationSourceHash", parcellationSourceHash], ["generationId", generationId],
  ]) {
    if (edgeBacked && raw == null) fail(`${name}.identity.${field} is required for an edge tube layer`);
    if (!edgeBacked && raw !== null) fail(`${name}.identity.${field} must be null for a layer that is not an edge tube layer`);
  }
  const identity = {
    sourcePopulation: normalizeText(
      requireOwn(rawIdentity, "sourcePopulation", `${name}.identity`),
      `${name}.identity.sourcePopulation`,
      { max: MAX_IDENTIFIER_LENGTH },
    ),
    // The served engine label is free text ("FILTER | corpus"). A record keeps
    // its digest instead: exact to compare, and no free text in the file.
    engineDigest: normalizeDigest(
      requireOwn(rawIdentity, "engineDigest", `${name}.identity`),
      `${name}.identity.engineDigest`,
    ),
    nReturned: integer(
      requireOwn(rawIdentity, "nReturned", `${name}.identity`),
      `${name}.identity.nReturned`,
      { min: 0, max: MAX_COUNT },
    ),
    lineCount: integer(
      requireOwn(rawIdentity, "lineCount", `${name}.identity`),
      `${name}.identity.lineCount`,
      { min: 0, max: MAX_COUNT },
    ),
    bankSourceHash: bankBacked ? normalizeHash(bankSourceHash, `${name}.identity.bankSourceHash`) : null,
    edgeSourceHash: edgeBacked ? normalizeHash(edgeSourceHash, `${name}.identity.edgeSourceHash`) : null,
    corpusSourceHash: edgeBacked ? normalizeHash(corpusSourceHash, `${name}.identity.corpusSourceHash`) : null,
    parcellationSourceHash: edgeBacked
      ? normalizeHash(parcellationSourceHash, `${name}.identity.parcellationSourceHash`) : null,
    generationId: edgeBacked ? normalizeIdentifier(generationId, `${name}.identity.generationId`) : null,
  };
  if (edgeBacked && identity.sourcePopulation !== layerKey) {
    fail(`${name}.identity.sourcePopulation ("${identity.sourcePopulation}") must match ${name}.layerKey ("${layerKey}")`);
  }
  return {
    layerKey,
    tractLabel: normalizeText(requireOwn(value, "tractLabel", name), `${name}.tractLabel`, { max: 160 }),
    request,
    identity,
  };
}

function normalizeGeneratedList(value) {
  if (value == null) return [];
  if (!Array.isArray(value) || value.length > MAX_GENERATED) {
    fail(`generated must contain at most ${MAX_GENERATED} generated or subset layers`);
  }
  const keys = new Set();
  return value.map((entry, index) => {
    const generated = normalizeGenerated(entry, index);
    if (keys.has(generated.layerKey)) fail(`generated contains duplicate layerKey "${generated.layerKey}"`);
    keys.add(generated.layerKey);
    return generated;
  });
}

function normalizePaintGrid(value) {
  assertOnlyKeys(value, PAINT_GRID_FIELDS, "paint.grid");
  const dims = requireOwn(value, "dims", "paint.grid");
  if (!Array.isArray(dims) || dims.length !== 3) fail("paint.grid.dims must be three voxel counts");
  const normalizedDims = dims.map((dim, index) => integer(dim, `paint.grid.dims[${index}]`, { min: 1, max: MAX_DIMENSION }));
  return {
    gridId: normalizeIdentifier(requireOwn(value, "gridId", "paint.grid"), "paint.grid.gridId"),
    dims: normalizedDims,
    affineDigest: normalizeDigest(requireOwn(value, "affineDigest", "paint.grid"), "paint.grid.affineDigest"),
  };
}

function normalizeRois(value, dims) {
  assertOnlyKeys(value, ROI_FIELDS, "paint.rois");
  const voxels = dims[0] * dims[1] * dims[2];
  const runsFor = (raw, name) => {
    const runs = normalizeRuns(raw, name);
    for (let pair = 0; pair < runs.length; pair += 2) {
      if (runs[pair] + runs[pair + 1] > voxels) fail(`${name} names a voxel outside paint.grid.dims`);
    }
    return runs;
  };
  // An AND region opened by "New AND region" is empty until it is painted.
  // It is kept so paint.andIndex still names the region the user is working in.
  const and = requireOwn(value, "and", "paint.rois");
  if (!Array.isArray(and) || and.length > MAX_AND_REGIONS) {
    fail(`paint.rois.and must contain at most ${MAX_AND_REGIONS} regions`);
  }
  return {
    seed: runsFor(requireOwn(value, "seed", "paint.rois"), "paint.rois.seed"),
    and: and.map((region, index) => runsFor(region, `paint.rois.and[${index}]`)),
    or: runsFor(requireOwn(value, "or", "paint.rois"), "paint.rois.or"),
    not: runsFor(requireOwn(value, "not", "paint.rois"), "paint.rois.not"),
  };
}

function normalizePaint(value) {
  if (value == null) return null;
  assertOnlyKeys(value, PAINT_FIELDS, "paint");
  const grid = normalizePaintGrid(requireOwn(value, "grid", "paint"));
  const rois = normalizeRois(requireOwn(value, "rois", "paint"), grid.dims);
  const andIndex = integer(requireOwn(value, "andIndex", "paint"), "paint.andIndex", {
    min: 0,
    max: MAX_AND_REGIONS - 1,
  });
  if (rois.and.length && andIndex >= rois.and.length) fail("paint.andIndex is outside paint.rois.and");
  if (!rois.and.length && andIndex !== 0) fail("paint.andIndex must be 0 when there are no AND regions");
  const paint = {
    grid,
    paintMode: boolean(requireOwn(value, "paintMode", "paint"), "paint.paintMode"),
    role: enumValue(requireOwn(value, "role", "paint"), "paint.role", ROI_ROLES),
    andIndex,
    brushRadiusMm: finiteNumber(requireOwn(value, "brushRadiusMm", "paint"), "paint.brushRadiusMm", {
      min: 0,
      max: 100,
    }),
    seedPresetId: nullable(
      optionalField(value, "seedPresetId"),
      (id) => normalizeIdentifier(id, "paint.seedPresetId"),
    ),
    rois,
  };
  const size = JSON.stringify(paint).length;
  if (size > MAX_PAINT_CHARS) {
    fail(
      `painted ROI state is ${size} characters, over the ${MAX_PAINT_CHARS}-character limit; `
      + "clear some paint or simplify the regions, then save again",
    );
  }
  return paint;
}

function normalizeOverlays(value) {
  if (value == null) return { priors: [], parcelNetwork: null };
  assertOnlyKeys(value, OVERLAYS_FIELDS, "overlays");
  const priors = requireOwn(value, "priors", "overlays");
  if (!Array.isArray(priors) || priors.length > MAX_PRIORS) {
    fail(`overlays.priors must contain at most ${MAX_PRIORS} prior ids`);
  }
  const seen = new Set();
  const ids = priors.map((id, index) => {
    const priorId = normalizeIdentifier(id, `overlays.priors[${index}]`);
    if (seen.has(priorId)) fail(`overlays.priors contains duplicate id "${priorId}"`);
    seen.add(priorId);
    return priorId;
  });
  return {
    priors: ids,
    parcelNetwork: nullable(
      requireOwn(value, "parcelNetwork", "overlays"),
      (network) => integer(network, "overlays.parcelNetwork", { min: 1, max: 64 }),
    ),
  };
}

function normalizeEnvelope(value, layerKeys) {
  if (value == null) return null;
  assertOnlyKeys(value, ENVELOPE_FIELDS, "envelope");
  const layerKey = normalizeLayerKey(requireOwn(value, "layerKey", "envelope"), "envelope.layerKey");
  if (!layerKeys.has(layerKey)) fail("envelope.layerKey must name a saved layer");
  return {
    layerKey,
    marginMm: finiteNumber(requireOwn(value, "marginMm", "envelope"), "envelope.marginMm", {
      min: 0.1,
      max: 100,
    }),
  };
}

function normalizeEvidence(value) {
  assertOnlyKeys(value, EVIDENCE_FIELDS, "evidence");
  return {
    preflightState: enumValue(
      requireOwn(value, "preflightState", "evidence"),
      "evidence.preflightState",
      PREFLIGHT_STATES,
    ),
    preflightText: normalizeText(requireOwn(value, "preflightText", "evidence"), "evidence.preflightText", {
      max: MAX_TEXT_LENGTH,
      allowEmpty: true,
    }),
    spaceLabel: normalizeText(requireOwn(value, "spaceLabel", "evidence"), "evidence.spaceLabel", {
      max: 400,
      allowEmpty: true,
    }),
  };
}

function normalizeUnsupported(value) {
  if (!Array.isArray(value) || value.length > MAX_UNSUPPORTED) {
    fail(`unsupported must contain at most ${MAX_UNSUPPORTED} plain-text explanations`);
  }
  return value.map((entry, index) => normalizeText(entry, `unsupported[${index}]`, { max: 400 }));
}

/**
 * Reviewer-entered outcome. Absent (undefined or null) reads as not-recorded
 * with an empty note; an unknown effect or unexpected field is refused, never
 * coerced, so a hand-edited file cannot slip a new category past the reader.
 */
function normalizeOutcome(value) {
  if (value == null) return { effect: "not-recorded", note: "" };
  assertOnlyKeys(value, OUTCOME_FIELDS, "outcome");
  const effect = enumValue(requireOwn(value, "effect", "outcome"), "outcome.effect", OUTCOME_EFFECT_SET);
  const note = Object.hasOwn(value, "note")
    ? normalizeText(value.note, "outcome.note", { max: MAX_OUTCOME_NOTE_LENGTH, allowEmpty: true })
    : "";
  return { effect, note };
}

function declaredVersion(value) {
  assertPlainObject(value, "record");
  const version = requireOwn(value, "schemaVersion", "record");
  if (!SUPPORTED_SCHEMA_VERSIONS.has(version)) {
    fail(`schemaVersion must be 1, 2 or ${SCHEMA_VERSION}`);
  }
  return version;
}

function normalizePayload(value, { forCreate = false, version = SCHEMA_VERSION } = {}) {
  if (version === 1 && !forCreate) {
    assertPlainObject(value, "record");
    for (const field of [...V2_OBJECT_FIELDS, ...V3_OBJECT_FIELDS]) {
      if (Object.hasOwn(value, field)) fail(`a schema 1 record cannot carry "${field}"`);
    }
  }
  if (version === 2 && !forCreate) {
    assertPlainObject(value, "record");
    for (const field of V3_OBJECT_FIELDS) {
      if (Object.hasOwn(value, field)) fail(`a schema 2 record cannot carry "${field}"`);
    }
  }
  const allowedFields = forCreate
    ? CREATE_FIELDS
    : ({ 1: RECORD_FIELDS_V1, 2: RECORD_FIELDS_V2 }[version] || RECORD_FIELDS_V3);
  assertOnlyKeys(value, allowedFields, "record");
  const { banks, bankIds } = normalizeBanks(requireOwn(value, "banks", "record"), {
    minimum: version === 1 ? 1 : 0,
  });
  const unsupported = Object.hasOwn(value, "unsupported") ? normalizeUnsupported(value.unsupported) : [];
  if (unsupported.length) {
    fail(`cannot restore a record with unsupported state: ${unsupported.join("; ")}`);
  }
  const core = {
    schemaVersion: version,
    createdAt: forCreate && !Object.hasOwn(value, "createdAt")
      ? new Date().toISOString()
      : normalizeCreatedAt(requireOwn(value, "createdAt", "record")),
    context: normalizeContext(requireOwn(value, "context", "record"), "context"),
    banks,
  };
  if (version === 1) {
    const layerKeys = new Set([...bankIds].map((id) => `bank:${id}`));
    return {
      ...core,
      view: normalizeView(requireOwn(value, "view", "record"), layerKeys, {
        requireCameraOffset: !forCreate,
        requireOverviewFraction: !forCreate,
        version: 1,
      }),
      evidence: normalizeEvidence(requireOwn(value, "evidence", "record")),
      unsupported,
    };
  }
  if (!forCreate) {
    for (const field of V2_OBJECT_FIELDS) {
      if (!Object.hasOwn(value, field)) fail(`record.${field} is required in a saved schema ${version} record`);
    }
  }
  const generated = normalizeGeneratedList(optionalField(value, "generated"));
  if (!banks.length && !generated.length) {
    fail("a record must carry at least one named bank or generated layer");
  }
  // One key names one layer. A bank and a generated layer sharing a key would
  // make focus, pin and envelope references ambiguous, so it is refused here
  // rather than resolved by whichever list happened to be read first.
  const layerKeys = new Set();
  for (const key of [...bankIds].map((id) => `bank:${id}`)) layerKeys.add(key);
  for (const entry of generated) {
    if (layerKeys.has(entry.layerKey)) {
      fail(`layer key "${entry.layerKey}" names both a saved bank and a generated layer`);
    }
    layerKeys.add(entry.layerKey);
  }
  const paint = normalizePaint(optionalField(value, "paint"));
  if (paint && paint.grid.gridId !== core.context.gridId) {
    fail("paint.grid.gridId must match the record's context grid");
  }
  for (const entry of generated) {
    if (entry.request.roiRef === "paint" && !paint) {
      fail(`generated layer "${entry.layerKey}" references painted ROIs that are not in the record`);
    }
  }
  const record = {
    ...core,
    generated,
    paint,
    overlays: normalizeOverlays(optionalField(value, "overlays")),
    envelope: normalizeEnvelope(optionalField(value, "envelope"), layerKeys),
    view: normalizeView(requireOwn(value, "view", "record"), layerKeys, {
      requireCameraOffset: !forCreate,
      requireOverviewFraction: !forCreate,
      // Schema 3 changed no view field; the view reader is shared from 2 on.
      version: 2,
    }),
    evidence: normalizeEvidence(requireOwn(value, "evidence", "record")),
    unsupported,
  };
  // A schema 2 record has no outcome key at all, so it reads back byte for byte.
  if (version >= 3) record.outcome = normalizeOutcome(optionalField(value, "outcome"));
  return record;
}

function displayValue(value, suffix = "") {
  return value == null ? "unknown" : `${value}${suffix}`;
}

function runVoxelCount(runs) {
  let total = 0;
  for (let pair = 1; pair < runs.length; pair += 2) total += runs[pair];
  return total;
}

function summariseV2Objects(record, lines) {
  lines.push("", "Generated and subset layers — request provenance only; no streamline geometry is saved.");
  if (!record.generated.length) {
    lines.push("None.");
  } else {
    for (const [index, entry] of record.generated.entries()) {
      const route = GENERATED_ROUTES.get(entry.request.route);
      lines.push(
        `${index + 1}. ${entry.tractLabel} (${entry.layerKey})`,
        `   Source population: ${entry.identity.sourcePopulation}`,
        `   Request: ${route.label} (${route.reproducible ? "reproducible" : `not reproducible, ${route.why}`})`,
        `   Request digest: ${entry.request.digest}`,
        `   Seed preset: ${displayValue(entry.request.seedPresetId)}; painted ROIs: ${entry.request.roiRef ? "yes" : "no"}`,
        `   Parameters: ${canonicalJson(entry.request.params)}`,
        `   Returned: engine digest ${entry.identity.engineDigest}; n ${entry.identity.nReturned}; displayed ${entry.identity.lineCount}`,
        `   Bank source hash: ${displayValue(entry.identity.bankSourceHash)}`,
      );
    }
  }

  lines.push("", "Painted and preset ROI state");
  if (!record.paint) {
    lines.push("None.");
  } else {
    const { grid, rois } = record.paint;
    lines.push(
      `Grid: ${grid.gridId}; dims ${grid.dims.join(" x ")}; affine digest ${grid.affineDigest}`,
      `Paint mode: ${record.paint.paintMode ? "on" : "off"}; role ${record.paint.role}; AND region ${record.paint.andIndex + 1}`,
      `Brush radius: ${record.paint.brushRadiusMm} mm; seed preset ${displayValue(record.paint.seedPresetId)}`,
      `SEED voxels: ${runVoxelCount(rois.seed)}; OR voxels: ${runVoxelCount(rois.or)}; NOT voxels: ${runVoxelCount(rois.not)}`,
      `AND regions: ${rois.and.length ? rois.and.map((region) => runVoxelCount(region)).join(", ") + " voxels" : "none"}`,
    );
  }

  lines.push(
    "",
    "Population overlays — population priors, never patient anatomy.",
    `Atlas priors: ${record.overlays.priors.length ? record.overlays.priors.join(", ") : "none"}`,
    `Parcellation network: ${displayValue(record.overlays.parcelNetwork)}`,
    "",
    "Display envelope",
    record.envelope
      ? `Layer ${record.envelope.layerKey} at ${record.envelope.marginMm} mm; display only, not clearance and not a resection margin.`
      : "None.",
  );
}

function buildSummary(record) {
  const version = record.schemaVersion;
  const lines = [
    version === 1 ? "TractLab named-bank review record" : "TractLab review record",
    `Schema: ${version}`,
    `Created: ${record.createdAt}`,
    "",
    "Context",
    `Case: ${record.context.caseId}`,
    `Grid: ${record.context.gridId}`,
    `Volume: ${record.context.volumeId}`,
    `Recipe: ${record.context.recipeHash}`,
    `Case source hash: ${record.context.caseSourceHash}`,
  ];
  if (record.context.buildId) lines.push(`Build: ${record.context.buildId}`);
  lines.push("", "Banks — Historical annotations only; do not treat saved measurements as current.");
  for (const [index, bank] of record.banks.entries()) {
    lines.push(
      `${index + 1}. ${bank.tractLabel} (${bank.bankId})`,
      `   Source population: ${bank.sourcePopulation}`,
      `   Source hash: ${bank.sourceHash}`,
      `   Counts: displayed ${displayValue(bank.displayedCount)}; analytic ${displayValue(bank.analyticCount)}; full ${displayValue(bank.fullCount)}`,
      `   p5: ${displayValue(bank.p5, " mm")}; floor: ${displayValue(bank.floorMm, " mm")}`,
      `   Clearance method: ${displayValue(bank.clearanceMethod)}; population: ${displayValue(bank.clearancePopulation)}`,
      `   Weighting: ${displayValue(bank.weighting)}; fidelity: ${displayValue(bank.fidelityStatus)}; operating point: ${displayValue(bank.operatingPoint)}`,
    );
  }
  const camera = record.view.camera;
  lines.push(
    "",
    "Saved view",
    `Focus: ${version === 1 ? record.view.focusBankId : record.view.focusLayerKey}; colour: ${record.view.colour}; underlay: ${record.view.underlay}`,
    `Camera position: ${camera.position.join(", ")}; target: ${camera.target.join(", ")}; up: ${camera.up.join(", ")}`,
    `Camera offset: ${camera.offset.join(", ")} normalized viewport fraction`,
    `Slices: ax ${record.view.slices.ax}; cor ${record.view.slices.cor}; sag ${record.view.slices.sag}; plane ${record.view.slicePlane}`,
    `Tube radius: ${record.view.radius}; near lesion: ${record.view.displayNearLesion ? "on" : "off"} (${record.view.nearLesionFrac}, ${record.view.nearRadiusMm} mm)`,
    `Low support: hide ${record.view.hideLowSupport ? "on" : "off"}; only ${record.view.onlyLowSupport ? "on" : "off"}`,
    `Hull: ${record.view.hullVisible ? record.view.hullVeil : "hidden"}; lesion: ${record.view.lesionVisible ? "visible" : "hidden"}`,
    `Layout: slices ${record.view.layout.slicesOpen ? "open" : "closed"}; dock ${record.view.layout.dockFraction}; overview ${record.view.layout.overviewFraction}; focused slice ${record.view.layout.focusedSlice ?? "none"}`,
  );
  if (record.view.pin) {
    const pinned = version === 1 ? record.view.pin.bankId : record.view.pin.layerKey;
    lines.push(`Pin: ${pinned} source #${record.view.pin.sourceIndex + 1} (saved source index ${record.view.pin.sourceIndex})`);
  } else {
    lines.push("Pin: none");
  }
  if (version !== 1) summariseV2Objects(record, lines);
  if (version >= 3) {
    lines.push(
      "",
      "Review outcome — reviewer self-report, not a measurement.",
      `${OUTCOME_QUESTION} ${OUTCOME_LABELS[record.outcome.effect]}`,
      `Outcome note: ${record.outcome.note || "none"}`,
    );
  }
  lines.push(
    "",
    "Evidence (saved plain text)",
    `Preflight state: ${record.evidence.preflightState}`,
    `Preflight text: ${record.evidence.preflightText || "unknown"}`,
    `Space label: ${record.evidence.spaceLabel || "unknown"}`,
    "",
    "Recorded floor is a display limit, not measured spatial accuracy; local residual error remains unmeasured. p5 describes reconstructed streamlines.",
    "Saved measurements are historical annotations only. Current geometry and measurements must be re-read from matching served named-bank sources before restoring this view.",
  );
  if (version !== 1) {
    lines.push(
      "Generated and subset layers are redrawn only by re-issuing their saved request and matching the identity the server returns; a mismatch is reported, never drawn.",
    );
  }
  const summary = lines.join("\n");
  const limit = version === 1 ? MAX_V1_INPUT_CHARS : MAX_INPUT_CHARS;
  if (utf8Bytes(summary) > limit) {
    fail(`summary exceeds the ${limit}-byte review-record limit`);
  }
  return summary;
}

function normalizeRecord(value, { verifySummary = false } = {}) {
  const version = declaredVersion(value);
  const record = normalizePayload(value, { version });
  if (verifySummary) {
    const savedSummary = normalizeText(requireOwn(value, "summary", "record"), "record.summary", {
      max: version === 1 ? MAX_V1_INPUT_CHARS : MAX_INPUT_CHARS,
      allowEmpty: false,
    });
    const expectedSummary = buildSummary(record);
    if (savedSummary !== expectedSummary) {
      fail("summary does not match the structured review record");
    }
  }
  return record;
}

/**
 * Upgrade a validated schema-2 record to the current schema. Pure.
 *
 * Schema 2 had no outcome, so the upgraded record states it as not recorded
 * rather than inferring one. Every schema-2 field is carried unchanged.
 */
function upgradeV2(record) {
  const upgraded = {
    ...record,
    schemaVersion: SCHEMA_VERSION,
    outcome: { effect: "not-recorded", note: "" },
  };
  const revalidated = normalizePayload(upgraded, { version: SCHEMA_VERSION });
  return { ...revalidated, summary: buildSummary(revalidated) };
}

/**
 * Upgrade a validated schema-1 record to schema 2, then on to the current schema. Pure.
 *
 * Schema 1 could not hold generated layers, painted ROIs, overlays, or an
 * envelope, so the upgraded record states all four as absent rather than
 * inferring any of them. A schema-1 focus/pin bank id becomes its layer key.
 */
function upgradeV1(record) {
  const { focusBankId, pin, ...view } = record.view;
  const upgraded = {
    ...record,
    schemaVersion: 2,
    view: {
      ...view,
      focusLayerKey: `bank:${focusBankId}`,
      pin: pin ? { layerKey: `bank:${pin.bankId}`, sourceIndex: pin.sourceIndex } : null,
    },
    generated: [],
    paint: null,
    overlays: { priors: [], parcelNetwork: null },
    envelope: null,
  };
  // Re-validate through the schema-2 reader so an upgrade can never widen what
  // the reader itself would accept.
  return upgradeV2(normalizePayload(upgraded, { version: 2 }));
}

/** Upgrade any supported saved record to the current schema. Pure. */
export function upgradeReviewRecord(value) {
  const version = declaredVersion(value);
  const record = normalizeRecord(value, { verifySummary: true });
  if (version === SCHEMA_VERSION) return { ...record, summary: buildSummary(record) };
  return version === 2 ? upgradeV2(record) : upgradeV1(record);
}

/** Create a serializable current-schema review record. */
export function createReviewRecord(input) {
  const record = normalizePayload(input, { forCreate: true });
  const complete = { ...record, summary: buildSummary(record) };
  // Save under exactly the cap reopen enforces, in the same units, so a record
  // can never be written that this viewer would then refuse to open.
  const bytes = utf8Bytes(JSON.stringify(complete));
  if (bytes > MAX_INPUT_CHARS) {
    fail(
      `this review is ${bytes} bytes, over the ${MAX_INPUT_CHARS}-byte limit a saved record must fit; `
      + "clear some painted ROIs or generated layers, then save again",
    );
  }
  return complete;
}

/**
 * Parse and strictly validate a downloaded review JSON document.
 * Schema 1 and 2 documents are accepted and returned upgraded to the current schema.
 */
export function parseReviewRecord(text) {
  if (typeof text !== "string" || text.length === 0) fail("input must be a non-empty JSON document");
  const bytes = utf8Bytes(text);
  if (bytes > MAX_INPUT_CHARS) {
    fail(`input must be a JSON document no larger than ${MAX_INPUT_CHARS} bytes`);
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    fail("input is not valid JSON");
  }
  // Schema 1 keeps the cap it was written under. Upgrading a document must not
  // retroactively widen what schema 1 was ever allowed to be.
  if (isPlainObject(parsed) && parsed.schemaVersion === 1 && bytes > MAX_V1_INPUT_CHARS) {
    fail(`a schema 1 document must be no larger than ${MAX_V1_INPUT_CHARS} bytes`);
  }
  return upgradeReviewRecord(parsed);
}

/** Assert that a review belongs to the current case/grid/volume/recipe context. */
export function assertReviewContext(record, currentContext) {
  const saved = normalizeRecord(record, { verifySummary: true });
  const current = normalizeContext(currentContext, "currentContext", { strict: false });
  const labels = {
    caseId: "case",
    gridId: "grid",
    volumeId: "volume",
    recipeHash: "recipe",
    caseSourceHash: "case source hash",
  };
  for (const [field, label] of Object.entries(labels)) {
    if (saved.context[field] !== current[field]) {
      fail(`cannot restore because the saved ${label} does not match the current ${label}`);
    }
  }
  if (saved.context.buildId && current.buildId && saved.context.buildId !== current.buildId) {
    fail("cannot restore because the saved build identity does not match the current build identity");
  }
  return record;
}

/** Assert that every saved named bank has the exact currently served source. */
export function assertReviewSources(record, currentBanks) {
  const saved = normalizeRecord(record, { verifySummary: true });
  if (!Array.isArray(currentBanks) || currentBanks.length !== saved.banks.length) {
    fail("cannot restore because the current named-bank set does not match the saved record");
  }
  const currentById = new Map();
  for (const [index, value] of currentBanks.entries()) {
    const name = `currentBanks[${index}]`;
    assertPlainObject(value, name);
    const bankId = normalizeIdentifier(requireOwn(value, "bankId", name), `${name}.bankId`);
    const sourcePopulation = normalizeIdentifier(
      requireOwn(value, "sourcePopulation", name),
      `${name}.sourcePopulation`,
    );
    if (sourcePopulation !== `bank:${bankId}`) {
      fail(`${name}.sourcePopulation is not a restorable named bank source`);
    }
    if (currentById.has(bankId)) fail(`currentBanks contains duplicate bankId "${bankId}"`);
    currentById.set(bankId, {
      sourcePopulation,
      sourceHash: normalizeHash(requireOwn(value, "sourceHash", name), `${name}.sourceHash`),
    });
  }
  for (const bank of saved.banks) {
    const current = currentById.get(bank.bankId);
    if (!current) fail(`cannot restore because named bank "${bank.bankId}" is not currently loaded`);
    if (current.sourcePopulation !== bank.sourcePopulation) {
      fail(`cannot restore because named bank "${bank.bankId}" has a different source population`);
    }
    if (current.sourceHash !== bank.sourceHash) {
      fail(`cannot restore because named bank "${bank.bankId}" has a different source hash`);
    }
  }
  return record;
}

/** Build the deterministic plain-text download summary for a validated record. */
export function reviewSummary(record) {
  return buildSummary(normalizeRecord(record));
}

/** The current record schema. */
export const REVIEW_SCHEMA_VERSION = SCHEMA_VERSION;

/** Byte-ish caps a caller may want to state in its own copy. */
export const REVIEW_LIMITS = Object.freeze({
  maxRecordBytes: MAX_INPUT_CHARS,
  maxPaintChars: MAX_PAINT_CHARS,
  maxGenerated: MAX_GENERATED,
});

/** Route facts for a saved generated layer: method, and whether it can be re-issued. */
export function generatedRoute(route) {
  const entry = GENERATED_ROUTES.get(route);
  if (!entry) fail(`"${route}" is not a route that produces a restorable layer`);
  return { route, ...entry };
}

/**
 * Compare a saved generated layer against what a re-issued request returned.
 *
 * Pure: the caller performs the fetch and passes the observed identity. A layer
 * is restorable only when the route is reproducible, the rebuilt request is
 * byte-identical to the saved one, and every identity field the server returned
 * matches. Anything else is reported, never drawn.
 */
export function matchGeneratedIdentity(saved, observed, { rebuiltRequest } = {}) {
  const entry = normalizeGenerated(saved, 0);
  const refuse = (reason) => ({ layerKey: entry.layerKey, label: entry.tractLabel, restored: false, reason });
  // A caller that forgets the rebuilt request must not get a pass: the
  // comparison is the whole point, so its absence is a programming error.
  if (rebuiltRequest === undefined) fail("a rebuilt request is required to match a generated layer");
  const route = GENERATED_ROUTES.get(entry.request.route);
  // Order matters for the reason a person reads: a route that can never be
  // re-issued says so, rather than reporting a digest that is moot.
  if (!route.reproducible) {
    return refuse(`not restored, ${route.why}; the request is kept as provenance`);
  }
  if (requestDigest(rebuiltRequest) !== entry.request.digest) {
    return refuse("not restored, the rebuilt request differs from the saved request");
  }
  if (!isPlainObject(observed)) fail("observed identity must be an object");
  if (observed.sourcePopulation !== entry.identity.sourcePopulation) {
    return refuse("not restored, the server returned a different source population");
  }
  if (observed.engineDigest !== entry.identity.engineDigest) {
    return refuse("not restored, the server returned a different engine");
  }
  for (const [field, label] of [["nReturned", "streamline count"], ["lineCount", "displayed count"]]) {
    const expected = entry.identity[field];
    if (!Number.isSafeInteger(observed[field])) {
      return refuse(`not restored, the server did not report a ${label}`);
    }
    if (observed[field] !== expected) {
      return refuse(`not restored, the ${label} changed from ${expected} to ${observed[field]}`);
    }
  }
  const observedHash = observed.bankSourceHash == null ? null : String(observed.bankSourceHash).toLowerCase();
  if (observedHash !== entry.identity.bankSourceHash) {
    return refuse(
      entry.identity.bankSourceHash === null
        ? "not restored, the server reported a bank source for a layer that has none"
        : "not restored, the bank source hash changed",
    );
  }
  const observedEdgeHash = observed.edgeSourceHash == null ? null : String(observed.edgeSourceHash).toLowerCase();
  if (observedEdgeHash !== entry.identity.edgeSourceHash) {
    return refuse(
      entry.identity.edgeSourceHash === null
        ? "not restored, the server reported an edge source for a layer that has none"
        : "not restored, the edge source hash changed",
    );
  }
  const observedCorpusHash = observed.corpusSourceHash == null ? null : String(observed.corpusSourceHash).toLowerCase();
  if (observedCorpusHash !== entry.identity.corpusSourceHash) {
    return refuse(
      entry.identity.corpusSourceHash === null
        ? "not restored, the server reported a corpus source for a layer that has none"
        : "not restored, the corpus source hash changed",
    );
  }
  const observedParcHash = observed.parcellationSourceHash == null
    ? null : String(observed.parcellationSourceHash).toLowerCase();
  if (observedParcHash !== entry.identity.parcellationSourceHash) {
    return refuse(
      entry.identity.parcellationSourceHash === null
        ? "not restored, the server reported a parcellation source for a layer that has none"
        : "not restored, the parcellation source hash changed",
    );
  }
  // Corpus and parcellation bytes can stay identical across a rebuild (same
  // inputs, re-run); the generation id is what catches that "same corpus,
  // different assignment lineage" case, since a new build publishes a new
  // gen_id even when nothing it read actually changed.
  const observedGenerationId = observed.generationId == null ? null : String(observed.generationId);
  if (observedGenerationId !== entry.identity.generationId) {
    return refuse(
      entry.identity.generationId === null
        ? "not restored, the server reported a generation id for a layer that has none"
        : "not restored, the connectome generation changed",
    );
  }
  return { layerKey: entry.layerKey, label: entry.tractLabel, restored: true, reason: null };
}

/**
 * One short line per restored or refused object, for the status panel.
 * Copy stays terse and free of middle-dot chains.
 */
export function restoreStatusLines(statuses) {
  if (!Array.isArray(statuses)) fail("restore statuses must be a list");
  return statuses.map((status, index) => {
    const name = `statuses[${index}]`;
    assertPlainObject(status, name);
    const label = normalizeText(requireOwn(status, "label", name), `${name}.label`, { max: 160 });
    const restored = boolean(requireOwn(status, "restored", name), `${name}.restored`);
    if (restored) return `${label}: restored`;
    const reason = normalizeText(requireOwn(status, "reason", name), `${name}.reason`, { max: 400 });
    return `${label}: ${reason}`;
  });
}
