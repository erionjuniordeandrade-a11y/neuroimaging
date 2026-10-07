/**
 * Along-tract profile panel — renders the served `/api/profile/<bank>` payload.
 *
 * Every number on screen is a served field. Nothing here recomputes a median,
 * a percentile, a distance or a count; the only arithmetic is the pixel
 * mapping of served values onto the SVG and the node-to-tube-point mapping the
 * interval highlight needs. The claim string is shown verbatim.
 *
 * Colour is reserved for evidence (CONTEXT.md): the band and line carry the
 * tract's own hue from viewer/tract_colour.js and nothing else in the panel is
 * hued. Out-of-interval tube vertices fall to the ghost grey the ghosted
 * supported trunk already uses (viewer/veil.js), never to a new palette.
 */

import { DIST_MAX, FAR_FADE_START_MM, rgbToHex } from "./tract_colour.js";
import { sideFromIdToken } from "./tract.js";

/** Plot geometry, in SVG user units. Exported so the drag handler and the
 * renderer share one mapping instead of two that can drift. */
export const CHART = Object.freeze({
  width: 300, height: 112, left: 30, right: 296, top: 8, bottom: 86,
});
export const TRACK = Object.freeze({
  width: 300, height: 44, left: 30, right: 296, top: 8, bottom: 32,
});

export const SUPPORTED_SCALARS = Object.freeze(["fa", "md"]);

/** The lesion-distance track shades the same two thresholds the proximity
 * ramp uses; both come from tract_colour.js, never from a new constant. */
export const NEAR_LESION_MM = FAR_FADE_START_MM;
export const TRACK_MAX_MM = DIST_MAX;

/**
 * Display axis for a scalar. The MD domain is a fixed display window in
 * 10^-3 mm^2/s, declared in the axis label — it is not derived from the data
 * and makes no claim about the tissue.
 */
export function scalarAxis(scalar) {
  if (String(scalar).toLowerCase() === "md") {
    return {
      scalar: "md",
      label: "MD (10⁻³ mm²/s)",
      max: 3,
      scale: 1000,
      ticks: [0, 1.5, 3],
      format: (value) => value.toFixed(1),
    };
  }
  return {
    scalar: "fa",
    label: "FA",
    max: 1,
    scale: 1,
    ticks: [0, 0.5, 1],
    format: (value) => value.toFixed(1),
  };
}

/**
 * The short ROI name an orientation record's ROI file carries, e.g.
 * "tracts/roi/fat_r_sfg_dil1.nii.gz" -> "sfg". Returns null when the record
 * declares no ROI file, so the caller can fall back rather than invent one.
 */
export function roiShortName(orientation) {
  const path = orientation?.roi_path;
  if (typeof path !== "string" || !path.trim()) return null;
  let stem = path.split("/").pop().replace(/\.nii(\.gz)?$/i, "");
  stem = stem.replace(/_dil\d+$/i, "");
  const parts = stem.split("_").filter(Boolean);
  const family = String(orientation?.family || "").toLowerCase();
  const side = String(orientation?.side || "").toLowerCase();
  if (parts.length && parts[0].toLowerCase() === family) parts.shift();
  if (parts.length && parts[0].toLowerCase() === side) parts.shift();
  return parts.length ? parts.join(" ") : stem;
}

/**
 * The two end labels the x axis carries, derived from the served orientation
 * rule. Node 0 is always the end nearer the served anchor (profile.py
 * orient_streamlines), so the start label names that end.
 */
export function endLabels(orientation) {
  const rule = String(orientation?.rule || "");
  if (rule === "inferior") return { start: "inferior", end: "superior" };
  if (rule === "anterior") return { start: "anterior", end: "posterior" };
  if (rule === "roi_centroid") {
    const roi = roiShortName(orientation);
    return { start: roi ? `ROI: ${roi}` : "ROI", end: "far end" };
  }
  return { start: "node 0", end: "node 99" };
}

/**
 * Contiguous runs of drawable nodes. A node drops out of the band when the
 * server reported no usable value there, or when it carries the served
 * zero-or-missing flag — the flag is drawn as a gap plus a marker rather than
 * smoothed over, so a reader cannot mistake it for measured tissue.
 */
export function bandSegments(nodes) {
  const median = nodes?.median || [];
  const p25 = nodes?.p25 || [];
  const p75 = nodes?.p75 || [];
  const flags = nodes?.zero_or_missing_flag || [];
  const segments = [];
  let open = null;
  for (let i = 0; i < median.length; i++) {
    const usable = Number.isFinite(median[i]) && Number.isFinite(p25[i])
      && Number.isFinite(p75[i]) && !flags[i];
    if (usable) {
      if (!open) { open = { from: i, to: i }; segments.push(open); }
      else open.to = i;
    } else {
      open = null;
    }
  }
  return segments;
}

/** Node indices the server flagged as zero-or-missing. */
export function flaggedNodes(nodes) {
  const flags = nodes?.zero_or_missing_flag || [];
  const out = [];
  for (let i = 0; i < flags.length; i++) if (flags[i]) out.push(i);
  return out;
}

/** One honest line per served refusal code. Never a diagnosis, never a retry claim. */
export function profileErrorLine(code, scalar = "fa") {
  const name = String(scalar).toLowerCase() === "md" ? "MD" : "FA";
  switch (String(code)) {
    case "scalar_missing": return `${name} map not available for this case`;
    case "scalar_unsupported": return `${name} is not a supported profile scalar`;
    case "orientation_unknown_family": return "No orientation rule for this tract family";
    case "invalid_bank_id": return "This tract id carries no readable side token";
    case "unknown_bank": return "This tract is not a named bank in this case";
    case "bank_source_changed":
    case "hash_mismatch": return "The tract file changed on disk, reload the case";
    case "lesion_invalid": return "The declared lesion mask could not be read";
    case "bank_mismatch": return "The server answered for a different tract";
    case "scalar_mismatch": return "The server answered for a different scalar";
    case "derivation_mismatch": return "This profile belongs to another derivation of the case";
    case "provenance_missing": return "This profile arrived without its full provenance";
    case "nodes_malformed": return "This profile arrived incomplete";
    case "measurements_invalid": return "This profile arrived with unusable measurements";
    case "claim_invalid": return "This profile did not carry the served claim";
    case "claim_forbidden": return "This profile claims validation, which this instrument never asserts";
    case "derivation_unknown": return "Derivation status unavailable";
    case "orientation_mismatch": return "The profile orientation does not match this tract's identity";
    case "bank_source_unknown": return "This tract carries no source digest to check";
    case "payload_invalid": return "The server sent no readable profile";
    case "request_failed": return "The profile request did not complete";
    default: return "Profile unavailable for this tract";
  }
}

/** Clamp a raw drag pair onto whole node indices inside the served range. */
export function normalizeInterval(a, b, nPoints) {
  const last = Math.max(0, Math.round(Number(nPoints)) - 1);
  const lo = Math.min(Math.max(0, Math.round(Number(a))), last);
  const hi = Math.min(Math.max(0, Math.round(Number(b))), last);
  return lo <= hi ? [lo, hi] : [hi, lo];
}

/**
 * The tube-point indices a node interval covers.
 *
 * Both the profile (n_points nodes) and the baked tube (k points per line) are
 * arc-length resamplings of the same streamline, so fractional position along
 * the line is the shared coordinate. When the interval is narrower than one
 * tube-point step the nearest single point is returned, so a selection is
 * never silently empty.
 */
export function pointRangeForInterval(k, nPoints, interval) {
  const points = Math.round(Number(k));
  const nodes = Math.round(Number(nPoints));
  if (!Number.isFinite(points) || points < 2 || !Number.isFinite(nodes) || nodes < 2) return null;
  if (!Array.isArray(interval) || interval.length !== 2) return null;
  const [a, b] = normalizeInterval(interval[0], interval[1], nodes);
  const toPoint = (node) => (node / (nodes - 1)) * (points - 1);
  let lo = Math.ceil(toPoint(a) - 1e-9);
  let hi = Math.floor(toPoint(b) + 1e-9);
  if (lo > hi) {
    const centre = Math.round(toPoint((a + b) / 2));
    lo = hi = Math.min(points - 1, Math.max(0, centre));
  }
  return { lo: Math.max(0, lo), hi: Math.min(points - 1, hi) };
}

/**
 * True when this streamline runs against the profile's node order: node 0 is
 * the endpoint nearer the served anchor (profile.py orient_streamlines), so a
 * line whose LAST packed point is nearer the anchor is reversed relative to it.
 */
export function isStreamlineFlipped(pts, lineIndex, k, anchorMm) {
  if (!pts || !anchorMm || anchorMm.length !== 3) return false;
  const base = lineIndex * k * 3;
  const tail = base + (k - 1) * 3;
  const d = (offset) => Math.hypot(
    pts[offset] - anchorMm[0], pts[offset + 1] - anchorMm[1], pts[offset + 2] - anchorMm[2],
  );
  return d(tail) < d(base);
}

/**
 * Per-tube-point mask for a node interval: 1 keeps the evidence colour, 0
 * falls to the ghost. Length is lineCount * k; the caller expands it across
 * the radial ring, which is why this stays per point and not per vertex.
 */
export function intervalPointMask({ pts, lineCount, k, nPoints, interval, anchorMm }) {
  const range = pointRangeForInterval(k, nPoints, interval);
  const mask = new Uint8Array(Math.max(0, lineCount * k));
  if (!range) return mask;
  const mirroredLo = k - 1 - range.hi;
  const mirroredHi = k - 1 - range.lo;
  for (let line = 0; line < lineCount; line++) {
    const flipped = isStreamlineFlipped(pts, line, k, anchorMm);
    const lo = flipped ? mirroredLo : range.lo;
    const hi = flipped ? mirroredHi : range.hi;
    for (let i = lo; i <= hi; i++) mask[line * k + i] = 1;
  }
  return mask;
}

/** A served sha256: exactly 64 hex characters, nothing shorter. */
export function isSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/i.test(value.trim());
}

/** First 8 characters of a served sha256. Anything else is not a digest. */
export function shortHash(value) {
  return isSha256(value) ? value.trim().slice(0, 8) : null;
}

/** The claim the route is contracted to send, verbatim (profile.py CLAIM). */
export const SERVED_CLAIM = "descriptive profile; not a normative abnormality score";

/** Statistics: a node with no usable sample is null, never a sentinel number. */
const STAT_ARRAYS = ["median", "p25", "p75", "mean"];
/** Counts: always present, always a whole non-negative number. */
const COUNT_ARRAYS = ["n", "n_valid", "n_zero", "n_nan"];
const NODE_ARRAYS = [...STAT_ARRAYS, ...COUNT_ARRAYS];

/** Orientation rules profile.py can declare (resolve_orientation). */
export const ORIENTATION_RULES = new Set(["inferior", "roi_centroid", "anterior"]);

/** Families profile.py will parse a bank id for (_KNOWN_FAMILIES). */
export const ORIENTATION_FAMILIES = new Set([
  "slf1", "slf2", "slf3", "ifof", "cing", "cst", "fat", "uf", "or",
]);

/**
 * The family token of a structured bank id: `bank_<family>_<side>[…]`.
 * Structure only, never label prose (CONTEXT.md laterality rule).
 */
export function familyFromBankId(bankId) {
  const parts = String(bankId || "").toLowerCase().split("_").filter(Boolean);
  if (parts.length < 3 || parts[0] !== "bank") return null;
  return ORIENTATION_FAMILIES.has(parts[1]) ? parts[1] : null;
}

/** 'l' | 'r' | null — the viewer's own token rule (tract.js sideFromIdToken). */
export function sideFromBankId(bankId) {
  const side = sideFromIdToken(bankId);
  return side === "L" || side === "R" ? side.toLowerCase() : null;
}

const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const isCount = (value) => Number.isSafeInteger(value) && value >= 0;

/**
 * Refuse a 200 that is not the profile that was asked for.
 *
 * A wrong bank, a wrong scalar, a different derivation, a missing digest or a
 * short node array is a mismatch, not a profile: the panel must show one
 * honest line rather than draw someone else's numbers under this tract's name.
 * Returns null when the payload is usable, otherwise the refusal code.
 */
export function validateProfilePayload(payload, { bankId, scalar, activeDerivation, sourceHash } = {}) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return "payload_invalid";
  if (payload.bank_id !== bankId) return "bank_mismatch";
  if (String(payload.scalar || "").toLowerCase() !== String(scalar || "").toLowerCase()) return "scalar_mismatch";
  if (!isSha256(payload.bank_sha256) || !isSha256(payload.scalar_sha256)) return "provenance_missing";
  if (typeof payload.bank_path !== "string" || !payload.bank_path.trim()) return "provenance_missing";
  if (typeof payload.scalar_path !== "string" || !payload.scalar_path.trim()) return "provenance_missing";
  if (typeof payload.active_derivation !== "string" || !payload.active_derivation.trim()) return "provenance_missing";
  // No derivation authority, no profile. Skipping the comparison because the
  // viewer failed to learn its own lineage would accept an artifact from any
  // other derivation of this case (CLAUDE.md: never mix derivations).
  if (!activeDerivation) return "derivation_unknown";
  if (payload.active_derivation !== activeDerivation) return "derivation_mismatch";

  const orientation = payload.orientation;
  if (!orientation || typeof orientation.rule !== "string" || !orientation.rule) return "provenance_missing";
  if (!Array.isArray(orientation.anchor_mm) || orientation.anchor_mm.length !== 3
    || !orientation.anchor_mm.every(isFiniteNumber)) return "provenance_missing";
  // A declared ROI or lesion must arrive with its digest; half a provenance
  // record is worse than none because it reads as complete.
  if (orientation.roi_path && !isSha256(orientation.roi_sha256)) return "provenance_missing";
  if (payload.lesion_path && !isSha256(payload.lesion_sha256)) return "provenance_missing";
  // Identity: the family and side the server oriented by must be the ones this
  // tract's structured id declares. Laterality comes from the id token only.
  if (!ORIENTATION_RULES.has(orientation.rule)) return "orientation_mismatch";
  const family = familyFromBankId(bankId);
  const side = sideFromBankId(bankId);
  if (!family || !side) return "orientation_mismatch";
  if (String(orientation.family || "").toLowerCase() !== family) return "orientation_mismatch";
  if (String(orientation.side || "").toLowerCase() !== side) return "orientation_mismatch";

  const nPoints = Number(payload.n_points);
  if (!Number.isSafeInteger(nPoints) || nPoints < 2) return "nodes_malformed";
  const nodes = payload.nodes;
  if (!nodes || typeof nodes !== "object") return "nodes_malformed";
  for (const name of NODE_ARRAYS) {
    if (!Array.isArray(nodes[name]) || nodes[name].length !== nPoints) return "nodes_malformed";
  }
  if (!Array.isArray(nodes.zero_or_missing_flag) || nodes.zero_or_missing_flag.length !== nPoints) {
    return "nodes_malformed";
  }
  // Values, not just shapes. A node with no usable sample is null (profile.py
  // aggregate_nodes); anything else must be a real number, and JSON can carry
  // Infinity through 1e999, so "is a number" is not enough.
  for (const name of STAT_ARRAYS) {
    for (const value of nodes[name]) {
      if (value !== null && !isFiniteNumber(value)) return "measurements_invalid";
    }
  }
  for (const name of COUNT_ARRAYS) {
    for (const value of nodes[name]) if (!isCount(value)) return "measurements_invalid";
  }
  for (const value of nodes.zero_or_missing_flag) {
    if (typeof value !== "boolean") return "measurements_invalid";
  }
  const track = payload.lesion_distance?.track_mm;
  if (track != null) {
    if (!Array.isArray(track) || track.length !== nPoints) return "nodes_malformed";
    // A distance is never negative and never missing per node: the whole
    // track is null, or every node has one.
    for (const value of track) if (!isFiniteNumber(value) || value < 0) return "measurements_invalid";
  }

  // The claim is contract text, not prose the panel paraphrases. Pilot
  // language is owner law: a payload asserting validation is refused outright.
  if (typeof payload.claim !== "string") return "claim_invalid";
  if (/\bvalidated\b/i.test(payload.claim)) return "claim_forbidden";
  if (payload.claim !== SERVED_CLAIM) return "claim_invalid";

  // The bank file this layer was drawn from must still be the one the server
  // hashed at boot. The digest domains differ (see the note in index.html), so
  // this compares the viewer's own recorded header across the request.
  if (!sourceHash?.expected) return "bank_source_unknown";
  if (sourceHash.expected !== sourceHash.current) return "bank_source_changed";
  return null;
}

/** The node where the served lesion track is at its minimum. Served values only. */
export function closestApproach(payload) {
  const track = payload?.lesion_distance?.track_mm;
  if (!Array.isArray(track) || !track.length) return null;
  let node = -1;
  let mm = Infinity;
  for (let i = 0; i < track.length; i++) {
    const value = Number(track[i]);
    if (!Number.isFinite(value) || value >= mm) continue;
    mm = value;
    node = i;
  }
  return node < 0 ? null : { node, mm };
}

/** "closest 0.6 mm, node 75" — both numbers straight off the served track. */
export function closestApproachLabel(payload) {
  const closest = closestApproach(payload);
  if (!closest) return null;
  const mm = closest.mm >= 10 ? closest.mm.toFixed(0) : closest.mm.toFixed(1);
  return `closest ${mm} mm, node ${closest.node}`;
}

/**
 * The served provenance the "i" card shows. Served values only, and only for
 * a payload that already passed validateProfilePayload — a row is never
 * rendered as "unrecorded", because a payload missing any of these is refused
 * before it reaches the panel.
 */
export function provenanceRows(payload) {
  const orientation = payload?.orientation || {};
  const rows = [
    ["bank file", String(payload?.bank_path || "")],
    ["bank sha256", shortHash(payload?.bank_sha256)],
    ["scalar file", String(payload?.scalar_path || "")],
    ["scalar sha256", shortHash(payload?.scalar_sha256)],
    ["orientation rule", String(orientation.rule || "")],
  ];
  if (orientation.roi_path) {
    rows.push(["orientation ROI", String(orientation.roi_path)]);
    rows.push(["ROI sha256", shortHash(orientation.roi_sha256)]);
  }
  rows.push(["streamlines flipped", String(orientation.n_flipped ?? "")]);
  if (payload?.lesion_path) {
    rows.push(["lesion mask", String(payload.lesion_path)]);
    rows.push(["lesion sha256", shortHash(payload.lesion_sha256)]);
    rows.push(["near band", `under ${NEAR_LESION_MM} mm, track axis to ${TRACK_MAX_MM} mm`]);
  }
  rows.push(["derivation", String(payload?.active_derivation || "")]);
  rows.push(["resampling", String(payload?.resample_method || "")]);
  return rows.filter(([, value]) => value !== null && value !== "");
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value != null) el.setAttribute(name, String(value));
  }
  return el;
}

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

const nodeX = (index, nPoints, box) =>
  box.left + (index / Math.max(1, nPoints - 1)) * (box.right - box.left);

/** Map a pointer x (already in SVG user units) back to a node index. */
export function xToNode(x, nPoints, box = CHART) {
  const span = box.right - box.left;
  const t = span > 0 ? (x - box.left) / span : 0;
  return Math.min(nPoints - 1, Math.max(0, Math.round(t * (nPoints - 1))));
}

function buildChart(payload, axis, hue, interval, closest) {
  const nodes = payload.nodes || {};
  const nPoints = Number(payload.n_points) || (nodes.median || []).length;
  const root = svg("svg", {
    id: "profileChart", class: "profile-chart", viewBox: `0 0 ${CHART.width} ${CHART.height}`,
    preserveAspectRatio: "none", role: "img",
    "aria-label": `${axis.label} along the bundle, node 0 to node ${nPoints - 1}`,
    "data-nodes": nPoints,
  });
  const valueY = (value) => {
    const scaled = Number(value) * axis.scale;
    const t = Math.min(1, Math.max(0, scaled / axis.max));
    return CHART.bottom - t * (CHART.bottom - CHART.top);
  };

  for (const tick of axis.ticks) {
    const y = CHART.bottom - (tick / axis.max) * (CHART.bottom - CHART.top);
    root.append(svg("line", {
      class: "profile-grid", x1: CHART.left, x2: CHART.right, y1: y, y2: y,
    }));
    const label = svg("text", {
      class: "profile-tick", x: CHART.left - 4, y: y + 3, "text-anchor": "end",
    });
    label.textContent = axis.format(tick);
    root.append(label);
  }

  if (interval) {
    const x1 = nodeX(interval[0], nPoints, CHART);
    const x2 = nodeX(interval[1], nPoints, CHART);
    root.append(svg("rect", {
      id: "profileSelection", class: "profile-selection", x: Math.min(x1, x2),
      y: CHART.top, width: Math.max(1, Math.abs(x2 - x1)), height: CHART.bottom - CHART.top,
    }));
  }

  const hex = rgbToHex(hue);
  for (const segment of bandSegments(nodes)) {
    const upper = [];
    const lower = [];
    for (let i = segment.from; i <= segment.to; i++) {
      upper.push(`${nodeX(i, nPoints, CHART).toFixed(2)},${valueY(nodes.p75[i]).toFixed(2)}`);
      lower.unshift(`${nodeX(i, nPoints, CHART).toFixed(2)},${valueY(nodes.p25[i]).toFixed(2)}`);
    }
    root.append(svg("polygon", {
      class: "profile-band", points: [...upper, ...lower].join(" "), fill: hex,
    }));
  }
  for (const segment of bandSegments(nodes)) {
    const line = [];
    for (let i = segment.from; i <= segment.to; i++) {
      line.push(`${nodeX(i, nPoints, CHART).toFixed(2)},${valueY(nodes.median[i]).toFixed(2)}`);
    }
    root.append(svg("polyline", {
      class: "profile-median", points: line.join(" "), stroke: hex, fill: "none",
    }));
  }
  for (const index of flaggedNodes(nodes)) {
    root.append(svg("rect", {
      class: "profile-gap", x: nodeX(index, nPoints, CHART) - 1, y: CHART.bottom - 5,
      width: 2, height: 5,
    }));
  }
  root.append(svg("line", {
    class: "profile-axis-line", x1: CHART.left, x2: CHART.right,
    y1: CHART.bottom, y2: CHART.bottom,
  }));
  // The closest-approach node ties the two rows together: one instrument, the
  // same x, read once off the served lesion track.
  if (closest) {
    const x = nodeX(closest.node, nPoints, CHART);
    root.append(svg("line", {
      id: "profileClosestTick", class: "profile-closest-tick",
      x1: x, x2: x, y1: CHART.top, y2: CHART.bottom,
    }));
  }
  return root;
}

function buildTrack(payload) {
  const track = payload?.lesion_distance?.track_mm;
  if (!Array.isArray(track) || !track.length) return null;
  const nPoints = track.length;
  const root = svg("svg", {
    id: "profileLesionTrack", class: "profile-track", viewBox: `0 0 ${TRACK.width} ${TRACK.height}`,
    preserveAspectRatio: "none", role: "img",
    "aria-label": "Distance from each node to the lesion surface, millimetres",
    "data-nodes": nPoints,
  });
  const mmY = (mm) => {
    const t = Math.min(1, Math.max(0, Number(mm) / TRACK_MAX_MM));
    return TRACK.bottom - t * (TRACK.bottom - TRACK.top);
  };
  const nearY = mmY(NEAR_LESION_MM);
  root.append(svg("rect", {
    class: "profile-near-band", x: TRACK.left, y: nearY,
    width: TRACK.right - TRACK.left, height: TRACK.bottom - nearY,
  }));
  root.append(svg("rect", {
    class: "profile-far-band", x: TRACK.left, y: mmY(TRACK_MAX_MM),
    width: TRACK.right - TRACK.left, height: nearY - mmY(TRACK_MAX_MM),
  }));
  const line = track.map((mm, index) =>
    `${nodeX(index, nPoints, TRACK).toFixed(2)},${mmY(mm).toFixed(2)}`);
  root.append(svg("polyline", { class: "profile-track-line", points: line.join(" "), fill: "none" }));
  const nearLabel = svg("text", {
    class: "profile-tick", x: TRACK.left - 4, y: nearY + 3, "text-anchor": "end",
  });
  nearLabel.textContent = String(NEAR_LESION_MM);
  root.append(nearLabel);
  const closest = closestApproach(payload);
  if (closest) {
    const x = nodeX(closest.node, nPoints, TRACK);
    const y = mmY(closest.mm);
    root.append(svg("circle", {
      id: "profileClosestMarker", class: "profile-closest-marker", cx: x, cy: y, r: 2.4,
    }));
    // Above the plot area, never on it: the track line runs between
    // TRACK.top and TRACK.bottom, so this band is always clear. The halo
    // (paint-order stroke, .profile-closest-label) keeps it readable if a
    // future layout moves it back over the line.
    const text = svg("text", {
      id: "profileClosestLabel", class: "profile-closest-label",
      x: Math.min(TRACK.right, Math.max(TRACK.left, x)),
      y: TRACK.top - 2,
      "text-anchor": x > (TRACK.left + TRACK.right) / 2 ? "end" : "start",
    });
    text.textContent = closestApproachLabel(payload);
    root.append(text);
  }
  return root;
}

/**
 * Mount or replace the profile panel inside `host`.
 *
 * `state.status` is one of loading | unsupported | error | ready. The panel is
 * always present for a focused layer so the reader never has to guess whether
 * it is missing or still working; what changes is the one honest line inside.
 */
export function renderProfilePanel(host, state = {}) {
  if (!host) return null;
  host.replaceChildren();
  // No focused layer at all: the panel is absent, not empty. A panel that
  // stays behind after the last tract is removed would read as stale evidence.
  if (state.status === "absent") return null;
  const panel = element("div", "section profile-panel");
  panel.id = "profilePanel";
  panel.setAttribute("aria-label", "Along-tract scalar profile");
  // No heading here: the dock's own <summary> already names this panel
  // (design pass — the title was printed twice).

  if (state.status === "unsupported") {
    panel.append(element("p", "section-hint", "Profile needs a named tract"));
    host.append(panel);
    return panel;
  }

  const scalars = element("div", "profile-scalars");
  scalars.setAttribute("role", "group");
  scalars.setAttribute("aria-label", "Profile scalar");
  for (const scalar of SUPPORTED_SCALARS) {
    const button = element("button", "profile-scalar", scalar.toUpperCase());
    button.type = "button";
    button.id = `profileScalar_${scalar}`;
    button.dataset.scalar = scalar;
    button.setAttribute("aria-pressed", String(scalar === state.scalar));
    scalars.append(button);
  }
  panel.append(scalars);

  if (state.status === "loading") {
    const skeleton = element("div", "profile-skeleton");
    skeleton.id = "profileSkeleton";
    skeleton.setAttribute("role", "status");
    skeleton.textContent = "Reading profile…";
    panel.append(skeleton);
    host.append(panel);
    return panel;
  }

  if (state.status === "error") {
    const line = element("p", "section-hint profile-error", profileErrorLine(state.code, state.scalar));
    line.id = "profileError";
    panel.append(line);
    host.append(panel);
    return panel;
  }

  const payload = state.payload || {};
  const axis = scalarAxis(payload.scalar || state.scalar);
  const nPoints = Number(payload.n_points) || 0;
  const closest = closestApproach(payload);

  // The row above the chart: the scalar names the y axis on the left, the
  // muted hint (or the live selection) sits on the right. The scalar label
  // used to be drawn inside the SVG at the top-left, where it sat on top of
  // the topmost y tick; out here it cannot collide with a tick at all.
  const head = element("div", "profile-chart-head");
  const name = element("span", "profile-scalar-name", axis.label);
  name.id = "profileScalarName";
  const hint = element("p", "profile-hint");
  hint.id = "profileHint";
  if (state.interval) {
    hint.textContent = `Nodes ${state.interval[0]} to ${state.interval[1]}, Escape clears`;
    hint.classList.add("selected");
    hint.id = "profileIntervalLine";
  } else {
    hint.textContent = "Drag to highlight";
  }
  head.append(name, hint);
  panel.append(head);

  panel.append(buildChart(payload, axis, state.hue || [0.7, 0.82, 1], state.interval || null, closest));

  const axisRow = element("div", "profile-axis");
  const labels = endLabels(payload.orientation);
  axisRow.append(
    element("span", "profile-axis-start", labels.start),
    element("span", "profile-axis-end", labels.end),
  );
  panel.append(axisRow);

  const track = buildTrack(payload);
  if (track) {
    panel.append(element("p", "profile-row-label", "Distance to lesion (mm)"));
    panel.append(track);
  } else {
    panel.append(element("p", "section-hint profile-no-lesion", "No lesion mask in this case"));
  }

  if (payload.zero_or_missing_warning) {
    const warning = element("p", "profile-warning", String(payload.zero_or_missing_warning));
    warning.id = "profileWarning";
    panel.append(warning);
  }

  const chips = element("div", "profile-chips");
  chips.id = "profileChips";
  const chip = (text, id) => {
    const span = element("span", "prov-chip", text);
    span.dataset.prov = "profile";
    span.dataset.state = "mute";
    if (id) span.id = id;
    chips.append(span);
  };
  // The scalar itself is already named by the pressed toggle, the y axis and
  // the scalar-digest chip; a fourth copy would be noise.
  chip(`${Number(payload.n_streamlines || 0).toLocaleString("en-US")} streamlines`, "profileChipCount");
  chip(`derivation ${payload.active_derivation}`, "profileChipDerivation");
  // The two digests the profile rests on, visible without opening the card.
  chip(`bank ${shortHash(payload.bank_sha256)}`, "profileChipBankHash");
  chip(`${String(payload.scalar).toLowerCase()} ${shortHash(payload.scalar_sha256)}`, "profileChipScalarHash");
  const info = element("button", "prov-chip", "i");
  info.type = "button";
  info.id = "profileChipInfo";
  info.dataset.prov = "profile";
  info.dataset.state = "mute";
  info.setAttribute("aria-expanded", "false");
  info.setAttribute("aria-label", "Show profile provenance");
  chips.append(info);
  panel.append(chips);

  const card = element("div", "profile-prov-card");
  card.id = "profileProvCard";
  card.hidden = true;
  card.textContent = provenanceRows(payload).map(([name, value]) => `${name}: ${value}`).join("\n");
  panel.append(card);

  const claim = element("p", "profile-claim", String(payload.claim || ""));
  claim.id = "profileClaim";
  panel.append(claim);

  panel.dataset.nodes = String(nPoints);
  host.append(panel);
  return panel;
}
