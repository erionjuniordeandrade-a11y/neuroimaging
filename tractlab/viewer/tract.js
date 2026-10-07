/**
 * Tract descriptor — single authority for laterality and chip identity.
 *
 * Domain:
 *   tract  — what the user sees (streamlines + identity)
 *   bank   — stored multi-ROI extract (source of a tract)
 *   live   — Commit / filter re-track (other source)
 *
 * Laterality authority: bank / seed-preset **id token only**.
 * Prose labels are display-only and never consulted for side.
 *
 * Unknown side is always '?' — never a defaulted hemisphere.
 */

/** @typedef {'L'|'R'|'?'} Side */
/** @typedef {'cst'|'fat'|'slf'|'ifof'|'uf'|'cing'|'cc'|'or'|'other'} TractGroup */

/**
 * @typedef {object} TractDescriptor
 * @property {Side} side
 * @property {string} short   — HUD / status stem (e.g. SLF-III-L)
 * @property {string} chip    — button text (may include n≈)
 * @property {TractGroup} group
 * @property {string} role    — structured role from bank record (e.g. true_slf3)
 * @property {string} id      — bank id or seed preset id
 * @property {'bank'|'live'} source
 */

/**
 * Side from a structured id (bank_slf3_l, seed_handknob_r, …).
 * Standalone path segments only — never substring match inside words.
 * @param {string} id
 * @returns {Side}
 */
export function sideFromIdToken(id) {
  const parts = String(id || "")
    .toLowerCase()
    .split(/[_-]+/)
    .filter(Boolean);
  const sides = new Set();
  for (const p of parts) {
    if (p === "l") sides.add("L");
    else if (p === "r") sides.add("R");
  }
  // Conflicting structured tokens are malformed metadata, not permission to
  // choose whichever hemisphere happened to occur last.
  return sides.size === 1 ? [...sides][0] : "?";
}

/**
 * Side token from a .tck path basename (conformance / diagnostics).
 * @param {string} path
 * @returns {Side}
 */
export function sideFromPathToken(path) {
  const base = String(path || "").split(/[/\\]/).pop() || "";
  const stem = base.replace(/\.tck$/i, "");
  return sideFromIdToken(stem);
}

/**
 * @param {string} id
 * @param {Side} side
 * @returns {string}
 */
function shortFromId(id, side) {
  const s = String(id || "");
  const soft = /soft/i.test(s);
  const softBit = soft ? " soft" : "";
  if (/strict/i.test(s) && /cst/i.test(s)) {
    return side === "?" ? "CST (strict)" : `CST-${side} (strict)`;
  }
  if (/cst/i.test(s)) return (side === "?" ? "CST" : `CST-${side}`) + softBit;
  if (/fat/i.test(s)) return (side === "?" ? "FAT" : `FAT-${side}`) + softBit;
  // Order matters: match component before bare "slf"
  if (/slf1/i.test(s)) return (side === "?" ? "SLF-I" : `SLF-I-${side}`) + softBit;
  if (/slf2/i.test(s)) return (side === "?" ? "SLF-II" : `SLF-II-${side}`) + softBit;
  if (/slf3/i.test(s)) return (side === "?" ? "SLF-III" : `SLF-III-${side}`) + softBit;
  if (/slf/i.test(s)) return (side === "?" ? "SLF" : `SLF-${side}`) + softBit;
  if (/ifof/i.test(s)) return (side === "?" ? "IFOF" : `IFOF-${side}`) + softBit;
  if (/uf_|uncinate/i.test(s)) return (side === "?" ? "UF" : `UF-${side}`) + softBit;
  if (/cing/i.test(s)) return (side === "?" ? "Cing" : `Cing-${side}`) + softBit;
  if (/or_.*meyer|meyer/i.test(s)) {
    return (side === "?" ? "OR Meyer" : `OR-${side} Meyer`) + softBit;
  }
  // Optic radiation: bank_or_l / bank_or_r (not "forceps" which contains "or")
  if (/(?:^|_)or(?:_|$)/i.test(s) || /optic/i.test(s)) {
    return (side === "?" ? "OR" : `OR-${side}`) + softBit;
  }
  if (/forceps_minor|forceps.minor/i.test(s)) return "Forceps minor" + softBit;
  if (/forceps_major|forceps.major/i.test(s)) return "Forceps major" + softBit;
  if (/cc_body|callos/i.test(s)) return "Callosum" + softBit;
  const cleaned = s.replace(/^bank_/, "").replace(/_/g, " ");
  return cleaned || "tract";
}

/**
 * @param {string} id
 * @param {string} role
 * @returns {TractGroup}
 */
function groupFromId(id, role) {
  const r = String(role || "");
  const s = String(id || "");
  if (r === "true_cst" || r === "soft_cst" || /cst/i.test(s)) return "cst";
  if (r === "true_fat" || r === "soft_fat" || /fat/i.test(s)) return "fat";
  if (
    r === "true_slf1" || r === "true_slf2" || r === "true_slf3" ||
    r === "soft_slf1" || r === "soft_slf2" || r === "soft_slf3" ||
    /slf/i.test(s)
  ) {
    return "slf";
  }
  if (r === "true_ifof" || /ifof/i.test(s)) return "ifof";
  if (r === "true_uf" || /uf_|uncinate/i.test(s)) return "uf";
  if (r === "true_cing" || /cing/i.test(s)) return "cing";
  if (r === "true_or" || /(?:^|_)or(?:_|$)/i.test(s) || /optic/i.test(s)) return "or";
  if (r === "true_cc" || /cc_|callos|forceps/i.test(s)) return "cc";
  return "other";
}

export const TRACT_GROUP_META = Object.freeze({
  cst: { label: "Corticospinal (CST)", order: 0 },
  fat: { label: "Frontal aslant (FAT)", order: 1 },
  slf: { label: "Superior longitudinal (SLF I–III)", order: 2 },
  ifof: { label: "Inferior fronto-occipital (IFOF)", order: 3 },
  uf: { label: "Uncinate fasciculus (UF)", order: 4 },
  cing: { label: "Cingulum", order: 5 },
  or: { label: "Optic radiation (OR)", order: 6 },
  cc: { label: "Callosal fibres", order: 7 },
  other: { label: "Other banks", order: 9 },
});

/**
 * Describe a bank (or bank-like) record. Id is required for identity.
 * Label fields on `rec` are ignored for side / short / group.
 *
 * @param {{ id: string, role?: string, nStreamlines?: number|null, path?: string, label?: string }} rec
 * @returns {Readonly<TractDescriptor>}
 */
export function describeTract(rec) {
  const id = String(rec?.id || "");
  if (!id) {
    return Object.freeze({
      side: "?",
      short: "tract",
      chip: "tract",
      group: "other",
      role: "",
      id: "",
      source: "bank",
    });
  }
  const side = sideFromIdToken(id);
  const role = String(rec?.role || "");
  const short = shortFromId(id, side);
  const n = rec?.nStreamlines;
  const chip =
    n != null && Number.isFinite(Number(n)) ? `${short} · n≈${n}` : short;
  return Object.freeze({
    side,
    short,
    chip,
    group: groupFromId(id, role),
    role,
    id,
    source: "bank",
  });
}

/**
 * Live Commit / filter — side only from seed_preset id tokens, never modeLabel.
 *
 * @param {{ seedPresetId?: string|null }} [opts]
 * @returns {Readonly<TractDescriptor>}
 */
export function describeLiveTrack(opts = {}) {
  const id = String(opts.seedPresetId || "");
  const side = id ? sideFromIdToken(id) : "?";
  const short =
    side === "?" ? "live" : `live ${side}`;
  return Object.freeze({
    side,
    short,
    chip: short,
    group: "other",
    role: "",
    id,
    source: "live",
  });
}

/**
 * Peri-lesional recovery — never a named multi-ROI bundle identity.
 * @param {{ radiusMm?: number, source?: string }} [opts]
 * @returns {Readonly<TractDescriptor>}
 */
export function describeRecoveryTrack(opts = {}) {
  const r = Number(opts.radiusMm);
  const radius = Number.isFinite(r) ? r : 8;
  const src = String(opts.source || "corpus");
  const short = `Recovery ≤${radius} mm`;
  return Object.freeze({
    side: "?",
    short,
    chip: short,
    group: "other",
    role: "recovery_perilesional",
    id: `recovery_perilesional_${src}`,
    source: "live",
  });
}

/**
 * An assigned edge's exact streamlines — a Tract with a parcel-pair label,
 * never a parcel (ADR-0003). "Edge" refers to the assignment, not laterality;
 * side stays "?" here — the structured id token carries no side information
 * for an edge (a parcel pair can span hemispheres).
 *
 * "ASSIGNED" and the build's radial-search radius (when known) are baked
 * directly into `short`/`chip` — the only fields every render site (HUD
 * chip, legend, proximity card) is guaranteed to show — so the ADR-0003
 * reminder and the assignment rule travel with the layer wherever it
 * appears, not only inside the Connections panel that created it.
 * @param {{ a?: number, b?: number, radiusMm?: number }} [opts]
 * @returns {Readonly<TractDescriptor>}
 */
export function describeEdgeTrack(opts = {}) {
  const a = Number(opts.a);
  const b = Number(opts.b);
  const radius = Number(opts.radiusMm);
  const pair = Number.isFinite(a) && Number.isFinite(b) ? `${a}–${b}` : "unknown";
  const short = Number.isFinite(radius)
    ? `ASSIGNED ${pair} (radial search ≤${radius} mm)`
    : `ASSIGNED ${pair}`;
  return Object.freeze({
    side: "?",
    short,
    chip: short,
    group: "other",
    role: "connectome_edge",
    id: Number.isFinite(a) && Number.isFinite(b) ? `edge_${a}_${b}` : "edge_unknown",
    source: "live",
  });
}

/**
 * The identity a saved generated or subset layer is redrawn under when a
 * review is reopened.
 *
 * A recovery result keeps the recovery identity and its visual contract; it is
 * never relabelled as a live track. A bank-backed subset keeps the bank's
 * identity. An assigned-edge tube layer keeps the edge identity. Everything
 * else is a live track.
 *
 * @param {{ route: string, bankId: (string|null), seedPresetId: (string|null),
 *           params: object }} request  the saved request record
 * @param {object|null} [bank]  the served bank record, when the layer is bank-backed
 * @returns {Readonly<TractDescriptor>}
 */
export function describeGeneratedLayer(request, bank = null) {
  if (request.bankId && bank) return describeTract(bank);
  if (request.route === "/api/recovery/perilesional") {
    return describeRecoveryTrack({
      radiusMm: request.params?.radiusMm,
      source: request.params?.source,
    });
  }
  if (request.route === "/api/connectome/edge/tubes") {
    return describeEdgeTrack({ a: request.params?.a, b: request.params?.b, radiusMm: request.params?.radiusMm });
  }
  return describeLiveTrack({ seedPresetId: request.seedPresetId || undefined });
}
