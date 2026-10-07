/**
 * Provenance strip — pure text/state builders (owner ruling 8, DESIGN doc
 * proposals D+E). No DOM here; index.html wires these into #provStrip /
 * #provCard. Every value is a served field, honestly summarised — never a
 * new claim, never an invented number (CLAUDE.md honesty rule).
 */

// The mm figure appears ONLY when it comes from a served, signed delta-QC
// median (derivation.py's _LABEL_RPE_SIGNED, e.g.
// "reverse-PE corrected — median shift 0.6 mm (signed delta QC)"). Any other
// state (uncorrected, or corrected-but-unsigned) never fabricates a number.
const SIGNED_SHIFT_RE = /median shift\s*([\d.]+)\s*mm.*signed delta QC/i;

export function spaceChipText(floorLabel) {
  const label = typeof floorLabel === "string" ? floorLabel : "";
  const m = label.match(SIGNED_SHIFT_RE);
  if (m) {
    const mm = Number(m[1]);
    return `DWI space, ${Number.isFinite(mm) ? mm.toFixed(1) : m[1]} mm shift`;
  }
  if (/reverse-PE corrected/i.test(label)) {
    // Corrected but not (yet) signed: never borrow the uncorrected-case
    // "~3 mm floor" wording (CLAUDE.md: don't apply an uncorrected warning
    // to a corrected derivation), and never invent a number.
    return "DWI space, corrected, unsigned";
  }
  const floor=label.match(/~?\s*(\d+(?:\.\d+)?)\s*mm\s+(?:geometric\s+)?floor/i);
  return floor ? `DWI space, recorded floor ${Number(floor[1])} mm` : "DWI space, floor unavailable";
}

export function spaceChipState(floorLabel) {
  return SIGNED_SHIFT_RE.test(typeof floorLabel === "string" ? floorLabel : "") ? "signed" : "caution";
}

export function preflightChipText(state) {
  if (state === "signed") return "Preflight signed";
  if (state === "unsigned") return "Preflight unsigned";
  if (state === "untestable") return "Preflight untestable";
  if (state === "stale") return "Preflight stale, review required";
  if (state === "unavailable") return "Preflight unavailable";
  if (state === "loading") return "Preflight loading";
  return "Preflight absent";
}

export function preflightChipState(state) {
  if (state === "signed") return "signed";
  if (["unsigned", "untestable", "stale", "unavailable"].includes(state)) return "caution";
  return "mute";
}

/** A returned fallback scorecard is evidence of untestability, not a signature. */
export function preflightState(payload) {
  const rows = payload?.verified?.criteria;
  if (!Array.isArray(rows) || !rows.length || rows.some(c => !c || !['pass','degraded','fail','untestable'].includes(c.verdict))) return 'unavailable';
  if (rows.every(c => c.verdict === 'untestable')) return 'untestable';
  if (!payload.stored) return 'absent';
  if (!Array.isArray(payload.drift) || payload.drift.length) return 'unsigned';
  const who = payload.stored.approved_by;
  return typeof who === 'string' && who.trim() ? 'signed' : 'unsigned';
}

export function operatingPointChipText(fidelitySummary) {
  const state = fidelitySummary?.state;
  const who = fidelitySummary?.who || "";
  if (state === "signed") return `Operating point: ${who}`;
  if (state === "pilot") {
    return /mixed/i.test(who) ? "Operating point: mixed" : "Operating point: pilot, unsigned";
  }
  return "Operating point: absent";
}

export function operatingPointChipState(fidelitySummary) {
  const state = fidelitySummary?.state;
  if (state === "signed") return "signed";
  if (state === "pilot") return "caution";
  return "mute";
}

/**
 * The one row of chips (DWI space · preflight · operating point ·
 * research-only) — the provenance card collapsed (ruling 8). Order is fixed.
 */
export function provenanceChips({ space, preflightState: pfState, fidelitySummary } = {}) {
  return [
    { text: spaceChipText(space), state: spaceChipState(space) },
    { text: preflightChipText(pfState), state: preflightChipState(pfState) },
    { text: operatingPointChipText(fidelitySummary), state: operatingPointChipState(fidelitySummary) },
    { text: "Research only. Not for navigation.", state: "research" },
  ];
}
