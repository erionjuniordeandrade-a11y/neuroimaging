/**
 * Short-form derivation copy — pure text builders (ADR-0007 / CLAUDE.md:
 * "do not apply an uncorrected-case warning to a corrected derivation").
 *
 * A short binary fragment ("no reverse-PE" vs "reverse-PE corrected") is
 * derived from the served /api/derivation `kind` field — never a hard-coded
 * constant. A fragment that needs the recorded mm figure is derived from the
 * served `floor_label` string instead (the only source with that number),
 * mirroring viewer/provenance.js's spaceChipText parsing so the two surfaces
 * can never disagree. No DOM here; index.html wires these into element text
 * and title attributes.
 */

export const KIND_UNCORRECTED = "uncorrected";
export const KIND_RPE_PAIR = "rpe_pair";
const UNAVAILABLE_NOTE = "derivation status unavailable";

/** "no reverse-PE" ONLY for an explicitly uncorrected lineage; "reverse-PE
 * corrected" ONLY for an explicitly rpe_pair lineage. A null/unknown kind
 * (API not yet loaded, fetch failed, unrecognized value) must never guess
 * "uncorrected" — an absent answer is not evidence of an uncorrected case,
 * so it reads as unavailable instead (CLAUDE.md: never apply an
 * uncorrected-case warning where the derivation is not actually known). */
export function reversePeNote(kind) {
  if (kind === KIND_RPE_PAIR) return "reverse-PE corrected";
  if (kind === KIND_UNCORRECTED) return "no reverse-PE";
  return UNAVAILABLE_NOTE;
}

export function researchChipTitle(kind) {
  return `Research preview only — geometric uncertainty, ${reversePeNote(kind)}`;
}

/** "few-mm uncertainty" is only ever asserted for an uncorrected lineage
 * (its fixed geometric floor) or a corrected lineage whose delta QC is
 * SIGNED (a measured shift, not an assumption). A corrected-but-unsigned
 * lineage states correction status only, in the served floor_label's own
 * "residual uncertainty unquantified" wording (derivation.py's
 * _LABEL_RPE_UNSIGNED) — never a borrowed few-mm claim. An unknown kind
 * carries no uncertainty claim at all. */
export function t1AnatomyTitle(kind, deltaQcSigned) {
  if (kind === KIND_RPE_PAIR) {
    return deltaQcSigned
      ? "T1 anatomy (QC-approved · reverse-PE corrected · few-mm uncertainty)"
      : "T1 anatomy (QC-approved · reverse-PE corrected · residual uncertainty unquantified)";
  }
  if (kind === KIND_UNCORRECTED) {
    return "T1 anatomy (QC-approved · no reverse-PE · few-mm uncertainty)";
  }
  return `T1 anatomy (QC-approved · ${UNAVAILABLE_NOTE})`;
}

/** Recovery-radius slider caption: the recorded floor mm figure ONLY when
 * floor_label actually carries one (uncorrected case); a corrected lineage
 * never borrows that floor wording — same rule as provenance.js. */
export function recoveryFloorNote(floorLabel) {
  const label = typeof floorLabel === "string" ? floorLabel : "";
  const floor = label.match(/~?\s*(\d+(?:\.\d+)?)\s*mm\s+(?:geometric\s+)?floor/i);
  if (floor) return `(from lesion surface · geom floor ~${Number(floor[1])} mm)`;
  if (/reverse-PE corrected/i.test(label)) return "(from lesion surface · reverse-PE corrected)";
  return "(from lesion surface · floor unavailable)";
}
