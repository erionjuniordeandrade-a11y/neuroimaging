/**
 * Atlas prior identity — separate wall from Tract (ADR-0001).
 * Laterality still from id tokens (norm_cst_r → R).
 */

import { sideFromIdToken } from "./tract.js";

/** @typedef {'L'|'R'|'?'} Side */

/**
 * @param {{ id: string, label?: string, officialName?: string }} rec
 */
export function describePrior(rec) {
  const id = String(rec?.id || "");
  if (!id) {
    return Object.freeze({
      side: "?",
      short: "prior",
      chip: "prior",
      id: "",
      source: "atlas",
      provenance: "POPULATION ATLAS",
    });
  }
  const side = sideFromIdToken(id);
  const short = shortFromPriorId(id, side);
  return Object.freeze({
    side,
    short,
    chip: short,
    id,
    source: "atlas",
    provenance: "POPULATION ATLAS",
  });
}

function shortFromPriorId(id, side) {
  const s = String(id || "").replace(/^norm_/, "");
  if (/cst/i.test(s)) return side === "?" ? "CST prior" : `CST-${side} prior`;
  if (/fa|aslant/i.test(s)) return side === "?" ? "FAT prior" : `FAT-${side} prior`;
  if (/slf3/i.test(s)) return side === "?" ? "SLF3 prior" : `SLF3-${side} prior`;
  if (/slf2/i.test(s)) return side === "?" ? "SLF2 prior" : `SLF2-${side} prior`;
  if (/slf1/i.test(s)) return side === "?" ? "SLF1 prior" : `SLF1-${side} prior`;
  if (/ifo/i.test(s)) return side === "?" ? "IFOF prior" : `IFOF-${side} prior`;
  if (/uf/i.test(s)) return side === "?" ? "UF prior" : `UF-${side} prior`;
  if (/cbd|cing/i.test(s)) return side === "?" ? "Cing prior" : `Cing-${side} prior`;
  if (/or/i.test(s)) return side === "?" ? "OR prior" : `OR-${side} prior`;
  if (/fmi|forceps.?minor/i.test(s)) return "Forceps min prior";
  if (/fma|forceps.?major/i.test(s)) return "Forceps maj prior";
  if (/af/i.test(s)) return side === "?" ? "AF prior" : `AF-${side} prior`;
  return s.replace(/_/g, " ") + " prior";
}

/** Fixed monochrome family for all priors — never direction-RGB. */
export const PRIOR_SHELL_COLOR = 0x8a9bb0;
export const PRIOR_CONTOUR_RGBA = "rgba(160, 180, 210, 0.85)";
export const PRIOR_PROB_THR = 0.25; // display convention (not tunable evidence)
