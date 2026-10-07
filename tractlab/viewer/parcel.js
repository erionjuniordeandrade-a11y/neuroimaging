/**
 * Parcellation prior UI helpers (A2). Sibling wall to tract priors (ADR-0001).
 * Single active network; Yeo-7 didactic names; never direction-RGB.
 * Parcel probe is lookup only — never clearance / tracking input.
 */

export const YEO7_NETWORKS = Object.freeze([
  { id: 1, name: "Visual" },
  { id: 2, name: "Somatomotor" },
  { id: 3, name: "Dorsal Attention" },
  { id: 4, name: "Ventral Attention" },
  { id: 5, name: "Limbic" },
  { id: 6, name: "Frontoparietal" },
  { id: 7, name: "Default" },
]);

/** Desaturated wash/ghost colours (0–1) — lowest visual weight of three classes. */
export const YEO7_COLORS = Object.freeze({
  1: [0.55, 0.45, 0.65],
  2: [0.55, 0.55, 0.45],
  3: [0.45, 0.55, 0.50],
  4: [0.60, 0.48, 0.42],
  5: [0.50, 0.55, 0.42],
  6: [0.45, 0.50, 0.60],
  7: [0.58, 0.48, 0.48],
});

export function networkColorHex(id) {
  const c = YEO7_COLORS[id] || [0.5, 0.5, 0.5];
  const r = Math.round(c[0] * 255);
  const g = Math.round(c[1] * 255);
  const b = Math.round(c[2] * 255);
  return (r << 16) | (g << 8) | b;
}

export function networkWashRgba(id, alpha = 0.22) {
  const c = YEO7_COLORS[id] || [0.5, 0.5, 0.5];
  return `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${alpha})`;
}

export const PARCEL_PROVENANCE = "POPULATION ATLAS";

/** When a patient tract is visible, dim population wash so tubes keep hierarchy. */
export const PARCEL_TRACT_DIM = 0.45;

/**
 * Effective wash/mesh opacity from user slider + optional patient-tract dim.
 * @param {number} slider 0–1 user intensity
 * @param {boolean} patientTractOn
 */
export function effectiveParcelOpacity(slider, patientTractOn = false) {
  const v = Number(slider);
  const base = Math.min(0.7, Math.max(0.04, Number.isFinite(v) ? v : 0.22));
  if (!patientTractOn) return base;
  return Math.max(0.04, base * PARCEL_TRACT_DIM);
}

/**
 * Shorten Schaefer order name for probe line.
 * e.g. 7Networks_RH_Cont_PFCl_1 → RH Cont PFCl 1
 * @param {string} name
 */
export function shortParcelName(name) {
  const s = String(name || "");
  return s
    .replace(/^7Networks_/, "")
    .replace(/_/g, " ")
    .trim() || "unlabeled";
}

/**
 * Build MPR probe readout from label id + LUT row.
 * @param {number} labelId
 * @param {Record<string, {name?: string, networkId?: number, networkName?: string, hemi?: string}>|null} lut
 * @returns {{ line: string, empty: boolean, labelId: number, networkId: number }}
 */
export function formatParcelProbe(labelId, lut) {
  const id = Number(labelId) || 0;
  if (id <= 0) {
    return {
      line: "Outside labeled cortex · POPULATION ATLAS",
      empty: true,
      labelId: 0,
      networkId: 0,
    };
  }
  const row = lut && (lut[String(id)] || lut[id]);
  const netId = Number(row?.networkId) || 0;
  const netName =
    row?.networkName ||
    YEO7_NETWORKS.find((n) => n.id === netId)?.name ||
    (netId ? `Network ${netId}` : "unknown network");
  const short = shortParcelName(row?.name || "");
  const hemi = row?.hemi && row.hemi !== "?" ? ` · ${row.hemi}` : "";
  return {
    line: `parcel ${id} · ${short}${hemi} · ${netName} · ${PARCEL_PROVENANCE}`,
    empty: false,
    labelId: id,
    networkId: netId,
  };
}