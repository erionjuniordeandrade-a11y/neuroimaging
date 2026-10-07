/**
 * Distance-legend ticks for the clinical proximity ramp (S-12).
 *
 * The middle tick is THIS case's served geometric floor (X-clearanceFloorMm),
 * never a literal: on an uncorrected case it reads 3, on a reverse-PE corrected
 * case with a signed 0.6 mm delta-QC median it reads 0.6, and when the server
 * refused to state a floor it reads "—" so the key never implies a number the
 * case did not record.
 */
export const DIST_LEGEND_MAX_MM = 25;
/** Schematic mid-ramp label (the ramp's own stop), not a case measurement. */
export const DIST_LEGEND_MID_MM = 12;

/** Format a floor for the key: integers bare, sub-integer floors to one decimal. */
export function formatFloorTick(floorMm) {
  const f = Number(floorMm);
  if (floorMm == null || !Number.isFinite(f) || f <= 0) return "—";
  return Number.isInteger(f) ? String(f) : f.toFixed(1);
}

/** [0, floor, ~mid, ≥max mm] — same shape as the four spans in #distLegend. */
export function legendTicks(floorMm) {
  return Object.freeze([
    "0",
    formatFloorTick(floorMm),
    `~${DIST_LEGEND_MID_MM}`,
    `≥${DIST_LEGEND_MAX_MM} mm`,
  ]);
}
