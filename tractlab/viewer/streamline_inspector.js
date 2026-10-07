import {
  FLAG_CROSSES_CAVITY,
  FLAG_CROSSES_LESION,
  isLowSupport,
  isUnmeasurable,
} from "./fidelity.js";

/** Decode the response's displayed-index -> source-index lookup table. */
export function decodeSourceOrdinals(buffer, headers, nDisplayed) {
  const offsetRaw = headers?.get?.("X-sourceOrdinalOffset");
  const countRaw = headers?.get?.("X-sourceOrdinalCount");
  const encoding = headers?.get?.("X-sourceOrdinalEncoding");
  const offset = Number(offsetRaw);
  const count = Number(countRaw);
  const expected = Number(nDisplayed);
  if (
    offsetRaw == null || countRaw == null || encoding !== "uint32le" ||
    !Number.isInteger(offset) || offset < 0 ||
    !Number.isInteger(count) || count < 0 ||
    !Number.isInteger(expected) || expected < 0 || count !== expected
  ) {
    return null;
  }
  const raw = buffer instanceof ArrayBuffer
    ? buffer
    : buffer?.buffer instanceof ArrayBuffer
      ? buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
      : null;
  if (!raw || offset + count * 4 > raw.byteLength) return null;
  const view = new DataView(raw, offset, count * 4);
  const ordinals = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) ordinals[i] = view.getUint32(i * 4, true);
  return ordinals;
}

/**
 * Three.LineSegments reports the index-buffer entry of the hit segment.
 * Convert that entry back to the zero-based displayed streamline index.
 */
export function streamlineIndexFromSegmentEntry(indexEntry, pointsPerLine, lineCount) {
  if (indexEntry == null) return null;
  const entry = Number(indexEntry);
  const k = Number(pointsPerLine);
  const n = Number(lineCount);
  if (
    !Number.isInteger(entry) || entry < 0 ||
    !Number.isInteger(k) || k < 2 ||
    !Number.isInteger(n) || n < 1
  ) {
    return null;
  }
  const segmentOrdinal = Math.floor(entry / 2);
  const streamlineIndex = Math.floor(segmentOrdinal / (k - 1));
  return streamlineIndex < n ? streamlineIndex : null;
}

function finiteAt(values, index) {
  if (!values || index < 0 || index >= values.length) return null;
  const value = Number(values[index]);
  return Number.isFinite(value) ? value : null;
}

function shortNumber(value, digits = 1) {
  if (Number.isInteger(value)) return String(value);
  return Number(value).toFixed(digits).replace(/\.0+$/, "");
}

/** Build copy-safe, semantics-safe text for a pinned streamline. */
export function summarizeStreamline({
  displayIndex,
  sourceOrdinals,
  sourcePopulation = "",
  lineCount,
  pointsPerLine,
  distances,
  fidelity,
  tract,
  geomFloorMm = null,
}) {
  const index = Number(displayIndex);
  const n = Number(lineCount);
  const k = Number(pointsPerLine);
  if (!Number.isInteger(index) || index < 0 || index >= n) {
    throw new RangeError("displayIndex is outside the displayed tract layer");
  }

  const mapped = sourceOrdinals && index < sourceOrdinals.length
    ? Number(sourceOrdinals[index])
    : null;
  const sourceIndex = Number.isInteger(mapped) && mapped >= 0 ? mapped : null;

  let minDistanceMm = null;
  if (distances && Number.isInteger(k) && k > 0 && distances.length >= (index + 1) * k) {
    for (let point = index * k; point < (index + 1) * k; point += 1) {
      const value = Number(distances[point]);
      if (Number.isFinite(value)) {
        minDistanceMm = minDistanceMm == null ? value : Math.min(minDistanceMm, value);
      }
    }
  }
  const hasRecordedFloor = (
    geomFloorMm != null && String(geomFloorMm).trim() !== ""
  );
  const floor = hasRecordedFloor ? Number(geomFloorMm) : null;
  const clearanceText = minDistanceMm == null
    ? "distance unavailable"
    : !Number.isFinite(floor)
      ? "geometric floor unrecorded — distance not reportable"
      : minDistanceMm < floor
      ? `below ${shortNumber(floor)} mm geometric floor`
      : `${shortNumber(minDistanceMm)} mm minimum to lesion`;

  let fidelityText = "fidelity: untested";
  let lowSupport = null;
  let crossesLesion = false;
  let crossesCavity = false;
  if (fidelity?.status === "ok") {
    const frac = finiteAt(fidelity.fracGeR, index);
    const p5 = finiteAt(fidelity.p5, index);
    const flags = finiteAt(fidelity.flags, index);
    if (flags != null) {
      crossesLesion = (flags & FLAG_CROSSES_LESION) !== 0;
      crossesCavity = (flags & FLAG_CROSSES_CAVITY) !== 0;
    }
    if (frac != null && p5 != null && flags != null && !isUnmeasurable(frac, flags, p5)) {
      lowSupport = isLowSupport(frac, flags, fidelity.minFrac);
      crossesLesion = (flags & FLAG_CROSSES_LESION) !== 0;
      crossesCavity = (flags & FLAG_CROSSES_CAVITY) !== 0;
      const bits = [
        `support ≥R ${Math.round(frac * 100)}%`,
        p5 == null ? null : `support p5 ratio ${p5.toFixed(2)}`,
        lowSupport ? "low-support" : "supported",
        crossesLesion ? "crosses lesion" : null,
        crossesCavity ? "crosses cavity" : null,
      ].filter(Boolean);
      fidelityText = bits.join(" · ");
    } else fidelityText = `support unmeasurable${crossesCavity ? ' · crosses cavity' : ''}${crossesLesion ? ' · crosses lesion' : ''}`;
  }

  return Object.freeze({
    displayIndex: index,
    displayOrdinal: index + 1,
    sourceIndex,
    sourceOrdinal: sourceIndex == null ? null : sourceIndex + 1,
    sourcePopulation: String(sourcePopulation || ""),
    tractLabel: String(tract?.short || "streamline"),
    minDistanceMm,
    clearanceText,
    fidelityText,
    lowSupport,
    crossesLesion,
    crossesCavity,
  });
}
