/**
 * E2 fidelity helpers — decode, partition, HUD copy.
 * Marking rule: low-support = frac_ge_R < minFrac OR crosses_cavity.
 * crosses_lesion is a count only (never a mark).
 */

export const FLAG_CROSSES_LESION = 1;
export const FLAG_CROSSES_CAVITY = 2;
export const FLAG_PROVENANCE_INCOMPLETE = 4;

export function isLowSupport(fracGeR, flags, minFrac) {
  // Server compares frac_ge_R against min_frac at float32 precision (numpy).
  // fracGeR here is already an exact float32 value read out of a
  // Float32Array, but minFrac arrives as a header string parsed to a JS
  // double — round both through Math.fround so a value that lands exactly
  // on the float32 operating point (e.g. frac == float32(0.7)) does not
  // flip to low-support just because double(0.7) != float32(0.7).
  return (Number.isFinite(fracGeR) && Math.fround(fracGeR) < Math.fround(Number(minFrac)))
    || (Number(flags) & FLAG_CROSSES_CAVITY) !== 0;
}

export function isUnmeasurable(frac, flags, p5 = 0) {
  return !Number.isFinite(frac) || frac < 0 || frac > 1 || !Number.isFinite(p5)
    || (Number(flags) & FLAG_PROVENANCE_INCOMPLETE) !== 0;
}

export function partitionLowSupport(fracGeR, flags, minFrac, p5 = null) {
  const n = fracGeR.length;
  const low = [];
  const high = [];
  const unknown = [];
  for (let i = 0; i < n; i++) {
    if ((flags[i] & FLAG_CROSSES_CAVITY) && !(flags[i] & FLAG_PROVENANCE_INCOMPLETE)) low.push(i);
    else if (isUnmeasurable(fracGeR[i], flags[i], p5 ? p5[i] : 0)) unknown.push(i);
    else if (isLowSupport(fracGeR[i], flags[i], minFrac)) low.push(i);
    else high.push(i);
  }
  return { low, high, unknown };
}

export function formatOperatingPointChip({ R, minFrac, opSource, approvedBy }) {
  if (R == null || minFrac == null) return "";
  const who = opSource === "signed" ? `signed${approvedBy ? " " + approvedBy : ""}` : "pilot · unsigned";
  return `R=${R} · min ${minFrac} · ${who}`;
}

// Summarize the "who" (HUD tail) / "chip" (operating-point badge) / "state"
// (badge styling hook) across ALL loaded layers' fidelity, so an aggregate
// count (sumLowSupportCount/totalTubeLineCount) is never attributed to a
// single bank's provenance when banks disagree on opSource/R/minFrac.
export function summarizeFidelity(fidelities) {
  const ok = (fidelities || []).filter((f) => f && f.status === "ok");
  if (!ok.length) return { state: "", who: "", chip: "" };
  const first = ok[0];
  const r0 = Math.fround(Number(first.R));
  const m0 = Math.fround(Number(first.minFrac));
  const op0 = first.opSource;
  const uniform = ok.every((f) =>
    f.opSource === op0
    && Math.fround(Number(f.R)) === r0
    && Math.fround(Number(f.minFrac)) === m0
  );
  if (!uniform) {
    // Mixed provenance is never presented as signed, even if some entries are.
    return { state: "pilot", who: "mixed operating points · unsigned", chip: "operating point: mixed" };
  }
  const state = op0 === "signed" ? "signed" : "pilot";
  let who;
  if (op0 === "signed") {
    const sameApprovedBy = ok.every((f) => f.approvedBy === first.approvedBy);
    who = sameApprovedBy ? `signed${first.approvedBy ? " " + first.approvedBy : ""}` : "signed";
  } else {
    who = "pilot · unsigned";
  }
  return { state, who, chip: formatOperatingPointChip(first) };
}

export function formatFidelityHud({ status, lowSupportCount, crossLesionCount, unmeasurableCount }) {
  if (status !== "ok") return "fidelity: untested";
  const bits = [];
  if (lowSupportCount) bits.push(`${lowSupportCount} low-support`);
  if (unmeasurableCount) bits.push(`${unmeasurableCount} support unmeasurable`);
  if (crossLesionCount) bits.push(`${crossLesionCount} cross lesion`);
  return bits.join(" · ");
}

export function decodeFidelity(buf, headers, lineCount) {
  const status = headers.get("X-fidelityStatus") || "absent";
  if (status !== "ok") return { status };
  const offRaw = headers.get("X-fidelityOffset");
  if (offRaw == null || !/^\d+$/.test(String(offRaw))) {
    return { status: "error" };
  }
  const off = Number(offRaw);
  const rRaw = headers.get("X-fidelityR");
  const minRaw = headers.get("X-fidelityMinFrac");
  const R = Number(rRaw);
  const minFrac = Number(minRaw);
  const displayedRaw = headers.get("X-nDisplayed");
  // The caller supplies the geometry count, never the optional display-count
  // header. A stale count cannot turn unmeasured geometry into supported rows.
  if (!Number.isSafeInteger(lineCount) || lineCount < 0
      || (displayedRaw != null && (!/^\d+$/.test(String(displayedRaw)) || Number(displayedRaw) !== lineCount))
      || rRaw == null || String(rRaw).trim() === '' || !Number.isFinite(R) || R < 0
      || minRaw == null || String(minRaw).trim() === '' || !Number.isFinite(minFrac)
      || minFrac < 0 || minFrac > 1) {
    return { status: "error" };
  }
  const n = lineCount;
  const need = n * 9;
  if (!(buf instanceof ArrayBuffer) && !ArrayBuffer.isView(buf)) return { status: "error" };
  // Copy the exact byte view, then read through DataView: the block may sit
  // at any byte offset and is always little-endian on the wire.
  const raw = buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  // The offset is a wire contract (4-byte aligned within the body); the view
  // itself may start at any byte.
  if (!Number.isSafeInteger(off) || off % 4 !== 0 || !Number.isSafeInteger(need) || off + need > raw.byteLength) {
    return { status: "error" };
  }
  const fracGeR = new Float32Array(n);
  const p5 = new Float32Array(n);
  const flags = new Uint8Array(n);
  const view = new DataView(raw);
  for (let i = 0; i < n; i++) {
    fracGeR[i] = view.getFloat32(off + i * 4, true);
    p5[i] = view.getFloat32(off + n * 4 + i * 4, true);
  }
  flags.set(new Uint8Array(raw, off + n * 8, n));
  return {
    status: "ok",
    fracGeR,
    p5,
    flags,
    R,
    minFrac,
    // "signed" only when the server read an owner-signed sheet; anything
    // else is the unsigned pilot point and the chip must say so.
    opSource: headers.get("X-fidelityOpSource") === "signed" ? "signed" : "pilot",
    approvedBy: headers.get("X-fidelityApprovedBy") || "",
    opDate: headers.get("X-fidelityOpDate") || "",
    lowSupportCount: Number(headers.get("X-lowSupportCount") || 0),
    crossLesionCount: Number(headers.get("X-crossesLesionCount") || 0),
    crossCavityCount: Number(headers.get("X-crossesCavityCount") || 0),
    unmeasurableCount: Number(headers.get("X-unmeasurableCount") || 0),
  };
}

export function extractStreamlines(flat, dist, lineCount, k, indices) {
  const stride = k * 3;
  const out = new Float32Array(indices.length * stride);
  let dOut = null;
  if (dist) dOut = new Float32Array(indices.length * k);
  for (let j = 0; j < indices.length; j++) {
    const i = indices[j];
    out.set(flat.subarray(i * stride, (i + 1) * stride), j * stride);
    if (dOut) dOut.set(dist.subarray(i * k, (i + 1) * k), j * k);
  }
  return { flat: out, dist: dOut, lineCount: indices.length };
}
