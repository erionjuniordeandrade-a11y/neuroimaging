/**
 * E2 fidelity marking gates (Task 10).
 * Partition / HUD / colour helpers — no browser.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { solidColorForBundle, lowSupportColor } from "../../viewer/tract_colour.js";
import {
  FLAG_CROSSES_CAVITY,
  FLAG_CROSSES_LESION,
  isLowSupport,
  formatFidelityHud,
  decodeFidelity,
  formatOperatingPointChip,
  summarizeFidelity,
  partitionLowSupport,
  FLAG_PROVENANCE_INCOMPLETE,
} from "../../viewer/fidelity.js";

test("invalid and incomplete support is never assigned to the supported population", () => {
  assert.deepEqual(partitionLowSupport(
    new Float32Array([0.9, 0.2, NaN, Infinity, 0.9, 0.9]),
    new Uint8Array([0, 0, 0, 0, FLAG_PROVENANCE_INCOMPLETE, 0]), 0.7,
    new Float32Array([0.8, 0.1, NaN, 0.1, 0.8, NaN]),
  ), { high: [0], low: [1], unknown: [2, 3, 4, 5] });
  assert.equal(isLowSupport(NaN, 0, 0.7), false);
});

test("fidelity decoding honors byte views and unaligned little-endian blocks", () => {
  const data = new Uint8Array(16);
  new DataView(data.buffer).setFloat32(5, 0.8, true);
  new DataView(data.buffer).setFloat32(9, 0.4, true);
  const headers = new Map([
    ['X-fidelityStatus', 'ok'], ['X-fidelityOffset', '4'],
    ['X-fidelityR', '0.3'], ['X-fidelityMinFrac', '0.7'],
  ]);
  const got = decodeFidelity(data.subarray(1, 14), headers, 1);
  assert.equal(got.status, 'ok');
  assert.equal(got.fracGeR[0], Math.fround(0.8));
  assert.equal(got.p5[0], Math.fround(0.4));
  headers.set('X-fidelityMinFrac', 'NaN');
  assert.equal(decodeFidelity(data, headers, 1).status, 'error');
});

test("lesion bit never marks; cavity bit marks", () => {
  assert.equal(isLowSupport(0.99, FLAG_CROSSES_LESION, 0.7), false);
  assert.equal(isLowSupport(0.99, FLAG_CROSSES_CAVITY, 0.7), true);
  assert.equal(isLowSupport(0.99, FLAG_CROSSES_LESION | FLAG_CROSSES_CAVITY, 0.7), true);
});

test("minFrac boundary is exclusive < not <=", () => {
  assert.equal(isLowSupport(0.7, 0, 0.7), false);
  assert.equal(isLowSupport(0.699, 0, 0.7), true);
  assert.equal(isLowSupport(0.7, FLAG_CROSSES_LESION, 0.7), false);
});

test("minFrac boundary compares at float32 precision, matching the server", () => {
  // frac_ge_R is stored as float32 server-side; the client decodes it out of
  // a Float32Array, so 0.7 here is exactly float32(0.7) — not the double
  // 0.7 a header string parses to. A naive double compare flips this row to
  // low-support even though the server (also float32) does not mark it.
  const fracGeR = new Float32Array([0.7]);
  assert.equal(
    isLowSupport(fracGeR[0], 0, 0.7), false,
    "exactly the float32 operating point must not be low-support",
  );
  assert.equal(isLowSupport(0.69, 0, 0.7), true);
});

test("low-support colour is same-hue, not recovery grey, not the pure hue", () => {
  const hue = solidColorForBundle("true_cst", "CST-L");
  const low = lowSupportColor(hue);
  const recovery = [0.6, 0.6, 0.58];
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  assert.ok(low.opacity === 0.35);
  assert.ok(dist(low.rgb, hue) > 0.15, "must desaturate away from the pure hue");
  assert.ok(dist(low.rgb, recovery) > 0.15, "must not collapse onto recovery grey");
});

test("decodeFidelity refuses a missing or non-numeric offset", () => {
  const headers = {
    get(k) {
      const h = {
        "X-fidelityStatus": "ok",
        "X-fidelityR": "0.30",
        "X-fidelityMinFrac": "0.70",
      };
      return h[k] ?? null;
    },
  };
  const buf = new ArrayBuffer(36);
  const out = decodeFidelity(buf, headers, 2);
  assert.equal(out.status, "error");
  assert.equal(
    formatFidelityHud({ status: out.status, lowSupportCount: 0, crossLesionCount: 0 }),
    "fidelity: untested",
  );
});

test("decodeFidelity refuses a short block", () => {
  const headers = {
    get(k) {
      const h = {
        "X-fidelityStatus": "ok",
        "X-fidelityOffset": "0",
        "X-fidelityR": "0.30",
        "X-fidelityMinFrac": "0.70",
      };
      return h[k] ?? null;
    },
  };
  const out = decodeFidelity(new ArrayBuffer(8), headers, 2);
  assert.equal(out.status, "error");
});

test("decoded support rows must match the actual geometry, with whole counts and aligned offsets", () => {
  const base = {
    'X-fidelityStatus': 'ok', 'X-fidelityOffset': '0',
    'X-fidelityR': '0.3', 'X-fidelityMinFrac': '0.7', 'X-nDisplayed': '2',
  };
  for (const [overrides, lineCount] of [
    [{'X-nDisplayed': '1'}, 2], [{'X-nDisplayed': '3'}, 2],
    [{'X-nDisplayed': '2.5'}, 2], [{}, 2.5],
    [{'X-fidelityOffset': '1'}, 2], [{'X-fidelityR': ''}, 2],
    [{'X-fidelityMinFrac': 'NaN'}, 2],
  ]) {
    const headers = new Map(Object.entries({...base, ...overrides}));
    assert.equal(decodeFidelity(new ArrayBuffer(64), headers, lineCount).status, 'error');
  }
  assert.equal(decodeFidelity(new ArrayBuffer(18), new Map(Object.entries(base)), 2).status, 'ok');
});

test("fidelity offsets are relative to the supplied buffer view", () => {
  const raw = new ArrayBuffer(32);
  const view = new Uint8Array(raw, 4, 18);
  new Float32Array(raw, 4, 4).set([0.2, 0.9, 0.1, 0.4]);
  const headers = new Map(Object.entries({
    'X-fidelityStatus': 'ok', 'X-fidelityOffset': '0',
    'X-fidelityR': '0.3', 'X-fidelityMinFrac': '0.7', 'X-nDisplayed': '2',
  }));
  const out = decodeFidelity(view, headers, 2);
  assert.equal(out.status, 'ok');
  assert.deepEqual([...out.fracGeR], [...new Float32Array([0.2, 0.9])]);
});

test("absent HUD is untested, never a 0 count", () => {
  assert.equal(
    formatFidelityHud({ status: "absent", lowSupportCount: 0, crossLesionCount: 0 }),
    "fidelity: untested",
  );
  assert.equal(
    formatFidelityHud({ status: "ok", lowSupportCount: 0, crossLesionCount: 0 }),
    "",
  );
  assert.equal(
    formatFidelityHud({ status: "ok", lowSupportCount: 41, crossLesionCount: 3 }),
    "41 low-support · 3 cross lesion",
  );
  assert.equal(
    formatFidelityHud({ status: "ok", lowSupportCount: 12, crossLesionCount: 0 }),
    "12 low-support",
  );
});

test("operating-point chip names its source: signed vs pilot", () => {
  assert.equal(
    formatOperatingPointChip({ R: 0.3, minFrac: 0.7, opSource: "pilot", approvedBy: "" }),
    "R=0.3 · min 0.7 · pilot · unsigned",
  );
  assert.equal(
    formatOperatingPointChip({ R: 0.45, minFrac: 0.9, opSource: "signed", approvedBy: "erion" }),
    "R=0.45 · min 0.9 · signed erion",
  );
  assert.equal(formatOperatingPointChip({ R: null, minFrac: null }), "");
});

test("decodeFidelity treats any non-'signed' source header as pilot", () => {
  const n = 1;
  const buf = new ArrayBuffer(n * 9);
  const mk = (src) => {
    const h = new Map([
      ["X-fidelityStatus", "ok"], ["X-fidelityOffset", "0"],
      ["X-fidelityR", "0.3"], ["X-fidelityMinFrac", "0.7"],
    ]);
    if (src != null) h.set("X-fidelityOpSource", src);
    return decodeFidelity(buf, h, n);
  };
  assert.equal(mk(null).opSource, "pilot");
  assert.equal(mk("pilot").opSource, "pilot");
  assert.equal(mk("SIGNED").opSource, "pilot");
  assert.equal(mk("signed").opSource, "signed");
});

test("summarizeFidelity: empty when nothing is ok", () => {
  assert.deepEqual(summarizeFidelity([]), { state: "", who: "", chip: "" });
  assert.deepEqual(
    summarizeFidelity([null, { status: "error" }, { status: "absent" }]),
    { state: "", who: "", chip: "" },
  );
});

test("summarizeFidelity: single signed entry with approvedBy", () => {
  const out = summarizeFidelity([
    { status: "ok", opSource: "signed", R: 0.45, minFrac: 0.9, approvedBy: "erion" },
  ]);
  assert.equal(out.state, "signed");
  assert.equal(out.who, "signed erion");
  assert.equal(out.chip, "R=0.45 · min 0.9 · signed erion");
});

test("summarizeFidelity: two identical pilot entries", () => {
  const f = { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.7, approvedBy: "" };
  const out = summarizeFidelity([f, { ...f }]);
  assert.equal(out.state, "pilot");
  assert.equal(out.who, "pilot · unsigned");
  assert.equal(out.chip, "R=0.3 · min 0.7 · pilot · unsigned");
});

test("summarizeFidelity: differing opSource is mixed, never presented as signed", () => {
  const out = summarizeFidelity([
    { status: "ok", opSource: "signed", R: 0.3, minFrac: 0.7, approvedBy: "erion" },
    { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.7, approvedBy: "" },
  ]);
  assert.equal(out.state, "pilot");
  assert.equal(out.who, "mixed operating points · unsigned");
  assert.equal(out.chip, "operating point: mixed");
});

test("summarizeFidelity: minFrac differing only below float32 precision is NOT mixed", () => {
  const out = summarizeFidelity([
    { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.7, approvedBy: "" },
    { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.7000000001, approvedBy: "" },
  ]);
  assert.equal(out.state, "pilot");
  assert.equal(out.who, "pilot · unsigned");
  assert.equal(out.chip, "R=0.3 · min 0.7 · pilot · unsigned");
});

test("summarizeFidelity: minFrac 0.7 vs 0.71 is mixed", () => {
  const out = summarizeFidelity([
    { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.7, approvedBy: "" },
    { status: "ok", opSource: "pilot", R: 0.3, minFrac: 0.71, approvedBy: "" },
  ]);
  assert.equal(out.state, "pilot");
  assert.equal(out.who, "mixed operating points · unsigned");
  assert.equal(out.chip, "operating point: mixed");
});
