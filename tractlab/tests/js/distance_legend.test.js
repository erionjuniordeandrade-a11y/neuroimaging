import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { legendTicks, formatFloorTick } from "../../viewer/distance_legend.js";

const here = dirname(fileURLToPath(import.meta.url));
const viewer = join(here, "..", "..", "viewer");

test("legend floor tick equals the served floor, never a literal", () => {
  assert.equal(legendTicks(3)[1], "3"); // uncorrected case: recorded 3 mm
  assert.equal(legendTicks(0.6033)[1], "0.6"); // rpe_pair, signed delta-QC median
  assert.equal(legendTicks(1.5)[1], "1.5");
  assert.deepEqual([...legendTicks(3)], ["0", "3", "~12", "≥25 mm"]);
});

test("a refused or blank floor renders an honest dash, not 3", () => {
  for (const v of [null, undefined, "", NaN, 0, -1, "abc"]) {
    assert.equal(formatFloorTick(v), "—", `floor ${String(v)}`);
  }
});

test("S-12: index.html no longer hard-codes the 3 mm tick", () => {
  const html = readFileSync(join(viewer, "index.html"), "utf8");
  assert.doesNotMatch(html, /<span>3<\/span>/);
  // main's scene legend draws the floor from each layer's served geomFloorMm.
  assert.match(html, /\.geomFloorMm\)/);
});

test("S-14: connectotomy.js carries no derivation-blind floor literal", () => {
  const js = readFileSync(join(viewer, "connectotomy.js"), "utf8");
  assert.doesNotMatch(js, /Geometry uncertain to ~3 mm/);
  assert.doesNotMatch(js, /CONNECTOTOMY_NOTE/);
  assert.doesNotMatch(js, /no reverse-PE/);
});

test("S-13: viewer sources carry no 'True CST' prose", () => {
  for (const f of ["index.html", "connectotomy.js", "tract.js", "tract_colour.js"]) {
    const src = readFileSync(join(viewer, f), "utf8");
    assert.doesNotMatch(src, /True CST/, f);
  }
});
