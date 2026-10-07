import { test } from "node:test";
import assert from "node:assert/strict";
import {
  YEO7_NETWORKS,
  networkColorHex,
  networkWashRgba,
  PARCEL_PROVENANCE,
  effectiveParcelOpacity,
  shortParcelName,
  formatParcelProbe,
  PARCEL_TRACT_DIM,
} from "../../viewer/parcel.js";

test("Yeo-7 didactic names are 7 networks", () => {
  assert.equal(YEO7_NETWORKS.length, 7);
  assert.equal(YEO7_NETWORKS[0].name, "Visual");
  assert.equal(YEO7_NETWORKS[6].name, "Default");
});

test("network colours are monochrome constants not RGB direction", () => {
  const h = networkColorHex(7);
  assert.equal(typeof h, "number");
  assert.match(networkWashRgba(1, 0.2), /^rgba\(/);
  assert.equal(PARCEL_PROVENANCE, "POPULATION ATLAS");
});

test("effectiveParcelOpacity dims when patient tract is on", () => {
  const base = effectiveParcelOpacity(0.4, false);
  const dim = effectiveParcelOpacity(0.4, true);
  assert.ok(dim < base);
  assert.ok(Math.abs(dim - base * PARCEL_TRACT_DIM) < 1e-9);
});

test("formatParcelProbe is lookup-only with POPULATION stamp", () => {
  const empty = formatParcelProbe(0, null);
  assert.equal(empty.empty, true);
  assert.match(empty.line, /POPULATION ATLAS/);
  const lut = {
    "142": {
      name: "7Networks_RH_Cont_PFCl_1",
      networkId: 6,
      networkName: "Frontoparietal",
      hemi: "R",
    },
  };
  const hit = formatParcelProbe(142, lut);
  assert.equal(hit.empty, false);
  assert.equal(hit.networkId, 6);
  assert.match(hit.line, /parcel 142/);
  assert.match(hit.line, /Frontoparietal/);
  assert.match(hit.line, /POPULATION ATLAS/);
  assert.equal(shortParcelName("7Networks_RH_Cont_PFCl_1"), "RH Cont PFCl 1");
});