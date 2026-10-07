/**
 * Laterality / tract descriptor suite.
 * Run: npm test   or   node --test tests/js/
 *
 * The control that matters: brokenShortBankLabelProseSide must FAIL the
 * SLF-III-L / frontal case — proving the suite can catch the defect class.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  describeTract,
  describeGeneratedLayer,
  describeLiveTrack,
  describeRecoveryTrack,
  describeEdgeTrack,
  sideFromIdToken,
  sideFromPathToken,
  TRACT_GROUP_META,
} from "../../viewer/tract.js";
import { brokenShortBankLabelProseSide } from "./fixtures/tract-prose-side-defect.js";
import { syntheticBankManifest } from "./fixtures/bank-catalog.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

// ── control: suite must be able to catch the bug ───────────────────────────

test("CONTROL: broken prose-side label misreads SLF-III-L (frontal)", () => {
  const wrong = brokenShortBankLabelProseSide({
    id: "bank_slf3_l",
    label: "SLF-III-L (SMG↔frontal)",
  });
  // Historical defect: returned SLF3-R because /slf.*r/ matched "frontal"
  assert.equal(
    wrong,
    "SLF3-R",
    "control fixture must still exhibit the frontal→R bug",
  );
});

test("CONTROL: describeTract does not exhibit the frontal→R bug", () => {
  const d = describeTract({
    id: "bank_slf3_l",
    label: "SLF-III-L (SMG↔frontal)",
    role: "true_slf3",
    nStreamlines: 2500,
  });
  assert.equal(d.side, "L");
  assert.equal(d.short, "SLF-III-L");
  assert.notEqual(d.short, "SLF-III-R");
  assert.equal(d.chip, "SLF-III-L · n≈2500");
  assert.equal(d.group, "slf");
  assert.equal(d.source, "bank");
});

// ── id authority (label ignored for side) ──────────────────────────────────

test("sideFromIdToken: standalone L/R segments only", () => {
  assert.equal(sideFromIdToken("bank_slf3_l"), "L");
  assert.equal(sideFromIdToken("bank_slf3_r"), "R");
  assert.equal(sideFromIdToken("bank_cst_r_strict"), "R");
  assert.equal(sideFromIdToken("bank_cc_body"), "?");
  assert.equal(sideFromIdToken("bank_cc_forceps_minor"), "?");
  // "frontal" is not an id segment — must not invent R
  assert.equal(sideFromIdToken("something_frontal"), "?");
  // Conflicting metadata is unknown, never "last token wins".
  assert.equal(sideFromIdToken("bank_fat_l_r"), "?");
  assert.equal(sideFromIdToken("seed_l_r"), "?");
});

test("describeTract ignores hostile label prose", () => {
  // Label claims right; id is left — id wins
  const d = describeTract({
    id: "bank_fat_l",
    label: "FAKE RIGHT FAT-R (should not win)",
    role: "true_fat",
  });
  assert.equal(d.side, "L");
  assert.equal(d.short, "FAT-L");
});

test("describeTract: CST full vs strict", () => {
  const full = describeTract({
    id: "bank_cst_r",
    role: "true_cst",
    nStreamlines: 1483,
  });
  const strict = describeTract({
    id: "bank_cst_r_strict",
    nStreamlines: 64,
  });
  assert.equal(full.side, "R");
  assert.equal(full.short, "CST-R");
  assert.equal(full.chip, "CST-R · n≈1483");
  assert.equal(strict.side, "R");
  assert.equal(strict.short, "CST-R (strict)");
  assert.equal(strict.chip, "CST-R (strict) · n≈64");
});

test("describeTract: bilateral / unknown side is ?", () => {
  for (const id of [
    "bank_cc_body",
    "bank_cc_forceps_minor",
    "bank_cc_forceps_major",
  ]) {
    const d = describeTract({ id });
    assert.equal(d.side, "?", id);
  }
});

test("describeTract returns frozen descriptor", () => {
  const d = describeTract({ id: "bank_slf3_r", role: "true_slf3" });
  assert.ok(Object.isFrozen(d));
  assert.throws(() => {
    // @ts-expect-error intentional mutation
    d.side = "L";
  });
});

test("describeLiveTrack: side from seed preset id only, never modeLabel", () => {
  const a = describeLiveTrack({ seedPresetId: "seed_handknob_r" });
  assert.equal(a.side, "R");
  assert.equal(a.source, "live");
  const b = describeLiveTrack({ seedPresetId: "seed_fa_l" });
  assert.equal(b.side, "L");
  // no preset → unknown
  const c = describeLiveTrack({});
  assert.equal(c.side, "?");
  assert.equal(c.short, "live");
});

test("TRACT_GROUP_META covers all groups", () => {
  for (const g of ["cst", "fat", "slf", "ifof", "uf", "cing", "or", "cc", "other"]) {
    assert.ok(TRACT_GROUP_META[g], g);
  }
});

test("describeTract: IFOF / UF / cing id sides", () => {
  assert.equal(describeTract({ id: "bank_ifof_r" }).side, "R");
  assert.equal(describeTract({ id: "bank_ifof_l" }).short, "IFOF-L");
  assert.equal(describeTract({ id: "bank_uf_l" }).side, "L");
  assert.equal(describeTract({ id: "bank_cing_r" }).group, "cing");
  assert.equal(describeTract({ id: "bank_cst_l" }).short, "CST-L");
});

test("describeTract: optic radiation L/R and Meyer", () => {
  const r = describeTract({
    id: "bank_or_r",
    role: "true_or",
    nStreamlines: 11257,
  });
  assert.equal(r.side, "R");
  assert.equal(r.short, "OR-R");
  assert.equal(r.group, "or");
  assert.equal(r.chip, "OR-R · n≈11257");
  const l = describeTract({ id: "bank_or_l", role: "true_or" });
  assert.equal(l.side, "L");
  assert.equal(l.short, "OR-L");
  const m = describeTract({ id: "bank_or_r_meyer", role: "true_or" });
  assert.equal(m.side, "R");
  assert.equal(m.short, "OR-R Meyer");
  assert.equal(m.group, "or");
  // forceps must not be mis-grouped as optic radiation (substring "or")
  const f = describeTract({ id: "bank_cc_forceps_minor", role: "true_cc" });
  assert.equal(f.group, "cc");
  assert.equal(f.short, "Forceps minor");
});

// ── manifest conformance ───────────────────────────────────────────────────

function loadCaseManifests() {
  // Real-case conformance is an explicit opt-in; ordinary npm test is portable
  // and must not discover or read a user's case directory.
  if(!process.env.TRACTLAB_CASE_MANIFEST_DIR) return [{caseId:'synthetic',man:syntheticBankManifest}];
  const casesDir = process.env.TRACTLAB_CASE_MANIFEST_DIR;
  if (!existsSync(casesDir)) return [];
  const out = [];
  for (const name of readdirSync(casesDir)) {
    const manPath = join(casesDir, name, "manifest.json");
    if (!existsSync(manPath)) continue;
    const man = JSON.parse(readFileSync(manPath, "utf8"));
    out.push({ caseId: name, manPath, man });
  }
  return out;
}

test("conformance: selected synthetic or explicitly supplied catalog has banks", () => {
  const cases = loadCaseManifests();
  assert.ok(
    cases.length > 0,
    "no manifests in selected catalog — suite cannot silently pass empty",
  );
  let bankCount = 0;
  for (const { man } of cases) {
    for (const k of Object.keys(man.inputs || {})) {
      if (k.startsWith("bank_")) bankCount++;
    }
  }
  assert.ok(
    bankCount > 0,
    "no bank_* entries in any case manifest — fail loud",
  );
});

test("conformance: id-derived side agrees with .tck path token", () => {
  const cases = loadCaseManifests();
  assert.ok(cases.length > 0, "no manifests");
  const rows = [];
  for (const { caseId, man } of cases) {
    for (const [id, meta] of Object.entries(man.inputs || {})) {
      if (!id.startsWith("bank_") || !meta || typeof meta !== "object") continue;
      const path = meta.path || "";
      const d = describeTract({
        id,
        role: meta.role,
        nStreamlines: meta.n_streamlines,
        path,
      });
      const pathSide = sideFromPathToken(path);
      rows.push({ caseId, id, path, idSide: d.side, pathSide });
      assert.equal(
        d.side,
        pathSide,
        `${caseId} ${id}: id side ${d.side} != path side ${pathSide} (${path})`,
      );
    }
  }
  assert.ok(rows.length > 0, "zero banks walked");
});

test("conformance: this case SLF L/R pair is correct", () => {
  const dL = describeTract({ id: "bank_slf3_l", role: "true_slf3" });
  const dR = describeTract({ id: "bank_slf3_r", role: "true_slf3" });
  assert.equal(dL.side, "L");
  assert.equal(dR.side, "R");
  assert.equal(dL.short, "SLF-III-L");
  assert.equal(dR.short, "SLF-III-R");
  const s1 = describeTract({ id: "bank_slf1_r", role: "true_slf1" });
  const s2 = describeTract({ id: "bank_slf2_l", role: "true_slf2" });
  assert.equal(s1.short, "SLF-I-R");
  assert.equal(s2.short, "SLF-II-L");
  assert.equal(s1.group, "slf");
  assert.equal(s2.group, "slf");
});

test("describeRecoveryTrack is never a named multi-ROI identity", () => {
  const d = describeRecoveryTrack({ radiusMm: 8, source: "filter_bank" });
  assert.equal(d.role, "recovery_perilesional");
  assert.equal(d.side, "?");
  assert.match(d.short, /Recovery/);
  assert.equal(d.source, "live");
  assert.notEqual(d.role, "true_cst");
  assert.notEqual(d.role, "true_slf3");
});

test("describeEdgeTrack always carries ASSIGNED and the assignment rule, side stays ?", () => {
  const d = describeEdgeTrack({ a: 10, b: 13, radiusMm: 4 });
  assert.equal(d.side, "?");
  assert.equal(d.role, "connectome_edge");
  assert.match(d.short, /^ASSIGNED\b/, "ASSIGNED must lead the chip/label text everywhere the layer appears");
  assert.match(d.chip, /^ASSIGNED\b/);
  assert.match(d.short, /radial search/i, "the assignment rule (radial search radius) is in the detail");
  assert.match(d.short, /≤4 mm/, "the radius value itself must be legible, not just the word radius");
  assert.equal(d.source, "live");
  assert.notEqual(d.role, "true_cst");

  // Radius unknown: still ASSIGNED, never a silent blank.
  const noRadius = describeEdgeTrack({ a: 10, b: 13 });
  assert.match(noRadius.short, /^ASSIGNED 10–13$/);
});

test("restore_edge_tubes_preserves_ASSIGNED_descriptor", () => {
  const restored = describeGeneratedLayer({
    route: "/api/connectome/edge/tubes",
    bankId: null,
    seedPresetId: null,
    params: { a: 10, b: 13, radiusMm: 4 },
  });
  assert.deepEqual(restored, describeEdgeTrack({ a: 10, b: 13, radiusMm: 4 }));
  assert.match(restored.short, /^ASSIGNED\b/);
  assert.equal(restored.side, "?");
});

test("soft bank chips are labeled soft and stay in family groups", () => {
  const d = describeTract({ id: "bank_slf3_r_soft", role: "soft_slf3" });
  assert.equal(d.short, "SLF-III-R soft");
  assert.equal(d.group, "slf");
  assert.equal(d.side, "R");
  const f = describeTract({ id: "bank_fat_r_soft", role: "soft_fat" });
  assert.equal(f.short, "FAT-R soft");
  assert.equal(f.group, "fat");
});

test("restore_recovery_preserves_recovery_descriptor", () => {
  // A reopened recovery layer keeps the recovery identity and its visual
  // contract; it is never redrawn as a generic live track.
  const recovery = describeGeneratedLayer({
    route: "/api/recovery/perilesional",
    bankId: null,
    seedPresetId: null,
    params: { radiusMm: 8, source: "filter_bank" },
  });
  assert.equal(recovery.role, "recovery_perilesional");
  assert.equal(recovery.id, "recovery_perilesional_filter_bank");
  assert.equal(recovery.short, "Recovery ≤8 mm");
  assert.equal(recovery.side, "?");
  assert.deepEqual(recovery, describeRecoveryTrack({ radiusMm: 8, source: "filter_bank" }));
  assert.notEqual(recovery.role, describeLiveTrack({}).role);

  // A filter result stays a live track.
  const live = describeGeneratedLayer({
    route: "/api/filter", bankId: null, seedPresetId: null, params: {},
  });
  assert.deepEqual(live, describeLiveTrack({}));

  // A bank-backed subset keeps the bank identity.
  const bank = { id: "bank_cst_r", label: "CST R", role: "cst" };
  assert.deepEqual(
    describeGeneratedLayer({ route: "/api/connectotomy/cut", bankId: "bank_cst_r", seedPresetId: null, params: {} }, bank),
    describeTract(bank),
  );
});
