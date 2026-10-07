#!/usr/bin/env node
/**
 * Derivation-copy honesty smoke (ADR-0007 / CLAUDE.md: "do not apply an
 * uncorrected-case warning to a corrected derivation").
 *
 * Requires the loopback server running on the demo case (kind rpe_pair):
 *   TRACTLAB_MANIFEST="$PWD/cases/demo-leipzig-sub-010005/manifest.json" \
 *     ./serve.sh 18996 --background
 *   node tests/perf/derivation_copy_smoke.mjs
 *   ./serve.sh 18996 --stop
 *
 * Asserts /api/derivation reports kind=rpe_pair for the demo case, then that
 * no element's visible text or title attribute in the loaded page contains
 * "no reverse-PE" — every such site must have been derived from the served
 * kind/floor_label instead (viewer/derivation_copy.js).
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { chromium } from "playwright";

const DEFAULT_URL = "http://127.0.0.1:18996/index.html";
const WAIT_MS = 60_000;

function arg(name, fallback) {
  const eq = process.argv.find((x) => x.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")
    ? process.argv[i + 1]
    : fallback;
}

function browserExecutable() {
  const candidates = [
    process.env.TRACTLAB_BROWSER,
    chromium.executablePath(),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p));
}

async function main() {
  const rawUrl = arg("url", process.env.TRACTLAB_URL || DEFAULT_URL);
  const url = new URL(rawUrl);
  url.hash = "test";

  const derivationRes = await fetch(new URL("/api/derivation", url.origin));
  assert.ok(derivationRes.ok, `/api/derivation fetch failed: HTTP ${derivationRes.status}`);
  const derivation = await derivationRes.json();
  assert.equal(
    derivation.kind,
    "rpe_pair",
    `demo case must be kind=rpe_pair for this smoke to be meaningful (got ${derivation.kind})`,
  );
  assert.match(
    String(derivation.floor_label || ""),
    /reverse-PE corrected/i,
    "served floor_label must read reverse-PE corrected for kind=rpe_pair",
  );

  const executablePath = browserExecutable();
  const launch = { headless: true };
  if (executablePath) launch.executablePath = executablePath;
  const browser = await chromium.launch(launch);
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });

  try {
    await page.goto(url.href, { waitUntil: "load", timeout: WAIT_MS });
    await page.waitForFunction(() => window.__tractlabTest?.settled, undefined, { timeout: WAIT_MS });
    // loadDerivationFloorLabel() runs during load(); #floorLabel's markup
    // default ("Derivation unavailable") is non-empty too, so wait for the
    // actual served text rather than a "length > 0" proxy.
    await page.waitForFunction(
      () => /reverse-PE corrected/i.test(document.getElementById("floorLabel")?.textContent || ""),
      undefined,
      { timeout: WAIT_MS },
    );

    const floorLabelText = await page.locator("#floorLabel").textContent();
    assert.match(floorLabelText || "", /reverse-PE corrected/i, "#floorLabel must reflect the corrected derivation");

    const offenders = await page.evaluate(() => {
      const hits = [];
      const needle = /no reverse-PE/i;
      if (needle.test(document.body.innerText || "")) {
        hits.push({ where: "document.body.innerText" });
      }
      for (const el of document.querySelectorAll("[title]")) {
        const title = el.getAttribute("title") || "";
        if (needle.test(title)) {
          hits.push({ where: "title", id: el.id || null, tag: el.tagName, title });
        }
      }
      return hits;
    });

    assert.deepEqual(
      offenders,
      [],
      `found "no reverse-PE" copy on a kind=rpe_pair (corrected) case: ${JSON.stringify(offenders)}`,
    );

    // Spot-check the two known derived copy sites landed the corrected text.
    const researchTitle = await page.locator("#provChipResearch").getAttribute("title");
    assert.match(researchTitle || "", /reverse-PE corrected/i, "research chip title must derive from kind");

    const recoveryNote = await page.locator("#recoveryFloorNote").textContent();
    assert.match(recoveryNote || "", /reverse-PE corrected/i, "recovery radius caption must derive from floor_label");

    // T1 button title: the demo case's delta_qc IS signed, so "few-mm
    // uncertainty" is a permitted claim here — but it must be driven by the
    // served delta_qc_signed flag, not a blanket rpe_pair assumption.
    assert.equal(derivation.delta_qc_signed, true, "demo case must be delta_qc_signed for this spot-check");
    const t1Btn = page.locator("#btnT1");
    if (await t1Btn.count()) {
      await page.waitForFunction(() => !document.getElementById("btnT1")?.disabled, undefined, { timeout: WAIT_MS });
      const t1Title = await t1Btn.getAttribute("title");
      assert.match(t1Title || "", /reverse-PE corrected/i, "T1 title must derive from kind");
      assert.match(t1Title || "", /few-mm uncertainty/i, "signed delta QC permits the few-mm uncertainty claim");
      assert.doesNotMatch(t1Title || "", /no reverse-PE/i);
    }

    console.log(JSON.stringify({
      url: url.href,
      derivationKind: derivation.kind,
      deltaQcSigned: derivation.delta_qc_signed,
      floorLabel: floorLabelText,
      researchTitle,
      recoveryNote,
      offenders,
    }, null, 2));
    console.log("derivation_copy_smoke PASS");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(`derivation_copy_smoke FAIL: ${error.stack || error.message}`);
  process.exitCode = 1;
});
