#!/usr/bin/env node
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";

const rawUrl = process.argv.find((value) => value.startsWith("--url="))?.slice(6)
  || process.env.TRACTLAB_URL
  ;
if (!rawUrl) throw new Error('Pass an isolated generated preview --url explicitly');
const waitMs = Number(
  process.argv.find((value) => value.startsWith("--timeout="))?.slice(10) || 120_000,
);
const executablePath = [
  process.env.TRACTLAB_BROWSER,
  chromium.executablePath(),
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean).find((candidate) => existsSync(candidate));

function modeUrl(mode = "") {
  const url = new URL(rawUrl);
  url.searchParams.set("test", "1");
  if (mode === "teaching") url.searchParams.set("teaching", "1");
  return url;
}

async function waitForLoadedMode(page, mode) {
  await page.waitForFunction(
    (expected) => {
      const state = window.__tractlabTest;
      return state?.settled && state?.lineCount > 0 && state?.displayMode === expected;
    },
    mode,
    { timeout: waitMs },
  );
}

async function idleDelta(page, milliseconds = 350) {
  const before = await page.evaluate(() => window.__tractlabTest.renderCount);
  await page.waitForTimeout(milliseconds);
  const after = await page.evaluate(() => window.__tractlabTest.renderCount);
  return after - before;
}

const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
try {
  // 1200 px reproduces the compact projector/laptop header where the longer
  // Teaching provenance chip previously escaped the fixed chrome row.
  const page = await browser.newPage({ viewport: { width: 1200, height: 746 } });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(modeUrl().href, { waitUntil: "load", timeout: waitMs });
  await waitForLoadedMode(page, "clinical");

  const clinical = await page.evaluate(() => ({
    state: structuredClone(window.__tractlabTest),
    modeChipHidden: document.querySelector("#provChipDisplay").hidden,
    modeButtons: [...document.querySelectorAll("#displayModeGroup [role=tab]")]
      .map((button) => [button.textContent.trim(), button.getAttribute("aria-selected")]),
  }));
  assert.equal(clinical.modeChipHidden, true);
  assert.equal(clinical.state.presentationGlowVisible, false);
  assert.equal(clinical.state.constellationVisible, false);
  assert.equal(clinical.state.tubeOpacity, 1);
  assert.deepEqual(clinical.modeButtons, [
    ["Clinical", "true"], ["Presenter", "false"], ["Teaching", "false"],
  ]);
  assert.ok(await idleDelta(page) <= 1, "Clinical must settle to an idle renderer");

  await page.locator("#btnModePresenter").click();
  await page.waitForFunction(
      () => window.__tractlabTest?.displayMode === "presenter"
      && !window.__tractlabTest?.presentationGlowVisible
      && window.__tractlabTest?.constellationVisible
      && window.__tractlabTest?.tubeOpacity < 0.9,
    undefined,
    { timeout: waitMs },
  );
  const presenter = await page.evaluate(() => ({
    chip: document.querySelector("#provChipDisplay").textContent.trim().toLowerCase(),
    hidden: document.querySelector("#provChipDisplay").hidden,
    presentation: new URL(location.href).searchParams.get("presentation"),
    teaching: new URL(location.href).searchParams.get("teaching"),
    profile: new URL(location.href).searchParams.get("profile"),
    evidenceDisabled: [...document.querySelectorAll("#evidenceLabel + .seg button")]
      .some((button) => button.disabled),
  }));
  assert.equal(presenter.hidden, false);
  assert.equal(presenter.chip, "presentation · evidence dimmed");
  assert.equal(presenter.presentation, null);
  assert.equal(presenter.profile, "presenter");
  assert.equal(presenter.teaching, null);
  assert.equal(presenter.evidenceDisabled, false);
  await page.waitForTimeout(100);
  assert.ok(await idleDelta(page) <= 1, "Presenter must not introduce a continuous render loop");

  // Exercise the Uint32 index path (>65,535 displayed vertices), not only the
  // small default CST bank, before creating the Teaching overlays.
  const beforeLargeBank = await page.evaluate(() => window.__tractlabTest.commitCount);
  await page.locator("#bank_bank_or_l").click({ force: true });
  await page.waitForFunction(
    (before) => window.__tractlabTest?.commitCount > before
      && window.__tractlabTest?.descriptor?.id === "bank_or_l"
      && window.__tractlabTest?.lineCount * window.__tractlabTest?.pointsPerLine > 65_535,
    beforeLargeBank,
    { timeout: waitMs },
  );

  await page.locator("#btnModeTeaching").click();
  await page.waitForFunction(
    () => window.__tractlabTest?.displayMode === "teaching"
      && window.__tractlabTest?.visibleTeachingTraceCount > 0
      && window.__tractlabTest?.traceAnimating
      && window.__tractlabTest?.tubeOpacity < 0.35,
    undefined,
    { timeout: waitMs },
  );
  const teachingStart = await page.evaluate(() => window.__tractlabTest.renderCount);
  await page.waitForTimeout(250);
  const teaching = await page.evaluate(() => ({
    renderDelta: window.__tractlabTest.renderCount,
    note: document.querySelector("#teachingModeBoundary").textContent,
    viewportNoteHidden: document.querySelector("#teachingViewportNote")?.hidden ?? null,
    viewportNote: document.querySelector("#teachingViewportNote")?.textContent ?? "",
    chipContained: (() => {
      const header = document.querySelector("#chrome")?.getBoundingClientRect();
      const chip = document.querySelector("#provChipDisplay")?.getBoundingClientRect();
      return Boolean(header && chip && chip.bottom <= header.bottom + 0.5);
    })(),
    presentation: new URL(location.href).searchParams.get("presentation"),
    teaching: new URL(location.href).searchParams.get("teaching"),
    profile: new URL(location.href).searchParams.get("profile"),
  }));
  assert.ok(teaching.renderDelta - teachingStart > 3, "Teaching should intentionally animate");
  assert.match(teaching.note, /symmetric/i);
  assert.match(teaching.note, /does not show axonal direction/i);
  assert.match(teaching.note, /neural conduction/i);
  assert.equal(teaching.viewportNoteHidden, false);
  assert.match(teaching.viewportNote, /not direction or conduction/i);
  assert.equal(teaching.chipContained, true, "Teaching provenance chip must stay inside the header");
  assert.equal(teaching.presentation, null);
  assert.equal(teaching.teaching, null);
  assert.equal(teaching.profile, "teaching");

  await page.locator('#btnTracePause').click();
  await page.waitForFunction(()=>!window.__tractlabTest.traceAnimating);
  await page.waitForTimeout(100);
  assert.ok(await idleDelta(page)<=1, 'explicit pause settles the renderer');
  await mkdir('output/playwright',{recursive:true});
  await page.screenshot({path:'output/playwright/case-teaching-final.png'});

  await page.locator("#btnHideLowSupport").click();
  await page.waitForFunction(
    () => window.__tractlabTest?.teachingEvidenceMode === 1,
    undefined,
    { timeout: 5_000 },
  );
  await page.locator("#btnModeClinical").click();
  await page.waitForFunction(
    () => window.__tractlabTest?.displayMode === "clinical"
      && !window.__tractlabTest?.traceAnimating,
    undefined,
    { timeout: 5_000 },
  );
  assert.equal(await page.locator("#btnHideLowSupport").getAttribute("aria-pressed"), "true");
  await page.waitForTimeout(100);
  assert.ok(await idleDelta(page) <= 1, "returning to Clinical must release the animation loop");
  await page.locator('#btnModePresenter').click();
  await page.waitForTimeout(100);
  await page.screenshot({path:'output/playwright/case-presenter-final.png'});
  assert.deepEqual(pageErrors, []);
  await page.close();

  const reducedPage = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const reducedErrors = [];
  reducedPage.on("pageerror", (error) => reducedErrors.push(error.message));
  await reducedPage.emulateMedia({ reducedMotion: "reduce" });
  await reducedPage.goto(modeUrl("teaching").href, { waitUntil: "load", timeout: waitMs });
  await waitForLoadedMode(reducedPage, "teaching");
  await reducedPage.waitForFunction(
    () => window.__tractlabTest?.teachingTraceCount > 0,
    undefined,
    { timeout: waitMs },
  );
  const reduced = await reducedPage.evaluate(() => ({
    animating: window.__tractlabTest.traceAnimating,
    note: document.querySelector("#teachingModeBoundary").textContent,
  }));
  assert.equal(reduced.animating, false);
  assert.match(reduced.note, /paused for reduced motion/i);
  await reducedPage.waitForTimeout(100);
  assert.ok(await idleDelta(reducedPage) <= 1, "reduced-motion Teaching must settle to idle");
  assert.deepEqual(reducedErrors, []);

  console.log(JSON.stringify({ clinical, presenter, teaching, reduced }, null, 2));
} finally {
  await browser.close();
}
