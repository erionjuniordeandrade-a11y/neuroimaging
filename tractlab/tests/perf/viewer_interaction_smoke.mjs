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
const url = new URL(rawUrl);
url.searchParams.set("test", "1");
const executablePath = [
  process.env.TRACTLAB_BROWSER,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  chromium.executablePath(),
].filter(Boolean).find((candidate) => existsSync(candidate));

const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const consoleErrors = [];
const pageErrors = [];
const httpErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("response", (response) => {
  if (response.status() >= 400) httpErrors.push(`${response.status()} ${response.url()}`);
});

try {
  await page.goto(url.href, { waitUntil: "load", timeout: waitMs });
  await page.waitForFunction(
    () => window.__tractlabTest?.mapReady && window.__tractlabTest?.settled && window.__tractlabTest?.lineCount > 0 && !window.__tractlabTest?.pipeline?.interacting,
    undefined,
    { timeout: waitMs },
  );

  const warningHierarchy = await page.evaluate(() => {
    const safety = document.querySelector("#provStrip");
    const interpretation = document.querySelector("#bankBlindSpot");
    const context = document.querySelector("#priorPanel");
    return {
      levels: [safety?.dataset.severity, interpretation?.dataset.severity, context?.dataset.severity],
      safetyBackground: safety ? getComputedStyle(safety).backgroundColor : null,
      interpretationBackground: interpretation ? getComputedStyle(interpretation).backgroundColor : null,
      interpretationRole: interpretation?.getAttribute("role"),
    };
  });
  assert.deepEqual(warningHierarchy.levels, ["safety", "interpretation", "context"]);
  assert.equal(warningHierarchy.interpretationRole, "note");
  assert.notEqual(warningHierarchy.safetyBackground, warningHierarchy.interpretationBackground);

  const beforeIdle = await page.evaluate(() => window.__tractlabTest?.renderCount);
  assert.ok(Number.isInteger(beforeIdle), "test seam exposes renderer.render count");
  await page.waitForTimeout(500);
  const afterIdle = await page.evaluate(() => window.__tractlabTest?.renderCount);
  assert.ok(
    afterIdle - beforeIdle <= 1,
    `idle scene rendered ${afterIdle - beforeIdle} extra frames in 500 ms`,
  );

  await page.setViewportSize({ width: 1390, height: 890 });
  await page.waitForFunction(
    (count) => window.__tractlabTest?.renderCount > count,
    afterIdle,
    { timeout: 5_000 },
  );

  const target = await page.evaluate(() => window.__tractlabPickProbe?.projectTarget?.());
  assert.ok(target && Number.isFinite(target.x) && Number.isFinite(target.y), "pick target available");
  await page.mouse.click(target.x, target.y);
  await page.waitForFunction(
    () => !document.querySelector("#streamlineInspector")?.hidden,
    undefined,
    { timeout: 5_000 },
  );
  const selection = await page.locator("#streamlineInspector").evaluate((element) => ({
    text: element.textContent,
    displayIndex: element.dataset.displayIndex,
    sourceIndex: element.dataset.sourceIndex,
    sourcePopulation: element.dataset.sourcePopulation,
  }));
  assert.match(selection.text, /displayed #\d+/i);
  assert.match(selection.text, /source #\d+/i);
  assert.match(selection.text, /bank:/i);
  assert.match(selection.displayIndex, /^\d+$/);
  assert.match(selection.sourceIndex, /^\d+$/);
  assert.match(selection.sourcePopulation, /^bank:/);

  // Walk focus/remove through the actual controls, then pin without canvas input.
  await page.locator('#bank_bank_cst_l').click();
  await page.waitForFunction(()=>document.querySelector('#inspectLayer')?.value==='bank:bank_cst_l');
  await page.locator('#bank_bank_cst_l').click();
  await page.waitForFunction(()=>document.querySelector('#inspectLayer')?.value==='bank:bank_cst_r');
  assert.match(await page.locator('#counts').textContent(), /CST-R.*900/);
  await page.locator('.keyboard-inspector summary').click();
  await page.locator('#inspectRow').fill('2');
  await page.locator('#inspectPin').focus();
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#streamlineInspector').getAttribute('data-display-index'),'1');
  await page.locator('#btnModeClinical').focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#btnModePresenter').getAttribute('aria-selected'),'true');
  await page.keyboard.press('ArrowLeft');
  assert.equal(await page.locator('#btnModeClinical').getAttribute('aria-selected'),'true');
  await mkdir('output/playwright',{recursive:true});
  await page.screenshot({path:'output/playwright/case-inspector-final.png'});
  const layout=[];
  for(const width of [320,768,1024]){
    await page.setViewportSize({width,height:900});
    const boxes=await page.evaluate(()=>{
      const rect=id=>document.getElementById(id).getBoundingClientRect().toJSON();
      return {width:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth,
        canvas:rect('gl'),hud:rect('hud'),inspector:rect('streamlineInspector')};
    });
    assert.equal(boxes.overflow,false,`no horizontal overflow at ${width}`);
    assert.ok(boxes.canvas.height>180 && boxes.canvas.width>180);
    assert.ok(boxes.inspector.bottom<=boxes.hud.top || boxes.hud.right<=boxes.inspector.left,
      `inspector and HUD do not overlap at ${width}`);
    layout.push(boxes);
    await page.screenshot({path:`output/playwright/case-${width}-final.png`});
  }

  assert.deepEqual(pageErrors, []);
  const optionalAbsent = new Set([
    "/api/volume/t1",
    "/api/volume/fa",
    "/api/volume/dec",
    "/api/volume/lesion",
    "/api/connectotomy",
    "/api/surface/lesion",
  ]);
  const allowedHttp = httpErrors.filter((entry) => {
    const match = entry.match(/^404 (.+)$/);
    return match && optionalAbsent.has(new URL(match[1]).pathname);
  });
  assert.deepEqual(
    httpErrors.filter((entry) => !allowedHttp.includes(entry)),
    [],
    "unexpected HTTP errors",
  );
  assert.equal(consoleErrors.length, allowedHttp.length, "one browser console 404 per absent optional resource");
  assert.ok(consoleErrors.every((entry) => /status of 404/.test(entry)), "no non-404 console errors");
  console.log(JSON.stringify({ url: url.href, warningHierarchy, beforeIdle, afterIdle, selection, keyboard:true, focusRemoval:true, layout }, null, 2));
} catch (error) {
  let pageState = null;
  try {
    pageState = await page.evaluate(() => ({
      test: window.__tractlabTest || null,
      status: document.querySelector("#status")?.textContent || null,
      counts: document.querySelector("#counts")?.textContent || null,
    }));
  } catch {}
  console.error(JSON.stringify({ pageState, consoleErrors, pageErrors, httpErrors }, null, 2));
  throw error;
} finally {
  await browser.close();
}
