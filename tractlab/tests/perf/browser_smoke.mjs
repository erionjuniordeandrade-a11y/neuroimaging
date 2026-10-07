#!/usr/bin/env node
/**
 * TractLab browser acceptance gates.
 *
 * Requires the loopback server to be running. The default is headed because
 * the FPS gate measures the real compositor; use --headless --skip-fps for a
 * structural smoke run in CI.
 *
 *   npm run test:browser
 *   npm run test:browser -- --url http://127.0.0.1:8770/index.html
 *   npm run test:browser -- --headless --skip-fps
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { chromium } from "playwright";

const DEFAULT_URL = "http://127.0.0.1:8770/index.html";
const FPS_MIN = 100;
const WAIT_MS = 120_000;

function arg(name, fallback) {
  const eq = process.argv.find((x) => x.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")
    ? process.argv[i + 1]
    : fallback;
}

function hasFlag(name) {
  return process.argv.includes(name);
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

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Decode the RGBA PNG emitted by Playwright without another native package. */
function decodePngRgba(buffer) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  assert.equal(buffer.subarray(0, 8).compare(sig), 0, "screenshot is not PNG");
  let off = 8;
  let width = 0;
  let height = 0;
  let depth = 0;
  let colorType = 0;
  let interlace = 0;
  const idat = [];
  while (off < buffer.length) {
    const n = buffer.readUInt32BE(off);
    const type = buffer.toString("ascii", off + 4, off + 8);
    const data = buffer.subarray(off + 8, off + 8 + n);
    off += 12 + n;
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      depth = data[8];
      colorType = data[9];
      interlace = data[12];
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
      break;
    }
  }
  assert.equal(depth, 8, "only 8-bit PNG screenshots are supported");
  assert.ok(colorType === 2 || colorType === 6, "expected RGB/RGBA PNG screenshot");
  assert.equal(interlace, 0, "interlaced PNG screenshot is unsupported");
  const bpp = colorType === 6 ? 4 : 3;
  const stride = width * bpp;
  const raw = inflateSync(Buffer.concat(idat));
  const rows = Buffer.alloc(height * stride);
  let src = 0;
  for (let y = 0; y < height; y += 1) {
    const filter = raw[src++];
    const row = rows.subarray(y * stride, (y + 1) * stride);
    const prev = y ? rows.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x += 1) {
      const left = x >= bpp ? row[x - bpp] : 0;
      const up = prev ? prev[x] : 0;
      const upLeft = prev && x >= bpp ? prev[x - bpp] : 0;
      const value = raw[src++];
      if (filter === 0) row[x] = value;
      else if (filter === 1) row[x] = (value + left) & 255;
      else if (filter === 2) row[x] = (value + up) & 255;
      else if (filter === 3) row[x] = (value + Math.floor((left + up) / 2)) & 255;
      else if (filter === 4) row[x] = (value + paeth(left, up, upLeft)) & 255;
      else throw new Error(`unsupported PNG filter ${filter}`);
    }
  }
  const pixels = Buffer.alloc(height * width * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const si = y * stride + x * bpp;
      const di = (y * width + x) * 4;
      pixels[di] = rows[si];
      pixels[di + 1] = rows[si + 1];
      pixels[di + 2] = rows[si + 2];
      pixels[di + 3] = bpp === 4 ? rows[si + 3] : 255;
    }
  }
  return { width, height, pixels };
}

function screenshotNonBackground(png) {
  const { width, height, pixels } = decodePngRgba(png);
  const bg = [11, 13, 17]; // viewer scene background (#0b0d11)
  let nonBackground = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (
      pixels[i + 3] > 0 &&
      Math.abs(pixels[i] - bg[0]) +
        Math.abs(pixels[i + 1] - bg[1]) +
        Math.abs(pixels[i + 2] - bg[2]) > 12
    ) {
      nonBackground += 1;
    }
  }
  return { width, height, nonBackground, fraction: nonBackground / (width * height) };
}

async function state(page) {
  return page.evaluate(() => structuredClone(window.__tractlabTest));
}

async function waitForBank(page, id, short, side, minCommit) {
  const before = await state(page);
  // Collapsible bank groups hide non-default families — open ancestor <details>
  await page.locator(`#${id}`).evaluate((el) => {
    let n = el.parentElement;
    while (n) {
      if (n.tagName === "DETAILS") n.open = true;
      n = n.parentElement;
    }
    el.scrollIntoView({ block: "center" });
  });
  await page.locator(`#${id}`).click({ force: true });
  // Multi-select bank loads (975860c) intentionally do NOT bump the live-track
  // request token (_requestSeq) — per-bank generations guard them instead. The
  // "a new response committed after this click" invariant is therefore proven
  // by commitCount strictly increasing past the pre-click snapshot, not by
  // requestSeq.
  await page.waitForFunction(
    ({ id, short, side, beforeCommit, minCommit }) => {
      const s = window.__tractlabTest;
      return (
        s &&
        s.settled &&
        s.commitCount > beforeCommit &&
        s.commitCount >= minCommit &&
        s.descriptor?.id === id.slice(5) &&
        s.descriptor?.short === short &&
        (side === "" || s.descriptor?.side === side) &&
        s.lineCount > 0 &&
        s.pointsPerLine > 0
      );
    },
    { id, short, side, beforeCommit: before.commitCount ?? 0, minCommit },
    { timeout: WAIT_MS },
  );
  return state(page);
}

async function measureFps(page, seconds = 3) {
  return page.evaluate(async (seconds) => {
    const gaps = [];
    let last = performance.now();
    const start = last;
    while (performance.now() - start < seconds * 1000) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const now = performance.now();
      gaps.push(now - last);
      last = now;
    }
    const sorted = [...gaps].sort((a, b) => a - b);
    const at = (p) => sorted[Math.floor(sorted.length * p)] || Infinity;
    return {
      frames: gaps.length,
      fps: gaps.length / ((last - start) / 1000),
      medianFrameMs: at(0.5),
      p95FrameMs: at(0.95),
      worstFrameMs: sorted[sorted.length - 1] || Infinity,
    };
  }, seconds);
}

async function readMpr(page) {
  await page.locator("#advLive").evaluate((el) => {
    el.open = true;
  });
  return page.locator("#cax").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let checksum = 0;
    let nonZero = 0;
    let alphaOk = true;
    let gray = true;
    for (let i = 0; i < data.length; i += 4) {
      checksum += data[i];
      if (data[i] !== 0) nonZero += 1;
      if (data[i + 3] !== 255) alphaOk = false;
      if (data[i] !== data[i + 1] || data[i] !== data[i + 2]) gray = false;
    }
    return { width: canvas.width, height: canvas.height, checksum, nonZero, alphaOk, gray };
  });
}

async function main() {
  const rawUrl = arg("url", process.env.TRACTLAB_URL || DEFAULT_URL);
  const url = new URL(rawUrl);
  url.hash = "test";
  const headed = !hasFlag("--headless");
  const skipFps = hasFlag("--skip-fps");
  // BLOCKED≠PASS: when this case is expected to ship signed atlas QC, hard-fail
  // if priors are absent so render gates cannot soft-skip green.
  const requirePriors = hasFlag("--require-priors");
  if (!headed && !skipFps) {
    throw new Error("FPS gate is headed-only; use a headed run or pass --skip-fps explicitly");
  }

  const executablePath = browserExecutable();
  const launch = { headless: !headed };
  if (executablePath) launch.executablePath = executablePath;
  const browser = await chromium.launch(launch);
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
  const origin = url.origin;
  const external = [];
  const consoleErrors = [];
  const pageErrors = [];
  const requestFailures = [];
  const httpErrors = [];
  const payloads = [];
  const payloadChecks = [];

  page.on("request", (request) => {
    const target = request.url();
    if (target.startsWith("data:") || target.startsWith("blob:")) return;
    let sameOrigin = false;
    try { sameOrigin = new URL(target).origin === origin; } catch {}
    if (!sameOrigin) external.push(target);
  });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => requestFailures.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText || "failed"}`));
  page.on("response", (response) => {
    if (response.status() >= 400) httpErrors.push(`${response.status()} ${response.url()}`);
    if (response.url().endsWith("/api/bank/load") && response.status() === 200) {
      payloadChecks.push((async () => {
        try {
          const body = await response.body();
          const lineCount = Number(response.headers()["x-linecount"] || 0);
          const pointsPerLine = Number(response.headers()["x-pointsperline"] || 0);
          const bytes = lineCount * pointsPerLine * 3 * 4;
          assert.ok(bytes > 0 && body.byteLength >= bytes, "bank geometry payload is truncated");
          const values = new Float32Array(body.buffer, body.byteOffset, bytes / 4);
          let min = Infinity;
          let max = -Infinity;
          let nonZero = 0;
          for (const value of values) {
            assert.ok(Number.isFinite(value), "bank geometry contains non-finite values");
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (value !== 0) nonZero += 1;
          }
          assert.ok(nonZero > values.length / 100, "bank geometry is effectively empty");
          assert.ok(max - min > 1, "bank geometry is degenerate");
          assert.ok(min >= -1000 && max <= 1000, "bank geometry is outside plausible world bounds");
          payloads.push({ lineCount, pointsPerLine, bytes: body.byteLength, min, max });
        } catch (error) {
          httpErrors.push(`bank payload: ${error.message}`);
        }
      })());
    }
  });

  try {
    await page.goto(url.href, { waitUntil: "load", timeout: WAIT_MS });
    await page.waitForSelector("#bank_bank_cst_r", { timeout: WAIT_MS });
    await page.waitForFunction(() => {
      const s = window.__tractlabTest;
      return s?.settled && s.descriptor?.id === "bank_cst_r" && s.lineCount > 0;
    }, undefined, { timeout: WAIT_MS });

    const initial = await state(page);
    assert.equal(
      await page.locator("#connectotomyPanel").count(),
      1,
      "C1 panel present on the lesion case",
    );
    const c1Text = await page.locator("#connectotomyPanel").innerText();
    assert.match(c1Text, /cuts \d+ of \d+ bank streamlines/);
    assert.doesNotMatch(c1Text, /ASSIGNED|Yeo|Schaefer|margin|at risk/i);
    assert.equal(
      await page.locator("#bankGroups #connectotomyPanel").count(),
      0,
      "connectotomy panel must sit outside tract chips",
    );
    assert.equal(initial.descriptor.short, "CST-R", "default bank descriptor");
    assert.equal(initial.descriptor.side, "R", "default bank laterality");
    assert.equal(initial.materialType, "MeshStandardMaterial", "tube material");
    assert.ok(initial.positionCount > 0, "tube position attribute is non-empty");

    // Exercise the request-token race: L is delayed before it reaches the
    // server, then R is dispatched while the UI is busy. Only R may commit.
    const raceBase = initial.commitCount;
    const delayLeft = async (route) => {
      let body = {};
      try { body = route.request().postDataJSON(); } catch {}
      // Hold L before it reaches the server long enough for R to commit. This
      // avoids manufacturing a server-side 409 while still making L stale.
      if (body.bankId === "bank_slf3_l") await new Promise((resolve) => setTimeout(resolve, 5000));
      await route.continue();
    };
    await page.route("**/api/bank/load", delayLeft);
    await page.evaluate(() => {
      for (const id of ["bank_bank_slf3_l", "bank_bank_slf3_r"]) {
        const el = document.querySelector(`#${id}`);
        if (!el) return;
        let n = el.parentElement;
        while (n) {
          if (n.tagName === "DETAILS") n.open = true;
          n = n.parentElement;
        }
      }
      document.querySelector("#bank_bank_slf3_l")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    await page.evaluate(() => document.querySelector("#bank_bank_slf3_r")?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    await page.waitForFunction(
      (base) => window.__tractlabTest?.settled && window.__tractlabTest?.descriptor?.id === "bank_slf3_r" && window.__tractlabTest.commitCount === base + 1,
      raceBase,
      { timeout: WAIT_MS },
    );
    await new Promise((resolve) => setTimeout(resolve, 5500));
    const race = await state(page);
    assert.equal(race.descriptor.side, "R", "stale delayed L response must not relabel R");
    assert.equal(race.commitCount, raceBase + 1, "stale delayed response must not commit geometry twice");
    await page.unroute("**/api/bank/load", delayLeft);

    // Fresh page for deterministic sequential chip and pixel gates.
    await page.reload({ waitUntil: "load", timeout: WAIT_MS });
    await page.waitForFunction(() => window.__tractlabTest?.settled && window.__tractlabTest?.descriptor?.id === "bank_cst_r", undefined, { timeout: WAIT_MS });
    const left = await waitForBank(page, "bank_bank_slf3_l", "SLF3-L", "L", 2);
    assert.equal(left.descriptor.side, "L", "SLF3-L descriptor side");
    assert.match(await page.locator("#hudTitle").textContent(), /SLF3-L/);
    assert.equal(left.materialType, "MeshStandardMaterial");
    // positionCount is the committed layer's geometry; lineCount is the
    // multi-select total across layers — assert per-layer (layerLineCount).
    assert.equal(left.positionCount, left.layerLineCount * left.pointsPerLine * 8, "tube vertex count (committed layer)");
    const right = await waitForBank(page, "bank_bank_slf3_r", "SLF3-R", "R", 3);
    assert.equal(right.descriptor.side, "R", "SLF3-R descriptor side");
    assert.match(await page.locator("#hudTitle").textContent(), /SLF3-R/);
    assert.equal(right.materialType, "MeshStandardMaterial");

    // MPR oracle before prior overlays (contours change pixels)
    const mpr = await readMpr(page);
    assert.equal(mpr.width, 185);
    assert.equal(mpr.height, 185);
    assert.equal(mpr.checksum, 671943, "axial MPR checksum at k=54");
    assert.equal(mpr.nonZero, 8883, "axial MPR non-zero pixels at k=54");
    assert.equal(mpr.alphaOk, true, "MPR alpha");
    assert.equal(mpr.gray, true, "b0 MPR remains grayscale");

    // ── Atlas prior gates (ADR-0001 / ADR-0002) ──────────────────────────
    // (1) fail-closed: empty /api/priors → panel absent
    await page.route("**/api/priors", async (route) => {
      const u = route.request().url();
      // only the list endpoint — not /volume or /mesh
      if (/\/api\/priors\/?$/.test(new URL(u).pathname)) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ priors: [], available: false, disclaimer: "test" }),
        });
        return;
      }
      await route.continue();
    });
    await page.reload({ waitUntil: "load", timeout: WAIT_MS });
    await page.waitForFunction(
      () => window.__tractlabTest?.settled && window.__tractlabTest?.descriptor?.id === "bank_cst_r",
      undefined,
      { timeout: WAIT_MS },
    );
    assert.equal(
      await page.locator("#priorPanel").evaluate((el) => el.classList.contains("visible")),
      false,
      "unapproved/empty priors must hide #priorPanel",
    );
    // panel must not live inside bank chip groups
    assert.equal(
      await page.locator("#bankGroups #priorPanel").count(),
      0,
      "prior panel must be physically outside #bankGroups",
    );
    await page.unroute("**/api/priors");

    // (2–4) approved case: panel visible, separate, ghost mesh not tubes, provenance
    await page.reload({ waitUntil: "load", timeout: WAIT_MS });
    await page.waitForFunction(
      () => window.__tractlabTest?.settled && window.__tractlabTest?.descriptor?.id === "bank_cst_r",
      undefined,
      { timeout: WAIT_MS },
    );
    const health = await page.evaluate(async () => {
      const h = await (await fetch("/api/health")).json();
      const p = await (await fetch("/api/priors")).json();
      return { hasPriors: h.hasPriors, priorCount: h.priorCount, n: (p.priors || []).length };
    });
    if (health.n > 0) {
      await page.waitForSelector("#priorPanel.visible", { timeout: WAIT_MS });
      assert.equal(
        await page.locator("#bankGroups #priorPanel").count(),
        0,
        "prior panel separate from tract chips",
      );
      // Population panel ships collapsed (lowest evidence tier) — expand it the
      // way a user would before reaching for a prior chip.
      const priorSummary = page.locator("#priorDetails > summary");
      if ((await priorSummary.count()) && !(await page.locator("#priorDetails[open]").count())) {
        await priorSummary.click();
      }
      const firstPrior = page.locator("#priorGroups button.prior").first();
      assert.ok(await firstPrior.count(), "at least one prior chip");
      await firstPrior.click();
      await page.waitForFunction(
        () => {
          // ghost meshes live in scene userData.kind === 'atlas-prior'
          // exposed via a probe on window for the harness
          return window.__tractlabPriorProbe?.activeCount > 0;
        },
        undefined,
        { timeout: WAIT_MS },
      );
      const priorProbe = await page.evaluate(() => window.__tractlabPriorProbe);
      assert.ok(priorProbe.activeCount >= 1, "prior ghost mesh present");
      assert.equal(priorProbe.anyTubeMaterial, false, "priors must not use tube/vertexColor material");
      assert.equal(priorProbe.anyDirectionRGB, false, "priors must not use direction-RGB vertex colors");
      assert.match(
        await page.locator("#hudMeta").textContent(),
        /POPULATION ATLAS/,
        "provenance chip text in HUD when prior visible",
      );
      // Two provenance badges exist (panel summary + nested parcel panel);
      // the summary badge is the always-visible one, even collapsed.
      assert.ok(
        await page.locator(".prior-prov").first().isVisible(),
        "POPULATION ATLAS badge visible in prior panel",
      );
    } else if (requirePriors) {
      assert.fail(
        `--require-priors: server reports hasPriors=false / priorCount=${health.priorCount} ` +
          `(n=${health.n}). Render gates 2–4 would soft-skip — BLOCKED, not PASS. ` +
          `Sign atlas_prior_qc after full-SyN prep, restart server, re-run.`,
      );
    } else {
      console.warn(
        "BLOCKED (not PASS): skip live prior render gates — hasPriors=false. " +
          "Re-run with --require-priors on the signed-QC case to hard-fail this path.",
      );
    }

    // C1 fail-closed: no lesion cavity → panel not in the DOM
    await page.route("**/api/connectotomy", async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/connectotomy") {
        await route.fulfill({ status: 404, contentType: "text/plain", body: "no lesion cavity" });
        return;
      }
      await route.continue();
    });
    await page.reload({ waitUntil: "load", timeout: WAIT_MS });
    await page.waitForFunction(
      () => window.__tractlabTest?.settled && window.__tractlabTest?.descriptor?.id === "bank_cst_r",
      undefined,
      { timeout: WAIT_MS },
    );
    assert.equal(
      await page.locator("#connectotomyPanel").count(),
      0,
      "C1 panel absent when /api/connectotomy is 404",
    );
    await page.unroute("**/api/connectotomy");

    const glPng = await page.locator("#gl").screenshot({ path: "/tmp/tractlab-browser-gl.png" });
    const pixels = screenshotNonBackground(glPng);
    assert.ok(pixels.fraction > 0.01, `WebGL screenshot is blank (${pixels.nonBackground}/${pixels.width * pixels.height})`);

    const fps = skipFps ? null : await measureFps(page, Number(arg("seconds", "3")));
    if (fps) assert.ok(fps.fps >= FPS_MIN, `FPS ${fps.fps.toFixed(1)} < ${FPS_MIN}`);

    await Promise.all(payloadChecks);
    assert.deepEqual(external, [], "external network requests");
    assert.deepEqual(requestFailures, [], "failed requests");
    assert.deepEqual(pageErrors, [], "page errors");
    // The deliberate L/R race fires concurrent /api/bank/load requests; the
    // server's single-flight contract 409s one and the client handles it
    // ("Bank busy"). Chrome still logs the non-2xx fetch, so tolerate 409s on
    // /api/bank/load ONLY — a 409 anywhere else (track, recovery) still fails.
    const allowed409 = (e) => e.startsWith("409 ") && e.endsWith("/api/bank/load");
    const allowedC1closed = (e) => e.startsWith("404 ") && e.includes("/api/connectotomy") && !e.includes("/cut");
    const unexpectedHttp = httpErrors.filter((e) => !allowed409(e) && !allowedC1closed(e));
    const n409Allowed = httpErrors.length - unexpectedHttp.length;
    const unexpectedConsole = [...consoleErrors];
    for (let i = 0; i < n409Allowed; i += 1) {
      const j = unexpectedConsole.findIndex((t) =>
        t.includes("status of 409") || (t.includes("status of 404") && t.includes("connectotomy")),
      );
      if (j >= 0) unexpectedConsole.splice(j, 1);
    }
    assert.deepEqual(unexpectedConsole, [], "console errors");
    assert.deepEqual(unexpectedHttp, [], "HTTP errors (409 bank race + C1 404 fail-closed allowed)");
    assert.ok(payloads.length >= 3, `expected bank payload observations, saw ${payloads.length}`);

    console.log(JSON.stringify({
      url: url.href,
      headed,
      initial,
      race,
      left,
      right,
      mpr,
      webgl: pixels,
      fps,
      payloads: payloads.slice(-5),
      external,
      consoleErrors,
      pageErrors,
      requestFailures,
      httpErrors,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(`browser_smoke FAIL: ${error.stack || error.message}`);
  process.exitCode = 1;
});
