import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from 'node:child_process';

const html = readFileSync(new URL("../../viewer/index.html", import.meta.url), "utf8");

test('the complete viewer module parses, including browser wiring',()=>{
  const script=html.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
  assert.ok(script);
  const result=spawnSync(process.execPath,['--input-type=module','--check'],{input:script,encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
});

test("viewer render loop is wired through the on-demand scheduler", () => {
  assert.match(html, /from ['"]\.\/render_scheduler\.js['"]/);
  assert.match(html, /createRenderScheduler\s*\(/);
  assert.doesNotMatch(
    html,
    /requestAnimationFrame\(loop\).*renderer\.render\(scene,camera\)/s,
    "the unconditional renderer loop must not return",
  );
});

test("streamline inspector is a named, persistent viewport region", () => {
  assert.match(
    html,
    /id="streamlineInspector"[^>]*role="status"[^>]*aria-live="polite"/,
  );
  assert.match(html, /id="streamlineInspectorTitle"/);
  assert.match(html, /id="streamlineInspectorMeta"/);
});

test("warnings expose safety, interpretation, and context as distinct levels", () => {
  // #spaceWarn was folded into the provenance strip (owner ruling 8) — the
  // safety-severity level now lives on #provStrip, which carries the DWI
  // space chip.
  assert.match(html, /id="provStrip"[^>]*data-severity="safety"/);
  assert.match(
    html,
    /id="bankBlindSpot"[^>]*class="interpretation-alert"[^>]*data-severity="interpretation"[^>]*role="note"/,
  );
  assert.match(html, /id="priorPanel"[^>]*data-severity="context"/);
  assert.equal(
    (html.match(/edema ≠ absence/g) || []).length,
    1,
    "the decision-changing warning should be prominent once, not repeated as boilerplate",
  );
});
