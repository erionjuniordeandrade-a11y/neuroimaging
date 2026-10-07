// SCENE GRAMMAR v2 smoke: injects a declarative v2 scene through the test-only
// window.__atlasTest.applyScene(scene) hook (available only under ?test=1) and asserts
// the multi-bundle/ghost, multi-region/highlight, camera-tween and reduced-motion
// contracts implemented in lesson_scene.js / atlas_scene.js / atlas_app.js.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.argv.find(v=>v.startsWith('--url='))?.slice(6);
if(!base)throw new Error('Pass the isolated preview --url explicitly');
const url=new URL('atlas.html?test=1&profile=teaching',base).href;
const browser=await chromium.launch({headless:true});
const evidence={};await mkdir('output/playwright',{recursive:true});

// L:8 and L:74 are both valid HCP-MMP1 left-hemisphere parcel ids (L:74 is already exercised
// by atlas_smoke.mjs), so this scene is guaranteed to resolve without needing case data.
const testScene={
  side:'L',
  bundles:['CST','ML'],
  ghost:['DRTT'],
  regions:[{id:8,hemi:'L'},{id:74,hemi:'L'}],
  camera:{view:'superior',zoom:1.3,tweenMs:400},
  durationSec:12,
  deepRegions:['THA-lh'],
  deep:true,
};

try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.waitForFunction(()=>window.__atlasTest?.ready);
  const state=()=>page.evaluate(()=>window.__atlasTest);

  const before=await state();
  assert.equal(await page.evaluate(()=>typeof window.__atlasTest.applyScene),'function',
    'applyScene is exposed under ?test=1');

  await page.evaluate(scene=>window.__atlasTest.applyScene(scene),testScene);
  await page.waitForTimeout(120); // mid-flight: tweenMs is 400
  const mid=await state();
  await page.waitForTimeout(500); // let the 400ms tween finish
  const after=await state();
  await page.screenshot({path:'output/playwright/grammar-v2.png'});

  // Two bundles visible + one ghosted with lower opacity.
  assert.deepEqual([...after.bundles].sort(),['CST_L','ML_L'],`two primary bundles visible, observed ${JSON.stringify(after.bundles)}`);
  assert.deepEqual(after.ghostBundles,['DRTT_L'],`one bundle ghosted, observed ${JSON.stringify(after.ghostBundles)}`);
  assert(after.bundleAlpha.DRTT_L<after.bundleAlpha.CST_L,
    `ghosted bundle has a lower render alpha than a primary one, observed ${JSON.stringify(after.bundleAlpha)}`);
  assert(after.bundleAlpha.DRTT_L<after.bundleAlpha.ML_L,'ghosted bundle dimmer than the other primary too');

  // Two regions highlighted: the first is picked (#pickedName/select), the second is the secondary tier.
  assert.deepEqual(after.selected,{hemi:'L',id:8},`first region is picked, observed ${JSON.stringify(after.selected)}`);
  assert.deepEqual(after.highlighted,[{hemi:'L',id:74}],`second region is secondary-highlighted, observed ${JSON.stringify(after.highlighted)}`);
  assert.match(await page.locator('#pickedName').textContent(),/Left/,'the picked name reflects the focus region');

  // Camera position changed after the tween, and was animating (not an instant jump) along the way.
  assert.notDeepEqual(mid.camera,before.camera,'camera has started moving mid-tween');
  assert.notDeepEqual(mid.camera,after.camera,'camera has not yet reached the final position mid-tween');
  assert.notDeepEqual(after.camera,before.camera,'camera position changed once the tween finished');

  // Deep highlight subset + deep visibility.
  assert.equal(after.deepVisible,true,'deep structures are visible for this scene');
  assert.deepEqual(after.deepHighlight,['THA-lh'],'requested deep-structure subset recorded');

  assert.deepEqual(errors,[],`no page errors, observed ${JSON.stringify(errors)}`);
  evidence.desktop={before,mid,after};

  // Reduced motion jumps instantly: the camera position right after applyScene (before any
  // waiting) must already equal the position after a full wait — no multi-frame lerp occurred.
  const reduced=await browser.newPage({viewport:{width:1200,height:900},reducedMotion:'reduce'});
  await reduced.goto(url);await reduced.waitForFunction(()=>window.__atlasTest?.ready);
  await reduced.evaluate(scene=>window.__atlasTest.applyScene(scene),testScene);
  const immediate=await reduced.evaluate(()=>window.__atlasTest.camera);
  await reduced.waitForTimeout(500);
  const later=await reduced.evaluate(()=>window.__atlasTest.camera);
  assert.deepEqual(immediate,later,`reduced motion jumps instantly (no further camera movement), observed ${JSON.stringify({immediate,later})}`);
  assert.notDeepEqual(immediate,before.camera,'reduced-motion jump still reached a different camera position than the starting one');
  evidence.reducedMotion={immediate,later};
  await reduced.close();

  console.log(JSON.stringify({passed:true,...evidence},null,2));
  await writeFile('output/playwright/grammar-v2.json',JSON.stringify(evidence,null,2)+'\n');
}finally{await browser.close();}
