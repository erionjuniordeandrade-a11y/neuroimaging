import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.argv.find(v=>v.startsWith('--url='))?.slice(6);
if(!base)throw new Error('Pass the isolated preview --url explicitly');
const url=new URL('atlas.html?test=1&profile=teaching',base).href;
const browser=await chromium.launch({headless:true});
const evidence={};await mkdir('output/playwright',{recursive:true});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],apiRequests=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))apiRequests.push(r.url());});
  await page.goto(url);await page.waitForFunction(()=>window.__atlasTest?.ready);
  const state=()=>page.evaluate(()=>window.__atlasTest);
  assert.deepEqual((await state()).vertices,[32492,32492]);assert.equal((await state()).playing,false);
  await page.screenshot({path:'output/playwright/atlas-desktop.png'});
  await page.locator('#atlasArea').selectOption('R:8');
  assert.match(await page.locator('#pickedName').textContent(),/Right · area 4/);
  await page.locator('#atlasHemisphere').selectOption('R');assert.equal((await state()).hemisphere,'R');
  // Multi-bundle picker (2026-09-07): chips toggle in pick order; the default CST stays selected.
  if((await state()).bundles.length)await page.locator('#pathwayClear').click();assert.deepEqual((await state()).bundles,[]);
  await page.locator('.bundle-chip[data-bundle="FAT_R"]').click();assert.deepEqual((await state()).bundles,['FAT_R']);
  await page.locator('.bundle-chip[data-bundle="AF_R"]').click();assert.deepEqual((await state()).bundles,['FAT_R','AF_R']);
  await page.locator('#pathwayFilter').fill('arcu');
  assert.equal(await page.locator('.bundle-chip:visible').count(),2,'filter keeps only the two arcuate chips');
  await page.locator('#pathwayFilter').fill('');
  await page.locator('.bundle-chip[data-bundle="FAT_R"]').click();assert.deepEqual((await state()).bundles,['AF_R']);
  await page.locator('#tracePlay').click();const before=(await state()).time;
  await page.waitForTimeout(350);assert((await state()).time>before);
  await page.locator('#tracePlay').click();const paused=(await state()).time;
  await page.waitForTimeout(350);assert.equal((await state()).time,paused);
  await page.locator('#lessonPlay').click();assert.equal((await state()).lesson.status,'playing');
  await page.locator('#atlasArea').selectOption('L:74');assert.equal((await state()).lesson.status,'paused');
  const nav=page.locator('#lessonNext');await nav.click();assert.equal((await state()).lesson.step,1);
  await page.reload();await page.waitForFunction(()=>window.__atlasTest?.ready);
  assert.equal((await state()).lesson.step,1);assert.equal((await state()).playing,false);
  await page.locator('#lessonSelect').selectOption('interoception');
  assert.deepEqual((await state()).bundles,[]);assert.equal((await state()).deepVisible,true);
  await page.locator('#lessonNext').click();assert.match(await page.locator('#pickedName').textContent(),/area PoI2/);
  await page.screenshot({path:'output/playwright/atlas-interoception.png'});
  await page.locator('#lessonExplore').click();assert.equal((await state()).lesson.status,'inactive');
  await page.locator('#lessonProfile').selectOption('presenter');assert.equal((await state()).playing,false);
  assert.equal(await page.locator('#tracePlay').isDisabled(),true);
  // Malicious metadata remains literal text; no live DOM nodes or handlers.
  const inert=await page.evaluate(async()=>{const {setHudMarkup}=await import('./ui_text.js');
    const target=document.createElement('div');document.body.append(target);
    setHudMarkup(target,'<b class="metric" onclick="window.__injected=1">5</b> <img src=x onerror="window.__injected=1"><span style="position:fixed">name</span>');
    const result={images:target.querySelectorAll('img').length,handlers:target.querySelectorAll('[onclick],[onerror],[style]').length,text:target.textContent,injected:!!window.__injected};target.remove();return result;});
  assert.equal(inert.images,0);assert.equal(inert.handlers,0);assert.equal(inert.injected,false);assert.match(inert.text,/<img/);
  assert.deepEqual(errors,[]);assert.deepEqual(apiRequests,[]);
  evidence.desktop={...await state(),inertMetadata:inert,apiRequests};
  for(const width of [320,768,1024]){
    await page.setViewportSize({width,height:900});
    await page.goto(url);await page.waitForFunction(()=>window.__atlasTest?.ready);
    const layout=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,
      canvas:document.querySelector('canvas').getBoundingClientRect().toJSON(),lesson:document.querySelector('#lessonPanel').getBoundingClientRect().toJSON()}));
    assert.equal(layout.overflow,false,`no horizontal overflow at ${width}`);
    assert(layout.canvas.height>180,`usable canvas at ${width}`);
    if(width===320)assert(layout.lesson.y>=layout.canvas.bottom,'narrow lesson follows canvas');
    await page.screenshot({path:`output/playwright/atlas-${width}.png`,fullPage:width===320});
    evidence[`width${width}`]=layout;
  }
  const reduced=await browser.newPage({viewport:{width:1200,height:900},reducedMotion:'reduce'});
  await reduced.goto(url);await reduced.waitForFunction(()=>window.__atlasTest?.ready);
  await reduced.locator('#lessonPlay').click();
  const t=await reduced.evaluate(()=>window.__atlasTest.time);await reduced.waitForTimeout(300);
  assert.equal(await reduced.evaluate(()=>window.__atlasTest.time),t);assert(await reduced.locator('#tracePlay').isDisabled());
  evidence.reducedMotion=true;await reduced.close();
  const bad=await browser.newPage();
  await bad.route('**/atlas/surface-labels.bin',async route=>{const response=await route.fetch();const bytes=Buffer.from(await response.body());bytes[0]^=1;await route.fulfill({response,body:bytes});});
  await bad.goto(url);await bad.waitForFunction(()=>document.querySelector('#atlasLoading')?.textContent.includes('integrity failed'));
  assert.equal(await bad.locator('canvas').count(),0);assert.equal(await bad.evaluate(()=>!!window.__atlasTest?.ready),false);
  evidence.corruptAssetRefused=true;await bad.close();
  console.log(JSON.stringify({passed:true,...evidence},null,2));
  await writeFile('output/playwright/atlas-verification.json',JSON.stringify(evidence,null,2)+'\n');
}finally{await browser.close();}
