import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {visibleUiMetrics} from './ui_metrics.mjs';

const baseline=process.argv.includes('--baseline');
const url=new URL(process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18993/');url.hash='test';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1200,height:600},deviceScaleFactor:2.5});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const snapshot=()=>page.evaluate(()=>{
  const rect=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};};
  return {workspace:rect('viewWorkspace'),scene:rect('scene'),dock:rect('anatomyDock'),legend:rect('mapScale'),
    mpr:[...document.querySelectorAll('.mpr canvas')].map(c=>({width:c.clientWidth,height:c.clientHeight})),
    cards:[...document.querySelectorAll('#mapLayers .map-layer')].map(r=>({key:r.dataset.layerKey,value:r.querySelector('.map-value').textContent})),
    map:window.__tractlabTest.map};
});
try{
  await page.goto(url.href);await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
  await page.waitForFunction(()=>!window.__tractlabTest.pipeline.interacting);
  await page.waitForTimeout(220); // Finish the existing 140 ms loading-to-ready opacity transition before capture.
  const initial=await snapshot();
  await page.screenshot({path:`/private/tmp/tractlab-composition-${baseline?'before':'after'}.png`,scale:'css'});
  await page.locator('#tab-display').click(); // cold start opens Tracts; veil controls live in Display
  await page.locator('#hullVeilSolid').click();
  await page.waitForTimeout(220);
  await page.screenshot({path:`/private/tmp/tractlab-hull-${baseline?'before':'after'}.png`,scale:'css'});
  await page.locator('#hullVeilGhost').click();
  console.log(JSON.stringify({stage:baseline?'baseline':'refined',...initial}));
  if(!baseline){
    assert.ok(initial.dock.height/initial.workspace.height<=.42,'short-window anatomy dock reserves most height for 3D');
    assert.ok(initial.legend.height<=120,'default legend is compact');
    // Plan 011 made Overview a compact 88-104 px context strip (canvas + 64 px plane label); inspection enlarges it.
    assert.ok(initial.mpr.every(p=>p.height>=80&&p.width>=160),'short-window slices retain usable space');
    const fat=page.locator('#mapLayers .map-layer').filter({hasText:'FAT-R'});
    assert.equal(await fat.locator('.map-population').isVisible(),true);
    assert.match(await fat.locator('.map-population').textContent(),/20,000.*27,528/);
    assert.equal(await page.locator('#provenanceBar #provChipPreflight').isVisible(),true);
    assert.equal(await page.locator('#provenanceBar #provChipOperating').isVisible(),true);
    await page.locator('#provChipPreflight').click();assert.equal(await page.locator('#provCard').isVisible(),true);await page.keyboard.press('Escape');
    const separator=page.locator('#anatomyResize');await separator.focus();await page.keyboard.press('ArrowUp');
    assert.ok((await snapshot()).dock.height>initial.dock.height+15,'keyboard resizes dock');
    const box=await separator.boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2,box.y-25,{steps:4});await page.mouse.up();
    const resized=await snapshot();assert.ok(resized.dock.height>initial.dock.height+35,'pointer resizes dock');
    await page.locator('#btnSlices').click();await page.locator('#btnSlices').click();
    assert.ok(Math.abs((await snapshot()).dock.height-resized.dock.height)<2,'toggle preserves selected dock size');
    assert.deepEqual((await snapshot()).map.target,initial.map.target,'resizing preserves lesion target');
    assert.deepEqual((await snapshot()).cards,initial.cards,'resizing preserves owned measurements');
    await page.locator('#btnDefaults').click();assert.ok(Math.abs((await snapshot()).dock.height-initial.dock.height)<2);
    await page.locator('#btnInspectMRI').click(); // underlay controls are hidden in the compact Overview strip (plan 011)
    await page.locator('#btnDEC').click();
    assert.equal(await page.locator('#sliceColourKey').isVisible(),true);
    assert.doesNotMatch(await page.locator('#mapScale').textContent(),/Slice DEC/);
    await page.locator('#btnInspectMRI').click(); // back to Overview
    await page.locator('#btnResetView').focus();await page.keyboard.press('1');
    assert.match(await page.locator('#mapScale').textContent(),/Fibre orientation/);
    await page.keyboard.press('3');
    const metrics=await page.evaluate(visibleUiMetrics);
    assert.deepEqual(metrics.contrastFailures,[]);assert.deepEqual(metrics.smallText,[]);assert.deepEqual(metrics.smallTargets,[]);
    await page.setViewportSize({width:1280,height:850});
    await page.waitForTimeout(220);
    // Plan 011: Overview keeps a compact context strip; painting room lives in the expanded inspection state.
    // The dock fraction chosen at 1200x600 is retained across the resize (plan 011); Defaults restores the desktop compact strip, Inspect MRI expands it.
    await page.locator('#btnDefaults').click();await page.waitForTimeout(220);
    await page.locator('#btnInspectMRI').click();await page.waitForTimeout(220);
    const expanded=(await snapshot()).mpr.filter(p=>p.width>0); // inspection shows one large plane; hidden planes measure 0
    assert.ok(expanded.length>0&&expanded.every(p=>p.height>=145),'desktop slices retain painting room');
    await page.locator('#btnInspectMRI').click();await page.waitForTimeout(220);
    await page.screenshot({path:'/private/tmp/tractlab-composition-desktop.png',scale:'css'});
    await page.setViewportSize({width:480,height:700});
    await page.locator('#provChipPreflight').click();
    const provenance=await page.evaluate(()=>({card:document.getElementById('provCard').getBoundingClientRect().bottom,bar:document.getElementById('provenanceBar').getBoundingClientRect().top}));
    assert.ok(provenance.card<=provenance.bar+1,'wrapped provenance footer does not sit behind its popover');
    await page.keyboard.press('Escape');
  }
  assert.deepEqual(errors,[]);
}finally{await browser.close();}
