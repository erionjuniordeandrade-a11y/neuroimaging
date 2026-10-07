import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {mkdir, writeFile} from 'node:fs/promises';

const out='/private/tmp/tractlab-visual-overview';await mkdir(out,{recursive:true});
// Serve URL is overridable so the harness can run against a case on another port.
const url=new URL(process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18993/');url.hash='test';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:870,height:550},deviceScaleFactor:2.5});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const settled=async()=>{
  await page.waitForFunction(()=>window.__tractlabTest?.pipeline?.interacting===false);
  await page.evaluate(()=>new Promise(resolve=>{
    let last=-1,stable=0;function frame(){const n=window.__tractlabTest.renderCount;stable=n===last?stable+1:0;last=n;if(stable>=6)resolve();else requestAnimationFrame(frame);}requestAnimationFrame(frame);
  }));
};
const read=()=>page.evaluate(()=>structuredClone(window.__tractlabTest));
const measurements=()=>page.locator('#mapLayers .map-layer').evaluateAll(rows=>rows.map(r=>({key:r.dataset.layerKey,value:r.querySelector('.map-value').textContent,population:r.querySelector('.map-population')?.textContent})));
const dimensions=()=>page.evaluate(()=>{
  const rect=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right};};
  return {viewport:[innerWidth,innerHeight],scene:rect('gl'),cards:rect('compactTracts'),dock:rect('anatomyDock'),cardsInsideScene:!!document.querySelector('#scene #compactTracts'),compact:document.getElementById('anatomyDock').classList.contains('anatomy-compact'),overflow:document.documentElement.scrollWidth>innerWidth};
});
const report={layouts:[],lighting:[]};
try{
  await page.goto(url.href);
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});await settled();
  report.buildId=await page.locator('meta[name="tractlab-build"]').getAttribute('content');
  report.radiusMm=Number(await page.locator('#fibre').inputValue());
  assert.equal(report.radiusMm,.22);
  const owned=await measurements();assert.equal(owned.length,3);
  for(const viewport of [{width:870,height:550},{width:870,height:507},{width:1280,height:850}]){
    await page.setViewportSize(viewport);await page.locator('#btnDefaults').click();await settled();
    const layout=await dimensions();report.layouts.push(layout);
    assert.equal(layout.cardsInsideScene,false,'measurement cards are outside the scene DOM');
    assert.equal(layout.compact,true,'Overview starts with compact anatomical context');
    assert.equal(layout.overflow,false);
    assert.ok(layout.dock.height<=136,'thumbnail context leaves height for Overview');
    assert.ok(layout.scene.height>=(viewport.width<=1000?260:600),'Overview has usable height');
    if(viewport.width<=1000)assert.ok(layout.cards.bottom<=layout.scene.y+.5,'reserved strip cannot overlap the canvas');
    await page.screenshot({path:`${out}/overview-${viewport.width}x${viewport.height}.png`,scale:'css'});
    const overview=await read();
    await page.locator('#btnInspectMRI').click();await settled();
    const inspected=await page.locator('#ccor').boundingBox();
    assert.ok(inspected.width>=500&&inspected.height>=145,`MRI inspection stays usable in a short window: ${JSON.stringify({viewport,inspected})}`);
    await page.locator('#btnPaintMode').click();
    await page.locator('#btnInspectMRI').click();await settled();
    assert.equal(await page.locator('#btnPaintMode').getAttribute('aria-pressed'),'false');
    const returning=await read();
    assert.deepEqual(returning.camera,overview.camera,'each viewport returns to its fitted Overview');
  }
  await page.locator('#anatomyResize').focus();
  // Compact layout now holds until the dock can show full controls without shrinking the images (audit F04); enlarge until it leaves compact.
  for(let i=0;i<10&&(await dimensions()).compact;i++){await page.keyboard.press('ArrowUp');await settled();}
  assert.equal((await dimensions()).compact,false);
  await page.locator('#btnPaintMode').click();await settled();
  assert.equal((await read()).viewLayout.focusedSlice,'cor','painting from a partially enlarged dock expands a usable plane');
  await page.locator('#btnPaintMode').click();await page.locator('#btnDefaults').click();await settled();
  await page.locator('#btnInspectMRI').click();await settled();
  assert.equal(await page.locator('#cax').isVisible(),false);
  const cor=await page.locator('#ccor').boundingBox();assert.ok(cor.width>=500&&cor.height>=300,'inspection exposes a usable MRI surface');
  await page.locator('#btnPaintMode').click();await page.locator('#ccor').click();
  assert.equal(await page.locator('#btnUndoPaint').isEnabled(),true);
  assert.equal((await read()).layerKeys.length,0,'a new ROI invalidates previous tract results');
  await page.locator('#btnUndoPaint').click();assert.equal(await page.locator('#btnUndoPaint').isDisabled(),true);
  await page.screenshot({path:`${out}/expanded-mri.png`,scale:'css'});
  await page.locator('#btnInspectMRI').click();await settled();
  assert.equal(await page.locator('#btnPaintMode').getAttribute('aria-pressed'),'false','returning to thumbnails exits paint mode');
  assert.equal((await dimensions()).compact,true);
  // Undo restores paint state, never stale analytic results. Reload the original
  // named sources for the matched comparison after exercising actual painting.
  await page.reload();await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});await settled();
  assert.deepEqual(await measurements(),owned,'fresh named sources retain their owned p5');
  await page.locator('#tab-display').click();
  for(const pose of ['overview','anterior']){
    if(pose==='anterior')await page.locator('[data-view="A"]').click();
    for(const colour of ['direction','distance']){
      await page.locator(colour==='direction'?'#btnDir':'#btnDist').click();await settled();
      const matched=await read();
      for(const mode of ['direct','environment']){
        await page.evaluate(mode=>window.__tractlabLightingProbe.setMode(mode),mode);
        await page.waitForFunction(mode=>window.__tractlabTest.lighting===mode,mode);await settled();
        const state=await read();
        for(const key of ['position','target','up','offset'])state.camera[key].forEach((value,i)=>assert.ok(Math.abs(value-matched.camera[key][i])<1e-8,'lighting comparison preserves the camera to numerical precision'));
        assert.equal(state.camera.autoFrame,matched.camera.autoFrame);
        assert.deepEqual(state.map.layers,matched.map.layers,'lighting comparison keeps sources and display geometry');
        assert.equal(state.commitCount,matched.commitCount);
        assert.equal(state.pipeline.focusOutlinePasses,0,'selection is carried by labels and leaders, not fibre contours');
        assert.deepEqual(await measurements(),owned);
        await page.screenshot({path:`${out}/${pose}-${colour}-${mode}.png`,scale:'css'});
        await page.locator('#gl').screenshot({path:`${out}/${pose}-${colour}-${mode}-native.png`});
        report.lighting.push({pose,colour,mode,camera:state.camera,pipeline:state.pipeline,commitCount:state.commitCount});
      }
    }
  }
  assert.deepEqual(errors,[]);
  await writeFile(`${out}/report.json`,JSON.stringify({...report,errors},null,2));
  console.log(JSON.stringify({out,layouts:report.layouts,comparisons:report.lighting.length,errors}));
}finally{await browser.close();}
