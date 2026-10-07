// Behavioral regressions for the ten findings in HEADLESS-FULL-AUDIT-2026-09-11.
// Uses the real case/server in a separate headless Chrome session. No app mutation.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const option=(key,fallback)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)||fallback;
const base=option('url','http://127.0.0.1:18993/');
const out=option('out','/private/tmp/tractlab-audit-fixes-20260911');
const selected=option('checks','drawer,rail,rounding,mri,semantics,envelope,bias,cancel,focus').split(',');
await mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,channel:'chrome'});
const context=await browser.newContext({viewport:{width:1179,height:557},deviceScaleFactor:Number(option('dpr','2.5'))});
const page=await context.newPage();page.setDefaultTimeout(15000);
const report={startedAt:new Date().toISOString(),browser:browser.version(),checks:[],pageErrors:[]};
page.on('pageerror',e=>report.pageErrors.push(e.message));
const state=()=>page.evaluate(()=>structuredClone(window.__tractlabTest));
const settle=()=>page.waitForTimeout(300);
const shot=async name=>{await settle();await page.screenshot({path:`${out}/${name}.png`,scale:'css'});};
const ready=()=>page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
const fresh=async()=>{await page.setViewportSize({width:1179,height:557});await page.goto(base+'#test');await page.reload();await ready();await settle();};
const panel=async name=>{
  if(await page.locator('#side').evaluate(e=>e.getBoundingClientRect().right<=0))await page.locator('#btnRail').click();
  await page.locator('#tab-'+name).click();
};
const open=async id=>{if(!await page.locator('#'+id).evaluate(e=>e.open))await page.locator('#'+id+' > summary').click();};
const choose=async id=>{
  await panel('tracts');await open('secBanks');await page.locator('#bankFilter').fill(id);await page.locator('#bank_'+id).click();
  await page.waitForFunction(id=>window.__tractlabTest.layerKeys?.includes('bank:'+id)&&!document.getElementById('bank_'+id).classList.contains('loading'),id,{timeout:120000});await settle();
};
const focus=async id=>{await page.locator(`#mapLayers [data-layer-key="bank:${id}"] .map-focus`).click();await settle();};
const measures=()=>page.locator('#mapLayers .map-layer').evaluateAll(es=>Object.fromEntries(es.map(e=>[e.dataset.layerKey,e.querySelector('.map-value').textContent])));
const dimensions=()=>page.evaluate(()=>{
  const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right};};
  return {dock:rect(document.getElementById('anatomyDock')),compact:document.getElementById('anatomyDock').classList.contains('anatomy-compact'),
    canvases:['cax','ccor','csag'].map(id=>rect(document.getElementById(id))),side:rect(document.getElementById('side')),nav:rect(document.getElementById('railNav')),
    panel:rect(document.querySelector('.rail-panel:not([hidden])')),mask:getComputedStyle(document.querySelector('.rail-panel:not([hidden])')).maskImage};
});
const check=async(name,fn)=>{
  if(!selected.includes(name))return;
  const row={name};const t=Date.now();
  try{await fresh();row.evidence=await fn();row.status='pass';}
  catch(e){row.status='fail';row.error=e.stack;await shot('FAIL-'+name).catch(()=>{});row.state=await state().catch(()=>null);}
  row.ms=Date.now()-t;report.checks.push(row);console.log(JSON.stringify({name,status:row.status,ms:row.ms,error:row.error?.split('\n')[0]}));
  await writeFile(out+'/regressions.json',JSON.stringify(report,null,2));
};
try{
  report.start=await (await context.request.get(base+'api/bootstrap')).json();
  assert.equal(report.start.runtimeStatus,'ok','Use a stable running build');assert.equal(report.start.busy,false);
  await check('drawer',async()=>{
    await panel('display');await page.setViewportSize({width:870,height:507});await settle();
    await page.locator('#chromeHelp').focus();const radius=await page.locator('#fibre').inputValue();const ids=[];
    for(let i=0;i<45;i++){
      await page.keyboard.press('Tab');const active=await page.evaluate(()=>({id:document.activeElement.id,inside:document.getElementById('side').contains(document.activeElement)}));
      ids.push(active.id);assert.equal(active.inside,false,'Closed drawer receives keyboard focus: '+active.id);
    }
    await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#fibre').inputValue(),radius);
    await page.locator('#btnRail').click();assert.equal(await page.locator('#side').evaluate(e=>e.inert),false);
    await page.locator('#tab-display').click();await page.locator('#btnDir').focus();await page.keyboard.press('Escape');
    assert.equal(await page.locator('#btnRail').evaluate(e=>document.activeElement===e),true,'Escape returns focus to drawer opener');
    await page.locator('#compactTracts button').nth(1).click();
    assert.equal(await page.locator('#side').evaluate(e=>e.inert),false,'Compact measurement chip must open an interactive drawer');
    await page.locator('#tab-display').click();await page.locator('#btnDir').focus();await page.keyboard.press('Escape');
    await page.setViewportSize({width:1179,height:557});await settle();
    assert.equal(await page.locator('#side').evaluate(e=>e.inert),false,'Desktop sidebar must be interactive after resize');
    await page.locator('#tab-display').click();await shot('drawer-desktop-restored');return {ids};
  });
  await check('rail',async()=>{
    const samples=[];
    const verify=async count=>{
      await panel('display');const d=await dimensions();samples.push({count,...d});
      assert.ok(d.nav.y>=d.side.y&&d.nav.bottom<=d.side.bottom+1,'Rail tabs must remain on screen');
      assert.ok(d.panel.height>=120&&d.panel.bottom<=d.side.bottom+1,`${count} banks leave only ${d.panel.height}px for controls`);
      if(count===3)assert.equal(await page.locator('#mapLayers .map-layer').evaluateAll(es=>es.every(e=>{
        const r=e.getBoundingClientRect(),p=document.getElementById('caseMap').getBoundingClientRect();return r.top>=p.top&&r.bottom<=p.bottom;
      })),true,'The default trio must all be visible without scrolling');
      assert.equal(d.mask,'none','Interactive rail content must not fade under an opacity mask');await shot('rail-'+count);
    };
    await verify(3);
    for(const id of ['bank_cst_r_strict','bank_slf3_r_soft'])await choose(id);await verify(5);
    const banks=(await (await context.request.get(base+'api/banks')).json()).banks;
    const loaded=new Set((await state()).layerKeys.map(k=>k.replace('bank:','')));
    for(const bank of banks.filter(b=>!loaded.has(b.id)).slice(0,5))await choose(bank.id);await verify(10);
    // Last card and first control must both be reachable with ordinary scrolling/focus.
    await page.locator('#mapLayers .map-focus').last().focus();
    await page.locator('#tab-display').click();await page.locator('#btnDir').focus();
    const b=await page.locator('#btnDir').boundingBox();assert.ok(b.y>=48&&b.y+b.height<=557);return samples;
  });
  await check('rounding',async()=>{
    const pending=page.waitForResponse(r=>r.url().endsWith('/api/bank/load')&&r.request().postDataJSON()?.bankId==='bank_slf3_r_soft');
    await choose('bank_slf3_r_soft');const r=await pending;const h=await r.allHeaders();
    const value=await page.locator('#mapLayers [data-layer-key="bank:bank_slf3_r_soft"] .map-value').textContent();
    assert.equal(value,h['x-clearancep5display']+' mm');assert.ok(h['x-clearancesummary'].includes(value));
    await shot('rounding-authority');return {raw:h['x-clearancep5'],display:h['x-clearancep5display'],value};
  });
  await check('mri',async()=>{
    const baseline=await dimensions(),samples=[baseline];await page.locator('#anatomyResize').focus();
    for(let i=0;i<8;i++){
      await page.keyboard.press('ArrowUp');await settle();const d=await dimensions(),prev=samples.at(-1);
      samples.push(d);for(let j=0;j<3;j++)assert.ok(d.canvases[j].height>=prev.canvases[j].height-1,`Growing dock ${prev.dock.height}→${d.dock.height} shrinks slice ${j}: ${prev.canvases[j].height}→${d.canvases[j].height}`);
    }
    await page.locator('#btnDefaults').click();await settle();
    for(const axis of ['ax','cor','sag']){
      await page.locator('#expand-'+axis).click();await settle();await shot('inspect-'+axis);
      await page.locator('#expand-'+axis).click();await settle();const d=await dimensions();
      for(let j=0;j<3;j++)assert.ok(Math.abs(d.canvases[j].height-baseline.canvases[j].height)<=1,`Returning from ${axis} changed slice ${j} height`);
    }
    await shot('mri-overview-restored');return samples;
  });
  await check('semantics',async()=>{
    await panel('display');await page.locator('#btnDir').click();await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#btnSolid').getAttribute('aria-selected'),'true');
    assert.equal(await page.locator('#btnSolid').evaluate(e=>e===document.activeElement),true);
    assert.deepEqual(await page.locator('#btnDir,#btnSolid,#btnDist').evaluateAll(es=>es.map(e=>e.tabIndex)),[-1,0,-1]);
    await page.keyboard.press('ArrowLeft');assert.equal(await page.locator('#btnDir').getAttribute('aria-selected'),'true');
    await page.locator('#chromeHelp').click();await page.keyboard.press('Escape');
    assert.equal(await page.locator('#chromeHelp').getAttribute('aria-expanded'),'false');
    assert.equal(await page.locator('#howto').evaluate(e=>e.hidden),true);
    await panel('tracts');await open('priorDetails');
    const prior=page.locator('#priorPanel button.chip').first();
    assert.equal(await prior.getAttribute('aria-pressed'),'false');await prior.click();await settle();
    assert.equal(await prior.getAttribute('aria-pressed'),'true');await prior.click();await settle();assert.equal(await prior.getAttribute('aria-pressed'),'false');
    const unlabelled=await page.locator('#priorPanel button.chip').evaluateAll(es=>es.filter(e=>!['true','false'].includes(e.getAttribute('aria-pressed'))).map(e=>e.id));
    assert.deepEqual(unlabelled,[]);
    const network=page.locator('#parcelNetworks button').first();await network.click();await settle();
    assert.equal(await network.getAttribute('aria-pressed'),'true');await network.click();assert.equal(await network.getAttribute('aria-pressed'),'false');
    await panel('tools');await open('advLive');
    for(const group of [['roleSeed','roleAnd','roleOr','roleNot'],['denSparse','denNormal','denDense']]){
      for(const id of group){await page.locator('#'+id).click();
        assert.deepEqual(await page.locator(group.map(x=>'#'+x).join(',')).evaluateAll(es=>es.map(e=>({id:e.id,pressed:e.getAttribute('aria-pressed')}))),
          group.map(x=>({id:x,pressed:String(x===id)})));
      }
    }
    return {unlabelled};
  });
  await check('envelope',async()=>{
    await focus('bank_slf3_r');await panel('tools');await open('researchTools');
    const pending=page.waitForResponse(r=>r.url().endsWith('/api/margin'));await page.locator('#btnMargin').click();
    await (await pending).finished();await page.waitForFunction(()=>document.getElementById('btnMargin').getAttribute('aria-checked')==='true');await settle();
    const radius=await page.locator('#marginMm').inputValue();let legend=await page.locator('#mapScale').textContent();
    assert.match(legend,new RegExp(`SLF-III-R envelope.*${radius} mm.*display only`));await shot('owned-envelope');
    // Editing a radius cannot relabel geometry before the new response is accepted.
    let release,entered;const held=new Promise(r=>release=r),arrived=new Promise(r=>entered=r);
    await page.route('**/api/margin',async route=>{entered();await held;await route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Audit fixture: unavailable envelope'})});});
    const number=page.locator('#marginMm').locator('..').locator('input[type=number]');
    await number.fill('10');await number.press('Tab');await arrived;
    await panel('display');await page.locator('#btnDir').click();
    assert.match(await page.locator('#mapScale').textContent(),new RegExp(`envelope.*${radius} mm`),'Pending radius must not relabel the accepted mesh');
    release();await page.waitForFunction(()=>document.getElementById('btnMargin').getAttribute('aria-checked')==='false');
    assert.doesNotMatch(await page.locator('#mapScale').textContent(),/envelope/);await page.unroute('**/api/margin');
    await focus('bank_fat_r');await settle();legend=await page.locator('#mapScale').textContent();
    // Changing focus must never relabel an existing SLF envelope as FAT.
    assert.ok(!/FAT-R envelope/.test(legend));
    return {radius,legend};
  });
  await check('bias',async()=>{
      const values=await measures(),samples=[];
    for(const id of ['bank_cst_r','bank_fat_r','bank_slf3_r']){
      await focus(id);await panel('tools');await open('researchTools');const before=await state();
      await page.locator('#displayNearLesion').click();
      await page.waitForFunction(n=>window.__tractlabTest.commitCount>=n+3&&window.__tractlabTest.layerKeys.length===3,before.commitCount,{timeout:120000});await settle();
      const after=await state();assert.equal(after.focusLayerKey,'bank:'+id,'Display-only reload changed focus');
      assert.equal(after.exportIdentity.bankId,id,'Export no longer targets intended bank');assert.deepEqual(await measures(),values);samples.push({id,export:after.exportIdentity});
    }
    return samples;
  });
  await check('cancel',async()=>{
    await panel('tools');await open('advLive');await page.locator('#presetRow button').first().click();await page.locator('#denSparse').click();
    const pending=page.waitForResponse(r=>r.url().endsWith('/api/cancel'));await page.locator('#track').click();await page.locator('#cancel').click();
    const response=await pending;await response.finished();assert.ok(response.ok());
    await page.waitForFunction(()=>!document.getElementById('track').disabled&&!/Cancelling/.test(document.getElementById('status').textContent));
    const status=await page.locator('#status').textContent();assert.match(status,/cancelled|canceled/i);
    await page.waitForTimeout(1200);assert.equal(await page.locator('#status').textContent(),status,'Aborted work must not repaint status');return {status};
  });
  await check('focus',async()=>{
    const values=await measures(),samples=[];
    for(const id of ['bank_cst_r','bank_fat_r','bank_slf3_r']){
      await focus(id);const s=await state();assert.equal(s.focusLayerKey,'bank:'+id);
      assert.equal(s.pipeline.focusOutlinePasses,0,'Focus must not draw per-fibre pale contours');
      assert.equal(await page.locator(`#mapLayers [data-layer-key="bank:${id}"] .map-focus`).getAttribute('data-focused'),'true');
      assert.ok(await page.locator('#mapLabels .is-focused').count()>0);assert.deepEqual(await measures(),values);
      samples.push({id,pipeline:s.pipeline});await shot('focus-'+id);
    }
    return samples; // Contour quality is additionally inspected from these matched screenshots.
  });
}finally{
  report.end=await (await context.request.get(base+'api/bootstrap')).json().catch(e=>({error:e.message}));
  report.stable=report.start?.buildId===report.end.buildId&&report.start?.bootId===report.end.bootId;
  report.finishedAt=new Date().toISOString();await writeFile(out+'/regressions.json',JSON.stringify(report,null,2));await browser.close();
}
assert.equal(report.stable,true,'Runtime changed during verification');assert.deepEqual(report.pageErrors,[]);
const failures=report.checks.filter(r=>r.status!=='pass');assert.equal(failures.length,0,failures.map(r=>r.name+': '+r.error?.split('\n')[0]).join('\n'));
console.log(`PASS ${report.checks.length} audit regression groups; evidence ${out}`);
