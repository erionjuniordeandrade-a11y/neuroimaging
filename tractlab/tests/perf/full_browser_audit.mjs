// Audit the running application without changing source or the user's browser.
// Writes local evidence; a failed stage is recorded and later stages continue.
import {chromium} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {visibleUiMetrics} from './ui_metrics.mjs';

const option=(key,fallback)=>process.argv.find(a=>a.startsWith('--'+key+'='))?.split('=').slice(1).join('=')||fallback;
const base=option('url','http://127.0.0.1:18993/');
const out=option('out','/private/tmp/tractlab-headless-audit-20260911');
const phases=option('phases','overview,mri,catalog,overlays,research,controls,regressions,responsive').split(',');
await mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,channel:'chrome'});
const dpr=Number(option('dpr','2.5'));
const context=await browser.newContext({viewport:{width:1179,height:557},deviceScaleFactor:dpr,acceptDownloads:true});
const page=await context.newPage();page.setDefaultTimeout(12000);
const report={startedAt:new Date().toISOString(),browser:browser.version(),dpr,phases,stages:[],responses:[],pageErrors:[],console:[],requestFailures:[],externalRequests:[]};
let activeStage='startup';
page.on('pageerror',e=>report.pageErrors.push({stage:activeStage,message:e.message}));
page.on('console',m=>{if(['warning','error'].includes(m.type()))report.console.push({stage:activeStage,type:m.type(),text:m.text()});});
page.on('request',r=>{if(new URL(r.url()).origin!==new URL(base).origin)report.externalRequests.push(r.url());});
page.on('requestfailed',r=>report.requestFailures.push({stage:activeStage,path:new URL(r.url()).pathname,error:r.failure()?.errorText}));
page.on('response',r=>{
  const path=new URL(r.url()).pathname;
  if(!path.startsWith('/api/'))return;
  const headers=r.headers();
  report.responses.push({stage:activeStage,path,status:r.status(),method:r.request().method(),
    request:r.request().postDataJSON(),headers:Object.fromEntries(Object.entries(headers).filter(([k])=>k.startsWith('x-')))});
});
await page.addInitScript(()=>{
  window.__auditLongTasks=[];
  new PerformanceObserver(list=>{for(const e of list.getEntries())window.__auditLongTasks.push({start:e.startTime,ms:e.duration});}).observe({type:'longtask',buffered:true});
});
const save=()=>writeFile(out+'/audit-'+phases.join('-')+'.json',JSON.stringify(report,null,2));
const settle=async()=>{await page.waitForTimeout(240);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));};
const state=()=>page.evaluate(()=>structuredClone(window.__tractlabTest));
const measure=()=>page.locator('#mapLayers .map-layer').evaluateAll(rows=>Object.fromEntries(rows.map(r=>[r.dataset.layerKey,r.querySelector('.map-value').textContent])));
const shot=async name=>{await settle();const path=out+'/'+name+'.png';await page.screenshot({path,scale:'css'});return path;};
const stage=async(name,fn)=>{
  activeStage=name;const start=Date.now(),row={name};
  try{row.evidence=await fn();row.status='pass';}
  catch(e){row.status='failed';row.error=e.message;row.screenshot=await shot('failure-'+name).catch(()=>null);row.state=await state().catch(()=>null);}
  row.elapsedMs=Date.now()-start;report.stages.push(row);await save();
  console.log(JSON.stringify({name,status:row.status,ms:row.elapsedMs,error:row.error}));
};
const fresh=async()=>{
  await page.setViewportSize({width:1179,height:557});
  if(page.url()===base+'#test')await page.reload({waitUntil:'load'});
  else await page.goto(base+'#test',{waitUntil:'load'});
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
  await settle();
};
const panel=async name=>{
  if(await page.locator('#btnRail').isVisible()&&!await page.locator('#side').evaluate(e=>e.getBoundingClientRect().right>0))await page.locator('#btnRail').click();
  await page.locator('#tab-'+name).click();
};
const open=async selector=>{if(!await page.locator(selector).evaluate(e=>e.open))await page.locator(selector+' > summary').click();};
const numeric=async(id,value)=>{const field=page.locator('#'+id).locator('..').locator('input[type=number]');await field.fill(String(value));await field.press('Tab');await settle();};
const rects=()=>page.evaluate(()=>{
  const r=id=>{const e=document.getElementById(id),b=e.getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height,hidden:!e.checkVisibility({checkVisibilityCSS:true})};};
  return {viewport:[innerWidth,innerHeight],scene:r('gl'),dock:r('anatomyDock'),compact:document.getElementById('anatomyDock').classList.contains('anatomy-compact'),mpr:['cax','ccor','csag'].map(r),strip:r('compactTracts'),rail:r('side'),legend:r('mapScale'),overflow:document.documentElement.scrollWidth>innerWidth,layout:window.__tractlabTest.viewLayout,pipeline:window.__tractlabTest.pipeline};
});
const removeAll=async()=>{
  // Desktop cards are outside the canvas; remove through their normal UI.
  while(await page.locator('#mapLayers .map-remove').count())await page.locator('#mapLayers .map-remove').first().click();
  await settle();
};
const chooseBank=async id=>{
  await panel('tracts');await open('#secBanks');
  await page.locator('#bankFilter').fill(id);
  await page.locator('#bank_'+id).click();
  await page.waitForFunction(key=>window.__tractlabTest.layerKeys?.includes(key)&&document.querySelector('.map-layer[data-layer-key="'+key+'"]')?.getAttribute('aria-busy')!=='true','bank:'+id,{timeout:120000});
  // aria-busy is set to false on settled cards, not removed in every build.
  await page.waitForFunction(id=>!document.getElementById('bank_'+id).classList.contains('loading'),id,{timeout:120000});
  await settle();
};
const hashCanvas=async selector=>page.locator(selector).evaluate(el=>{const data=el.getContext('2d').getImageData(0,0,el.width,el.height).data;let sum=0,n=0;for(let i=0;i<data.length;i+=4){sum=(sum+data[i]+data[i+1]*3+data[i+2]*7)>>>0;if(data[i]+data[i+1]+data[i+2]>30)n++;}return {sum,n,width:el.width,height:el.height};});
try{
  report.bootstrapStart=await (await context.request.get(base+'api/bootstrap')).json();
  assert.equal(report.bootstrapStart.runtimeStatus,'ok');
  assert.equal(report.bootstrapStart.busy,false,'Do not compete with an active server job');
  await stage('startup',async()=>{const start=Date.now();await fresh();return {readyMs:Date.now()-start,state:await state(),rects:await rects(),screenshot:await shot('cold-overview'),longTasks:await page.evaluate(()=>window.__auditLongTasks)};});

  if(phases.includes('overview')){
    await stage('focus-colour-named-views',async()=>{
      const initial=await measure(),states=[];
      for(const id of ['bank_cst_r','bank_fat_r','bank_slf3_r']){
        await page.locator('[data-layer-key="bank:'+id+'"] .map-focus').click();
        await panel('display');
        for(const mode of ['Dir','Solid','Dist']){
          await page.locator('#btn'+mode).click();await settle();
          assert.deepEqual(await measure(),initial,'Focus and colour must preserve measurements');
          states.push({id,mode,focus:(await state()).focusLayerKey,screenshot:await shot('focus-'+id+'-'+mode)});
        }
      }
      await page.locator('[data-layer-key="bank:bank_fat_r"] .map-focus').click();await page.locator('#btnDir').click();
      for(const view of ['L','R','A','P','S','I']){
        await page.locator('[data-view="'+view+'"]').click();await settle();
        assert.deepEqual(await measure(),initial);
        states.push({view,screenshot:await shot('named-'+view),camera:(await state()).camera});
      }
      await page.locator('#btnResetView').click();
      return states;
    });
    await stage('surface-radius-support',async()=>{
      await panel('display');const values=await measure(),samples=[];
      for(const mode of ['Ghost','Mid','Solid']){
        await page.locator('#hullVeil'+mode).click();samples.push({mode,screenshot:await shot('hull-'+mode)});
      }
      await page.locator('#hullVeilGhost').click();
      for(const id of ['btnLesionVis','btnHullVis']){await page.locator('#'+id).click();await settle();await page.locator('#'+id).click();}
      for(const radius of [.12,.22,.6]){await numeric('fibre',radius);samples.push({radius,state:(await state()).pipeline});}
      await page.locator('#fibre').locator('..').getByRole('button').click();
      assert.deepEqual(await measure(),values);
      const support=await page.locator('#btnAllSupport,#btnHideLowSupport,#btnOnlyLowSupport').evaluateAll(es=>es.map(e=>({id:e.id,disabled:e.disabled,title:e.title})));
      assert.ok(support.every(s=>s.disabled),'Absent support cannot be an enabled filter');
      return {samples,support,note:await page.locator('#evidenceAvailability').innerText()};
    });
  }
  if(phases.includes('mri')){
    await fresh();
    await stage('anatomy-dock-thresholds',async()=>{
      const sizes=[await rects()];
      await page.locator('#anatomyResize').focus();
      for(let i=0;i<5;i++){await page.keyboard.press('ArrowUp');await settle();sizes.push(await rects());if(i===1)await shot('dock-threshold-152');}
      await page.locator('#btnInspectMRI').click();sizes.push(await rects());
      return sizes;
    });
    await stage('MRI-underlays-slices-inspection',async()=>{
      await page.locator('#btnDefaults').click();await page.locator('#expand-cor').click();const values=await measure(),samples=[];
      for(const id of ['btnB0','btnT1','btnFA','btnDEC']){await page.locator('#'+id).click();await settle();samples.push({underlay:id,hash:await hashCanvas('#ccor')});}
      assert.equal(new Set(samples.map(s=>s.hash.sum)).size,4);
      for(const axis of ['ax','cor','sag']){
        await page.locator('#slicePlane').selectOption(axis);await settle();
        samples.push({axis,state:(await state()).map.slices,screenshot:await shot('plane-'+axis)});
      }
      await page.locator('#slice-cor').focus();await page.keyboard.press('ArrowRight');
      await page.locator('#ccor').hover();await page.mouse.wheel(0,90);await settle();
      await page.locator('#ccor').click({position:{x:400,y:100}});await settle();
      assert.deepEqual(await measure(),values);
      await page.locator('#slicePlane').selectOption('off');await page.locator('#btnB0').click();await page.locator('#expand-cor').click();
      for(const axis of ['ax','cor','sag']){await page.locator('#expand-'+axis).click();samples.push({expanded:axis,rects:await rects()});await page.locator('#expand-'+axis).click();}
      return samples;
    });
  }
  if(phases.includes('catalog')){
    await fresh();
    await stage('all-32-banks',async()=>{
      const catalog=await (await context.request.get(base+'api/banks')).json(),rows=[];
      await removeAll();
      for(const bank of catalog.banks){
        const before=Date.now();await chooseBank(bank.id);
        const s=await state(),response=report.responses.findLast(r=>r.path==='/api/bank/load'&&r.request?.bankId===bank.id&&r.status===200);
        assert.equal(s.focusLayerKey,'bank:'+bank.id);
        assert.equal(s.exportIdentity.bankId,bank.id);
        const headers=response?.headers||{};
        const row={id:bank.id,n:bank.nStreamlines,ms:Date.now()-before,values:await measure(),layer:s.map?.layers,headers};
        rows.push(row);report.catalogProgress=rows;await save();
        if(rows.length%8===0)console.log(JSON.stringify({catalogProgress:rows.length,total:catalog.banks.length}));
        await removeAll();
      }
      await panel('tracts');await page.locator('#bankFilter').fill('no-such-tract-zz');
      const noMatches=await page.locator('#bankGroups button:visible').count();
      assert.equal(noMatches,0);
      await page.locator('#bankFilter').fill('');
      return {count:rows.length,rows,noMatches};
    });
  }
  if(phases.includes('overlays')){
    await fresh();await panel('tracts');
    await stage('all-22-population-priors',async()=>{
      await open('#priorDetails');const ids=await page.locator('#priorGroups button').evaluateAll(es=>es.map(e=>e.id));const rows=[],values=await measure();
      for(const id of ids){
        await page.locator('#'+id).click();
        await page.waitForFunction(()=>window.__tractlabPriorProbe?.activeCount===1,undefined,{timeout:30000});
        const probe=await page.evaluate(()=>window.__tractlabPriorProbe);
        assert.equal(probe.anyTubeMaterial,false);assert.equal(probe.anyDirectionRGB,false);
        rows.push({id,probe,ariaPressed:await page.locator('#'+id).getAttribute('aria-pressed')});
        if(rows.length===1)await shot('population-prior');
        await page.locator('#'+id).click();await page.waitForFunction(()=>window.__tractlabPriorProbe.activeCount===0);
        assert.deepEqual(await measure(),values);
      }
      return {count:rows.length,rows};
    });
    await stage('seven-territories-and-intersections',async()=>{
      await open('#priorDetails');const rows=[];
      const nets=await page.locator('#parcelNetworks button').evaluateAll(es=>es.map(e=>e.dataset.networkId));
      for(const id of nets){
        await page.locator('#parcelNetworks button[data-network-id="'+id+'"]').click();
        await page.waitForFunction(id=>window.__tractlabParcelProbe?.networkId===Number(id)&&window.__tractlabParcelProbe.activeCount===1,id,{timeout:30000});
        const probe=await page.evaluate(()=>window.__tractlabParcelProbe);assert.equal(probe.anyTubeMaterial,false);rows.push(probe);
      }
      await numeric('parcelIntensity',.7);await shot('territory-overlay');
      await page.locator('#parcelNetworks button[data-network-id="'+nets.at(-1)+'"]').click();
      await open('#connectotomyDetails');
      await page.waitForFunction(()=>document.getElementById('connectotomyHost').dataset.state==='ready',undefined,{timeout:60000});
      const cuts=await page.locator('#connectotomyHost button[data-bank-id]').evaluateAll(es=>es.map(e=>({id:e.dataset.bankId,text:e.innerText})));
      await shot('lesion-intersections');
      const cut=page.locator('#connectotomyHost button[data-bank-id]').first();await cut.click();
      await page.waitForFunction(()=>window.__tractlabTest.sourcePopulation?.startsWith('cut-subset:'),undefined,{timeout:30000});
      return {networks:rows,cuts,cutState:await state()};
    });
  }
  if(phases.includes('research')){
    await fresh();
    await stage('owned-exports-and-envelope',async()=>{
      const rows=[];
      for(const id of ['bank_cst_r','bank_fat_r','bank_slf3_r']){
        await page.locator('[data-layer-key="bank:'+id+'"] .map-focus').click();await panel('tools');await open('#researchTools');
        const downloading=page.waitForEvent('download');await page.locator('#btnExport').click();const download=await downloading;
        const bytes=await readFile(await download.path()),response=report.responses.findLast(r=>r.path==='/api/export/tck');
        assert.equal(response.request.bankId,id);assert.equal(response.headers['x-bankid'],id);
        assert.ok(bytes.subarray(0,64).toString().includes('mrtrix tracks'));
        rows.push({id,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),headers:response.headers});
      }
      await page.locator('#btnMargin').click();await page.waitForFunction(()=>document.getElementById('btnMargin').getAttribute('aria-checked')==='true',undefined,{timeout:60000});
      for(const mm of [2,15,5]){const pending=page.waitForResponse(r=>r.url().endsWith('/api/margin')&&r.request().postDataJSON()?.marginMm===mm,{timeout:60000});await numeric('marginMm',mm);const response=await pending;await response.finished();await settle();}
      const envelope=await shot('tract-envelope');await page.locator('#btnMargin').click();
      return {exports:rows,envelope,toolHint:await page.locator('#toolHint').innerText()};
    });
    await stage('recovery-boundaries-and-replace',async()=>{
      await panel('tools');const rows=[];
      for(const mm of [3,8,20]){
        await numeric('recoveryMm',mm);await page.locator('#btnRecovery').click();
        await page.waitForFunction(()=>!document.getElementById('btnRecovery').disabled,undefined,{timeout:120000});
        rows.push({mm,status:await page.locator('#recoveryStatus').innerText(),state:await state()});
        await shot('recovery-'+mm);
      }
      await page.locator('#recoveryKeepBanks').uncheck();await page.locator('#btnRecovery').click();
      await page.waitForFunction(()=>!document.getElementById('btnRecovery').disabled,undefined,{timeout:120000});
      const replaced=await state();assert.ok(replaced.layerKeys.every(k=>!k.startsWith('bank:')));
      return {rows,replaced};
    });
    await fresh();
    await stage('presets-live-and-cancel',async()=>{
      await panel('tools');await open('#advLive');
      const ids=await page.locator('#presetRow button').evaluateAll(es=>es.map(e=>e.id)),presets=[];
      for(const id of ids){
        await page.locator('#'+id).click();await settle();presets.push({id,status:await page.locator('#status').textContent(),state:await state()});
      }
      await page.locator('#'+ids[0]).click();await page.locator('#denSparse').click();
      const liveResponse=page.waitForResponse(r=>r.url().endsWith('/api/track'),{timeout:120000});
      await page.locator('#track').click();
      const during=await page.locator('#track,#btnFilter,#btnRecovery').evaluateAll(es=>es.map(e=>({id:e.id,disabled:e.disabled})));
      const response=await liveResponse;await response.finished();
      await page.waitForFunction(()=>!document.getElementById('track').disabled,undefined,{timeout:120000});
      const live={http:response.status(),state:await state(),status:await page.locator('#status').textContent(),screenshot:await shot('sparse-live')};
      await page.locator('#track').click();await page.locator('#cancel').click();
      await page.waitForFunction(()=>!document.getElementById('track').disabled,undefined,{timeout:30000});
      return {presets,during,live,cancelStatus:await page.locator('#status').textContent()};
    });
    await fresh();
    await stage('paint-all-roles-filter-undo',async()=>{
      await page.locator('#expand-cor').click();await page.locator('#btnPaintMode').click();await panel('tools');await open('#advLive');
      const samples=[];
      for(const [i,role] of ['Seed','And','Or','Not'].entries()){
        await page.locator('#role'+role).click();if(role==='And')await page.locator('#btnNewAnd').click();
        const b=await page.locator('#ccor').boundingBox();await page.mouse.click(b.x+b.width*(.48+i*.015),b.y+b.height*.42);await settle();
        samples.push({role,state:await state(),undoDisabled:await page.locator('#btnUndoPaint').isDisabled()});
      }
      const painted=await shot('paint-all-roles');
      // Return to a seed-only mask through actual undo, then filter the corpus.
      for(let i=0;i<3;i++)await page.locator('#btnUndoPaint').click();
      const filtering=page.waitForResponse(r=>r.url().endsWith('/api/filter'),{timeout:120000});
      await page.locator('#btnFilter').click();const response=await filtering;await response.finished();
      await page.waitForFunction(()=>!document.getElementById('btnFilter').disabled,undefined,{timeout:120000});
      const filtered={http:response.status(),state:await state(),status:await page.locator('#status').textContent()};
      await page.locator('#clear').click();await settle();const clearUndo=await page.locator('#btnUndoPaint').isEnabled();
      if(clearUndo)await page.locator('#btnUndoPaint').click();
      return {samples,painted,filtered,clearUndo};
    });
  }
  if(phases.includes('controls')){
    await fresh();
    await stage('display-bias-and-live-parameters',async()=>{
      const original=await measure(),focused=(await state()).focusLayerKey;
      await panel('tools');await open('#researchTools');
      const commits=(await state()).commitCount;
      await page.locator('#displayNearLesion').uncheck();
      await page.waitForFunction(n=>window.__tractlabTest.commitCount>=n+3&&window.__tractlabTest.layerKeys.length===3,commits,{timeout:120000});await settle();
      const afterBias={measurements:await measure(),focus:(await state()).focusLayerKey,export:(await state()).exportIdentity};
      assert.deepEqual(afterBias.measurements,original);
      await open('#advLive');const parameters=[];
      for(const [id,lo,hi] of [['brush',1,12],['cutoff',.02,.20],['angle',15,60],['minlen',10,80]]){
        await numeric(id,lo);await numeric(id,hi);
        const control=page.locator('#'+id).locator('..');
        await control.locator('input[type=number]').fill('999');await control.locator('input[type=number]').press('Tab');
        const rejected=await page.locator('#'+id).inputValue();
        await control.getByRole('button').click();
        parameters.push({id,lo,hi,rejected,reset:await page.locator('#'+id).inputValue()});
      }
      const selected=[];
      for(const density of ['Sparse','Dense','Normal']){await page.locator('#den'+density).click();selected.push(await page.locator('#densityHint').textContent());}
      await page.locator('#presetRow button').first().click();await page.locator('#denSparse').click();
      await numeric('cutoff',.20);await numeric('angle',15);await numeric('minlen',10);
      const pending=page.waitForResponse(r=>r.url().endsWith('/api/track'),{timeout:120000});
      await page.locator('#track').click();const response=await pending;await response.finished();
      await page.waitForFunction(()=>!document.getElementById('track').disabled,undefined,{timeout:120000});
      return {focused,afterBias,parameters,selected,live:{request:response.request().postDataJSON().params,headers:response.headers(),status:await page.locator('#status').textContent()}};
    });
  }
  if(phases.includes('regressions')){
    await fresh();
    await stage('variant-names-and-rounding',async()=>{
      await page.setViewportSize({width:1440,height:900});
      await chooseBank('bank_cst_r_strict');
      const variants=await page.locator('#mapLayers .map-layer').evaluateAll(es=>es.map(e=>({key:e.dataset.layerKey,name:e.querySelector('.map-name').textContent,value:e.querySelector('.map-value').textContent})));
      const variantShot=await shot('cst-variant-names');
      await chooseBank('bank_slf3_r_soft');
      const rounding={card:await page.locator('#mapLayers .map-layer[data-layer-key="bank:bank_slf3_r_soft"]').innerText(),headers:report.responses.findLast(r=>r.path==='/api/bank/load'&&r.request?.bankId==='bank_slf3_r_soft')?.headers,screenshot:await shot('soft-p5-rounding')};
      await page.setViewportSize({width:1179,height:557});await settle();
      const shortWindow={rects:await rects(),railPanel:await page.locator('.rail-panel:visible').boundingBox(),screenshot:await shot('five-banks-short-window')};
      return {variants,variantShot,rounding,shortWindow};
    });
    await fresh();
    await stage('MRI-return-and-closed-drawer-keyboard',async()=>{
      const before=await rects();
      await page.locator('#expand-cor').click();await settle();await page.locator('#expand-cor').click();await settle();
      const after=await rects();
      await page.locator('#btnDefaults').click();await settle();const defaults=await rects();
      await panel('display');await page.locator('#btnDir').focus();await page.keyboard.press('ArrowRight');
      const colourTabs=await page.locator('#btnDir,#btnSolid,#btnDist').evaluateAll(es=>es.map(e=>({id:e.id,tabIndex:e.tabIndex,selected:e.getAttribute('aria-selected'),focused:document.activeElement===e})));
      await page.setViewportSize({width:870,height:507});await settle();
      await page.locator('#chromeHelp').click();await page.keyboard.press('Escape');
      const help={hidden:await page.locator('#howto').evaluate(e=>e.hidden),expanded:await page.locator('#chromeHelp').getAttribute('aria-expanded')};
      await page.locator('#chromeHelp').focus();const keyboard=[];
      for(let i=0;i<18;i++){await page.keyboard.press('Tab');keyboard.push(await page.evaluate(()=>{const e=document.activeElement,b=e.getBoundingClientRect();return {id:e.id,tag:e.tagName,x:b.x,y:b.y,value:e.value};}));if(await page.locator('#fibre').evaluate(e=>document.activeElement===e))break;}
      const radiusBefore=await page.locator('#fibre').inputValue();await page.keyboard.press('ArrowRight');const radiusAfter=await page.locator('#fibre').inputValue();
      const screenshot=await shot('hidden-drawer-keyboard');await page.keyboard.press('ArrowLeft');
      return {before,after,defaults,colourTabs,help,keyboard,radiusBefore,radiusAfter,screenshot};
    });
    await fresh();
    await stage('cancel-completion-and-idle',async()=>{
      await panel('tools');await open('#advLive');await page.locator('#presetRow button').first().click();await page.locator('#denSparse').click();
      const cancelResponse=page.waitForResponse(r=>r.url().endsWith('/api/cancel'));
      await page.locator('#track').click();await page.locator('#cancel').click();const response=await cancelResponse;await response.finished();
      await page.waitForTimeout(1200);
      const health=await (await context.request.get(base+'api/bootstrap')).json();
      const cancelled={http:response.status(),busy:health.busy,ui:await page.locator('#status').textContent(),state:await state()};
      await fresh();const first=(await state()).renderCount;await page.waitForTimeout(1000);const last=(await state()).renderCount;
      return {cancelled,idleExtraFrames:last-first};
    });
  }
  if(phases.includes('responsive')){
    await fresh();
    for(const [width,height] of [[1440,900],[1179,557],[1000,650],[870,507],[768,600],[480,700],[390,844],[844,390]]){
      await stage('responsive-'+width+'x'+height,async()=>{
        await page.setViewportSize({width,height});await page.locator('#btnDefaults').click();await settle();
        const dimensions=await rects(),metrics=await page.evaluate(visibleUiMetrics),screenshot=await shot('responsive-'+width+'x'+height);
        const outline=await page.locator('h1,h2,h3,[role=main],nav').evaluateAll(es=>es.map(e=>({tag:e.tagName,role:e.getAttribute('role'),text:e.textContent.slice(0,100)})));
        await page.locator('#chromeHelp').click();const help=await shot('help-'+width+'x'+height);await page.keyboard.press('Escape');
        const tabSequence=[];
        await page.locator('#chromeHelp').focus();
        for(let i=0;i<24;i++){
          await page.keyboard.press('Tab');
          tabSequence.push(await page.evaluate(()=>{const e=document.activeElement,b=e.getBoundingClientRect();return {id:e.id,tag:e.tagName,text:e.textContent.trim().slice(0,40),x:b.x,y:b.y,w:b.width,h:b.height,offscreen:b.right<=0||b.left>=innerWidth||b.bottom<=0||b.top>=innerHeight};}));
        }
        return {dimensions,metrics,outline,screenshot,help,tabSequence};
      });
    }
  }
}finally{
  report.bootstrapEnd=await (await context.request.get(base+'api/bootstrap')).json().catch(e=>({error:e.message}));
  report.identityStable=report.bootstrapStart?.buildId===report.bootstrapEnd?.buildId&&report.bootstrapStart?.bootId===report.bootstrapEnd?.bootId;
  report.finishedAt=new Date().toISOString();await save();await browser.close();
  console.log(JSON.stringify({output:out+'/audit-'+phases.join('-')+'.json',passed:report.stages.filter(s=>s.status==='pass').length,failed:report.stages.filter(s=>s.status==='failed').length,identityStable:report.identityStable,pageErrors:report.pageErrors.length}));
}
