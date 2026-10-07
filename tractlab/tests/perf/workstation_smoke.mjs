import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {visibleUiMetrics} from './ui_metrics.mjs';

const url=new URL(process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18992/');
url.hash='test';
const quick=process.argv.includes('--quick');
const perf=process.argv.includes('--perf');
const audit=process.argv.includes('--audit');
const displayRadius=process.argv.find(a=>a.startsWith('--radius='))?.slice(9);
// The headless shell forces SwiftShader on this Mac. Performance needs the
// real Chrome compositor; structural checks can still use the headless shell.
const browser=await chromium.launch((perf||process.platform==='darwin')?{headless:true,channel:'chrome'}:{headless:true});
const page=await browser.newPage({viewport:{width:1280,height:850},deviceScaleFactor:2.5});
await page.addInitScript(()=>{window.__startupLongTasks=[];new PerformanceObserver(list=>{for(const e of list.getEntries())window.__startupLongTasks.push({start:Math.round(e.startTime),ms:Math.round(e.duration)});}).observe({type:'longtask',buffered:true});});
page.setDefaultTimeout(15000);
const errors=[],consoleErrors=[],failedResponses=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text());});
page.on('response',r=>{if(r.status()>=400)failedResponses.push({path:new URL(r.url()).pathname,status:r.status()});});
const measurements=()=>page.locator('#mapLayers .map-layer').evaluateAll(rows=>Object.fromEntries(rows.filter(row=>row.getAttribute('aria-busy')!=='true').map(row=>[row.dataset.layerKey,row.querySelector('.map-value').textContent])));
try {
  const started=Date.now();
  await page.goto(url.href,{waitUntil:'load'});
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000}).catch(error=>{throw new Error(`${error.message}; page errors: ${errors.join('; ')}; browser: ${consoleErrors.slice(0,4).join('; ')}`);});
  const readyMs=Date.now()-started;
  const startup=await page.evaluate(()=>({resources:performance.getEntriesByType('resource').map(r=>({path:new URL(r.name).pathname,start:Math.round(r.startTime),ms:Math.round(r.duration)})),longTasks:window.__startupLongTasks}));
  assert.equal(startup.resources.some(r=>r.path==='/api/connectotomy'),false,'optional intersection analysis cannot delay the default map');
  if(displayRadius){const input=page.locator('#fibre').locator('..').locator('input[type=number]');await input.fill(displayRadius);await input.press('Tab');}
  await page.waitForTimeout(220); // Let the documented 160 ms state transitions finish.
  await page.screenshot({path:`/private/tmp/tractlab-workstation-desktop${displayRadius?'-radius-'+displayRadius:''}.png`,scale:'css'});
  assert.deepEqual(errors,[]);
  const original=await measurements();assert.equal(Object.keys(original).length,3);
  const desktop=await page.evaluate(()=>({
    rail:{height:document.getElementById('side').clientHeight,content:document.getElementById('side').scrollHeight},
    scene:{width:document.getElementById('scene').clientWidth,height:document.getElementById('scene').clientHeight},
    mpr:[...document.querySelectorAll('.mpr canvas')].map(e=>({id:e.id,width:e.clientWidth,height:e.clientHeight,backing:[e.width,e.height]})),
    mprPainted:[...document.querySelectorAll('.mpr canvas')].every(e=>e.getContext('2d').getImageData(0,0,e.width,e.height).data.some((v,i)=>i%4!==3&&v>30)),
    pipeline:window.__tractlabTest?.pipeline,
    gpu:(()=>{const gl=document.getElementById('gl').getContext('webgl2');const ext=gl.getExtension('WEBGL_debug_renderer_info');return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);})(),
  }));
  console.log(JSON.stringify({stage:'loaded',readyMs,pipeline:desktop.pipeline,gpu:desktop.gpu,startup}));
  assert.equal(desktop.mprPainted,true,'all physical MPR canvases actually paint anatomy');
  assert.equal(desktop.pipeline.aoEnabled,true);assert.equal(desktop.pipeline.samples,4);assert.ok(desktop.pipeline.pixelRatio>=2.4,'settled image retains near-native detail within the pixel budget');
  const accessibility={display:await page.evaluate(visibleUiMetrics)};
  for(const panel of ['tracts','tools']){await page.locator(`#tab-${panel}`).click();accessibility[panel]=await page.evaluate(visibleUiMetrics);}
  await page.locator('#tab-display').click();
  for(const [panel,metrics] of Object.entries(accessibility)){
    assert.deepEqual(metrics.contrastFailures,[],`${panel}: visible text contrast`);
    assert.deepEqual(metrics.smallText,[],`${panel}: 12 px text floor`);
    assert.deepEqual(metrics.smallTargets,[],`${panel}: 32 px targets`);
  }
  if(audit){
    const failedPanels=[];
    for(const name of ['tracts','display','tools']){
      await page.locator(`#tab-${name}`).click();
      const panel=page.locator('.rail-panel:visible');
      const restore=await panel.evaluate(el=>{const details=[...el.querySelectorAll('details:not(#connectotomyDetails)')];const state=details.map(d=>d.open);details.forEach(d=>d.open=true);return state;});
      const size=await panel.evaluate(el=>({height:el.clientHeight,total:el.scrollHeight}));
      const failures=[];
      for(let offset=0;offset<size.total;offset+=Math.max(100,size.height*.75)){
        await panel.evaluate((el,y)=>el.scrollTop=y,offset);
        const m=await page.evaluate(visibleUiMetrics);
        if(m.contrastFailures.length||m.smallText.length||m.smallTargets.length)failures.push({offset,...m});
      }
      await panel.evaluate((el,state)=>{[...el.querySelectorAll('details:not(#connectotomyDetails)')].forEach((d,i)=>d.open=state[i]);el.scrollTop=0;},restore);
      const unique=items=>[...new Map(items.map(item=>[JSON.stringify(item),item])).values()];
      console.log(JSON.stringify({stage:'expanded-audit',panel:name,contrast:unique(failures.flatMap(f=>f.contrastFailures)),text:unique(failures.flatMap(f=>f.smallText)),targets:unique(failures.flatMap(f=>f.smallTargets))}));
      if(failures.length)failedPanels.push(name);
    }
    await page.locator('#tab-display').click();
    assert.equal(failedPanels.length,0,`Expanded/scrolled audit failures: ${failedPanels.join(', ')}`);
  }
  let performanceReport=null;
  if(perf){
    await page.evaluate(()=>{window.__frameProbe=[];let lastCount=-1,previous;function sample(t){const s=window.__tractlabTest;if(s.renderCount!==lastCount){if(previous!=null)window.__frameProbe.push({ms:t-previous,cpu:s.frameCpuMs,dpr:s.pipeline.pixelRatio,active:s.pipeline.interacting});previous=t;lastCount=s.renderCount;}window.__probeRaf=requestAnimationFrame(sample);}window.__probeRaf=requestAnimationFrame(sample);});
    const box=await page.locator('#gl').boundingBox();const x=box.x+box.width*.55,y=box.y+box.height*.45;
    await page.mouse.move(x,y);await page.mouse.down();
    for(let i=1;i<=24;i++){await page.mouse.move(x+Math.sin(i/10)*140,y+i*2);await page.waitForTimeout(35);}
    await page.mouse.up();
    await page.waitForFunction(()=>!window.__tractlabTest.pipeline.interacting,undefined,{timeout:10000});
    performanceReport=await page.evaluate(()=>{cancelAnimationFrame(window.__probeRaf);const rows=window.__frameProbe.filter(r=>r.active),a=rows.map(r=>r.ms).sort((a,b)=>a-b);const cpu=rows.map(r=>r.cpu).sort((a,b)=>a-b);return {frames:rows.length,frameMedianMs:a[Math.floor(a.length*.5)],frameP95Ms:a[Math.floor(a.length*.95)],submissionMedianMs:cpu[Math.floor(cpu.length*.5)],interactionDpr:Math.min(...rows.map(r=>r.dpr)),settledDpr:window.__tractlabTest.pipeline.pixelRatio};});
    await page.locator('#btnResetView').click();
  }
  if(!quick){
    assert.ok(desktop.scene.height>=600,'Overview reserves height for 3D');
    await page.locator('#expand-cor').click();
    const inspected=await page.locator('#ccor').boundingBox();
    assert.ok(inspected.width>=500&&inspected.height>=300,'explicit inspection exposes a usable MPR surface');
    const compassBefore=await page.locator('#orientationCube').innerHTML();
    await page.locator('[data-view="A"]').click();
    await page.waitForFunction(before=>document.getElementById('orientationCube').innerHTML!==before,compassBefore);
    await page.locator('#btnResetView').click();
    await page.locator('#slicePlane').selectOption('cor');
    await page.locator('#slice-cor').focus();await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(220);
    await page.screenshot({path:'/private/tmp/tractlab-workstation-slice-plane.png',scale:'css'});
    for(const [key,title] of [['1','Fibre orientation'],['2','Tract identity'],['3','Distance to lesion']]){
      await page.locator('#btnResetView').focus();await page.keyboard.press(key);
      await page.waitForFunction(expected=>document.querySelector('#mapScale p')?.textContent===expected,title);
      assert.deepEqual(await measurements(),original,'display modes preserve owned measurements');
    }
    // Retained recovery is a separate generated result; it must not silently
    // change the named source used by Export and Envelope.
    const focusedKey=Object.keys(original).find(key=>/cst/i.test(key));
    await page.locator(`.map-layer[data-layer-key="${focusedKey}"] .map-focus`).click();
    await page.locator('#tab-tools').click();
    await page.locator('#btnRecovery').click();
    await page.waitForFunction(()=>window.__tractlabTest.layerKeys.includes('__recovery__')&&!document.getElementById('btnRecovery').disabled,undefined,{timeout:120000});
    assert.equal(await page.evaluate(()=>window.__tractlabTest.focusLayerKey),focusedKey);
    assert.equal(await page.evaluate(()=>window.__tractlabTest.exportIdentity.bankId),focusedKey.slice(5));
    const withRecovery=await measurements();
    for(const [key,value] of Object.entries(original))assert.equal(withRecovery[key],value);
    assert.match(await page.locator('#recoveryStatus').textContent(),/\d.* of .* streamlines found/);
    await page.locator('.map-layer[data-layer-key="__recovery__"] .map-remove').click();
    await page.locator('#tab-display').click();
    // Verify true data-space hit mapping, independent of the presentation backing resolution.
    await page.locator('#ccor').click();
    assert.deepEqual(await measurements(),original,'slice inspection keeps tract layers');
    await page.locator('#btnPaintMode').click();
    await page.locator('#ccor').click();
    await page.locator('#btnUndoPaint').click();
    assert.equal(await page.locator('#btnUndoPaint').isDisabled(),true);
    await page.locator('#btnPaintMode').click();
    await page.reload({waitUntil:'load'});await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
    await page.locator('#tab-tracts').click();
    await page.locator('#connectotomyDetails summary').click();
    await page.waitForFunction(()=>document.getElementById('connectotomyHost').dataset.state==='ready',undefined,{timeout:60000});
    await page.locator('#connectotomyDetails summary').click();
  }
  await page.setViewportSize({width:851,height:563});
  await page.locator('#btnSlices').click(); // Collapse the open desktop dock for compact viewing.
  await page.waitForTimeout(220);
  await page.locator('#compactTracts button').first().click();
  const drawerHeight=await page.locator('.rail-panel:visible').evaluate(el=>el.clientHeight);
  assert.ok(drawerHeight>=300,'compact drawer reserves space for focused controls');
  await page.locator('#btnCloseRail').click();
  await page.waitForTimeout(220);
  await page.screenshot({path:'/private/tmp/tractlab-workstation-compact.png',scale:'css'});
  const compact=await page.evaluate(()=>({sceneWidth:document.getElementById('scene').clientWidth,windowWidth:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth}));
  assert.equal(compact.overflow,false);assert.ok(compact.sceneWidth>=compact.windowWidth-2);
  assert.deepEqual(await measurements(),original);
  assert.deepEqual(errors,[]);assert.deepEqual(consoleErrors,[],'browser console stays clean');
  console.log(JSON.stringify({readyMs,desktop,compact,drawerHeight,accessibility,performanceReport,errors,consoleErrors,failedResponses,quick},null,2));
}catch(error){console.error(JSON.stringify({stage:'browser-failure',message:error.message,state:await page.evaluate(()=>({pipeline:window.__tractlabTest?.pipeline,status:document.getElementById('status')?.textContent,legend:document.querySelector('#mapScale p')?.textContent})),errors,consoleErrors,failedResponses}));throw error;}finally{await browser.close();}
