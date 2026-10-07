import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const baseline=process.argv.includes('--baseline');
const url=process.argv.find(v=>v.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18993/';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1280,height:850},deviceScaleFactor:2.5,locale:'pt-BR'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const bankHeaders=new Map();page.on('response',response=>{
  if(new URL(response.url()).pathname==='/api/bank/load'&&response.status()===200)bankHeaders.set(response.request().postDataJSON().bankId,response.headers());
});
try{
  await page.goto(`${url}#test`);await page.waitForFunction(()=>window.__tractlabTest?.mapReady&&window.__tractlabTest?.pipeline&&!window.__tractlabTest.pipeline.interacting,undefined,{timeout:120000});
  const original=await page.evaluate(()=>({map:window.__tractlabTest.map.layers,values:[...document.querySelectorAll('.map-value')].map(e=>e.textContent)}));
  await page.evaluate(async()=>{
    const THREE=await import('/vendor/three.module.js');
    window.__reviewDisposed=0;
    const dispose=THREE.BufferGeometry.prototype.dispose;
    THREE.BufferGeometry.prototype.dispose=function(){window.__reviewDisposed++;return dispose.call(this);};
  });
  const timing=await page.evaluate(()=>{
    const el=document.getElementById('fibre'),samples=[];
    for(const value of ['0.12','0.18','0.3','0.46','0.22']){
      el.value=value;const start=performance.now();el.dispatchEvent(new Event('input',{bubbles:true}));samples.push(performance.now()-start);
    }
    return {samples,disposed:window.__reviewDisposed};
  });
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  assert.deepEqual(await page.evaluate(()=>({map:window.__tractlabTest.map.layers,values:[...document.querySelectorAll('.map-value')].map(e=>e.textContent)})),original);
  console.log(JSON.stringify({baseline,radius:timing,diagnostics:await page.evaluate(()=>window.__tractlabTest.pipeline)}));
  if(!baseline){
    for(const [bank,p5] of [['bank_cst_r',8.67],['bank_fat_r',6.03],['bank_slf3_r',6]]){
      const header=bankHeaders.get(bank);assert.ok(header);
      assert.equal(Number(header['x-clearancep5']),p5,'recorded bank p5 stays unchanged');
      assert.match(header['x-clearancemethod'],/all vertices; no seed exclusion/);
    }
    assert.equal(timing.disposed,0,'radius never replaces geometry or pick proxies');
    assert.ok(Math.max(...timing.samples)<32,'radius input fits within two display frames');
    await page.locator('#tab-display').click();
    for(const id of ['btnAllSupport','btnHideLowSupport','btnOnlyLowSupport'])assert.equal(await page.locator(`#${id}`).isDisabled(),true,'untested banks cannot be filtered as evidence');
    assert.match(await page.locator('#provChipPreflight').textContent(),/untestable/i);
    assert.doesNotMatch(await page.locator('#distLegend').textContent(),/red = near|cyan = far/);
    assert.ok((await page.locator('#counts').textContent()).includes(await page.evaluate(()=>window.__tractlabTest.descriptor.short)),'activity badge uses its actual response descriptor');
    assert.doesNotMatch(await page.locator('#counts').textContent(),/SLF3-/);
    assert.match(await page.locator('#hudTitle').textContent(),/SLF-III-R/);
    assert.match(await page.locator('.map-footnote').textContent(),/reconstructed|recipe/i);
    // A selected source remains selected when only the tube display width changes.
    const pick=await page.evaluate(()=>window.__tractlabPickProbe.projectTarget());await page.mouse.click(pick.x,pick.y);
    await page.waitForFunction(()=>window.__tractlabTest.selected);
    const selection=await page.evaluate(()=>window.__tractlabTest.selected);
    for(const [tag,value] of [['thin','0.04'],['thick','0.5']]){
      await page.locator('#fibre').evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      assert.deepEqual(await page.evaluate(()=>window.__tractlabTest.selected),selection);
      await page.screenshot({path:`/private/tmp/tractlab-radius-${tag}.png`,scale:'css'});
    }
    await page.locator('#btnClearStreamline').click();
    await page.locator('#fibre').evaluate(el=>{el.value=el.defaultValue;el.dispatchEvent(new Event('input',{bubbles:true}));});
  }
  assert.deepEqual(errors,[]);
  await page.close();
  if(!baseline){
    const fixture=await browser.newPage({viewport:{width:1280,height:850}}),fixtureErrors=[];
    fixture.on('pageerror',e=>fixtureErrors.push(e.message));
    fixture.on('console',m=>{if(m.type()==='error')fixtureErrors.push(m.text());});
    const attack='<img src=x onerror="window.__reviewInjected=1">';
    let declaredCountDelta=0;
    await fixture.route('**/api/bank/load',async route=>{
      const response=await route.fetch();
      if(response.status()!==200||route.request().postDataJSON().bankId!=='bank_fat_r'){await route.fulfill({response});return;}
      const headers=response.headers(),body=await response.body(),n=Number(headers['x-linecount'])+declaredCountDelta;
      const off=Math.ceil(body.length/4)*4,withEvidence=Buffer.alloc(off+n*9);body.copy(withEvidence);
      for(let i=0;i<n;i++){withEvidence.writeFloatLE(i%2?.9:.2,off+i*4);withEvidence.writeFloatLE(.1,off+n*4+i*4);}
      Object.assign(headers,{'x-ndisplayed':String(n),'x-fidelitystatus':'ok','x-fidelityoffset':String(off),'x-fidelityr':'0.3','x-fidelityminfrac':'0.7','x-fidelityopsource':'signed',
        'x-fidelityapprovedby':attack,'x-lowsupportcount':String(Math.ceil(n/2)),'x-crosseslesioncount':'0','x-crossescavitycount':'0','content-length':String(withEvidence.length)});
      await route.fulfill({response,headers,body:withEvidence});
    });
    await fixture.goto(`${url}#test`);await fixture.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
    await fixture.locator('#tab-display').click();
    for(const id of ['btnAllSupport','btnHideLowSupport','btnOnlyLowSupport'])assert.equal(await fixture.locator(`#${id}`).isDisabled(),false,'a measured layer enables support filtering');
    assert.match(await fixture.locator('#hudEvidence').textContent(),/750 of 1,500 measured/);
    assert.match(await fixture.locator('#hudEvidence').textContent(),/2,322 untested/);
    assert.ok((await fixture.locator('#provChipOperating').textContent()).includes(attack),'served provenance remains literal text');
    assert.equal(await fixture.locator('#provChipOperating img, #counts img, #hudMeta img').count(),0);
    assert.equal(await fixture.evaluate(()=>window.__reviewInjected),undefined);
    await fixture.locator('#btnOnlyLowSupport').click();
    await fixture.locator('#fibre').evaluate(el=>{el.value='0.4';el.dispatchEvent(new Event('input',{bubbles:true}));});
    await fixture.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    await fixture.screenshot({path:'/private/tmp/tractlab-radius-evidence-fixture.png',scale:'css'});
    await fixture.locator('#btnHideLowSupport').click();await fixture.locator('#btnAllSupport').click();
    await fixture.locator('[data-layer-key="bank:bank_fat_r"] .map-remove').click();
    assert.equal(await fixture.locator('#btnOnlyLowSupport').isDisabled(),true,'removing the final measured layer disables filters');
    assert.equal(await fixture.locator('#btnAllSupport').getAttribute('aria-pressed'),'true');
    declaredCountDelta=-1;
    await fixture.reload();await fixture.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
    await fixture.locator('#tab-display').click();
    assert.equal(await fixture.locator('#btnOnlyLowSupport').isDisabled(),true,'support rows from a different geometry population remain untested');
    assert.match(await fixture.locator('#hudEvidence').textContent(),/No measured support data/);
    assert.match(await fixture.locator('#provChipOperating').textContent(),/absent/);
    assert.deepEqual(fixtureErrors,[]);await fixture.close();

    // Deliberate isolated responses: failed preflight is not absence, and
    // timeout/empty warnings must never be interpreted as markup.
    for(const outcome of ['timeout','empty']){
      const failed=await browser.newPage(),pageErrors=[];failed.on('pageerror',e=>pageErrors.push(e.message));
      await failed.route('**/api/preflight',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"fixture"}'}));
      await failed.route('**/api/bank/load',route=>route.fulfill({status:200,headers:{'X-outcome':outcome==='empty'?'ok':'timeout','X-warning':attack,'X-lineCount':'0','X-pointsPerLine':'64'},body:Buffer.alloc(0)}));
      await failed.goto(`${url}#test`);await failed.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
      assert.match(await failed.locator('#provChipPreflight').textContent(),/unavailable/i);
      assert.ok((await failed.locator('#status').textContent()).includes(attack));
      assert.equal(await failed.locator('#status img').count(),0);
      assert.equal(await failed.evaluate(()=>window.__reviewInjected),undefined);assert.deepEqual(pageErrors,[]);
      await failed.close();
    }
    console.log(JSON.stringify({fixtures:'mixed evidence, removal, literal served text, unavailable preflight: passed'}));
  }
}finally{await browser.close();}
