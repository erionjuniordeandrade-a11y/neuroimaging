#!/usr/bin/env node
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const base=new URL(process.argv.find(arg=>arg.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18992/');
base.search=''; base.hash='test'; // Instrumentation only: no bank, colour or view override.
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1400,height:900}});
const errors=[],recorded=new Map();
page.on('pageerror',error=>errors.push(error.message));
page.on('response',async response=>{
  if(new URL(response.url()).pathname!=='/api/bank/load'||response.status()!==200) return;
  recorded.set(response.request().postDataJSON().bankId,response.headers());
});
const trio=['bank_cst_r','bank_fat_r','bank_slf3_r'];
const waitReady=()=>page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
const values=()=>page.locator('.map-layer').evaluateAll(rows=>Object.fromEntries(rows.map(row=>[
  row.dataset.layerKey,{value:row.querySelector('.map-value').textContent,counts:row.querySelector('.map-counts').textContent,state:row.querySelector('.map-focus').dataset.state},
])));
try{
  await page.goto(base.href,{waitUntil:'load'}); await waitReady();
  const state=await page.evaluate(()=>window.__tractlabTest);
  assert.deepEqual(state.layerKeys,trio.map(id=>`bank:${id}`),'bare 3T URL opens the trio');
  assert.equal(state.map.colour,'distance');
  assert.equal(state.map.lesionVisible,true);
  assert.deepEqual(state.map.target,state.map.origin.world,'camera orbits the lesion centroid');
  const [i,j,k]=state.map.origin.voxel;
  assert.deepEqual(state.map.slices,{sag:Math.round(i),cor:Math.round(j),ax:Math.round(k)});
  assert.equal(await page.locator('#mapScale').isVisible(),true);
  assert.equal(await page.locator('#secBanks').getAttribute('open'),null,'catalog is secondary');
  assert.equal(await page.locator('#mapLabels text[data-layer-key]').count(),3,'all three visible populations are labelled');
  assert.ok(state.map.layers.every(layer=>layer.sourcePopulation===layer.key&&layer.distanceCount>0&&!layer.hasGlow));
  const original=await values();
  for(const id of trio){
    const headers=recorded.get(id); assert.ok(headers,id);
    const p5=Number(headers['x-clearancep5']),floor=Number(headers['x-clearancefloormm']);
    const actual=original[`bank:${id}`];
    if(p5<floor){assert.equal(actual.state,'below-floor');assert.doesNotMatch(actual.value,/≥|>=/);}
    else assert.equal(actual.value,`${p5>=10?p5.toFixed(0):p5.toFixed(1)} mm`,`${id} owns its served p5`);
    assert.ok(actual.counts.includes(Number(headers['x-nanalytic']).toLocaleString('en-US')));
  }
  await page.locator('[data-layer-key="bank:bank_cst_r"] .map-focus').click();
  const identity=await page.evaluate(()=>window.__tractlabTest.exportIdentity);
  await page.locator('#tab-display').click(); // cold start opens Tracts; colour tabs live in Display
  await page.locator('#btnDir').click(); await page.locator('#btnDist').click();
  assert.deepEqual(await values(),original,'recolouring preserves all measurements');
  assert.deepEqual(await page.evaluate(()=>window.__tractlabTest.exportIdentity),identity,'recolouring preserves focused export');
  await page.locator('[data-layer-key="bank:bank_fat_r"] .map-remove').click();
  const remaining=await values();
  assert.deepEqual(Object.keys(remaining),['bank:bank_cst_r','bank:bank_slf3_r']);
  for(const key of Object.keys(remaining)) assert.deepEqual(remaining[key],original[key]);
  await page.waitForFunction(()=>document.querySelectorAll('#mapLabels text[data-layer-key="bank:bank_fat_r"]').length===0);
  assert.equal(await page.locator('#mapLabels text[data-layer-key="bank:bank_fat_r"]').count(),0);

  // Cancel while the fetched response is held: no orphan geometry or p5.
  await page.locator('#tab-tracts').click();
  await page.locator('#secBanks summary').first().click();
  await page.locator('#bankFilter').fill('CST-L');
  let headersResolve,headersReject,releaseBody,finishBody,delayedError=null,cancellationIssued=false;
  const headersReady=new Promise((resolve,reject)=>{headersResolve=resolve;headersReject=reject;});
  const bodyGate=new Promise(resolve=>releaseBody=resolve),bodyDone=new Promise(resolve=>finishBody=resolve);
  await page.route('**/api/bank/load',async route=>{
    if(route.request().postDataJSON().bankId!=='bank_cst_l')return route.continue();
    try{
      const response=await route.fetch();headersResolve();await bodyGate;
      await route.fulfill({response});
    }catch(error){
      headersReject(error);
      // Chrome may have completed the route as soon as the client aborted it.
      if(!cancellationIssued||!error.message.includes('Route is already handled'))delayedError=error;
    }finally{finishBody();}
  });
  await page.locator('#bank_bank_cst_l').click();
  await headersReady;cancellationIssued=true;
  await page.locator('[data-layer-key="bank:bank_cst_l"] .map-remove').click();
  releaseBody();await bodyDone;if(delayedError)throw delayedError;
  assert.equal(await page.locator('[data-layer-key="bank:bank_cst_l"]').count(),0);
  assert.deepEqual(await values(),remaining);
  await page.unroute('**/api/bank/load');

  // Measurement faults are simulated independently for each visible bank.
  await page.route('**/api/bank/load',async route=>{
    const response=await route.fetch(),id=route.request().postDataJSON().bankId;
    const headers={...response.headers()};
    if(id===trio[0]) headers['x-clearancefloormm']='NaN';
    if(id===trio[1]) Object.assign(headers,{'x-clearancep5':'.4','x-clearancep5display':'>=3','x-clearancefloormm':'3'});
    if(id===trio[2]) headers['x-clearancerefusal']='<img src=x onerror=alert(1)>';
    await route.fulfill({response,headers});
  });
  await page.reload({waitUntil:'load'}); await waitReady();
  const limited=await values();
  assert.equal(limited[`bank:${trio[0]}`].state,'unknown-floor');
  assert.equal(limited[`bank:${trio[1]}`].state,'below-floor');
  assert.equal(limited[`bank:${trio[2]}`].state,'refused');
  assert.doesNotMatch(await page.locator('#mapLayers').textContent(),/0\.4|>=3|<img|onerror/);
  assert.equal(await page.locator('#mapLayers img').count(),0);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({defaultTrio:true,lesionOrigin:true,proximityColour:true,ownedMeasurements:true,recolourRemovalAndCancellation:true,syntheticFaults:true,pageErrors:errors}));
}finally{await browser.close();}
