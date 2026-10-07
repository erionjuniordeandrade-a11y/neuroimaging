#!/usr/bin/env node
// BVH tube picking gate: each tube hit must map to the streamline geometrically nearest the
// hit point (within the tube radius). Runs only against the loopback demo case.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const arg=(name,fallback)=>{
  const eq=process.argv.find(value=>value.startsWith(`--${name}=`));
  if(eq)return eq.slice(name.length+3);
  const at=process.argv.indexOf(`--${name}`);
  return at>=0&&process.argv[at+1]&&!process.argv[at+1].startsWith('--')?process.argv[at+1]:fallback;
};
const url=new URL(arg('url','http://127.0.0.1:18995/index.html'));
assert.equal(url.hostname,'127.0.0.1','pick gate only accepts loopback');
const bootstrap=await fetch(new URL('/api/bootstrap',url));
assert.equal(bootstrap.status,200,'demo bootstrap unavailable');
assert.equal((await bootstrap.json()).caseId,'demo-leipzig-sub-010005','refusing non-demo case');
url.searchParams.set('test','1');
const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
const consoleErrors=[],pageErrors=[];
page.on('console',message=>{if(message.type()==='error'&&!/status of 404/.test(message.text()))consoleErrors.push(message.text());});
page.on('pageerror',error=>pageErrors.push(error.message));
try{
  await page.goto(url.href,{waitUntil:'load',timeout:120_000});
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady&&window.__tractlabTest?.settled
    &&window.__tractlabTest?.lineCount>0,undefined,{timeout:120_000});
  const result=await page.evaluate(()=>{
    const probe=window.__tractlabPickProbe,rows=[];
    for(const point of probe.candidates(100)){
      const detail=probe.hitDetail(point.x,point.y);
      if(!detail)continue;
      rows.push({displayIndex:detail.displayIndex,nearest:detail.nearest[0],radius:detail.radius});
      if(rows.length>=20)break;
    }
    return rows;
  });
  const agree=result.filter(row=>row.nearest?.displayIndex===row.displayIndex&&row.nearest.dist<=row.radius*1.05);
  console.log(`BVH_GEOMETRY ${agree.length}/${result.length} hits map to the nearest streamline within the tube radius`);
  assert.equal(result.length,20,'need 20 BVH-positive pointer positions');
  assert.ok(agree.length>=18,`only ${agree.length}/20 hits map to the nearest streamline`);
  const target=await page.evaluate(()=>window.__tractlabPickProbe.projectTarget());
  assert.ok(target,'no pickable projected target');
  const expected=await page.evaluate(({x,y})=>window.__tractlabPickProbe.pickAt(x,y),target);
  assert.ok(expected,'projected target has no BVH hit');
  await page.waitForFunction(()=>!window.__tractlabTest.pipeline?.interacting);
  await page.mouse.move(target.x,target.y);
  await page.waitForFunction(({layerKey,displayIndex})=>{
    const hover=window.__tractlabTest.hover;
    return hover?.layerKey===layerKey&&hover.displayIndex===displayIndex&&
      !document.querySelector('#streamlineHoverTip')?.hidden;
  },expected);
  await page.mouse.move(0,0);
  await page.waitForFunction(()=>window.__tractlabTest.hover===null&&
    document.querySelector('#streamlineHoverTip')?.hidden);
  console.log(`HOVER PASS: layer=${expected.layerKey} displayIndex=${expected.displayIndex}; empty clears`);
  await page.mouse.click(target.x,target.y);
  await page.waitForFunction(()=>!document.querySelector('#streamlineInspector')?.hidden);
  const shown=await page.locator('#streamlineInspector').evaluate(element=>Number(element.dataset.displayIndex));
  assert.equal(shown,expected.displayIndex,'inspector display index differs from the tube hit');
  console.log(`PICK_INSPECTOR PASS: layer=${expected.layerKey} displayIndex=${expected.displayIndex}`);
  const letters=await page.locator('#orientationCube text').allTextContents();
  assert.ok(letters.length>=3&&letters.every(letter=>'LRAPSI'.includes(letter)),'orientation letters missing');
  console.log(`ORIENTATION PASS: ${letters.join('')}`);
  assert.deepEqual(consoleErrors,[],'console errors');
  assert.deepEqual(pageErrors,[],'page errors');
  console.log('PASS bvh pick check');
}finally{await browser.close();}
