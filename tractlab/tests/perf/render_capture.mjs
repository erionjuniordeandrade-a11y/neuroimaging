#!/usr/bin/env node
// Before/after look captures of the demo case on the machine's real GPU (Metal ANGLE).
// Not a regression gate: visual_regression.mjs owns pixel baselines. Images go outside the repo.
// Usage: node tests/perf/render_capture.mjs --out <dir> [--tag before] [--url http://127.0.0.1:18995/index.html]
//        [--ao-diff] [--veils] [--views L,S,A] [--view-veil Mid]
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
import path from 'node:path';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {PNG} from 'pngjs';
import pixelmatch from 'pixelmatch';

const arg=(name,fallback)=>{
  const at=process.argv.indexOf(`--${name}`);
  return at>=0&&process.argv[at+1]?process.argv[at+1]:fallback;
};
const source=new URL(arg('url','http://127.0.0.1:18995/index.html'));
const out=arg('out');
const tag=arg('tag','capture');
assert.ok(out,'--out is required (a directory outside the repository)');
assert.equal(source.hostname,'127.0.0.1','captures only accept the loopback demo server');
const identity=await (await fetch(new URL('/api/bootstrap',source))).json();
assert.equal(identity.caseId,'demo-leipzig-sub-010005','refusing to capture a non-demo case');
source.searchParams.set('test','1');
await mkdir(out,{recursive:true});

const browser=await chromium.launch({headless:true,args:['--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist']});
const page=await (await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:2})).newPage();
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
page.on('console',message=>{if(message.type()==='error'&&!/status of 404/.test(message.text()))errors.push(message.text());});

async function settle(){
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady&&window.__tractlabTest?.settled
    &&window.__tractlabTest?.lineCount>0&&!window.__tractlabTest?.pipeline?.interacting,
    undefined,{timeout:120_000});
  await page.waitForTimeout(600);
}
async function shot(name){
  await settle();
  const file=path.join(out,`${tag}-${name}.png`);
  await page.locator('#scene').screenshot({path:file,animations:'disabled'});
  console.log(file);
}

try{
  await page.goto(source.href,{waitUntil:'load',timeout:120_000});
  console.log('gl:',await page.evaluate(()=>{
    const gl=document.getElementById('gl').getContext('webgl2');
    const ext=gl?.getExtension('WEBGL_debug_renderer_info');
    return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):'unknown';
  }));
  await shot('default');
  if(process.argv.includes('--ao-diff')){
    // Prove the AO pass: same frame with strength 0, then the share of pixels that changed.
    await page.evaluate(()=>window.__tractlabLightingProbe.setAoStrength(0));
    await page.waitForTimeout(400);
    await shot('ao-off');
    const [on,off]=await Promise.all(['default','ao-off'].map(async n=>PNG.sync.read(await readFile(path.join(out,`${tag}-${n}.png`)))));
    const changed=pixelmatch(on.data,off.data,null,on.width,on.height,{threshold:0.02});
    console.log(`ao-diff: ${changed} px changed (${(100*changed/(on.width*on.height)).toFixed(2)}%)`);
    await page.evaluate(()=>window.__tractlabLightingProbe.setAoStrength(.45));
  }
  if(process.argv.includes('--veils')){
    for(const veil of ['Mid','Solid','Ghost']){
      await page.evaluate(id=>document.getElementById(id).click(),`hullVeil${veil}`);
      if(veil!=='Ghost')await shot(`veil-${veil.toLowerCase()}`);
    }
  }
  const viewVeil=arg('view-veil');
  if(viewVeil)await page.evaluate(id=>document.getElementById(id).click(),`hullVeil${viewVeil}`);
  for(const view of (arg('views','L,S,A')).split(',')){
    await page.locator(`#viewCluster [data-view="${view}"]`).click();
    await shot(`view-${view}`);
  }
  assert.deepEqual(errors,[],'browser errors');
}finally{await browser.close();}
