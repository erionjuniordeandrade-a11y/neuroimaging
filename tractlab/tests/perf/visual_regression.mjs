#!/usr/bin/env node
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import pixelmatch from 'pixelmatch';
import {PNG} from 'pngjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const arg=(name,fallback)=>{
  const eq=process.argv.find(value=>value.startsWith(`--${name}=`));
  if(eq)return eq.slice(name.length+3);
  const at=process.argv.indexOf(`--${name}`);
  return at>=0&&process.argv[at+1]&&!process.argv[at+1].startsWith('--')?process.argv[at+1]:fallback;
};
const source=new URL(arg('url','http://127.0.0.1:18995/index.html'));
const out=path.resolve(arg('out','output/visual'));
const update=process.argv.includes('--update');
// Baselines are rendered screenshots and never enter the repository (the PHI pre-commit hook
// refuses image files). They live per machine under ~/.cache/tractlab; run once with --update.
const baselines=path.resolve(arg('baselines',path.join(process.env.HOME||root,'.cache/tractlab/visual-baselines')));
assert.equal(source.hostname,'127.0.0.1','visual gate only accepts the loopback demo server');
const bootstrap=await fetch(new URL('/api/bootstrap',source));
assert.equal(bootstrap.status,200,'demo bootstrap unavailable');
const identity=await bootstrap.json();
assert.equal(identity.caseId,'demo-leipzig-sub-010005','refusing to screenshot a non-demo case');
source.searchParams.set('test','1');

await mkdir(out,{recursive:true});
if(update)await mkdir(baselines,{recursive:true});
const browser=await chromium.launch({headless:true,args:[
  '--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader',
]});
const context=await browser.newContext({viewport:{width:1280,height:800},deviceScaleFactor:1});
const page=await context.newPage(),errors=[];
page.on('pageerror',error=>errors.push(error.message));
page.on('console',message=>{if(message.type()==='error'&&!/status of 404/.test(message.text()))errors.push(message.text());});

async function settle(){
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady&&window.__tractlabTest?.settled
    &&window.__tractlabTest?.lineCount>0&&!window.__tractlabTest?.pipeline?.interacting,
    undefined,{timeout:120_000});
  await page.evaluate(async()=>{await document.fonts.ready;});
  await page.waitForTimeout(300);
}
async function scene(name){
  await settle();
  const screenshot=await page.screenshot({animations:'disabled',caret:'hide'});
  const baseline=path.join(baselines,`${name}.png`);
  if(update){await writeFile(baseline,screenshot);console.log(`BASELINE ${name}: ${baseline}`);return;}
  let stored;
  try{stored=await readFile(baseline);}catch(error){
    if(error.code==='ENOENT')throw Error(`Missing baseline ${baseline}; run once with --update on this machine`);throw error;}
  const expected=PNG.sync.read(stored),actual=PNG.sync.read(screenshot);
  assert.equal(actual.width,1280);assert.equal(actual.height,800);
  assert.equal(expected.width,actual.width);assert.equal(expected.height,actual.height);
  const diff=new PNG({width:actual.width,height:actual.height});
  const changed=pixelmatch(expected.data,actual.data,diff.data,actual.width,actual.height,{threshold:.1});
  const fraction=changed/(actual.width*actual.height);
  await writeFile(path.join(out,`${name}.png`),screenshot);
  await writeFile(path.join(out,`${name}.diff.png`),PNG.sync.write(diff));
  console.log(`${fraction>.005?'FAIL':'PASS'} ${name}: ${changed}/${actual.width*actual.height} pixels (${(fraction*100).toFixed(3)}%)`);
  assert.ok(fraction<=.005,`${name} changed more than 0.5% of pixels`);
}

try{
  await page.goto(source.href,{waitUntil:'load',timeout:120_000});
  await scene('demo-default');
  const target=await page.evaluate(()=>window.__tractlabPickProbe?.projectTarget?.());
  assert.ok(target&&Number.isFinite(target.x)&&Number.isFinite(target.y),'no deterministic tube target');
  await page.mouse.click(target.x,target.y);
  await page.waitForFunction(()=>!document.querySelector('#streamlineInspector')?.hidden);
  await scene('tract-selected');
  // The public demo case ships no lesion map, so the third scene is a canonical view change
  // (camera-controls setLookAt path) rather than the lesion map panel.
  await page.locator('#viewCluster [data-view="S"]').click();
  await page.waitForFunction(()=>!window.__tractlabTest?.pipeline?.interacting);
  await scene('view-superior');
  assert.deepEqual(errors,[],'browser errors');
  console.log(`${update?'BASELINES':'PASS'} visual gate: 3 scenes, demo case confirmed`);
}finally{await browser.close();}
