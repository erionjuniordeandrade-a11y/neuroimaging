#!/usr/bin/env node
// Along-tract profile panel, against a served case (default: the demo subject
// on 18994). Reads the same debug seam the other harnesses use: the panel's
// own state on window.__tractlabTest, and the geometry probe that only exists
// under the ?test / #test flag.
//
//   node tests/perf/profile_panel_smoke.mjs --url=http://127.0.0.1:18994/
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const url=new URL(process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18994/');
url.hash='test';
const bankId=process.argv.find(a=>a.startsWith('--bank='))?.slice(7)||'bank_cst_r';
const layerKey=`bank:${bankId}`;

const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1400,height:900}});
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
page.on('console',message=>{if(message.type()==='error'&&!/Failed to load resource/.test(message.text()))errors.push(message.text());});

const profileState=()=>page.evaluate(()=>window.__tractlabTest.profile);
const probe=(method,...args)=>page.evaluate(
  ([name,rest])=>window.__tractlabProfileProbe[name](...rest),[method,args]);
const medianPointCounts=()=>page.evaluate(()=>[...document.querySelectorAll('#profileChart .profile-median')]
  .map(node=>node.getAttribute('points').trim().split(/\s+/).length));
const lineColours=(line,k)=>page.evaluate(([key,index,count])=>
  [...Array(count).keys()].map(point=>window.__tractlabProfileProbe.vertexColour(key,index,point)),
  [layerKey,line,k]);
const report={};

try{
  await page.goto(url.href,{waitUntil:'load'});
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});

  // Focus the named bank, then wait for its profile.
  await page.waitForFunction(key=>window.__tractlabTest.layerKeys.includes(key),layerKey,{timeout:120000});
  await page.evaluate(key=>{
    const card=document.querySelector(`[data-layer-key="${key}"] .map-focus`);
    if(card)card.click();
  },layerKey);
  await page.waitForFunction(
    id=>window.__tractlabTest.profile?.status==='ready'&&window.__tractlabTest.profile.bankId===id,
    bankId,{timeout:120000});

  const ready=await profileState();
  assert.equal(ready.scalar,'fa','the panel opens on FA');
  assert.equal(ready.nodes,100,'the served profile has 100 nodes');
  report.fa={...ready};

  // Every served node is drawn: the band may be cut into segments by the
  // served zero-or-missing flags, so the total across segments is the check.
  const faPoints=await medianPointCounts();
  assert.equal(faPoints.reduce((sum,count)=>sum+count,0),100,'the FA median draws all 100 nodes');
  report.faSegments=faPoints;

  const labels=await page.evaluate(()=>[
    document.querySelector('.profile-axis-start')?.textContent,
    document.querySelector('.profile-axis-end')?.textContent,
  ]);
  report.endLabels=labels;
  if(bankId.startsWith('bank_cst_')) assert.deepEqual(labels,['inferior','superior'],'cst end labels come from the orientation rule');
  else assert.ok(labels[0]&&labels[1],'both end labels are present');
  assert.equal(await page.locator('#profileClaim').textContent(),
    'descriptive profile; not a normative abnormality score','the served claim is shown verbatim');

  // Design pass: the dock summary is the only title, the scalar names the y
  // axis (not the middle of the x axis), and the hint sits above the chart.
  assert.equal(await page.locator('#profilePanel .section-title').count(),0,'the title is not printed twice');
  assert.equal(await page.locator('#profileScalarName').textContent(),'FA','the y axis names the scalar');
  assert.equal(await page.locator('#profilePanel .profile-axis').innerText(),'inferior\nsuperior');
  assert.equal(await page.locator('#profileHint').textContent(),'Drag to highlight');

  // Provenance is visible without opening anything, and the card carries the
  // served paths and digests in full.
  assert.match(await page.locator('#profileChipBankHash').textContent(),/^bank [0-9a-f]{8}$/);
  assert.match(await page.locator('#profileChipScalarHash').textContent(),/^fa [0-9a-f]{8}$/);
  await page.locator('#profileChipInfo').click();
  const card=await page.locator('#profileProvCard').textContent();
  for(const row of ['bank file:','bank sha256:','scalar file:','scalar sha256:','orientation rule:','derivation:'])
    assert.ok(card.includes(row),`the provenance card lists ${row}`);
  assert.equal(/unrecorded/.test(card),false,'no provenance row is rendered as "unrecorded"');
  report.provenance=card.split('\n');
  await page.locator('#profileChipInfo').click();

  // Drag an interval and check the tube colour buffer, not the panel text.
  await page.locator('#profileChart').scrollIntoViewIfNeeded();
  const info=await probe('info',layerKey);
  assert.ok(info&&info.k>=2,'the focused layer has baked tube geometry');
  assert.equal(info.hasMask,false,'no interval is applied before the drag');
  const before=await lineColours(0,info.k);

  const box=await page.locator('#profileChart').boundingBox();
  await page.mouse.move(box.x+box.width*0.30,box.y+box.height/2);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width*0.55,box.y+box.height/2,{steps:8});
  await page.mouse.up();
  await page.waitForFunction(key=>window.__tractlabProfileProbe.info(key)?.hasMask===true,layerKey,{timeout:15000});

  const masked=await probe('info',layerKey);
  assert.ok(Array.isArray(masked.interval),'the drag recorded a node interval');
  assert.ok(masked.interval[0]<masked.interval[1],'the drag selected more than one node');
  assert.ok(masked.maskSum>0&&masked.maskSum<masked.lineCount*masked.k,
    'the interval covers part of the geometry, not all or none');
  report.interval=masked.interval;
  report.maskSum=masked.maskSum;

  const after=await lineColours(0,info.k);
  const mask=await page.evaluate(([key,count])=>
    [...Array(count).keys()].map(point=>window.__tractlabProfileProbe.maskAt(key,0,point)),[layerKey,info.k]);
  const inside=mask.findIndex(value=>value===1);
  const outside=mask.findIndex(value=>value===0);
  assert.ok(inside>=0&&outside>=0,'line 0 has both in-interval and out-of-interval points');
  assert.deepEqual(after[inside],before[inside],'in-interval vertices keep their evidence colour');
  assert.notDeepEqual(after[outside],before[outside],'out-of-interval vertices fall to the ghost');
  const ghost=after[outside];
  assert.ok(ghost[0]===ghost[1]||Math.abs(ghost[0]-ghost[2])<0.2,'the ghost is a grey, not a new hue');

  const intervalLine=await page.locator('#profileIntervalLine').textContent();
  assert.match(intervalLine,new RegExp(`Nodes ${masked.interval[0]} to ${masked.interval[1]}`));

  // Escape clears the interval and restores every vertex exactly.
  await page.keyboard.press('Escape');
  await page.waitForFunction(key=>window.__tractlabProfileProbe.info(key)?.hasMask===false,layerKey,{timeout:15000});
  assert.deepEqual(await lineColours(0,info.k),before,'clearing restores the baked colours exactly');

  // MD: the route answers, or the panel says so in one line.
  await page.locator('#profileScalar_md').click();
  await page.waitForFunction(()=>window.__tractlabTest.profile?.scalar==='md'
    &&window.__tractlabTest.profile.status!=='loading',undefined,{timeout:120000});
  const md=await profileState();
  report.md={...md};
  if(md.status==='ready'){
    assert.equal(md.nodes,100,'the MD profile has 100 nodes');
    const mdPoints=await medianPointCounts();
    assert.equal(mdPoints.reduce((sum,count)=>sum+count,0),100,'the MD median draws all 100 nodes');
    assert.match(await page.locator('#profileScalarName').textContent(),/MD/,'the y axis declares MD units');
    report.mdSegments=mdPoints;
  }else{
    assert.equal(md.status,'error');
    assert.equal(await page.locator('#profileError').textContent(),'MD map not available for this case');
  }

  // Unfocus: with no layer left there is no panel to read.
  await page.evaluate(()=>{
    for(const button of document.querySelectorAll('#mapLayers .map-remove'))button.click();
  });
  await page.waitForFunction(()=>window.__tractlabTest.profile?.status==='absent',undefined,{timeout:30000});
  assert.equal(await page.locator('#profilePanel').count(),0,'the panel is gone once no tract is focused');
  assert.equal(await page.locator('#profileDetails').isVisible(),false,'the dock section hides with it');

  assert.deepEqual(errors,[],'no page errors');
  console.log(JSON.stringify({stage:'profile-panel',url:url.href,...report}));
}catch(error){
  console.error(JSON.stringify({stage:'profile-panel-failure',message:error.message,
    profile:await profileState().catch(()=>null),errors}));
  throw error;
}finally{
  await browser.close();
}
