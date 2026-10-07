#!/usr/bin/env node
// Profile panel against SYNTHETIC responses.
//
// Every payload here is built in this file and served through page.route, so
// the refusal paths, the retry path and the abort race are exercised without
// needing a case that happens to be broken. The demo smoke
// (profile_panel_smoke.mjs) still covers the real route end to end; this one
// covers what a real case cannot be asked to produce on demand.
//
//   node tests/perf/profile_panel_mock.mjs --url=http://127.0.0.1:18994/
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const url=new URL(process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18994/');
url.hash='test';
const BANK='bank_cst_r';
const LAYER=`bank:${BANK}`;
const SHA=(seed)=>seed.repeat(64).slice(0,64);

function payload(overrides={}){
  const n=100;
  const column=value=>Array.from({length:n},()=>value);
  return {
    bank_id:BANK,
    bank_path:'tracts/bank/cst_r_motor_pons.tck',
    bank_sha256:SHA('8bf1f924'),
    scalar:'fa',
    scalar_path:'work/profiles/fa_casemask.nii.gz',
    scalar_sha256:SHA('7cbc153a'),
    n_points:n,
    n_streamlines:8303,
    orientation:{rule:'inferior',anchor_mm:[-3.1,-20.8,-73.0],family:'cst',side:'r',n_flipped:6194},
    active_derivation:'d1',
    resample_method:'numpy_arc_length',
    claim:'descriptive profile; not a normative abnormality score',
    nodes:{
      median:Array.from({length:n},(_,i)=>0.3+0.2*Math.sin(i/9)),
      p25:column(0.25),p75:column(0.6),mean:column(0.42),
      n:column(8303),n_valid:column(8303),n_zero:column(0),n_nan:column(0),
      zero_or_missing_flag:column(false),
    },
    zero_or_missing_threshold:0.2,
    zero_or_missing_warning:null,
    streamline_mean_histogram:{counts:[],edges:[],n:0},
    lesion_distance:{track_mm:null,reason:'no lesion mask'},
    lesion_path:null,
    lesion_sha256:null,
    ...overrides,
  };
}

const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1400,height:900}});
const errors=[];
page.on('pageerror',error=>errors.push(error.message));

// One mutable script the route handler follows, so each stage can decide what
// the next /api/profile request receives.
let plan={mode:'ok'};
const seen=[];
await page.route('**/api/profile/**',async route=>{
  const target=new URL(route.request().url());
  const scalar=target.searchParams.get('scalar')||'fa';
  seen.push({path:target.pathname,scalar,mode:plan.mode});
  const json=(status,body)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
  if(plan.mode==='busy') return json(409,{error:'a track job is already running'});
  if(plan.mode==='wrong-bank') return json(200,payload({bank_id:'bank_fat_r'}));
  if(plan.mode==='wrong-scalar') return json(200,payload({scalar:'md'}));
  if(plan.mode==='wrong-derivation') return json(200,payload({active_derivation:'d9'}));
  if(plan.mode==='no-hash') return json(200,payload({scalar_sha256:'7cbc153a'}));
  if(plan.mode==='half-roi') return json(200,payload({
    orientation:{rule:'roi_centroid',anchor_mm:[1,2,3],family:'cst',side:'r',n_flipped:10,
      roi_path:'tracts/roi/cst_r_pons_dil1.nii.gz'},
  }));
  if(plan.mode==='short-nodes'){
    const body=payload();
    body.nodes.p25=body.nodes.p25.slice(0,40);
    return json(200,body);
  }
  if(plan.mode==='null-body') return route.fulfill({status:200,contentType:'application/json',body:'null'});
  if(plan.mode==='lesion') return json(200,payload({
    lesion_path:'nifti/lesion.nii.gz',
    lesion_sha256:SHA('abcd1234'),
    lesion_distance:{track_mm:Array.from({length:100},(_,i)=>i===75?0.6:2+Math.abs(i-75)*0.3),reason:null},
  }));
  if(plan.mode==='slow'){
    await new Promise(resolve=>setTimeout(resolve,plan.delayMs??1500));
    return json(200,payload());
  }
  return json(200,payload());
});

const state=()=>page.evaluate(()=>window.__tractlabTest.profile);
const settle=(predicate,arg)=>page.waitForFunction(predicate,arg,{timeout:30000});
const errorLine=()=>page.locator('#profileError').textContent();
// goto() to a URL that differs only by its #hash does NOT reload the document,
// so every load after the first has to be an explicit reload.
let opened=false;
const reload=async()=>{
  if(opened) await page.reload({waitUntil:'load'});
  else{ await page.goto(url.href,{waitUntil:'load'}); opened=true; }
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
};
const refocus=()=>page.evaluate(key=>{
  document.querySelector(`[data-layer-key="${key}"] .map-focus`)?.click();
},LAYER);
const report={};

try{
  // --- malformed and mismatched 200s are refused, one honest line each ---
  const refusals=[
    ['wrong-bank','bank_mismatch','The server answered for a different tract'],
    ['wrong-scalar','scalar_mismatch','The server answered for a different scalar'],
    ['wrong-derivation','derivation_mismatch','This profile belongs to another derivation of the case'],
    ['no-hash','provenance_missing','This profile arrived without its full provenance'],
    ['half-roi','provenance_missing','This profile arrived without its full provenance'],
    ['short-nodes','nodes_malformed','This profile arrived incomplete'],
    ['null-body','payload_invalid','The server sent no readable profile'],
  ];
  for(const [mode,code,line] of refusals){
    // Arm the route before the load so the panel's own first request is the
    // one that receives the bad payload.
    plan={mode};
    await reload();
    await settle(c=>window.__tractlabTest.profile?.status==='error'&&window.__tractlabTest.profile.code===c,code);
    assert.equal((await state()).code,code,mode);
    assert.equal(await errorLine(),line,mode);
    // A refused payload is never drawn and never cached as ready.
    assert.equal(await page.locator('#profileChart').count(),0,`${mode} draws no chart`);
    assert.equal(await page.locator('#profileProvCard').count(),0,`${mode} shows no provenance`);
  }
  report.refusals=refusals.map(([mode])=>mode);

  // --- a 409 is transient: it is not cached, and refocus retries ---
  plan={mode:'busy'};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='error'
    &&window.__tractlabTest.profile.code==='http_409');
  const before=seen.length;
  plan={mode:'ok'};
  await refocus();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  assert.ok(seen.length>before,'refocus re-issued the request instead of reusing the refusal');
  assert.equal((await state()).nodes,100);
  report.retryRequests=seen.length-before;

  // --- an aborted key can restart at once (A, B, A back to back) ---
  plan={mode:'slow',delayMs:1500};
  await reload();
  await page.waitForTimeout(150);
  // Abort by switching scalar twice in quick succession, ending back on FA.
  await page.locator('#profileScalar_md').click();
  await page.waitForTimeout(80);
  await page.locator('#profileScalar_fa').click();
  plan={mode:'ok'};
  await settle(()=>window.__tractlabTest.profile?.status==='ready'
    &&window.__tractlabTest.profile.scalar==='fa');
  assert.equal((await state()).nodes,100,'the restarted key resolved rather than hanging on a dead request');

  // --- a lesion track draws its closest approach from the served numbers ---
  plan={mode:'lesion'};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  await page.locator('#profileChart').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#profileClosestLabel').textContent(),'closest 0.6 mm, node 75');
  assert.equal(await page.locator('#profileClosestMarker').count(),1);
  assert.equal(await page.locator('#profileClosestTick').count(),1,'the same node is ticked on the scalar chart');
  assert.equal(await page.locator('.profile-no-lesion').count(),0);
  await page.locator('#profileChipInfo').click();
  const card=await page.locator('#profileProvCard').textContent();
  assert.ok(card.includes('lesion mask: nifti/lesion.nii.gz'),'the card names the lesion mask');
  assert.ok(card.includes(`lesion sha256: ${SHA('abcd1234').slice(0,8)}`),'the card carries the lesion digest');
  assert.ok(/near band: under \d/.test(card),'the near band threshold lives in the card, not the caption');
  report.closest='closest 0.6 mm, node 75';

  // --- a bank load issued during a profile fetch is not refused ---
  plan={mode:'slow',delayMs:2000};
  await reload();
  await page.waitForTimeout(200);
  const load=await page.evaluate(async bank=>{
    const health=await (await fetch('/api/health')).json();
    const started=performance.now();
    const response=await fetch('/api/bank/load',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({bankId:bank,gridId:health.gridId,volumeId:health.volumeId})});
    return {status:response.status,ms:Math.round(performance.now()-started)};
  },BANK);
  assert.equal(load.status,200,`a bank load during a profile fetch answered HTTP ${load.status}`);
  assert.ok(load.ms<10000,`the bank load took ${load.ms} ms`);
  report.bankLoadDuringProfile=load;

  assert.deepEqual(errors,[],'no page errors');
  console.log(JSON.stringify({stage:'profile-panel-mock',url:url.href,requests:seen.length,...report}));
}catch(error){
  console.error(JSON.stringify({stage:'profile-panel-mock-failure',message:error.message,
    plan,profile:await state().catch(()=>null),seen:seen.slice(-4),errors}));
  throw error;
}finally{
  await browser.close();
}
