#!/usr/bin/env node
// Profile panel with NO case server at all.
//
// Every request the page makes — the document, its modules, the vendored
// three.js, and every /api route — is fulfilled from this file. The viewer
// runs against a 10x10x10 volume and six straight synthetic streamlines, so
// the refusal paths can be driven exactly and nothing here depends on a case
// existing, on a port being free, or on patient data.
//
//   node tests/perf/profile_panel_synthetic.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const VIEWER=fileURLToPath(new URL('../../viewer/', import.meta.url));
const ORIGIN='http://tractlab.synthetic';
const BANK='bank_cst_r';
const LAYER=`bank:${BANK}`;
const BUILD='b'.repeat(64);
const CASE_SOURCE='c'.repeat(64);
const SOURCE_A='a'.repeat(64);
const SOURCE_B='d'.repeat(64);
const DIM=10, N_VOX=DIM*DIM*DIM;
const LINES=6, K=32, N_POINTS=100;
const SERVED_CLAIM='descriptive profile; not a normative abnormality score';

// ---- synthetic case -------------------------------------------------------

const b0=Buffer.alloc(N_VOX);
for(let i=0;i<N_VOX;i++) b0[i]=40+(i%160);
// Identity affine: voxel index is millimetres, so the tubes below sit inside
// the volume and the viewer's world-space maths needs no special case.
const AFFINE=[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

/** Six straight streamlines running inferior to superior, packed line-major. */
function tubeGeometry(){
  const positions=new Float32Array(LINES*K*3);
  let at=0;
  for(let line=0;line<LINES;line++){
    const x=2+line*1.2, y=4+(line%2)*0.8;
    for(let point=0;point<K;point++){
      const t=point/(K-1);
      positions[at++]=x;
      positions[at++]=y;
      positions[at++]=1+t*7;   // z from 1 to 8 mm
    }
  }
  return Buffer.from(positions.buffer);
}
const TUBES=tubeGeometry();

/** A box hull enclosing the volume: 8 vertices, 12 triangles. loadHull() needs
 * a real surface (it reads X-center / X-radius unguarded), so the fixture
 * provides the smallest one that encloses the synthetic tubes. */
function hullSurface(){
  const corners=[];
  for(const x of [0,DIM]) for(const y of [0,DIM]) for(const z of [0,DIM]) corners.push([x,y,z]);
  const vertices=new Float32Array(corners.flat());
  // index bits: x=4, y=2, z=1
  const faces=new Uint32Array([
    0,1,3, 0,3,2,   4,7,5, 4,6,7,
    0,4,5, 0,5,1,   2,3,7, 2,7,6,
    0,2,6, 0,6,4,   1,5,7, 1,7,3,
  ]);
  return {
    vertexCount:corners.length, faceCount:faces.length/3,
    body:Buffer.concat([Buffer.from(vertices.buffer),Buffer.from(faces.buffer)]),
  };
}
const HULL=hullSurface();

function profilePayload(overrides={}){
  const column=value=>Array.from({length:N_POINTS},()=>value);
  return {
    bank_id:BANK,
    bank_path:'tracts/bank/cst_r.tck',
    bank_sha256:'8'.repeat(64),
    scalar:'fa',
    scalar_path:'work/profiles/fa.nii.gz',
    scalar_sha256:'7'.repeat(64),
    n_points:N_POINTS,
    n_streamlines:LINES,
    orientation:{rule:'inferior',anchor_mm:[2,4,1],family:'cst',side:'r',n_flipped:0},
    active_derivation:'d1',
    mrtrix_version:null,
    resample_method:'numpy_arc_length',
    claim:SERVED_CLAIM,
    nodes:{
      median:Array.from({length:N_POINTS},(_,i)=>0.3+0.2*Math.sin(i/9)),
      p25:column(0.25), p75:column(0.6), mean:column(0.42),
      n:column(LINES), n_valid:column(LINES), n_zero:column(0), n_nan:column(0),
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

const BOOTSTRAP={
  status:'ok', runtimeStatus:'ok',
  viewerProtocol:1, expectedViewerProtocol:1,
  buildId:BUILD, expectedClientBuildId:BUILD,
  caseId:'synthetic', gridId:'grid-synthetic', volumeId:'volume-synthetic',
  recipeHash:'recipe-synthetic', caseSourceHash:CASE_SOURCE, spaceId:'synthetic',
  presetCount:0, bankCount:1, hasFilterBank:false, filterBankCount:0,
  hasLesion:false, sift2BankCount:0,
  recovery:{perilesional:false,radiusMinMm:3,radiusMaxMm:20,radiusDefaultMm:8,
    note:'Fibres within radius of lesion — not named-bundle proof'},
  displayBias:{nearLesionFracDefault:0.5,nearRadiusMmDefault:12,
    note:'Display-only sample bias toward peri-lesional streamlines'},
  hasFa:false, hasDec:false, hasT1:false, hasPriors:false, priorCount:0,
  hasParcellation:false, busy:false, activeJobId:null, activeJobAgeS:null,
  uncertainty:'synthetic fixture; rigid T1 only; aid not navigation',
  liveDefaults:{cutoff:0.08,cutoffMin:0.02,cutoffMax:0.4,
    cutoffKind:'FOD amplitude (not tensor FA)',angle:45,minlength:20,
    density:'normal',densities:{normal:{seeds:1,select:1}}},
};

const CATALOG={
  banks:[{id:BANK,label:'True CST-R (synthetic)',nStreamlines:LINES,
    engine:'SYNTHETIC',note:'Synthetic fixture. Research only.',default:true,
    role:'true_cst',hasSift2:false}],
  defaultBankId:BANK,
};

// ---- mutable plan the routes follow --------------------------------------

let plan={profile:'ok', bankSourceHash:SOURCE_A, delayMs:0};
const seen=[];

const STATIC_TYPES={'.html':'text/html','.js':'application/javascript',
  '.css':'text/css','.svg':'image/svg+xml','.json':'application/json'};

async function staticBody(pathname){
  const rel=pathname==='/'?'index.html':pathname.replace(/^\/+/,'');
  if(rel.includes('..')) return null;
  let body;
  try{ body=await readFile(VIEWER+rel); }catch{ return null; }
  const dot=rel.lastIndexOf('.');
  const type=STATIC_TYPES[rel.slice(dot)]||'application/octet-stream';
  if(rel==='index.html'){
    // The real server injects the boot fingerprint from memory rather than
    // from the HTML on disk (runtime_identity.inject_document_meta).
    const meta=`<meta name="tractlab-build" content="${BUILD}">`
      +'<meta name="tractlab-protocol" content="1">';
    body=Buffer.from(body.toString('utf8').replace('</head>',`${meta}</head>`),'utf8');
  }
  return {body,type};
}

function bankLoadHeaders(){
  return {
    'X-encoding':'float32le',
    'X-lineCount':String(LINES),
    'X-pointsPerLine':String(K),
    'X-layout':'line-major, then point, then xyz',
    'X-spaceId':'synthetic',
    'X-sourcePopulation':`bank:${BANK}`,
    'X-outcome':'ok',
    'X-bankId':BANK,
    'X-engine':'SYNTHETIC | prebuilt',
    'X-nReturned':String(LINES),
    'X-nDisplayed':String(LINES),
    'X-nAnalytic':String(LINES),
    'X-nAnalyticFull':String(LINES),
    'X-clearanceRefusal':'synthetic fixture records no geometric floor',
    ...(plan.bankSourceHash?{'X-bankSourceHash':plan.bankSourceHash}:{}),
  };
}

const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1400,height:900}});
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
if(process.env.TRACTLAB_SYNTHETIC_DEBUG){
  page.on('console',m=>console.error('[console]',m.type(),m.text()));
  page.on('requestfailed',r=>console.error('[failed]',r.url(),r.failure()?.errorText));
}

await page.route('**/*',async route=>{
  const target=new URL(route.request().url());
  const path=target.pathname;
  const json=(status,body)=>route.fulfill({status,contentType:'application/json',
    body:JSON.stringify(body)});

  if(path.startsWith('/api/profile/')){
    const scalar=target.searchParams.get('scalar')||'fa';
    seen.push({scalar,mode:plan.profile});
    if(plan.delayMs) await new Promise(resolve=>setTimeout(resolve,plan.delayMs));
    switch(plan.profile){
      case 'busy': return json(409,{error:'a track job is already running'});
      case 'wrong-bank': return json(200,profilePayload({bank_id:'bank_fat_r'}));
      case 'wrong-scalar': return json(200,profilePayload({scalar:'md'}));
      case 'wrong-derivation': return json(200,profilePayload({active_derivation:'d9'}));
      case 'orientation-family':
        return json(200,profilePayload({orientation:{rule:'inferior',anchor_mm:[0,0,0],
          family:'fat',side:'r',n_flipped:0}}));
      case 'orientation-side':
        return json(200,profilePayload({orientation:{rule:'inferior',anchor_mm:[0,0,0],
          family:'cst',side:'l',n_flipped:0}}));
      case 'orientation-rule':
        return json(200,profilePayload({orientation:{rule:'superior',anchor_mm:[0,0,0],
          family:'cst',side:'r',n_flipped:0}}));
      case 'nonfinite-median':{
        // JSON has no Infinity literal, but 1e999 parses back as Infinity —
        // a value the wire can carry and the panel must never draw. Written
        // as text because JSON.stringify would emit it as null.
        const body=profilePayload();
        const median=body.nodes.median.map((value,i)=>i===7?'1e999':value.toFixed(4));
        return route.fulfill({status:200,contentType:'application/json',
          body:JSON.stringify(body).replace(/"median":\[[^\]]*\]/,
            `"median":[${median.join(',')}]`)});
      }
      case 'negative-track':
        return json(200,profilePayload({lesion_path:'nifti/lesion.nii.gz',
          lesion_sha256:'e'.repeat(64),
          lesion_distance:{track_mm:Array.from({length:N_POINTS},(_,i)=>i===12?-3:2),reason:null}}));
      case 'null-count':{
        const body=profilePayload();
        body.nodes.n_valid[3]=null;
        return json(200,body);
      }
      case 'claim-missing': return json(200,profilePayload({claim:'descriptive profile'}));
      case 'claim-forbidden':
        return json(200,profilePayload({claim:'validated abnormality score'}));
      case 'lesion':
        return json(200,profilePayload({lesion_path:'nifti/lesion.nii.gz',
          lesion_sha256:'e'.repeat(64),
          lesion_distance:{track_mm:Array.from({length:N_POINTS},
            (_,i)=>i===75?0.6:2+Math.abs(i-75)*0.3),reason:null}}));
      default: return json(200,profilePayload());
    }
  }

  if(path==='/api/bootstrap'||path==='/api/health') return json(200,BOOTSTRAP);
  if(path==='/api/derivation') return json(200,{active:'d1',kind:'rpe_pair',
    derivations:[{id:'d1',kind:'rpe_pair',qc_signed:{}}],
    floor_label:'reverse-PE corrected — median shift 0.6 mm (signed delta QC)',
    delta_qc_signed:true});
  if(path==='/api/preflight') return json(200,{stored:null,verified:{criteria:[]},drift:[]});
  if(path==='/api/presets') return json(200,{presets:[]});
  if(path==='/api/banks') return json(200,CATALOG);
  if(path==='/api/priors') return json(200,{priors:[]});
  if(path==='/api/volume/b0'){
    return route.fulfill({status:200,contentType:'application/octet-stream',
      headers:{'X-Shape':`${DIM},${DIM},${DIM}`,'X-Affine':AFFINE.join(';'),
        'X-Grid-Id':BOOTSTRAP.gridId},
      body:b0});
  }
  if(path==='/api/surface/brain'){
    return route.fulfill({status:200,contentType:'application/octet-stream',
      headers:{'X-vertexCount':String(HULL.vertexCount),'X-faceCount':String(HULL.faceCount),
        'X-center':`${DIM/2},${DIM/2},${DIM/2}`,'X-radius':String(DIM)},
      body:HULL.body});
  }
  if(path==='/api/bank/load'){
    return route.fulfill({status:200,contentType:'application/octet-stream',
      headers:bankLoadHeaders(),body:TUBES});
  }
  // Optional case data this fixture does not provide. The viewer treats each
  // of these as absent, which is the path a lesion-free case already takes.
  if(path.startsWith('/api/')) return route.fulfill({status:404,contentType:'text/plain',body:'absent'});

  const file=await staticBody(path);
  if(file) return route.fulfill({status:200,contentType:file.type,body:file.body});
  return route.fulfill({status:404,contentType:'text/plain',body:'not found'});
});

const state=()=>page.evaluate(()=>window.__tractlabTest.profile);
const settle=(predicate,arg)=>page.waitForFunction(predicate,arg,{timeout:30000});
const errorLine=()=>page.locator('#profileError').textContent();
let opened=false;
const reload=async()=>{
  // goto() to a URL differing only by #hash does not reload the document.
  if(opened) await page.reload({waitUntil:'load'});
  else{ await page.goto(`${ORIGIN}/#test`,{waitUntil:'load'}); opened=true; }
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
};
const refocus=()=>page.evaluate(key=>{
  document.querySelector(`[data-layer-key="${key}"] .map-focus`)?.click();
},LAYER);
const report={};

try{
  // The fixture itself must be a working viewer before any refusal means
  // anything: geometry loaded, a profile drawn, the served claim shown.
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:0};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  const ready=await state();
  assert.equal(ready.bankId,BANK);
  assert.equal(ready.nodes,N_POINTS);
  assert.deepEqual(await page.evaluate(()=>window.__tractlabTest.layerKeys),[LAYER]);
  assert.equal(await page.locator('#profileClaim').textContent(),SERVED_CLAIM);
  const points=await page.evaluate(()=>[...document.querySelectorAll('#profileChart .profile-median')]
    .map(node=>node.getAttribute('points').trim().split(/\s+/).length));
  assert.equal(points.reduce((sum,count)=>sum+count,0),N_POINTS);
  report.geometry=await page.evaluate(key=>window.__tractlabProfileProbe.info(key),LAYER);
  assert.equal(report.geometry.lineCount,LINES);
  assert.equal(report.geometry.k,K);

  // --- items 1 to 4: every refusal path, one honest line each ---
  const refusals=[
    ['wrong-bank','bank_mismatch','The server answered for a different tract'],
    ['wrong-scalar','scalar_mismatch','The server answered for a different scalar'],
    ['wrong-derivation','derivation_mismatch','This profile belongs to another derivation of the case'],
    ['orientation-family','orientation_mismatch',"The profile orientation does not match this tract's identity"],
    ['orientation-side','orientation_mismatch',"The profile orientation does not match this tract's identity"],
    ['orientation-rule','orientation_mismatch',"The profile orientation does not match this tract's identity"],
    ['nonfinite-median','measurements_invalid','This profile arrived with unusable measurements'],
    ['null-count','measurements_invalid','This profile arrived with unusable measurements'],
    ['negative-track','measurements_invalid','This profile arrived with unusable measurements'],
    ['claim-missing','claim_invalid','This profile did not carry the served claim'],
    ['claim-forbidden','claim_forbidden','This profile claims validation, which this instrument never asserts'],
  ];
  for(const [mode,code,line] of refusals){
    plan={profile:mode,bankSourceHash:SOURCE_A,delayMs:0};
    await reload();
    await settle(c=>window.__tractlabTest.profile?.status==='error'
      &&window.__tractlabTest.profile.code===c,code);
    assert.equal(await errorLine(),line,mode);
    assert.equal(await page.locator('#profileChart').count(),0,`${mode} draws no chart`);
    assert.equal(await page.locator('#profileProvCard').count(),0,`${mode} shows no provenance`);
  }
  report.refusals=refusals.map(([mode])=>mode);

  // --- item 2: no derivation authority, no profile ---
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:0,derivationDown:true};
  await page.route('**/api/derivation',route=>route.fulfill({status:503,
    contentType:'application/json',body:JSON.stringify({error:'unavailable'})}));
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='error'
    &&window.__tractlabTest.profile.code==='derivation_unknown');
  assert.equal(await errorLine(),'Derivation status unavailable');
  await page.unroute('**/api/derivation');
  report.derivationDown='derivation_unknown';

  // --- item 3: a layer with no source digest cannot be checked ---
  plan={profile:'ok',bankSourceHash:null,delayMs:0};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='error'
    &&window.__tractlabTest.profile.code==='bank_source_unknown');
  assert.equal(await errorLine(),'This tract carries no source digest to check');

  // --- item 3: a profile read on source A is never left on screen once the
  // layer has been reloaded from source B. (The in-flight case — an answer
  // for A arriving after the reload — is pinned by the unit test
  // profile_response_started_on_source_A_refuses_after_reload_B, because
  // removing the layer here aborts the request before it can land.)
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:0};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  await page.locator('#profileChart').scrollIntoViewIfNeeded();
  const chartA=await page.locator('#profileChart').boundingBox();
  await page.mouse.move(chartA.x+chartA.width*0.3,chartA.y+chartA.height/2);
  await page.mouse.down();
  await page.mouse.move(chartA.x+chartA.width*0.6,chartA.y+chartA.height/2,{steps:5});
  await page.mouse.up();
  await settle(key=>window.__tractlabProfileProbe.info(key)?.hasMask===true,LAYER);

  const beforeReload=seen.length;
  plan={profile:'ok',bankSourceHash:SOURCE_B,delayMs:0};
  await page.evaluate(()=>document.getElementById('secBanks')?.setAttribute('open',''));
  await page.evaluate(id=>document.getElementById(`bank_${id}`)?.click(),BANK);   // unload
  await settle(()=>window.__tractlabTest.layerKeys.length===0);
  await page.evaluate(id=>document.getElementById(`bank_${id}`)?.click(),BANK);   // reload as B
  await settle(key=>window.__tractlabTest.layerKeys.includes(key),LAYER);
  await settle(()=>window.__tractlabTest.profile?.status==='error'
    &&window.__tractlabTest.profile.code==='bank_source_changed');
  assert.equal(await errorLine(),'The tract file changed on disk, reload the case');
  assert.equal(seen.length,beforeReload,
    'the profile read on the old source is refused, not silently re-read');
  assert.equal((await page.evaluate(key=>window.__tractlabProfileProbe.info(key),LAYER)).hasMask,
    false,'the highlight from the old source does not survive the reload');
  report.sourceReload='bank_source_changed';

  // --- 409 is transient: not cached, refocus retries ---
  plan={profile:'busy',bankSourceHash:SOURCE_A,delayMs:0};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='error'
    &&window.__tractlabTest.profile.code==='http_409');
  const before=seen.length;
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:0};
  await refocus();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  assert.ok(seen.length>before,'refocus re-issued the request');
  report.retryRequests=seen.length-before;

  // --- abort race: FA, MD, FA back to back must not hang on a dead request ---
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:1200};
  await reload();
  await page.waitForTimeout(120);
  await page.locator('#profileScalar_md').click();
  await page.waitForTimeout(60);
  await page.locator('#profileScalar_fa').click();
  plan={profile:'ok',bankSourceHash:SOURCE_A,delayMs:0};
  await settle(()=>window.__tractlabTest.profile?.status==='ready'
    &&window.__tractlabTest.profile.scalar==='fa');
  assert.equal((await state()).nodes,N_POINTS,'the restarted key resolved');

  // --- a served lesion track still draws its closest approach ---
  plan={profile:'lesion',bankSourceHash:SOURCE_A,delayMs:0};
  await reload();
  await settle(()=>window.__tractlabTest.profile?.status==='ready');
  await page.locator('#profileChart').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#profileClosestLabel').textContent(),'closest 0.6 mm, node 75');
  assert.equal(await page.locator('#profileClosestMarker').count(),1);

  // --- and the interval still masks the tubes it should ---
  const box=await page.locator('#profileChart').boundingBox();
  await page.mouse.move(box.x+box.width*0.3,box.y+box.height/2);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width*0.6,box.y+box.height/2,{steps:6});
  await page.mouse.up();
  await settle(key=>window.__tractlabProfileProbe.info(key)?.hasMask===true,LAYER);
  const masked=await page.evaluate(key=>window.__tractlabProfileProbe.info(key),LAYER);
  assert.ok(masked.maskSum>0&&masked.maskSum<masked.lineCount*masked.k);
  report.interval=masked.interval;
  report.maskSum=masked.maskSum;

  assert.deepEqual(errors,[],'no page errors');
  console.log(JSON.stringify({stage:'profile-panel-synthetic',requests:seen.length,...report}));
}catch(error){
  console.error(JSON.stringify({stage:'profile-panel-synthetic-failure',message:error.message,
    plan,profile:await state().catch(()=>null),seen:seen.slice(-4),errors}));
  throw error;
}finally{
  await browser.close();
}
