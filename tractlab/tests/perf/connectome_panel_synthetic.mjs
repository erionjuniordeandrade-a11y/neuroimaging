#!/usr/bin/env node
// Connections panel with NO case server at all — modelled on
// tests/perf/profile_panel_synthetic.mjs.
//
// Reproduces the owner's 3T case exactly: a lesion mask with a real nonzero
// voxel, which puts the viewer in "case-map" (lesion-centred workstation)
// mode — body.case-map — same as it would on any case whose lesion volume
// actually has lesion voxels. The Tracts tab and its dock (intersections /
// profile — open by default, real chart rendered — / connections) work the
// same in that mode; ROOT CAUSE: an UNCONDITIONAL CSS rule,
// `body.case-map #side #connectomeHost { display:none; }`, hid the
// Connections host entirely (not gated behind the narrow-viewport media
// query the rest of case-map's layout changes live in). Every element
// inside it still had its OWN computed style exactly as authored (svg
// display:inline, panel display:block — what devtools' Computed panel
// reports, independent of ancestor visibility) and the full DOM (49
// rect[data-net-a] cells, provenance chip text) was genuinely present —
// but nothing inside a display:none ancestor is ever part of the render
// tree, so getBoundingClientRect measured 0x0 everywhere and the <details>
// itself measured as summary-only (32px: the host contributed zero height).
// Runs the whole page twice, at 1440x900 and at 1280x720, and asserts every
// one of the 49 heatmap cells has a non-zero bounding box in both.
//
//   node tests/perf/connectome_panel_synthetic.mjs
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
const DIM=10, N_VOX=DIM*DIM*DIM;
const LINES=6, K=32, N_POINTS=100;
const SERVED_CLAIM='descriptive profile; not a normative abnormality score';

// ---- synthetic case (shared geometry helpers, same as profile_panel_synthetic.mjs) ----

const b0=Buffer.alloc(N_VOX);
for(let i=0;i<N_VOX;i++) b0[i]=40+(i%160);
const AFFINE=[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

function tubeGeometry(lines,k,zBase=1,zSpan=7){
  const positions=new Float32Array(lines*k*3);
  let at=0;
  for(let line=0;line<lines;line++){
    const x=2+line*1.2, y=4+(line%2)*0.8;
    for(let point=0;point<k;point++){
      const t=point/(k-1);
      positions[at++]=x; positions[at++]=y; positions[at++]=zBase+t*zSpan;
    }
  }
  return Buffer.from(positions.buffer);
}
const TUBES=tubeGeometry(LINES,K);
const EDGE_TUBES=tubeGeometry(2,16); // the edge Show loads — 2 short streamlines

function hullSurface(scale=DIM,offset=0){
  const corners=[];
  for(const x of [0,scale]) for(const y of [0,scale]) for(const z of [0,scale]) corners.push([x+offset,y+offset,z+offset]);
  const vertices=new Float32Array(corners.flat());
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
const HULL=hullSurface(DIM);
const LESION_SURFACE=hullSurface(1.5,4); // a small lesion mesh inside the hull

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
  // hasLesion:true makes "Lesion intersections" visible; the ACTUAL
  // case-map trigger is a nonzero voxel in /api/volume/lesion (below) —
  // both are true here, matching the owner's case exactly.
  hasLesion:true, sift2BankCount:0,
  recovery:{perilesional:true,radiusMinMm:3,radiusMaxMm:20,radiusDefaultMm:8,
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

const CONNECTOTOMY_REPORT={
  banks:[{id:BANK,n_cut:2,n_bank:LINES,label:'True CST-R (synthetic)'}],
  note:'Lesion cavity. Geometry uncertain to ~3 mm. Research/preview only — not a resection plan.',
};

// ---- synthetic C1b connectome: 4 nodes, nodes 1,2 -> network 1 (Visual),
// nodes 3,4 -> network 2 (Somatomotor). Same shape as the pure-function unit
// tests in tests/js/connectome_panel.test.js. Edge 1-2 (count 5) is the
// pair Show is clicked on. ----
const MATRIX=[
  [0,5,2,0],
  [5,0,0,1],
  [2,0,0,3],
  [0,1,3,0],
];
const NODE_LABELS=['7Networks_LH_Vis_1','7Networks_LH_Vis_2','7Networks_LH_SomMot_3','7Networks_LH_SomMot_4'];
const GEN_ID='20260915T000000Z-synthetic1';
const CORPUS_SHA='c'.repeat(64), PARC_SHA='9'.repeat(64), LUT_SHA='7'.repeat(64);
const MATRIX_SHA='1'.repeat(64), ASSIGN_SHA='2'.repeat(64);
const EDGE_CONTENT_SHA='e'.repeat(64);

const CONNECTOME_PROVENANCE={
  case_id:'synthetic', active_derivation:'d1',
  corpus:{key:'WHOLEBRAIN_100K',path:'tracts/connectome/wholebrain.tck',sha256:CORPUS_SHA},
  parcellation:{key:'parc_schaefer200_yeo7',path:'normative/parc.nii.gz',sha256:PARC_SHA,
    lut_path:'normative/lut.txt',lut_sha256:LUT_SHA,label:'Synthetic Schaefer/Yeo-7',n_parcels:4,n_networks:2},
  matrix:{path:'matrix.csv',sha256:MATRIX_SHA},
  assignments:{path:'assignments.txt',sha256:ASSIGN_SHA},
  radius_mm:4.0, mrtrix_version:'synthetic', n_streamlines:11, n_assigned:11,
  unassigned_fraction:0.0, node_order:NODE_LABELS.map((name,i)=>({id:i+1,name})),
  weighting:'none (raw counts)', assignment_method:'radial_search',
  sift2:'none — synthetic fixture', gen_id:GEN_ID,
  label_note:'ASSIGNED — parcel tag on a patient streamline, not cortex identity (ADR-0003)',
};

const LUT_LABELS={
  1:{id:1,name:NODE_LABELS[0],networkId:1,networkName:'Visual',hemi:'L'},
  2:{id:2,name:NODE_LABELS[1],networkId:1,networkName:'Visual',hemi:'L'},
  3:{id:3,name:NODE_LABELS[2],networkId:2,networkName:'Somatomotor',hemi:'L'},
  4:{id:4,name:NODE_LABELS[3],networkId:2,networkName:'Somatomotor',hemi:'L'},
};

function edgeTubesHeaders(){
  return {
    'X-encoding':'float32le', 'X-lineCount':'2', 'X-pointsPerLine':'16',
    'X-layout':'line-major, then point, then xyz', 'X-spaceId':'synthetic',
    'X-sourcePopulation':'edge:1-2', 'X-outcome':'ok',
    'X-a':'1', 'X-b':'2',
    'X-matrixCount':'5','X-assignmentRowCount':'5','X-extractedCount':'5',
    'X-edgeSourceHash':EDGE_CONTENT_SHA, 'X-corpusSourceHash':CORPUS_SHA,
    'X-parcellationSourceHash':PARC_SHA, 'X-generationId':GEN_ID,
    'X-assignmentRadiusMm':'4',
    'X-matrixPath':'matrix.csv','X-matrixSha256':MATRIX_SHA,
    'X-assignmentsPath':'assignments.txt','X-assignmentsSha256':ASSIGN_SHA,
    'X-lesionHits':'1','X-lesionTotal':'2','X-lesionReason':'',
    'X-engine':'CONNECTOME | edge 1-2',
    'X-nReturned':'2','X-nDisplayed':'2',
    'X-clearanceRefusal':'synthetic fixture records no geometric floor',
  };
}

// ---- mutable plan the routes follow --------------------------------------

let plan={bankSourceHash:SOURCE_A};

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
    const meta=`<meta name="tractlab-build" content="${BUILD}">`
      +'<meta name="tractlab-protocol" content="1">';
    body=Buffer.from(body.toString('utf8').replace('</head>',`${meta}</head>`),'utf8');
  }
  return {body,type};
}

function bankLoadHeaders(){
  return {
    'X-encoding':'float32le', 'X-lineCount':String(LINES), 'X-pointsPerLine':String(K),
    'X-layout':'line-major, then point, then xyz', 'X-spaceId':'synthetic',
    'X-sourcePopulation':`bank:${BANK}`, 'X-outcome':'ok', 'X-bankId':BANK,
    'X-engine':'SYNTHETIC | prebuilt', 'X-nReturned':String(LINES), 'X-nDisplayed':String(LINES),
    'X-nAnalytic':String(LINES), 'X-nAnalyticFull':String(LINES),
    'X-clearanceRefusal':'synthetic fixture records no geometric floor',
    ...(plan.bankSourceHash?{'X-bankSourceHash':plan.bankSourceHash}:{}),
  };
}

/** Registers the full synthetic route table on one page. */
async function installRoutes(page){
  await page.route('**/*',async route=>{
    const target=new URL(route.request().url());
    const path=target.pathname;
    const json=(status,body)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});

    if(path.startsWith('/api/profile/')) return json(200,profilePayload());
    if(path==='/api/bootstrap'||path==='/api/health') return json(200,BOOTSTRAP);
    if(path==='/api/derivation') return json(200,{active:'d1',kind:'rpe_pair',
      derivations:[{id:'d1',kind:'rpe_pair',qc_signed:{}}],
      floor_label:'reverse-PE corrected — median shift 0.6 mm (signed delta QC)',
      delta_qc_signed:true});
    if(path==='/api/preflight') return json(200,{stored:null,verified:{criteria:[]},drift:[]});
    if(path==='/api/presets') return json(200,{presets:[]});
    if(path==='/api/banks') return json(200,CATALOG);
    if(path==='/api/priors') return json(200,{priors:[]});
    if(path==='/api/connectotomy') return json(200,CONNECTOTOMY_REPORT);
    if(path==='/api/connectome') return json(200,{matrix:MATRIX,nodeLabels:NODE_LABELS,provenance:CONNECTOME_PROVENANCE});
    if(path==='/api/parcellation/lut') return json(200,{provenance:'population-atlas',
      disclaimer:'POPULATION ATLAS — Schaefer labels, not patient cortex', nLabels:4,
      labels:LUT_LABELS, lutSha256:LUT_SHA});
    if(/^\/api\/connectome\/edge\/1\/2\/tubes$/.test(path)){
      return route.fulfill({status:200,contentType:'application/octet-stream',
        headers:edgeTubesHeaders(),body:EDGE_TUBES});
    }
    if(path==='/api/volume/b0'){
      return route.fulfill({status:200,contentType:'application/octet-stream',
        headers:{'X-Shape':`${DIM},${DIM},${DIM}`,'X-Affine':AFFINE.join(';'),
          'X-Grid-Id':BOOTSTRAP.gridId},body:b0});
    }
    // A real nonzero voxel — this is what actually flips the viewer into
    // case-map (lesion-centred workstation) mode via lesionOrigin(), the
    // exact condition that exposed the bug (see the header comment).
    if(path==='/api/volume/lesion'){
      const lesion=Buffer.alloc(N_VOX);
      lesion[5*DIM*DIM+5*DIM+5]=1; // one voxel near the volume centre
      return route.fulfill({status:200,contentType:'application/octet-stream',body:lesion});
    }
    if(path==='/api/surface/brain'){
      return route.fulfill({status:200,contentType:'application/octet-stream',
        headers:{'X-vertexCount':String(HULL.vertexCount),'X-faceCount':String(HULL.faceCount),
          'X-center':`${DIM/2},${DIM/2},${DIM/2}`,'X-radius':String(DIM)},body:HULL.body});
    }
    if(path==='/api/surface/lesion'){
      return route.fulfill({status:200,contentType:'application/octet-stream',
        headers:{'X-vertexCount':String(LESION_SURFACE.vertexCount),'X-faceCount':String(LESION_SURFACE.faceCount)},
        body:LESION_SURFACE.body});
    }
    if(path==='/api/bank/load'){
      return route.fulfill({status:200,contentType:'application/octet-stream',
        headers:bankLoadHeaders(),body:TUBES});
    }
    if(path.startsWith('/api/')) return route.fulfill({status:404,contentType:'text/plain',body:'absent'});

    const file=await staticBody(path);
    if(file) return route.fulfill({status:200,contentType:file.type,body:file.body});
    return route.fulfill({status:404,contentType:'text/plain',body:'not found'});
  });
}

/** Runs the whole scenario once, at the given viewport, returning the cell bounding boxes measured. */
async function runAtViewport(browser,width,height){
  const page=await browser.newPage({viewport:{width,height}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  if(process.env.TRACTLAB_SYNTHETIC_DEBUG){
    page.on('console',m=>console.error('[console]',m.type(),m.text()));
    page.on('requestfailed',r=>console.error('[failed]',r.url(),r.failure()?.errorText));
  }
  await installRoutes(page);

  await page.goto(`${ORIGIN}/#test`,{waitUntil:'load'});
  await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
  await page.waitForFunction(()=>window.__tractlabTest?.layerKeys?.length>0,undefined,{timeout:60000});
  const settled=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));

  // Confirms the fixture actually reproduces the trigger condition — a real
  // lesion mask puts the viewer in case-map mode, the same state the owner's
  // case was in (never assume; verify the body class landed).
  assert.equal(await page.evaluate(()=>document.body.classList.contains('case-map')),true,
    'the synthetic lesion voxel must put the viewer in case-map mode');

  await page.locator('#tab-tracts').click();await settled();

  // The exact dock stack the owner's case hit: intersections (collapsed,
  // present because hasLesion), profile (open by default, a real chart
  // rendered), THEN connections opened last, pushed furthest down.
  await page.locator('#profileDetails').waitFor({state:'attached',timeout:30000});
  assert.equal(await page.locator('#profileDetails').getAttribute('open')!==null,true,
    'the profile details is open by default, exactly like the owner\'s case');
  await page.waitForFunction(()=>document.getElementById('profileChart'),undefined,{timeout:30000});

  await page.locator('#connectomeDetails summary').click();
  await page.locator('#connectomePanel').waitFor({state:'attached',timeout:30000});
  const cellCount=await page.locator('#connectomePanel rect[data-net-a]').count();
  assert.equal(cellCount,49,`the heatmap has 49 cells at ${width}x${height} (got ${cellCount})`);

  // The actual regression: every cell's bounding box must be non-zero. The
  // bug measured svg 0x0 and every rect 0x0 despite all 49 existing in the
  // DOM with real provenance chip text.
  const boxes=await page.evaluate(()=>[...document.querySelectorAll('#connectomePanel rect[data-net-a]')]
    .map(el=>{const r=el.getBoundingClientRect();return {w:r.width,h:r.height};}));
  const zeroBoxes=boxes.filter(b=>b.w<=0||b.h<=0);
  assert.equal(zeroBoxes.length,0,
    `every cell must have a non-zero bounding box at ${width}x${height} (${zeroBoxes.length} of ${boxes.length} were 0x0)`);
  const svgBox=await page.locator('#connectomePanel svg').boundingBox();
  assert.ok(svgBox&&svgBox.width>0&&svgBox.height>0,
    `the heatmap svg itself must be non-zero at ${width}x${height} (got ${JSON.stringify(svgBox)})`);

  const chipsText=await page.locator('#connectomePanel .connectome-chips').first().textContent();
  assert.match(chipsText,/corpus/,`provenance chips are rendered (${chipsText})`);

  // --- click a cell, Show, assert the edge layer upserts -------------------
  await page.locator('#connectomePanel rect[data-net-a="1"][data-net-b="1"]').click();
  await settled();
  const pairRows=page.locator('#connectomeDrilldown .connectome-pair-row');
  assert.ok(await pairRows.count()>=1,'network1->network1 has at least the 1-2 pair');
  await pairRows.first().locator('.connectome-show-btn').click();
  await page.waitForFunction(key=>window.__tractlabTest?.sourcePopulation===key,'edge:1-2',{timeout:30000});
  const descriptor=await page.evaluate(()=>window.__tractlabTest.descriptor);
  assert.match(descriptor.short,/^ASSIGNED 1–2/,`the edge layer descriptor is rendered (${descriptor.short})`);
  const infoText=await page.locator('#connectomeDrilldown .connectome-edge-info').first().innerText();
  assert.match(infoText,/matrix 5.*assignment rows 5.*extracted 5/,`the counts row is populated (${infoText})`);

  assert.deepEqual(errors,[],`no page errors at ${width}x${height}`);
  await page.close();
  return {cellCount,zeroBoxCount:zeroBoxes.length,svgBox};
}

const browser=await chromium.launch({headless:true,channel:'chrome'});
try{
  const report={};
  report.wide=await runAtViewport(browser,1440,900);
  report.narrow=await runAtViewport(browser,1280,720);
  console.log(JSON.stringify({stage:'connectome-panel-synthetic',...report}));
}catch(error){
  console.error(JSON.stringify({stage:'connectome-panel-synthetic-failure',message:error.message}));
  throw error;
}finally{
  await browser.close();
}
