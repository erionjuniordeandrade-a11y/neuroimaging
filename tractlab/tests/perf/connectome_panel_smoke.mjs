/**
 * C1b Connections panel browser smoke: open the panel, check the 7x7
 * heatmap's rendered counts sum to the served matrix total, drill into one
 * network cell, load one edge's tubes as a generated layer, and confirm it
 * round-trips through save/reopen (review record v2 identity check).
 *
 * Usage: node tests/perf/connectome_panel_smoke.mjs [--url=http://127.0.0.1:18995/]
 * Needs a served case whose connectome has already been built
 * (scripts/build_connectome.py) — the demo case ships one under
 * cases/demo-leipzig-sub-010005/work/connectome/.
 */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {matrixTotal, aggregateNetworkMatrix, nodeNetworkIds} from '../../viewer/connectome_panel.js';

const rawUrl=process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18995/';
const base=rawUrl.replace(/\/$/,'');
const url=new URL(rawUrl); url.hash='test';

const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
const settled=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
const idle=()=>page.waitForFunction(()=>!document.getElementById('main').inert);
const ready=()=>page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:180000});
const dismiss=async()=>{if(await page.locator('#reviewStatus').isVisible())await page.locator('.review-dismiss').click();};

async function saveRecord(){
  const saving=page.waitForEvent('download',{timeout:30000});
  await page.locator('#btnSaveReview').click();
  const file=await saving;await idle();
  const record=JSON.parse(await readFile(await file.path(),'utf8'));
  await dismiss();
  return record;
}

try{
  // Independent ground truth: fetch the same JSON the page fetches, straight
  // from the server, and reduce it with the same pure helper the page uses —
  // but never trust what the page rendered without checking it here.
  const connectome=await (await fetch(base+'/api/connectome')).json();
  const lutResp=await (await fetch(base+'/api/parcellation/lut')).json();
  const netIds=nodeNetworkIds(connectome.matrix.length,lutResp.labels);
  const cells=aggregateNetworkMatrix(connectome.matrix,netIds);
  const servedTotal=matrixTotal(connectome.matrix);
  assert.equal(cells.flat().reduce((a,b)=>a+b,0),servedTotal,
    'every parcel has a network id 1-7, so the 7x7 aggregation accounts for the whole matrix');
  let bestP=0,bestQ=0,bestCount=-1;
  for(let p=0;p<7;p++)for(let q=0;q<7;q++)if(cells[p][q]>bestCount){bestCount=cells[p][q];bestP=p+1;bestQ=q+1;}
  assert.ok(bestCount>0,'the demo connectome has at least one non-empty network cell');

  await page.goto(url.href);
  await ready();
  await page.locator('#tab-tracts').click();await settled();

  await page.locator('#connectomeDetails summary').click();
  await page.locator('#connectomePanel').waitFor({state:'attached',timeout:60000});

  const cellHandles=page.locator('#connectomePanel rect[data-net-a]');
  assert.equal(await cellHandles.count(),49,'the heatmap is exactly 7x7');

  const renderedTitles=await cellHandles.evaluateAll(
    rects=>rects.map(r=>r.querySelector('title')?.textContent||''));
  const renderedTotal=renderedTitles.reduce((sum,text)=>{
    const m=text.match(/:\s*(\d+)\s*$/);
    return sum+(m?Number(m[1]):0);
  },0);
  assert.equal(renderedTotal,servedTotal,
    `sum of rendered cell counts (${renderedTotal}) must equal the served matrix sum (${servedTotal})`);

  await page.locator(`#connectomePanel rect[data-net-a="${bestP}"][data-net-b="${bestQ}"]`).click();
  await settled();
  const pairRows=page.locator('#connectomeDrilldown .connectome-pair-row');
  const pairCount=await pairRows.count();
  assert.ok(pairCount>=1 && pairCount<=12,`the drill-down list has between 1 and 12 rows (got ${pairCount})`);

  // Item 2 (fix round): Show must extract once — a single request to the
  // /tubes route, never a second request to the plain edge JSON route.
  const edgeJsonPath=/\/api\/connectome\/edge\/\d+\/\d+$/;
  const tubesPath=/\/api\/connectome\/edge\/\d+\/\d+\/tubes$/;
  const seenRequests=[];
  const onRequest=req=>{const p=new URL(req.url()).pathname; if(edgeJsonPath.test(p)||tubesPath.test(p))seenRequests.push(p);};
  page.on('request',onRequest);
  const tubesResponse=page.waitForResponse(r=>tubesPath.test(new URL(r.url()).pathname));
  await pairRows.first().locator('.connectome-show-btn').click();
  const tubesResp=await tubesResponse;
  await settled();
  page.off('request',onRequest);
  assert.equal(seenRequests.filter(p=>tubesPath.test(p)).length,1,`exactly one /tubes request (saw ${JSON.stringify(seenRequests)})`);
  assert.equal(seenRequests.filter(p=>edgeJsonPath.test(p)).length,0,
    `Show must never call the plain edge JSON route (saw ${JSON.stringify(seenRequests)})`);

  const tubesHeaders=tubesResp.headers();
  assert.match(tubesHeaders['x-engine']||'',/^CONNECTOME \| edge \d+-\d+$/,
    `the tube response carries the edge engine token (got ${tubesHeaders['x-engine']})`);
  const [, edgeA, edgeB]=new URL(tubesResp.url()).pathname.match(/edge\/(\d+)\/(\d+)\/tubes$/);
  const expectedLayerKey=`edge:${edgeA}-${edgeB}`;

  await page.waitForFunction(
    key=>window.__tractlabTest?.sourcePopulation===key, expectedLayerKey, {timeout:60000});
  await page.waitForFunction(()=>{
    const el=document.querySelector('#connectomeDrilldown .connectome-edge-info');
    return !!el && el.textContent.trim()!=='' && el.textContent!=='Loading…';
  },undefined,{timeout:30000});

  const infoText=await page.locator('#connectomeDrilldown .connectome-edge-info').first().innerText();
  const counts=[...infoText.matchAll(/(\d+)/g)].map(m=>Number(m[1]));
  assert.ok(counts.length>=3,`the counts row reports at least 3 numbers (got: ${infoText})`);
  const [matrixCount,assignmentRowCount,extractedCount]=counts;
  assert.equal(matrixCount,assignmentRowCount,`matrixCount must equal assignmentRowCount (${infoText})`);
  assert.equal(assignmentRowCount,extractedCount,`assignmentRowCount must equal extractedCount (${infoText})`);

  // --- save + reopen: the edge layer restores (identity check passes) -----
  const saved=await saveRecord();
  const edgeEntry=saved.generated.find(g=>g.layerKey===expectedLayerKey);
  assert.ok(edgeEntry,`the saved record contains the edge layer ${expectedLayerKey} (layers: ${saved.generated.map(g=>g.layerKey).join(',')})`);
  assert.equal(edgeEntry.request.route,'/api/connectome/edge/tubes');
  assert.equal(edgeEntry.request.params.a,Number(edgeA));
  assert.equal(edgeEntry.request.params.b,Number(edgeB));
  assert.ok(edgeEntry.identity.edgeSourceHash,'the edge layer carries its own source hash identity');
  assert.equal(edgeEntry.identity.bankSourceHash,null,'an edge layer is not bank-backed');

  await page.reload();
  await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved)),
  });
  await idle();
  const statusText=await page.locator('#reviewStatus').innerText();
  assert.match(statusText,/Review reopened/,`the restore reports through the existing status panel (${statusText})`);
  const rows=await page.evaluate(()=>[...document.querySelectorAll('#reviewStatus .review-restore-list li')]
    .map(li=>({state:li.dataset.state,text:li.textContent})));
  const edgeRow=rows.find(row=>row.text.startsWith(edgeEntry.tractLabel+':'));
  assert.ok(edgeRow,`the edge layer has its own restore status row (${JSON.stringify(rows)})`);
  assert.equal(edgeRow.state,'ok',`the edge layer's identity check passes on reopen (${edgeRow.text})`);
  await dismiss();

  // --- fix round 2, item 2: edge_show_uses_assignment_radius_from_same_tubes_generation
  // The ASSIGNED label's radius must come from THIS Show's own tubes response
  // (X-assignmentRadiusMm), never a stale earlier /api/connectome fetch —
  // proven by rewriting the header in flight to a distinguishable value and
  // checking the rendered descriptor reflects exactly that value.
  await page.reload();
  await ready();
  await page.locator('#tab-tracts').click();await settled();
  await page.locator('#connectomeDetails summary').click();
  await page.locator('#connectomePanel').waitFor({state:'attached',timeout:60000});
  await page.locator(`#connectomePanel rect[data-net-a="${bestP}"][data-net-b="${bestQ}"]`).click();
  await settled();
  const rewrittenRadius='99';
  await page.route(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`, async route=>{
    const response=await route.fetch();
    route.fulfill({response,headers:{...response.headers(),'x-assignmentradiusmm':rewrittenRadius}});
  });
  await page.locator('#connectomeDrilldown .connectome-pair-row').first().locator('.connectome-show-btn').click();
  await page.waitForFunction(
    val=>window.__tractlabTest?.descriptor?.short?.includes(val),
    `≤${rewrittenRadius} mm`, {timeout:60000});
  const radiusDescriptor=await page.evaluate(()=>window.__tractlabTest.descriptor.short);
  assert.match(radiusDescriptor,/^ASSIGNED\b/,`the descriptor still carries ASSIGNED (${radiusDescriptor})`);
  assert.match(radiusDescriptor,new RegExp(`\\u2264${rewrittenRadius} mm`),
    `the rendered radius must come from THIS response's X-assignmentRadiusMm, not the earlier /api/connectome fetch (${radiusDescriptor})`);
  await page.unroute(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`);

  // --- fix round 2, item 3: load_edge_tubes_refuses_mismatched_pair_source_or_generation_before_upsert
  // A response whose X-a disagrees with the requested pair must be refused
  // BEFORE the layer is upserted — the commit count must not advance and the
  // drill-down must report the failure.
  const commitCountBefore=await page.evaluate(()=>window.__tractlabTest.commitCount);
  await page.route(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`, async route=>{
    const response=await route.fetch();
    route.fulfill({response,headers:{...response.headers(),'x-a':'999999'}});
  });
  await page.locator('#connectomeDrilldown .connectome-pair-row').first().locator('.connectome-show-btn').click();
  await page.waitForFunction(()=>{
    const el=document.querySelector('#connectomeDrilldown .connectome-edge-info');
    return !!el && /^Failed:/.test(el.textContent);
  },undefined,{timeout:30000});
  const failedInfoText=await page.locator('#connectomeDrilldown .connectome-edge-info').first().innerText();
  assert.match(failedInfoText,/Failed:.*(does not match|identity)/i,
    `the guard's refusal reason is shown in the panel's own status line (${failedInfoText})`);
  const commitCountAfter=await page.evaluate(()=>window.__tractlabTest.commitCount);
  assert.equal(commitCountAfter,commitCountBefore,
    'a mismatched-identity response must never reach upsertTubeLayer (commit count must not advance)');
  await page.unroute(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`);

  // --- final round, item 1a: load_edge_tubes_refuses_missing_radius_or_count_provenance_before_upsert
  // Stripping X-matrixSha256 (one of the count-provenance fields) must
  // refuse before upsert, exactly like a corrupted pair/source identity.
  const commitCountBeforeProvenance=await page.evaluate(()=>window.__tractlabTest.commitCount);
  await page.route(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`, async route=>{
    const response=await route.fetch();
    route.fulfill({response,headers:{...response.headers(),'x-matrixsha256':''}});
  });
  await page.locator('#connectomeDrilldown .connectome-pair-row').first().locator('.connectome-show-btn').click();
  await page.waitForFunction(()=>{
    const el=document.querySelector('#connectomeDrilldown .connectome-edge-info');
    return !!el && /^Failed:/.test(el.textContent);
  },undefined,{timeout:30000});
  assert.equal(await page.evaluate(()=>window.__tractlabTest.commitCount),commitCountBeforeProvenance,
    'missing count-source provenance (X-matrixSha256) must refuse before upsert');
  await page.unroute(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`);

  // --- final round, item 1b: edge_zero_line_response_refuses_incomplete_identity
  // A zero-streamline response with a corrupted identity header must still
  // be refused by the guard — never slip past to the empty-tract branch.
  const commitCountBeforeZero=await page.evaluate(()=>window.__tractlabTest.commitCount);
  await page.route(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`, async route=>{
    const response=await route.fetch();
    route.fulfill({response,headers:{...response.headers(),'x-linecount':'0','x-edgesourcehash':''}});
  });
  await page.locator('#connectomeDrilldown .connectome-pair-row').first().locator('.connectome-show-btn').click();
  await page.waitForFunction(()=>{
    const el=document.querySelector('#connectomeDrilldown .connectome-edge-info');
    return !!el && /^Failed:/.test(el.textContent);
  },undefined,{timeout:30000});
  const zeroLineFailText=await page.locator('#connectomeDrilldown .connectome-edge-info').first().innerText();
  assert.match(zeroLineFailText,/Failed:.*(does not match|identity)/i,
    `a zero-line response with broken identity is refused by the guard, not shown as "no streamlines" (${zeroLineFailText})`);
  assert.equal(await page.evaluate(()=>window.__tractlabTest.commitCount),commitCountBeforeZero);
  await page.unroute(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`);

  // --- final round, item 1c: load_edge_tubes_refuses_panel_generation_mismatch
  // A response whose X-generationId disagrees with the panel's own last-
  // fetched generation must refuse with the specific "reopen the panel"
  // reason and trigger a fresh /api/connectome fetch.
  const commitCountBeforeGen=await page.evaluate(()=>window.__tractlabTest.commitCount);
  const connectomeRefetch=page.waitForRequest(req=>new URL(req.url()).pathname==='/api/connectome',{timeout:15000});
  await page.route(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`, async route=>{
    const response=await route.fetch();
    route.fulfill({response,headers:{...response.headers(),'x-generationid':'a-different-generation-id'}});
  });
  await page.locator('#connectomeDrilldown .connectome-pair-row').first().locator('.connectome-show-btn').click();
  await page.waitForFunction(()=>{
    const el=document.querySelector('#connectomeDrilldown .connectome-edge-info');
    return !!el && /^Failed:/.test(el.textContent);
  },undefined,{timeout:30000});
  const genMismatchText=await page.locator('#connectomeDrilldown .connectome-edge-info').first().innerText();
  assert.match(genMismatchText,/reopen the panel/i,`the refusal names the reason (${genMismatchText})`);
  assert.equal(await page.evaluate(()=>window.__tractlabTest.commitCount),commitCountBeforeGen);
  await connectomeRefetch; // the fire-and-forget panel refresh must issue a fresh /api/connectome fetch
  await page.unroute(`**/api/connectome/edge/${edgeA}/${edgeB}/tubes`);

  assert.deepEqual(errors,[],'no page errors');
  console.log(`connectome panel smoke OK: 49 cells, rendered total ${renderedTotal} == served total ${servedTotal}, `
    + `cell (${bestP},${bestQ}) drill-down ${pairCount} row(s), edge ${edgeA}-${edgeB} loaded and restored, `
    + `radius-from-response and pre-render identity guard both proven`);
}finally{
  await browser.close();
}
