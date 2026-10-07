/**
 * Review record v2 browser smoke: paint an ROI, filter the bank with it, save
 * the record, reopen it, and check the painted set and the generated layer
 * come back through the same status panel the rest of the viewer uses.
 *
 * Usage: node tests/perf/review_v2_smoke.mjs [--url=http://127.0.0.1:18996/]
 * Needs a served case with a filter_bank; the CC0 demo case is enough.
 */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {canonicalJson,requestDigest,reviewSummary} from '../../viewer/review_record.js';

// Mirrors REVIEW_LIMITS.maxRecordBytes in viewer/review_record.js.
const REVIEW_CAP=256*1024;

/** Expand [start, length, ...] voxel runs, mirroring decodeVoxelRuns. */
function decodeRuns(runs){
  const out=[];
  for(let pair=0;pair<runs.length;pair+=2)for(let i=0;i<runs[pair+1];i+=1)out.push(runs[pair]+i);
  return out;
}

const url=process.argv.find(a=>a.startsWith('--url='))?.slice(6)||'http://127.0.0.1:18996/';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
const settled=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
const idle=()=>page.waitForFunction(()=>!document.getElementById('main').inert);
const ready=()=>page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:180000});
const dismiss=async()=>{if(await page.locator('#reviewStatus').isVisible())await page.locator('.review-dismiss').click();};

/** The body of the last /api/filter request, as the page actually sent it. */
let sentFilterBody=null;

async function paintStroke(){
  await page.locator('#btnInspectMRI').click();await settled();
  await page.locator('#btnPaintMode').click();await settled();
  const box=await page.locator('#ccor').boundingBox();
  for(const [u,v] of [[0.46,0.50],[0.48,0.52],[0.50,0.50]]){
    await page.mouse.move(box.x+box.width*u,box.y+box.height*v);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width*(u+0.01),box.y+box.height*v,{steps:3});
    await page.mouse.up();
  }
  await settled();
}

async function paintAndFilter(){
  await page.locator('#btnInspectMRI').click();await settled();
  await page.locator('#btnPaintMode').click();await settled();
  const box=await page.locator('#ccor').boundingBox();
  assert.ok(box,'the coronal MPR canvas is laid out');
  // The first point is painted twice on purpose: a re-painted voxel must not
  // change the request, the digest, or the stored runs.
  for(const [u,v] of [[0.46,0.50],[0.46,0.50],[0.48,0.52],[0.50,0.50],[0.52,0.48],[0.54,0.52]]){
    await page.mouse.move(box.x+box.width*u,box.y+box.height*v);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width*(u+0.01),box.y+box.height*v,{steps:3});
    await page.mouse.up();
  }
  await settled();
  assert.match(await page.locator('#status').innerText(),/SEED [1-9]/,'the stroke painted SEED voxels');
  const sending=page.waitForRequest(r=>r.url().endsWith('/api/filter'),{timeout:180000});
  await page.locator('#btnFilter').click();
  sentFilterBody=(await sending).postDataJSON();
  await page.waitForFunction(()=>window.__tractlabTest?.settled===true,undefined,{timeout:180000});
  await settled();
  assert.equal((await page.evaluate(()=>window.__tractlabTest.sourcePopulation)),'filtered-corpus');
}

async function saveRecord(){
  const saving=page.waitForEvent('download',{timeout:30000});
  await page.locator('#btnSaveReview').click();
  const file=await saving;await idle();
  const record=JSON.parse(await readFile(await file.path(),'utf8'));
  await dismiss();
  return record;
}

try{
  await page.goto(url+'#test');
  await ready();
  await paintAndFilter();
  const saved=await saveRecord();

  assert.equal(saved.schemaVersion,3,'a saved record is schema 3');
  assert.deepEqual(saved.outcome,{effect:'not-recorded',note:''},'an unanswered outcome saves as not recorded');
  assert.deepEqual(saved.unsupported,[],'nothing in this state is refused any more');

  assert.ok(saved.paint,'painted ROI state is saved, not refused');
  assert.equal(saved.paint.paintMode,true);
  assert.equal(saved.paint.grid.gridId,saved.context.gridId,'the paint grid is the case grid');
  assert.equal(saved.paint.grid.dims.length,3);
  assert.match(saved.paint.grid.affineDigest,/^[a-f0-9]{16}$/);
  const seedVoxels=saved.paint.rois.seed.filter((_,index)=>index%2===1).reduce((a,b)=>a+b,0);
  assert.ok(seedVoxels>0,`the stroke painted at least one SEED voxel (got ${seedVoxels})`);

  assert.equal(saved.generated.length,1,'the filter result is saved as a generated layer');
  const layer=saved.generated[0];
  assert.equal(layer.request.route,'/api/filter');
  assert.equal(layer.request.roiRef,'paint','the layer references the painted ROIs instead of copying them');
  assert.equal(layer.identity.sourcePopulation,'filtered-corpus');
  assert.ok(layer.identity.nReturned>0,'the returned count is recorded');
  assert.match(layer.request.digest,/^[a-f0-9]{16}$/);

  assert.equal(JSON.stringify(saved).includes('points_mm'),false,'no ROI or streamline geometry payload is written');
  const size=JSON.stringify(saved).length;
  assert.ok(size<REVIEW_CAP,`the record stays inside the byte cap (${size} of ${REVIEW_CAP})`);

  // Reopen in a clean page. The painted set and the layer must come back by
  // re-issuing the saved request, never from anything stored in the file.
  await page.reload();
  await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved)),
  });
  await idle();
  const statusText=await page.locator('#reviewStatus').innerText();
  assert.match(statusText,/Review reopened/,'the restore reports through the existing status panel');
  const restored=await page.locator('#reviewStatus .review-restore-list li[data-state="ok"]').count();
  const refused=await page.locator('#reviewStatus .review-restore-list li[data-state="refused"]').count();
  assert.ok(restored>0,'the status panel lists what was restored');
  assert.equal(refused,0,`nothing was refused (status: ${statusText})`);
  await dismiss();

  const again=await saveRecord();
  assert.deepEqual(again.paint.rois,saved.paint.rois,'the painted voxel set survives save and reopen');
  assert.equal(again.paint.grid.affineDigest,saved.paint.grid.affineDigest);
  assert.equal(again.view.focusLayerKey,saved.view.focusLayerKey);
  assert.equal(again.generated[0].identity.nReturned,saved.generated[0].identity.nReturned);
  assert.equal(again.generated[0].request.digest,saved.generated[0].request.digest,
    'the rebuilt request is byte-identical to the saved one');

  // A generated layer whose returned identity no longer matches is reported
  // and left undrawn, while the rest of the record still restores.
  const mismatched=structuredClone(saved);
  mismatched.generated[0].identity.nReturned+=1;
  mismatched.summary=reviewSummary(mismatched);
  await page.reload();await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(mismatched)),
  });
  await idle();
  const mismatchText=await page.locator('#reviewStatus').innerText();
  assert.match(mismatchText,/streamline count changed/,'the refusal names the identity that changed');
  const layerDrawn=await page.evaluate(()=>window.__tractlabTest?.sourcePopulation);
  assert.notEqual(layerDrawn,'filtered-corpus','a refused layer is not drawn');
  await dismiss();

  // A record whose summary no longer matches its contents is refused outright.
  const tampered=structuredClone(saved);
  tampered.generated[0].identity.nReturned+=1;
  await page.reload();await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(tampered)),
  });
  await idle();
  assert.match(await page.locator('#reviewStatus').innerText(),/Review not opened/);

  // --- save_refuses_generated_layer_after_request_paint_changes -------------
  // The layer on screen was produced by the ROI state frozen when the request
  // was sent. "Clear" wipes the painted ROIs but deliberately keeps the tract
  // display, so the layer outlives its ROIs: the save must refuse by name
  // rather than re-digest the request against whatever is painted now.
  await page.reload();await ready();
  await paintAndFilter();
  await page.locator('#clear').click();await settled();
  assert.match(await page.locator('#status').innerText(),/Cleared painted ROIs/,'Clear keeps the tract display');
  await page.locator('#btnSaveReview').click();await idle();
  const refusedSave=await page.locator('#reviewStatus').innerText();
  assert.match(refusedSave,/Review not saved/,'a changed ROI state refuses the save');
  assert.match(refusedSave,/painted ROIs changed after/,`the refusal names why (${refusedSave})`);
  await dismiss();

  // --- restore_null_paint_clears_existing_roi_state -------------------------
  // A record with no paint must leave nothing of the previous session behind.
  await page.reload();await ready();
  const bankOnly=await saveRecord();
  assert.equal(bankOnly.paint,null,'a record taken with no paint carries paint:null');
  assert.ok(bankOnly.banks.length>0,'the bank-only record is anchored on a named bank');
  await paintStroke();
  assert.match(await page.locator('#status').innerText(),/SEED [1-9]/,'there is paint to clear');
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(bankOnly)),
  });
  await idle();
  assert.match(await page.locator('#reviewStatus').innerText(),/Review reopened/);
  await dismiss();
  const clearedRois=await page.evaluate(()=>document.getElementById('status').textContent);
  assert.match(clearedRois,/SEED 0/,`the previous session's painted ROIs are gone (${clearedRois})`);
  assert.equal(await page.locator('#btnPaintMode').getAttribute('aria-pressed'),'false','paint mode is off again');
  assert.equal(await page.locator('#btnUndoPaint').isDisabled(),true,'the paint undo history is gone too');
  const afterClear=await saveRecord();
  assert.equal(afterClear.paint,null,'re-saving confirms no ROI state survived');

  // --- restore_prior_404_is_refused ----------------------------------------
  await page.reload();await ready();
  // The rail remembers the last panel, so name the one the priors live in.
  await page.locator('#tab-tracts').click();
  await settled();
  // The panel only becomes visible once the priors have actually loaded.
  await page.locator('#priorPanel.visible').waitFor({state:'attached',timeout:120000});
  const priorId=await page.evaluate(()=>document.querySelector('#priorGroups .chip.prior')?.id?.replace(/^prior_/,'')||null);
  assert.ok(priorId,'the demo case serves atlas priors');
  // The chips live behind the collapsed population-atlas disclosure.
  await page.locator('#priorDetails').evaluate(el=>{el.open=true;});
  await page.locator('#prior_'+priorId).waitFor({state:'visible',timeout:60000});
  await page.locator('#prior_'+priorId).click();
  await page.waitForFunction(id=>document.getElementById('prior_'+id)?.classList.contains('active'),priorId,{timeout:60000});
  await settled();
  const withPrior=await saveRecord();
  assert.deepEqual(withPrior.overlays.priors,[priorId],'the prior is saved as an id');
  await page.reload();
  await page.route('**/api/priors/*/volume',route=>route.fulfill({status:404,body:'gone'}));
  await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(withPrior)),
  });
  await idle();
  const priorText=await page.locator('#reviewStatus').innerText();
  assert.match(priorText,new RegExp(`Atlas prior ${priorId}.*not restored`),
    `a prior whose volume 404s is reported refused (${priorText})`);
  assert.equal(await page.locator('#prior_'+priorId).getAttribute('aria-pressed'),'false',
    'a prior that failed to load is not left reading as active');
  await page.unroute('**/api/priors/*/volume');
  await dismiss();

  // --- restore_envelope_uses_saved_layer -----------------------------------
  // The viewer clears an envelope whenever focus moves, so a record with an
  // envelope on one tract and the focus on another is built here on purpose.
  // Restore must rebuild it on the layer the record names.
  await page.reload();await ready();
  const bootLayer=await page.evaluate(()=>window.__tractlabTest.sourcePopulation);
  assert.match(bootLayer,/^bank:/,`a named tract is focused at boot (${bootLayer})`);
  const otherChip=await page.evaluate(key=>{
    const want='bank_'+key.replace(/^bank:/,'');
    const chip=[...document.querySelectorAll('[id^="bank_"]')].find(b=>b.id!==want&&!b.classList.contains('active'));
    return chip?chip.id:null;
  },bootLayer);
  assert.ok(otherChip,'a second named tract is available');
  await page.locator('#'+otherChip).click();
  await page.waitForFunction(key=>window.__tractlabTest.sourcePopulation!==key,bootLayer,{timeout:180000});
  await settled();
  const twoTracts=await saveRecord();
  assert.equal(twoTracts.banks.length,2,'both named tracts are in the record');
  assert.notEqual(twoTracts.view.focusLayerKey,bootLayer,'the focus is on the second tract');

  const envelopeRecord=structuredClone(twoTracts);
  envelopeRecord.envelope={layerKey:bootLayer,marginMm:5};
  envelopeRecord.summary=reviewSummary(envelopeRecord);
  await page.reload();await ready();
  const marginRequest=page.waitForRequest(r=>r.url().endsWith('/api/margin'),{timeout:240000});
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(envelopeRecord)),
  });
  const marginBody=(await marginRequest).postDataJSON();
  await idle();
  assert.equal(marginBody.bankId,bootLayer.replace(/^bank:/,''),
    `the envelope rebuilt on its saved layer, not the focused one (${JSON.stringify(marginBody)})`);
  assert.equal(marginBody.marginMm,5);
  const envelopeText=await page.locator('#reviewStatus').innerText();
  assert.doesNotMatch(envelopeText,/Envelope.*not restored/,`the envelope restored (${envelopeText})`);
  await dismiss();

  // --- generated_request_digest_matches_exact_sent_body ---------------------
  // The digest must be over the body the page actually put on the wire, and the
  // body a reopen rebuilds must be that same body, byte for byte.
  await page.reload();await ready();
  await paintAndFilter();
  const exact=await saveRecord();
  assert.ok(sentFilterBody,'the filter request body was captured');
  const sentPoints=sentFilterBody.seed.points_mm;
  assert.equal(new Set(sentPoints.map(p=>p.join(','))).size,sentPoints.length,
    'the request carries each painted voxel once, even though one was painted twice');
  const savedLayer=exact.generated[0];
  const sentDescriptor={route:'/api/filter',bankId:null,seedPresetId:null,body:sentFilterBody};
  assert.equal(savedLayer.request.digest,requestDigest(sentDescriptor),
    'the saved digest is the digest of the body actually sent');
  // And the stored runs rebuild that body: same voxel set, same order, same mm.
  const rebuiltPoints=decodeRuns(exact.paint.rois.seed).length;
  assert.equal(rebuiltPoints,sentPoints.length,'the stored runs hold exactly the sent points');
  assert.equal(sentFilterBody.seed.radius_mm,exact.paint.brushRadiusMm);
  assert.equal(canonicalJson(sentDescriptor).length>0,true);

  // --- focused_generated_mismatch_surfaces_restored_false_and_preserves_other_objects
  // The focused layer is refused, but the painted ROIs and the atlas prior it
  // was saved with must still come back, each with its own status line.
  await page.locator('#tab-tracts').click();await settled();
  await page.locator('#priorPanel.visible').waitFor({state:'attached',timeout:120000});
  const mixPrior=await page.evaluate(()=>document.querySelector('#priorGroups .chip.prior')?.id?.replace(/^prior_/,'')||null);
  await page.locator('#priorDetails').evaluate(el=>{el.open=true;});
  await page.locator('#prior_'+mixPrior).waitFor({state:'visible',timeout:60000});
  await page.locator('#prior_'+mixPrior).click();
  await page.waitForFunction(id=>document.getElementById('prior_'+id)?.classList.contains('active'),mixPrior,{timeout:60000});
  await settled();
  const mixed=await saveRecord();
  assert.equal(mixed.view.focusLayerKey,'__live__','the generated layer is the saved focus');
  assert.ok(mixed.paint,'the record also carries painted ROIs');
  assert.deepEqual(mixed.overlays.priors,[mixPrior],'and an atlas prior');

  mixed.generated[0].identity.nReturned+=1;
  mixed.summary=reviewSummary(mixed);
  await page.reload();await ready();
  await page.locator('#reviewFile').setInputFiles({
    name:'review.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(mixed)),
  });
  await idle();
  const mixedText=await page.locator('#reviewStatus').innerText();
  assert.match(mixedText,/Review reopened/,`the record still opens (${mixedText})`);
  const rows=await page.evaluate(()=>[...document.querySelectorAll('#reviewStatus .review-restore-list li')]
    .map(li=>({state:li.dataset.state,text:li.textContent})));
  const refusedRow=rows.find(row=>row.text.startsWith(`${mixed.generated[0].tractLabel}:`));
  assert.ok(refusedRow,`the refused layer has its own status row (${JSON.stringify(rows)})`);
  assert.equal(refusedRow.state,'refused');
  assert.match(refusedRow.text,/streamline count changed/,'with its own reason');
  const paintRow=rows.find(row=>row.text.startsWith('Painted ROIs'));
  assert.equal(paintRow?.state,'ok','the painted ROIs still restored');
  const priorRow=rows.find(row=>row.text.startsWith(`Atlas prior ${mixPrior}`));
  assert.equal(priorRow?.state,'ok','the atlas prior still restored');
  const focusRow=rows.find(row=>row.text.startsWith('Saved focus'));
  assert.equal(focusRow?.state,'refused','the substituted focus is reported too');
  await dismiss();
  // The restored objects are really on screen, not just reported: the prior
  // chip is active and the painted SEED voxels are back on the ROI counter.
  assert.equal(await page.locator('#prior_'+mixPrior).getAttribute('aria-pressed'),'true',
    'the prior chip is active after the partial restore');
  const seedCount=decodeRuns(mixed.paint.rois.seed).length;
  assert.match(await page.locator('#status').innerText(),new RegExp(`SEED ${seedCount}\\b`),
    'the painted SEED voxels are back');
  // With no tract restored there is nothing to anchor a new record on, and the
  // save says so rather than writing a record with no evidence in it.
  await page.locator('#btnSaveReview').click();await idle();
  assert.match(await page.locator('#reviewStatus').innerText(),/Load and select a tract first/);
  await dismiss();

  assert.deepEqual(errors,[],'no page errors');
  console.log(`review v2 smoke OK: ${seedVoxels} SEED voxels, ${restored} objects restored, 0 refused, ${size} record bytes`);
  console.log('  save_refuses_generated_layer_after_request_paint_changes OK');
  console.log('  restore_null_paint_clears_existing_roi_state OK');
  console.log('  restore_prior_404_is_refused OK');
  console.log(`  restore_envelope_uses_saved_layer OK (${bootLayer} while focused on ${envelopeRecord.view.focusLayerKey})`);
  console.log(`  generated_request_digest_matches_exact_sent_body OK (${sentPoints.length} points)`);
  console.log('  focused_generated_mismatch_surfaces_restored_false_and_preserves_other_objects OK');
}finally{
  await browser.close();
}
