import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.argv.find(v=>v.startsWith('--url='))?.slice(6);
if(!base)throw new Error('Pass the isolated preview --url explicitly');
const url=new URL('atlas.html?test=1&profile=teaching',base).href;
const browser=await chromium.launch({headless:true});
const evidence={};await mkdir('output/playwright',{recursive:true});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],apiRequests=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/api/'))apiRequests.push(r.url());});
  await page.goto(url);await page.waitForFunction(()=>window.__atlasTest?.ready);
  const state=()=>page.evaluate(()=>window.__atlasTest);
  const picked=()=>page.locator('#pickedName').textContent();
  const snap=n=>page.screenshot({path:`output/playwright/journey-${String(n).padStart(2,'0')}.png`});

  // 1. Land on atlas.html?test=1&profile=teaching. Motor lesson 'motor-cst' preselected, status paused, step 0.
  let s=await state();
  assert.equal(s.lesson.lessonId,'motor-cst',"step 1: 'motor-cst' preselected on landing");
  assert.equal(s.lesson.status,'paused','step 1: lesson status paused on landing');
  assert.equal(s.lesson.step,0,'step 1: lesson step 0 on landing');
  await snap(1);evidence.step1=s;

  // 2. Click #lessonPlay (tour playing). Step 0: hemisphere control value unchanged ('both' or 'L'),
  //    picked name matches /Left · area 4/, bundles == [] (step 1 shows cortex only).
  await page.locator('#lessonPlay').click();
  s=await state();
  assert.equal(s.lesson.status,'playing','step 2: lesson playing after #lessonPlay click');
  assert.equal(s.lesson.step,0,'step 2: still on step 0 right after play');
  assert(['both','L'].includes(s.hemisphere),`step 2: hemisphere control unchanged ('both' or 'L'), observed ${JSON.stringify(s.hemisphere)}`);
  const picked2=await picked();
  assert.match(picked2,/Left · area 4/,`step 2: picked name matches /Left · area 4/, observed ${JSON.stringify(picked2)}`);
  assert.deepEqual(s.bundles,[],`step 2: bundles == [] at step 0 (cortex only), observed ${JSON.stringify(s.bundles)}`);
  // 2b. Auto-advance: a playing tour moves to step 1 on its own within STEP_MS (20 s) and keeps playing.
  assert.equal(await page.locator('.lesson-pace').count(),1,'step 2b: pace bar visible while playing');
  const stepMs=await page.evaluate(async()=>{const {LESSONS}=await import('./lesson_content.js');const l=LESSONS.find(x=>x.id==='motor-cst');return (l.steps[0].scene.durationSec||20)*1000;});
  await page.waitForFunction(()=>window.__atlasTest.lesson.step===1,null,{timeout:stepMs+8000});
  s=await state();
  assert.equal(s.lesson.status,'playing','step 2b: still playing after auto-advance');
  assert.deepEqual(s.bundles,[],'step 2b: step 1 (second step) is still cortex only');
  await page.locator('#lessonPlay').click(); // pause, then continue the manual journey from step 1
  await page.locator('#lessonStep-0').click();
  await snap(2);evidence.step2={...s,picked:picked2};

  // 3. Click #lessonNext twice -> step 2 (third step): bundles == ['CST_L'].
  await page.locator('#lessonNext').click();
  await page.locator('#lessonNext').click();
  s=await state();
  assert.equal(s.lesson.step,2,`step 3: lesson step == 2 after two #lessonNext clicks, observed ${s.lesson.step}`);
  assert.deepEqual(s.bundles,['CST_L'],`step 3: bundles == ['CST_L'], observed ${JSON.stringify(s.bundles)}`);
  await snap(3);evidence.step3=s;

  // 4. Select #atlasHemisphere = 'R' during the motor lesson -> lesson status 'paused', bundles == ['CST_R'],
  //    picked matches /Right · area 4/, view == 'right'.
  await page.locator('#atlasHemisphere').selectOption('R');
  s=await state();
  assert.equal(s.lesson.status,'paused',`step 4: lesson status 'paused' after hemisphere -> R, observed ${JSON.stringify(s.lesson.status)}`);
  assert.deepEqual(s.bundles,['CST_R'],`step 4: bundles == ['CST_R'], observed ${JSON.stringify(s.bundles)}`);
  const picked4=await picked();
  assert.match(picked4,/Right · area 4/,`step 4: picked name matches /Right · area 4/, observed ${JSON.stringify(picked4)}`);
  assert.equal(s.view,'right',`step 4: view == 'right', observed ${JSON.stringify(s.view)}`);
  await snap(4);evidence.step4={...s,picked:picked4};

  // 5. Select lesson 'interoception' via #lessonSelect, click #lessonPlay: deepVisible true, bundles [],
  //    picked matches /Right · area (AVI|PoI2)/ because hemisphere is still R.
  //    Click #lessonNext: picked still starts with 'Right'.
  await page.locator('#lessonSelect').selectOption('interoception');
  await page.locator('#lessonPlay').click();
  s=await state();
  assert.equal(s.deepVisible,true,`step 5: deepVisible true for interoception, observed ${JSON.stringify(s.deepVisible)}`);
  assert.deepEqual(s.bundles,[],`step 5: bundles == [], observed ${JSON.stringify(s.bundles)}`);
  const picked5a=await picked();
  assert.match(picked5a,/Right · area (AVI|PoI2)/,`step 5: picked matches /Right · area (AVI|PoI2)/ (hemisphere still R), observed ${JSON.stringify(picked5a)}`);
  await page.locator('#lessonNext').click();
  const picked5b=await picked();
  assert(picked5b.startsWith('Right'),`step 5: picked still starts with 'Right' after #lessonNext, observed ${JSON.stringify(picked5b)}`);
  await snap(5);evidence.step5={...s,pickedBeforeNext:picked5a,pickedAfterNext:picked5b};

  // 6. Select lesson 'motor-cst' again, click #lessonPlay: deepVisible == false (no leak from interoception),
  //    bundles == [] at step 0.
  await page.locator('#lessonSelect').selectOption('motor-cst');
  await page.locator('#lessonPlay').click();
  s=await state();
  assert.equal(s.deepVisible,false,`step 6: deepVisible == false, no leak from interoception, observed ${JSON.stringify(s.deepVisible)}`);
  assert.deepEqual(s.bundles,[],`step 6: bundles == [] at step 0, observed ${JSON.stringify(s.bundles)}`);
  await snap(6);evidence.step6=s;

  // 7. Click #lessonNext twice (step 2), click the AF_R chip (exploration): lesson paused, AF_R ADDED to
  //    whatever the step showed (multi-bundle picker, 2026-09-07: a manual pick adds, it does not replace).
  await page.locator('#lessonNext').click();
  await page.locator('#lessonNext').click();
  const shownBeforePick=(await state()).bundles;
  await page.locator('.bundle-chip[data-bundle="AF_R"]').click();
  s=await state();
  assert.equal(s.lesson.status,'paused',`step 7: lesson paused after exploration pathway pick, observed ${JSON.stringify(s.lesson.status)}`);
  assert.deepEqual(s.bundles,[...shownBeforePick,'AF_R'],`step 7: bundles == step bundles + AF_R, observed ${JSON.stringify(s.bundles)}`);
  await snap(7);evidence.step7=s;

  // 8. page.reload(); wait ready: lesson step == 2, status paused, hemisphere == 'R', bundles == ['CST_R'],
  //    view == 'right' (reload restores the AUTHORED view of the step on the persisted side; the exploration
  //    deviation AF_R is NOT persisted anywhere in the URL/state -- it must NOT survive the reload).
  await page.reload();await page.waitForFunction(()=>window.__atlasTest?.ready);
  s=await state();
  assert.equal(s.lesson.step,2,`step 8: lesson step == 2 restored after reload, observed ${s.lesson.step}`);
  assert.equal(s.lesson.status,'paused',`step 8: lesson status 'paused' after reload, observed ${JSON.stringify(s.lesson.status)}`);
  assert.equal(s.hemisphere,'R',`step 8: hemisphere == 'R' restored after reload, observed ${JSON.stringify(s.hemisphere)}`);
  // Exploration deviation (AF_R) is not part of the persisted lesson/URL state; reload must show the
  // step's authored bundle (CST_R), not the AF_R pathway that was picked during exploration in step 7.
  assert.deepEqual(s.bundles,['CST_R'],`step 8: bundles == ['CST_R'] (authored view; AF_R exploration deviation not persisted), observed ${JSON.stringify(s.bundles)}`);
  assert.equal(s.view,'right',`step 8: view == 'right' restored after reload, observed ${JSON.stringify(s.view)}`);
  await snap(8);evidence.step8=s;

  // 9. Click #lessonExplore ('Return to exploration'): lesson status 'inactive'.
  await page.locator('#lessonExplore').click();
  s=await state();
  assert.equal(s.lesson.status,'inactive',`step 9: lesson status 'inactive' after #lessonExplore, observed ${JSON.stringify(s.lesson.status)}`);
  await snap(9);evidence.step9=s;

  // 10. No page errors, no /api/ requests across the whole journey.
  assert.deepEqual(errors,[],`step 10: no page errors, observed ${JSON.stringify(errors)}`);
  assert.deepEqual(apiRequests,[],`step 10: no /api/ requests, observed ${JSON.stringify(apiRequests)}`);
  await snap(10);evidence.step10={errors,apiRequests};

  console.log(JSON.stringify({passed:true,...evidence},null,2));
  await writeFile('output/playwright/atlas-journey.json',JSON.stringify(evidence,null,2)+'\n');
}finally{await browser.close();}
