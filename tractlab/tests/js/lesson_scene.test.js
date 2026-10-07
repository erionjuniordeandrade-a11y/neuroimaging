import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_SCENE,LESSONS} from '../../viewer/lesson_content.js';
import {resolveScene} from '../../viewer/lesson_scene.js';

const motor=LESSONS.find(l=>l.id==='motor-cst');
const fat=LESSONS.find(l=>l.id==='fat-language');
const interoception=LESSONS.find(l=>l.id==='interoception');

test('follow-side resolution tracks the current hemisphere control',()=>{
  const step={scene:{...DEFAULT_SCENE,side:'follow'}};
  assert.equal(resolveScene(step,'L').side,'L');
  assert.equal(resolveScene(step,'R').side,'R');
  assert.equal(resolveScene(step,'both').side,'L');
});

test('fixed-side steps ignore the hemisphere control',()=>{
  assert.equal(resolveScene(fat.steps[0],'R').side,'L');
  assert.equal(resolveScene(fat.steps[0],'L').side,'L');
  assert.equal(resolveScene(fat.steps[2],'R').bundleId,'FAT_L');
});

test('view follow maps L to left and R to right',()=>{
  const step={scene:{...DEFAULT_SCENE,side:'follow',view:'follow'}};
  assert.equal(resolveScene(step,'L').view,'left');
  assert.equal(resolveScene(step,'R').view,'right');
});

test('deep and surface fall back to declared defaults',()=>{
  const r=resolveScene({scene:{...DEFAULT_SCENE}},'L');
  assert.equal(r.deep,false);
  assert.equal(r.surface,null);
});

test('motor v2: step 1 lights the precentral parcels with no bundle; step 3 reveals CST on the current side',()=>{
  const s1=resolveScene(motor.steps[0],'L');
  assert.deepEqual(s1.bundleIds,[]);assert.equal(s1.regions[0].id,8);assert(s1.regions.length>1);
  assert.deepEqual(resolveScene(motor.steps[2],'L').bundleIds,['CST_L']);
  assert.deepEqual(resolveScene(motor.steps[2],'R').bundleIds,['CST_R']);
  assert.equal(motor.steps.length,10);
});

test('interoception region follows the resolved side',()=>{
  assert.deepEqual(resolveScene(interoception.steps[0],'L').region,{id:111,hemi:'L'});
  assert.deepEqual(resolveScene(interoception.steps[0],'R').region,{id:111,hemi:'R'});
  assert.deepEqual(resolveScene(interoception.steps[1],'L').region,{id:106,hemi:'L'});
  assert.deepEqual(resolveScene(interoception.steps[1],'R').region,{id:106,hemi:'R'});
});

test('motor v2: deep structures appear only on the steps that declare them, with side-resolved ids',()=>{
  const deepSteps=motor.steps.map((s,i)=>resolveScene(s,'R').deep?i:-1).filter(i=>i>=0);
  assert(deepSteps.length>0&&deepSteps.length<motor.steps.length);
  assert.equal(resolveScene(motor.steps[0],'R').deep,false);
  const r=resolveScene(motor.steps[5],'R');
  assert(r.deepRegionIds.length>0);assert(r.deepRegionIds.every(id=>id.endsWith('-rh')));
  assert(resolveScene(motor.steps[5],'L').deepRegionIds.every(id=>id.endsWith('-lh')));
});

test('every installed lesson step resolves without throwing for both sides',()=>{
  for(const lesson of LESSONS) for(const s of lesson.steps) for(const hemi of ['L','R'])
    assert.doesNotThrow(()=>resolveScene(s,hemi),`${lesson.id} step "${s.title}" hemi ${hemi}`);
});

// SCENE GRAMMAR v2 -----------------------------------------------------------------

test('v2 bundles: a family list resolves to per-side ids, superseding a v1 bundle',()=>{
  const step={scene:{bundles:['CST','ML']}};
  assert.deepEqual(resolveScene(step,'L').bundleIds,['CST_L','ML_L']);
  assert.deepEqual(resolveScene(step,'R').bundleIds,['CST_R','ML_R']);
  const superseded={scene:{bundle:'FAT',bundles:['CST']}};
  assert.deepEqual(resolveScene(superseded,'L').bundleIds,['CST_L']);
  assert.equal(resolveScene(superseded,'L').bundleId,'FAT_L','v1 bundleId keeps its own v1 meaning');
});

test('v2 bundles: midline families never take a side suffix',()=>{
  const step={scene:{bundles:['CC','MCP','CST']}};
  assert.deepEqual(resolveScene(step,'L').bundleIds,['CC','MCP','CST_L']);
  assert.deepEqual(resolveScene(step,'R').bundleIds,['CC','MCP','CST_R']);
});

test('v2 ghost: a separate dimmed family list resolves independently of bundles',()=>{
  const step={scene:{bundles:['CST'],ghost:['ML','DRTT']}};
  const resolved=resolveScene(step,'L');
  assert.deepEqual(resolved.bundleIds,['CST_L']);
  assert.deepEqual(resolved.ghostIds,['ML_L','DRTT_L']);
});

test('v2 bundles/ghost: an unknown family throws',()=>{
  assert.throws(()=>resolveScene({scene:{bundles:['NOPE']}},'L'),/Unknown bundle family: NOPE/);
  assert.throws(()=>resolveScene({scene:{ghost:['NOPE']}},'L'),/Unknown bundle family: NOPE/);
});

test('v2 regions: the first entry is the focus and follow resolves per side; v1 region keeps its own v1 meaning',()=>{
  const step={scene:{region:{id:99,hemi:'follow'},regions:[{id:8,hemi:'follow'},{id:74,hemi:'R'}]}};
  const resolvedL=resolveScene(step,'L');
  assert.deepEqual(resolvedL.regions,[{id:8,hemi:'L'},{id:74,hemi:'R'}],'first entry is the focus');
  assert.deepEqual(resolvedL.region,{id:99,hemi:'L'},'v1 region field is untouched by v2 regions');
  const resolvedR=resolveScene(step,'R');
  assert.deepEqual(resolvedR.regions,[{id:8,hemi:'R'},{id:74,hemi:'R'}]);
});

test('v2 regions: absent falls back to the v1 single region unchanged',()=>{
  const step={scene:{region:{id:8,hemi:'follow'}}};
  assert.deepEqual(resolveScene(step,'L').regions,[{id:8,hemi:'L'}]);
  assert.deepEqual(resolveScene(step,'R').regions,[{id:8,hemi:'R'}]);
  assert.deepEqual(resolveScene({scene:{}},'L').regions,[]);
});

test('v2 camera: follow resolves to left/right by side and supplies zoom/tweenMs defaults',()=>{
  assert.deepEqual(resolveScene({scene:{camera:{view:'follow'}}},'L').camera,{view:'left',zoom:1,tweenMs:900});
  assert.deepEqual(resolveScene({scene:{camera:{view:'follow'}}},'R').camera,{view:'right',zoom:1,tweenMs:900});
  assert.deepEqual(resolveScene({scene:{camera:{view:'superior',zoom:1.6,tweenMs:400}}},'L').camera,
    {view:'superior',zoom:1.6,tweenMs:400});
  assert.equal(resolveScene({scene:{}},'L').camera,null);
});

test('v2 durationSec: defaults to 20s, an override is passed through unchanged',()=>{
  assert.equal(resolveScene({scene:{}},'L').durationSec,20);
  assert.equal(resolveScene({scene:{durationSec:45}},'L').durationSec,45);
});

test('v2 deepRegions: passes the subcortex id list through; empty by default',()=>{
  assert.deepEqual(resolveScene({scene:{}},'L').deepRegionIds,[]);
  assert.deepEqual(resolveScene({scene:{deepRegions:['THA-lh','THA-rh']}},'L').deepRegionIds,['THA-lh','THA-rh']);
});
