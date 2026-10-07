import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {SphereGeometry} from '../../viewer/vendor/three.module.js';
import {streamlineLesionDistance,layerLesionDistances} from '../../viewer/lesion_distance.js';

// The browser import map resolves the pinned BVH's bare `three` import. Node's
// test runner uses the exact vendored module with that one specifier resolved.
const threeUrl=pathToFileURL(resolve('viewer/vendor/three.module.js')).href;
const bvhSource=(await readFile(resolve('viewer/vendor/three-mesh-bvh.js'),'utf8'))
  .replace("from 'three'",`from '${threeUrl}'`);
const {MeshBVH}=await import(`data:text/javascript;base64,${Buffer.from(bvhSource).toString('base64')}`);
const geometry=new SphereGeometry(10,64,32);
const bvh=new MeshBVH(geometry);

test('sphere surface distance, crossing segment, and sampled refinement',()=>{
  const outside=new Float32Array([14,-5,0,14,0,0,14,5,0]);
  const four=streamlineLesionDistance(outside,bvh);
  assert.ok(Math.abs(four.minMm-4)<.15,`surface distance ${four.minMm}`);
  const crossing=new Float32Array([18,0,0,14,0,0,-14,0,0,-18,0,0]);
  const exact=streamlineLesionDistance(crossing,bvh,{sample:1});
  const sampled=streamlineLesionDistance(crossing,bvh,{sample:3});
  assert.ok(exact.minMm<.15,`crossing distance ${exact.minMm}`);
  assert.ok(Math.abs(sampled.minMm-exact.minMm)<.2);
});

test('layer distances yield, report progress, and cache by geometry and lesion',async()=>{
  const layer={flat:new Float32Array([14,0,0,14,1,0,18,0,0,10,0,0]),lineCount:2,k:2};
  const progress=[];
  const values=await layerLesionDistances(layer,bvh,{onProgress:(n)=>progress.push(n),
    frame:callback=>setImmediate(callback),budgetMs:0});
  assert.equal(values.length,2);
  assert.ok(Math.abs(values[0]-4)<.15);
  assert.ok(values[1]<.15);
  assert.deepEqual(progress,[1,2]);
  assert.equal(await layerLesionDistances(layer,bvh),values);
});
