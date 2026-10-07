import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../viewer/vendor/three.module.js';
import {createSceneLighting} from '../../viewer/scene_lighting.js';

test('environment lighting keeps the direct rig when float targets are unavailable',()=>{
  const scene=new THREE.Scene(),previous=new THREE.Texture();
  scene.environment=previous;scene.environmentIntensity=.8;
  const lighting=createSceneLighting(THREE,{extensions:{has:()=>false}},scene);
  assert.equal(lighting.mode,'direct');
  assert.equal(scene.children.filter(light=>light.visible).length,5);
  assert.equal(scene.environment,previous);
  assert.equal(scene.environmentIntensity,.8);
  assert.throws(()=>lighting.setMode('environment'),/floating-point/);
  assert.equal(lighting.mode,'direct','a failed switch preserves the working rig');
  lighting.dispose();lighting.dispose();assert.equal(scene.children.length,0);
  assert.equal(scene.environment,previous);
});
