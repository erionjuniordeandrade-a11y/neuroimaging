import test from 'node:test';
import assert from 'node:assert/strict';
import {configTubeMaterial, configureTubeRadius, setTubeRadius} from '../../viewer/scene_materials.js';

test('radius uniform composes with the proximity hatch and updates without shader or geometry replacement',()=>{
  const material={userData:{},onBeforeCompile(shader){shader.uniforms.existing={value:1};},customProgramCacheKey(){return 'base';}};
  configTubeMaterial(material,{caseMap:true,geomFloorMm:3});
  configureTubeRadius(material,{bakedRadius:.22});
  const shader={uniforms:{},vertexShader:'#include <begin_vertex>',fragmentShader:'#include <color_fragment>'};
  material.onBeforeCompile(shader);
  assert.match(shader.vertexShader,/transformed \+= normal \* uTractlabRadiusDelta/);
  assert.match(shader.fragmentShader,/tractlabHatch/);
  assert.equal(shader.uniforms.existing.value,1);
  const hook=material.onBeforeCompile,key=material.customProgramCacheKey();material.needsUpdate=false;
  setTubeRadius(material,.5);
  assert.equal(shader.uniforms.uTractlabRadiusDelta.value,.5-.22);
  assert.equal(material.onBeforeCompile,hook);assert.equal(material.customProgramCacheKey(),key);assert.equal(material.needsUpdate,false);
  setTubeRadius(material,.04);assert.equal(shader.uniforms.uTractlabRadiusDelta.value,.04-.22);
});

test('radius works on unlit selection and ghost materials and rejects invalid widths',()=>{
  const material={userData:{}};configureTubeRadius(material,{bakedRadius:.3,radius:.6});
  const shader={uniforms:{},vertexShader:'#include <begin_vertex>',fragmentShader:''};material.onBeforeCompile(shader);
  assert.equal(shader.uniforms.uTractlabRadiusDelta.value,.3);
  assert.throws(()=>setTubeRadius(material,NaN),/radius/);
  assert.throws(()=>setTubeRadius(material,-.1),/radius/);
});
