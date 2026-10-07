import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../viewer/vendor/three.module.js';
import {createFocusOutline} from '../../viewer/focus_outline.js';

test('focus contour shares source geometry, preserves materials, and restores renderer state',()=>{
  let target=null;const clear=new THREE.Color('#123456'),calls=[];
  const renderer={autoClear:true,info:{autoReset:true},alpha:1,
    getRenderTarget:()=>target,setRenderTarget:value=>target=value,
    getDrawingBufferSize:size=>size.set(1730,614),
    getClearColor:color=>color.copy(clear),getClearAlpha(){return this.alpha;},
    setClearColor(color,alpha){clear.set(color);this.alpha=alpha;},clear(){},render(scene){calls.push({target,children:[...scene.children]});}};
  const outline=createFocusOutline(THREE,renderer),geometry=new THREE.CylinderGeometry(.22,.22,5);
  geometry.userData.bakedRadius=.22;
  const original=new THREE.MeshStandardMaterial({color:'#5533aa'}),mesh=new THREE.Mesh(geometry,original);mesh.userData.radiusScale=1;
  const contextGeometry=new THREE.CylinderGeometry(.16,.16,3),contextOriginal=new THREE.MeshStandardMaterial({color:'#336655'});
  contextGeometry.userData.bakedRadius=.16;
  const contextMesh=new THREE.Mesh(contextGeometry,contextOriginal);contextMesh.userData.radiusScale=.8;
  const translucentMesh=new THREE.Mesh(new THREE.CylinderGeometry(.1,.1,2),new THREE.MeshBasicMaterial({transparent:true,opacity:.4}));
  translucentMesh.geometry.userData.bakedRadius=.1;translucentMesh.userData.radiusScale=.5;
  let sourceDisposals=0,contextDisposals=0;geometry.addEventListener('dispose',()=>sourceDisposals++);contextGeometry.addEventListener('dispose',()=>contextDisposals++);
  outline.render([mesh],new THREE.PerspectiveCamera(),865,307,.22,[mesh,contextMesh,translucentMesh]);
  assert.equal(calls.length,7,'renders selection mask, separable closing, occupancy, and exterior edge');
  assert.equal(outline.diagnostics.passes,calls.length,'reports the GPU passes used by the active contour');
  assert.equal(calls[0].children[0].geometry,geometry);assert.equal(mesh.material,original);assert.equal(contextMesh.material,contextOriginal);
  assert.equal(calls[0].target.width,1730);assert.equal(calls[0].target.height,614,'uses the drawing-buffer resolution for a smooth contour');
  assert.equal(new Set(calls.slice(0,6).map(call=>call.target)).size,3,'bounds morphology memory to one mask and two scratch targets');
  assert.equal(calls[1].target,calls[3].target,'reuses the first scratch target only after its input has been consumed');
  assert.equal(calls[2].target,calls[4].target,'keeps the closed silhouette in the second scratch target');
  assert.equal(calls[5].children.length,2,'occupancy combines selected and other visible opaque tract meshes only');
  assert.equal(calls[5].children[0].geometry,geometry);assert.equal(calls[5].children[1].geometry,contextGeometry);
  assert.equal(calls[6].target,null,'composites the contour into the previous render target');
  assert.equal(calls[6].children[0].material.uniforms.source.value,calls[4].target.texture,
    'the exterior edge reads the closed silhouette, not the sparse source mask');
  assert.equal(calls[6].children[0].material.uniforms.occupancy.value,calls[5].target.texture,
    'suppresses the contour where the visible tract union occupies the edge');
  assert.equal(renderer.autoClear,true);assert.equal(renderer.info.autoReset,true);assert.equal(target,null);assert.equal(clear.getHexString(),'123456');
  const contourTargets=[...new Set(calls.slice(0,6).map(call=>call.target))];let contourTargetDisposals=0;
  contourTargets.forEach(contourTarget=>contourTarget.addEventListener('dispose',()=>contourTargetDisposals++));
  const maskMesh=calls[0].children[0],occupancyMesh=calls[5].children[0];outline.render([mesh],new THREE.PerspectiveCamera(),865,307,.3,[mesh,contextMesh,translucentMesh]);
  assert.equal(calls[7].children[0],maskMesh,'display changes reuse the selection mask mesh');
  assert.equal(calls[12].children[0],occupancyMesh,'display changes reuse the occupancy mask mesh');
  assert.equal(sourceDisposals,0);mesh.userData.ghost=true;
  outline.render([mesh],new THREE.PerspectiveCamera(),865,307,.3,[mesh,contextMesh,translucentMesh]);assert.equal(outline.active,false,'ghosted support is not outlined');
  outline.dispose();outline.dispose();assert.equal(contourTargetDisposals,3);assert.equal(sourceDisposals,0);assert.equal(contextDisposals,0);
});
