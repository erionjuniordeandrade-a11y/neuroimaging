/** A display-only selection contour outside the projected bundle silhouette. */
import {configureTubeRadius,setTubeRadius} from './scene_materials.js';

const ACTIVE_PASSES=7;
const QUAD_VERTEX='varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';

/** Separable closing radius in drawing pixels. One pixel only bridged adjacent
 * streamlines, so a porous mask produced a lacework of per-fibre strokes; three
 * pixels close the gaps between fibres into a bundle silhouette while the
 * exterior edge pass still draws a single boundary. */
export const CLOSE_RADIUS_PX=3;
function morphFragment(reduce,radius=CLOSE_RADIUS_PX){
  const taps=[];for(let o=-radius;o<=radius;o++)taps.push(`sampleMask(${o.toFixed(1)})`);
  return `uniform sampler2D source;uniform vec2 pixel;uniform vec2 axis;varying vec2 vUv;
    float sampleMask(float offset){return texture2D(source,clamp(vUv+axis*pixel*offset,vec2(0.),vec2(1.))).r;}
    void main(){
      float value=${taps.reduce((acc,t)=>acc?`${reduce}(${acc},${t})`:t,'')};
      gl_FragColor=vec4(vec3(value),value);
    }`;
}

function maskTarget(THREE,depthBuffer=false){
  const target=new THREE.WebGLRenderTarget(1,1,{depthBuffer,stencilBuffer:false});
  target.texture.generateMipmaps=false;
  return target;
}

function visibleMesh(mesh){return !!(mesh?.visible&&!mesh.userData?.ghost&&mesh.geometry&&mesh.material);}
function visibleOpaqueMesh(mesh){
  if(!visibleMesh(mesh))return false;
  const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];
  return materials.every(material=>material&&!material.transparent&&material.opacity!==0);
}
function sameSources(next,current){return next.length===current.length&&next.every((mesh,i)=>mesh===current[i]);}

export function createFocusOutline(THREE,renderer){
  const mask=maskTarget(THREE,true),scratchA=maskTarget(THREE),scratchB=maskTarget(THREE);
  const selectedScene=new THREE.Scene(),occupancyScene=new THREE.Scene(),quadCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  const geometry=new THREE.PlaneGeometry(2,2),colour=new THREE.Color('#afd6ff'),pixel=new THREE.Vector2(1,1),drawingBufferSize=new THREE.Vector2();
  const makeMorph=(source,axis,reduce)=>new THREE.ShaderMaterial({depthTest:false,depthWrite:false,toneMapped:false,
    uniforms:{source:{value:source},pixel:{value:pixel},axis:{value:new THREE.Vector2(...axis)}},
    vertexShader:QUAD_VERTEX,fragmentShader:morphFragment(reduce)});
  const dilateXMaterial=makeMorph(mask.texture,[1,0],'max'),dilateYMaterial=makeMorph(scratchA.texture,[0,1],'max');
  const erodeXMaterial=makeMorph(scratchB.texture,[1,0],'min'),closedMaterial=makeMorph(scratchA.texture,[0,1],'min');
  const edgeMaterial=new THREE.ShaderMaterial({transparent:true,depthTest:false,depthWrite:false,toneMapped:false,
    uniforms:{source:{value:scratchB.texture},occupancy:{value:mask.texture},pixel:{value:pixel},colour:{value:colour}},vertexShader:QUAD_VERTEX,
    fragmentShader:`uniform sampler2D source;uniform sampler2D occupancy;uniform vec2 pixel;uniform vec3 colour;varying vec2 vUv;
      float sampleMask(vec2 offset){return texture2D(source,clamp(vUv+pixel*offset,vec2(0.),vec2(1.))).r;}
      float sampleOccupancy(){return texture2D(occupancy,vUv).r;}
      void main(){
        float centre=sampleMask(vec2(0.));
        float outside=max(max(sampleMask(vec2(-1.25,-1.25)),sampleMask(vec2(0.,-1.25))),max(sampleMask(vec2(1.25,-1.25)),max(sampleMask(vec2(-1.25,0.)),max(sampleMask(vec2(1.25,0.)),max(sampleMask(vec2(-1.25,1.25)),max(sampleMask(vec2(0.,1.25)),sampleMask(vec2(1.25,1.25))))))));
        float edge=smoothstep(.03,.30,max(0.,outside-centre));
        float exposed=edge*(1.-smoothstep(.45,.85,sampleOccupancy()));
        gl_FragColor=vec4(colour,exposed*.70);
        #include <colorspace_fragment>
      }`});
  const makeQuadScene=material=>{const targetScene=new THREE.Scene();targetScene.add(new THREE.Mesh(geometry,material));return targetScene;};
  const dilateXScene=makeQuadScene(dilateXMaterial),dilateYScene=makeQuadScene(dilateYMaterial);
  const erodeXScene=makeQuadScene(erodeXMaterial),closedScene=makeQuadScene(closedMaterial),edgeScene=makeQuadScene(edgeMaterial);
  let selectedSources=[],selectedCopies=[],occupancySources=[],occupancyCopies=[],width=0,height=0,disposed=false;
  const clearCopies=(targetScene,copies)=>{for(const mesh of copies)mesh.material.dispose();targetScene.clear();};
  const copyMaskMesh=source=>{
    const mat=new THREE.MeshBasicMaterial({color:0xffffff,toneMapped:false,fog:false});
    configureTubeRadius(mat,{bakedRadius:source.geometry.userData.bakedRadius});
    const mesh=new THREE.Mesh(source.geometry,mat);mesh.renderOrder=source.renderOrder;return mesh;
  };
  const syncSelected=next=>{if(sameSources(next,selectedSources))return;clearCopies(selectedScene,selectedCopies);selectedSources=next;selectedCopies=next.map(copyMaskMesh);selectedCopies.forEach(mesh=>selectedScene.add(mesh));};
  const syncOccupancy=next=>{if(sameSources(next,occupancySources))return;clearCopies(occupancyScene,occupancyCopies);occupancySources=next;occupancyCopies=next.map(copyMaskMesh);occupancyCopies.forEach(mesh=>occupancyScene.add(mesh));};
  const clearSelected=()=>{clearCopies(selectedScene,selectedCopies);selectedSources=[];selectedCopies=[];};
  const clearOccupancy=()=>{clearCopies(occupancyScene,occupancyCopies);occupancySources=[];occupancyCopies=[];};
  const resize=(w,h)=>{for(const target of [mask,scratchA,scratchB])target.setSize(w,h);pixel.set(1/w,1/h);};
  const renderSize=(w,h)=>{
    let nextWidth=Math.max(1,Math.round(w)),nextHeight=Math.max(1,Math.round(h));
    const actual=renderer.getDrawingBufferSize?.(drawingBufferSize)||drawingBufferSize;
    if(Number.isFinite(actual?.x)&&Number.isFinite(actual?.y)&&actual.x>0&&actual.y>0){nextWidth=Math.round(actual.x);nextHeight=Math.round(actual.y);}
    return [nextWidth,nextHeight];
  };
  const syncTransforms=(copies,sources,radius)=>copies.forEach((copy,i)=>{
    copy.matrixAutoUpdate=false;copy.matrix.copy(sources[i].matrixWorld);
    const scale=Number(sources[i].userData.radiusScale);setTubeRadius(copy.material,radius*(Number.isFinite(scale)?scale:1));
  });
  const diagnostics=()=>({passes:!disposed&&selectedCopies.length?ACTIVE_PASSES:0,closeRadiusPx:1,edgeWidthPx:1.25,targets:3,disposed});
  function render(meshes,camera,w,h,radius,contextMeshes=[]){
    if(disposed)return;
    syncSelected(meshes.filter(visibleMesh));
    if(!selectedCopies.length){clearOccupancy();return;}
    syncOccupancy((Array.isArray(contextMeshes)?contextMeshes:[]).filter(visibleOpaqueMesh));
    const [nextWidth,nextHeight]=renderSize(w,h);
    if(nextWidth!==width||nextHeight!==height){width=nextWidth;height=nextHeight;resize(width,height);}
    syncTransforms(selectedCopies,selectedSources,radius);syncTransforms(occupancyCopies,occupancySources,radius);
    const target=renderer.getRenderTarget(),autoClear=renderer.autoClear,infoAutoReset=renderer.info.autoReset;
    const clearColour=renderer.getClearColor(new THREE.Color()),alpha=renderer.getClearAlpha();
    try{
      renderer.autoClear=false;renderer.info.autoReset=false;renderer.setClearColor(0x000000,0);
      renderer.setRenderTarget(mask);renderer.clear(true,true,true);renderer.render(selectedScene,camera);
      renderer.setRenderTarget(scratchA);renderer.clear(true,true,true);renderer.render(dilateXScene,quadCamera);
      renderer.setRenderTarget(scratchB);renderer.clear(true,true,true);renderer.render(dilateYScene,quadCamera);
      renderer.setRenderTarget(scratchA);renderer.clear(true,true,true);renderer.render(erodeXScene,quadCamera);
      renderer.setRenderTarget(scratchB);renderer.clear(true,true,true);renderer.render(closedScene,quadCamera);
      renderer.setRenderTarget(mask);renderer.clear(true,true,true);renderer.render(occupancyScene,camera);
      renderer.setRenderTarget(target);renderer.render(edgeScene,quadCamera);
    }finally{renderer.setRenderTarget(target);renderer.setClearColor(clearColour,alpha);renderer.autoClear=autoClear;renderer.info.autoReset=infoAutoReset;}
  }
  const outline={render,dispose(){if(disposed)return;disposed=true;clearSelected();clearOccupancy();mask.dispose();scratchA.dispose();scratchB.dispose();dilateXMaterial.dispose();dilateYMaterial.dispose();erodeXMaterial.dispose();closedMaterial.dispose();edgeMaterial.dispose();geometry.dispose();}};
  Object.defineProperties(outline,{active:{enumerable:true,get:()=>selectedCopies.length>0},diagnostics:{enumerable:true,get:diagnostics}});
  return outline;
}
