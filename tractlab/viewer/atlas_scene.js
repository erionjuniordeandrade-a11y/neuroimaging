import {BUNDLE_TINTS,MAX_BUNDLES} from './bundle_picker.js';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {DRACOLoader} from 'three/addons/loaders/DRACOLoader.js';
import {ATLAS_VIEWS,decodeAtlasLabels,decodeAtlasBundle} from './atlas_data.js';
const MANIFEST_SHA256='f0bb6486476c5f0e010f5fc5ba03314703a4f5d0ac136a4cf75a1660646858b9';
const sha256=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');

// `selected` is 0 (none), 1 (primary/picked — drives #pickedName) or 2 (secondary
// highlight from a v2 `regions` list). One attribute, two visual tiers: extending it
// keeps a single per-vertex mechanism instead of a second highlight mesh.
const pointVertex=`
attribute float selected; varying float chosen; varying float ringed; varying float facing;
uniform float pixelRatio;
void main(){
  vec4 mv=modelViewMatrix*vec4(position,1.0);
  vec3 n=normalize(normalMatrix*normal);
  facing=max(0.0,dot(n,normalize(-mv.xyz)));
  chosen=selected>1.5?0.0:selected; ringed=selected>1.5?1.0:0.0;
  gl_Position=projectionMatrix*mv;
  gl_PointSize=(1.25+0.65*chosen+0.4*ringed)*pixelRatio;
}`;
const pointFragment=`
varying float chosen; varying float ringed; varying float facing; uniform float surface;
void main(){
  float d=length(gl_PointCoord-0.5); if(d>0.5)discard;
  float edge=1.0-smoothstep(0.25,0.5,d);
  vec3 tint=mix(vec3(0.30,0.64,0.87),vec3(0.98,0.76,0.38),chosen);
  tint=mix(tint,vec3(0.55,0.86,0.58),ringed);
  float a=surface*(0.25+0.75*facing)+chosen*0.4+ringed*0.28;
  gl_FragColor=vec4(tint,edge*min(a,0.9));
}`;

/** Owns one reference scene. Its inputs never include a case result or mask. */
export async function createAtlasScene(mount,{onPick=()=>{},onStatus=()=>{},onInteraction=()=>{}}={}) {
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const scene=new THREE.Scene();
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x070c12,0);
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.localClippingEnabled=true;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
  renderer.domElement.setAttribute('aria-label','Interactive HCP reference atlas. Use the labelled area and view controls for keyboard access.');
  mount.append(renderer.domElement);
  const camera=new THREE.PerspectiveCamera(35,1,.1,2000);camera.up.set(0,0,1);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=.12;controls.enablePan=false;
  controls.minDistance=150;controls.maxDistance=900;
  scene.add(new THREE.AmbientLight(0xc7def5,1.2));
  const light=new THREE.DirectionalLight(0xd2e7ff,2);light.position.set(-150,200,300);scene.add(light);
  const centre=new THREE.Vector3(0,-18,8),hemis={},deep=[],bundles=new Map(),frameGeometry=[];
  let playing=false,profile='teaching',frame=0,last=0,time=0,disposed=false,view='left';
  let selected={hemi:'L',id:8},highlighted=[],visibleHemi='both',surface=.6,deepVisible=false,framed=false,autoFrame=true;
  let deepHighlightIds=[],cameraTween=null;
  const traces=[];let frames=0;
  const bundleIdsBy=ghost=>[...bundles].filter(([,v])=>v.ghost===ghost).map(([id])=>id);
  function draw(now=0){
    frame=0;if(disposed)return;
    const moving=playing&&profile==='teaching'&&!reduced.matches&&!document.hidden;
    if(moving&&last)time+=Math.min((now-last)/1000,.05);
    last=moving?now:0;
    for(const trace of traces)trace.material.uniforms.time.value=time;
    const changed=controls.update();renderer.render(scene,camera);frames++;
    if(moving||changed)requestDraw();
  }
  function requestDraw(){if(!frame&&!disposed)frame=requestAnimationFrame(draw);}
  controls.addEventListener('change',requestDraw);
  controls.addEventListener('start',()=>{autoFrame=false;onInteraction();});
  reduced.addEventListener('change',requestDraw);
  document.addEventListener('visibilitychange',requestDraw);
  const resize=()=>{const w=mount.clientWidth,h=mount.clientHeight;
    if(w<=0||h<=0)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);
    if(framed&&autoFrame)setView(view);requestDraw();};
  const observer=new ResizeObserver(resize);observer.observe(mount);
  function disposeBase(){disposed=true;cameraTween=null;cancelAnimationFrame(frame);observer.disconnect();controls.dispose();
    reduced.removeEventListener('change',requestDraw);document.removeEventListener('visibilitychange',requestDraw);
    renderer.dispose();renderer.domElement.remove();}
  resize();
  // Shared frustum-fit math for an instant cut (setView) and an animated flight (flyTo).
  function computeViewTarget(key,zoom=1) {
    const resolved=key in ATLAS_VIEWS?key:'left';
    const direction=new THREE.Vector3(...ATLAS_VIEWS[resolved]);
    if(resolved==='medial'&&visibleHemi==='R')direction.x*=-1;
    direction.normalize();
    const right=new THREE.Vector3().crossVectors(camera.up,direction).normalize();
    const up=new THREE.Vector3().crossVectors(direction,right).normalize();
    const tanV=Math.tan(THREE.MathUtils.degToRad(camera.fov/2)),tanH=tanV*camera.aspect;
    let distance=250;const p=new THREE.Vector3();
    for(const geo of frameGeometry){const pos=geo.attributes.position;
      for(let i=0;i<pos.count;i++){p.fromBufferAttribute(pos,i).sub(centre);
        distance=Math.max(distance,p.dot(direction)+Math.abs(p.dot(up))/tanV,
          p.dot(direction)+Math.abs(p.dot(right))/tanH);}}
    distance=distance*1.08/Math.max(zoom,.1);
    return {view:resolved,position:new THREE.Vector3().copy(centre).addScaledVector(direction,distance)};
  }
  function setView(key='left') {
    cameraTween=null;
    const target=computeViewTarget(key,1);
    view=target.view;camera.position.copy(target.position);autoFrame=true;
    controls.target.copy(centre);controls.update();requestDraw();
  }
  const easeInOutQuad=t=>t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
  /** Animate the camera to a named view; prefers-reduced-motion jumps instantly. */
  function flyTo({view:key='left',zoom=1,tweenMs=900}={}) {
    const target=computeViewTarget(key,zoom);
    view=target.view;autoFrame=false;
    if(reduced.matches||tweenMs<=0){
      cameraTween=null;camera.position.copy(target.position);controls.target.copy(centre);
      controls.update();requestDraw();return;
    }
    const fromPosition=camera.position.clone(),fromTarget=controls.target.clone(),start=performance.now();
    const token=cameraTween={};
    function step(now){
      if(cameraTween!==token||disposed)return;
      const t=Math.min(1,(now-start)/tweenMs),eased=easeInOutQuad(t);
      camera.position.lerpVectors(fromPosition,target.position,eased);
      controls.target.lerpVectors(fromTarget,centre,eased);
      controls.update();requestDraw();
      if(t<1)requestAnimationFrame(step);else cameraTween=null;
    }
    requestAnimationFrame(step);
  }
  setView();onStatus('Loading the reference atlas…');
  let manifest;
  try{
    const metaResponse=await fetch('./atlas/manifest.json');
    if(!metaResponse.ok)throw new Error(`Atlas manifest unavailable (${metaResponse.status})`);
    const manifestBytes=await metaResponse.arrayBuffer();
    if(await sha256(manifestBytes)!==MANIFEST_SHA256)throw new Error('Atlas manifest integrity failed');
    manifest=JSON.parse(new TextDecoder().decode(manifestBytes));
  }catch(error){disposeBase();throw error;}
  async function checked(path){
    const record=manifest.assets.find(a=>a.path===path);if(!record)throw new Error('Unlisted atlas asset');
    const response=await fetch(`./atlas/${path}`);if(!response.ok)throw new Error(`Atlas asset unavailable: ${path}`);
    const bytes=await response.arrayBuffer();
    const hash=await sha256(bytes);
    if(hash!==record.sha256||bytes.byteLength!==record.bytes)throw new Error(`Atlas asset integrity failed: ${path}`);
    return bytes;
  }
  const json=async path=>JSON.parse(new TextDecoder().decode(await checked(path)));
  const draco=new DRACOLoader();draco.setDecoderPath('./vendor/addons/libs/draco/gltf/');draco.setWorkerLimit(2);
  const loader=new GLTFLoader();loader.setDRACOLoader(draco);
  async function geometry(path){
    const gltf=await loader.parseAsync(await checked(path),'');
    const mesh=gltf.scene.getObjectByProperty('isMesh',true);if(!mesh)throw new Error('Empty atlas mesh');
    gltf.scene.updateMatrixWorld(true);
    const geo=mesh.geometry;geo.applyMatrix4(mesh.matrixWorld);geo.computeVertexNormals();
    mesh.material?.dispose();return geo;
  }
  let surfaceMeta,tractMeta,tractBuffer,labels,sub;
  try {
    [surfaceMeta,tractMeta,tractBuffer]=await Promise.all([json('surface.json'),json('tracts.json'),checked('tracts.bin')]);
    const [left,right,labelBuffer]=await Promise.all([geometry('cortex-L.glb'),geometry('cortex-R.glb'),checked('surface-labels.bin')]);
    labels=decodeAtlasLabels(surfaceMeta,labelBuffer,[left.attributes.position.count,right.attributes.position.count]);
    for(const [i,h] of ['L','R'].entries()) {
      const geo=[left,right][i],count=geo.attributes.position.count;
      geo.setAttribute('selected',new THREE.BufferAttribute(new Float32Array(count),1));
      const points=new THREE.Points(geo,new THREE.ShaderMaterial({
        uniforms:{pixelRatio:{value:renderer.getPixelRatio()},surface:{value:surface}},
        vertexShader:pointVertex,fragmentShader:pointFragment,transparent:true,depthWrite:false,
        blending:THREE.NormalBlending,toneMapped:false,
      }));
      const shell=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0x31536f,
        roughness:.88,metalness:0,transparent:true,opacity:.10,depthWrite:false,side:THREE.DoubleSide}));
      const group=new THREE.Group();group.add(shell,points);scene.add(group);
      hemis[h]={geo,points,shell,group,count};
      frameGeometry.push(geo);
    }
    const context=new THREE.Mesh(await geometry('inferior-context.glb'),new THREE.MeshStandardMaterial({
      color:0x33506a,roughness:.8,transparent:true,opacity:.16,depthWrite:false,side:THREE.DoubleSide,
      clippingPlanes:[new THREE.Plane(new THREE.Vector3(0,0,-1),0)]}));
    scene.add(context);
    frameGeometry.push(context.geometry);
    const bounds=new THREE.Box3();for(const geo of frameGeometry){geo.computeBoundingBox();bounds.union(geo.boundingBox);}bounds.getCenter(centre);
    sub=await json('subcortex.json');
    const deepColours={THA:0xc69d71,HIP:0x68afae,AMY:0xd28b90,PUT:0xa19dbb,CAU:0x829eba,GP:0xb5ab78,NAc:0xc39178};
    await Promise.all(sub.structures.map(async entry=>{
      const mesh=new THREE.Mesh(await geometry(entry.mesh),new THREE.MeshStandardMaterial({
        color:deepColours[entry.id.split('-')[0]],roughness:.5,transparent:true,opacity:.7,depthWrite:false}));
      mesh.visible=false;mesh.userData={...entry,kind:'deep'};scene.add(mesh);deep.push(mesh);
    }));
  } catch(error){
    scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});draco.dispose();disposeBase();throw error;
  }
  function applySelection(){
    for(const h of ['L','R']){const a=hemis[h].geo.attributes.selected;
      for(let i=0;i<a.count;i++){const id=labels[h][i];
        a.array[i]=(id!==0&&h===selected.hemi&&id===selected.id)?1
          :(id!==0&&highlighted.some(r=>r.hemi===h&&r.id===id))?2:0;
      }
      a.needsUpdate=true;
    }requestDraw();
  }
  function select(hemi,id){
    if(!(id in surfaceMeta.sets.glasser.regions[hemi]))return;
    selected={hemi,id};highlighted=[];applySelection();
  }
  /** Light a set of parcels as secondary context (tier 2), leaving the primary
   * `select()` picked region (tier 1) untouched. Empty list clears the set. */
  function highlight(list=[]){
    highlighted=(list||[]).filter(r=>r&&['L','R'].includes(r.hemi)&&
      Number.isInteger(r.id)&&r.id in surfaceMeta.sets.glasser.regions[r.hemi]);
    applySelection();
  }
  function setHemisphere(h){
    visibleHemi=['L','R','both'].includes(h)?h:'both';
    for(const key of ['L','R'])hemis[key].group.visible=visibleHemi==='both'||visibleHemi===key;
    for(const mesh of deep)mesh.visible=deepVisible&&(visibleHemi==='both'||visibleHemi===mesh.userData.hemisphere);
    requestDraw();
  }
  function setDeep(on){deepVisible=!!on;setHemisphere(visibleHemi);}
  /** Dim deep structures not named in `ids` (others stay at full opacity); empty = all as today. */
  function setDeepHighlight(ids=[]){
    deepHighlightIds=Array.isArray(ids)?ids:[];
    for(const mesh of deep)mesh.material.opacity=
      (deepHighlightIds.length&&!deepHighlightIds.includes(mesh.userData.id))?.15:.7;
    requestDraw();
  }
  function clearBundles(){
    for(const {group} of bundles.values()){scene.remove(group);group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});}
    bundles.clear();traces.length=0;
  }
  const GHOST_TINT=0x818d99;
  /** Show a set of bundles at once. `ghost` ids are rendered dimmer, untinted-by-index and
   * without the animated trace — a "kept visible but de-emphasised" layer under the primaries. */
  function setBundles(ids,{ghost=[]}={}) {
    clearBundles();
    const ghostSet=new Set(ghost);
    const primaryIds=[...new Set(ids)].filter(id=>!ghostSet.has(id));
    const ghostIds=[...new Set(ghost)].filter(id=>!primaryIds.includes(id));
    const ordered=[...primaryIds.map(id=>({id,isGhost:false})),...ghostIds.map(id=>({id,isGhost:true}))].slice(0,MAX_BUNDLES);
    let colourIndex=0;
    for(const {id,isGhost} of ordered) {
      const meta=tractMeta.bundles.find(b=>b.id===id);if(!meta)continue;
      const lines=decodeAtlasBundle(meta,tractBuffer),positions=[],arc=[],tracePositions=[],traceArc=[];
      for(const [li,line] of lines.entries()) {
        let length=0;const lengths=[0];
        for(let i=1;i<line.length;i++){length+=Math.hypot(...line[i].map((x,j)=>x-line[i-1][j]));lengths.push(length);}
        for(let i=1;i<line.length;i++) {
          positions.push(...line[i-1],...line[i]);arc.push(lengths[i-1]/length,lengths[i]/length);
          if(li%Math.ceil(lines.length/18)===0){tracePositions.push(...line[i-1],...line[i]);traceArc.push(lengths[i-1]/length,lengths[i]/length);}
        }
      }
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
      geo.setAttribute('arc',new THREE.Float32BufferAttribute(arc,1));
      const tint=isGhost?GHOST_TINT:BUNDLE_TINTS[colourIndex%BUNDLE_TINTS.length];
      const alpha=isGhost?.12:.30;
      if(!isGhost)colourIndex++;
      const group=new THREE.Group();group.add(new THREE.LineSegments(geo,new THREE.ShaderMaterial({
        uniforms:{tint:{value:new THREE.Color(tint)},alpha:{value:alpha}},
        vertexShader:'attribute float arc; varying float t; void main(){t=arc;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
        fragmentShader:'uniform vec3 tint; uniform float alpha; varying float t; void main(){float feather=smoothstep(0.0,0.06,min(t,1.0-t));gl_FragColor=vec4(tint,alpha*feather);}',
        transparent:true,depthWrite:false,blending:THREE.NormalBlending,toneMapped:false})));
      if(!isGhost){
        const traceGeo=new THREE.BufferGeometry();traceGeo.setAttribute('position',new THREE.Float32BufferAttribute(tracePositions,3));
        traceGeo.setAttribute('arc',new THREE.Float32BufferAttribute(traceArc,1));
        const material=new THREE.ShaderMaterial({uniforms:{time:{value:time},phase:{value:colourIndex*.7}},
          vertexShader:'attribute float arc; varying float t; void main(){t=arc;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
          fragmentShader:`uniform float time; uniform float phase; varying float t;
            void main(){float p=0.18+0.27*(1.0-cos(time*0.7+phase));
              float a=max(exp(-pow((t-p)/0.025,2.0)),exp(-pow((t-(1.0-p))/0.025,2.0)));
              if(a<0.02)discard; gl_FragColor=vec4(1.0,0.78,0.36,a*0.85);}`,
          transparent:true,depthWrite:false,depthTest:false,blending:THREE.NormalBlending,toneMapped:false});
        const trace=new THREE.LineSegments(traceGeo,material);trace.visible=profile==='teaching';trace.renderOrder=5;
        group.add(trace);traces.push(trace);
      }
      bundles.set(id,{group,ghost:isGhost,alpha});scene.add(group);
    }requestDraw();
  }
  const ray=new THREE.Raycaster(),ndc=new THREE.Vector2();let down;
  renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
  renderer.domElement.addEventListener('pointerup',e=>{
    if(!down||e.button!==0||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5){down=null;return;}down=null;
    const rect=renderer.domElement.getBoundingClientRect();ndc.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);
    ray.setFromCamera(ndc,camera);
    const candidates=Object.values(hemis).filter(h=>h.group.visible).map(h=>h.shell);
    const hit=ray.intersectObjects(deep.filter(x=>x.visible),false)[0]||ray.intersectObjects(candidates,false)[0];if(!hit)return;
    if(hit.object.userData.kind==='deep'){onPick({deep:hit.object.userData});return;}
    const h=Object.keys(hemis).find(k=>hemis[k].shell===hit.object),pos=hemis[h].geo.attributes.position;
    const p=new THREE.Vector3();let nearest=hit.face.a,distance=Infinity;
    for(const i of [hit.face.a,hit.face.b,hit.face.c]){p.fromBufferAttribute(pos,i);const d=p.distanceToSquared(hit.point);if(d<distance){nearest=i;distance=d;}}
    select(h,labels[h][nearest]);onPick({hemi:h,id:labels[h][nearest],vertex:nearest});
  });
  select('L',8);framed=true;resize();setView();onStatus('Atlas ready');
  function snapshot(){return {selected:{...selected},highlighted:highlighted.map(r=>({...r})),hemisphere:visibleHemi,
    bundles:bundleIdsBy(false),ghostBundles:bundleIdsBy(true),deepHighlight:[...deepHighlightIds],
    surface,deepVisible,view,camera:camera.position.toArray(),target:controls.target.toArray(),up:camera.up.toArray()};}
  function restore(state){
    cameraTween=null;
    select(state.selected.hemi,state.selected.id);highlight(state.highlighted||[]);
    setHemisphere(state.hemisphere);setBundles(state.bundles,{ghost:state.ghostBundles||[]});
    setDeep(state.deepVisible);setDeepHighlight(state.deepHighlight||[]);
    surface=state.surface;for(const h of Object.values(hemis)){h.points.material.uniforms.surface.value=surface;h.shell.material.opacity=.03+surface*.13;}
    camera.position.fromArray(state.camera);controls.target.fromArray(state.target);camera.up.fromArray(state.up);view=state.view;
    autoFrame=false;controls.update();requestDraw();
  }
  return {surfaceMeta,tractMeta,subMeta:sub,manifest,select,highlight,setHemisphere,setDeep,setDeepHighlight,
    setBundles,setView,flyTo,snapshot,restore,
    setSurface(value){surface=Math.max(.08,Math.min(.95,value));for(const h of Object.values(hemis)){
      h.points.material.uniforms.surface.value=surface;h.shell.material.opacity=.03+surface*.13;}requestDraw();},
    setProfile(value){profile=value==='presenter'?'presenter':'teaching';for(const t of traces)t.visible=profile==='teaching';requestDraw();},
    setPlaying(value){playing=!!value;last=0;requestDraw();},
    get state(){return {ready:true,profile,playing:playing&&profile==='teaching'&&!reduced.matches,
      reducedMotion:reduced.matches,time,frames,selected,highlighted:highlighted.map(r=>({...r})),hemisphere:visibleHemi,
      bundles:bundleIdsBy(false),ghostBundles:bundleIdsBy(true),
      bundleAlpha:Object.fromEntries([...bundles].map(([id,v])=>[id,v.alpha])),
      vertices:Object.values(hemis).map(h=>h.count),view,deepVisible,deepHighlight:[...deepHighlightIds],
      camera:camera.position.toArray()};},
    dispose(){draco.dispose();scene.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});disposeBase();},
  };
}
