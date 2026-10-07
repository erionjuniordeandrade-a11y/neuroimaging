// NiiVue 2D scene (canvas #gl): radiological convention, layers = base + overlay + visible masks, always in
// each layer's own reading colormap. The 3D view is a second NiiVue instance (render3d.js) placed over its tile.
const CT_PRESETS=[['Cérebro',40,80],['Subdural',75,215],['AVC',35,40],['Partes moles',40,400],['Osso',600,2800],['Osso temporal',700,4000]];
// mask = the render mask (manifest role "render") that the preset applies to the 3D copy.
const PRESETS_3D={
  'ct-bone':{key:'preset3dBone',kind:'CT',mask:'head',min:150,max:1650,colormap:'cc-bone',smooth:true},
  'ct-skin':{key:'preset3dSkin',kind:'CT',mask:'head',min:-500,max:1500,colormap:'cc-skinbone',smooth:true,shellMm:6,boneHU:200},
  'mr-brain':{key:'preset3dBrain',kind:'MR',mask:'brain',auto:true,colormap:'cc-brain',smooth:true,baseSigma:.6},
  'surfaces':{key:'preset3dSurfaces',kind:'any',surfaceOnly:true,auto:true,colormap:'gray'},
  'mr-skin':{key:'preset3dSkinMR',kind:'MR',mask:'head',auto:true,colormap:'cc-skin',smooth:true,baseSigma:.6},
  // Brain plus the bright voxels of a thin shell outside the brain mask: on contrast-enhanced T1 these are
  // superficial veins and enhancing dura, coloured apart from cortex. Scalp and skull are never in the shell.
  'mr-vessels':{key:'preset3dVessels',kind:'MR',mask:'brain',vessels:true,requiresVessels:true,vesselColormap:'cc-brain',marginMm:3,auto:true,colormap:'cc-vessels',smooth:true,baseSigma:.6},
  // Angio-CT: pack-time subtraction vessels as a surface, seen through the skull (meshXray); offered only
  // when the capsule carries that mask, since a single CT cannot separate contrast from bone.
  'ct-vessels':{key:'preset3dCTA',kind:'CT',mask:'head',vessels:true,requiresVessels:true,meshXray:.75,min:150,max:1650,colormap:'cc-bone',smooth:true}
};
const LAYOUTS=['2x2','3d','axial','coronal','sagittal'];
const DEFAULT_CAMERA={azimuth:110,elevation:15,zoom:1};
function isCT(meta){return meta.kind==='CT'}
function autoWindow(meta){const s=meta.stats||{},lo=s.p01??s.min??0,hi=s.p99??s.max??1;return {center:(lo+hi)/2,width:Math.max(1e-3,hi-lo)}}
function windowPresets(meta){
  let presets=(meta.window_presets||[]).map(p=>({name:p.name,center:p.center,width:p.width}));
  if(isCT(meta)){const names=new Set(presets.map(p=>p.name));presets=[...presets,...CT_PRESETS.filter(p=>!names.has(p[0])).map(([name,center,width])=>({name,center,width}))]}
  else{presets=[{name:tr('auto'),...autoWindow(meta)},...presets.filter(p=>p.name!=='Auto')]}
  return presets;
}
function defaultWindow(meta){if(!isCT(meta))return autoWindow(meta);const p=windowPresets(meta);return {...(p.find(x=>x.name==='Partes moles')||p[0])}}
function presets3dFor(meta){
  const cropped=!!state.manifest?.grid?.crop;
  return Object.entries(PRESETS_3D).filter(([key,p])=>{
    if(p.kind!=='any'&&p.kind!==(isCT(meta)?'CT':'MR'))return false;
    if(key==='mr-brain'&&state.manifest?.brain_mask_qc?.verdict==='FAIL')return false;
    if(key==='surfaces'&&!cropped)return false;
    if(cropped&&(key==='mr-skin'||key==='ct-skin'))return false;
    const vessels=p.vessels?renderMaskFor(meta.id,'vessels'):null;
    if(p.requiresVessels&&!vessels)return false;
    if(p.vessels&&state.mode==='patient'&&!isValidReview(vessels))return false;
    if(meta.skull_stripped_input===true&&key==='mr-skin')return false;
    return true;
  });
}
function default3d(meta){
  const offered=presets3dFor(meta).map(([key])=>key);
  if(state.manifest?.grid?.crop&&offered.includes('surfaces'))return 'surfaces';
  const preferred=isCT(meta)?(offered.includes('ct-vessels')?'ct-vessels':'ct-bone'):
    (renderMaskFor(meta.id,'brain')?'mr-brain':'mr-skin');
  return offered.includes(preferred)?preferred:offered[0]||null;
}
async function initNiivue(){
  if(!window.niivue)throw Error('NiiVue ausente');
  const probe=document.createElement('canvas').getContext('webgl2');if(!probe)throw Error('WebGL2 unavailable');
  const nv=new niivue.Niivue({
    backColor:[0,0,0,1],crosshairColor:[.79,.66,.3,1],crosshairWidth:1,show3Dcrosshair:true,
    isRadiologicalConvention:true,multiplanarShowRender:niivue.SHOW_RENDER.ALWAYS,multiplanarLayout:2,
    dragAndDropEnabled:false,isColorbar:false,loadingText:'',logLevel:'error',meshThicknessOn2D:state.tractSlabMm,
    fontColor:[.86,.84,.78,1],measureLineColor:[.79,.66,.3,1],measureTextColor:[.95,.9,.75,1],
    viewModeHotKey:'',clipPlaneHotKey:'',cycleClipPlaneHotKey:'',isOrientationTextVisible:true,
    clipPlaneColor:[0,0,0,0],isAlphaClipDark:true
  });
  await nv.attachToCanvas($('gl'));
  state.nv=nv;
  nv.onLocationChange=data=>{if(!data?.mm)return;state.crosshair=[data.mm[0],data.mm[1],data.mm[2]];syncCrosshair3d();renderReadout()};
  return nv;
}
function volumeImage(id){return state.volumes.get(id)?.nvimg}
function applyLayers(){
  const nv=state.nv,base=state.volumes.get(state.base);if(!nv||!base)return;
  const list=[],bi=base.nvimg;
  // Colormap first: NiiVue's colormap setter recomputes cal_min/cal_max from the header, so the range must be set after it.
  const w=state.windows[state.base];bi.colormap='gray';bi.cal_min=w.center-w.width/2;bi.cal_max=w.center+w.width/2;
  bi.opacity=1;list.push(bi);
  if(state.overlay&&state.overlay!==state.base){
    const o=state.volumes.get(state.overlay),w=state.windows[state.overlay];
    o.nvimg.colormap=state.overlayColormap;o.nvimg.cal_min=w.center-w.width/2;o.nvimg.cal_max=w.center+w.width/2;o.nvimg.opacity=state.overlayOpacity;list.push(o.nvimg);
  }
  for(const m of visibleMasks()){if(m.id!==state.editingMask){const l=state.masks.get(m.id).nvimg;list.push(l.fill,l.rim)}}
  if(typeof anatomyLayers==='function')list.push(...anatomyLayers());
  nv.volumes=list;nv.back=list[0];nv.overlays=list.slice(1);
  nv.updateGLVolume();
  if(typeof applyMeshes3d==='function')applyMeshes3d();
  if(typeof renderMaskLegend==='function')renderMaskLegend();
}
// Tiles as canvas fractions [left, top, width, height]; the 3D instance covers the rect that has no tile.
const TILE_LAYOUTS={
  '2x2':{tiles:[['axial',[0,0,.5,.5]],['coronal',[.5,0,.5,.5]],['sagittal',[0,.5,.5,.5]]],render:[.5,.5,.5,.5]},
  '3d':{tiles:[['axial',[.72,0,.28,1/3]],['coronal',[.72,1/3,.28,1/3]],['sagittal',[.72,2/3,.28,1/3]]],render:[0,0,.72,1]},
  '3d-patient':{tiles:[],render:[0,0,1,1]}
};
// Every 2D tile shows its whole slice at ONE shared mm scale with a margin (NiiVue's own multiplanar does
// this; a fractional custom layout fits each slice to its cell, so the same head was drawn up to 36% larger
// in the coronal/sagittal cells and touched the cell edges). Cells are recomputed whenever the canvas resizes.
const TILE_MARGIN=.04;
function sliceExtentsMm(){
  const b=state.nv?.back;
  if(b?.dimsRAS&&b?.pixDimsRAS)return [1,2,3].map(a=>b.dimsRAS[a]*b.pixDimsRAS[a]);
  const g=state.manifest.grid;return g.dims.map((d,i)=>d*(g.spacing_mm?.[i]??1));
}
function fitTiles(tiles){
  const c=$('gl'),W=c.clientWidth||c.width,H=c.clientHeight||c.height,e=sliceExtentsMm();
  const fov={axial:[e[0],e[1]],coronal:[e[0],e[2]],sagittal:[e[1],e[2]]};
  let s=Infinity;for(const [t,[,,w,h]] of tiles)s=Math.min(s,w*W/fov[t][0],h*H/fov[t][1]);
  s*=1-2*TILE_MARGIN;
  return tiles.map(([t,[l,top,w,h]])=>{const fw=fov[t][0]*s/W,fh=fov[t][1]*s/H;return [t,[l+(w-fw)/2,top+(h-fh)/2,fw,fh]]});
}
function applyTileLayout(){
  const nv=state.nv,tl=state.tileLayout;if(!nv||!tl?.tiles.length)return;
  const types={axial:nv.sliceTypeAxial,coronal:nv.sliceTypeCoronal,sagittal:nv.sliceTypeSagittal};
  nv.setCustomLayout(fitTiles(tl.tiles).map(([t,position])=>({sliceType:types[t],position})));
}
function watchTileResize(){
  if(typeof ResizeObserver!=='function')return;
  new ResizeObserver(()=>{if(state.tileLayout?.tiles.length){applyTileLayout();state.nv.drawScene()}}).observe($('gl-wrap'));
}
function placeRender(rect){
  const wrap=$('gl3d-wrap');
  if(!rect||!state.preset3d){wrap.style.visibility='hidden';Object.assign($('mask-legend').style,{left:'12px',top:'12px'});return}
  wrap.style.visibility='';Object.assign(wrap.style,{left:rect[0]*100+'%',top:rect[1]*100+'%',width:rect[2]*100+'%',height:rect[3]*100+'%'});
  Object.assign($('mask-legend').style,{left:`calc(${rect[0]*100}% + 12px)`,top:`calc(${rect[1]*100}% + 12px)`});
  requestAnimationFrame(()=>{state.nv3d?.resizeListener?.();state.nv3d?.drawScene()});
}
function setLayout(layout){
  if(layout==='three')layout='3d';
  if(!LAYOUTS.includes(layout))throw Error('Layout inválido: '+layout);
  state.layout=layout;const nv=state.nv,o=nv.opts;o.heroImageFraction=0;
  const tl=TILE_LAYOUTS[layout==='3d'&&state.mode==='patient'?'3d-patient':layout],types={axial:nv.sliceTypeAxial,coronal:nv.sliceTypeCoronal,sagittal:nv.sliceTypeSagittal};
  $('gl-wrap').style.visibility=tl&&!tl.tiles.length?'hidden':'';
  state.tileLayout=tl||null;
  if(tl){if(tl.tiles.length)applyTileLayout();else nv.clearCustomLayout()}
  else{nv.clearCustomLayout();nv.setSliceType(types[layout])}
  placeRender(tl?tl.render:null);
  applyLayers();
  for(const b of document.querySelectorAll('[data-layout]'))b.classList.toggle('active',b.dataset.layout===layout);
  if(typeof renderWindowPanel==='function')renderWindowPanel();
  return layout;
}
function setBase(id){
  if(!state.volumes.has(id))throw Error('Volume não encontrado');
  if(id!==state.base)state.window3d=null;state.base=id;if(state.overlay===id)state.overlay=null;
  const meta=state.volumes.get(id).meta;if(!presets3dFor(meta).some(([key])=>key===state.preset3d))state.preset3d=default3d(meta);
  applyLayers();updateRender();renderReadout();if(typeof renderFusionControls==='function')renderFusionControls();return id;
}
function setOverlay(id,options={}){
  if(id&&!state.volumes.has(id))throw Error('Volume não encontrado');
  state.overlay=id&&id!==state.base?id:null;
  if(options.colormap){if(!['hot','warm','cool','viridis'].includes(options.colormap))throw Error('Mapa inválido');state.overlayColormap=options.colormap}
  if(options.opacity!==undefined)state.overlayOpacity=clamp(Number(options.opacity),0,1);
  applyLayers();renderReadout();if(typeof renderFusionControls==='function')renderFusionControls();return state.overlay;
}
function setWindow(volumeId,center,width){
  if(!state.volumes.has(volumeId))throw Error('Volume não encontrado');
  state.windows[volumeId]={center:Number(center),width:Math.max(1e-3,Number(width))};applyLayers();return state.windows[volumeId];
}
function setPreset3d(key){
  const base=state.volumes.get(state.base).meta,p=PRESETS_3D[key];
  if(!p)throw Error('Predefinição 3D inválida');
  if(!presets3dFor(base).some(([available])=>available===key))throw Error('Predefinição indisponível para este volume');
  state.preset3d=key;state.window3d=null;const sel=$('preset3d');if(sel)sel.value=key;return updateRender().then(()=>key);
}
async function setIllumination(amount){state.illumination=clamp(Number(amount),0,1);await state.nv3d.setVolumeRenderIllumination(state.illumination);state.nv3d.drawScene()}
function setCrosshairRAS(x,y,z){
  if(!state.manifest?.volumes?.length)return;
  const p=rasToVoxel(x,y,z),d=state.manifest.grid.dims;
  state.crosshair=affinePoint(...p.map((v,i)=>clamp(v,0,d[i]-1)));
  syncCrosshair();renderReadout();return state.crosshair;
}
function syncCrosshair(){
  const nv=state.nv;if(!nv?.back)return;
  const f=nv.mm2frac(state.crosshair);nv.scene.crosshairPos=[clamp(f[0],0,1),clamp(f[1],0,1),clamp(f[2],0,1)];
  nv.drawScene();syncCrosshair3d();
}
// Clip plane through the crosshair. NiiVue's plane lives in RAS texture space [0,1]^3 with
// normal n = sph2cart(azimuth+180, elevation) and depth d where dot(n, p-0.5) + d = 0.
const CLIP_ANGLES={axial:[0,90],coronal:[0,0],sagittal:[90,0]};
// Cut away the half facing the camera (checked visually: NiiVue removes the side opposite the normal;
// camera at azimuth a sits at RAS (-sin a, -cos a), elevation > 0 is above).
function clipAngles(axis){
  const s=state.nv3d.scene,a=s.renderAzimuth*Math.PI/180;
  if(axis==='sagittal')return -Math.sin(a)<0?[270,0]:[90,0];
  if(axis==='coronal')return -Math.cos(a)>0?[180,0]:[0,0];
  return s.renderElevation>=0?[0,90]:[0,-90];
}
function clipNormal(az,el){const A=-el*Math.PI/180,g=((az+180-90)%360)*Math.PI/180;return [Math.cos(A)*Math.cos(g),Math.cos(A)*Math.sin(g),Math.sin(A)]}
function applyClip(draw=true){
  const nv=state.nv3d;if(!nv)return;
  if(state.trajClip&&typeof trajClipPlane==='function'){const p=trajClipPlane();if(p){nv.scene.clipPlaneDepthAziElevs=[p];nv.setClipPlane(p);if(draw)nv.drawScene();return}}
  if(!state.clip){nv.setClipPlane([2,0,0]);return}
  const [az,el]=clipAngles(state.clip),n=clipNormal(az,el),f=nv.scene.crosshairPos;
  const depth=-(n[0]*(f[0]-.5)+n[1]*(f[1]-.5)+n[2]*(f[2]-.5));
  nv.scene.clipPlaneDepthAziElevs=[[depth,az,el]];
  nv.setClipPlane([depth,az,el]);if(draw)nv.drawScene();
}
function setClip(axis){
  if(axis&&!CLIP_ANGLES[axis])throw Error('Eixo inválido');state.clip=axis||null;if(state.clip)state.trajClip=false;applyClip();if(typeof renderTrajectoryPanel==='function')renderTrajectoryPanel();
  for(const b of document.querySelectorAll('[data-clip]'))b.classList.toggle('active',(b.dataset.clip||null)===state.clip);return state.clip;
}
function getCamera(){const s=state.nv3d.scene;return {azimuth:Number(s.renderAzimuth.toFixed(2)),elevation:Number(s.renderElevation.toFixed(2)),zoom:Number(s.volScaleMultiplier.toFixed(3))}}
function setCamera(c={}){const nv=state.nv3d,cur=getCamera();if(state.surgeonView){state.surgeonView=false;nv.position=null;if(typeof renderTrajectoryPanel==='function')renderTrajectoryPanel()}nv.setScale(c.zoom??cur.zoom);nv.setRenderAzimuthElevation(c.azimuth??cur.azimuth,c.elevation??cur.elevation);if(state.clip||state.trajClip)applyClip();return getCamera()}
function resetView(){setCamera(DEFAULT_CAMERA);state.nv.scene.pan2Dxyzmm=[0,0,0,1];state.nv.drawScene();state.nv3d.drawScene()}
function fmtValue(meta,v){return v===null?'—':isCT(meta)?String(Math.round(v)):Math.abs(v)>=100?v.toFixed(0):v.toFixed(1)}
function units(meta){return meta.units||(isCT(meta)?'HU':'a.u.')}
function resolutionReadout(meta){
  if(meta.resampled!==true||!Array.isArray(meta.source_spacing_mm)||!state.manifest.grid.spacing_mm?.length)return '';
  const format=value=>new Intl.NumberFormat('pt-BR',{maximumFractionDigits:2}).format(Number(value));
  const source=meta.source_spacing_mm.map(format).join('×'),grid=format(state.manifest.grid.spacing_mm[0]);
  return tr('resampledReadout').replace('{source}',source).replace('{grid}',grid);
}
function renderReadout(){
  const el=$('readout');if(!el||!state.manifest)return;
  if(state.mode==='patient'){el.textContent='';el.title='';return}
  const [x,y,z]=state.crosshair,parts=[`${tr('readoutCursor')} RAS ${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)} mm`];
  for(const id of visibleVolumeIds()){
    const meta=state.volumes.get(id).meta,resolution=resolutionReadout(meta);
    parts.push(`${meta.label}: ${fmtValue(meta,valueAtRAS(id,x,y,z))} ${units(meta)}${resolution?` (${resolution})`:''}`);
  }
  if(typeof anatomyAtRAS==='function'){const a=anatomyAtRAS(x,y,z);if(a.length)parts.push(`${tr('anatomy')}: ${a.map(q=>q.name).join(', ')}`)}
  el.textContent=parts.join('   ·   ');el.title=el.textContent;
}
function drawNow(){state.nv?.drawScene();state.nv3d?.drawScene()}
