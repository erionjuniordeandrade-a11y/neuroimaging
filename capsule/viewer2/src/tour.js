// Surgeon-authored pt-BR tour. A step stores layout, crosshair, visible volumes/masks/tracts,
// the 3D camera (azimuth/elevation/zoom) and the clip plane. Patient mode shows reviewed content only.
function captureView(){
  const w=state.windows[state.base];
  return {layout:state.layout,crosshair_ras:state.crosshair.map(v=>Number(v.toFixed(3))),
    window:{volume:state.base,center:w.center,width:w.width},
    overlay:state.overlay?{volume:state.overlay,colormap:state.overlayColormap,opacity:state.overlayOpacity,center:state.windows[state.overlay].center,width:state.windows[state.overlay].width}:null,
    preset3d:state.preset3d,window3d:state.window3d,camera:getCamera(),clip:state.clip,
    visible_volumes:visibleVolumeIds(),visible_masks:visibleMasks().map(m=>m.id),visible_tracts:visibleTracts().map(t=>t.id),
    visible_annotations:[]};
}
function addTourStep(title,text){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const step={id:'step-'+crypto.randomUUID(),title:String(title||tr('step')),text:String(text||''),view:captureView()};
  state.manifest.tour.push(step);renderStepList();return step;
}
function applyTourStep(index){
  const tour=state.manifest.tour;
  if(!tour.length){state.tourIndex=0;renderTourPanel();return null}
  state.tourIndex=clamp(index,0,tour.length-1);const step=tour[state.tourIndex],view=step.view||{};
  const vols=(view.visible_volumes||[]).filter(id=>state.volumes.has(id));
  const baseId=view.window?.volume&&state.volumes.has(view.window.volume)?view.window.volume:vols[0]||state.base;
  state.base=baseId;
  if(view.window&&view.window.volume===baseId)state.windows[baseId]={center:view.window.center,width:view.window.width};
  const ov=view.overlay&&state.volumes.has(view.overlay.volume)?view.overlay:null;
  state.overlay=ov?ov.volume:(vols.find(id=>id!==baseId)||null);
  if(ov){state.overlayColormap=ov.colormap||state.overlayColormap;state.overlayOpacity=ov.opacity??state.overlayOpacity;if(ov.center!==undefined)state.windows[ov.volume]={center:ov.center,width:ov.width}}
  const baseMeta=state.volumes.get(baseId).meta;
  state.preset3d=view.preset3d&&presets3dFor(baseMeta).some(([key])=>key===view.preset3d)?view.preset3d:default3d(baseMeta);
  state.window3d=state.preset3d===view.preset3d&&view.window3d?{min:Number(view.window3d.min),max:Number(view.window3d.max)}:null;
  const filter=state.mode==='patient';state.tourMaskFilter=filter?view.visible_masks??null:null;state.tourTractFilter=filter?view.visible_tracts??null:null;
  updateRender();setLayout(view.layout||'3d');
  if(view.crosshair_ras)setCrosshairRAS(...view.crosshair_ras);
  if(view.camera)setCamera({azimuth:view.camera.azimuth,elevation:view.camera.elevation,zoom:view.camera.zoom});
  applyTracts();setClip(view.clip||null);
  renderTourPanel();renderFusionControls();return step;
}
async function setMode(mode){
  if(!['patient','surgeon'].includes(mode))throw Error('Modo inválido');
  if(mode==='patient'){if(state.tool!=='crosshair')await setTool('crosshair');if(state.editingMask)editMask(null)}
  state.mode=mode;document.body.classList.toggle('patient',mode==='patient');
  const baseMeta=state.volumes.get(state.base)?.meta;
  if(baseMeta&&!presets3dFor(baseMeta).some(([key])=>key===state.preset3d)){state.preset3d=default3d(baseMeta);state.window3d=null;updateRender()}
  $('mode-button').textContent=mode==='patient'?tr('surgeon'):tr('patient');
  hideMeasurements(mode==='patient');
  if(mode==='patient'){
    if(state.manifest.tour.length)applyTourStep(state.tourIndex);
    else{state.tourMaskFilter=null;state.tourTractFilter=null;setLayout('3d');applyTracts();renderTourPanel()}
  }else{state.tourMaskFilter=null;state.tourTractFilter=null;setLayout(state.layout);applyTracts()}
  renderReadout();renderMaskList();renderTractList();renderAnatomyList();renderTrajectoryPanel();renderFusionControls();
  requestAnimationFrame(()=>{state.nv.resizeListener?.();state.nv3d.resizeListener?.()});
  return mode;
}
function renderTourPanel(){
  const tour=state.manifest.tour,step=tour[state.tourIndex];
  $('step-count').textContent=step?`${tr('step')} ${state.tourIndex+1} / ${tour.length}`:'';
  $('step-title').textContent=step?step.title:'';$('step-text').textContent=step?step.text:tr('noTour');
  $('prev-step').disabled=!step||state.tourIndex===0;$('next-step').disabled=!step||state.tourIndex>=tour.length-1;
  $('prev-step').hidden=$('next-step').hidden=!step;
}
function renderStepList(){
  const list=$('step-list');list.replaceChildren();
  if(!state.manifest.tour.length){const p=document.createElement('p');p.className='empty-note';p.textContent=tr('noSteps');list.append(p);return}
  for(const [i,step] of state.manifest.tour.entries()){
    const div=document.createElement('div');div.className='step-card';const row=document.createElement('div');row.className='row';
    const title=document.createElement('span');title.textContent=`${i+1}. ${step.title}`;title.style.cursor='pointer';title.onclick=()=>{applyTourStep(i)};row.append(title);
    for(const [text,delta] of [['↑',-1],['↓',1]]){const b=document.createElement('button');b.type='button';b.textContent=text;b.disabled=i+delta<0||i+delta>=state.manifest.tour.length;b.onclick=()=>{const t=state.manifest.tour;[t[i],t[i+delta]]=[t[i+delta],t[i]];renderStepList()};row.append(b)}
    const del=document.createElement('button');del.type='button';del.textContent=tr('delete');del.onclick=()=>{state.manifest.tour.splice(i,1);renderStepList()};row.append(del);
    div.append(row);list.append(div);
  }
}
