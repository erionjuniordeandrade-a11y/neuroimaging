// Loading sequence, panels and event wiring.
function option(select,value,label){const el=document.createElement('option');el.value=value;el.textContent=label;select.append(el);return el}
function renderFusionControls(){
  const base=$('base-volume'),over=$('overlay-volume');base.replaceChildren();over.replaceChildren();
  option(over,'',tr('none'));
  for(const meta of state.manifest.volumes){option(base,meta.id,`${meta.label} (${meta.kind})`);if(meta.id!==state.base)option(over,meta.id,`${meta.label} (${meta.kind})`)}
  base.value=state.base;over.value=state.overlay||'';over.disabled=state.manifest.volumes.length<2;
  $('overlay-colormap').value=state.overlayColormap;$('overlay-opacity').value=Math.round(state.overlayOpacity*100);$('overlay-opacity-out').textContent=Math.round(state.overlayOpacity*100)+'%';
  renderWindowPanel();renderPreset3d();renderReadout();
}
function windowTarget(){return state.windowLayer==='overlay'&&state.overlay?state.overlay:state.base}
function renderWindowPanel(){
  const layer=$('window-layer');if(!layer||!state.base)return;layer.replaceChildren();option(layer,'base',tr('layerBase'));
  if(state.overlay)option(layer,'overlay',tr('layerOverlay'));if(!state.overlay)state.windowLayer='base';layer.value=state.windowLayer;
  const id=windowTarget(),meta=state.volumes.get(id).meta,w=state.windows[id],presets=windowPresets(meta),sel=$('window-preset');sel.replaceChildren();
  for(const [i,p] of presets.entries())option(sel,String(i),p.name);option(sel,'manual',tr('manual'));sel._presets=presets;
  const match=presets.findIndex(p=>Math.abs(p.center-w.center)<1e-6&&Math.abs(p.width-w.width)<1e-6);sel.value=match>=0?String(match):'manual';
  const step=isCT(meta)?1:Math.max(.01,Number(((meta.stats?.p99??100)/200).toPrecision(1)));
  for(const x of ['level-number','width-number'])$(x).step=step;
  const dp=Math.abs(w.width)>=100?0:2;$("level-number").value=Number(w.center.toFixed(dp));$("width-number").value=Number(w.width.toFixed(dp));
}
function renderPreset3d(){
  const sel=$('preset3d'),meta=state.volumes.get(state.base).meta;sel.replaceChildren();
  const available=presets3dFor(meta);
  if(!available.some(([key])=>key===state.preset3d))state.preset3d=default3d(meta);
  for(const [key,p] of available)option(sel,key,tr(p.key));sel.value=state.preset3d;
  let note=$('brain-mask-qc-note');
  if(!isCT(meta)&&state.manifest.brain_mask_qc?.verdict==='FAIL'){
    if(!note){note=document.createElement('div');note.id='brain-mask-qc-note';note.className='hint';sel.parentElement.after(note)}
    note.textContent=tr('brainMaskQcFail');note.hidden=false;
  }else note?.remove();
  renderVesselReview();
}
function renderVesselReview(){
  const row=$('vessel-review-control');if(!row)return;
  const meta=state.manifest?.masks?.find(m=>m.for_volume===state.base&&m.role==='render'&&
    (m.id==='vessels'||m.id.startsWith('vessels_'))&&state.masks.has(m.id));
  row.hidden=!meta;if(!meta)return;
  const box=$('vessel-review-checkbox'),badge=$('vessel-review-status'),signature=reviewStatus(meta);
  box.checked=isValidReview(meta);box.disabled=!state.reviewer.trim();
  badge.textContent=signature==='signed'?'':tr(signature==='unsigned'?'reviewedUnsigned':'notReviewed');
  box.onchange=async()=>{try{await setMaskReviewed(meta.id,box.checked)}catch(e){console.warn(e.message)}renderPreset3d()};
}
function wire(){
  for(const b of document.querySelectorAll('[data-layout]'))b.onclick=()=>setLayout(b.dataset.layout);
  $('mode-button').onclick=()=>setMode(state.mode==='patient'?'surgeon':'patient');
  $('save-button').onclick=async()=>{try{await save()}catch(e){console.warn(e.message)}};
  $('shot-button').onclick=()=>screenshot();
  $('base-volume').onchange=e=>setBase(e.target.value);
  $('overlay-volume').onchange=e=>setOverlay(e.target.value||null);
  $('overlay-colormap').onchange=e=>setOverlay(state.overlay,{colormap:e.target.value});
  $('overlay-opacity').oninput=e=>{state.overlayOpacity=Number(e.target.value)/100;$('overlay-opacity-out').textContent=e.target.value+'%';applyLayers()};
  $('window-layer').onchange=e=>{state.windowLayer=e.target.value;renderWindowPanel()};
  $('window-preset').onchange=e=>{const p=e.target._presets[Number(e.target.value)];if(p){setWindow(windowTarget(),p.center,p.width);renderWindowPanel()}};
  for(const id of ['level-number','width-number'])$(id).onchange=()=>{setWindow(windowTarget(),$('level-number').value,$('width-number').value);renderWindowPanel()};
  $('preset3d').onchange=e=>setPreset3d(e.target.value);
  $('reviewer-name').value=state.reviewer;
  $('reviewer-name').oninput=e=>{
    state.reviewer=e.target.value;const disabled=!state.reviewer.trim();
    for(const box of document.querySelectorAll('.reviewed input[type="checkbox"]'))box.disabled=disabled;
    renderTractList();renderMaskList();renderAnatomyList();renderPreset3d();
  };
  $('illumination').oninput=e=>setIllumination(Number(e.target.value)/100);
  for(const id of ['window3d-min','window3d-max'])$(id).oninput=()=>setWindow3dFromSliders();
  for(const b of document.querySelectorAll('[data-clip]'))b.onclick=()=>setClip(b.dataset.clip||null);
  $('reset-view').onclick=()=>resetView();
  $('tract-slab').onchange=e=>setTractSlab(e.target.value);
  $('tract-density-toggle').onclick=()=>setTractDensityMode(state.tractDensityMode==='proportional'?'equalized':'proportional');
  $('tract-xray').oninput=e=>setTractXray(Number(e.target.value)/100);
  $('show-outliers').onchange=e=>setShowOutliers(e.target.checked);
  $('new-mask').onclick=async()=>{const meta=await createMask();editMask(meta.id)};
  for(const b of document.querySelectorAll('[data-tool]'))b.onclick=()=>setTool(b.dataset.tool);
  $('pen-size').onchange=e=>{state.nv.opts.penSize=Number(e.target.value)||1};
  $('undo-draw').onclick=()=>{state.nv.drawUndo();syncDrawing();applyLayers()};
  $('add-step').onclick=()=>{addTourStep($('new-step-title').value.trim()||`${tr('step')} ${state.manifest.tour.length+1}`,$('new-step-text').value.trim());$('new-step-title').value='';$('new-step-text').value=''};
  $('prev-step').onclick=()=>applyTourStep(state.tourIndex-1);$('next-step').onclick=()=>applyTourStep(state.tourIndex+1);
  const nv=state.nv;
  nv.onMeasurementCompleted=onMeasurement;nv.onAngleCompleted=onAngle;
  nv.onDrawingChanged=()=>{if(state.editingMask)syncDrawing()};
}
// NiiVue keeps its own copy of the voxels (nvimg.img). When it is the same type, length and voxel order as the
// decoded blob, the volume reads that one buffer (nothing writes volume voxels) and the duplicate is released.
function shareVoxels(data,nvimg){
  const img=nvimg?.img;if(!img||img.constructor!==data.constructor||img.length!==data.length)return data;
  for(let n=0;n<data.length;n++)if(img[n]!==data[n])return data;
  return img;
}
async function loadVolumes(){
  const m=state.manifest,[nx,ny,nz]=m.grid.dims,total=nx*ny*nz;
  for(const [i,meta] of m.volumes.entries()){
    status('loading',tr('loading'),`${tr('loadingVolume')} ${i+1}/${m.volumes.length}`);await new Promise(r=>setTimeout(r,0));
    const Typed=NIFTI_TYPES[meta.dtype]?.[2];if(!Typed||meta.dtype==='uint8')throw Error('volume dtype');
    const bytes=await blobBytes(meta.blob);if(bytes.length!==total*Typed.BYTES_PER_ELEMENT)throw Error('volume length');
    let data=new Typed(bytes.buffer,bytes.byteOffset,total),slope=Number(meta.slope??1),inter=Number(meta.intercept??0);
    const w=defaultWindow(meta);state.windows[meta.id]={center:w.center,width:w.width};
    const nvimg=await niivue.NVImage.new(niftiBytes(m.grid,meta.dtype,data,{slope,inter,calMin:w.center-w.width/2,calMax:w.center+w.width/2}),`${meta.id}.nii`,'gray',1,null,w.center-w.width/2,w.center+w.width/2);
    data=shareVoxels(data,nvimg);
    state.volumes.set(meta.id,{meta:{...meta,slope,intercept:inter},data,nvimg});
  }
}
async function start(){
  localizeStatic();status('loading',tr('loading'),'');
  const watchdog=setTimeout(()=>{if(state.status==='loading')status('loading',tr('loading'),tr('timeout'))},60000);
  try{
    const node=$('capsule-manifest');if(!node)throw Error('manifest missing');
    const manifest=JSON.parse(node.textContent);if(manifest.schema!=='case-capsule/1')throw Error('schema');
    state.manifest=manifest;manifest.masks??=[];manifest.annotations??=[];manifest.tour??=[];manifest.seg_prompts??=[];
    state.reviewer=typeof manifest.case?.reviewer==='string'?manifest.case.reviewer:'';
    const deid=manifest.case?.deidentification,anonymous=$('anonymous');
    const faceRemoved=deid?.face_removed===true;
    anonymous.textContent=tr(faceRemoved?'anonymized':'pseudonymized');
    if(faceRemoved)anonymous.removeAttribute('title');else anonymous.title=tr('pseudonymizedTitle');
    $('case-label').textContent=$('case-label').title=manifest.case?.label||'Caso';setVersionBadge(manifest.version??1);
    if(!manifest.volumes?.length){status('empty',tr('empty'));clearTimeout(watchdog);readyResolve(state);return}
    const [nx,ny,nz]=manifest.grid.dims,total=nx*ny*nz;if(!Number.isSafeInteger(total)||total<=0||total>150000000)throw Error('grid');
    const a=manifest.grid.affine_ras;inverseAffine=inverse3([a[0][0],a[0][1],a[0][2],a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]]);
    await initNiivue();await initNiivue3d();watchTileResize();initTrajectory();initSegPrompts();
    await loadVolumes();await loadMasks();await loadAnatomy();
    state.base=manifest.volumes[0].id;state.preset3d=default3d(state.volumes.get(state.base).meta);
    state.crosshair=foregroundCenterRAS(state.volumes.get(state.base));
    await loadTracts();
    state.tractXray=(manifest.tracts||[]).length||lesionMasks().length?.6:0;
    wire();wireTrajectory();wireSegPrompts();applyLayers();
    status('loading',tr('loading'),tr('preparing3d'));await new Promise(r=>setTimeout(r,0));
    await updateRender();applyTracts();
    await state.nv3d.setVolumeRenderIllumination(state.illumination);
    setLayout('2x2');syncCrosshair();
    renderFusionControls();renderTractList();setTractXray(state.tractXray);renderMaskList();renderAnatomyList();renderAnnotationList();renderTrajectoryPanel();renderSegPrompts();renderStepList();renderTourPanel();await setTool('crosshair');
    $('mode-button').textContent=tr('patient');
    clearTimeout(watchdog);status('ready','');state.nv.resizeListener?.();state.nv3d.resizeListener?.();drawNow();readyResolve(state);
  }catch(err){
    clearTimeout(watchdog);console.warn(err);
    status('error',err.message==='WebGL2 unavailable'?tr('gpu'):tr('bad'),err.message);readyReject(err);
  }
}
