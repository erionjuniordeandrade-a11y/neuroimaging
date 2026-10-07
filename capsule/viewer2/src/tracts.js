// Tractography: each manifest tract is an MRtrix .tck blob loaded as an NVMesh (world RAS mm).
const TRACT_TUBE_MM=.45;
// Tubes multiply the triangle count ~12x: a software rasteriser (SwiftShader, llvmpipe) stalls on them, so it keeps lines.
function softwareGL(gl){try{const ext=gl.getExtension('WEBGL_debug_renderer_info');
  return /swiftshader|llvmpipe|software/i.test(ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER))}catch{return false}}
function tractStreamlineBounds(streamlines){
  const bounds=new Float32Array(streamlines.length*6);
  for(let s=0;s<streamlines.length;s++){
    const line=streamlines[s],off=s*6;let x0=Infinity,y0=Infinity,z0=Infinity,x1=-Infinity,y1=-Infinity,z1=-Infinity;
    for(let i=0;i+2<line.length;i+=3){const x=line[i],y=line[i+1],z=line[i+2];
      if(x<x0)x0=x;if(y<y0)y0=y;if(z<z0)z0=z;if(x>x1)x1=x;if(y>y1)y1=y;if(z>z1)z1=z}
    bounds[off]=x0;bounds[off+1]=y0;bounds[off+2]=z0;bounds[off+3]=x1;bounds[off+4]=y1;bounds[off+5]=z1;
  }
  return bounds;
}
const XTRACT_PTBR={
  cst:'Trato corticoespinal (CST)',af:'Fascículo arqueado (AF)',slf1:'Fascículo longitudinal superior I (SLF I)',
  slf2:'Fascículo longitudinal superior II (SLF II)',slf3:'Fascículo longitudinal superior III (SLF III)',
  ifo:'Fascículo fronto-occipital inferior (IFOF)',ilf:'Fascículo longitudinal inferior (ILF)',
  uf:'Fascículo uncinado (UF)',or:'Radiação óptica (OR)',fa:'Aslant frontal (FAT)',fx:'Fórnix (FX)',
  ar:'Radiação acústica (AR)',atr:'Radiação talâmica anterior (ATR)',str:'Radiação talâmica superior (STR)',
  mdlf:'Fascículo longitudinal médio (MdLF)',vof:'Fascículo occipital vertical (VOF)',
  fma:'Forceps major (FMA)',fmi:'Forceps minor (FMI)',cbd:'Cíngulo dorsal (CBD)',
  cbp:'Cíngulo parahipocampal (CBP)',cbt:'Cíngulo temporal (CBT)'
};
// Capsules carry opaque ids (t01…); the XTRACT bundle key (fa_l) lives in the label.
function tractKey(meta){const label=typeof meta?.label==='string'?meta.label.trim():'';return label||String(meta?.id||'')}
function tractDisplayName(meta){
  const key=tractKey(meta),side=/^(.*)_(l|r)$/.exec(key),name=XTRACT_PTBR[side?side[1]:key];
  if(!name)return key;
  return name+(side?(side[2]==='l'?' E':' D'):'');
}
function sourceStreamlineCount(meta){const n=meta?.n_streamlines_source;return Number.isSafeInteger(n)&&n>=0?n:null}
function evenlySpacedIndices(n,k){
  const indices=new Uint32Array(k);if(k===1){indices[0]=0;return indices}
  for(let i=0;i<k;i++)indices[i]=Math.floor(i*(n-1)/(k-1));return indices;
}
function packTractDisplay(streamlines,indices){
  let coordinates=0;for(const i of indices)coordinates+=streamlines[i].length;
  const pts=new Float32Array(coordinates),offsets=new Uint32Array(indices.length+1);let write=0,points=0;
  for(let i=0;i<indices.length;i++){const line=streamlines[indices[i]];offsets[i]=points;pts.set(line,write);write+=line.length;points+=line.length/3}
  offsets[indices.length]=points;return {pts,offsets};
}
function updateTractMeshDisplay(mesh,gl,geometry){
  mesh.pts=geometry.pts;mesh.offsetPt0=geometry.offsets;mesh.fiberLengths=null;mesh.fiberDensity=null;
  // Keep the full-bundle extents so density changes do not move the camera framing.
  // NiiVue's default 2 mm filter silently drops short selected streamlines; a negative floor keeps all selected lines.
  mesh.fiberLength=-1;mesh.fiberDecimationStride=1;mesh.updateMesh?.(gl);
}
function syncTractDisplay(visible=visibleTracts()){
  const sources=visible.map(sourceStreamlineCount).filter(n=>n!==null),maxSource=sources.length?Math.max(...sources):0;
  for(const meta of visible){
    const entry=state.tracts.get(meta.id);if(!entry)continue;
    const n=entry.streamlines.length,source=sourceStreamlineCount(meta);
    let count=n;
    if(state.tractDensityMode==='proportional'&&source!==null&&maxSource>0){
      count=Math.min(n,Math.max(Math.min(30,n),Math.round(n*source/maxSource)));
    }
    const signature=n+':'+count;
    if(entry.displaySignature!==signature){
      const indices=evenlySpacedIndices(n,count),geometry=packTractDisplay(entry.streamlines,indices);
      if(count>0){updateTractMeshDisplay(entry.mesh,state.nv.gl,geometry);updateTractMeshDisplay(entry.mesh3d,state.nv3d.gl,geometry)}
      entry.displayIndices=indices;entry.displayGeometry=geometry;entry.displayCount=count;entry.displaySignature=signature;
    }
  }
}
function updateTractDensityUi(){
  const equalized=state.tractDensityMode==='equalized',button=$('tract-density-toggle'),warning=$('density-warning');
  if(button){button.setAttribute('aria-pressed',String(equalized));for(const option of button.querySelectorAll('[data-density-mode]'))option.classList.toggle('active',option.dataset.densityMode===(equalized?'equalized':'proportional'))}
  if(warning)warning.hidden=!equalized;
}
async function loadTracts(){
  const tracts=state.manifest.tracts||[];
  for(const [i,meta] of tracts.entries()){
    status('loading',tr('loading'),`${tr('loadingTract')} ${i+1}/${tracts.length}`);
    const bytes=await blobBytes(meta.blob);await registerBlobIntegrity(meta,bytes);
    const info=tckInfo(bytes),streamlines=tckStreamlines(bytes),streamlineBounds=tractStreamlineBounds(streamlines),[r,g,b]=hexToRgb(meta.color);
    // Meshes are bound to one GL context: one copy for the 2D slab view, one for the 3D instance.
    // 2D slabs keep 1 px lines; the 3D copy is drawn as shaded tubes so bundles read as solid fibres.
    const make=async(data,gl,tube,name)=>{const buffer=data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
      // NiiVue's fibre dither re-rolls Math.random on every rebuild; proportional density rebuilds on visibility
      // changes, so dither would make identical views render differently.
      const mesh=await niivue.NVMesh.readMesh(buffer,`${name}.tck`,gl,1,new Uint8Array([r,g,b,255]),true);
      mesh.fiberColor='Fixed';mesh.fiberDither=0;mesh.fiberRadius=tube?TRACT_TUBE_MM:0;mesh.rgba255=new Uint8Array([r,g,b,255]);
      if(tube){mesh.fiberSides=4;mesh.fiberOcclusion=.15;surfaceShader(mesh)}
      mesh.updateMesh?.(gl);return mesh};
    state.tractTubes??=!softwareGL(state.nv3d.gl);
    const entry={meta,mesh:await make(bytes,state.nv.gl,false,meta.id),
      mesh3d:await make(bytes,state.nv3d.gl,state.tractTubes,meta.id),count:info.count,streamlines,streamlineBounds};
    if(meta.outlier_blob){
      const dropped=await blobBytes(meta.outlier_blob);
      if(meta.outlier_blob_sha256&&await sha256Hex(dropped)!==meta.outlier_blob_sha256)throw Error('outlier blob hash');
      entry.outlierMesh=await make(dropped,state.nv.gl,false,`${meta.id}-outlier`);
      entry.outlierMesh3d=await make(dropped,state.nv3d.gl,state.tractTubes,`${meta.id}-outlier`);
    }
    state.tracts.set(meta.id,entry);
  }
}
function applyTracts(){
  const nv=state.nv;if(!nv)return;
  const visible=visibleTracts();syncTractDisplay(visible);updateTractDensityUi();
  const show=new Set(visible.map(t=>t.id));
  // NiiVue 0.69 still draws meshes flagged visible=false in the 3D pass, so hidden tracts are detached from nv.meshes.
  const includeOutliers=state.mode==='surgeon'&&state.showOutliers;
  const meshes=[];
  for(const [id,t] of state.tracts){
    const visible=show.has(id),draw=visible&&t.displayCount>0;t.mesh.visible=draw;if(draw)meshes.push(t.mesh);
    if(t.outlierMesh){t.outlierMesh.visible=visible&&includeOutliers;if(t.outlierMesh.visible)meshes.push(t.outlierMesh)}
  }
  nv.meshes=meshes;
  nv.opts.meshThicknessOn2D=state.tractSlabMm;
  nv.updateGLVolume();applyMeshes3d();
}
function setTractDensityMode(mode){
  if(!['proportional','equalized'].includes(mode))throw Error('Modo de densidade inválido');
  state.tractDensityMode=mode;applyTracts();renderTractList();return mode;
}
function setTractVisible(id,visible,{rerender=true}={}){if(!state.tracts.has(id))throw Error('Trato não encontrado');state.tractVisible[id]=!!visible;applyTracts();if(rerender)renderTractList();else updateTractTrustChip();return visibleTracts().some(t=>t.id===id)}
async function setTractReviewed(id,reviewed){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const t=state.tracts.get(id);if(!t)throw Error('Trato não encontrado');
  const valid=await setItemReview(t.meta,reviewed,await blobBytes(t.meta.blob));applyTracts();renderTractList();return valid;
}
function setShowOutliers(show){state.showOutliers=!!show;$('show-outliers').checked=state.showOutliers;applyTracts();return state.showOutliers}
// X-ray: NiiVue draws occluded mesh fragments (tracts, lesion surfaces) through the opaque volume render,
// so deep tracts and a tumour inside the brain stay visible in 3D.
function setTractXray(v){state.tractXray=clamp(Number(v)||0,0,1);state.nv3d.opts.meshXRay=state.tractXray;const pct=Math.round(state.tractXray*100);$('tract-xray').value=pct;$('tract-xray-out').textContent=pct+'%';state.nv3d.drawScene();return state.tractXray}
function setTractSlab(mm){state.tractSlabMm=clamp(Number(mm)||5,.5,200);applyTracts()}
function eyeIcon(on){return on?'<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M1.5 10S5 4 10 4s8.5 6 8.5 6-3.5 6-8.5 6-8.5-6-8.5-6z"/><circle cx="10" cy="10" r="2.6"/></svg>':'<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 3l14 14M8 4.3A8 8 0 0 1 10 4c5 0 8.5 6 8.5 6a15 15 0 0 1-2.6 3.2M5.3 6.2A15 15 0 0 0 1.5 10S5 16 10 16a8 8 0 0 0 3.4-.8"/></svg>'}
// The eye repaints in place: rebuilding the list on every toggle swapped the button between mousedown and mouseup,
// so quick successive clicks were lost. onToggle reads the current state and returns the resulting visibility.
function paintEye(eye,on){eye.className='eye'+(on?'':' off');eye.innerHTML=eyeIcon(on);eye.title=on?tr('hide'):tr('show');eye.setAttribute('aria-pressed',String(!on))}
function itemRow({color,name,meta,visible,onToggle,reviewed,onReviewed,signature='unreviewed',rowTitle='',metaTitle='',extra}){
  const row=document.createElement('div');row.className='item';
  const eye=document.createElement('button');eye.type='button';paintEye(eye,visible);eye.onclick=()=>{const on=onToggle();if(typeof on==='boolean')paintEye(eye,on)};
  const sw=document.createElement('span');sw.className='swatch';sw.style.background=color;
  const label=document.createElement('span');label.className='name';label.textContent=name;label.title=[name,rowTitle].filter(Boolean).join('\n');
  const info=document.createElement('span');info.className='meta';info.textContent=meta;info.title=metaTitle;
  const rev=document.createElement('label');rev.className='reviewed';
  if(signature!=='signed'){const badge=document.createElement('span');badge.className='badge unreviewed-badge';badge.textContent=tr(signature==='unsigned'?'reviewedUnsigned':'notReviewed');rev.append(badge)}
  const box=document.createElement('input');box.type='checkbox';box.checked=!!reviewed;box.disabled=!state.reviewer.trim();box.onchange=async()=>{try{await onReviewed(box.checked)}catch(e){console.warn(e.message)}renderTractList()};rev.append(box,document.createTextNode(tr('reviewed')));
  row.title=rowTitle;row.append(eye,sw,label,info,...(extra?[extra]:[]),rev);return row;
}
function compactTractProvenance(tracts){
  const summaries=new Set(),valueText=value=>Array.isArray(value)?value.map(valueText).filter(Boolean).join(', '):
    (typeof value==='string'||typeof value==='number'||typeof value==='boolean')?String(value):'';
  for(const meta of tracts){
    const p=meta.provenance;if(p?.recorded!==true)continue;
    const fields=[],software=valueText(p.software),version=valueText(p.version),algorithm=valueText(p.algorithm),bValues=valueText(p.b_values);
    if(software||version)fields.push([software,version&&!software?`v${version}`:version].filter(Boolean).join(' '));
    if(algorithm)fields.push(algorithm);
    if(bValues)fields.push(`b=${bValues}`);
    if(fields.length<3)for(const [key,value] of Object.entries(p)){
      if(key==='recorded'||['software','version','algorithm','b_values'].includes(key))continue;
      const text=valueText(value);if(text)fields.push(`${key} ${text}`);if(fields.length===3)break;
    }
    if(fields.length)summaries.add(fields.join(' · '));
  }
  return [...summaries].join(' / ');
}
function tractAsymmetryWarnings(tracts){
  const warnings=new Map();if(state.mode!=='surgeon')return warnings;
  const positions=new Map(tracts.map((meta,index)=>[tractKey(meta),{meta,index}]));
  for(const [key,leftEntry] of positions){
    const match=/^(.*)_l$/.exec(key);if(!match)continue;
    const rightEntry=positions.get(match[1]+'_r');if(!rightEntry)continue;
    // r17 capsules flag low yield on each tract; the pair warning stays only for older capsules.
    if([leftEntry,rightEntry].some(entry=>typeof entry.meta.trust?.yield_flag==='boolean'))continue;
    const left=sourceStreamlineCount(leftEntry.meta),right=sourceStreamlineCount(rightEntry.meta);
    if(left===null||right===null)continue;const larger=Math.max(left,right);if(larger===0||Math.min(left,right)/larger>=1/3)continue;
    const counts=left.toLocaleString('pt-BR')+' vs '+right.toLocaleString('pt-BR');
    warnings.set(Math.max(leftEntry.index,rightEntry.index),tr('asymmetryWarning').replace('{counts}',counts));
  }
  return warnings;
}
function tractYieldNote(trust){
  const ratio=trust.metrics?.yield_ratio;
  if(!trust.yield_flag||!Number.isFinite(ratio)||ratio<=0)return '';
  return tr('tractTrustYield').replace('{r}',(1/ratio).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1}));
}
function tractTrustPresentation(meta){
  const trust=meta.trust;if(!trust)return {note:tr('qcMissing')};
  const presentation=tractTrustVerdictPresentation(trust),yieldNote=tractYieldNote(trust);
  if(yieldNote)presentation.note=[presentation.note,yieldNote].filter(Boolean).join(' · ');
  return presentation;
}
function tractTrustVerdictPresentation(trust){
  if(trust.verdict==='FAIL'){
    const ratio=trust.metrics?.tortuosity_ratio;
    const ratioText=Number.isFinite(ratio)?ratio.toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1}):'—';
    const reasons={
      tortuosity_ratio:()=>tr('tractTrustRatio').replace('{r}',ratioText),
      median_tortuosity:()=>tr('tractTrustTortuous'),
      cap_fraction:()=>tr('tractTrustCap')
    };
    return {rejected:true,failure:(Array.isArray(trust.failing)?trust.failing:[]).map(code=>reasons[code]?.()).filter(Boolean).join(', ')};
  }
  if(trust.verdict==='PASS')return {note:trust.rim_flag?tr('tractTrustRim'):''};
  if(trust.verdict==='UNAVAILABLE')return {note:tr('qcUnavailable')};
  return {note:tr('qcMissing')};
}
function updateTractTrustChip(){
  const list=$('tract-list');if(!list)return;
  let chip=$('tract-trust-chip');
  if(!visibleTracts().some(meta=>meta.trust?.verdict==='FAIL')){chip?.remove();return}
  if(!chip){chip=document.createElement('span');chip.id='tract-trust-chip';chip.className='badge gold-badge';list.prepend(chip)}
  chip.textContent=tr('tractTrustChip');
}
function renderTractList(){
  const list=$('tract-list');if(!list)return;list.replaceChildren();updateTractDensityUi();
  const tracts=state.manifest.tracts||[];
  if(!tracts.length){const p=document.createElement('p');p.className='empty-note';p.textContent=tr('noTracts');list.append(p);return}
  if(state.mode==='surgeon'){
    const note=document.createElement('p'),provenance=compactTractProvenance(tracts);
    note.className='hint surgeon-only tract-import-note';note.style.margin='0';
    note.textContent=[tr('tractImportNote'),provenance].filter(Boolean).join(' · ');list.append(note);
  }
  const warnings=tractAsymmetryWarnings(tracts);
  for(const [index,meta] of tracts.entries()){
    const t=state.tracts.get(meta.id),visible=tractIsVisible(meta),trust=tractTrustPresentation(meta);
    const extra=trust.note?document.createElement('span'):null;if(extra){extra.className='trust-note'+(meta.trust?.yield_flag?' warn':'');extra.textContent=trust.note}
    const n=meta.n_streamlines??t?.count??0,src=sourceStreamlineCount(meta);
    const filtered=state.mode==='surgeon'&&meta.n_streamlines_outliers>0?
      ' · −'+(100*meta.n_streamlines_outliers/Math.max(1,src||0)).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+'% '+tr('filteredShare'):'';
    let provenance=tr('tractProvenanceMissing');
    if(meta.provenance?.recorded){const fields=Object.entries(meta.provenance).filter(([k])=>k!=='recorded').map(([k,v])=>k+' '+v).join(' · ');provenance=tr('tractProvenance')+': '+fields}
    const sourceText=src!==null&&src!==n?' / '+src.toLocaleString('pt-BR'):'';
    list.append(itemRow({color:meta.color,name:tractDisplayName(meta)+(trust.rejected?` — ${tr('tractTrustRejected')} (${trust.failure})`:''),extra,
      meta:n.toLocaleString('pt-BR')+' '+tr('streamlines')+sourceText+filtered,
      visible,onToggle:()=>setTractVisible(meta.id,!tractIsVisible(meta),{rerender:false}),reviewed:isValidReview(meta),signature:reviewStatus(meta),
      onReviewed:v=>setTractReviewed(meta.id,v),rowTitle:state.mode==='surgeon'?provenance:'',metaTitle:filtered?meta.outlier_filter||'': ''}));
    // Tract names are long (bundle + endpoints): the count moves under the name instead of truncating it.
    list.lastElementChild.classList.add('stacked');
    if(warnings.has(index)){const warning=document.createElement('p');warning.className='tract-warning';warning.textContent=warnings.get(index);list.append(warning)}
  }
  updateTractTrustChip();
}
