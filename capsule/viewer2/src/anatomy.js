// Anatomy: manifest.anatomy items are automatic label maps (uint8, one value per structure) on the capsule grid.
// 2D: per item one label-coloured fill layer (40% opacity) and one opaque rim layer, like lesion masks (NiiVue 0.69
// ignores per-label alpha in 2D, so hidden labels are zeroed in the display copies). 3D: one surface mesh per label,
// built on first show and cached; opacity per group. Default: only the ventricles are shown. Patient mode hides an
// item until the surgeon signs its current blob. Save carries the blobs byte-for-byte; review fields may change.
const ANATOMY_GROUPS=['Ventrículos','Tronco e cerebelo','Núcleos profundos','Estruturas límbicas','Osso','Seios e cavidades','Vasos','Nervos e órbita'];
const ANATOMY_DEFAULT_GROUPS=new Set(['Ventrículos']);
const anatomyMeshes=new Map();
function groupRank(g){const i=ANATOMY_GROUPS.indexOf(g);return i<0?99:i}
function anatomyItems(){return state.manifest?.anatomy||[]}
function anatomyKey(id,value){return `${id}:${value}`}
function anatomyItemVisibilityKey(id){return `${id}:item`}
function failedGyriQc(item){return item.id==='anat_mr_gyri'&&item.qc?.verdict==='FAIL'}
function anatomyItemVisible(item){return !failedGyriQc(item)||state.anatomyVisible[anatomyItemVisibilityKey(item.id)]===true}
function anatomyLabelVisible(item,label){if(failedGyriQc(item))return anatomyItemVisible(item);const v=state.anatomyVisible[anatomyKey(item.id,label.value)];return v===undefined?ANATOMY_DEFAULT_GROUPS.has(label.group):v}
function anatomyGroupOpacity(group){return state.anatomyOpacity[group]??1}
function visibleAnatomyItems(){return anatomyItems().filter(it=>state.anatomy.has(it.id)&&anatomyItemVisible(it)&&!patientHides(it))}
function visibleAnatomyLabels(){const out=[];for(const it of visibleAnatomyItems())for(const l of it.labels||[])if(anatomyLabelVisible(it,l))out.push([it,l]);return out}
function labelDisplay(data,shown,fill,rim){
  // Same two-voxel rim as maskDisplay, per label: a voxel is edge when a face neighbour is outside or another label.
  const [nx,ny,nz]=state.manifest.grid.dims,sl=nx*ny;
  const nb=(n,i,j,k,f)=>f(i>0?n-1:-1)||f(i<nx-1?n+1:-1)||f(j>0?n-nx:-1)||f(j<ny-1?n+nx:-1)||f(k>0?n-sl:-1)||f(k<nz-1?n+sl:-1);
  const edge=new Uint8Array(data.length);
  for(let k=0,n=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++,n++){
    const v=data[n],on=v&&shown[v];fill[n]=on?v:0;rim[n]=0;
    if(on&&nb(n,i,j,k,m=>m<0||data[m]!==v)){rim[n]=v;edge[n]=1}
  }
  for(let k=0,n=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++,n++){const v=data[n];if(v&&shown[v]&&!edge[n]&&nb(n,i,j,k,m=>m>=0&&edge[m]&&data[m]===v))rim[n]=v}
}
async function anatomyLayer(item,disp,part,opacity){
  const labels=item.labels||[],max=Math.max(1,...labels.map(l=>l.value));
  const img=await niivue.NVImage.new(niftiBytes(state.manifest.grid,'uint8',disp,{intent:1002,calMin:0,calMax:max}),`anatomy-${item.id}-${part}.nii`,'gray',1,null,0,max);
  img.img=disp;const R=[0],G=[0],B=[0],A=[0],I=[0],names=[''];
  for(const l of labels){const [r,g,b]=hexToRgb(l.color);R.push(r);G.push(g);B.push(b);A.push(255);I.push(l.value);names.push(l.name||l.key)}
  img.setColormapLabel({R,G,B,A,I,labels:names});img.opacity=opacity;return img;
}
async function loadAnatomy(){
  const total=state.manifest.grid.dims.reduce((a,b)=>a*b,1);
  for(const item of anatomyItems()){
    const data=await blobBytes(item.blob);if(data.length!==total)throw Error('anatomy length');
    await registerBlobIntegrity(item,data);
    const fill=new Uint8Array(total),rim=new Uint8Array(total);
    const entry={meta:item,data,fill:await anatomyLayer(item,fill,'fill',MASK_FILL_OPACITY),rim:await anatomyLayer(item,rim,'rim',1),shownKey:null};
    state.anatomy.set(item.id,entry);refreshAnatomyDisplay(entry);
  }
}
function refreshAnatomyDisplay(entry){
  const shown=new Uint8Array(256);for(const l of entry.meta.labels||[])if(anatomyLabelVisible(entry.meta,l))shown[l.value]=1;
  const key=shown.join('');if(key===entry.shownKey)return false;
  labelDisplay(entry.data,shown,entry.fill.img,entry.rim.img);entry.shownKey=key;return true;
}
// Layers appended by applyLayers (view.js) after the lesion masks.
function anatomyLayers(){
  const out=[];
  for(const it of visibleAnatomyItems()){const e=state.anatomy.get(it.id);if(!(it.labels||[]).some(l=>anatomyLabelVisible(it,l)))continue;refreshAnatomyDisplay(e);out.push(e.fill,e.rim)}
  return out;
}
function anatomyMesh(item,label){
  const key=anatomyKey(item.id,label.value);if(anatomyMeshes.has(key))return anatomyMeshes.get(key);
  const data=state.anatomy.get(item.id).data,bin=new Uint8Array(data.length);for(let n=0;n<data.length;n++)bin[n]=data[n]===label.value?1:0;
  const surf=maskSurface(bin,state.manifest.grid.dims);let mesh=null;
  if(surf&&surf.tris.length){const [r,g,b]=hexToRgb(label.color);mesh=new niivue.NVMesh(surf.pts,surf.tris,`anatomy-${key}`,new Uint8Array([r,g,b,255]),1,true,state.nv3d.gl);surfaceShader(mesh)}
  anatomyMeshes.set(key,mesh);return mesh;
}
// Meshes appended by applyMeshes3d (render3d.js).
function anatomyMeshList(){
  const out=[];
  for(const [it,l] of visibleAnatomyLabels()){const mesh=anatomyMesh(it,l);if(!mesh)continue;const op=anatomyGroupOpacity(l.group);if(mesh.opacity!==op){mesh.opacity=op;mesh.updateMesh?.(state.nv3d.gl)}out.push(mesh)}
  return out;
}
function anatomyItem(id){const e=state.anatomy.get(id);if(!e)throw Error('Anatomia não encontrada');return e.meta}
function anatomyChanged(){applyLayers();renderAnatomyList();renderReadout()}
function setAnatomyVisible(id,value,visible){
  const item=anatomyItem(id),label=(item.labels||[]).find(l=>l.value===Number(value));if(!label)throw Error('Estrutura não encontrada');
  state.anatomyVisible[anatomyKey(id,label.value)]=!!visible;anatomyChanged();return anatomyLabelVisible(item,label);
}
function setAnatomyItemVisible(id,visible){
  const item=anatomyItem(id);if(!failedGyriQc(item))throw Error('Controle QC não aplicável');
  state.anatomyVisible[anatomyItemVisibilityKey(id)]=!!visible;anatomyChanged();return anatomyItemVisible(item);
}
function setAnatomyGroupVisible(id,group,visible){
  const item=anatomyItem(id),labels=(item.labels||[]).filter(l=>l.group===group);if(!labels.length)throw Error('Grupo não encontrado');
  for(const l of labels)state.anatomyVisible[anatomyKey(id,l.value)]=!!visible;anatomyChanged();return labels.length;
}
function setAnatomyOpacity(group,opacity){state.anatomyOpacity[group]=clamp(Number(opacity),0,1);if(typeof applyMeshes3d==='function')applyMeshes3d();state.nv3d?.drawScene();renderAnatomyList();return state.anatomyOpacity[group]}
async function setAnatomyReviewed(id,reviewed){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const item=anatomyItem(id),data=state.anatomy.get(id).data;
  const valid=await setItemReview(item,reviewed,data);anatomyChanged();return valid;
}
function visibleAnatomy(){return visibleAnatomyLabels().map(([it,l])=>anatomyKey(it.id,l.value))}
function anatomyAtRAS(x,y,z){
  const n=voxelToIndex(...rasToVoxel(x,y,z));if(n<0)return [];const out=[];
  for(const it of visibleAnatomyItems()){const v=state.anatomy.get(it.id).data[n];const l=v&&(it.labels||[]).find(q=>q.value===v);if(l)out.push({item:it.id,value:l.value,name:l.name||l.key,group:l.group})}
  return out;
}
function anatomyTitle(item){const vol=state.volumes.get(item.for_volume)?.meta;return `${tr('anatomyAuto')}${vol?` · ${vol.label}`:''}`}
function renderAnatomyList(){
  const list=$('anatomy-list');if(!list||!state.manifest)return;list.replaceChildren();
  const items=anatomyItems().filter(it=>state.anatomy.has(it.id));
  $('anatomy-group').hidden=!items.length;
  for(const item of items){
    const card=document.createElement('div');card.className='anatomy-item';card.dataset.item=item.id;
    const head=document.createElement('div');head.className='anatomy-head';
    const title=document.createElement('span');title.className='name';title.textContent=anatomyTitle(item);title.title=[item.method,item.licence].filter(Boolean).join(' · ');head.append(title);
    const qc=item.qc;
    if(item.id==='anat_mr_gyri'&&(qc?.verdict==='UNAVAILABLE'||!qc)){const note=document.createElement('span');note.className='anatomy-qc-note';note.textContent=tr(qc?'qcUnavailable':'qcMissing');head.append(note)}
    const rev=document.createElement('label');rev.className='reviewed';
    const signature=reviewStatus(item);
    if(signature!=='signed'){const b=document.createElement('span');b.className='badge unreviewed-badge';b.textContent=tr(signature==='unsigned'?'reviewedUnsigned':'notReviewed');rev.append(b)}
    const box=document.createElement('input');box.type='checkbox';box.checked=isValidReview(item);box.disabled=!state.reviewer.trim();
    box.onchange=async()=>{try{await setAnatomyReviewed(item.id,box.checked)}catch(e){console.warn(e.message)}renderAnatomyList()};
    rev.append(box,document.createTextNode(tr('reviewed')));
    card.append(head,rev);
    if(failedGyriQc(item)){
      const row=document.createElement('div');row.className='anatomy-qc-row';
      const message=document.createElement('span');message.className='name';message.textContent=tr('parcQcFail').replace('{n}',String(qc.n_failing));
      const eye=document.createElement('button'),shown=anatomyItemVisible(item);eye.type='button';eye.className='eye'+(shown?'':' off');eye.innerHTML=eyeIcon(shown);eye.title=shown?tr('hide'):tr('show');eye.setAttribute('aria-label',eye.title);eye.onclick=()=>setAnatomyItemVisible(item.id,!shown);
      row.append(eye,message);
      if(shown){const chip=document.createElement('span');chip.className='badge anatomy-qc-chip';chip.textContent=tr('parcQcRejected');row.append(chip)}
      card.append(row);list.append(card);continue;
    }
    const groups=[...new Set((item.labels||[]).map(l=>l.group))].sort((a,b)=>groupRank(a)-groupRank(b));
    for(const group of groups){
      const labels=item.labels.filter(l=>l.group===group),on=labels.filter(l=>anatomyLabelVisible(item,l)).length;
      const g=document.createElement('div');g.className='anatomy-grp';g.dataset.group=group;
      const gh=document.createElement('div');gh.className='anatomy-grp-head';
      const eye=document.createElement('button');eye.type='button';eye.className='eye'+(on?'':' off');eye.innerHTML=eyeIcon(!!on);eye.title=on?tr('hide'):tr('show');
      eye.onclick=()=>setAnatomyGroupVisible(item.id,group,!on);
      const name=document.createElement('span');name.className='name';name.textContent=group;
      const count=document.createElement('span');count.className='meta';count.textContent=`${on}/${labels.length}`;
      const op=document.createElement('input');op.type='range';op.min=0;op.max=100;op.step=5;op.value=Math.round(anatomyGroupOpacity(group)*100);op.title=`${tr('opacity3d')} ${op.value}%`;op.setAttribute('aria-label',`${tr('opacity3d')} · ${group}`);
      op.oninput=()=>{state.anatomyOpacity[group]=Number(op.value)/100;op.title=`${tr('opacity3d')} ${op.value}%`;applyMeshes3d();state.nv3d.drawScene()};
      gh.append(eye,name,count,op);g.append(gh);
      for(const l of labels){
        const vis=anatomyLabelVisible(item,l),row=document.createElement('div');row.className='item anatomy-label';row.dataset.value=l.value;
        const e=document.createElement('button');e.type='button';e.className='eye'+(vis?'':' off');e.innerHTML=eyeIcon(vis);e.title=vis?tr('hide'):tr('show');e.onclick=()=>setAnatomyVisible(item.id,l.value,!vis);
        const sw=document.createElement('span');sw.className='swatch';sw.style.background=l.color;
        const nm=document.createElement('span');nm.className='name';nm.textContent=l.name||l.key;nm.title=l.name||l.key;
        const ml=document.createElement('span');ml.className='meta';
        const qcFailed=qc?.verdict==='PASS'&&qc.failing?.includes(l.key);
        ml.textContent=qcFailed?tr('qcPassFailed'):formatVolumeMl(l.volume_ml);
        row.append(e,sw,nm,ml);g.append(row);
      }
      card.append(g);
    }
    list.append(card);
  }
}
