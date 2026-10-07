// Masks: lesion masks are uint8 overlays in their colour, drawn in 2D as a translucent fill plus an opaque
// rim band (voxels within two face steps of the outside), and in 3D as a solid mesh. NiiVue 0.69 ignores the alpha
// of a label colormap entry in 2D (measured: A=20 and A=90 render identically), so fill and rim are two layers
// and the translucency comes from the fill layer's opacity.
// Render masks (role "render") are display-only inputs of the 3D copy: kept for save, never listed or drawn.
// Editing uses NiiVue's drawing layer (pen, eraser, click-to-segment); every change is copied back to the
// capsule's native C-order array.
const MASK_FILL_OPACITY=.4;
function maskDisplay(data,fill,rim){
  // Rim = mask voxels within two face steps of the outside. NiiVue samples the overlay with linear interpolation,
  // so a one-voxel ring renders as a faint tent; a two-voxel band keeps a clearly visible outline.
  const [nx,ny,nz]=state.manifest.grid.dims,sl=nx*ny;
  const nb=(n,i,j,k,f)=>f(i>0?n-1:-1)||f(i<nx-1?n+1:-1)||f(j>0?n-nx:-1)||f(j<ny-1?n+nx:-1)||f(k>0?n-sl:-1)||f(k<nz-1?n+sl:-1);
  for(let k=0,n=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++,n++){
    fill[n]=data[n]?1:0;rim[n]=data[n]&&nb(n,i,j,k,m=>m<0||!data[m])?1:0;
  }
  const edge=rim.slice();
  for(let k=0,n=0;k<nz;k++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++,n++)if(data[n]&&!edge[n]&&nb(n,i,j,k,m=>m>=0&&edge[m]))rim[n]=1;
}
async function maskLayer(meta,disp,part,opacity){
  const img=await niivue.NVImage.new(niftiBytes(state.manifest.grid,'uint8',disp,{intent:1002,calMin:0,calMax:1}),`mask-${meta.id}-${part}.nii`,'gray',1,null,0,1);
  img.img=disp;const [r,g,b]=hexToRgb(meta.color);
  img.setColormapLabel({R:[0,r],G:[0,g],B:[0,b],A:[0,255],I:[0,1],labels:['',meta.label||meta.id]});
  img.opacity=opacity;return img;
}
async function maskImage(meta,data){
  const fill=new Uint8Array(data.length),rim=new Uint8Array(data.length);maskDisplay(data,fill,rim);
  return {fill:await maskLayer(meta,fill,'fill',MASK_FILL_OPACITY),rim:await maskLayer(meta,rim,'rim',1)};
}
function isRenderMask(meta){return meta.role==='render'}
async function loadMasks(){
  const total=state.manifest.grid.dims.reduce((a,b)=>a*b,1);
  for(const meta of state.manifest.masks){
    const data=await blobBytes(meta.blob);if(data.length!==total)throw Error('mask length');
    await registerBlobIntegrity(meta,data);
    state.masks.set(meta.id,{meta,data,version:0,nvimg:isRenderMask(meta)?null:await maskImage(meta,data)});
  }
}
function refreshMask(m){m.version=(m.version||0)+1;if(m.nvimg){maskDisplay(m.data,m.nvimg.fill.img,m.nvimg.rim.img)}}
function countMask(mask){let n=0;for(let i=0;i<mask.length;i++)n+=mask[i]?1:0;return n}
function updateMaskVolume(meta,redraw=true){
  const m=state.masks.get(meta.id);invalidateReview(meta);refreshMask(m);state.maskRev=(state.maskRev||0)+1;meta.volume_ml=Number((countMask(m.data)*voxelVolumeML()).toFixed(3));
  renderMaskList();if(redraw)applyLayers();return meta.volume_ml;
}
function setMaskVisible(id,visible,{rerender=true}={}){if(!state.masks.has(id)||isRenderMask(state.masks.get(id).meta))throw Error('Máscara não encontrada');state.maskVisible[id]=!!visible;applyLayers();if(rerender)renderMaskList();return visibleMasks().some(m=>m.id===id)}
async function setMaskReviewed(id,reviewed){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const m=state.masks.get(id);if(!m||isRenderMask(m.meta)&&m.meta.id!=='vessels'&&!m.meta.id.startsWith('vessels_'))throw Error('Máscara não encontrada');
  const valid=await setItemReview(m.meta,reviewed,m.data);applyLayers();renderMaskList();renderPreset3d();return valid;
}
async function createMask(label=tr('newMaskLabel'),color='#E4572E'){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const id='mask-'+crypto.randomUUID().slice(0,8),blob='mask_'+id.replace(/[^A-Za-z0-9_-]/g,'_'),total=state.manifest.grid.dims.reduce((a,b)=>a*b,1);
  const meta={id,blob,label,color,volume_ml:0,source:'surgeon',reviewed:false},data=new Uint8Array(total);
  meta.blob_sha256=await sha256Hex(data);state.blobIntegrity.set(blob,true);
  state.manifest.masks.push(meta);state.masks.set(id,{meta,data,nvimg:await maskImage(meta,data)});renderMaskList();applyLayers();return meta;
}
// Native (capsule) order <-> NiiVue RAS drawing order, mirroring NiiVue's loadDrawing LUTs.
function drawLuts(){
  const img=state.nv.back,perm=img.permRAS,dims=img.hdr.dims,layout=[0,0,0];
  for(let s=0;s<3;s++)for(let a=0;a<3;a++)if(Math.abs(perm[s])-1===a)layout[a]=s*Math.sign(perm[s]);
  let stride=1;const instride=[1,1,1],flip=[false,false,false];
  for(let s=0;s<3;s++)for(let a=0;a<3;a++)if(Math.abs(layout[a])===s){instride[a]=stride;if(layout[a]<0||Object.is(layout[a],-0))flip[a]=true;stride*=dims[a+1]}
  const lut=(n,st,f)=>{const out=new Int32Array(n);for(let c=0;c<n;c++)out[c]=(f?n-1-c:c)*st;return out};
  return {dims,x:lut(dims[1],instride[0],flip[0]),y:lut(dims[2],instride[1],flip[1]),z:lut(dims[3],instride[2],flip[2])};
}
function nativeToDrawing(data){const L=drawLuts(),out=new Uint8Array(data.length);let o=0;for(let k=0;k<L.dims[3];k++)for(let j=0;j<L.dims[2];j++)for(let i=0;i<L.dims[1];i++,o++)out[L.x[i]+L.y[j]+L.z[k]]=data[o]?1:0;return out}
function drawingToNative(bitmap,out){const L=drawLuts();let o=0;for(let k=0;k<L.dims[3];k++)for(let j=0;j<L.dims[2];j++)for(let i=0;i<L.dims[1];i++,o++)out[o]=bitmap[L.x[i]+L.y[j]+L.z[k]]?1:0;return out}
function editMask(id){
  const nv=state.nv;
  if(state.editingMask){syncDrawing();nv.closeDrawing?.();nv.setDrawingEnabled(false)}
  state.editingMask=null;
  if(id){
    if(state.mode==='patient')throw Error(tr('editBlocked'));
    const m=state.masks.get(id);if(!m||isRenderMask(m.meta))throw Error('Máscara não encontrada');
    state.editingMask=id;state.maskVisible[id]=true;applyLayers();
    const [r,g,b]=hexToRgb(m.meta.color);nv.setDrawColormap({R:[0,r],G:[0,g],B:[0,b],A:[0,255],I:[0,1]});
    nv.createEmptyDrawing();nv.drawBitmap.set(nativeToDrawing(m.data));nv.drawAddUndoBitmap?.();nv.refreshDrawing(true);nv.setDrawOpacity(.65);
  }else applyLayers();
  renderMaskList();return state.editingMask;
}
function syncDrawing(){
  const nv=state.nv,m=state.masks.get(state.editingMask);if(!m||!nv.drawBitmap)return;
  drawingToNative(nv.drawBitmap,m.data);updateMaskVolume(m.meta,false);
}
function renderMaskList(){
  const list=$('mask-list');if(!list)return;list.replaceChildren();
  const masks=lesionMasks();
  if(!masks.length){const p=document.createElement('p');p.className='empty-note';p.textContent=tr('noMasks');list.append(p);return}
  for(const meta of masks){
    const visible=state.maskVisible[meta.id]!==false,edit=document.createElement('button');edit.type='button';edit.className='eye';
    edit.textContent=state.editingMask===meta.id?tr('editing'):tr('edit');edit.classList.toggle('active',state.editingMask===meta.id);
    edit.onclick=()=>editMask(state.editingMask===meta.id?null:meta.id);
    const row=itemRow({color:meta.color,name:meta.label||meta.id,meta:formatVolumeMl(meta.volume_ml),visible,
      onToggle:()=>setMaskVisible(meta.id,state.maskVisible[meta.id]===false,{rerender:false}),reviewed:isValidReview(meta),signature:reviewStatus(meta),
      onReviewed:v=>setMaskReviewed(meta.id,v),extra:edit});
    row.classList.toggle('editing',state.editingMask===meta.id);list.append(row);
  }
}
// Viewport legend for visible lesion masks; in surgeon mode an unreviewed one carries its badge.
function renderMaskLegend(){
  const box=$('mask-legend');if(!box||!state.manifest)return;box.replaceChildren();
  for(const meta of visibleMasks()){
    const row=document.createElement('div');row.className='legend-row';row.dataset.mask=meta.id;
    const sw=document.createElement('span');sw.className='swatch';sw.style.background=meta.color;
    const name=document.createElement('span');name.textContent=meta.label||meta.id;row.append(sw,name);
    if(reviewStatus(meta)!=='signed'&&state.mode!=='patient'){const b=document.createElement('span');b.className='badge unreviewed-badge';b.textContent=tr(reviewStatus(meta)==='unsigned'?'reviewedUnsigned':'notReviewed');row.append(b)}
    box.append(row);
  }
  box.hidden=!box.childElementCount;
}
// Scripted equivalents kept from v1 (six-connected grow, slice brush); volume = voxels × |det(affine)|.
function growRegionRAS(x,y,z,options={}){
  const volumeId=options.volumeId||state.base,vol=state.volumes.get(volumeId);if(!vol)throw Error('Volume não encontrado');
  const seed=rasToVoxel(x,y,z).map(Math.round),seedIndex=voxelToIndex(...seed);if(seedIndex<0)throw Error('Semente fora do volume');
  const tolerance=Number(options.tolerance??50),radius=Number(options.radiusMm??30);if(!(tolerance>0&&radius>0))throw Error('Tolerância ou raio inválido');
  const neighbours=[];for(let dk=-1;dk<=1;dk++)for(let dj=-1;dj<=1;dj++)for(let di=-1;di<=1;di++){const n=voxelToIndex(seed[0]+di,seed[1]+dj,seed[2]+dk);if(n>=0)neighbours.push(vol.data[n]*vol.meta.slope+vol.meta.intercept)}
  neighbours.sort((a,b)=>a-b);
  const center=neighbours[Math.floor(neighbours.length/2)],total=vol.data.length,seen=new Uint8Array(total),queue=new Uint32Array(total),mask=new Uint8Array(total),a=state.manifest.grid.affine_ras;
  const col=c=>Math.hypot(a[0][c],a[1][c],a[2][c]),spacing=[col(0),col(1),col(2)];
  let head=0,tail=0;queue[tail++]=seedIndex;seen[seedIndex]=1;
  while(head<tail){
    const n=queue[head++],[i,j,k]=indexToVoxel(n),dx=(i-seed[0])*spacing[0],dy=(j-seed[1])*spacing[1],dz=(k-seed[2])*spacing[2];
    if(dx*dx+dy*dy+dz*dz>radius*radius)continue;
    const value=vol.data[n]*vol.meta.slope+vol.meta.intercept;if(Math.abs(value-center)>tolerance)continue;mask[n]=1;
    for(const q of [[i-1,j,k],[i+1,j,k],[i,j-1,k],[i,j+1,k],[i,j,k-1],[i,j,k+1]]){const idx=voxelToIndex(...q);if(idx>=0&&!seen[idx]){seen[idx]=1;queue[tail++]=idx}}
  }
  return (async()=>{
    const meta=options.maskId?state.manifest.masks.find(m=>m.id===options.maskId):await createMask(options.label||'Crescimento 3D');
    if(!meta)throw Error('Máscara não encontrada');state.masks.get(meta.id).data.set(mask);
    const volume_ml=updateMaskVolume(meta);return {id:meta.id,volume_ml,voxel_count:countMask(mask)};
  })();
}
async function brushAtRAS(x,y,z,erase=false,radiusMm=4,view='axial'){
  let meta=lesionMasks().at(-1);if(!meta)meta=await createMask('Pincel');
  const mask=state.masks.get(meta.id).data,ijk=rasToVoxel(x,y,z),a=state.manifest.grid.affine_ras,dims=state.manifest.grid.dims;
  const minSp=Math.min(...[0,1,2].map(c=>Math.hypot(a[0][c],a[1][c],a[2][c]))),r=Math.max(1,Math.ceil(radiusMm/minSp));
  const axis=view==='axial'?2:view==='coronal'?1:0,fixed=Math.round(ijk[axis]);
  for(let k=Math.max(0,Math.floor(ijk[2]-r));k<=Math.min(dims[2]-1,Math.ceil(ijk[2]+r));k++)for(let j=Math.max(0,Math.floor(ijk[1]-r));j<=Math.min(dims[1]-1,Math.ceil(ijk[1]+r));j++)for(let i=Math.max(0,Math.floor(ijk[0]-r));i<=Math.min(dims[0]-1,Math.ceil(ijk[0]+r));i++){
    if([i,j,k][axis]!==fixed)continue;const p=affinePoint(i,j,k);if(norm([p[0]-x,p[1]-y,p[2]-z])<=radiusMm)mask[voxelToIndex(i,j,k)]=erase?0:1;
  }
  return updateMaskVolume(meta);
}
