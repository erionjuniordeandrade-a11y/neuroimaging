// Six-connected seeded growing and slice brush. Volumes use affine determinant (mm³).
function countMask(mask){let n=0;for(const b of mask)n+=b?1:0;return n}
function updateMaskVolume(meta){const arr=state.masks.get(meta.id);meta.volume_ml=Number((countMask(arr)*voxelVolumeML()).toFixed(3));state.glData.maskDirty=true;renderMaskList();renderAll();return meta.volume_ml}
function createMask(label='Nova máscara',color='#E4572E'){
  const id='mask-'+crypto.randomUUID(),blob='mask_'+id,total=state.manifest.grid.dims.reduce((a,b)=>a*b,1);
  const meta={id,blob,label,color,volume_ml:0,source:'surgeon',reviewed:false};state.manifest.masks.push(meta);state.masks.set(id,new Uint8Array(total));state.glData.maskDirty=true;renderMaskList();return meta;
}
function growRegionRAS(x,y,z,options={}){
  const volumeId=options.volumeId||state.base,vol=state.volumes.get(volumeId);if(!vol)throw Error('Volume não encontrado');
  const seed=rasToVoxel(x,y,z).map(Math.round),dims=state.manifest.grid.dims,seedIndex=voxelToIndex(...seed);if(seedIndex<0)throw Error('Semente fora do volume');
  const tolerance=Number(options.tolerance??$('grow-tolerance').value),radius=Number(options.radiusMm??$('grow-radius').value);
  if(!(tolerance>0&&radius>0))throw Error('Tolerância ou raio inválido');
  const neighbours=[];for(let dk=-1;dk<=1;dk++)for(let dj=-1;dj<=1;dj++)for(let di=-1;di<=1;di++){const n=voxelToIndex(seed[0]+di,seed[1]+dj,seed[2]+dk);if(n>=0)neighbours.push(vol.data[n]*vol.meta.slope+vol.meta.intercept)}neighbours.sort((a,b)=>a-b);
  const center=neighbours[Math.floor(neighbours.length/2)],total=vol.data.length,seen=new Uint8Array(total),queue=new Uint32Array(total),mask=new Uint8Array(total),spacing=state.manifest.grid.spacing_mm;
  let head=0,tail=0;queue[tail++]=seedIndex;seen[seedIndex]=1;
  while(head<tail){
    const n=queue[head++],[i,j,k]=indexToVoxel(n),dx=(i-seed[0])*spacing[0],dy=(j-seed[1])*spacing[1],dz=(k-seed[2])*spacing[2];
    if(dx*dx+dy*dy+dz*dz>radius*radius)continue;
    const value=vol.data[n]*vol.meta.slope+vol.meta.intercept;if(Math.abs(value-center)>tolerance)continue;mask[n]=1;
    for(const q of [[i-1,j,k],[i+1,j,k],[i,j-1,k],[i,j+1,k],[i,j,k-1],[i,j,k+1]]){const idx=voxelToIndex(...q);if(idx>=0&&!seen[idx]){seen[idx]=1;queue[tail++]=idx}}
  }
  const meta=options.maskId?state.manifest.masks.find(m=>m.id===options.maskId):createMask(options.label||'Crescimento 3D');if(!meta)throw Error('Máscara não encontrada');
  state.masks.set(meta.id,mask);const volume_ml=updateMaskVolume(meta);return {id:meta.id,volume_ml,voxel_count:countMask(mask)};
}
function brushAtRAS(x,y,z,erase=false,radiusMm=Number($('tool-radius').value),view=state.activeView){
  let meta=state.manifest.masks.at(-1);if(!meta)meta=createMask('Pincel');const mask=state.masks.get(meta.id),ijk=rasToVoxel(x,y,z),r=Math.max(1,Math.ceil(radiusMm/Math.min(...state.manifest.grid.spacing_mm))),dims=state.manifest.grid.dims;
  const axis=view==='axial'?2:view==='coronal'?1:0,fixed=Math.round(ijk[axis]);
  for(let k=Math.max(0,Math.floor(ijk[2]-r));k<=Math.min(dims[2]-1,Math.ceil(ijk[2]+r));k++)for(let j=Math.max(0,Math.floor(ijk[1]-r));j<=Math.min(dims[1]-1,Math.ceil(ijk[1]+r));j++)for(let i=Math.max(0,Math.floor(ijk[0]-r));i<=Math.min(dims[0]-1,Math.ceil(ijk[0]+r));i++){
    const q=[i,j,k];if(q[axis]!==fixed)continue;const p=affinePoint(i,j,k);if(norm(p.map((v,t)=>v-[x,y,z][t]))<=radiusMm)mask[voxelToIndex(i,j,k)]=erase?0:1;
  }
  return updateMaskVolume(meta);
}
