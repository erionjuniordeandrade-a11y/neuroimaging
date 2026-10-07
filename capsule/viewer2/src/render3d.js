// 3D render instance. NiiVue colours every tile from one colormapped texture per layer, so the 2D slices
// (canvas #gl, original values, reading window) and the 3D view (canvas #gl3d) are two NiiVue instances.
// The 3D instance only ever holds a display copy of the base volume: voxels outside the preset's render
// mask are set to background, CT is smoothed along its acquired slice axis. Values shown or read in 2D
// never come from this copy.
const RENDER_COLORMAPS={
  // NiiVue classifies BEFORE trilinear sampling: each voxel becomes an RGBA8 texel and the ray interpolates
  // un-premultiplied colour. A transparent node coloured black therefore bleeds dark into every surface texel
  // by an amount that depends on the surface's sub-voxel phase, which paints one ring per voxel layer on the
  // calvarium (and stripes on the brain). Transparent nodes carry the first visible colour, so interpolation
  // never mixes in black (GPU probe, vertex crop band-pass |box3-box11|: 2.57 -> 0.68).
  // Ivory bone (range 150..1650 HU): transparent below ~240 HU, then a gradual ramp.
  'cc-bone':{R:[176,176,176,206,226,240],G:[160,160,160,188,208,224],B:[126,126,126,152,172,190],A:[0,0,10,60,150,190],I:[0,15,26,50,95,255]},
  // Skin shell (translucent, warm) over ivory bone (range -500..1500 HU).
  'cc-skinbone':{R:[222,222,222,226,214,240,255],G:[170,170,170,176,198,232,250],B:[150,150,150,154,170,212,236],A:[0,0,12,14,40,230,255],I:[0,36,44,90,98,112,255]},
  // MR brain surface: CSF transparent, cortex warm grey-pink, white matter lighter.
  'cc-brain':{R:[168,168,168,214,236,248],G:[142,142,142,184,212,232],B:[136,136,136,172,198,220],A:[0,0,60,210,250,255],I:[0,50,70,110,170,255]},
  // MR skin surface.
  'cc-skin':{R:[200,200,200,226,240],G:[160,160,160,184,206],B:[140,140,140,160,184],A:[0,0,80,230,255],I:[0,22,34,70,255]},
  // MR brain + vessels (auto range p01..p99): cortex warm grey-pink as cc-brain; the brightest band (enhanced
  // veins and dura on contrast T1, above ~55% of the range once smoothed) turns blue.
  'cc-vessels':{R:[178,178,178,226,232,58,70],G:[150,150,150,196,204,96,120],B:[142,142,142,184,192,214,240],A:[0,0,70,225,240,255,255],I:[0,50,70,122,134,148,255]}
};
function lesionMasks(){return (state.manifest?.masks||[]).filter(m=>(m.role||'lesion')!=='render')}
// Render masks are resolved by name for the base volume ("head", "head_mr", "head-mr_3"); a mask made for a
// co-registered sibling of the same modality is used when the volume has none of its own.
function renderMaskFor(volumeId,name){
  // Shared contract: role "render", for_volume === the volume, id === name or name + "_<suffix>".
  return (state.manifest?.masks||[]).find(m=>m.role==='render'&&m.for_volume===volumeId&&state.masks.has(m.id)&&
    (m.id===name||m.id.startsWith(name+'_')))||null;
}
async function initNiivue3d(){
  const nv=new niivue.Niivue({
    backColor:[0,0,0,1],crosshairColor:[.79,.66,.3,1],crosshairWidth:1,show3Dcrosshair:true,
    isRadiologicalConvention:true,dragAndDropEnabled:false,isColorbar:false,loadingText:'',logLevel:'error',
    viewModeHotKey:'',clipPlaneHotKey:'',cycleClipPlaneHotKey:'',isOrientationTextVisible:false,isOrientCube:true,
    clipPlaneColor:[0,0,0,0],isAlphaClipDark:true
  });
  await nv.attachToCanvas($('gl3d'));
  for(const [name,cm] of Object.entries(RENDER_COLORMAPS))nv.addColormap(name,cm);
  nv.setSliceType(nv.sliceTypeRender);
  nv.onAzimuthElevationChange=()=>{if(state.clip)applyClip(false)};
  state.nv3d=nv;return nv;
}
// --- volume processing -------------------------------------------------------------------------------
// Acquired slice spacing hides in a resampled grid as periodic kinks: linear interpolation between thick
// slices leaves near-zero second differences except at the original slice positions. Per axis, the mean
// |second difference| in tissue, and the period of its per-slice profile, give the axis to smooth and by how much.
function sliceAxisEstimate(values,dims){
  const [nx,ny,nz]=dims,step=[1,nx,nx*ny],out=[];
  for(let a=0;a<3;a++){
    // Profile of mean |second difference| in tissue, per index along axis a (full resolution along a).
    const len=dims[a],prof=new Float64Array(len),cnt=new Float64Array(len);
    for(let k=1;k<nz-1;k+=(a===2?1:2))for(let j=1;j<ny-1;j+=(a===1?1:2))for(let i=1;i<nx-1;i+=(a===0?1:2)){
      const n=(k*ny+j)*nx+i,v=values[n];if(v<-300||v>2500)continue;
      const t=a===0?i:a===1?j:k;prof[t]+=Math.abs(values[n+step[a]]-2*v+values[n-step[a]]);cnt[t]++;
    }
    // Detrended log profile: each slice against the median of its neighbourhood, so skull-base/vertex
    // trends do not swamp the periodic kinks left by interpolation between thick slices.
    const lp=[];for(let t=0;t<len;t++)lp.push(cnt[t]>50?Math.log1p(prof[t]/cnt[t]):NaN);
    const det=lp.map((x,t)=>{if(!Number.isFinite(x))return NaN;const w=lp.slice(Math.max(0,t-4),t+5).filter(Number.isFinite).sort((p,q)=>p-q);return x-w[w.length>>1]});
    let best=1,bestScore=0;const scores={};
    for(let lag=2;lag<=8;lag++){
      let top=-Infinity;
      for(let ph=0;ph<lag;ph++){let on=0,no=0,off=0,nf=0;det.forEach((x,t)=>{if(!Number.isFinite(x))return;if(t%lag===ph){on+=x;no++}else{off+=x;nf++}});
        if(no>=4&&nf)top=Math.max(top,on/no-off/nf)}
      scores[lag]=top;
    }
    const max=Math.max(...Object.values(scores));
    if(max>Math.log(3))for(let lag=2;lag<=8;lag++)if(scores[lag]>=.8*max){best=lag;bestScore=scores[lag];break}
    out.push({period:best,score:+bestScore.toFixed(3)});
  }
  return out;
}
function gaussianKernel(sigma){if(sigma<.3)return [1];const r=Math.ceil(sigma*2.5),k=[];let s=0;for(let x=-r;x<=r;x++){const w=Math.exp(-x*x/(2*sigma*sigma));k.push(w);s+=w}return k.map(w=>w/s)}
function blurAxis(src,dst,dims,axis,kernel){
  const [nx,ny,nz]=dims,st=[1,nx,nx*ny][axis],len=dims[axis],r=(kernel.length-1)/2;
  const [o1,o2]=[[1,2],[0,2],[0,1]][axis],n1=dims[o1],n2=dims[o2],s1=[1,nx,nx*ny][o1],s2=[1,nx,nx*ny][o2],line=new Float32Array(len);
  for(let b=0;b<n2;b++)for(let a=0;a<n1;a++){
    const base=a*s1+b*s2;for(let t=0;t<len;t++)line[t]=src[base+t*st];
    for(let t=0;t<len;t++){let acc=0;for(let q=-r;q<=r;q++){const u=Math.min(len-1,Math.max(0,t+q));acc+=line[u]*kernel[q+r]}dst[base+t*st]=acc}
  }
}
function smoothVolume(values,dims,sigmas){
  let a=values,b=new Float32Array(values.length);
  for(let axis=0;axis<3;axis++){const k=gaussianKernel(sigmas[axis]);if(k.length===1)continue;blurAxis(a,b,dims,axis,k);[a,b]=[b,a]}
  return a;
}
// Two-pass chamfer distance (voxels) from outside the mask, capped; used for the translucent skin shell.
function depthInside(mask,dims,cap=30,edgeOutside=true){
  const [nx,ny,nz]=dims,d=new Float32Array(mask.length);for(let n=0;n<mask.length;n++)d[n]=mask[n]?cap:0;
  const off=[];for(let dk=-1;dk<=1;dk++)for(let dj=-1;dj<=1;dj++)for(let di=-1;di<=1;di++){if(!di&&!dj&&!dk)continue;off.push([di,dj,dk,Math.hypot(di,dj,dk)])}
  const fwd=off.filter(([di,dj,dk])=>dk<0||(dk===0&&(dj<0||(dj===0&&di<0)))),bwd=off.filter(o=>!fwd.includes(o));
  const pass=(ks,js,is,list)=>{for(const k of ks)for(const j of js)for(const i of is){const n=(k*ny+j)*nx+i;if(!d[n])continue;let v=d[n];
    for(const [di,dj,dk,w] of list){const a=i+di,b=j+dj,c=k+dk;if(a<0||b<0||c<0||a>=nx||b>=ny||c>=nz){if(edgeOutside)v=Math.min(v,w);continue}const q=d[(c*ny+b)*nx+a]+w;if(q<v)v=q}d[n]=v}};
  const rng=(n,rev)=>{const r=[];for(let x=0;x<n;x++)r.push(x);return rev?r.reverse():r};
  pass(rng(nz),rng(ny),rng(nx),fwd);pass(rng(nz,true),rng(ny,true),rng(nx,true),bwd);return d;
}
// --- render copy -------------------------------------------------------------------------------------
const renderCache=new Map();
// Each entry holds one int16 copy of the grid inside NiiVue. Small grids keep four (fast preset switching);
// past RENDER_CACHE_BYTES a single entry stays, so old + new never coexist.
const RENDER_CACHE_BYTES=64e6;
function trimRenderCache(keep,total){
  const max=Math.max(1,Math.min(4,Math.floor(RENDER_CACHE_BYTES/(total*2))));
  for(const k of [...renderCache.keys()])if(k!==keep&&(k.split('|')[0]!==state.base||(keep===null?renderCache.size>=max:renderCache.size>max)))renderCache.delete(k);
}
function renderSpec(){
  if(!state.base||!state.preset3d)return null;
  const vol=state.volumes.get(state.base),meta=vol.meta,p=PRESETS_3D[state.preset3d],kind=isCT(meta)?'CT':'MR';
  const preset=p&&(p.kind===kind||p.kind==='any')?p:PRESETS_3D[default3d(meta)];
  if(!preset)return null;
  const mask=preset.mask?renderMaskFor(state.base,preset.mask):null;
  const head=preset.marginMm?renderMaskFor(state.base,'head'):null;
  const vessels=preset.vessels?renderMaskFor(state.base,'vessels'):null;
  return {vol,meta,preset,mask,head,vessels,key:`${state.base}|${state.preset3d}|${mask?.id||'-'}|${vessels?.id||'-'}`};
}
async function renderImage(){
  const spec=renderSpec();if(!spec)return null;
  if(renderCache.has(spec.key))return renderCache.get(spec.key);
  const {vol,meta,preset,mask,head,vessels}=spec,dims=state.manifest.grid.dims,total=vol.data.length;
  // Big grids keep only the current render copy: drop the others before building the next one.
  trimRenderCache(null,total);
  // Voxels outside the render mask take the volume minimum (shared contract; 0 would be water on CT).
  let values=new Float32Array(total),bg=Infinity;for(let n=0;n<total;n++){const v=vol.data[n]*meta.slope+meta.intercept;values[n]=v;if(v<bg)bg=v}
  const m=mask?state.masks.get(mask.id).data:null;
  if(preset.shellMm){
    // Skin shell: soft tissue deeper than shellMm below the head surface is removed; bone stays.
    const head=m||Uint8Array.from(values,v=>v>-300?1:0),depth=depthInside(head,dims,preset.shellMm+2);
    for(let n=0;n<total;n++)if(values[n]<preset.boneHU&&depth[n]>preset.shellMm)values[n]=bg;
  }
  let keep=m;const w=autoWindow(meta);
  // With a pack-time vessel mask the vessels are drawn as a surface mesh (vesselMesh); the volume keeps only the
  // brain, in the plain brain colours, so Limiar/Teto peel the cortex away from the vessels.
  if(m&&vessels){/* brain only */}
  else if(m&&preset.marginMm){
    // Shell outside the mask: chamfer distance from the mask, in voxels of the finest axis; volume edges are
    // not mask. Limited to the head mask when there is one, so air never enters the shell.
    const vox=Math.min(...state.manifest.grid.spacing_mm),r=preset.marginMm/vox,hm=head?state.masks.get(head.id).data:null;
    // Only voxels brighter than the brain's own top percentile stay in the shell (enhancing vessels); the rest
    // of the meninges and CSF would otherwise wrap the gyri in a flat grey sheet.
    const inside=[];for(let n=0;n<total;n+=7)if(m[n])inside.push(values[n]);inside.sort((a,b)=>a-b);
    const thr=inside.length?inside[Math.floor(inside.length*(preset.shellPct??.97))]:-Infinity;
    const d=depthInside(Uint8Array.from(m,x=>x?0:1),dims,r+2,false);keep=new Uint8Array(total);
    for(let n=0;n<total;n++)keep[n]=m[n]||(d[n]<=r&&values[n]>=thr&&(!hm||hm[n]))?1:0;
  }
  if(keep)for(let n=0;n<total;n++)if(!keep[n])values[n]=bg;
  let info=null;
  if(preset.smooth){
    info=sliceAxisEstimate(values,dims);
    const sig=info.map(a=>a.period>1?a.period*.5:(preset.baseSigma??1));values=smoothVolume(values,dims,sig);info.sigmas=sig;
  }
  const lo=preset.auto?w.center-w.width/2:preset.min,hi=preset.auto?w.center+w.width/2:preset.max;
  const scale=preset.auto?Math.max(1e-6,(hi-lo)/30000):1,inter=preset.auto?lo:0;
  const data=new Int16Array(total);for(let n=0;n<total;n++)data[n]=Math.max(-32768,Math.min(32767,Math.round((values[n]-inter)/scale)));
  const img=await niivue.NVImage.new(niftiBytes(state.manifest.grid,'int16',data,{slope:scale,inter,calMin:lo,calMax:hi}),`render-${state.base}.nii`,'gray',1,null,lo,hi);
  img.colormap=preset.colormap;img.cal_min=lo;img.cal_max=hi;img.opacity=preset.surfaceOnly?0:1;
  const entry={img,key:spec.key,mask:mask?.id||null,preset:state.preset3d,smoothing:info,lo,hi,colormap:vessels&&preset.vesselColormap||preset.colormap,vessels:vessels?.id||null};
  renderCache.set(spec.key,entry);trimRenderCache(spec.key,total);return entry;
}
let renderPending=Promise.resolve();
function updateRender(){
  renderPending=renderPending.then(async()=>{
    const nv=state.nv3d;if(!nv||!state.base)return;
    if(!state.preset3d){state.render3d=null;placeRender(null);return}
    const entry=await renderImage();if(!entry){state.render3d=null;placeRender(null);return}
    const img=entry.img,p=PRESETS_3D[entry.preset]||{};
    img.colormap=entry.colormap||p.colormap||'gray';const w3=state.window3d;img.cal_min=w3?w3.min:entry.lo;img.cal_max=w3?w3.max:entry.hi;
    if(nv.volumes[0]!==img){nv.volumes=[img];nv.back=img;nv.overlays=[]}
    state.render3d={mask:entry.mask,preset:entry.preset,smoothing:entry.smoothing&&entry.smoothing.map(a=>a.period),sigmas:entry.smoothing?.sigmas||null};
    placeRender(state.tileLayout?.render||null);nv.updateGLVolume();syncCrosshair3d();applyMeshes3d();renderWindow3d();
  }).catch(e=>console.warn(e));
  return renderPending;
}
// --- 3D window ---------------------------------------------------------------------------------------
// Moves only the render copy's cal_min/cal_max: every render colormap ramps its opacity across that range, so
// raising the threshold peels the dimmer tissue off in depth order. Sliders span the preset's range widened by
// half of it on each side; switching preset or base returns to the preset's own range.
function window3dSpan(){const spec=renderSpec();if(!spec)return null;const e=renderCache.get(spec.key);if(!e)return null;const w=e.hi-e.lo;return {lo:e.lo,hi:e.hi,min:e.lo-w/2,max:e.hi+w/2}}
function setWindow3d(min,max){
  const s=window3dSpan();if(!s)throw Error('3D não pronto');
  min=clamp(Number(min),s.min,s.max);max=clamp(Number(max),s.min,s.max);if(max-min<(s.hi-s.lo)/100)max=min+(s.hi-s.lo)/100;
  state.window3d={min,max};const img=state.nv3d?.volumes[0];
  if(img){img.cal_min=min;img.cal_max=max;state.nv3d.updateGLVolume()}
  renderWindow3d();return state.window3d;
}
function setWindow3dFromSliders(){
  const s=window3dSpan();if(!s)return;const at=id=>s.min+(s.max-s.min)*Number($(id).value)/1000;
  setWindow3d(at('window3d-min'),at('window3d-max'));
}
function renderWindow3d(){
  const s=window3dSpan(),a=$('window3d-min'),b=$('window3d-max');if(!s||!a||!b)return;
  const w=state.window3d||{min:s.lo,max:s.hi},pos=v=>String(Math.round(1000*(v-s.min)/(s.max-s.min)));
  a.value=pos(w.min);b.value=pos(w.max);
}
// --- lesion meshes (3D) ------------------------------------------------------------------------------
// Naive surface nets on a lightly smoothed copy of the binary mask: solid colour in 3D, shown through the
// volume by NiiVue's mesh x-ray pass. Cached per mask until it is edited.
const lesionMeshes=new Map();
function maskSurface(data,dims,iso=.5){
  const [nx,ny,nz]=dims;let x0=nx,y0=ny,z0=nz,x1=-1,y1=-1,z1=-1;
  for(let k=0;k<nz;k++)for(let j=0;j<ny;j++){const row=(k*ny+j)*nx;for(let i=0;i<nx;i++)if(data[row+i]){if(i<x0)x0=i;if(i>x1)x1=i;if(j<y0)y0=j;if(j>y1)y1=j;if(k<z0)z0=k;if(k>z1)z1=k}}
  if(x1<0)return null;
  x0-=2;y0-=2;z0-=2;x1+=2;y1+=2;z1+=2;
  const W=x1-x0+1,H=y1-y0+1,D=z1-z0+1,f=new Float32Array(W*H*D);
  const at=(i,j,k)=>i<0||j<0||k<0||i>=nx||j>=ny||k>=nz?0:data[(k*ny+j)*nx+i]?1:0;
  for(let k=0;k<D;k++)for(let j=0;j<H;j++)for(let i=0;i<W;i++)f[(k*H+j)*W+i]=at(i+x0,j+y0,k+z0);
  const g=smoothVolume(f,[W,H,D],[.8,.8,.8]),vid=new Int32Array(W*H*D).fill(-1),pts=[];
  const G=(i,j,k)=>g[(k*H+j)*W+i];
  const corners=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]],edges=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
  for(let k=0;k<D-1;k++)for(let j=0;j<H-1;j++)for(let i=0;i<W-1;i++){
    const v=corners.map(([a,b,c])=>G(i+a,j+b,k+c));let inside=0;for(const x of v)if(x>iso)inside++;
    if(!inside||inside===8)continue;
    let sx=0,sy=0,sz=0,n=0;
    for(const [a,b] of edges){const va=v[a],vb=v[b];if((va>iso)===(vb>iso))continue;const t=(iso-va)/(vb-va),A=corners[a],B=corners[b];
      sx+=A[0]+t*(B[0]-A[0]);sy+=A[1]+t*(B[1]-A[1]);sz+=A[2]+t*(B[2]-A[2]);n++}
    vid[(k*H+j)*W+i]=pts.length/3;const p=affinePoint(i+x0+sx/n,j+y0+sy/n,k+z0+sz/n);pts.push(p[0],p[1],p[2]);
  }
  const tris=[],cell=(i,j,k)=>i<0||j<0||k<0||i>=W-1||j>=H-1||k>=D-1?-1:vid[(k*H+j)*W+i];
  for(let k=0;k<D;k++)for(let j=0;j<H;j++)for(let i=0;i<W;i++){
    const inP=G(i,j,k)>iso;
    for(let a=0;a<3;a++){
      const q=[i,j,k];q[a]++;if(q[0]>=W||q[1]>=H||q[2]>=D)continue;if(inP===(G(...q)>iso))continue;
      const b=(a+1)%3,c=(a+2)%3,p0=[i,j,k],p1=[i,j,k],p2=[i,j,k],p3=[i,j,k];p1[b]--;p2[b]--;p2[c]--;p3[c]--;
      const q0=cell(...p0),q1=cell(...p1),q2=cell(...p2),q3=cell(...p3);if(q0<0||q1<0||q2<0||q3<0)continue;
      if(inP)tris.push(q0,q1,q2,q0,q2,q3);else tris.push(q0,q2,q1,q0,q3,q2);
    }
  }
  // Orient outward whatever the affine's handedness: positive signed volume.
  let vol6=0;for(let t=0;t<tris.length;t+=3){const a=tris[t]*3,b=tris[t+1]*3,c=tris[t+2]*3;
    vol6+=pts[a]*(pts[b+1]*pts[c+2]-pts[b+2]*pts[c+1])-pts[a+1]*(pts[b]*pts[c+2]-pts[b+2]*pts[c])+pts[a+2]*(pts[b]*pts[c+1]-pts[b+1]*pts[c])}
  if(vol6<0)for(let t=0;t<tris.length;t+=3){const x=tris[t+1];tris[t+1]=tris[t+2];tris[t+2]=x}
  return {pts:new Float32Array(pts),tris:new Uint32Array(tris)};
}
// NiiVue's default Phong shader darkens saturated lesion colours to brown; Diffuse keeps them true.
function surfaceShader(mesh){const i=state.nv3d.meshShaders.findIndex(s=>s.Name==='Diffuse');if(i>=0)mesh.meshShaderIndex=i;return mesh}
function lesionMesh(meta){
  const m=state.masks.get(meta.id),cached=lesionMeshes.get(meta.id);
  if(cached&&cached.version===m.version)return cached.mesh;
  const surf=maskSurface(m.data,state.manifest.grid.dims);let mesh=null;
  if(surf&&surf.tris.length){const [r,g,b]=hexToRgb(meta.color);mesh=new niivue.NVMesh(surf.pts,surf.tris,`lesion-${meta.id}`,new Uint8Array([r,g,b,255]),1,true,state.nv3d.gl);surfaceShader(mesh)}
  lesionMeshes.set(meta.id,{version:m.version,mesh});return mesh;
}
const VESSEL_COLOR='#3A60D6';
// Active only in a vessels preset whose base volume has a pack-time vessel render mask.
function vesselMesh(){
  const p=PRESETS_3D[state.preset3d];if(!p?.vessels||!state.base)return null;
  const meta=renderMaskFor(state.base,'vessels');if(!meta)return null;
  const m=state.masks.get(meta.id),cached=lesionMeshes.get(meta.id);
  if(cached&&cached.version===m.version)return cached.mesh;
  const surf=maskSurface(m.data,state.manifest.grid.dims,.35);let mesh=null;
  if(surf&&surf.tris.length){const [r,g,b]=hexToRgb(meta.color||VESSEL_COLOR);mesh=new niivue.NVMesh(surf.pts,surf.tris,`vessels-${meta.id}`,new Uint8Array([r,g,b,255]),1,true,state.nv3d.gl);surfaceShader(mesh)}
  lesionMeshes.set(meta.id,{version:m.version,mesh});return mesh;
}
function applyMeshes3d(){
  const nv=state.nv3d;if(!nv)return;
  const show=new Set(visibleTracts().map(t=>t.id)),list=[];
  for(const [id,t] of state.tracts)if(show.has(id)){
    if(t.mesh3d&&t.displayCount>0)list.push(t.mesh3d);
    if(state.mode==='surgeon'&&state.showOutliers&&t.outlierMesh3d)list.push(t.outlierMesh3d);
  }
  for(const meta of visibleMasks()){if(meta.id===state.editingMask)continue;const mesh=lesionMesh(meta);if(mesh)list.push(mesh)}
  const vm=vesselMesh();if(vm)list.push(vm);
  if(typeof anatomyMeshList==='function')list.push(...anatomyMeshList());
  if(typeof trajectoryMeshList==='function')list.push(...trajectoryMeshList());
  for(const m of list)m.visible=true;
  nv.meshes=list;nv.opts.meshXRay=Math.max(state.tractXray,vm?PRESETS_3D[state.preset3d].meshXray||0:0);nv.updateGLVolume();
  // The 3D pivot follows the scene extents (meshes included): keep the surgeon's target centred.
  if(typeof recentreSurgeonView==='function')recentreSurgeonView();
}
function syncCrosshair3d(){
  const nv=state.nv3d;if(!nv?.back)return;
  const f=nv.mm2frac(state.crosshair);nv.scene.crosshairPos=[clamp(f[0],0,1),clamp(f[1],0,1),clamp(f[2],0,1)];
  if(state.clip)applyClip(false);nv.drawScene();
}
