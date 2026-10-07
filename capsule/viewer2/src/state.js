// Mutable viewer state. The manifest object is retained as-is so unknown fields survive a save.
const $=id=>document.getElementById(id);
const state={
  manifest:null,volumes:new Map(),masks:new Map(),tracts:new Map(),nv:null,nv3d:null,render3d:null,
  mode:'surgeon',tractDensityMode:'proportional',layout:'2x2',tileLayout:null,crosshair:[0,0,0],base:null,overlay:null,overlayOpacity:.5,overlayColormap:'hot',
  windows:{},windowLayer:'base',preset3d:null,window3d:null,illumination:.6,clip:null,
  maskVisible:{},anatomy:new Map(),anatomyVisible:{},anatomyOpacity:{},tractVisible:{},tractSlabMm:5,tractXray:0,tool:'crosshair',editingMask:null,
  reviewer:'',blobIntegrity:new Map(),showOutliers:false,
  activeTrajectory:null,trajectoryDraft:null,activeSegPrompt:null,surgeonView:false,trajClip:false,trajClipDepth:null,corridorRadius:5,
  tourIndex:0,tourMaskFilter:null,tourTractFilter:null,measurementsStash:null,status:'loading'
};
let readyResolve,readyReject;
const ready=new Promise((resolve,reject)=>{readyResolve=resolve;readyReject=reject});
const api={
  setCrosshairRAS:(...a)=>setCrosshairRAS(...a),valueAtRAS:(...a)=>valueAtRAS(...a),setMode:(...a)=>setMode(...a),
  setTractVisible:(...a)=>setTractVisible(...a),setTractReviewed:(...a)=>setTractReviewed(...a),setTractDensityMode:(...a)=>setTractDensityMode(...a),
  setMaskVisible:(...a)=>setMaskVisible(...a),setMaskReviewed:(...a)=>setMaskReviewed(...a),
  addTourStep:(...a)=>addTourStep(...a),applyTourStep:(...a)=>applyTourStep(...a),save:(...a)=>save(...a),
  setLayout:(...a)=>setLayout(...a),setBase:(...a)=>setBase(...a),setOverlay:(...a)=>setOverlay(...a),
  setWindow:(...a)=>setWindow(...a),setPreset3d:(...a)=>setPreset3d(...a),setWindow3d:(...a)=>setWindow3d(...a),setClip:(...a)=>setClip(...a),
  setCamera:(...a)=>setCamera(...a),setTool:(...a)=>setTool(...a),createMask:(...a)=>createMask(...a),
  growRegionRAS:(...a)=>growRegionRAS(...a),brushAtRAS:(...a)=>brushAtRAS(...a),editMask:(...a)=>editMask(...a),
  visibleMasks:()=>visibleMasks().map(m=>m.id),visibleTracts:()=>visibleTracts().map(t=>t.id),
  screenshot:(...a)=>screenshot(...a),setTractXray:(...a)=>setTractXray(...a),setShowOutliers:(...a)=>setShowOutliers(...a),render:()=>drawNow(),
  renderReady:()=>renderPending,renderCacheSize:()=>renderCache.size,
  setAnatomyVisible:(...a)=>setAnatomyVisible(...a),setAnatomyGroupVisible:(...a)=>setAnatomyGroupVisible(...a),
  setAnatomyReviewed:(...a)=>setAnatomyReviewed(...a),setAnatomyOpacity:(...a)=>setAnatomyOpacity(...a),
  visibleAnatomy:()=>visibleAnatomy(),anatomyAtRAS:(...a)=>anatomyAtRAS(...a),renderMaskFor:(v,n)=>renderMaskFor(v,n)?.id??null,
  addTrajectory:(...a)=>addTrajectory(...a),surgeonView:(...a)=>surgeonView(...a),setTrajectoryClip:(...a)=>setTrajectoryClip(...a),
  setCorridorRadius:(...a)=>setCorridorRadius(...a),corridorReport:()=>corridorReport(),pickTrajectoryPoint:(...a)=>pickTrajectoryPoint(...a),
  addSegPrompt:(...a)=>addSegPrompt(...a),addSegPoint:(...a)=>addSegPoint(...a),segPrompts:()=>segPrompts(),
  setActiveSegPrompt:(...a)=>setActiveSegPrompt(...a),removeSegPrompt:(...a)=>removeSegPrompt(...a)
};
ready.catch(()=>{});
window.__capsule={ready,state,api};
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function norm(v){return Math.hypot(...v)}
function affinePoint(i,j,k){const a=state.manifest.grid.affine_ras;return [a[0][0]*i+a[0][1]*j+a[0][2]*k+a[0][3],a[1][0]*i+a[1][1]*j+a[1][2]*k+a[1][3],a[2][0]*i+a[2][1]*j+a[2][2]*k+a[2][3]]}
function inverse3(m){const [a,b,c,d,e,f,g,h,i]=m,det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);if(Math.abs(det)<1e-10)throw Error('Affine singular');return [(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det,(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det,(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]}
let inverseAffine;
function rasToVoxel(x,y,z){const a=state.manifest.grid.affine_ras,p=[x-a[0][3],y-a[1][3],z-a[2][3]],m=inverseAffine;return [dot(m.slice(0,3),p),dot(m.slice(3,6),p),dot(m.slice(6,9),p)]}
function voxelToIndex(i,j,k){const [nx,ny,nz]=state.manifest.grid.dims;i=Math.round(i);j=Math.round(j);k=Math.round(k);return i<0||j<0||k<0||i>=nx||j>=ny||k>=nz?-1:(k*ny+j)*nx+i}
function indexToVoxel(n){const [nx,ny]=state.manifest.grid.dims;return [n%nx,Math.floor(n/nx)%ny,Math.floor(n/(nx*ny))]}
function scalarAt(vol,i,j,k){const n=voxelToIndex(i,j,k);return n<0?NaN:vol.data[n]*vol.meta.slope+vol.meta.intercept}
// Values come from the capsule's own decoded arrays (not from GPU textures): the readout is the source of truth.
function valueAtRAS(volId,x,y,z){const vol=state.volumes.get(volId);if(!vol)return null;const value=scalarAt(vol,...rasToVoxel(x,y,z));return Number.isNaN(value)?null:value}
function voxelVolumeML(){const a=state.manifest.grid.affine_ras,m=[a[0][0],a[0][1],a[0][2],a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]];return Math.abs(m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]))/1000}
// Start position: centroid of the base volume's foreground (CT > -300 HU; MR > 10% of p99), so the cursor
// opens inside the anatomy even when the grid extends far beyond it (table, head holder, padding).
function foregroundCenterRAS(vol){
  const [nx,ny,nz]=state.manifest.grid.dims,s=vol.meta.stats||{},ct=vol.meta.kind==='CT';
  const cut=ct?-300:0.1*(s.p99??s.max??0),slope=vol.meta.slope,inter=vol.meta.intercept,d=vol.data;
  let n=0,si=0,sj=0,sk=0;
  for(let k=0;k<nz;k+=2)for(let j=0;j<ny;j+=2){const row=(k*ny+j)*nx;for(let i=0;i<nx;i+=2){if(d[row+i]*slope+inter>cut){n++;si+=i;sj+=j;sk+=k}}}
  const v=n?[si/n,sj/n,sk/n]:[(nx-1)/2,(ny-1)/2,(nz-1)/2];
  return affinePoint(...v.map(Math.round));
}
const SHA256_HEX=/^[0-9a-f]{64}$/;
// WebCrypto can be missing in in-app viewers (non-secure context): the capsule still opens, reviews just can't be verified.
async function sha256Hex(bytes){
  if(!globalThis.crypto?.subtle)return null;
  const digest=await crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('');
}
async function registerBlobIntegrity(meta,bytes){
  const actual=await sha256Hex(bytes);
  if(actual===null){state.blobIntegrity.set(meta.blob,false);return null}
  if(!meta.blob_sha256)meta.blob_sha256=actual;
  state.blobIntegrity.set(meta.blob,meta.blob_sha256===actual);
  return actual;
}
function isValidReview(meta){
  const review=meta?.review;
  if(meta?.reviewed!==true||!review||!SHA256_HEX.test(meta.blob_sha256||'')||
     state.blobIntegrity.get(meta.blob)!==true||review.blob_sha256!==meta.blob_sha256||
     !SHA256_HEX.test(review.blob_sha256||'')||typeof review.by!=='string'||!review.by.trim()||
     typeof review.at_utc!=='string')return false;
  const timestamp=Date.parse(review.at_utc);
  if(!Number.isFinite(timestamp))return false;
  try{return new Date(timestamp).toISOString()===review.at_utc}catch{return false}
}
function reviewStatus(meta){
  if(isValidReview(meta))return 'signed';
  if(meta?.reviewed===true&&!meta.review)return 'unsigned';
  return 'unreviewed';
}
async function setItemReview(meta,reviewed,bytes){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const reviewer=state.reviewer.trim();
  if(reviewed&&!reviewer)throw Error(tr('reviewerRequired'));
  const actual=await sha256Hex(bytes);
  if(actual===null)throw Error('Revisão indisponível neste navegador (sem WebCrypto)');
  meta.blob_sha256=actual;state.blobIntegrity.set(meta.blob,true);
  meta.reviewed=!!reviewed;
  if(reviewed)meta.review={by:reviewer,at_utc:new Date().toISOString(),blob_sha256:actual};
  else delete meta.review;
  return isValidReview(meta);
}
function invalidateReview(meta){
  if(!meta)return;
  state.blobIntegrity.set(meta.blob,false);meta.reviewed=false;delete meta.review;
}
function patientHides(meta){return state.mode==='patient'&&!isValidReview(meta)}
function visibleMasks(){return state.manifest.masks.filter(m=>m.role!=='render'&&state.masks.has(m.id)&&state.maskVisible[m.id]!==false&&!patientHides(m)&&(!state.tourMaskFilter||state.tourMaskFilter.includes(m.id)))}
function tractIsVisible(meta){return state.tractVisible[meta.id]??(meta.trust?.verdict!=='FAIL')}
function visibleTracts(){return (state.manifest.tracts||[]).filter(t=>state.tracts.has(t.id)&&tractIsVisible(t)&&!patientHides(t)&&(!state.tourTractFilter||state.tourTractFilter.includes(t.id)))}
function visibleVolumeIds(){return [state.base,state.overlay].filter((id,i,a)=>id&&a.indexOf(id)===i)}
function setVersionBadge(n){const b=$('version-badge');b.textContent=`${tr('capsuleVersion')} ${n}`;b.title=tr('capsuleVersionTitle')}
function status(kind,message,detail=''){
  state.status=kind;const box=$('status');box.classList.toggle('hidden',kind==='ready');box.classList.toggle('error',kind==='error');
  box.querySelector('h2').textContent=message;$('status-detail').textContent=detail;
}
