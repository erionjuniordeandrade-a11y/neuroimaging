// Mutable document and linked-view state. The manifest object itself is retained to preserve unknown fields.
const $=id=>document.getElementById(id);
const state={manifest:null,volumes:new Map(),masks:new Map(),gl:null,glData:null,mode:'surgeon',activeView:'axial',maximized:null,crosshair:[0,0,0],base:null,overlay:null,overlayOpacity:.35,colormap:'grey',window:{center:40,width:80},invert:false,slab:'thin',slabMm:1,oblique:0,zoom:{axial:1,coronal:1,sagittal:1,three:1},pan:{axial:[0,0],coronal:[0,0],sagittal:[0,0]},camera:{yaw:.45,pitch:.2},mode3d:'bone',cut3d:'off',planes3d:true,threshold3d:300,opacity3d:.45,tool:'crosshair',pending:[],tourIndex:0,tourMaskFilter:null,tourAnnotationFilter:null,status:'loading'};
let readyResolve,readyReject;
const ready=new Promise((resolve,reject)=>{readyResolve=resolve;readyReject=reject});
const api={setCrosshairRAS,valueAtRAS,growRegionRAS,addAnnotation,setMode,applyTourStep,save,render:renderAll,createMask,brushAtRAS};
window.__capsule={ready,state,api};
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function norm(v){return Math.hypot(...v)}
function status(kind,message){state.status=kind;$('status').classList.toggle('hidden',kind==='ready');$('status').querySelector('h2').textContent=message;$('status-detail').textContent=kind==='loading'?'':message}
function affinePoint(i,j,k){const a=state.manifest.grid.affine_ras;return [a[0][0]*i+a[0][1]*j+a[0][2]*k+a[0][3],a[1][0]*i+a[1][1]*j+a[1][2]*k+a[1][3],a[2][0]*i+a[2][1]*j+a[2][2]*k+a[2][3]]}
function inverse3(m){const [a,b,c,d,e,f,g,h,i]=m,det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);if(Math.abs(det)<1e-10)throw Error('Affine singular');return [(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det,(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det,(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]}
let inverseAffine;
function rasToVoxel(x,y,z){const a=state.manifest.grid.affine_ras,p=[x-a[0][3],y-a[1][3],z-a[2][3]],m=inverseAffine;return [dot(m.slice(0,3),p),dot(m.slice(3,6),p),dot(m.slice(6,9),p)]}
function voxelToIndex(i,j,k){const [nx,ny,nz]=state.manifest.grid.dims;i=Math.round(i);j=Math.round(j);k=Math.round(k);return i<0||j<0||k<0||i>=nx||j>=ny||k>=nz?-1:(k*ny+j)*nx+i}
function indexToVoxel(n){const [nx,ny]=state.manifest.grid.dims;return [n%nx,Math.floor(n/nx)%ny,Math.floor(n/(nx*ny))]}
function scalarAt(vol,i,j,k){const n=voxelToIndex(i,j,k);return n<0?NaN:vol.data[n]*vol.meta.slope+vol.meta.intercept}
function valueAtRAS(volId,x,y,z){const vol=state.volumes.get(volId);if(!vol)return null;const value=scalarAt(vol,...rasToVoxel(x,y,z));return Number.isNaN(value)?null:value}
function setCrosshairRAS(x,y,z){if(!state.manifest)return;const p=rasToVoxel(x,y,z),d=state.manifest.grid.dims;state.crosshair=affinePoint(...p.map((v,i)=>clamp(v,0,d[i]-1)));renderAll();return state.crosshair}
function voxelVolumeML(){const a=state.manifest.grid.affine_ras,m=[a[0][0],a[0][1],a[0][2],a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]];return Math.abs(m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]))/1000}
function visibleMasks(){return state.manifest.masks.filter(m=>state.masks.has(m.id)&&(state.mode!=='patient'||m.reviewed)&&(!state.tourMaskFilter||state.tourMaskFilter.includes(m.id)))}
function visibleAnnotations(){return state.manifest.annotations.filter(a=>state.mode!=='patient'||(state.tourAnnotationFilter||[]).includes(a.id))}
function renderAll(){if(state.status==='error'||!state.manifest?.volumes?.length)return;for(const view of ['axial','coronal','sagittal'])renderMPR(view);render3D();renderReadout()}
