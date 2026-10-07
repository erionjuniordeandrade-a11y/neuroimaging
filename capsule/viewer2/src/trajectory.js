// Trajectory planning: annotation type "trajectory" (points_ras [entry, target], optional corridor_radius_mm).
// 2D: drawn on a transparent canvas over #gl in every slice tile, projected with NiiVue's own tile mapping
// (frac2canvasPosWithTile), solid where the path lies within 1.5 mm of the shown plane, dashed elsewhere.
// 3D: a gold tube. "Visão do cirurgião" puts the 3D camera beyond the entry looking down entry -> target, with
// the target centred, and optionally clips perpendicular to the path at a chosen depth from the entry.
// The corridor report is computed, never saved: lesion and vessel masks, anatomy labels, and loaded tracts,
// visible or not, with geometry within R mm of the segment.
const TRAJ_COLOR='#C9A84C',TRAJ_TUBE_MM=1,TRAJ_IN_PLANE_MM=1.5,TRAJ_SAMPLE_MM=.5;
function trajectories(){return (state.manifest?.annotations||[]).filter(a=>a.type==='trajectory')}
function activeTrajectory(){const all=trajectories();return all.find(a=>a.id===state.activeTrajectory)||all.at(-1)||null}
function vsub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function vadd(a,b,s=1){return [a[0]+s*b[0],a[1]+s*b[1],a[2]+s*b[2]]}
function trajGeometry(a){const [e,t]=a.points_ras,d=vsub(t,e),L=norm(d);return {entry:e,target:t,L,w:L>0?d.map(x=>x/L):[0,0,1]}}
function trajRadius(a){return clamp(Number(a?.corridor_radius_mm??state.corridorRadius),0,15)}
function addTrajectory(entry,target,options={}){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const e=entry.map(Number),t=target.map(Number);if(e.length!==3||t.length!==3||[...e,...t].some(v=>!Number.isFinite(v)))throw Error('Pontos inválidos');
  const L=norm(vsub(t,e));if(L<1)throw Error('Entrada e alvo coincidem');
  const radius=clamp(Number(options.radius??state.corridorRadius),0,15);
  const item={id:'ann-'+crypto.randomUUID().slice(0,8),type:'trajectory',points_ras:[e,t],label:tr('trajectory'),value:Number(L.toFixed(2)),units:'mm',corridor_radius_mm:radius};
  state.manifest.annotations.push(item);state.activeTrajectory=item.id;state.trajectoryDraft=null;state.trajClipDepth=null;
  trajectoryChanged();renderAnnotationList();return item;
}
function trajectoryChanged(){
  corridorCache=null;applyMeshes3d();if(state.surgeonView)surgeonView(true);else if(state.trajClip)applyClip();
  state.nv?.drawScene();state.nv3d?.drawScene();renderTrajectoryPanel();
}
function removeTrajectory(a){
  if(state.activeTrajectory===a.id)state.activeTrajectory=null;
  const mesh=trajMeshes.get(a.id);if(mesh)trajMeshes.delete(a.id);
  if(!activeTrajectory()){state.surgeonView=false;state.trajClip=false;if(state.nv3d)state.nv3d.position=null;applyClip()}
  trajectoryChanged();
}
function setCorridorRadius(mm){
  const a=activeTrajectory(),r=clamp(Number(mm),0,15);state.corridorRadius=r;if(a)a.corridor_radius_mm=r;
  corridorCache=null;renderTrajectoryPanel();return r;
}
// --- 2D ------------------------------------------------------------------------------------------
function trajOverlay(){
  let c=$('traj-overlay');
  if(!c){c=document.createElement('canvas');c.id='traj-overlay';c.setAttribute('aria-hidden','true');$('gl-wrap').append(c)}
  return c;
}
function initTrajectory(){
  const nv=state.nv,core=nv.drawScene.bind(nv);trajOverlay();
  nv.drawScene=function(){const r=core();drawTrajectoryOverlay();return r};
  let down=null;const gl=$('gl');
  gl.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY]});
  gl.addEventListener('pointerup',e=>{
    if(state.tool!=='trajectory'||!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;
    const r=gl.getBoundingClientRect(),dpr=gl.width/Math.max(1,r.width);
    pickTrajectoryPoint((e.clientX-r.left)*dpr,(e.clientY-r.top)*dpr);
  });
}
function canvasToMm(x,y){const f=state.nv.canvasPos2frac([x,y]);if(!f||f[0]<0)return null;const m=state.nv.frac2mm(f);return [m[0],m[1],m[2]]}
function pickTrajectoryPoint(x,y){
  const p=canvasToMm(x,y);if(!p)return null;
  if(!state.trajectoryDraft){state.trajectoryDraft=p;$('tool-hint').textContent=tr('hintTrajectoryTarget');state.nv.drawScene();return {entry:p}}
  const entry=state.trajectoryDraft;state.trajectoryDraft=null;$('tool-hint').textContent=tr('hintTrajectory');
  try{return addTrajectory(entry,p)}catch(err){console.warn(err.message);state.nv.drawScene();return null}
}
// Same mapping as NiiVue's frac2canvasPosWithTile, for any point (not only points in the shown plane).
function sliceProject(s,p){
  const r=s.axCorSag===0?[p[0],p[1],p[2]]:s.axCorSag===1?[p[0],p[2],p[1]]:[p[1],p[2],p[0]];
  const fx=(r[0]-s.leftTopMM[0])/s.fovMM[0],fy=(r[1]-s.leftTopMM[1])/s.fovMM[1],c=s.leftTopWidthHeight.slice();
  let flip=false;if(c[2]<0){flip=true;c[0]+=c[2];c[2]=-c[2]}
  const n=s.AxyzMxy,plane=n&&n.length>=5?n[2]+n[4]*(r[1]-n[1])-n[3]*(r[0]-n[0]):NaN;
  return {x:c[0]+(flip?1-fx:fx)*c[2],y:c[1]+(1-fy)*c[3],off:r[2]-plane};
}
function sliceTiles(){return (state.nv?.screenSlices||[]).filter(s=>s.axCorSag<=2&&s.leftTopWidthHeight&&s.fovMM&&s.AxyzMxy?.length>=4)}
function drawTrajectoryOverlay(){
  const c=$('traj-overlay'),gl=$('gl');if(!c||!gl)return;
  if(c.width!==gl.width||c.height!==gl.height){c.width=gl.width;c.height=gl.height}
  const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
  if(state.mode==='patient'||$('gl-wrap').style.visibility==='hidden')return;
  const dpr=gl.width/Math.max(1,gl.getBoundingClientRect().width)||1,items=[];
  const a=activeTrajectory();for(const t of trajectories())items.push([t,t===a]);
  for(const s of sliceTiles()){
    const lt=s.leftTopWidthHeight,x0=Math.min(lt[0],lt[0]+lt[2]),w=Math.abs(lt[2]);
    ctx.save();ctx.beginPath();ctx.rect(x0,lt[1],w,lt[3]);ctx.clip();
    for(const [t,active] of items){
      const {entry,L,w:dir}=trajGeometry(t),n=Math.max(2,Math.ceil(L/TRAJ_SAMPLE_MM));
      ctx.strokeStyle=TRAJ_COLOR;ctx.globalAlpha=active?1:.55;ctx.lineWidth=2*dpr;ctx.lineCap='round';
      let prev=sliceProject(s,entry);
      for(let k=1;k<=n;k++){
        const q=sliceProject(s,vadd(entry,dir,L*k/n)),inPlane=Math.abs((prev.off+q.off)/2)<=TRAJ_IN_PLANE_MM;
        ctx.setLineDash(inPlane?[]:[5*dpr,4*dpr]);ctx.lineDashOffset=-k*L/n*dpr;
        ctx.beginPath();ctx.moveTo(prev.x,prev.y);ctx.lineTo(q.x,q.y);ctx.stroke();prev=q;
      }
      ctx.setLineDash([]);
      const e=sliceProject(s,entry),g=sliceProject(s,t.points_ras[1]);
      ctx.beginPath();ctx.arc(e.x,e.y,5*dpr,0,2*Math.PI);ctx.stroke();
      ctx.fillStyle=TRAJ_COLOR;ctx.beginPath();ctx.arc(g.x,g.y,4*dpr,0,2*Math.PI);ctx.fill();
    }
    if(state.trajectoryDraft){const e=sliceProject(s,state.trajectoryDraft);ctx.globalAlpha=1;ctx.strokeStyle=TRAJ_COLOR;ctx.lineWidth=2*dpr;ctx.beginPath();ctx.arc(e.x,e.y,5*dpr,0,2*Math.PI);ctx.stroke()}
    ctx.restore();
  }
}
// --- 3D ------------------------------------------------------------------------------------------
const trajMeshes=new Map();
function tubeMesh(a){
  const key=a.id+'|'+a.points_ras.flat().join(',');const hit=trajMeshes.get(a.id);if(hit&&hit.key===key)return hit.mesh;
  const {entry,L,w}=trajGeometry(a),seg=20,rad=TRAJ_TUBE_MM;
  const ref=Math.abs(w[2])<.9?[0,0,1]:[1,0,0],u0=[w[1]*ref[2]-w[2]*ref[1],w[2]*ref[0]-w[0]*ref[2],w[0]*ref[1]-w[1]*ref[0]],un=norm(u0),u=u0.map(x=>x/un);
  const v=[w[1]*u[2]-w[2]*u[1],w[2]*u[0]-w[0]*u[2],w[0]*u[1]-w[1]*u[0]],pts=[],tris=[];
  for(const d of [0,L])for(let i=0;i<seg;i++){const th=2*Math.PI*i/seg,c=Math.cos(th)*rad,s=Math.sin(th)*rad;pts.push(entry[0]+w[0]*d+u[0]*c+v[0]*s,entry[1]+w[1]*d+u[1]*c+v[1]*s,entry[2]+w[2]*d+u[2]*c+v[2]*s)}
  pts.push(...entry,...vadd(entry,w,L));const cE=2*seg,cT=2*seg+1;
  for(let i=0;i<seg;i++){const j=(i+1)%seg;tris.push(i,seg+i,seg+j,i,seg+j,j,cE,j,i,cT,seg+i,seg+j)}
  const [r,g,b]=hexToRgb(TRAJ_COLOR),mesh=new niivue.NVMesh(new Float32Array(pts),new Uint32Array(tris),`trajectory-${a.id}`,new Uint8Array([r,g,b,255]),1,true,state.nv3d.gl);
  surfaceShader(mesh);trajMeshes.set(a.id,{key,mesh});return mesh;
}
// Appended by applyMeshes3d (render3d.js); planning artefacts are not shown in patient mode.
function trajectoryMeshList(){return state.mode==='patient'?[]:trajectories().map(tubeMesh)}
// Camera at azimuth a, elevation e sits in direction (-sin a cos e, -cos a cos e, sin e) from the pivot (view.js).
function surgeonAngles(a){const {entry,target}=trajGeometry(a),d=vsub(entry,target),u=d.map(x=>x/norm(d));
  const el=Math.asin(clamp(u[2],-1,1))*180/Math.PI,az=((Math.atan2(-u[0],-u[1])*180/Math.PI)%360+360)%360;return {azimuth:az,elevation:el}}
function centre3dOn(p){
  // NiiVue's model matrix is mirror * T(0,0,-dist) * T(position) * rotations * T(-pivot): position is an
  // eye-space pan. Solve for the pan that puts p on the view axis.
  const nv=state.nv3d,s=nv.scene;nv.position=null;
  const h=nv.calculateMvpMatrix(null,[0,0,nv.gl.canvas.width,nv.gl.canvas.height],s.renderAzimuth,s.renderElevation)[1];
  const ex=h[0]*p[0]+h[4]*p[1]+h[8]*p[2]+h[12],ey=h[1]*p[0]+h[5]*p[1]+h[9]*p[2]+h[13];
  nv.position=[ex,-ey,0];
}
function surgeonView(on=true){
  const a=activeTrajectory(),nv=state.nv3d;
  if(!on||!a){state.surgeonView=false;nv.position=null;applyClip();nv.drawScene();renderTrajectoryPanel();return null}
  state.surgeonView=true;const {azimuth,elevation}=surgeonAngles(a);
  nv.setRenderAzimuthElevation(azimuth,elevation);centre3dOn(a.points_ras[1]);applyClip(false);nv.drawScene();renderTrajectoryPanel();
  return {azimuth,elevation,position:[...nv.position]};
}
function recentreSurgeonView(){const a=activeTrajectory();if(state.surgeonView&&a&&state.nv3d?.back)centre3dOn(a.points_ras[1])}
// Clip perpendicular to the path at depth d from the entry, cutting away the entry side (the half facing the
// surgeon's camera). NiiVue's plane lives in frac space: n_f ∝ A^T u for frac -> mm Jacobian A.
function trajClipDepth(a){const L=trajGeometry(a).L;return clamp(state.trajClipDepth??L/2,0,L)}
function trajClipPlane(){
  const a=activeTrajectory(),nv=state.nv3d;if(!a||!nv?.back)return null;
  const {entry,w}=trajGeometry(a),o=nv.frac2mm([0,0,0]),col=[[1,0,0],[0,1,0],[0,0,1]].map(e=>vsub(nv.frac2mm(e),o));
  const nf=col.map(c=>-dot(c,w.map(x=>-x))),nn=norm(nf),n=nf.map(x=>x/nn);
  const f0=nv.mm2frac(vadd(entry,w,trajClipDepth(a))),depth=-(n[0]*(f0[0]-.5)+n[1]*(f0[1]-.5)+n[2]*(f0[2]-.5));
  const el=-Math.asin(clamp(n[2],-1,1))*180/Math.PI,az=((Math.atan2(n[1],n[0])*180/Math.PI-90)%360+360)%360;
  return [depth,az,el];
}
function setTrajectoryClip(on,depthMm){
  if(depthMm!==undefined&&depthMm!==null)state.trajClipDepth=Number(depthMm);
  state.trajClip=!!on&&!!activeTrajectory();if(state.trajClip)state.clip=null;
  for(const b of document.querySelectorAll('[data-clip]'))b.classList.toggle('active',(b.dataset.clip||null)===state.clip);
  applyClip();renderTrajectoryPanel();return state.trajClip?trajClipPlane():null;
}
// --- corridor report -----------------------------------------------------------------------------
let corridorCache=null;
const corridorObjectIds=new WeakMap();let corridorObjectSeq=0;
function corridorObjectId(value){
  if(!value||typeof value!=='object')return 0;
  let id=corridorObjectIds.get(value);if(!id){id=++corridorObjectSeq;corridorObjectIds.set(value,id)}return id;
}
function corridorStructures(){
  const out=[];
  for(const meta of lesionMasks()){
    const m=state.masks.get(meta.id);if(!m)continue;
    const reviewed=isValidReview(meta);
    out.push({kind:'mask',id:meta.id,name:meta.label||meta.id,color:meta.color,data:m.data,value:null,meta,
      reviewed,visible:state.maskVisible[meta.id]!==false&&!(state.mode==='patient'&&!reviewed)});
  }
  if(state.mode!=='patient'){
    const preset=PRESETS_3D[state.preset3d],activeVessel=preset?.vessels?renderMaskFor(state.base,'vessels')?.id:null;
    for(const meta of state.manifest?.masks||[]){
      if(meta.role!=='render'||!(meta.id==='vessels'||meta.id.startsWith('vessels')))continue;
      const m=state.masks.get(meta.id);if(!m)continue;
      out.push({kind:'vessels',id:meta.id,name:meta.label||meta.id,color:meta.color,data:m.data,value:null,group:tr('corridorGroupVessels'),meta,
        reviewed:isValidReview(meta),visible:meta.id===activeVessel});
    }
  }
  for(const it of anatomyItems()){
    const e=state.anatomy.get(it.id);if(!e)continue;const reviewed=isValidReview(it);
    for(const l of it.labels||[])out.push({kind:'anatomy',id:anatomyKey(it.id,l.value),name:l.name||l.key,color:l.color,data:e.data,value:l.value,group:l.group,
      meta:it,reviewed,visible:anatomyLabelVisible(it,l)&&!(state.mode==='patient'&&!reviewed)});
  }
  if(state.mode!=='patient')for(const [id,t] of state.tracts){
    if(!t?.streamlines?.length)continue;const meta=t.meta||state.manifest?.tracts?.find(x=>x.id===id);if(!meta)continue;
    const reviewed=isValidReview(meta);
    out.push({kind:'tract',id,name:tractDisplayName(meta),color:meta.color,group:tr('corridorGroupTracts'),meta,entry:t,
      streamlines:t.streamlines,streamlineBounds:t.streamlineBounds,reviewed,
      visible:tractIsVisible(meta)&&!(state.mode==='patient'&&!reviewed)&&(!state.tourTractFilter||state.tourTractFilter.includes(id))});
  }
  return out;
}
function segmentPathClosest(entry,w,L,ax,ay,az,bx,by,bz,out){
  const ux=w[0]*L,uy=w[1]*L,uz=w[2]*L,vx=bx-ax,vy=by-ay,vz=bz-az,rx=entry[0]-ax,ry=entry[1]-ay,rz=entry[2]-az;
  const a=L*L,b=ux*vx+uy*vy+uz*vz,c=vx*vx+vy*vy+vz*vz,d=ux*rx+uy*ry+uz*rz,e=vx*rx+vy*ry+vz*rz;
  if(c<=1e-20)return false;
  const den=a*c-b*b;let sn,sd=den,tn,td=den;
  if(den<1e-20){sn=0;sd=1;tn=e;td=c}
  else{
    sn=b*e-c*d;tn=a*e-b*d;
    if(sn<0){sn=0;tn=e;td=c}
    else if(sn>sd){sn=sd;tn=e+b;td=c}
  }
  if(tn<0){
    tn=0;
    if(-d<0)sn=0;else if(-d>a)sn=sd;else{sn=-d;sd=a}
  }else if(tn>td){
    tn=td;
    if(-d+b<0)sn=0;else if(-d+b>a)sn=sd;else{sn=-d+b;sd=a}
  }
  const s=Math.abs(sn)<1e-12?0:sn/sd,t=Math.abs(tn)<1e-12?0:tn/td;
  const dx=rx+s*ux-t*vx,dy=ry+s*uy-t*vy,dz=rz+s*uz-t*vz;
  out[0]=dx*dx+dy*dy+dz*dz;out[1]=s*L;return true;
}
function pointPathContact(entry,w,L,x,y,z,r2){
  const rx=x-entry[0],ry=y-entry[1],rz=z-entry[2],along=rx*w[0]+ry*w[1]+rz*w[2],depth=clamp(along,0,L);
  const dx=rx-depth*w[0],dy=ry-depth*w[1],dz=rz-depth*w[2];
  if(dx*dx+dy*dy+dz*dz>r2)return Infinity;
  const perp2=Math.max(0,rx*rx+ry*ry+rz*rz-along*along);
  return clamp(along-Math.sqrt(Math.max(0,r2-perp2)),0,L);
}
function segmentPathContact(entry,w,L,ax,ay,az,bx,by,bz,r2){
  let first=Math.min(pointPathContact(entry,w,L,ax,ay,az,r2),pointPathContact(entry,w,L,bx,by,bz,r2));
  const vx=bx-ax,vy=by-ay,vz=bz-az,c=vx*vx+vy*vy+vz*vz;if(c<=1e-20)return first;
  const rx=entry[0]-ax,ry=entry[1]-ay,rz=entry[2]-az,u0=(rx*vx+ry*vy+rz*vz)/c,ur=(w[0]*vx+w[1]*vy+w[2]*vz)/c;
  let lo=0,hi=L;
  if(Math.abs(ur)<1e-14){if(u0<0||u0>1)return first}
  else{
    const t0=-u0/ur,t1=(1-u0)/ur;lo=Math.max(lo,Math.min(t0,t1));hi=Math.min(hi,Math.max(t0,t1));if(lo>hi)return first;
  }
  const px=rx-u0*vx,py=ry-u0*vy,pz=rz-u0*vz,wx=w[0]-ur*vx,wy=w[1]-ur*vy,wz=w[2]-ur*vz;
  const qa=wx*wx+wy*wy+wz*wz,qb=2*(px*wx+py*wy+pz*wz),qc=px*px+py*py+pz*pz-r2;
  if(qa<=1e-24){if(qc<=0)first=Math.min(first,lo)}
  else{
    const disc=qb*qb-4*qa*qc;
    if(disc>=0){const root=Math.sqrt(disc),enter=(-qb-root)/(2*qa),leave=(-qb+root)/(2*qa),candidate=Math.max(lo,enter);
      if(candidate<=Math.min(hi,leave))first=Math.min(first,candidate)}
  }
  return first;
}
function tractCorridorMetrics(s,entry,w,L,R){
  const lines=s.streamlines,bounds=s.streamlineBounds,r2=R*R,closest=[0,0];
  const pathEnd=vadd(entry,w,L),loX=Math.min(entry[0],pathEnd[0])-R,loY=Math.min(entry[1],pathEnd[1])-R,loZ=Math.min(entry[2],pathEnd[2])-R;
  const hiX=Math.max(entry[0],pathEnd[0])+R,hiY=Math.max(entry[1],pathEnd[1])+R,hiZ=Math.max(entry[2],pathEnd[2])+R;
  let best2=Infinity,closestDepth=Infinity,first=Infinity;
  for(let sidx=0;sidx<lines.length;sidx++){
    const line=lines[sidx],bo=sidx*6;
    if(bounds&&(bounds[bo]>hiX||bounds[bo+1]>hiY||bounds[bo+2]>hiZ||bounds[bo+3]<loX||bounds[bo+4]<loY||bounds[bo+5]<loZ))continue;
    for(let i=0;i+2<line.length;i+=3){
      const x=line[i],y=line[i+1],z=line[i+2],rx=x-entry[0],ry=y-entry[1],rz=z-entry[2],along=rx*w[0]+ry*w[1]+rz*w[2],depth=clamp(along,0,L);
      const dx=rx-depth*w[0],dy=ry-depth*w[1],dz=rz-depth*w[2],d2=dx*dx+dy*dy+dz*dz;
      if(d2<best2-1e-14||(Math.abs(d2-best2)<=1e-14&&depth<closestDepth)){best2=d2;closestDepth=depth}
      if(d2<=r2||d2<=1e-12){
        const perp2=Math.max(0,rx*rx+ry*ry+rz*rz-along*along),contact=clamp(along-Math.sqrt(Math.max(0,r2-perp2)),0,L);
        if(contact<first)first=contact;
      }
      if(i+5<line.length){
        const ax=x,ay=y,az=z,bx=line[i+3],by=line[i+4],bz=line[i+5];
        if(segmentPathClosest(entry,w,L,ax,ay,az,bx,by,bz,closest)){
          if(closest[0]<best2-1e-14||(Math.abs(closest[0]-best2)<=1e-14&&closest[1]<closestDepth)){best2=closest[0];closestDepth=closest[1]}
        }
        const contact=segmentPathContact(entry,w,L,ax,ay,az,bx,by,bz,r2);if(contact<first)first=contact;
      }
    }
  }
  const distance=Math.sqrt(best2);if(distance<=1e-6&&!Number.isFinite(first))first=closestDepth;
  return {distance,closestDepth,first};
}
function corridorReport(a=activeTrajectory(),radius){
  if(!a)return [];const R=radius===undefined?trajRadius(a):clamp(Number(radius),0,15);
  const {entry,L,w}=trajGeometry(a),structs=corridorStructures(),S=structs.length,dims=state.manifest.grid.dims,[nx,ny,nz]=dims;
  const structureKey=structs.map(s=>[s.kind,s.id,s.name,s.color,s.group,s.reviewed,s.visible,s.meta?.blob_sha256||'',
    s.kind==='tract'?corridorObjectId(s.entry):0,s.kind==='tract'?corridorObjectId(s.streamlines):0]);
  const key=JSON.stringify([a.id,a.points_ras,R,state.masks.size,state.maskRev||0,state.base,state.preset3d,state.mode,structureKey]);
  if(corridorCache?.key===key)return corridorCache.rows;
  const minD=new Float64Array(S).fill(Infinity),contact=new Float64Array(S).fill(Infinity),pierce=new Float64Array(S).fill(Infinity);
  const byValue=new Map();structs.forEach((s,i)=>{if(s.kind==='anatomy'){const k=s.data;if(!byValue.has(k))byValue.set(k,new Map());byValue.get(k).set(s.value,i)}});
  const masks=structs.map((s,i)=>[s,i]).filter(([s])=>s.kind==='mask'||s.kind==='vessels'),labelMaps=[...byValue.entries()];
  const hits=(n,cb)=>{for(const [s,i] of masks)if(s.data[n])cb(i);for(const [data,map] of labelMaps){const v=data[n];if(v){const i=map.get(v);if(i!==undefined)cb(i)}}};
  // Path samples every <= 0.5 mm: a sample inside a structure's voxel means the path crosses it (distance 0).
  const steps=Math.max(1,Math.ceil(L/TRAJ_SAMPLE_MM));
  const inside=(i,d)=>{const n=voxelToIndex(...rasToVoxel(...vadd(entry,w,d)));let hit=false;if(n>=0)hits(n,x=>{if(x===i)hit=true});return hit};
  for(let k=0;k<=steps;k++){const d=L*k/steps,p=vadd(entry,w,d),n=voxelToIndex(...rasToVoxel(...p));if(n>=0)hits(n,i=>{
    if(d>=pierce[i])return;let lo=Math.max(0,L*(k-1)/steps),hi=d;
    // Refine the crossing to the voxel boundary between the previous sample (outside) and this one.
    if(k>0)for(let it=0;it<12;it++){const mid=(lo+hi)/2;if(inside(i,mid))hi=mid;else lo=mid}
    pierce[i]=hi})}
  // Voxels near the segment: distance of each voxel centre to the segment, and the first path depth at which the
  // corridor (radius R around the path) reaches it.
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const p of [entry,vadd(entry,w,L)])for(const dx of [-R-2,R+2])for(const dy of [-R-2,R+2])for(const dz of [-R-2,R+2]){const v=rasToVoxel(p[0]+dx,p[1]+dy,p[2]+dz);for(let c=0;c<3;c++){lo[c]=Math.min(lo[c],v[c]);hi[c]=Math.max(hi[c],v[c])}}
  const i0=Math.max(0,Math.floor(lo[0])-1),i1=Math.min(nx-1,Math.ceil(hi[0])+1),j0=Math.max(0,Math.floor(lo[1])-1),j1=Math.min(ny-1,Math.ceil(hi[1])+1),k0=Math.max(0,Math.floor(lo[2])-1),k1=Math.min(nz-1,Math.ceil(hi[2])+1);
  // Voxels count by their inscribed ball (centre distance minus half the smallest spacing), so a structure
  // enters the corridor when its voxel faces do, not only its voxel centres (conservative by up to half a voxel).
  const aff=state.manifest.grid.affine_ras,half=.5*Math.min(...[0,1,2].map(c=>Math.hypot(aff[0][c],aff[1][c],aff[2][c]))),Rv=R+half,R2=Rv*Rv;
  for(let k=k0;k<=k1;k++)for(let j=j0;j<=j1;j++){const row=(k*ny+j)*nx;for(let i=i0;i<=i1;i++){
    const n=row+i;let any=false;hits(n,()=>{any=true});if(!any)continue;
    const P=affinePoint(i,j,k),rel=vsub(P,entry),tu=dot(rel,w),tc=clamp(tu,0,L),q=vsub(rel,w.map(x=>x*tc)),d2=dot(q,q);if(d2>R2)continue;
    const d=Math.max(0,Math.sqrt(d2)-half),perp2=Math.max(0,dot(rel,rel)-tu*tu),s=clamp(tu-Math.sqrt(Math.max(0,R2-perp2)),0,L);
    hits(n,x=>{if(d<minD[x])minD[x]=d;if(s<contact[x])contact[x]=s});
  }}
  const rows=[];
  structs.forEach((s,i)=>{
    if(s.kind==='tract'){
      const m=tractCorridorMetrics(s,entry,w,L,R);if(m.distance>R)return;
      const crossing=m.distance<=half?Number(m.closestDepth.toFixed(2)):null;
      rows.push({kind:s.kind,id:s.id,name:s.name,color:s.color,group:s.group??null,
        min_distance_mm:m.distance<=1e-6?0:Number(m.distance.toFixed(2)),
        first_contact_mm:Number((Number.isFinite(m.first)?m.first:m.closestDepth).toFixed(2)),
        crossing_mm:crossing,reviewed:!!s.reviewed,visible:!!s.visible});
      return;
    }
    const crossed=Number.isFinite(pierce[i]);if(!crossed&&!Number.isFinite(minD[i]))return;
    rows.push({kind:s.kind,id:s.id,name:s.name,color:s.color,group:s.group??null,
      min_distance_mm:crossed?0:Number(minD[i].toFixed(2)),
      first_contact_mm:Number(Math.min(contact[i],crossed?pierce[i]:Infinity).toFixed(2)),
      crossing_mm:crossed?Number(pierce[i].toFixed(2)):null,reviewed:!!s.reviewed,visible:!!s.visible});
  });
  rows.sort((p,q)=>p.first_contact_mm-q.first_contact_mm||p.min_distance_mm-q.min_distance_mm);
  corridorCache={key,rows};return rows;
}
// --- panel ---------------------------------------------------------------------------------------
const fmtMm=v=>Number(v).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1});
function renderTrajectoryPanel(){
  const box=$('trajectory-panel');if(!box||!state.manifest)return;
  const a=activeTrajectory();box.hidden=!a;if(!a){$('corridor-report').replaceChildren();return}
  const {L}=trajGeometry(a),R=trajRadius(a);
  $('traj-length').textContent=`${tr('trajectoryLength')}: ${fmtMm(L)} mm`;
  $('surgeon-view').classList.toggle('active',!!state.surgeonView);
  $('traj-clip').checked=!!state.trajClip;const dep=$('traj-depth');dep.max=L.toFixed(1);dep.value=trajClipDepth(a).toFixed(1);dep.disabled=!state.trajClip;
  $('traj-depth-out').textContent=`${fmtMm(trajClipDepth(a))} mm`;
  $('corridor-radius').value=R;$('corridor-radius-out').textContent=`${fmtMm(R)} mm`;
  const list=$('corridor-report');list.replaceChildren();
  const head=document.createElement('p');head.className='hint';head.textContent=tr('corridorHeader').replace('{r}',fmtMm(R));list.append(head);
  const rows=corridorReport(a,R);
  if(!rows.length){const p=document.createElement('p');p.className='empty-note';p.textContent=tr('corridorEmpty');list.append(p);return}
  for(const r of rows){
    const row=document.createElement('div');row.className='corridor-row';row.dataset.id=r.id;
    const sw=document.createElement('span');sw.className='swatch';sw.style.background=r.color;
    const nm=document.createElement('span');nm.className='name';nm.textContent=r.name;nm.title=r.name;
    const label=document.createElement('div');label.className='corridor-label';label.append(nm);
    if(r.group){const group=document.createElement('span');group.className='corridor-group';group.textContent=r.group;label.append(group)}
    if(!r.reviewed){const badge=document.createElement('span');badge.className='badge unreviewed-badge';badge.textContent=tr('notReviewed');label.append(badge)}
    const info=document.createElement('span');info.className='meta';
    info.textContent=`${tr('corridorContact')} ${fmtMm(r.first_contact_mm)} mm · ${r.crossing_mm!==null?`${tr('corridorCrosses')} ${fmtMm(r.crossing_mm)} mm`:`${tr('corridorMin')} ${fmtMm(r.min_distance_mm)} mm`}`;
    row.append(sw,label,info);list.append(row);
  }
}
function wireTrajectory(){
  $('surgeon-view').onclick=()=>surgeonView(!state.surgeonView);
  $('traj-clip').onchange=e=>setTrajectoryClip(e.target.checked);
  $('traj-depth').oninput=e=>{state.trajClipDepth=Number(e.target.value);$('traj-depth-out').textContent=`${fmtMm(state.trajClipDepth)} mm`;applyClip()};
  $('corridor-radius').oninput=e=>setCorridorRadius(e.target.value);
}
