// Tools map onto NiiVue drag modes (measurement, angle) and its drawing layer (pen, eraser, click-to-segment).
const TOOLS=['crosshair','distance','angle','pen','erase','segment','trajectory','segprompt'];
const TOOL_HINTS={crosshair:'hintCursor',distance:'hintDistance',angle:'hintAngle',pen:'hintPen',erase:'hintErase',segment:'hintSegment',trajectory:'hintTrajectory',segprompt:'hintSegPrompt'};
async function setTool(tool){
  if(!TOOLS.includes(tool))throw Error('Ferramenta inválida');
  if(state.mode==='patient'&&tool!=='crosshair')throw Error(tr('editBlocked'));
  const nv=state.nv,D=niivue.DRAG_MODE,o=nv.opts;state.tool=tool;
  if(tool!=='trajectory'&&state.trajectoryDraft){state.trajectoryDraft=null;nv.drawScene()}
  o.dragModePrimary=tool==='distance'?D.measurement:tool==='angle'?D.angle:D.crosshair;
  o.clickToSegment=tool==='segment';
  if(['pen','erase','segment'].includes(tool)){
    if(!state.editingMask){const meta=lesionMasks().at(-1)||await createMask();editMask(meta.id)}
    o.penSize=Number($('pen-size')?.value||3);
    nv.setPenValue(tool==='erase'?0:1,false);nv.setDrawingEnabled(true);
  }else nv.setDrawingEnabled(false);
  for(const b of document.querySelectorAll('[data-tool]'))b.classList.toggle('active',b.dataset.tool===tool);
  const hint=$('tool-hint');if(hint)hint.textContent=tr(TOOL_HINTS[tool]);
  return tool;
}
function onMeasurement(r){
  const item={id:'ann-'+crypto.randomUUID().slice(0,8),type:'distance',points_ras:[Array.from(r.startMM),Array.from(r.endMM)],label:tr('distance'),value:Number(r.distance.toFixed(2)),units:'mm'};
  state.manifest.annotations.push(item);renderAnnotationList();
}
function onAngle(r){
  const pts=[r.firstLineMM.start,r.firstLineMM.end,r.secondLineMM.end].map(p=>Array.from(p));
  const item={id:'ann-'+crypto.randomUUID().slice(0,8),type:'angle',points_ras:pts,label:tr('angle'),value:Number(r.angle.toFixed(1)),units:'°'};
  state.manifest.annotations.push(item);renderAnnotationList();
}
function renderAnnotationList(){
  const list=$('annotation-list');if(!list)return;list.replaceChildren();
  if(!state.manifest.annotations.length){const p=document.createElement('p');p.className='empty-note';p.textContent=tr('noAnnotations');list.append(p);return}
  for(const [i,a] of state.manifest.annotations.entries()){
    const row=document.createElement('div');row.className='row step-card';const span=document.createElement('span');
    span.textContent=`${a.label||a.type}: ${a.value??''} ${a.units||''}`;
    if(a.type==='trajectory'){row.classList.add('trajectory-row');row.classList.toggle('active',a===activeTrajectory());span.onclick=()=>{state.activeTrajectory=a.id;trajectoryChanged();renderAnnotationList()}}const del=document.createElement('button');del.type='button';del.textContent=tr('delete');
    del.onclick=()=>{state.manifest.annotations.splice(i,1);removeDrawnMeasurement(a);renderAnnotationList()};row.append(span,del);list.append(row);
  }
}
function removeDrawnMeasurement(a){
  const doc=state.nv.document,near=(p,q)=>p&&q&&Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2])<1e-3;
  if(a.type==='distance')doc.completedMeasurements=doc.completedMeasurements.filter(m=>!(near(Array.from(m.startMM),a.points_ras[0])&&near(Array.from(m.endMM),a.points_ras[1])));
  if(a.type==='trajectory')removeTrajectory(a);
  if(a.type==='angle')doc.completedAngles=doc.completedAngles.filter(m=>!near(Array.from(m.firstLineMM.start),a.points_ras[0]));
  state.nv.drawScene();
}
// Patient mode shows no numbers: measurements drawn by NiiVue are stashed and restored.
function hideMeasurements(hide){
  const doc=state.nv.document;
  if(hide&&!state.measurementsStash){state.measurementsStash={m:doc.completedMeasurements,a:doc.completedAngles};doc.completedMeasurements=[];doc.completedAngles=[]}
  else if(!hide&&state.measurementsStash){doc.completedMeasurements=state.measurementsStash.m;doc.completedAngles=state.measurementsStash.a;state.measurementsStash=null}
  state.nv.drawScene();
}
function downloadBlob(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000)}
// PNG of the workspace as seen: the 2D canvas with the 3D canvas composited over its tile.
function composite(){
  const ws=$('workspace').getBoundingClientRect(),out=document.createElement('canvas'),dpr=window.devicePixelRatio||1;
  out.width=Math.round(ws.width*dpr);out.height=Math.round(ws.height*dpr);const ctx=out.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,out.width,out.height);
  for(const nv of [state.nv,state.nv3d]){
    const c=nv.canvas,r=c.getBoundingClientRect();if(getComputedStyle(c.parentElement).visibility==='hidden'||!r.width)continue;
    nv.drawScene();ctx.drawImage(c,(r.left-ws.left)*dpr,(r.top-ws.top)*dpr,r.width*dpr,r.height*dpr);
  }
  return out;
}
function screenshot(){
  return new Promise((resolve,reject)=>{composite().toBlob(blob=>{if(!blob)return reject(Error('PNG indisponível'));const name=`case-capsule-v${state.manifest.version}-${state.layout}.png`;downloadBlob(blob,name);resolve({filename:name,size:blob.size})},'image/png')});
}
