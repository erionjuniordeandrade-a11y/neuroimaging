// Measurement models and direct canvas gestures.
function addAnnotation(type,points,label=''){
  if(state.mode==='patient')throw Error('Ferramentas indisponíveis no modo paciente');
  if(typeof type==='object'){const item=type;type=item.type;points=item.points_ras;label=item.label||''}
  if(!['distance','angle','point','trajectory','roi'].includes(type))throw Error('Tipo de anotação inválido');
  const p=points.map(q=>q.map(Number)),distance=(a,b)=>norm(a.map((v,i)=>v-b[i]));let value=null,units='';
  if(type==='distance'||type==='trajectory'){value=distance(p[0],p[1]);units='mm'}
  if(type==='angle'){const a=p[0].map((v,i)=>v-p[1][i]),b=p[2].map((v,i)=>v-p[1][i]);value=Math.acos(clamp(dot(a,b)/(norm(a)*norm(b)),-1,1))*180/Math.PI;units='°'}
  if(type==='roi'){const center=p[0],radius=distance(p[0],p[1]),vol=state.volumes.get(state.base),q=rasToVoxel(...center),spacing=state.manifest.grid.spacing_mm,rr=Math.ceil(radius/Math.min(...spacing)),samples=[],axis=state.activeView==='axial'?2:state.activeView==='coronal'?1:0,axes=[0,1,2].filter(n=>n!==axis);for(let u=Math.floor(q[axes[0]]-rr);u<=q[axes[0]]+rr;u++)for(let v=Math.floor(q[axes[1]]-rr);v<=q[axes[1]]+rr;v++){const ijk=q.map(Math.round);ijk[axes[0]]=u;ijk[axes[1]]=v;const pos=affinePoint(...ijk);if(distance(center,pos)<=radius){const scalar=scalarAt(vol,...ijk);if(Number.isFinite(scalar))samples.push(scalar)}}const mean=samples.reduce((a,b)=>a+b,0)/samples.length,sd=Math.sqrt(samples.reduce((a,b)=>a+(b-mean)**2,0)/samples.length);value={mean:Number(mean.toFixed(2)),sd:Number(sd.toFixed(2))};units=vol.meta.units||'a.u.'}
  const item={id:'annotation-'+crypto.randomUUID(),type:type==='roi'?'point':type,points_ras:p,label:label||({distance:'Distância',angle:'Ângulo',point:'Ponto',trajectory:'Trajetória',roi:'ROI'}[type]),value:typeof value==='number'?Number(value.toFixed(2)):value,units};if(type==='roi')item.subtype='roi';
  state.manifest.annotations.push(item);renderAnnotationList();renderAll();return item;
}
function renderReadout(p=state.crosshair){
  if(state.mode==='patient'){$('readout').textContent='';return}
  const ijk=rasToVoxel(...p).map(Math.round),base=state.volumes.get(state.base),value=valueAtRAS(state.base,...p),overlay=state.overlay?state.volumes.get(state.overlay):null,other=overlay?valueAtRAS(state.overlay,...p):null;
  $('readout').textContent=`RAS ${p.map(x=>x.toFixed(1)).join(', ')} mm · voxel ${ijk.join(', ')} · ${base.meta.label}: ${value===null?'—':value.toFixed(1)} ${base.meta.units||''}`+(overlay?` · ${overlay.meta.label}: ${other===null?'—':other.toFixed(1)} ${overlay.meta.units||''}`:'');
}
function renderAnnotationList(){const list=$('annotation-list');list.replaceChildren();for(const item of state.manifest.annotations){const div=document.createElement('div');div.className='chip';const value=typeof item.value==='object'?`${item.value.mean} ± ${item.value.sd}`:item.value;div.textContent=`${item.label}: ${value??''} ${item.units||''}`;list.append(div)}}
function canvasPoint(event,canvas){const rect=canvas.getBoundingClientRect();return [(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height]}
function handleCanvasTool(view,event,canvas){
  const [x,y]=canvasPoint(event,canvas),p=pixelToRAS(view,x,y,canvas),tool=state.tool;state.activeView=view;renderReadout(p);
  if(tool==='crosshair'){setCrosshairRAS(...p);return}
  if(tool==='grow'){growRegionRAS(...p);return}
  if(tool==='brush'||tool==='erase'){brushAtRAS(...p,tool==='erase',Number($('tool-radius').value),view);return}
  state.pending.push(p);const need=tool==='angle'?3:tool==='point'?1:2;
  if(state.pending.length>=need){addAnnotation(tool,state.pending);state.pending=[]}
}
function installCanvasGestures(){
  for(const view of ['axial','coronal','sagittal','three']){
    const canvas=$(view+'-canvas');let drag=null;
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>{
      if(state.mode==='patient')return;canvas.setPointerCapture(e.pointerId);const xy=canvasPoint(e,canvas);
      let kind=view==='three'?'rotate':e.button===2?'window':e.button===1||e.shiftKey?'pan':'select';
      if(view==='axial'&&kind==='select'){const [cx,cy]=rasToPixel(view,state.crosshair,canvas),hx=cx+35*Math.cos(state.oblique*Math.PI/180),hy=cy-35*Math.sin(state.oblique*Math.PI/180);if(Math.hypot(xy[0]-hx,xy[1]-hy)<13)kind='oblique'}
      drag={kind,xy,last:xy};if(kind==='select')handleCanvasTool(view,e,canvas);
    });
    canvas.addEventListener('pointermove',e=>{
      if(state.mode==='patient')return;const xy=canvasPoint(e,canvas);if(view!=='three')renderReadout(pixelToRAS(view,...xy,canvas));
      if(!drag)return;const dx=xy[0]-drag.last[0],dy=xy[1]-drag.last[1];drag.last=xy;
      if(drag.kind==='rotate'){state.camera.yaw+=dx*.009;state.camera.pitch=clamp(state.camera.pitch+dy*.009,-1.5,1.5);render3D()}
      else if(drag.kind==='window'){state.window.center+=dx;state.window.width=clamp(state.window.width+dy*2,1,10000);syncWindow();renderAll()}
      else if(drag.kind==='pan'){const f=planeFrame(view,canvas);state.pan[view][0]-=dx/f.scale;state.pan[view][1]-=dy/f.scale;renderMPR(view)}
      else if(drag.kind==='oblique'){const [cx,cy]=rasToPixel(view,state.crosshair,canvas);state.oblique=clamp(-Math.atan2(xy[1]-cy,xy[0]-cx)*180/Math.PI,-60,60);$('oblique').value=state.oblique;renderAll()}
      else if(drag.kind==='select'&&state.tool==='crosshair')setCrosshairRAS(...pixelToRAS(view,...xy,canvas));
      else if(drag.kind==='select'&&['brush','erase'].includes(state.tool))brushAtRAS(...pixelToRAS(view,...xy,canvas),state.tool==='erase',Number($('tool-radius').value),view);
    });
    canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
    canvas.addEventListener('wheel',e=>{if(state.mode==='patient')return;e.preventDefault();if(view==='three'||e.ctrlKey||e.metaKey||e.altKey){state.zoom[view]=clamp(state.zoom[view]*(e.deltaY<0?1.1:.9),.3,6);renderAll();return}const q=rasToVoxel(...state.crosshair),axis=view==='axial'?2:view==='coronal'?1:0;q[axis]=clamp(Math.round(q[axis])+Math.sign(e.deltaY),0,state.manifest.grid.dims[axis]-1);setCrosshairRAS(...affinePoint(...q))},{passive:false});
  }
}
