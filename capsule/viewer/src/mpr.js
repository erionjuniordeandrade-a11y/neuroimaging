// MPR canvases: pixel coordinates are mapped through RAS before sampling the common grid.
function planeFrame(view,canvas){
  const d=state.manifest.grid.dims,s=state.manifest.grid.spacing_mm,c=affinePoint((d[0]-1)/2,(d[1]-1)/2,(d[2]-1)/2),p=state.crosshair;
  const angle=state.oblique*Math.PI/180,co=Math.cos(angle),si=Math.sin(angle);
  let origin,h,v,extentH,extentV;
  if(view==='axial'){origin=[c[0]+state.pan.axial[0],c[1]+state.pan.axial[1],p[2]];h=[-1,0,0];v=[0,-1,0];extentH=d[0]*s[0];extentV=d[1]*s[1]}
  else if(view==='coronal'){origin=[p[0]+state.pan.coronal[0],p[1]+state.pan.coronal[1],p[2]];h=[-co,si,0];v=[0,0,-1];extentH=d[0]*s[0];extentV=d[2]*s[2]}
  else{origin=[p[0]+state.pan.sagittal[0],p[1]+state.pan.sagittal[1],p[2]];h=[si,-co,0];v=[0,0,-1];extentH=d[1]*s[1];extentV=d[2]*s[2]}
  const scale=Math.min(canvas.width/(extentH*1.12),canvas.height/(extentV*1.12))*state.zoom[view];
  return {origin,h,v,scale};
}
function pixelToRAS(view,x,y,canvas){const f=planeFrame(view,canvas),u=(x-canvas.width/2)/f.scale,w=(y-canvas.height/2)/f.scale;return f.origin.map((n,i)=>n+f.h[i]*u+f.v[i]*w)}
function rasToPixel(view,p,canvas){const f=planeFrame(view,canvas),r=p.map((n,i)=>n-f.origin[i]);return [canvas.width/2+dot(r,f.h)*f.scale,canvas.height/2+dot(r,f.v)*f.scale]}
function sampleSlab(vol,p,view){
  if(state.slab==='thin'||state.slabMm<=1)return valueAtRAS(vol.meta.id,...p);
  const count=Math.min(40,Math.max(1,Math.round(state.slabMm))),axis=view==='axial'?2:view==='coronal'?1:0;
  let lo=Infinity,hi=-Infinity,sum=0,n=0;
  for(let t=0;t<count;t++){const q=p.slice();q[axis]+=(t-(count-1)/2)*(state.slabMm/count);const val=valueAtRAS(vol.meta.id,...q);if(val===null)continue;lo=Math.min(lo,val);hi=Math.max(hi,val);sum+=val;n++}
  return n?(state.slab==='mip'?hi:state.slab==='minip'?lo:sum/n):null;
}
function windowValue(value,center,width){return clamp((value-center)/Math.max(width,1)+.5,0,1)}
function colorMap(g,mode){if(mode==='hot')return [255*clamp(g*1.6,0,1),255*clamp((g-.28)*2,0,1),255*clamp((g-.65)*3,0,1)];if(mode==='cool')return [255*g*.25,255*g*.85,255*g];return [255*g,255*g,255*g]}
function renderMPR(view){
  const canvas=$(view+'-canvas'),box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
  const w=Math.max(1,Math.min(520,Math.round(box.width))),h=Math.max(1,Math.min(420,Math.round(box.height)));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const ctx=canvas.getContext('2d',{willReadFrequently:true}),img=ctx.createImageData(w,h),px=img.data,base=state.volumes.get(state.base),overlay=state.volumes.get(state.overlay),masks=visibleMasks().map(m=>({meta:m,data:state.masks.get(m.id)}));
  const f=planeFrame(view,canvas),a=state.manifest.grid.affine_ras,m=inverseAffine,ox=a[0][3],oy=a[1][3],oz=a[2][3];
  const stepH=f.h.map(n=>n/f.scale),stepV=f.v.map(n=>n/f.scale);
  let row=f.origin.map((n,i)=>n-stepH[i]*w/2-stepV[i]*h/2);
  for(let y=0;y<h;y++){
    let p=row.slice();for(let x=0;x<w;x++){
      const rx=p[0]-ox,ry=p[1]-oy,rz=p[2]-oz;
      const i=m[0]*rx+m[1]*ry+m[2]*rz,j=m[3]*rx+m[4]*ry+m[5]*rz,k=m[6]*rx+m[7]*ry+m[8]*rz,n=voxelToIndex(i,j,k),idx=(y*w+x)*4;
      if(n>=0){
        let value=state.slab==='thin'?base.data[n]*base.meta.slope+base.meta.intercept:sampleSlab(base,p,view);
        let grey=windowValue(value,state.window.center,state.window.width);if(state.invert)grey=1-grey;
        let rgb=[grey*255,grey*255,grey*255];
        if(overlay){value=overlay.data[n]*overlay.meta.slope+overlay.meta.intercept;const st=overlay.meta.stats||{};const g=windowValue(value,((st.p01??0)+(st.p99??1000))/2,Math.max(1,(st.p99??1000)-(st.p01??0))),col=colorMap(g,state.colormap),alpha=state.overlayOpacity;rgb=rgb.map((v,t)=>v*(1-alpha)+col[t]*alpha)}
        for(const mask of masks)if(mask.data[n]){const hex=mask.meta.color||'#E4572E',col=[1,3,5].map(t=>parseInt(hex.slice(t,t+2),16));rgb=rgb.map((v,t)=>v*.25+col[t]*.75)}
        px[idx]=rgb[0];px[idx+1]=rgb[1];px[idx+2]=rgb[2];px[idx+3]=255;
      }else{px[idx]=6;px[idx+1]=10;px[idx+2]=16;px[idx+3]=255}
      p[0]+=stepH[0];p[1]+=stepH[1];p[2]+=stepH[2];
    }row[0]+=stepV[0];row[1]+=stepV[1];row[2]+=stepV[2];
  }
  ctx.putImageData(img,0,0);
  if(state.mode==='surgeon'){
    const [cx,cy]=rasToPixel(view,state.crosshair,canvas);ctx.strokeStyle='#C9A84C88';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.moveTo(0,cy);ctx.lineTo(w,cy);ctx.stroke();ctx.fillStyle='#C9A84C';ctx.beginPath();ctx.arc(cx,cy,3,0,Math.PI*2);ctx.fill();
    if(view==='axial'){const [hx,hy]=rasToPixel(view,state.crosshair,canvas);ctx.fillStyle='#C9A84C';ctx.beginPath();ctx.arc(hx+35*Math.cos(state.oblique*Math.PI/180),hy-35*Math.sin(state.oblique*Math.PI/180),6,0,Math.PI*2);ctx.fill()}
  }
  drawAnnotations(ctx,view,canvas);
}
function drawAnnotations(ctx,view,canvas){
  ctx.lineWidth=2;ctx.font='12px system-ui';
  for(const item of visibleAnnotations()){
    const pts=item.points_ras||[];if(!pts.length)continue;const xy=pts.map(p=>rasToPixel(view,p,canvas));
    ctx.strokeStyle=item.type==='trajectory'?'#64d9e6':'#C9A84C';ctx.fillStyle=ctx.strokeStyle;
    ctx.beginPath();xy.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));if(item.subtype==='roi'&&xy.length>1){ctx.moveTo(xy[0][0]+Math.hypot(xy[1][0]-xy[0][0],xy[1][1]-xy[0][1]),xy[0][1]);ctx.arc(xy[0][0],xy[0][1],Math.hypot(xy[1][0]-xy[0][0],xy[1][1]-xy[0][1]),0,Math.PI*2)}ctx.stroke();
    for(const [x,y] of xy){ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}
    if(state.mode==='surgeon'&&item.label)ctx.fillText(item.label,xy[0][0]+7,xy[0][1]-7);
  }
}
