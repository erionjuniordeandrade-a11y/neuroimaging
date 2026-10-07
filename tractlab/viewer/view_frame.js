/** Perspective framing of displayed bounds, retaining the anatomical orbit origin. */
const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
const subtract=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const vector=value=>Array.isArray(value)&&value.length===3&&value.every(Number.isFinite);
const unit=a=>{const n=Math.hypot(...a);if(n<1e-9)throw new Error('Degenerate frame direction');return a.map(v=>v/n);};

export function fitViewFrame({points,target,direction,up,width,height,fov=40,insets={}}){
  if(!points?.length||!points.every(vector)||![target,direction,up].every(vector)
    ||!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0||!Number.isFinite(fov)||fov<=0||fov>=170
    ||!Object.values(insets).every(v=>Number.isFinite(v)&&v>=0))
    throw new Error('Invalid view bounds');
  const forward=unit(direction),right=unit(cross(up,forward)),vertical=cross(forward,right);
  const projected=points.map(p=>{const r=subtract(p,target);return [dot(r,right),dot(r,vertical),dot(r,forward)];});
  const left=insets.left||0,top=insets.top||0,usableWidth=width-left-(insets.right||0),usableHeight=height-top-(insets.bottom||0);
  if(usableWidth<=0||usableHeight<=0)throw new Error('No space for anatomy');
  const tangent=Math.tan(fov*Math.PI/360),aspect=width/height;
  const bounds=distance=>{
    const xs=projected.map(p=>p[0]/((distance-p[2])*tangent*aspect));
    const ys=projected.map(p=>p[1]/((distance-p[2])*tangent));
    return {minX:Math.min(...xs),maxX:Math.max(...xs),minY:Math.min(...ys),maxY:Math.max(...ys)};
  };
  const fits=b=>(b.maxX-b.minX)<=usableWidth/width*2*.94&&(b.maxY-b.minY)<=usableHeight/height*2*.94;
  let lo=Math.max(1,Math.max(...projected.map(p=>p[2]))+1),hi=Math.max(20,lo*2);
  while(!fits(bounds(hi)))hi*=2;
  for(let i=0;i<48;i++){const mid=(lo+hi)/2;if(fits(bounds(mid)))hi=mid;else lo=mid;}
  const distance=Math.max(20,hi),b=bounds(distance);
  const centreX=(left+usableWidth/2)/width*2-1,centreY=1-(top+usableHeight/2)/height*2;
  const offset=[((b.minX+b.maxX)/2-centreX)/2,(centreY-(b.minY+b.maxY)/2)/2];
  return {target:[...target],position:target.map((v,i)=>v+forward[i]*distance),direction:forward,up:vertical,right,
    distance,offset,width,height,fov};
}

export function projectFramePoint(point,frame){
  const r=subtract(point,frame.target),depth=frame.distance-dot(r,frame.direction),tangent=Math.tan(frame.fov*Math.PI/360);
  const x=dot(r,frame.right)/(depth*tangent*frame.width/frame.height)-2*frame.offset[0];
  const y=dot(r,frame.up)/(depth*tangent)+2*frame.offset[1];
  return {x:(x+1)*frame.width/2,y:(1-y)*frame.height/2};
}
