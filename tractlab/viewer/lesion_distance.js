import {Vector3,Ray,DoubleSide} from './vendor/three.module.js';

/** Minimum surface distance along a displayed polyline, in world millimetres. */
export function streamlineLesionDistance(points, bvh, {sample=1}={}) {
  if (!(points instanceof Float32Array) || points.length%3 || !bvh ||
      !Number.isInteger(sample) || sample<1) throw new TypeError('invalid streamline distance input');
  const count=points.length/3;
  if (!count) return {minMm:Infinity,argVertex:-1};
  const point=new Vector3(), hit={};
  let minMm=Infinity,argVertex=-1;
  const measure=i=>{
    point.fromArray(points,i*3);
    const result=bvh.closestPointToPoint(point,hit);
    if (result && result.distance<minMm) {minMm=result.distance;argVertex=i;}
  };
  for(let i=0;i<count;i+=sample)measure(i);
  if((count-1)%sample)measure(count-1);
  // The coarse pass finds a neighbourhood; use every displayed vertex there.
  if(sample>1){
    const start=Math.max(0,argVertex-sample),end=Math.min(count-1,argVertex+sample);
    for(let i=start;i<=end;i++)measure(i);
  }
  if(minMm===0)return {minMm,argVertex};
  // A displayed segment can cross the surface between two sampled vertices.
  // Only segments long enough to reach the current closest surface distance
  // need an intersection query; ordinary short distant segments are skipped.
  const a=new Vector3(),b=new Vector3(),ray=new Ray();
  for(let i=0;i<count-1;i++){
    a.fromArray(points,i*3);b.fromArray(points,(i+1)*3);
    const length=b.distanceTo(a);
    if(length<minMm)continue;
    ray.origin.copy(a);ray.direction.subVectors(b,a).divideScalar(length);
    if(bvh.raycastFirst(ray,DoubleSide,0,length)){
      minMm=0;argVertex=i;break;
    }
  }
  return {minMm,argVertex};
}

/** Yield between short chunks; a replaced layer or lesion abandons its old result. */
export async function layerLesionDistances(layer,bvh,{onProgress=()=>{},sample=1,
  frame=callback=>requestAnimationFrame(callback),now=()=>performance.now(),budgetMs=8}={}) {
  if(layer.lesionDistanceCache?.bvh===bvh && layer.lesionDistanceCache.flat===layer.flat)
    return layer.lesionDistanceCache.values;
  const flat=layer.flat,values=new Float32Array(layer.lineCount),token={};
  layer.lesionDistanceToken=token;
  const started=now();
  let computeMs=0;
  for(let i=0;i<layer.lineCount;){
    const chunkStart=now();
    do {
      const start=i*layer.k*3;
      values[i]=streamlineLesionDistance(flat.subarray(start,start+layer.k*3),bvh,{sample}).minMm;
      i++;
    }while(i<layer.lineCount && now()-chunkStart<budgetMs);
    computeMs+=now()-chunkStart;
    if(layer.lesionDistanceToken!==token || layer.flat!==flat)return null;
    onProgress(i,layer.lineCount);
    if(i<layer.lineCount)await new Promise(resolve=>frame(resolve));
  }
  if(layer.lesionDistanceToken!==token)return null;
  layer.lesionDistanceCache={bvh,flat,values,computeMs,elapsedMs:now()-started};
  return values;
}
