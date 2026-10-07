/** Display decisions only. Coordinates and measurements come from the loaded case. */
export const MAP_TRIO=Object.freeze(['bank_cst_r','bank_fat_r','bank_slf3_r']);

export function initialMapBanks(catalog,hasLesion,search,defaultId){
  const ids=new Set(catalog.map(bank=>bank.id)), query=new URLSearchParams(search);
  if(query.has('bank')) return [...new Set(query.get('bank').split(','))].filter(id=>ids.has(id));
  if(hasLesion && MAP_TRIO.every(id=>ids.has(id))) return [...MAP_TRIO];
  return ids.has(defaultId)?[defaultId]:[];
}

export function lesionOrigin(mask,shape,affine){
  if(!mask || shape?.length!==3 || mask.length!==shape[0]*shape[1]*shape[2]
    || affine?.length!==16 || !affine.every(Number.isFinite)) return null;
  const [nx,ny]=shape, sum=[0,0,0]; let n=0;
  for(let index=0;index<mask.length;index++) if(mask[index]){
    sum[0]+=index%nx; sum[1]+=Math.floor(index/nx)%ny; sum[2]+=Math.floor(index/(nx*ny)); n++;
  }
  if(!n) return null;
  const voxel=sum.map(value=>value/n);
  const world=[0,1,2].map(row=>voxel.reduce((value,v,col)=>value+affine[row*4+col]*v,affine[row*4+3]));
  return Object.freeze({voxel:Object.freeze(voxel),world:Object.freeze(world)});
}

const numeric=value=>{
  if(value==null || String(value).trim()==='') return null;
  const n=Number(value); return Number.isFinite(n)&&n>=0?n:null;
};
const count=value=>{ const n=numeric(value); return Number.isSafeInteger(n)?n:null; };

export function layerMeasurement(layer,key){
  const h=layer?.responseHeaders, get=name=>h?.get(name);
  const floor=numeric(get('X-clearanceFloorMm')), p5=numeric(get('X-clearanceP5'));
  const data={analytic:count(get('X-nAnalytic')),full:count(get('X-nAnalyticFull')),
    loaded:count(layer?.lineCount),floor,
    population:['full','random_sample'].includes(get('X-clearancePopulation'))?get('X-clearancePopulation'):'unrecorded'};
  const result=(state,value,detail)=>Object.freeze({...data,state,value,detail});
  if(!layer || !layer.sourcePopulation || (layer.bankId && (key!==`bank:${layer.bankId}`
    || layer.tract?.id!==layer.bankId || layer.sourcePopulation!==key)))
    return result('identity-unavailable','Unassigned','Measurement ownership is unavailable for this layer.');
  if(get('X-clearanceRefusal')) return result('refused','Unavailable','The server refused this clearance measurement.');
  if(p5===null) return result('unavailable','Unavailable','No usable lesion clearance was recorded.');
  if(!floor) return result('unknown-floor','Floor unknown','Numeric clearance is withheld until geometric uncertainty is recorded.');
  const shown=String(get('X-clearanceP5Display')??'').trim();
  // The numeric header is rounded to two decimals; a true sub-floor value
  // can round onto the floor. Preserve the producer's unresolved marker.
  const markedBelow=/^>=\d+(?:\.\d+)?$/.test(shown)&&Math.abs(p5-floor)<=.005001;
  if(p5<floor||markedBelow) return result('below-floor',`Below ${floor} mm floor`,'Distance is unresolved within the recorded geometric uncertainty.');
  // The server formats p5 from full precision (clearance.format_p05); re-rounding the
  // two-decimal X-clearanceP5 header in JavaScript can disagree at a .x5 boundary
  // (6.35 -> "6.3" here, "6.4" there). Its display string is the single authority
  // when it is a plain above-floor number; the local rule is only a fallback.
  const precision=p5>=9.995?.5:.05;
  const consistent=/^\d+(\.\d)?$/.test(shown)&&Number(shown)>=floor
    &&Math.abs(Number(shown)-p5)<=precision+.005001;
  const value=consistent?shown:(p5>=10?p5.toFixed(0):p5.toFixed(1));
  return result('recorded',`${value} mm`,
    `p5 to lesion, recorded floor ${floor} mm`);
}

/** A label points to a real packed vertex near this layer's local centre. */
export function layerLabelPoint(flat,origin){
  if(!flat?.length || !origin) return null;
  const sum=[0,0,0]; let n=0;
  for(let i=0;i<flat.length;i+=3){
    if(Math.hypot(flat[i]-origin[0],flat[i+1]-origin[1],flat[i+2]-origin[2])>65) continue;
    for(let axis=0;axis<3;axis++) sum[axis]+=flat[i+axis]; n++;
  }
  const centre=n?sum.map(v=>v/n):origin;
  let best=0, distance=Infinity;
  for(let i=0;i<flat.length;i+=3){
    const d=Math.hypot(flat[i]-centre[0],flat[i+1]-centre[1],flat[i+2]-centre[2]);
    if(d<distance){distance=d;best=i;}
  }
  return Array.from(flat.subarray(best,best+3));
}
