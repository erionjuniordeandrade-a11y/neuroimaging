// Pure contracts for the versioned public reference package. No case API.
export function decodeAtlasLabels(meta, buffer, counts) {
  const set=meta.sets.glasser;
  if(!meta.built_from_compressed_mesh || !Array.isArray(counts) || counts.length!==2 ||
    counts.some((n,i)=>n!==set.counts[i]) || set.count!==counts[0]+counts[1] ||
    set.offset+set.count*2!==buffer.byteLength) throw new Error('Atlas mesh / label correspondence failed');
  const view=new DataView(buffer), out={};
  let offset=set.offset;
  for(const [hi,h] of ['L','R'].entries()) {
    out[h]=new Uint16Array(counts[hi]);
    for(let i=0;i<counts[hi];i++,offset+=2) {
      const id=view.getUint16(offset,true);
      if(!(id in set.regions[h]))throw new Error(`Unknown ${h} cortical label ${id}`);
      out[h][i]=id;
    }
  }
  return out;
}

export function regionIdentity(meta,hemi,id) {
  const raw=meta.sets.glasser.regions[hemi]?.[id];
  if(raw===undefined)throw new Error('Unknown atlas region');
  return {hemi,id,code:id===0?'Medial wall':raw.replace(/^[LR]_/,'').replace(/_ROI$/,''),raw};
}

export function decodeAtlasBundle(meta,buffer) {
  const {offset,lines,points_per_line:n,bytes}=meta;
  if(![offset,lines,n,bytes].every(Number.isSafeInteger) || offset<0 || lines<1 ||
    lines>220 || n<2 || n>256 || bytes!==lines*n*6 || offset+bytes>buffer.byteLength)
    throw new Error('Invalid atlas pathway layout');
  const view=new DataView(buffer,offset,bytes), out=[];
  let cursor=0;
  for(let row=0;row<lines;row++) {
    const points=[];
    for(let p=0;p<n;p++) {
      const xyz=[];
      for(let axis=0;axis<3;axis++,cursor+=2)xyz.push(view.getInt16(cursor,true)/100);
      points.push(xyz);
    }
    out.push(points);
  }
  return out;
}

export const ATLAS_VIEWS=Object.freeze({
  left:[-1,-.22,.18],right:[1,-.22,.18],superior:[-.04,-.12,1],
  anterior:[-.03,1,.08],posterior:[.02,-1,.12],medial:[1,-.08,.12],
});

// Bundle catalogue mirrors ./atlas/tracts.json (87 ids as shipped; see atlas-sources.html for
// provenance). Families here are the SCENE GRAMMAR v2 vocabulary for `scene.bundles`/`scene.ghost`:
// most resolve to a per-side id (`${family}_L` / `${family}_R`); the five midline families below
// are already bare ids in tracts.json and never take a side suffix.
export const MIDLINE_BUNDLE_FAMILIES=Object.freeze(['AC','CC','MCP','SCP','V']);
export const LATERAL_BUNDLE_FAMILIES=Object.freeze([
  // Association
  'AF','C_FPH','C_FP','C_PHP','C_PH','C_PO','EMC','FAT','IFOF','ILF','MdLF','PAT',
  'SLF1','SLF2','SLF3','UF','VOF',
  // Cerebellar (peduncles + CB)
  'CB','ICP',
  // Cranial nerves
  'CNII','CNIII','CNV','CNVII','CNVIII',
  // Projection
  'AR','CBT','CPT_F','CPT_O','CPT_P','CST','CS_A','CS_P','CS_S','DRTT','F','ML','OR','RST',
  'TR_A','TR_P','TR_S',
]);
export const BUNDLE_FAMILIES=Object.freeze([...MIDLINE_BUNDLE_FAMILIES,...LATERAL_BUNDLE_FAMILIES]);

/** Resolve a scene-grammar bundle family to a concrete tracts.json id for one side. */
export function resolveBundleId(family,side){
  return MIDLINE_BUNDLE_FAMILIES.includes(family)?family:`${family}_${side==='R'?'R':'L'}`;
}

// Subcortical structure ids mirror ./atlas/subcortex.json (Melbourne Subcortex Atlas scale 1).
export const SUBCORTEX_IDS=Object.freeze(['HIP-lh','HIP-rh','AMY-lh','AMY-rh','THA-lh','THA-rh',
  'NAc-lh','NAc-rh','GP-lh','GP-rh','PUT-lh','PUT-rh','CAU-lh','CAU-rh']);

export function atlasSelectionFromSearch(search) {
  const p=new URLSearchParams(search), hemi=['L','R','both'].includes(p.get('hemi'))?p.get('hemi'):'both';
  const area=Number(p.get('area')??8);
  return {hemi,area:Number.isInteger(area)&&area>=0&&area<=180?area:8,
    profile:p.get('profile')==='presenter'?'presenter':'teaching',playing:false};
}
