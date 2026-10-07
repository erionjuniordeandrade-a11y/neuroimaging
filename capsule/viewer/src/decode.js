// Payload decoding and grid validation. C-order payloads use x as the fastest axis.
function bytesFrom64(s){const raw=atob(s.trim().replace(/\s/g,'')),data=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)data[i]=raw.charCodeAt(i);return data}
function base64FromBytes(bytes){let out='';for(let i=0;i<bytes.length;i+=32768)out+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(out)}
async function gunzip(encoded){const stream=new Blob([bytesFrom64(encoded)]).stream().pipeThrough(new DecompressionStream('gzip'));return new Uint8Array(await new Response(stream).arrayBuffer())}
async function gzip(data){const stream=new Blob([data]).stream().pipeThrough(new CompressionStream('gzip'));return new Uint8Array(await new Response(stream).arrayBuffer())}
async function loadCapsule(){
  try{
    const node=$('capsule-manifest');if(!node)throw Error('manifest missing');
    const manifest=JSON.parse(node.textContent);if(manifest.schema!=='case-capsule/1')throw Error('schema');
    state.manifest=manifest;manifest.masks??=[];manifest.annotations??=[];manifest.tour??=[];
    if(!manifest.volumes?.length){status('empty',tr('empty'));readyResolve(state);return}
    const [nx,ny,nz]=manifest.grid.dims,total=nx*ny*nz;
    if(!Number.isSafeInteger(total)||total<=0||total>150000000)throw Error('grid');
    const a=manifest.grid.affine_ras;
    inverseAffine=inverse3([a[0][0],a[0][1],a[0][2],a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]]);
    for(let i=0;i<manifest.volumes.length;i++){
      const meta=manifest.volumes[i],blob=$('capsule-blob-'+meta.blob);if(!blob)throw Error('missing volume blob');
      status('loading',`${tr('loading')} ${i+1}/${manifest.volumes.length}`);await new Promise(r=>setTimeout(r,0));
      const bytes=await gunzip(blob.textContent);if(bytes.length!==total*2)throw Error('volume length');
      const aligned=bytes.slice().buffer;
      const data=meta.dtype==='int16'?new Int16Array(aligned):meta.dtype==='uint16'?new Uint16Array(aligned):null;
      if(!data)throw Error('volume dtype');state.volumes.set(meta.id,{meta,data});
    }
    for(const meta of manifest.masks){
      const blob=$('capsule-blob-'+meta.blob);if(!blob)throw Error('missing mask blob');
      const bytes=await gunzip(blob.textContent);if(bytes.length!==total)throw Error('mask length');state.masks.set(meta.id,bytes);
    }
    state.base=manifest.volumes[0].id;state.crosshair=affinePoint((nx-1)/2,(ny-1)/2,(nz-1)/2);
    const preset=manifest.volumes[0].window_presets?.[0],s=manifest.volumes[0].stats;
    state.window=preset?{center:preset.center,width:preset.width}:{center:((s?.p01??0)+(s?.p99??1000))/2,width:Math.max(1,(s?.p99??1000)-(s?.p01??0))};
    initGL();initUI();status('ready','');renderAll();readyResolve(state);
  }catch(err){status('error',err.message==='WebGL2 unavailable'?tr('gpu'):tr('bad')+' '+err.message);readyReject(err)}
}
