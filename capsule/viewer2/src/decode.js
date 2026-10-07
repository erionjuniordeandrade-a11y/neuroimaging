// Payload decoding (gzip+base64 via DecompressionStream) and in-memory NIfTI-1 construction.
function bytesFrom64(s){const raw=atob(s.trim().replace(/\s/g,'')),data=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)data[i]=raw.charCodeAt(i);return data}
function base64FromBytes(bytes){let out='';for(let i=0;i<bytes.length;i+=32768)out+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(out)}
async function gunzip(encoded){const stream=new Blob([bytesFrom64(encoded)]).stream().pipeThrough(new DecompressionStream('gzip'));return new Uint8Array(await new Response(stream).arrayBuffer())}
async function gzip(data){const stream=new Blob([data]).stream().pipeThrough(new CompressionStream('gzip'));return new Uint8Array(await new Response(stream).arrayBuffer())}
async function blobBytes(blobId){const node=$('capsule-blob-'+blobId);if(!node)throw Error('blob ausente: '+blobId);return gunzip(node.textContent)}
const NIFTI_TYPES={uint8:[2,8,Uint8Array],int16:[4,16,Int16Array],uint16:[512,16,Uint16Array]};
// NIfTI-1 single file: 348-byte header + 4 extension bytes, voxels at 352, sform_code 1 = grid.affine_ras.
function niftiBytes(grid,dtype,data,{slope=1,inter=0,intent=0,calMin=0,calMax=0,description=''}={}){
  const [code,bitpix]=NIFTI_TYPES[dtype]||[];if(!code)throw Error('dtype '+dtype);
  const [nx,ny,nz]=grid.dims,a=grid.affine_ras,buf=new ArrayBuffer(352+data.byteLength),v=new DataView(buf);
  v.setInt32(0,348,true);
  [3,nx,ny,nz,1,1,1,1].forEach((d,i)=>v.setInt16(40+2*i,d,true));
  v.setInt16(68,intent,true);v.setInt16(70,code,true);v.setInt16(72,bitpix,true);
  const zoom=[0,1,2].map(c=>Math.hypot(a[0][c],a[1][c],a[2][c]));
  [1,zoom[0],zoom[1],zoom[2],1,1,1,1].forEach((p,i)=>v.setFloat32(76+4*i,p,true));
  v.setFloat32(108,352,true);v.setFloat32(112,slope,true);v.setFloat32(116,inter,true);
  v.setUint8(123,10);v.setFloat32(124,calMax,true);v.setFloat32(128,calMin,true);
  const text=new TextEncoder().encode(description.slice(0,79));new Uint8Array(buf,148,80).set(text);
  v.setInt16(252,0,true);v.setInt16(254,1,true);
  for(let r=0;r<3;r++)for(let c=0;c<4;c++)v.setFloat32(280+16*r+4*c,a[r][c],true);
  new Uint8Array(buf,344,4).set([0x6e,0x2b,0x31,0]);
  new Uint8Array(buf,352).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
  return buf;
}
// Minimal .tck header check; NiiVue parses the points.
function tckInfo(bytes){
  const head=new TextDecoder('latin1').decode(bytes.subarray(0,Math.min(bytes.length,4096)));
  if(!head.startsWith('mrtrix tracks'))throw Error('tck inválido');
  const count=Number((head.match(/\ncount:\s*(\d+)/)||[])[1]);return {count};
}
// Keep world-RAS streamline points alongside the render meshes for exact corridor checks.
function tckStreamlines(bytes){
  let headerEnd=-1;
  for(let i=0;i+3<bytes.length;i++){
    if((i===0||bytes[i-1]===10)&&bytes[i]===69&&bytes[i+1]===78&&bytes[i+2]===68){
      let j=i+3;if(bytes[j]===13)j++;if(bytes[j]===10){headerEnd=j+1;break}
    }
  }
  if(!bytes.subarray(0,Math.min(bytes.length,13)).every((v,i)=>v===[109,114,116,114,105,120,32,116,114,97,99,107,115][i])||headerEnd<0)throw Error('tck inválido');
  const header=new TextDecoder('latin1').decode(bytes.subarray(0,headerEnd));
  const dtype=(header.match(/^datatype:\s*(Float32LE|Float32BE)\s*$/m)||[])[1];
  const offset=Number((header.match(/^file:\s*\.\s+(\d+)\s*$/m)||[])[1]);
  if(!dtype||!Number.isInteger(offset)||offset<headerEnd||offset>bytes.length)throw Error('tck inválido');
  const little=dtype==='Float32LE',view=new DataView(bytes.buffer,bytes.byteOffset+offset,bytes.byteLength-offset);
  const streamlines=[];let line=[];
  for(let p=0;p+12<=view.byteLength;p+=12){
    const x=view.getFloat32(p,little),y=view.getFloat32(p+4,little),z=view.getFloat32(p+8,little);
    if(!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(z)){
      if(line.length)streamlines.push(Float32Array.from(line));line=[];
      if(!Number.isNaN(x)&&!Number.isNaN(y)&&!Number.isNaN(z))break; // NaN triplet separates streamlines; Inf ends the file
      continue;
    }
    line.push(x,y,z);
  }
  if(line.length)streamlines.push(Float32Array.from(line));
  return streamlines;
}
function hexToRgb(hex){const m=/^#?([0-9a-f]{6})$/i.exec(hex||'');const n=m?parseInt(m[1],16):0xC9A84C;return [n>>16&255,n>>8&255,n&255]}
