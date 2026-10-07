// Save is a new blob download; the opened file is never overwritten.
function downloadBlob(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000)}
async function save(){
  if(state.mode==='patient')throw Error('Salvamento indisponível no modo paciente');
  const next=Number(state.manifest.version||0)+1,manifest=structuredClone(state.manifest),clone=document.documentElement.cloneNode(true);
  manifest.version=next;
  for(const meta of manifest.masks){const data=state.masks.get(meta.id),node=clone.querySelector('#capsule-blob-'+CSS.escape(meta.blob));if(!data)throw Error('Máscara não encontrada');const encoded=base64FromBytes(await gzip(data));if(node)node.textContent=encoded;else{const script=document.createElement('script');script.id='capsule-blob-'+meta.blob;script.type='application/octet-stream';script.dataset.encoding='gzip+base64';script.textContent=encoded;clone.querySelector('head').append(script)}}
  clone.querySelector('#capsule-manifest').textContent=JSON.stringify(manifest).replace(/</g,'\\u003c');
  const html='<!doctype html>\n'+clone.outerHTML,blob=new Blob([html],{type:'text/html;charset=utf-8'}),stem=decodeURIComponent(location.pathname.split('/').pop()||'case').replace(/(?:\.v\d+)?\.capsule\.html$|\.html$/,'');
  const filename=`${stem}.v${next}.capsule.html`;downloadBlob(blob,filename);state.manifest.version=next;$('version-badge').textContent='v'+next;return {filename,blob};
}
function screenshot(){const canvas=$(state.activeView+'-canvas');canvas.toBlob(blob=>{if(blob)downloadBlob(blob,`case-capsule-${state.activeView}.png`)},'image/png')}
