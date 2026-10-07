// Save = a new downloaded file <name>.v<n+1>.capsule.html; the opened file is never overwritten.
// The manifest object is cloned whole, so unknown fields survive; tract blobs are carried byte-for-byte.
const DYNAMIC_CONTAINERS=['tract-list','mask-list','anatomy-list','annotation-list','seg-prompt-list','corridor-report','traj-length','traj-depth-out','corridor-radius-out','step-list','base-volume','overlay-volume','window-layer','window-preset','preset3d','readout','mask-legend','case-label','step-count','step-title','step-text','tool-hint'];
async function save(){
  if(state.mode==='patient')throw Error(tr('patientSaveBlocked'));
  if(state.editingMask)syncDrawing();
  for(const meta of state.manifest.masks){
    const m=state.masks.get(meta.id);if(!m)throw Error('Máscara não encontrada');
    const digest=await sha256Hex(m.data);
    if(meta.blob_sha256!==digest){meta.reviewed=false;delete meta.review;meta.blob_sha256=digest}
    state.blobIntegrity.set(meta.blob,true);
  }
  state.manifest.case??={};
  if(state.reviewer.trim())state.manifest.case.reviewer=state.reviewer.trim();else delete state.manifest.case.reviewer;
  const next=Number(state.manifest.version||0)+1,manifest=structuredClone(state.manifest),clone=document.documentElement.cloneNode(true);
  manifest.version=next;
  for(const meta of manifest.masks){
    const m=state.masks.get(meta.id);if(!m)throw Error('Máscara não encontrada');
    const encoded=base64FromBytes(await gzip(m.data)),node=clone.querySelector('#capsule-blob-'+CSS.escape(meta.blob));
    if(node)node.textContent=encoded;
    else{const script=document.createElement('script');script.id='capsule-blob-'+meta.blob;script.type='application/octet-stream';script.dataset.encoding='gzip+base64';script.textContent=encoded;clone.querySelector('#capsule-manifest').after(script)}
  }
  for(const meta of manifest.anatomy||[])if(!clone.querySelector('#capsule-blob-'+CSS.escape(meta.blob)))throw Error('Anatomia sem dados: '+meta.id);
  for(const meta of manifest.tracts||[]){
    if(!clone.querySelector('#capsule-blob-'+CSS.escape(meta.blob)))throw Error('Trato sem dados: '+meta.id);
    if(meta.outlier_blob&&!clone.querySelector('#capsule-blob-'+CSS.escape(meta.outlier_blob)))throw Error('Fibras filtradas sem dados: '+meta.id);
  }
  clone.querySelector('#capsule-manifest').textContent=JSON.stringify(manifest).replace(/</g,'\\u003c');
  // Reset runtime UI so the saved file opens exactly like a fresh capsule.
  const body=clone.querySelector('body');body.className='';body.removeAttribute('style');
  for(const id of DYNAMIC_CONTAINERS){const el=clone.querySelector('#'+id);if(el)el.replaceChildren()}
  const outlierToggle=clone.querySelector('#show-outliers');if(outlierToggle)outlierToggle.checked=false;
  const densityToggle=clone.querySelector('#tract-density-toggle');if(densityToggle){densityToggle.setAttribute('aria-pressed','false');for(const option of densityToggle.querySelectorAll('[data-density-mode]'))option.classList.toggle('active',option.dataset.densityMode==='proportional')}
  const st=clone.querySelector('#status');st.className='glass';
  const wrap=clone.querySelector('#gl-wrap');wrap.removeAttribute('style');wrap.replaceChildren(Object.assign(document.createElement('canvas'),{id:'gl'}));
  wrap.querySelector('canvas').setAttribute('aria-label','Visualização NiiVue');
  const wrap3d=clone.querySelector('#gl3d-wrap'),densityWarning=clone.querySelector('#density-warning');wrap3d.removeAttribute('style');wrap3d.replaceChildren(Object.assign(document.createElement('canvas'),{id:'gl3d'}));
  if(densityWarning){densityWarning.hidden=true;wrap3d.append(densityWarning)}
  wrap3d.querySelector('canvas').setAttribute('aria-label','Visualização 3D');clone.querySelector('#mask-legend').hidden=true;clone.querySelector('#anatomy-group').hidden=true;clone.querySelector('#trajectory-panel').hidden=true;clone.querySelector('#segmentation-panel').hidden=true;
  for(const el of clone.querySelectorAll('.active'))el.classList.remove('active');
  for(const el of clone.querySelectorAll('body > :not(header):not(main):not(#status):not(script)'))el.remove();
  const html='<!doctype html>\n'+clone.outerHTML,blob=new Blob([html],{type:'text/html;charset=utf-8'});
  const stem=decodeURIComponent(location.pathname.split('/').pop()||'case').replace(/(?:\.v\d+)?\.capsule\.html$|\.html$/,'');
  const filename=`${stem}.v${next}.capsule.html`;downloadBlob(blob,filename);
  state.manifest.version=next;setVersionBadge(next);return {filename,blob};
}
