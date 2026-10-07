// Surgeon-authored nnInteractive point prompts. RAS points stay in the manifest; inference happens in the Python pipeline.
const SEG_PROMPT_COLORS=['#43B581','#4C9BE8','#C9A84C','#B06AD9','#E4572E','#3CC8C8'];
const SEG_POINT_POSITIVE='#35D07F',SEG_POINT_NEGATIVE='#F25F5C';
function segPromptItems(){return Array.isArray(state.manifest?.seg_prompts)?state.manifest.seg_prompts:[]}
function segPrompts(){return structuredClone(segPromptItems())}
function addSegPrompt(name){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const label=String(name??'').trim();if(!label)throw Error(tr('segPromptNameRequired'));
  state.manifest.seg_prompts??=[];
  const item={id:`segprompt-${crypto.randomUUID().slice(0,8)}`,name:label,for_volume:null,
    color:SEG_PROMPT_COLORS[state.manifest.seg_prompts.length%SEG_PROMPT_COLORS.length],points:[],status:'pending'};
  state.manifest.seg_prompts.push(item);state.activeSegPrompt=item.id;renderSegPrompts();return structuredClone(item);
}
function setActiveSegPrompt(id){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const item=segPromptItems().find(prompt=>prompt.id===id);
  if(!item||item.status!=='pending')throw Error(tr('segPromptNeedActive'));
  state.activeSegPrompt=id;renderSegPrompts();return structuredClone(item);
}
function removeSegPrompt(id){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const prompts=segPromptItems(),index=prompts.findIndex(prompt=>prompt.id===id);if(index<0)return false;
  prompts.splice(index,1);
  if(state.activeSegPrompt===id)state.activeSegPrompt=prompts.filter(prompt=>prompt.status==='pending').at(-1)?.id??null;
  renderSegPrompts();state.nv?.drawScene();return true;
}
function segPromptVolume(){
  const id=state.windowLayer==='overlay'&&state.overlay?state.overlay:state.base;
  return state.volumes.has(id)?id:null;
}
function addSegPoint(id,x,y,z,positive=true){
  if(state.mode==='patient')throw Error(tr('editBlocked'));
  const prompt=segPromptItems().find(item=>item.id===id);
  if(!prompt||prompt.status!=='pending')throw Error(tr('segPromptNeedActive'));
  const ras=[x,y,z].map(Number);if(ras.some(value=>!Number.isFinite(value)))throw Error('Coordenada RAS inválida');
  prompt.points??=[];
  if(!prompt.points.length){
    const volume=segPromptVolume();if(!volume)throw Error('Nenhum volume visível para o ponto');
    prompt.for_volume=volume;
  }
  const point={ras,positive:!!positive};prompt.points.push(point);state.activeSegPrompt=id;
  renderSegPrompts();state.nv?.drawScene();return structuredClone(point);
}
function renderSegPrompts(){
  const list=$('seg-prompt-list');if(!list)return;list.replaceChildren();
  const items=segPromptItems();
  if(!state.activeSegPrompt)state.activeSegPrompt=items.filter(item=>item.status==='pending').at(-1)?.id??null;
  if(!items.length){const empty=document.createElement('p');empty.className='empty-note';empty.textContent=tr('noSegPrompts');list.append(empty);return}
  for(const item of items){
    const row=document.createElement('div');row.className='seg-prompt-row';row.classList.toggle('active',item.id===state.activeSegPrompt);
    const select=document.createElement('button');select.type='button';select.className='prompt-name';select.textContent=item.name;
    select.disabled=item.status!=='pending';select.setAttribute('aria-pressed',String(item.id===state.activeSegPrompt));
    select.onclick=()=>{setActiveSegPrompt(item.id);setTool('segprompt')};
    const statusKey=item.status==='done'?'segPromptDone':item.status==='empty'?'segPromptEmpty':'segPromptPending';
    const meta=document.createElement('span');meta.className='prompt-meta';
    meta.textContent=`${item.points?.length||0} ponto(s) · ${tr(statusKey)}`;
    const remove=document.createElement('button');remove.type='button';remove.textContent=tr('delete');remove.setAttribute('aria-label',`${tr('delete')} ${item.name}`);
    remove.onclick=()=>removeSegPrompt(item.id);row.append(select,meta,remove);list.append(row);
  }
}
function wireSegPrompts(){
  $('add-seg-prompt').onclick=async()=>{
    try{addSegPrompt($('seg-prompt-name').value);$('seg-prompt-name').value='';await setTool('segprompt')}
    catch(error){$('tool-hint').textContent=error.message}
  };
}
function initSegPrompts(){
  const canvas=document.createElement('canvas');canvas.id='seg-prompt-overlay';canvas.setAttribute('aria-hidden','true');$('gl-wrap').append(canvas);
  const nv=state.nv,draw=nv.drawScene.bind(nv);
  nv.drawScene=function(){const result=draw();drawSegPromptOverlay();return result};
  const gl=$('gl');let down=null;
  gl.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY]});
  gl.addEventListener('pointercancel',()=>{down=null});
  gl.addEventListener('pointerup',event=>{
    if(state.tool!=='segprompt'||!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>4)return;
    down=null;const rect=gl.getBoundingClientRect(),dpr=gl.width/Math.max(1,rect.width);
    const point=canvasToMm((event.clientX-rect.left)*dpr,(event.clientY-rect.top)*dpr);
    if(!point)return;
    const prompt=segPromptItems().find(item=>item.id===state.activeSegPrompt&&item.status==='pending');
    if(!prompt){$('tool-hint').textContent=tr('segPromptNeedActive');return}
    try{addSegPoint(prompt.id,...point,!event.shiftKey)}catch(error){$('tool-hint').textContent=error.message}
  });
}
function drawSegPromptOverlay(){
  const canvas=$('seg-prompt-overlay'),gl=$('gl');if(!canvas||!gl)return;
  if(canvas.width!==gl.width||canvas.height!==gl.height){canvas.width=gl.width;canvas.height=gl.height}
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if(state.mode==='patient'||$('gl-wrap').style.visibility==='hidden')return;
  const dpr=gl.width/Math.max(1,gl.getBoundingClientRect().width)||1;
  for(const slice of sliceTiles()){
    const tile=slice.leftTopWidthHeight,x0=Math.min(tile[0],tile[0]+tile[2]),width=Math.abs(tile[2]);
    ctx.save();ctx.beginPath();ctx.rect(x0,tile[1],width,tile[3]);ctx.clip();
    for(const prompt of segPromptItems())for(const point of prompt.points||[]){
      const projection=sliceProject(slice,point.ras);if(Math.abs(projection.off)>2.5)continue;
      ctx.globalAlpha=prompt.status==='pending'?1:.55;ctx.fillStyle=point.positive?SEG_POINT_POSITIVE:SEG_POINT_NEGATIVE;
      ctx.strokeStyle='#10151b';ctx.lineWidth=1.5*dpr;ctx.beginPath();ctx.arc(projection.x,projection.y,4*dpr,0,2*Math.PI);ctx.fill();ctx.stroke();
    }
    ctx.restore();
  }
}
