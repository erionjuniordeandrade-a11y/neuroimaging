import {LESSONS,SOURCES,REGIONS} from './lesson_content.js';
import {createLessonController,lessonSearch} from './lesson_state.js';

function el(tag,attrs={},text=''){
  const node=document.createElement(tag);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));
  if(text)node.textContent=text;
  return node;
}
const button=(text,id,action)=>{const b=el('button',{type:'button',id},text);b.addEventListener('click',action);return b;};
const label=key=>key.replaceAll('_',' ');

/** Local content-only player. Callbacks expose display actions, never tracking. */
export const STEP_MS=20000;
export function mountLessonPlayer(root,{onStep=()=>{},onPlayback=()=>{},onRestore=()=>{},
  onClose=()=>{},capability=()=>({available:false,message:'Reconstruction unavailable'}),onCandidate=()=>{},
  getProfile=()=> 'clinical',onProfile=()=>{},
  profiles=[['clinical','Clinical'],['presenter','Presenter'],['teaching','Teaching']],
  atlasCapability=()=>null,candidateLabel='Show available candidate',onExplore=null}={}){
  let timer=null,lastStep='',controller;
  const cancelTimer=()=>{if(timer!==null)clearTimeout(timer);timer=null;};
  function render(state){
    cancelTimer();
    const focusId=root.contains(document.activeElement)?document.activeElement.id:null;
    root.replaceChildren();
    const header=el('div',{class:'lesson-top'});
    header.append(el('span',{class:'lesson-eyebrow'},'GUIDED ANATOMY'),button('Close','lessonClose',()=>{controller.pause();onClose();}));
    root.append(header);
    const profile=el('select',{id:'lessonProfile','aria-label':'Display profile'});
    for(const [id,name] of profiles)profile.append(el('option',{value:id},`${name} display`));
    profile.value=getProfile();profile.addEventListener('change',()=>onProfile(profile.value));root.append(profile);
    const select=el('select',{id:'lessonSelect','aria-label':'Choose a guided lesson'});
    select.append(el('option',{value:''},'Choose a lesson'));
    for(const l of LESSONS) select.append(el('option',{value:l.id},`${l.title} · ${l.minutes} min`));
    select.value=state.lessonId||'';
    select.addEventListener('change',()=>select.value?controller.select(select.value):controller.close());
    root.append(select);
    const review=el('p',{class:'lesson-review'},'Educational draft · anatomical review pending');
    root.append(review);
    const lesson=LESSONS.find(l=>l.id===state.lessonId);
    if(!lesson){
      root.append(el('h2',{},'A map, a pathway, a question.'),
        el('p',{class:'lesson-lead'},'Six guided journeys connect the scene to anatomy, methods and the experiments behind a claim. Start with a question; keep its evidence beside the picture.'));
      if(state.error)root.append(el('p',{role:'status',class:'lesson-capability'},state.error));
      const list=el('div',{class:'lesson-catalog'});
      for(const l of LESSONS){const b=button(l.title,`lessonStart-${l.id}`,()=>controller.select(l.id));
        b.append(el('span',{},`${l.steps.length} steps · auto-advance ${Math.round(STEP_MS/1000)} s each · ~${l.minutes} min taught`),el('small',{},l.summary));list.append(b);}
      root.append(list);lastStep='';onStep(null,null);onPlayback(false);return;
    }
    const current=lesson.steps[state.step];
    root.append(el('h2',{},lesson.title));
    const progress=el('div',{class:'lesson-progress','aria-label':'Lesson steps'});
    lesson.steps.forEach((s,i)=>{const b=button(String(i+1),`lessonStep-${i}`,()=>controller.move(i-controller.state.step));
      b.title=s.title;b.setAttribute('aria-label',`Step ${i+1}: ${s.title}`);
      b.setAttribute('aria-current',i===state.step?'step':'false');progress.append(b);});root.append(progress);
    root.append(el('p',{class:'lesson-evidence'},`${label(current.evidenceClass)} · step ${state.step+1} of ${lesson.steps.length}`));
    const article=el('article',{id:'lessonCurrent','aria-live':'polite','aria-atomic':'true'});
    article.append(el('h3',{},current.title),el('p',{class:'lesson-lead'},current.text));
    root.append(article);
    if(current.diagram){
      const diagram=el('div',{class:'lesson-diagram',role:'img','aria-label':current.diagram==='pipeline'?'Data processing stages, not neural transmission':current.diagram==='sampling'?'One straight path with two endpoints or added collinear points':'Bodily-state and regional concept cards; no subject circuit geometry'});
      const items=current.diagram==='pipeline'?['Diffusion model','Candidate corpus','Recipe selection','Analytic result','Display sample']
        :current.diagram==='sampling'?['● ────────────────── ●','● ─ ● ─ ● ─ ● ─ ● ─ ●','Same physical segment']
        :['Bodily-state framework','Thalamic territory','Posterior / anterior insula','Wider cortical relationships'];
      for(const text of items)diagram.append(el('span',{},text));
      diagram.append(el('small',{},'Original schematic · explanatory order only'));root.append(diagram);
    }
    if(current.needsAtlas || lesson.referenceOnly){
      const atlas=atlasCapability(current,lesson);
      root.append(el('p',{class:'lesson-capability',role:'note'},atlas?.message||'This case view has no registered cortical atlas. Open reference anatomy to explore labelled group parcels; no atlas boundary or functional marker is placed on the case reconstruction.'));
      if(!atlas)root.append(el('a',{href:`./atlas.html?profile=${getProfile()==='presenter'?'presenter':'teaching'}&lesson=${encodeURIComponent(lesson.id)}&step=${state.step}`,class:'lesson-atlas-link'},'Open the reference atlas ↗'));
    }
    if(current.group){
      const cap=capability(current.group,current.side);
      root.append(el('p',{class:'lesson-capability',id:'lessonCandidateStatus'},cap.message));
      if(cap.available)root.append(button(candidateLabel,'lessonCandidate',()=>onCandidate(current.group,current.side)));
    }
    if(current.regions.length){
      const regions=el('div',{class:'lesson-regions'});
      for(const id of current.regions){const region=REGIONS[id];const d=el('details',{});
        d.append(el('summary',{},region.name),el('p',{},region.text),el('small',{},`${label(region.evidenceClass)} · ${region.sources.join(', ')}`));regions.append(d);}root.append(regions);
    }
    if(current.question){const answer=el('details',{id:'lessonAnswer',class:'lesson-answer'});
      answer.append(el('summary',{},'Reveal explanation'),el('p',{},current.answer));root.append(answer);}
    const nav=el('div',{class:'lesson-nav'});
    const prev=button('Previous','lessonPrev',()=>controller.move(-1));prev.disabled=state.step===0;
    const play=button(state.status==='playing'?'Pause tour':'Play tour','lessonPlay',()=>state.status==='playing'?controller.pause():controller.play());
    const next=button('Next','lessonNext',()=>controller.move(1));next.disabled=state.step===lesson.steps.length-1;
    nav.append(prev,play,next);root.append(nav);
    root.append(button('Restore authored view','lessonRestore',onRestore));
    if(onExplore)root.append(button('Return to exploration','lessonExplore',()=>{controller.close();onExplore();}));
    const sources=el('details',{class:'lesson-sources',open:''});sources.append(el('summary',{},`Sources & limits (${current.sources.length})`));
    if(!current.sources.length)sources.append(el('p',{},'Local software contract and generated-data regression; no external biological claim.'));
    for(const id of current.sources){const s=SOURCES[id];const row=el('div',{});
      row.append(el('a',{href:s.url,target:'_blank',rel:'noopener noreferrer'},`${id} · ${s.title}`),el('p',{},s.scope));sources.append(row);}root.append(sources);
    const notes=el('details',{class:'lesson-notes','data-audience':'presenter'});
    notes.append(el('summary',{},'Presenter notes · operator panel'),el('p',{},current.notes));root.append(notes);
    root.append(el('p',{class:'lesson-version'},`Draft ${lesson.version} · ${lesson.audience} · pace is instructional, never neural timing`));
    const identity=`${lesson.id}:${state.step}`;
    if(identity!==lastStep){lastStep=identity;onStep(lesson,current);}
    onPlayback(state.status==='playing');
    if(state.status==='playing'){
      // durationSec (SCENE GRAMMAR v2) overrides the default per-step pace; STEP_MS otherwise.
      const stepMs=current.scene.durationSec?current.scene.durationSec*1000:STEP_MS;
      const bar=el('div',{class:'lesson-pace',role:'timer','aria-label':`Next step in ${Math.round(stepMs/1000)} seconds`});
      const fill=el('span',{class:'lesson-pace-fill'});bar.append(fill);
      const evidenceLine=root.querySelector('.lesson-evidence');evidenceLine?.after(bar);
      requestAnimationFrame(()=>{fill.style.transitionDuration=`${stepMs}ms`;fill.style.width='100%';});
      timer=setTimeout(()=>{const atEnd=controller.state.step===lesson.steps.length-1;
        if(atEnd)controller.pause();else{controller.move(1);controller.play();}},stepMs);
    }
    if(focusId)root.querySelector(`#${CSS.escape(focusId)}`)?.focus({preventScroll:true});
  }
  controller=createLessonController({search:location.search,onChange:state=>{
    history.replaceState(history.state,'',`${location.pathname}${lessonSearch(location.search,state)}${location.hash}`);render(state);
  }});
  root.addEventListener('keydown',event=>{
    if(event.key==='Escape'){controller.pause();onClose();return;}
    if(/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName))return;
    if(event.key==='ArrowRight'){event.preventDefault();controller.move(1);}
    else if(event.key==='ArrowLeft'){event.preventDefault();controller.move(-1);}
    else if(event.key===' ' && event.target.tagName!=='BUTTON' && event.target.tagName!=='SUMMARY'){
      event.preventDefault();controller.state.status==='playing'?controller.pause():controller.play();}
  });
  render(controller.state);
  return {controller,refresh:({restoreView=false}={})=>{if(restoreView)lastStep='';render(controller.state);},dispose:cancelTimer};
}
