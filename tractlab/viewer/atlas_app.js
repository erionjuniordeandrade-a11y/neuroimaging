import {createAtlasScene} from './atlas_scene.js';
import {atlasSelectionFromSearch,regionIdentity} from './atlas_data.js';
import {mountLessonPlayer} from './lesson_player.js';
import {resolveScene} from './lesson_scene.js';
import {DEFAULT_SCENE} from './lesson_content.js';
import {toggleBundle,MAX_BUNDLES,tintForIndex,tintCss,filterBundles,selectionCaption} from './bundle_picker.js';

const $=id=>document.getElementById(id),initial=atlasSelectionFromSearch(location.search);
let scene,player,profile=initial.profile,playing=false,currentStep=null,currentLesson=null,exploration=null,lastResolvedTrace=false;
// Ordered primary selection (pick order = tint order in the scene) and the ghost set that
// came with the current lesson step. Manual picks keep the ghosts unless a ghost is promoted.
let selectedBundles=[],currentGhosts=[];
const currentHemisphere=()=>$('atlasHemisphere').value==='R'?'R':'L';
// Friendly names for families with a confirmed, standard expansion. A family left out here
// (e.g. the cingulum subdivisions C_FPH/C_FP/C_PHP/C_PH/C_PO, PAT, CB, V, CPT_F/O/P, CS_A/P/S,
// F, TR_A/P/S) falls back to its raw tracts.json id via bundleLabel below rather than a guess —
// see the handoff report for the full flagged list.
const bundleNames={
  AF:'Arcuate',EMC:'Extreme capsule',FAT:'Frontal aslant',IFOF:'Inferior fronto-occipital',
  ILF:'Inferior longitudinal',MdLF:'Middle longitudinal',SLF1:'Superior longitudinal I',
  SLF2:'Superior longitudinal II',SLF3:'Superior longitudinal III',UF:'Uncinate',VOF:'Vertical occipital',
  CC:'Corpus callosum',AC:'Anterior commissure',
  ICP:'Inferior cerebellar peduncle',MCP:'Middle cerebellar peduncle',SCP:'Superior cerebellar peduncle',
  CNII:'Cranial nerve II (optic)',CNIII:'Cranial nerve III (oculomotor)',CNV:'Cranial nerve V (trigeminal)',
  CNVII:'Cranial nerve VII (facial)',CNVIII:'Cranial nerve VIII (vestibulocochlear)',
  CST:'Corticospinal',OR:'Optic radiation',ML:'Medial lemniscus',DRTT:'Dentatorubrothalamic',
  RST:'Rubrospinal',AR:'Acoustic radiation',CBT:'Corticobulbar',
};
const bundleLabel=id=>`${bundleNames[id.replace(/_[LR]$/,'')]||id}${id.endsWith('_L')?' · left':id.endsWith('_R')?' · right':''}`;
const option=(value,text)=>{const n=document.createElement('option');n.value=value;n.textContent=text;return n;};
function syncURL(){const p=new URLSearchParams(location.search);p.set('profile',profile);
  p.set('hemi',$('atlasHemisphere').value);const [h,id]=$('atlasArea').value.split(':');p.set('area',id||'8');p.set('areaHemi',h||'L');
  history.replaceState(history.state,'',`${location.pathname}?${p}`);}
function motionUI(){const canPlay=profile==='teaching'&&!matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('tracePlay').disabled=!canPlay;$('tracePlay').textContent=playing&&canPlay?'Pause display trace':'Play display trace';
  $('tracePlay').setAttribute('aria-pressed',String(playing&&canPlay));
  $('traceNote').textContent=profile==='presenter'?'Presenter · static reference anatomy.':!canPlay?'Reduced motion · static reference anatomy.':
    `${playing?'Playing':'Paused'} · paired trace follows shape; no direction or conduction.`;
  scene?.setPlaying(playing&&canPlay);
}
function setProfile(value){profile=value==='presenter'?'presenter':'teaching';scene?.setProfile(profile);motionUI();syncURL();}
function readRegion(hemi,id){
  const region=regionIdentity(scene.surfaceMeta,hemi,id);scene.select(hemi,id);
  $('pickedClass').textContent='ATLAS PARCEL · HCP–MMP1';
  $('pickedName').textContent=`${hemi==='L'?'Left':'Right'} · ${id===0?'medial wall':`area ${region.code}`}`;
  $('pickedDescription').textContent=id===0?'Unlabelled medial wall in this atlas.':
    `${region.raw} · Group reference boundary. Individual function and tract endpoints require separate evidence.`;
  $('atlasArea').value=`${hemi}:${id}`;syncURL();
}
function showDeep(entry){
  $('pickedClass').textContent='ATLAS STRUCTURE · MELBOURNE S1';
  $('pickedName').textContent=`${entry.hemisphere==='L'?'Left':'Right'} · ${entry.name}`;
  $('pickedDescription').textContent=entry.name==='Thalamus'?'Merged thalamic atlas territory. No specific relay nucleus or circuit is identified.':
    'Group reference surface. Its appearance does not establish a functional state or connection.';
}
/** Apply one or more bundles at once (v2 `bundles`+`ghost`); a single id keeps the exact
 * v1 pathway-summary text. Owns `selectedBundles` (pick order = tint order) and syncs the chips. */
function applyBundles(ids,ghostIds=[]){
  selectedBundles=[...new Set(ids)].slice(0,MAX_BUNDLES);currentGhosts=[...new Set(ghostIds)].filter(id=>!selectedBundles.includes(id));
  ids=selectedBundles;ghostIds=currentGhosts;
  scene.setBundles(ids,{ghost:ghostIds});
  syncPicker();
  if(!ids.length&&!ghostIds.length){
    $('pathwaySummary').textContent='Cortical reference · no pathway displayed';return;
  }
  const metaOf=id=>scene.tractMeta.bundles.find(x=>x.id===id);
  const ghostNote=ghostIds.length?`${ghostIds.length} ghosted for context`:'';
  let summary;
  if(ids.length===1){const b=metaOf(ids[0]);summary=b?`${bundleLabel(ids[0])} · ${b.lines} sampled atlas paths`:bundleLabel(ids[0]);}
  else if(ids.length){
    // Several primaries: one total, then the names in pick (= tint) order — the per-bundle
    // "220 sampled atlas paths" repeated N times was unreadable past three picks.
    const total=ids.reduce((n,id)=>n+(metaOf(id)?.lines||0),0);
    summary=`${ids.length} pathways · ${total.toLocaleString('en-GB')} sampled atlas paths · ${ids.map(bundleLabel).join(', ')}`;
  }
  // Ghost-only steps (no primary bundle) must not open with a dangling separator.
  $('pathwaySummary').textContent=summary
    ? summary+(ghostNote?` · ${ghostNote}`:'')
    : `Cortical reference · ${ghostNote}`;
}
function showBundle(id){applyBundles(id?[id]:[]);}
/** Reflect the selection on the chips: pressed state, tint swatch in pick order, count, limit. */
function syncPicker(){
  const full=selectedBundles.length>=MAX_BUNDLES;
  for(const chip of document.querySelectorAll('#pathwayGroups .bundle-chip')){
    const i=selectedBundles.indexOf(chip.dataset.bundle),on=i>=0;
    chip.setAttribute('aria-pressed',String(on));chip.style.setProperty('--chip',on?tintCss(tintForIndex(i)):'');
    chip.disabled=!on&&full;
    chip.title=on?'Click to remove':full?`Limit of ${MAX_BUNDLES} pathways reached`:'Click to add';
  }
  $('pathwayCount').textContent=selectionCaption(selectedBundles.length);
  $('pathwayClear').disabled=!selectedBundles.length;
}
/** Manual chip click: toggle in the ordered set. A ghosted bundle that gets picked is promoted. */
function togglePathway(id){
  pauseForExploration();
  const next=toggleBundle(selectedBundles,id);
  if(next===selectedBundles)return; // limit refused the add; syncPicker already disabled the chip
  applyBundles(next,currentGhosts.filter(g=>g!==id));
}
function applyPathwayFilter(){
  const q=$('pathwayFilter').value;
  for(const group of document.querySelectorAll('#pathwayGroups .picker-group')){
    const chips=[...group.querySelectorAll('.bundle-chip')];
    const keep=new Set(filterBundles(chips.map(c=>({id:c.dataset.bundle,label:c.dataset.label})),q,b=>b.label).map(b=>b.id));
    let any=false;for(const c of chips){c.hidden=!keep.has(c.dataset.bundle);any=any||!c.hidden;}
    group.hidden=!any;
  }
}
/** Apply a resolved scene (from resolveScene) to the live atlas: bundles+ghost, the region
 * set (first = focus, rest = secondary highlight), deep structures, surface and camera. */
function applySceneEffects(resolved){
  applyBundles(resolved.bundleIds,resolved.ghostIds);
  if(resolved.regions.length){
    readRegion(resolved.regions[0].hemi,resolved.regions[0].id);
    scene.highlight(resolved.regions.slice(1));
  }else{
    scene.highlight([]);
  }
  scene.setDeep(resolved.deep);$('deepStructures').checked=resolved.deep;
  scene.setDeepHighlight(resolved.deepRegionIds);
  if(resolved.surface!=null){scene.setSurface(resolved.surface);$('surfaceLevel').value=String(Math.round(resolved.surface*100));}
  if(resolved.camera)scene.flyTo(resolved.camera);
  else if(resolved.view)scene.setView(resolved.view==='top'?'superior':resolved.view);
}
function applyStep(lesson,step){
  if(lesson&&!currentLesson)exploration=scene.snapshot();
  currentStep=step;currentLesson=lesson;if(!step){lastResolvedTrace=false;return;}
  const resolved=resolveScene(step,currentHemisphere());
  applySceneEffects(resolved);
  lastResolvedTrace=resolved.trace;playing=lastResolvedTrace;motionUI();
}
function pauseForExploration(){
  if(player?.controller.state.status==='playing')player.controller.pause();
  playing=false;motionUI();
}
function restoreExploration(){
  if(!exploration)return;scene.restore(exploration);
  $('atlasHemisphere').value=exploration.hemisphere;$('surfaceLevel').value=String(exploration.surface*100);
  $('deepStructures').checked=exploration.deepVisible;readRegion(exploration.selected.hemi,exploration.selected.id);
  applyBundles(exploration.bundles||[],exploration.ghostBundles||[]);playing=false;motionUI();exploration=null;
}
function toggleLesson(open){$('atlasWorkspace').dataset.lessonHidden=String(!open);
  $('toggleLesson').textContent=open?'Hide lesson':'Show lesson';$('toggleLesson').setAttribute('aria-expanded',String(open));
  if(!open)player?.controller.pause();}
$('toggleLesson').addEventListener('click',()=>toggleLesson($('atlasWorkspace').dataset.lessonHidden==='true'));
$('tracePlay').addEventListener('click',()=>{playing=!playing;motionUI();});
matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',motionUI);
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{
  pauseForExploration();
  if(b.dataset.view==='medial'&&$('atlasHemisphere').value==='both'){
    $('atlasHemisphere').value='L';scene?.setHemisphere('L');syncURL();}
  scene?.setView(b.dataset.view);
}));

try{
  scene=await createAtlasScene($('atlasCanvas'),{onStatus:text=>{$('atlasLoading').textContent=text;},
    onPick:pick=>{pauseForExploration();pick.deep?showDeep(pick.deep):readRegion(pick.hemi,pick.id);},onInteraction:pauseForExploration});
  $('atlasLoading').hidden=true;
  const area=$('atlasArea');area.replaceChildren();
  for(const h of ['L','R']){const group=document.createElement('optgroup');group.label=h==='L'?'Left hemisphere':'Right hemisphere';
    for(const id of Object.keys(scene.surfaceMeta.sets.glasser.regions[h])){
      const r=regionIdentity(scene.surfaceMeta,h,Number(id));group.append(option(`${h}:${id}`,`${h} · ${r.code}`));}
    area.append(group);}area.disabled=false;
  area.addEventListener('change',()=>{const [h,id]=area.value.split(':');
    pauseForExploration();
    if($('atlasHemisphere').value!=='both'){$('atlasHemisphere').value=h;scene.setHemisphere(h);}readRegion(h,Number(id));});
  $('atlasHemisphere').value=initial.hemi;scene.setHemisphere(initial.hemi);
  $('atlasHemisphere').addEventListener('change',()=>{
    const h=$('atlasHemisphere').value;scene.setHemisphere(h);
    const stepSide=currentLesson&&currentStep?currentStep.scene.side:null;
    pauseForExploration();
    if(stepSide==='follow'){
      applyStep(currentLesson,currentStep);player?.refresh();
    }else{
      if(h!=='both'){readRegion(h,Number(area.value.split(':')[1]));scene.setView(h==='L'?'left':'right');}
      if(stepSide==='L'||stepSide==='R'){
        const note=` · this lesson's evidence is ${stepSide==='L'?'left':'right'}-sided`;
        const target=selectedBundles.length?$('pathwaySummary'):$('traceNote');
        if(!target.textContent.includes(note))target.textContent+=note;
      }
    }
    syncURL();});
  // Multi-bundle chip picker, grouped by the atlas manifest's own `group` field (Association/
  // Cerebellum/Commissural/Cranial nerve/Projection) so every shipped bundle is reachable. No
  // separate "brainstem" bucket exists in the shipped metadata (CBT/DRTT/RST/ML are Projection).
  const groups=$('pathwayGroups');groups.replaceChildren();
  const pathwayGroups=new Map();
  for(const b of scene.tractMeta.bundles){
    if(!pathwayGroups.has(b.group)){const g=document.createElement('div');g.className='picker-group';
      const title=document.createElement('span');title.textContent=b.group;g.append(title);pathwayGroups.set(b.group,g);groups.append(g);}
    const chip=document.createElement('button');chip.type='button';chip.className='bundle-chip';chip.dataset.bundle=b.id;
    chip.dataset.label=bundleLabel(b.id);chip.setAttribute('aria-pressed','false');
    const dot=document.createElement('i');chip.append(dot,document.createTextNode(bundleLabel(b.id)));
    chip.addEventListener('click',()=>togglePathway(b.id));pathwayGroups.get(b.group).append(chip);
  }
  $('pathwayFilter').disabled=false;$('pathwayFilter').addEventListener('input',applyPathwayFilter);
  $('pathwayClear').addEventListener('click',()=>{pauseForExploration();applyBundles([],currentGhosts);});
  $('surfaceLevel').addEventListener('input',()=>{pauseForExploration();scene.setSurface(Number($('surfaceLevel').value)/100);});
  $('deepStructures').addEventListener('change',()=>{pauseForExploration();scene.setDeep($('deepStructures').checked);});
  const sub=scene.subMeta;
  for(const d of sub.structures)$('deepRegion').append(option(d.id,`${d.hemisphere} · ${d.name}`));
  $('deepRegion').addEventListener('change',()=>{const entry=sub.structures.find(d=>d.id===$('deepRegion').value);if(!entry)return;
    pauseForExploration();
    $('deepStructures').checked=true;scene.setDeep(true);showDeep(entry);});
  const requestedH=new URLSearchParams(location.search).get('areaHemi');
  readRegion(['L','R'].includes(requestedH)?requestedH:initial.hemi==='R'?'R':'L',initial.area);
  showBundle(initial.hemi==='R'?'CST_R':'CST_L');setProfile(profile);
  if(!new URLSearchParams(location.search).has('lesson')){
    const p=new URLSearchParams(location.search);p.set('lesson','motor-cst');history.replaceState(history.state,'',`${location.pathname}?${p}`);
  }
  player=mountLessonPlayer($('lessonPanel'),{
    profiles:[['presenter','Presenter'],['teaching','Teaching']],getProfile:()=>profile,
    onProfile:setProfile,onStep:applyStep,onPlayback:value=>{playing=value||lastResolvedTrace;motionUI();},
    onRestore:()=>{pauseForExploration();applyStep(currentLesson,currentStep);},onExplore:restoreExploration,
    onClose:()=>{toggleLesson(false);$('toggleLesson').focus();},
    atlasCapability:()=>({message:'HCP-MMP1 parcels and gross Melbourne subcortical structures are installed. Functional territories, precise relay nuclei and proposed circuit edges are not mapped.'}),
    capability:(group,side)=>({available:true,message:`Reference atlas sample · ${bundleLabel(`${group==='or'?'OR':group.toUpperCase()}_${side||currentHemisphere()}`)}. No case support or proximity metric.`}),
    candidateLabel:'Show atlas pathway',onCandidate:(group,side)=>showBundle(`${group==='or'?'OR':group.toUpperCase()}_${side||currentHemisphere()}`),
  });
  scene.setProfile(profile);motionUI();
  window.addEventListener('popstate',()=>{const s=atlasSelectionFromSearch(location.search);
    profile=s.profile;scene.setProfile(profile);$('atlasHemisphere').value=s.hemi;scene.setHemisphere(s.hemi);
    readRegion(s.hemi==='R'?'R':'L',s.area);player.controller.restore(location.search);});
  if(new URLSearchParams(location.search).has('test'))Object.defineProperty(window,'__atlasTest',{get:()=>({...scene.state,lesson:player.controller.state,
    // Test-only: inject a v2 declarative scene directly, bypassing lesson content, so Playwright
    // can exercise SCENE GRAMMAR v2 (bundles/ghost/regions/camera/deepRegions) in isolation.
    applyScene:sceneInput=>applySceneEffects(resolveScene({scene:{...DEFAULT_SCENE,...sceneInput}},currentHemisphere())),
    togglePathway,selectedBundles:[...selectedBundles]})});
  window.addEventListener('pagehide',()=>{scene.dispose();player.dispose();},{once:true});
}catch(error){
  $('atlasLoading').hidden=false;$('atlasLoading').textContent=`Reference atlas unavailable: ${error.message}. Reload to retry.`;
  $('tracePlay').disabled=true;console.error(error);
}
