/** DOM structure only. Tract identity and measurements stay with the viewer state. */
export function mountWorkstation() {
  const $=id=>document.getElementById(id);
  const node=(tag,id,text)=>{const el=document.createElement(tag); if(id)el.id=id; if(text)el.textContent=text; return el;};
  const main=$('main'),side=$('side'),scene=$('scene');
  main.setAttribute('role','main'); scene.setAttribute('aria-label','Three-dimensional case view');
  const workspace=node('div','viewWorkspace'); scene.before(workspace);
  const compactTracts=node('div','compactTracts');compactTracts.setAttribute('role','group');compactTracts.setAttribute('aria-label','Tract measurements');
  workspace.append(compactTracts,scene);
  const provenance=node('footer','provenanceBar');provenance.setAttribute('aria-label','Case provenance');
  provenance.append(node('span',null,'Provenance'),$('provStrip'));$('app').append(provenance);
  $('chrome').insertBefore($('provChipResearch'),$('chromeHelp'));
  const identity=node('div','runtimeIdentity','Opening case…');identity.setAttribute('aria-label','Case and build');
  $('chrome').insertBefore(identity,$('provChipResearch'));
  const reviewTools=node('div','reviewTools');reviewTools.setAttribute('role','group');reviewTools.setAttribute('aria-label','Saved review');
  for(const [id,text]of [['btnSaveReview','Save review'],['btnOpenReview','Open review']]){const button=node('button',id,text);button.type='button';button.disabled=true;reviewTools.append(button);}
  const reviewFile=node('input','reviewFile');reviewFile.type='file';reviewFile.accept='.json,application/json';reviewFile.hidden=true;reviewTools.append(reviewFile);
  $('chrome').insertBefore(reviewTools,$('provChipResearch'));
  const reviewStatus=node('div','reviewStatus');reviewStatus.setAttribute('role','status');reviewStatus.setAttribute('aria-live','polite');reviewStatus.hidden=true;workspace.append(reviewStatus);
  provenance.append($('provCard'));
  const nav=node('nav','railNav'); nav.setAttribute('aria-label','Workstation sections');
  nav.setAttribute('role','tablist'); side.insertBefore(nav,$('secBanks'));
  const closeRail=node('button','btnCloseRail','Close controls');closeRail.type='button';side.prepend(closeRail);
  const sections={tracts:node('section','panelTracts'),display:node('section','panelDisplay'),tools:node('section','panelTools')};
  const showPanel=key=>{
    for(const [name,panel] of Object.entries(sections)){
      panel.hidden=name!==key; const btn=$(`tab-${name}`); btn.setAttribute('aria-selected',String(name===key)); btn.tabIndex=name===key?0:-1;
    }
    try{sessionStorage.setItem('tractlab.panel',key);}catch{}
  };
  for(const [key,label] of [['tracts','Tracts'],['display','Display'],['tools','Tools']]){
    const btn=node('button',`tab-${key}`,label); btn.type='button'; btn.setAttribute('role','tab'); btn.setAttribute('aria-controls',sections[key].id);
    btn.addEventListener('click',()=>showPanel(key));
    btn.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return; e.preventDefault(); const keys=Object.keys(sections),i=keys.indexOf(key); const next=keys[(i+(e.key==='ArrowRight'?1:2))%3]; showPanel(next); $(`tab-${next}`).focus();});
    nav.append(btn); const panel=sections[key]; panel.className='rail-panel'; panel.setAttribute('role','tabpanel'); panel.setAttribute('aria-labelledby',btn.id); side.append(panel);
  }
  const intersections=node('details','connectotomyDetails');intersections.append(node('summary',null,'Lesion intersections'),$('connectotomyHost'));
  // Same dock chrome as the C1 panel. Open by default: the profile is the
  // focused tract's own evidence, not a secondary report about other layers.
  const profile=node('details','profileDetails');profile.open=true;profile.append(node('summary',null,'Along-tract profile'),$('profileHost'));
  const connections=node('details','connectomeDetails');connections.append(node('summary',null,'Connections'),$('connectomeHost'));
  const focus=node('div','focusDetails'); focus.setAttribute('aria-label','Focused tract details'); sections.tracts.append(focus,$('secBanks'),intersections,profile,connections,$('priorPanel'));
  const inspector=$('streamlineInspector');sections.tracts.prepend(inspector);
  const clearSelection=node('button','btnClearStreamline','×');clearSelection.type='button';clearSelection.setAttribute('aria-label','Clear streamline selection');inspector.prepend(clearSelection);
  const locateSelection=node('button','btnLocateStreamline','Locate in MRI');locateSelection.type='button';locateSelection.className='ghost';
  locateSelection.title='Centre the slices on this streamline’s nearest displayed point to the lesion';inspector.append(locateSelection);
  sections.display.append($('secView'));
  sections.tools.append($('secRecovery'),$('advLive'),$('secStatus'));
  $('secBanks').querySelector('summary').textContent='Choose tracts';
  $('secBanks').querySelector('.section-title').remove();
  for(const [id,title] of [['secView','Display'],['secRecovery','Recovery'],['secStatus','Activity']]){
    const old=$(id).querySelector('.section-title'); if(old){const heading=node('h2',null,title); old.replaceWith(heading);}
  }
  // Keep scene controls together, and move analytical tools beside their focused identity.
  const research=node('details','researchTools'),researchSummary=node('summary',null,'Focused tract tools'); research.append(researchSummary);
  const view=$('secView'),start=$('btnExport').parentElement;
  const exportLabel=start.previousElementSibling; if(exportLabel?.classList.contains('section-hint'))exportLabel.remove();
  const end=$('layerToggles'); let item=start;
  while(item&&item!==end){const next=item.nextElementSibling; research.append(item);item=next;}
  const bias=$('displayNearLesion').closest('label'); research.append(bias);
  sections.tools.prepend(research);
  $('btnAllSupport').textContent='All'; $('btnHideLowSupport').textContent='Higher'; $('btnOnlyLowSupport').textContent='Lower'; $('evidenceLabel').textContent='Measured support';
  const evidenceNote=node('p','evidenceAvailability');evidenceNote.className='section-hint';evidenceNote.hidden=true;$('btnAllSupport').parentElement.after(evidenceNote);
  for(const el of document.querySelectorAll('.step'))el.remove();
  const fibreLabel=document.querySelector('label[for="fibre"]'); fibreLabel.childNodes[0].textContent='Tube radius ';
  fibreLabel.querySelector('u').textContent=' mm, display only';
  research.append($('toolHint'));
  $('bankBlindSpotDetails').hidden=true;
  $('recoveryHint').textContent='Search the recorded whole-brain CSD corpus within the chosen distance. Results form a separate, unnamed layer.';
  const persistentStatus=$('status').parentElement;persistentStatus.removeAttribute('aria-live');persistentStatus.removeAttribute('aria-atomic');
  $('status').setAttribute('role','status');$('status').setAttribute('aria-live','polite');
  $('recoveryStatus').setAttribute('role','status');
  // Consolidate routine limits here; individual controls retain hover help.
  document.querySelector('label[for="btnMargin"] span span')?.remove();
  document.querySelector('label[for="marginMm"] span[style]')?.remove();
  $('caseMap').querySelector('.map-footnote').textContent='p5: 5th percentile of minimum distances in the reconstructed tract. Recipes may miss fibres near edema.';
  const limits=node('details','limitsPanel'); limits.append(node('summary',null,'Limits of this display'),node('p',null,'Research preview; navigation requires a validated system. Tractography estimates pathways. A missing tract does not establish anatomical absence. p5 is a population statistic, not a functional or resection margin. The recorded floor is a display limit; it is not measured spatial accuracy. Local residual error is unmeasured.'));
  sections.tools.append(limits);
  const dock=node('aside','anatomyDock'); dock.setAttribute('aria-label','Linked anatomical slices'); workspace.append(dock);
  const divider=node('div','anatomyResize');divider.tabIndex=0;divider.setAttribute('role','separator');divider.setAttribute('aria-orientation','horizontal');divider.setAttribute('aria-controls','anatomyDock');divider.setAttribute('aria-label','Resize anatomy panel');divider.title='Drag to resize anatomy. Arrow keys adjust. Double-click resets.';dock.append(divider);
  const head=node('div','anatomyHead'); head.append(node('h2',null,'Anatomy'));
  const inspect=node('button','btnInspectMRI','Inspect MRI');inspect.type='button';inspect.className='ghost';
  inspect.setAttribute('aria-controls','anatomyDock');inspect.setAttribute('aria-expanded','false');head.append(inspect,$('underToggle'));
  const paint=node('button','btnPaintMode','Paint ROI'); paint.type='button'; paint.className='ghost'; paint.setAttribute('aria-pressed','false');
  const undo=node('button','btnUndoPaint','Undo'); undo.type='button'; undo.className='ghost'; undo.disabled=true;
  dock.append(head);
  const settings=node('div','sliceSettings'); const label=node('label',null,'3D slice '); label.htmlFor='slicePlane'; label.style.margin='0';
  const planes=node('select','slicePlane'); for(const [value,name] of [['off','Off'],['ax','Axial'],['cor','Coronal'],['sag','Sagittal']]){const o=node('option',null,name);o.value=value;planes.append(o);} planes.value='off';
  settings.append(label,planes);head.append(settings,paint,undo);
  const colourKey=node('span','sliceColourKey');colourKey.hidden=true;colourKey.setAttribute('aria-label','Slice direction colours');
  for(const [text,colour,full]of [['R/L','#ff3030','Right / left'],['A/P','#20ed45','Anterior / posterior'],['S/I','#5475ff','Superior / inferior']]){const key=node('span',null,text);key.title=full;const sw=node('i');sw.style.background=colour;key.prepend(sw);colourKey.append(key);}head.append(colourKey);
  const hint=node('span','mprMessage','Click to locate. Scroll to change slice.');hint.className='assistive';dock.append(hint);
  head.title='Click a slice to locate. Scroll to change slice. Paint ROI to draw.';
  const mpr=document.querySelector('.mpr'); mpr.querySelector('figure:last-child').remove(); dock.append(mpr);
  for(const [id,axis,labelText] of [['cax','ax','Axial'],['ccor','cor','Coronal'],['csag','sag','Sagittal']]){
    const figure=$(id).parentElement;figure.dataset.axis=axis;$(id).setAttribute('aria-describedby','mprMessage');const range=node('input',`slice-${axis}`); range.type='range'; range.min='0'; range.max='0';range.value='0';range.setAttribute('aria-label',`${labelText} slice`);
    const caption=figure.querySelector('figcaption'),captionText=node('span',caption.id,caption.textContent);captionText.className='slice-position';caption.removeAttribute('id');caption.className='slice-footer';
    const planeName=node('span',null,labelText);planeName.className='slice-plane-name';
    const expand=node('button',`expand-${axis}`,'Expand');expand.type='button';expand.setAttribute('aria-label',`Expand ${labelText.toLowerCase()} slice`);expand.setAttribute('aria-pressed','false');expand.title=`Enlarge ${labelText.toLowerCase()} for inspection and painting`;expand.addEventListener('click',()=>focusSlice(focusedSlice===axis?null:axis));caption.replaceChildren(planeName,captionText,range,expand);
  }
  const tools=node('nav','workstationTools');tools.setAttribute('aria-label','View tools');
  for(const [id,text] of [['btnRail','Tracts'],['btnSlices','Anatomy'],['btnDefaults','Defaults']]){const b=node('button',id,text);b.type='button';tools.append(b);}
  scene.append(tools);
  const scale=node('div','worldScale');scale.innerHTML='<div class="scale-line"></div><small></small>';scene.append(scale);
  $('orientationCube').replaceWith(document.createElementNS('http://www.w3.org/2000/svg','svg'));
  const compass=scene.querySelector('svg:last-of-type');compass.id='orientationCube';compass.setAttribute('viewBox','0 0 100 100');compass.setAttribute('role','img');compass.setAttribute('aria-label','Camera-linked anatomical orientation');
  $('mapScale').setAttribute('role','region');$('mapScale').setAttribute('aria-label','Colour and overlay legend');
  $('mapScale').hidden=false;
  // Fractions use the space below the reserved measurement strip. The existing
  // review layout record therefore captures compact and inspected states alike.
  const availableHeight=()=>Math.max(1,workspace.clientHeight-compactTracts.getBoundingClientRect().height);
  const compactFraction=()=>Math.min(.20,104/availableHeight());
  let dockFraction=compactFraction();
  let focusedSlice=null,overviewFraction=dockFraction;
  const dockBounds=()=>{const h=availableHeight();return {h,min:Math.min(88,h*.3),max:h*.68};};
  const sizeDock=()=>{
    const {h,min,max}=dockBounds();const px=Math.round(Math.max(min,Math.min(max,h*dockFraction)));
    workspace.style.setProperty('--anatomy-height',`${px}px`);
    // Compact images cap at 104px (CSS). At the 190px transition the full
    // toolbar/footer layout has at least that image height: growing the dock
    // must not shrink the actual slice. An uncapped compact layout cannot
    // satisfy this invariant merely by moving the threshold.
    const compact=!focusedSlice&&px<=190;
    dock.classList.toggle('anatomy-compact',compact);
    inspect.textContent=compact?'Inspect MRI':'Overview';inspect.setAttribute('aria-expanded',String(!compact));
    if(compact&&dock.classList.contains('paint-mode'))paint.click();
    divider.setAttribute('aria-valuemin',String(Math.round(min)));divider.setAttribute('aria-valuemax',String(Math.round(max)));divider.setAttribute('aria-valuenow',String(px));divider.setAttribute('aria-valuetext',`Anatomy ${px} pixels`);
  };
  const setDockSize=px=>{const {h,min,max}=dockBounds();dockFraction=Math.max(min,Math.min(max,px))/Math.max(1,h);sizeDock();};
  const focusSlice=axis=>{
    if(axis&&!['ax','cor','sag'].includes(axis))return;
    if(axis&&!focusedSlice){overviewFraction=dockFraction;dockFraction=Math.max(.68,dockFraction);}
    if(!axis&&focusedSlice)dockFraction=overviewFraction;
    focusedSlice=axis;dock.classList.toggle('slice-focused',!!axis);
    for(const figure of mpr.querySelectorAll('figure')){
      const ownAxis=figure.dataset.axis,button=$(`expand-${ownAxis}`),active=axis===ownAxis;
      figure.hidden=!!axis&&!active;button.textContent=active?'Overview':'Expand';button.setAttribute('aria-pressed',String(active));button.setAttribute('aria-label',active?'Return to overview':`Expand ${ownAxis==='ax'?'axial':ownAxis==='cor'?'coronal':'sagittal'} slice`);
    }
    toggleSlices(true);
  };
  const resetLayout=()=>{if(focusedSlice)focusSlice(null);dockFraction=compactFraction();sizeDock();};
  inspect.addEventListener('click',()=>{if(dock.classList.contains('anatomy-compact'))focusSlice('cor');else if(focusedSlice)focusSlice(null);else resetLayout();});
  const getLayout=()=>({slicesOpen:!dock.hidden,dockFraction,overviewFraction:focusedSlice?overviewFraction:dockFraction,focusedSlice});
  const restoreLayout=layout=>{
    focusSlice(layout.focusedSlice);
    dockFraction=layout.dockFraction;overviewFraction=layout.overviewFraction;
    sizeDock();toggleSlices(layout.slicesOpen);
  };
  let drag=null;
  divider.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();divider.focus();drag={y:e.clientY,height:dock.getBoundingClientRect().height};divider.setPointerCapture(e.pointerId);});
  divider.addEventListener('pointermove',e=>{if(drag)setDockSize(drag.height+drag.y-e.clientY);});
  divider.addEventListener('lostpointercapture',()=>drag=null);
  divider.addEventListener('pointerup',e=>{drag=null;if(divider.hasPointerCapture(e.pointerId))divider.releasePointerCapture(e.pointerId);});
  divider.addEventListener('dblclick',resetLayout);
  divider.addEventListener('keydown',e=>{
    if(!['ArrowUp','ArrowDown','Home','End'].includes(e.key))return;e.preventDefault();
    const {min,max}=dockBounds();setDockSize(e.key==='Home'?min:e.key==='End'?max:dock.getBoundingClientRect().height+(e.key==='ArrowUp'?24:-24));
  });
  const dockObserver=new ResizeObserver(sizeDock);dockObserver.observe(workspace);dockObserver.observe(compactTracts);
  const toggleSlices=(on)=>{dock.hidden=!on;workspace.classList.toggle('slices-open',on);$('btnSlices').setAttribute('aria-pressed',String(on));sizeDock();window.dispatchEvent(new Event('resize'));};
  $('btnSlices').addEventListener('click',()=>toggleSlices(dock.hidden));
  $('btnRail').setAttribute('aria-controls','side');$('btnRail').setAttribute('aria-expanded','false');
  // In docked mode (sidebar not a drawer) btnRail duplicates the rail tab, so it
  // is hidden there. btnSlices is a real toggle (show/hide the anatomy dock) and
  // stays in every mode; its label and aria-pressed state say what it does.
  const toolDrawerQuery=matchMedia('(max-width:1000px)');
  const updateToolVisibility=()=>{$('btnRail').hidden=!toolDrawerQuery.matches;};
  // A closed drawer is translated offscreen but stays keyboard/AT reachable
  // unless made inert. Keep it inert whenever it is acting as a drawer
  // (narrow layout) and is not open; docked mode is always interactive.
  const updateDrawerInert=()=>{side.toggleAttribute('inert',toolDrawerQuery.matches&&!document.body.classList.contains('rail-open'));};
  const closeControls=()=>{
    const hadFocusInside=toolDrawerQuery.matches&&side.contains(document.activeElement);
    document.body.classList.remove('rail-open');$('btnRail').setAttribute('aria-expanded','false');
    updateDrawerInert();
    if(hadFocusInside)$('btnRail').focus();
  };
  closeRail.addEventListener('click',closeControls);
  const openDrawer=()=>{document.body.classList.add('rail-open');$('btnRail').setAttribute('aria-expanded','true');updateDrawerInert();closeRail.focus();};
  const showInspector=()=>{
    showPanel('tracts');sections.tracts.scrollTop=0;
    if(toolDrawerQuery.matches)openDrawer();
  };
  $('btnRail').addEventListener('click',()=>{
    showPanel('tracts');
    if(toolDrawerQuery.matches){
      if(document.body.classList.contains('rail-open'))closeControls();else openDrawer();
    }
  });
  updateToolVisibility();updateDrawerInert();
  toolDrawerQuery.addEventListener('change',()=>{updateToolVisibility();updateDrawerInert();});
  // Other code (outside this module) also flips body.rail-open directly
  // (e.g. the compact tract bar opens the drawer on its own). Watch the class
  // itself so inert never desyncs from whichever code path opened/closed it.
  new MutationObserver(updateDrawerInert).observe(document.body,{attributes:true,attributeFilter:['class']});
  scene.addEventListener('pointerdown',e=>{if(!e.target.closest('#btnRail,#compactTracts'))closeControls();});
  // Keep the shared safety statement visible even when the drawer is closed.
  $('provChipResearch').textContent='Research only. Not for navigation.';
  // Single place that closes the help panel: keeps hidden/aria-expanded/focus
  // in agreement, however help gets closed (currently: Escape).
  const closeHelp=()=>{
    const howtoEl=$('howto'),helpBtn=$('chromeHelp');
    const wasOpen=!howtoEl.hidden;
    howtoEl.hidden=true;helpBtn.setAttribute('aria-expanded','false');
    if(wasOpen)helpBtn.focus();
  };
  window.addEventListener('keydown',e=>{
    const editing=/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||e.target.isContentEditable;
    if(e.target.type==='range'&&['[',']'].includes(e.key)){e.preventDefault();e.target[e.key==='['?'stepDown':'stepUp']();e.target.dispatchEvent(new Event('input',{bubbles:true}));e.target.dispatchEvent(new Event('change',{bubbles:true}));return;}
    if(e.key==='Escape'){closeControls();closeHelp();$('provCard').hidden=true;}
    if(e.ctrlKey||e.metaKey||e.altKey)return;
    if(!editing&&e.key.toLowerCase()==='t')$('btnRail').click();
    if(!editing&&e.key.toLowerCase()==='m')$('btnSlices').click();
    if(!editing&&e.key==='?')$('chromeHelp').click();
  });
  const help=$('howto');help.innerHTML='<h2>Workstation keys</h2><p><kbd>1</kbd> direction, <kbd>2</kbd> tract, <kbd>3</kbd> distance<br><kbd>R</kbd> reset camera, <kbd>M</kbd> slices, <kbd>T</kbd> tract drawer<br><kbd>Alt 1–6</kbd> named views, <kbd>Esc</kbd> close panels<br><kbd>Ctrl/Cmd Z</kbd> undo paint</p><p>Click a slice to locate a point in all three planes. Select Paint ROI to draw. Arrow keys or [ / ] adjust a focused slider.</p>';
  scene.append(help);
  // Convert source-authored tips into native hover/focus explanations too.
  for(const el of document.querySelectorAll('[data-tip]'))if(!el.title)el.title=el.dataset.tip;
  let initial='tracts';try{const saved=sessionStorage.getItem('tractlab.panel');if(saved in sections)initial=saved;}catch{}
  showPanel(initial);toggleSlices(true);
  // Overflow cue: rail panels sit in a fixed-height sidebar; mark when more content lies below.
  const markOverflow=panel=>{panel.dataset.more=String(panel.scrollHeight-panel.clientHeight-panel.scrollTop>4);};
  for(const panel of Object.values(sections)){panel.addEventListener('scroll',()=>markOverflow(panel),{passive:true});new ResizeObserver(()=>markOverflow(panel)).observe(panel);}
  return {showPanel,toggleSlices,resetLayout,focusSlice,showInspector,getLayout,restoreLayout};
}

export function addNumericControls(){
  for(const range of document.querySelectorAll('input[type="range"]:not([id^="slice-"])')){
    if(range.closest('.range-control'))continue;
    const wrap=document.createElement('div');wrap.className='range-control';range.before(wrap);wrap.append(range);
    // Native range semantics follow the live min/max; duplicated ARIA bounds
    // would become stale when case-specific display limits are applied.
    range.removeAttribute('aria-valuemin');range.removeAttribute('aria-valuemax');
    const labelNode=range.labels?.[0]?.cloneNode(true);labelNode?.querySelectorAll('.val').forEach(el=>el.remove());
    const label=(labelNode?.textContent||range.getAttribute('aria-label')||range.id).replace(/\s+/g,' ').trim();
    const number=document.createElement('input');number.type='number'; number.setAttribute('aria-label',`${label} exact value`);
    number.lang=document.documentElement.lang||'en';
    // The exact field is the single visible value. Native decimal formatting
    // follows the browser locale; a duplicate text label could disagree.
    range.labels?.[0]?.querySelectorAll('.val').forEach(el=>{el.hidden=true;el.setAttribute('aria-hidden','true');});
    for(const name of ['min','max','step'])number[name]=range[name];number.value=range.value;number.disabled=range.disabled;
    range.addEventListener('input',()=>number.value=range.value);
    number.addEventListener('change',()=>{if(!number.checkValidity()||number.value===''){number.value=range.value;return;}range.value=number.value;range.dispatchEvent(new Event('input',{bubbles:true}));range.dispatchEvent(new Event('change',{bubbles:true}));});
    const reset=document.createElement('button');reset.type='button';reset.textContent='↺';reset.title='Reset to default';reset.setAttribute('aria-label',`Reset ${label}`);reset.disabled=range.disabled;
    reset.addEventListener('click',()=>{range.value=range.defaultValue;range.dispatchEvent(new Event('input',{bubbles:true}));range.dispatchEvent(new Event('change',{bubbles:true}));});
    const defaults=document.createElement('datalist');defaults.id=`defaults-${range.id}`;const tick=document.createElement('option');tick.value=range.defaultValue;tick.label='Default';defaults.append(tick);range.setAttribute('list',defaults.id);
    new MutationObserver(()=>{number.disabled=range.disabled;reset.disabled=range.disabled;tick.value=range.defaultValue; for(const n of ['min','max','step'])number[n]=range[n];}).observe(range,{attributes:true,attributeFilter:['disabled','min','max','step','value']});
    wrap.append(number,reset,defaults);
  }
}
