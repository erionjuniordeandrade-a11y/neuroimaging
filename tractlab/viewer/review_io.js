/** Local file interaction; state and source verification belong to the caller. */
import {createReviewRecord,parseReviewRecord,assertReviewContext,reviewSummary,restoreStatusLines,
  REVIEW_LIMITS,OUTCOME_EFFECTS,OUTCOME_LABELS,OUTCOME_QUESTION} from './review_record.js';

// Reviewer self-report saved with the record (schema 3). It sits beside Save so
// it is answered at export; a reopened record puts its answer back here so a
// re-save never drops it. Plain wording, no clinical claim.
function installOutcomeControl(save){
  const wrap=document.createElement('span'),label=document.createElement('label'),select=document.createElement('select'),note=document.createElement('input');
  wrap.className='review-outcome';wrap.style.cssText='display:inline-flex;align-items:center;gap:6px;font-size:var(--fs-caption)';
  label.htmlFor=select.id='reviewOutcome';label.textContent=OUTCOME_QUESTION;
  for(const effect of OUTCOME_EFFECTS){const option=document.createElement('option');option.value=effect;option.textContent=OUTCOME_LABELS[effect];select.append(option);}
  note.type='text';note.id='reviewOutcomeNote';note.maxLength=500;note.placeholder='Note (optional)';
  note.setAttribute('aria-label','Outcome note (optional)');note.style.width='14em';
  wrap.append(label,select,note);save.parentNode.insertBefore(wrap,save);
  const write=outcome=>{select.value=outcome?.effect??'not-recorded';note.value=outcome?.note??'';};
  write(null);
  return {read:()=>({effect:select.value,note:note.value}),write,disable:on=>{select.disabled=note.disabled=on;}};
}

export function installReviewIO({capture,context,prepare,commit,setBusy,validateRuntime}){
  const $=id=>document.getElementById(id),save=$('btnSaveReview'),open=$('btnOpenReview'),file=$('reviewFile'),status=$('reviewStatus');
  let busy=false,ready=false;
  const outcome=installOutcomeControl(save);
  const message=(text,error=false)=>{
    status.replaceChildren(document.createTextNode(text));status.hidden=false;status.dataset.state=error?'error':'ok';
    const close=document.createElement('button');close.type='button';close.textContent='Dismiss';close.className='review-dismiss';
    close.addEventListener('click',()=>status.hidden=true);status.append(close);
  };
  const refresh=()=>{save.disabled=open.disabled=!ready||busy;outcome.disable(!ready||busy);};
  const pending=on=>{busy=on;setBusy(on);refresh();};
  document.addEventListener('keydown',event=>{if(busy){event.preventDefault();event.stopImmediatePropagation();}},true);
  const download=(text,type,name)=>{
    const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
  };
  // Per-object restore outcome, in the same status panel as everything else.
  const showRestoreStatus=statuses=>{
    if(!statuses?.length)return;
    const lines=restoreStatusLines(statuses);
    const refused=statuses.filter(item=>!item.restored).length;
    const heading=document.createElement('p');
    heading.className='review-restore-heading';
    heading.textContent=refused
      ? `${statuses.length-refused} of ${statuses.length} restored, ${refused} refused`
      : `All ${statuses.length} restored`;
    const list=document.createElement('ul');
    list.className='review-restore-list';
    for(const [index,line] of lines.entries()){
      const item=document.createElement('li');
      item.textContent=line;
      item.dataset.state=statuses[index].restored?'ok':'refused';
      list.append(item);
    }
    status.append(heading,list);
  };
  const showSummary=(record,text,statuses=null)=>{
    message(text);
    showRestoreStatus(statuses);
    const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre'),button=document.createElement('button');
    summary.textContent='Measurement and evidence summary';pre.textContent=reviewSummary(record);
    button.type='button';button.textContent='Download summary';button.addEventListener('click',()=>download(pre.textContent,'text/plain',`tractlab-${record.context.caseId}-review.txt`));
    details.append(summary,pre,button);status.append(details);
  };
  save.addEventListener('click',async()=>{
    if(busy||!ready)return;
    try{
      pending(true);message('Checking review sources…');
      await validateRuntime();
      const record=createReviewRecord({...capture(),outcome:outcome.read()});
      // Use the same strict reader as reopening before offering a file.
      const json=JSON.stringify(record,null,2);parseReviewRecord(json);
      download(json,'application/json',`tractlab-${record.context.caseId}-review.json`);
      showSummary(record,'Review saved locally. The file includes this measurement and evidence summary.');
    }catch(error){message(`Review not saved. ${error.message}`,true);}
    finally{pending(false);}
  });
  open.addEventListener('click',()=>{if(!busy&&ready)file.click();});
  file.addEventListener('change',async()=>{
    const selected=file.files?.[0];file.value='';if(!selected||busy||!ready)return;
    try{
      pending(true);message('Checking the saved case and named tract sources…');
      if(selected.size>REVIEW_LIMITS.maxRecordBytes)throw new Error('The review file is too large.');
      const record=parseReviewRecord(await selected.text());
      await validateRuntime();assertReviewContext(record,context());
      const staged=await prepare(record);
      const statuses=await commit(record,staged);
      outcome.write(record.outcome);
      const refused=(statuses||[]).filter(item=>!item.restored).length;
      showSummary(record,refused
        ? 'Review reopened. Some saved objects were refused and are not drawn.'
        : 'Review reopened from matching sources. Measurements on the cards were read again from the case.',
        statuses);
    }catch(error){message(`Review not opened. ${error.message}`,true);}
    finally{pending(false);}
  });
  return {setReady:on=>{ready=on;refresh();},message};
}
