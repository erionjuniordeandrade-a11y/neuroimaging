import { LESSONS, CONTENT_VERSION } from './lesson_content.js';

export function lessonStateFromSearch(search=''){
  const params=new URLSearchParams(search);
  const id=params.get('lesson');
  const lesson=LESSONS.find(l=>l.id===id);
  if(!id) return Object.freeze({status:'inactive',lessonId:null,step:0,error:''});
  if(!lesson) return Object.freeze({status:'inactive',lessonId:null,step:0,error:'That lesson is not installed.'});
  const version=params.get('lessonVersion');
  if(version && version!==lesson.version) return Object.freeze({status:'inactive',lessonId:null,step:0,error:'This lesson link uses a different content version. Choose the installed draft.'});
  const raw=Number(params.get('step')||0);
  const step=Number.isInteger(raw) && raw>=0 && raw<lesson.steps.length ? raw : 0;
  return Object.freeze({status:'paused',lessonId:id,step,error:''});
}

export function lessonSearch(search,state){
  const params=new URLSearchParams(search);
  for(const key of ['lesson','lessonVersion','step','playing']) params.delete(key);
  if(state.lessonId){
    params.set('lesson',state.lessonId);params.set('lessonVersion',CONTENT_VERSION);params.set('step',String(state.step));
  }
  const encoded=params.toString();return encoded?`?${encoded}`:'';
}

export function createLessonController({search='',onChange=()=>{}}={}){
  let state=lessonStateFromSearch(search);
  const emit=next=>{state=Object.freeze(next);onChange(state);return state;};
  return {
    get state(){return state;},
    restore(search){return emit(lessonStateFromSearch(search));},
    select(id){const lesson=LESSONS.find(l=>l.id===id);if(!lesson)throw new Error('Unknown lesson');
      return emit({status:'paused',lessonId:id,step:0,error:''});},
    move(delta){if(!state.lessonId)return state;const lesson=LESSONS.find(l=>l.id===state.lessonId);
      return emit({...state,step:Math.max(0,Math.min(lesson.steps.length-1,state.step+delta)),status:'paused'});},
    play(){return state.lessonId?emit({...state,status:'playing'}):state;},
    pause(){return state.lessonId?emit({...state,status:'paused'}):state;},
    close(){return emit({status:'inactive',lessonId:null,step:0,error:''});},
  };
}
