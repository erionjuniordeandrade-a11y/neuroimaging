/** Original educational drafts derived from the reviewed audit source pack.
 * No atlas meshes, patient images, functional coordinates or copied figures.
 * A source supports only its stated claim, modality and population.
 */
import {BUNDLE_FAMILIES,SUBCORTEX_IDS} from './atlas_data.js';
import {MOTOR_CST_V2,MOTOR_CST_V2_SOURCES,MOTOR_CST_V2_REGIONS} from './lessons/motor-cst.v2.js';
export const CONTENT_VERSION='2026-09-07.1';
export const EVIDENCE_CLASSES=Object.freeze(['atlas','reconstruction','association',
  'schematic','recovery_overlay','functional_measurement','model_metric',
  'experimental_anatomy','conceptual_model']);

const source=(id,title,url,evidenceClass,scope)=>Object.freeze({id,title,url,evidenceClass,scope});
const SOURCES_V1={
  S1:source('S1','Glasser et al., 2016 · multimodal cortical parcellation','https://doi.org/10.1038/nature18933','atlas','HCP population reference, combining cortical features. Atlas boundaries do not establish an individual functional boundary.'),
  S2:source('S2','Gordon et al., 2023 · motor and action organization','https://doi.org/10.1038/s41586-023-05964-2','functional_measurement','Precision functional MRI and connectivity. Effector and inter-effector observations are task and population dependent.'),
  S3:source('S3','Usuda et al., 2022 · cortical contributions to CST','https://doi.org/10.1016/j.neures.2022.06.008','reconstruction','Healthy-volunteer tractography. Streamline contributions depend on model, acquisition and ROI selection; they are not axon counts.'),
  S4:source('S4','Catani et al., 2013 · FAT and verbal fluency','https://doi.org/10.1093/brain/awt163','association','Primary progressive aphasia cohorts and controls; diffusion measures related to language tests. A cohort association is not a universal deficit prediction.'),
  S5:source('S5','Chernoff et al., 2019 · sentence planning and stimulation','https://doi.org/10.1080/02643294.2019.1619544','functional_measurement','One awake-mapping case near a tractography-defined left FAT. Planning and articulation were measured separately; no stimulation location is transferred to this scene.'),
  S6:source('S6',"Nilsson et al., 2007 · Meyer’s loop variation",'https://doi.org/10.1016/j.eplepsyres.2007.07.012','reconstruction','Small DTI study of anterior extent. Its method and sample do not define a universal millimetre boundary.'),
  S7:source('S7','Maier-Hein et al., 2017 · tractography validation challenge','https://doi.org/10.1038/s41467-017-01285-x','reconstruction','Challenge data with defined ground truth and participating methods. Its error rates are not an estimate of this dataset’s error rate.'),
  S8:source('S8','Smith et al., 2015 · SIFT2','https://doi.org/10.1016/j.neuroimage.2015.06.092','model_metric','Model-derived streamline contributions to a fibre-density fit. Weights are not probabilities that entire connections are true.'),
  S9:source('S9','Warrington et al., 2020 · XTRACT protocols','https://doi.org/10.1016/j.neuroimage.2020.116923','reconstruction','Standardized tractography recipes with species and method-specific constraints. A protocol label alone does not verify a subject’s pathway.'),
  S10:source('S10','Schaefer et al., 2018 · resting-state parcellation','https://doi.org/10.1093/cercor/bhx179','atlas','Population parcellation using resting-state functional connectivity. Label identity belongs to the specified release and table.'),
  S11:source('S11','Craig, 2002 · interoception framework','https://doi.org/10.1038/nrn894','conceptual_model','Review and conceptual synthesis, not a primary tractography or timing experiment.'),
  S12:source('S12','Mesulam & Mufson, 1982 · insular cortical outputs','https://cpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/7/2577/files/2018/08/14-Mesulam-2o4vkfe.pdf','experimental_anatomy','Tracer experiments in old-world monkeys. Species and experimental scope limit translation to an individual human reconstruction.'),
};
export const SOURCES=Object.freeze({...SOURCES_V1,...MOTOR_CST_V2_SOURCES});

const REGIONS_V1={
  area4:{name:'Area 4 reference',text:'Area 4 in HCP-MMP1 is a population reference label. The separate atlas view preserves its paired hemisphere table; case registration and individual function remain separate questions.',sources:['S1'],evidenceClass:'atlas'},
  motor:{name:'Effector / inter-effector territories',text:'Functional organization described by precision fMRI. These territories are discussed here; they are not painted onto the reconstruction.',sources:['S2'],evidenceClass:'functional_measurement'},
  cst:{name:'Corticospinal candidate',text:'A recipe-selected descending reconstruction. Read its actual side, extent and source rows; it does not specify a single body part.',sources:['S3','S9'],evidenceClass:'reconstruction'},
  frontal:{name:'Inferior and medial frontal relationship',text:'Reference anatomy for the FAT discussion. A schematic endpoint is not a measured termination of a displayed streamline.',sources:['S4'],evidenceClass:'schematic'},
  fat:{name:'Frontal aslant candidate',text:'The language examples concern the cited left-sided experiments. Changing hemisphere does not mirror those claims.',sources:['S4','S5'],evidenceClass:'reconstruction'},
  optic:{name:'Optic radiation / Meyer candidate',text:'Inspect the reconstructed anterior extent. An unavailable or short loop cannot establish anatomical absence.',sources:['S6','S7'],evidenceClass:'reconstruction'},
  parcels:{name:'Atlas parcels',text:'HCP-MMP1 and Schaefer use different defining methods. Region names require the correct atlas table, version and space.',sources:['S1','S10'],evidenceClass:'atlas'},
  support:{name:'Local model support',text:'TractLab summarizes sampled support at a recorded operating point. Missing or unmeasurable support is not zero support.',sources:['S8'],evidenceClass:'model_metric'},
  body:{name:'Bodily-state framework',text:'A conceptual account of internal physiological state and subjective feeling. This is an explanatory card, not an observed neural event.',sources:['S11'],evidenceClass:'conceptual_model'},
  thalamus:{name:'Thalamic relay territory',text:'An exact subdivision needs a verified atlas and its own reviewed claim. No available structure is renamed VMpo by appearance.',sources:['S11'],evidenceClass:'conceptual_model'},
  insula:{name:'Posterior / anterior insula',text:'Regional distinctions in an interoceptive framework. The atlas view permits inspection of HCP-MMP1 insular parcels; it supplies no subject-specific connection or measured sequence of activity.',sources:['S11','S12'],evidenceClass:'conceptual_model'},
};
export const REGIONS=Object.freeze({...REGIONS_V1,...MOTOR_CST_V2_REGIONS});

// Declarative per-step scene state. 'follow' means: use the current hemisphere
// control (R if the select is 'R', else L). A fixed 'L'/'R' means the lesson's
// evidence is side-specific and must not silently track the control.
//
// SCENE GRAMMAR v2 adds: bundles (family list, supersedes `bundle`), ghost (dimmed
// context families), regions (list, first = focus, supersedes `region`), camera
// (flight, supersedes `view`), durationSec (per-step auto-advance pace) and
// deepRegions (subcortex highlight subset). Every v1 lesson leaves these at their
// default below, so resolveScene's v1 behaviour is unchanged.
export const DEFAULT_SCENE=Object.freeze({side:'follow',bundle:null,region:null,
  deep:false,surface:null,view:null,trace:false,
  bundles:Object.freeze([]),ghost:Object.freeze([]),regions:Object.freeze([]),
  camera:null,durationSec:null,deepRegions:Object.freeze([])});

const step=(title,text,notes,evidenceClass,sources,regions=[],extra={})=>{
  const {scene={},...rest}=extra;
  return {title,text,notes,evidenceClass,sources,regions,...rest,scene};
};
const quiz=(title,question,answer,sources,extra={})=>
  step(title,question,'Invite a prediction before revealing the explanation. Ask the learner to name the modality, population and unresolved question.','schematic',sources,[],{question,answer,...extra});
const lesson=(id,title,minutes,summary,goals,steps,extra={},sceneDefaults={})=>({id,title,minutes,summary,
  goals,steps:steps.map(s=>({...s,scene:{...DEFAULT_SCENE,...sceneDefaults,...s.scene}})),
  version:CONTENT_VERSION,reviewStatus:'draft',audience:'Residents and fellows',...extra});

export const LESSONS=Object.freeze([
  {...MOTOR_CST_V2,draftRevision:MOTOR_CST_V2.version,version:CONTENT_VERSION,steps:MOTOR_CST_V2.steps.map(s=>({...s,scene:{...DEFAULT_SCENE,...s.scene}}))},
  lesson('fat-language','FAT & language planning',7,
    'Distinguish frontal anatomy, cohort association and a task-specific stimulation result.',
    ['Describe the frontal relationship.','Separate association from stimulation evidence.','Keep left-sided task claims source-specific.'],[
      step('Follow the frontal relationship','The frontal aslant pathway relates inferior frontal and medial superior frontal regions in the anatomical framework studied here. That relationship invites questions about speech organization and initiation.',
        'Begin with the regions, then ask what the experiments measured. A schematic endpoint must not be attached to an individual reconstructed termination.','reconstruction',['S4'],['frontal','fat'],{group:'fat',side:'L'}),
      step('Open the association card','In the studied primary progressive aphasia groups, FAT measures were associated with verbal fluency. This relates a diffusion-derived measure to performance in a particular cohort.',
        'Ask what was correlated, in whom, and using which task. The cohort belongs beside the claim. This does not locate all language function in one bundle.','association',['S4'],['fat']),
      step('Compare a stimulation experiment','One case reported sentence-planning disruption during stimulation near a tractography-defined left FAT, while measured articulation timing was not similarly changed.',
        'The task distinguished planning from articulation. Keep the single-case and localization limits visible. No stimulation coordinates from that case are placed on this brain.','functional_measurement',['S5'],['fat']),
      step('Let the task sharpen the question','Planning an utterance, retrieving words, repeating material and executing articulation are distinguishable behaviors. The tract picture alone cannot say which task will be affected.',
        'Invite a task-based comparison of interpretations. This is an educational reasoning exercise, not an intraoperative protocol. Right-sided selection must not silently reuse left-sided language claims.','schematic',['S4','S5'],['frontal']),
      quiz('Why is “FAT = speech arrest” inadequate?','Explain the difference between an anatomical relationship, a fluency correlation and a planning effect during stimulation.',
        'They are different evidence classes with different populations and tasks. A universal label erases those conditions. Describe the observed behavior and modality before generalizing.',['S4','S5']),
    ],{},{side:'L',view:'left',bundle:'FAT'}),
  lesson('optic-radiation','Optic radiation & reconstructed extent',6,
    'Inspect a curved pathway without turning its visible extent into a universal anatomical boundary.',
    ['Identify the actual reconstruction recipe.','Distinguish visible extent from completeness.','Test a display effect using the same source rows.'],[
      step('Locate the candidate','What does this reconstruction include? Its endpoints and anterior extent depend on acquisition, model and selection recipe. The named candidate is the object being inspected.',
        'Rotate once, restore the authored view, and identify the most anterior visible segment. That observation describes these streamlines, not a functional boundary.','reconstruction',['S6','S7'],['optic'],{group:'or',scene:{bundle:'OR',view:'follow'}}),
      step('Compare individuals, not a fixed boundary','The studied individuals differed in the anterior extent of Meyer’s loop. A group estimate is not a universal boundary for the next individual.',
        'The source used a small sample and a particular DTI method. Use it to motivate individual review, not a fixed temporal-pole distance.','reconstruction',['S6'],['optic'],{needsAtlas:true}),
      step('Inspect a missing segment','A shorter or missing visible segment may reflect acquisition, orientation modeling, tracking, extraction, filtering or display. Absence in this view does not establish absence of pathway or function.',
        'Change one display condition at a time and retain the result identity. The validation challenge motivates caution; its numerical error rates do not transfer to this case.','reconstruction',['S7','S9'],['optic'],{group:'or',scene:{bundle:'OR'}}),
      quiz('The anterior loop disappears','What directly tests whether the disappearance was caused by rendering or selection?',
        'Restore the same source result and compare its selection recipe and ordered source rows. That tests the software/data question. Increasing glow or displayed lines cannot replace independent functional evidence.',['S6','S7']),
    ],{},{side:'follow',deep:false,surface:null}),
  lesson('evidence-classes','Atlas, reconstruction, association, function',5,
    'Name the evidence class before interpreting the brain image.',
    ['Identify the atlas method.','Read reconstruction provenance.','Keep an experiment’s task and population beside its claim.'],[
      step('Which atlas produced the label?','Schaefer parcels organize resting-state similarity. HCP-MMP1 uses multiple cortical features. These label systems answer related but different questions.',
        'Ask for the atlas version, reference space and label table. A nickname is not a verified parcel identity.','atlas',['S1','S10'],['parcels'],{needsAtlas:true}),
      step('What are these lines?','Streamlines are reconstructed from diffusion data. An anatomically plausible appearance is useful for inspection, but plausibility alone does not establish a true connection.',
        'The challenge study had defined ground truth and participating methods. Do not assign its error rate to this dataset.','reconstruction',['S7'],['cst']),
      step('What was measured functionally?','A functional observation includes modality, task, location, population and measurement. It is more specific than a tract name and narrower than a universal statement about one function.',
        'Compare the precision-fMRI observations with the single-case stimulation report. Neither supplies functional coordinates for the current reconstruction.','functional_measurement',['S2','S5'],['motor','fat']),
      quiz('Classify four statements','Classify: “parcel 4 in this atlas”; “recipe-selected streamlines”; “a fluency correlation”; “a task disruption during stimulation.”',
        'Atlas label; reconstruction; association; localized experimental functional evidence. A change of rendering mode does not turn one class into another.',['S1','S4','S5','S7']),
    ],{},{side:'follow',deep:false,surface:null}),
  lesson('sampling-support','Empty, sparse & low support',6,
    'Follow the selection pipeline and keep measurements bound to the rows that produced them.',
    ['Separate analytic and displayed populations.','Recognize a sampling-invariant path.','Interpret support without turning it into a truth probability.'],[
      step('Follow the selection pipeline','Diffusion model → candidate corpus → recipe/ROI selection → analytic population → display sample. An empty extraction and a sparse display are different observations.',
        'Recovery uses a different predicate and source population. Nearby lines do not inherit named-bundle identity. This diagram represents data processing, not neural transmission.','schematic',['S7','S9'],[],{diagram:'pipeline'}),
      step('One path, different samples','Adding collinear points does not change a physical segment. A geometry consumer must declare its approximation and preserve the same path interpretation.',
        'Use the generated straight-path regression, not clinical data. The historical audit found different answers; the remediation gate requires segment-based consistency and an explicit error bound.','schematic',[],[],{diagram:'sampling'}),
      step('Read SIFT2 and fidelity separately','SIFT2 assigns model-derived contributions to a fibre-density fit. TractLab fidelity summarizes local support relative to an operating point. Neither value is a probability that a whole connection is true.',
        'Ask whether the metric was computed for these exact rows. Missing or unmeasurable support is untested, not zero. A signed setting records review, not clinical validation.','model_metric',['S8'],['support']),
      quiz('Two banks and one recovery overlay','What must be specified before exporting or drawing an envelope?',
        'The explicit source result and selection. The viewer may show multiple populations while an artifact belongs to one. The focused result, its row ordering and source digest must accompany the operation.',['S7','S8']),
    ],{},{side:'follow',deep:false,surface:null}),
  lesson('interoception','Interoception · reading a proposed circuit',8,
    'A conceptual tour with explicit experimental, species and missing-atlas boundaries.',
    ['Distinguish a model from a measured signal sequence.','Recognize branching relationships.','Separate installed reference anatomy from missing circuit evidence.'],[
      step('Start with the question','Interoception concerns the body’s internal physiological state. Craig’s framework relates bodily-state representations to subjective feeling. We examine it as a model with distinct kinds of evidence.',
        'This is not a recording of activity travelling through the screen. The introductory source is a conceptual review, not a new diffusion or timing experiment.','conceptual_model',['S11'],['body'],{diagram:'interoception',scene:{region:{id:111,hemi:'follow'}}}),
      step('Give each region its own card','Distinguish an ascending bodily-state framework from the way insular regions integrate information. Moving the teaching focus organizes the explanation; it does not measure latency or establish a serial route.',
        'An exact thalamic subdivision needs a matching atlas and reviewed claim. Do not rename an available structure as VMpo from appearance.','conceptual_model',['S11'],['thalamus','insula'],{needsAtlas:true,scene:{region:{id:106,hemi:'follow'}}}),
      step('Read a wider network','Insular relationships participate in a wider cortical system. Tracer experiments in monkeys contribute anatomical evidence, while species and experimental scope limit direct translation to an individual human.',
        'The primary source describes insular cortical outputs in old-world monkeys. Keep experimental anatomy separate from human diffusion reconstruction. No per-edge atlas geometry is asserted here.','experimental_anatomy',['S12'],['insula'],{scene:{region:{id:106,hemi:'follow'}}}),
      step('What is missing from this scene?','The reference atlas supplies cortical parcels and gross deep structures. Exact relay nuclei, per-relationship evidence and an individual circuit remain unresolved. A spatial resemblance does not complete that evidence.',
        'Do not rename the merged thalamic territory as VMpo. A bright hull or an available CST/IFOF/UF cannot substitute for this circuit. The case view has no patient-to-atlas registration.','schematic',['S11','S12'],['insula','thalamus'],{needsAtlas:true,scene:{region:{id:106,hemi:'follow'}}}),
      quiz('What did the animation establish?','The explanation visits posterior insula, anterior insula and an association region. What has that ordering established?',
        'The order of the explanation. Biological direction, timing, causality and a subject-specific connection require their own evidence. Identify which relationships are conceptual and which come from a named experiment.',['S11','S12'],{scene:{region:{id:111,hemi:'follow'}}}),
    ],{referenceOnly:true},{side:'follow',bundle:null,deep:true,surface:.24}),
]);

const SCENE_SIDES=Object.freeze(['follow','L','R']);
const SCENE_BUNDLES=Object.freeze([null,'CST','FAT','OR','AF']);
const SCENE_VIEWS=Object.freeze([null,'follow','left','right','top','medial']);
const SCENE_REGION_HEMIS=Object.freeze(['follow','L','R']);
const SCENE_CAMERA_VIEWS=Object.freeze(['left','right','superior','anterior','posterior','medial','follow']);

function validRegionEntry(r){
  return r && typeof r==='object' && Number.isInteger(r.id) && r.id>=0 && SCENE_REGION_HEMIS.includes(r.hemi);
}

function validateScene(scene,lessonId){
  if(!scene || typeof scene!=='object') throw new Error(`Missing scene in ${lessonId}`);
  if(!SCENE_SIDES.includes(scene.side)) throw new Error(`Invalid scene side in ${lessonId}`);
  if(!SCENE_BUNDLES.includes(scene.bundle)) throw new Error(`Invalid scene bundle in ${lessonId}`);
  if(!SCENE_VIEWS.includes(scene.view)) throw new Error(`Invalid scene view in ${lessonId}`);
  if(typeof scene.deep!=='boolean') throw new Error(`Invalid scene deep flag in ${lessonId}`);
  if(typeof scene.trace!=='boolean') throw new Error(`Invalid scene trace flag in ${lessonId}`);
  if(scene.surface!==null && (typeof scene.surface!=='number' || Number.isNaN(scene.surface) || scene.surface<0 || scene.surface>1))
    throw new Error(`Invalid scene surface range in ${lessonId}`);
  if(scene.region!==null){
    if(typeof scene.region!=='object' || !Number.isInteger(scene.region.id) || scene.region.id<0 ||
      !SCENE_REGION_HEMIS.includes(scene.region.hemi)) throw new Error(`Invalid scene region in ${lessonId}`);
  }
  // SCENE GRAMMAR v2
  if(!Array.isArray(scene.bundles) || scene.bundles.some(f=>!BUNDLE_FAMILIES.includes(f)))
    throw new Error(`Invalid scene bundles in ${lessonId}`);
  if(!Array.isArray(scene.ghost) || scene.ghost.some(f=>!BUNDLE_FAMILIES.includes(f)))
    throw new Error(`Invalid scene ghost in ${lessonId}`);
  if(!Array.isArray(scene.regions) || scene.regions.some(r=>!validRegionEntry(r)))
    throw new Error(`Invalid scene regions in ${lessonId}`);
  if(scene.camera!==null){
    const c=scene.camera;
    if(typeof c!=='object' || !SCENE_CAMERA_VIEWS.includes(c.view) ||
      (c.zoom!=null && (typeof c.zoom!=='number' || Number.isNaN(c.zoom) || c.zoom<=0)) ||
      (c.tweenMs!=null && (typeof c.tweenMs!=='number' || Number.isNaN(c.tweenMs) || c.tweenMs<0)))
      throw new Error(`Invalid scene camera in ${lessonId}`);
  }
  if(scene.durationSec!==null && (typeof scene.durationSec!=='number' || Number.isNaN(scene.durationSec) || scene.durationSec<=0))
    throw new Error(`Invalid scene durationSec in ${lessonId}`);
  if(!Array.isArray(scene.deepRegions) || scene.deepRegions.some(id=>!SUBCORTEX_IDS.includes(id)&&!SUBCORTEX_IDS.includes(`${id}-lh`)))
    throw new Error(`Invalid scene deepRegions in ${lessonId}`);
}

export function validateLessons(lessons=LESSONS){
  const ids=new Set();
  for(const l of lessons){
    if(ids.has(l.id) || !l.version || l.reviewStatus!=='draft' || !l.steps.length) throw new Error('Invalid lesson identity or review state');
    ids.add(l.id);
    for(const s of l.steps){
      if(!s.title || !s.text || !s.notes || !EVIDENCE_CLASSES.includes(s.evidenceClass)) throw new Error(`Invalid step in ${l.id}`);
      if(s.sources.some(id=>!SOURCES[id]) || s.regions.some(id=>!REGIONS[id])) throw new Error(`Unresolved reference in ${l.id}`);
      validateScene(s.scene,l.id);
    }
  }
  return true;
}
validateLessons();
