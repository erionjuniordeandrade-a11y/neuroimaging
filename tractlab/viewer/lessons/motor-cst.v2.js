/** motor-cst v2 — the full-standard rewrite of the motor lesson, and the template
 * for every lesson that follows. Standalone by design: it imports nothing, so it
 * can be reviewed, unit-tested and dropped into LESSONS without touching
 * lesson_content.js. Merge is Object.assign into SOURCES / REGIONS; no key here
 * collides with the keys already defined there.
 *
 * Everything below is an educational DRAFT awaiting the surgeon-owner's
 * anatomical review. Nothing here is validated.
 *
 * What this lesson is allowed to point at, and nothing else:
 *   - HCP S1200 group-average cortex with HCP-MMP1 parcels (viewer/atlas/surface.json)
 *   - HCP1065 population-averaged bundles (viewer/atlas/tracts.json)
 *   - Melbourne Subcortex Atlas scale 1 gross structures (viewer/atlas/subcortex.json)
 * There is NO patient data, NO case-to-atlas registration and NO functional map
 * installed. Where a claim needs one of those, the step says so in plain words.
 *
 * Companion prose + Vancouver reference list: ./motor-cst.v2.sources.md
 */

/* -------------------------------------------------------------------------- */
/* Scene grammar v2                                                            */
/* -------------------------------------------------------------------------- */
/**
 * scene: {
 *   side:        'follow' | 'L' | 'R'      — 'follow' tracks the hemisphere control
 *   bundles:     string[]                  — bundle FAMILIES, no side suffix.
 *                                            The resolver appends _L / _R from the
 *                                            resolved side (CST -> CST_L), exactly as
 *                                            scene.bundle did in grammar v1.
 *                                            Sideless families (MCP, SCP, CC) are used
 *                                            verbatim.
 *   ghost:       string[]                  — same vocabulary, drawn as context
 *   regions:     [{id:int, hemi:'follow'|'L'|'R'}] — HCP-MMP1 ids; FIRST ENTRY IS THE FOCUS
 *   deep:        boolean                   — show subcortical meshes
 *   deepRegions: string[]                  — subcortex STEMS, no hemisphere suffix.
 *                                            The resolver appends -lh / -rh from the
 *                                            resolved side (THA -> THA-lh), mirroring
 *                                            the bundle-family rule. Full ids live in
 *                                            viewer/atlas/subcortex.json.
 *   surface:     0.08 .. 0.95              — cortical opacity
 *   camera:      {view, zoom?, tweenMs?}   — view in VIEWS below
 *   trace:       boolean                   — shape sweep; carries no physiology
 *   durationSec: 25 .. 40                  — authored dwell time
 * }
 */
export const SCENE_GRAMMAR_VERSION = 'v2';
export const SCENE_SIDES = Object.freeze(['follow', 'L', 'R']);
export const SCENE_VIEWS = Object.freeze(['left', 'right', 'superior', 'anterior', 'posterior', 'medial', 'follow']);
export const SCENE_HEMIS = Object.freeze(['follow', 'L', 'R']);
/** Bundle families this lesson is permitted to name (all present in tracts.json). */
export const LESSON_BUNDLE_FAMILIES = Object.freeze([
  'CST', 'CPT_F', 'CPT_P', 'CPT_O', 'CS_A', 'CS_P', 'CS_S',
  'ML', 'DRTT', 'TR_A', 'TR_P', 'TR_S', 'MCP', 'SCP', 'CC',
]);
/** Melbourne S1 stems this lesson is permitted to name. */
export const LESSON_DEEP_STEMS = Object.freeze(['THA', 'PUT', 'GP', 'CAU']);

/** HCP-MMP1 ids used here, read from viewer/atlas/surface.json.
 * The L and R tables number these identically, so 'follow' is safe for all of them. */
export const PARCELS = Object.freeze({
  a4: 8,      // *_4_ROI      — primary motor
  a6d: 54,    // *_6d_ROI     — dorsal premotor
  a6v: 56,    // *_6v_ROI     — ventral premotor
  a6mp: 55,   // *_6mp_ROI    — posterior medial premotor (SMA proper territory)
  a6ma: 44,   // *_6ma_ROI    — anterior medial premotor (pre-SMA territory)
  SCEF: 43,   // *_SCEF_ROI   — supplementary/cingulate eye field
  a55b: 12,   // *_55b_ROI    — inferior precentral, between 6v and 6d
  a3a: 53,    // *_3a_ROI
  a3b: 9,     // *_3b_ROI
  a1: 51,     // *_1_ROI
  a2: 52,     // *_2_ROI
  a5L: 39,    // *_5L_ROI     — superior parietal
  a7PC: 47,   // *_7PC_ROI    — postcentral parietal
  a24dd: 40,  // *_24dd_ROI   — dorsal posterior cingulate motor territory
});

const R = (id, hemi = 'follow') => Object.freeze({ id, hemi });

/* -------------------------------------------------------------------------- */
/* Sources — every PMID checked against its NCBI esummary record (2026-09-07)  */
/* -------------------------------------------------------------------------- */
const source = (id, title, url, evidenceClass, scope) =>
  Object.freeze({ id, title, url, evidenceClass, scope });
const pubmed = pmid => `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;

export const MOTOR_CST_V2_SOURCES = Object.freeze({
  // --- primary teaching references -----------------------------------------
  M1: source('M1', 'Glasser et al., 2016 · multi-modal parcellation of human cerebral cortex',
    pubmed('27437579'), 'atlas',
    'HCP population reference built from converging cortical features. A parcel boundary here is a group boundary; it does not locate one person’s functional border and it has not been registered to any individual in this viewer.'),
  M2: source('M2', 'Gordon et al., 2023 · somato-cognitive action network in motor cortex',
    pubmed('37076628'), 'functional_measurement',
    'Precision functional MRI and connectivity in a small set of deeply sampled participants. Effector and inter-effector observations are task, modality and population dependent, and none of that map is installed in this viewer.'),
  M3: source('M3', 'Penfield & Rasmussen, 1950 · The Cerebral Cortex of Man (Macmillan, New York)',
    null, 'functional_measurement',
    'Historical open-cortex electrical stimulation in surgical patients, published as a book and therefore carrying no PMID — stated here rather than hidden. The homunculus figure is a schematic summary of stimulation responses, not a measured anatomical map.'),
  M4: source('M4', 'Usuda et al., 2022 · corticospinal tracts arising from different cortical areas',
    pubmed('35787428'), 'reconstruction',
    'Healthy-volunteer tractography comparing descending contributions by cortical origin. Streamline distributions depend on acquisition, model and ROI recipe; they are not axon counts and they are not transferable to a patient.'),
  M5: source('M5', 'Holodny et al., 2005 · somatotopic organization of corticospinal tracts in the internal capsule',
    pubmed('15665224'), 'reconstruction',
    'Early DTI tractography of capsular organization, published explicitly against prior reports. A group ordering derived from one method is not a map you can transfer to an individual capsule.'),
  M6: source('M6', 'Behrens et al., 2003 · non-invasive mapping of thalamus–cortex connections',
    pubmed('12808459'), 'reconstruction',
    'Probabilistic diffusion tractography relating thalamic territory to cortical target. A connectivity-defined territory is a model output, not a nuclear boundary, and this viewer ships no thalamic nuclei.'),
  M7: source('M7', 'Schmahmann & Pandya, 1997 · basilar pontine projections from prefrontal cortices',
    pubmed('8987769'), 'experimental_anatomy',
    'Tracer experiments in rhesus monkey. Species and experimental scope limit translation; a monkey tracer result does not verify a human population reconstruction.'),
  M8: source('M8', 'Morris et al., 2009 · proximal dentatothalamocortical tract involvement in posterior fossa syndrome',
    pubmed('19805491'), 'association',
    'Clinical–imaging correlation in a paediatric posterior fossa cohort. An association between imaged tract involvement and a postoperative syndrome is not a prediction rule for an individual, and nothing in this scene measures it.'),
  M9: source('M9', 'Krainik et al., 2001 · supplementary motor area and motor deficit after medial frontal surgery',
    pubmed('11552019'), 'association',
    'Surgical series correlating resection extent with postoperative deficit and recovery. The recovery pattern belongs to that series and its follow-up, not to any picture on this screen.'),
  M10: source('M10', 'Farquharson et al., 2013 · why we need to move beyond DTI',
    pubmed('23540269'), 'reconstruction',
    'Neurosurgical comparison of tensor and higher-order fibre-orientation models. Demonstrates that the orientation model changes which pathways appear; a rendered bundle inherits its model’s failure modes.'),
  M11: source('M11', 'Maier-Hein et al., 2017 · the challenge of mapping the connectome with diffusion tractography',
    pubmed('29116093'), 'reconstruction',
    'Challenge dataset with a defined ground truth and many submitted pipelines. Its false-positive findings motivate caution; its error figures belong to that challenge and are not an error estimate for this atlas.'),
  M12: source('M12', 'Nimsky et al., 2005 · pre- and intraoperative DTI-based fiber tracking in glioma surgery',
    pubmed('15617595'), 'reconstruction',
    'Operative series using DTI tractography inside neuronavigation, reporting intraoperative displacement of the reconstructed tracts. A preoperative bundle position is not a position during the resection.'),

  // --- dataset provenance for what is actually installed ---------------------
  D1: source('D1', 'Van Essen et al., 2013 · the WU-Minn Human Connectome Project overview',
    pubmed('23684880'), 'atlas',
    'Provenance for the S1200 group-average surface rendered here. A group-average midthickness surface is a population object; individual gyral and sulcal anatomy differs from it.'),
  D2: source('D2', 'Yeh, 2022 · population-based tract-to-region connectome of the human brain',
    pubmed('35995773'), 'atlas',
    'Provenance for the HCP1065 population-averaged bundles rendered here. Every displayed line is an average-space reconstruction sampled for display; it is not a patient’s pathway and carries no direction of conduction.'),
  D3: source('D3', 'Tian et al., 2020 · topographic organization of the human subcortex',
    pubmed('32989295'), 'atlas',
    'Provenance for the Melbourne Subcortex Atlas scale-1 meshes rendered here. Scale 1 is the coarsest level: anterior and posterior thalamus are merged and no brainstem nucleus is included.'),
});

/* -------------------------------------------------------------------------- */
/* Region cards — keys are lesson-scoped so a merge into REGIONS is collision-free */
/* -------------------------------------------------------------------------- */
export const MOTOR_CST_V2_REGIONS = Object.freeze({
  atlasProvenance: {
    name: 'What is actually installed',
    text: 'A group-average cortical surface with a population parcellation, a population-averaged bundle atlas, and gross subcortical meshes. No patient image, no registration, no functional map, no brainstem or thalamic nuclei.',
    sources: ['D1', 'D2', 'D3'], evidenceClass: 'atlas',
  },
  motorParcels: {
    name: 'Precentral and postcentral parcels',
    text: 'HCP-MMP1 divides the central region by several cortical features at once, so 4, 6d, 6v, 6mp, 6ma, SCEF, 55b, 3a, 3b, 1 and 2 are separate labels. They are population reference labels, not this patient’s borders.',
    sources: ['M1', 'D1'], evidenceClass: 'atlas',
  },
  effectorMap: {
    name: 'Effector and inter-effector territories',
    text: 'A functional organization measured with precision fMRI in other participants. It is discussed here and deliberately not painted onto the surface, because that measurement is not part of this dataset.',
    sources: ['M2', 'M3'], evidenceClass: 'functional_measurement',
  },
  cstCandidate: {
    name: 'Corticospinal candidate',
    text: 'A recipe-selected descending reconstruction from a population atlas. Read its side, its family and its provenance. It is a candidate pathway, never “the” corticospinal tract of any person, and it specifies no body part.',
    sources: ['D2', 'M4', 'M11'], evidenceClass: 'reconstruction',
  },
  coronaCapsule: {
    name: 'Corona radiata and internal capsule',
    text: 'The convergence where descending families stop being separable by eye. Capsular ordering described in the literature is a group finding from a particular method; it is not resolvable in this average-space rendering.',
    sources: ['M5', 'D2'], evidenceClass: 'reconstruction',
  },
  corticopontine: {
    name: 'Corticopontine families',
    text: 'Frontopontine, parietopontine and occipitopontine reconstructions travelling with the descending corridor. Their anatomical account rests partly on primate tracer work, which is experimental anatomy in another species.',
    sources: ['M7', 'D2'], evidenceClass: 'experimental_anatomy',
  },
  striatopallidal: {
    name: 'Deep grey around the capsule',
    text: 'Putamen, globus pallidus, caudate and thalamus at the coarsest scale of a group subcortical parcellation. Anterior and posterior thalamus are merged into one mesh; no nucleus is individually named or targetable here.',
    sources: ['D3'], evidenceClass: 'atlas',
  },
  sensoryReturn: {
    name: 'Ascending and thalamocortical families',
    text: 'Medial lemniscus and the anterior, superior and posterior thalamic radiations. The render encodes geometry only: it cannot show direction of conduction, so “ascending” is anatomy you bring to the picture.',
    sources: ['M6', 'D2'], evidenceClass: 'reconstruction',
  },
  cerebellarLoop: {
    name: 'Cortico-ponto-cerebello-thalamo-cortical loop',
    text: 'Dentato-rubro-thalamic and the cerebellar peduncles. The clinical syndromes attached to this loop come from cited cohorts and surgical series; the scene shows only reconstructed geometry.',
    sources: ['M8', 'M9', 'M7'], evidenceClass: 'reconstruction',
  },
  displaySupport: {
    name: 'What the display and the trace encode',
    text: 'Colour is local orientation, opacity is a setting, line count is a sampling cap, and the trace is a shape sweep at a fixed rate. None of it is conduction, direction, velocity or fibre density.',
    sources: ['M10', 'M11', 'M12'], evidenceClass: 'model_metric',
  },
});

/* -------------------------------------------------------------------------- */
/* Steps                                                                       */
/* -------------------------------------------------------------------------- */
const scene = ({ side = 'follow', bundles = [], ghost = [], regions = [], deep = false,
  deepRegions = [], surface = 0.9, camera = { view: 'follow' }, trace = false, durationSec = 30 }) =>
  Object.freeze({
    side,
    bundles: Object.freeze([...bundles]),
    ghost: Object.freeze([...ghost]),
    regions: Object.freeze([...regions]),
    deep,
    deepRegions: Object.freeze([...deepRegions]),
    surface,
    camera: Object.freeze({ ...camera }),
    trace,
    durationSec,
  });

const step = (title, text, notes, evidenceClass, sources, regions, sceneSpec, extra = {}) =>
  Object.freeze({ title, text, notes, evidenceClass, sources, regions, ...extra, scene: scene(sceneSpec) });

const STEPS = Object.freeze([

  step(
    'Name the surface before you name a tract',
    'Before any fibre appears, say what you are looking at. This is an HCP S1200 group-average cortex carrying HCP-MMP1 parcels. Area 4 is lit as the focus, with 6d, 6v, 6mp, 6ma, SCEF and 55b around it, because the precentral wall is not one strip: the atlas divides it using cytoarchitecture, myelin, cortical thickness and resting-state features together. That matters at surgery because the sulcal landmark you can actually see does not tell you where a parcel boundary sits. This is a population reference. It does not say where this patient’s area 4 begins, and no boundary here has been registered to any individual brain.',
    'Ask the room to point at the central sulcus before you reveal the parcel colours, then ask how confident they are to within a centimetre on a swollen, shifted hemisphere. Common misconception: that a parcel edge is a functional edge. It is a group statistical boundary from a multimodal atlas, and the absence of any registration is the whole point of this first step.',
    'atlas', ['M1', 'D1'], ['motorParcels', 'atlasProvenance'],
    {
      regions: [R(PARCELS.a4), R(PARCELS.a6d), R(PARCELS.a6v), R(PARCELS.a6mp), R(PARCELS.a6ma), R(PARCELS.SCEF), R(PARCELS.a55b)],
      surface: 0.92, camera: { view: 'follow', zoom: 1.4, tweenMs: 900 }, durationSec: 30,
    },
    { needsAtlas: true },
  ),

  step(
    'Somatotopy is a measurement, not a colour on this brain',
    'The classic homunculus came from open-cortex stimulation and was drawn as a continuous strip. Precision functional MRI in the cited work reports something else: effector-specific foci for foot, hand and mouth, separated by inter-effector regions that behave more like an action-control system than a body map. Neither picture is a parcel you can switch on here. The atlas gives you 4, 6mp, SCEF and 24dd as anatomical labels; the effector map is a functional measurement made in other participants with another modality, and it is not installed in this viewer. Say that out loud rather than letting parcel colours stand in for a map nobody has loaded.',
    'Ask which of the two pictures they would trust to set a resection margin, and let them arrive at the honest answer — neither, without individual mapping in this patient. Common misconception: that the homunculus is drawn somewhere in the atlas. It is not present at all. Keep Penfield’s figure labelled as a 1950 stimulation summary and the effector account as task- and population-dependent.',
    'functional_measurement', ['M2', 'M3', 'M1'], ['effectorMap', 'motorParcels'],
    {
      regions: [R(PARCELS.a6mp), R(PARCELS.a4), R(PARCELS.SCEF), R(PARCELS.a24dd), R(PARCELS.a6ma)],
      surface: 0.92, camera: { view: 'medial', zoom: 1.3, tweenMs: 1000 }, durationSec: 34,
    },
    { needsAtlas: true },
  ),

  step(
    'Reveal the descending reconstruction',
    'Now the corticospinal bundle appears. Be precise about what arrived: a population-averaged reconstruction, built by averaging across the HCP subjects recorded in the atlas manifest and shipped as a display sample of streamlines. It is not this patient’s axons and not any patient’s axons. Call it a corticospinal candidate. Drop the cortical opacity so the course is visible from the centrum semiovale down toward the crus. What you gain is a spatial expectation — where such a bundle tends to run relative to the parcels above it. What you do not gain is a calibre, an ordering inside the bundle, or evidence that any displayed line carries motor command in a living person.',
    'Ask what actually changed when the bundle appeared: geometry, or knowledge? Common misconception: that a denser-looking bundle means more fibres. The line count is a per-bundle display cap and the geometry is an average across subjects. Insist on the phrase “a corticospinal candidate” every time somebody in the room says “the CST”, and make them say why the article matters.',
    'reconstruction', ['D2', 'M11'], ['cstCandidate', 'atlasProvenance'],
    {
      bundles: ['CST'],
      regions: [R(PARCELS.a4), R(PARCELS.a6d), R(PARCELS.a6mp)],
      surface: 0.35, camera: { view: 'follow', zoom: 1.2, tweenMs: 1200 }, durationSec: 32,
    },
  ),

  step(
    'The candidate has more than one cortical origin',
    'Light the parcels and ghost the bundle. Descending fibres of the pyramidal system do not arise from area 4 alone. The cited quantitative comparison tracks corticospinal contributions from premotor 6d and 6v and from postcentral 3a, 3b, 1 and 2 as well as from 4, and reports that they differ by origin. That reframes the resection question: a precentral gyrus left intact while a premotor or postcentral origin is transected still costs descending fibres. Those distributions are streamline counts from one tractography method in healthy volunteers, dependent on model and ROI choice. They are not axon counts, and this scene does not colour any fibre by its origin.',
    'Ask where the room thinks the corticospinal tract “starts”, then ask what fraction they would guess comes from area 4 — and then refuse to supply a number, because the cited proportions are method-bound. Common misconception: precentral gyrus equals pyramidal tract. Premotor and postcentral contributions are exactly why a “we stayed behind the sulcus” argument does not hold.',
    'reconstruction', ['M4', 'M1', 'D2'], ['cstCandidate', 'motorParcels'],
    {
      ghost: ['CST'],
      regions: [R(PARCELS.a4), R(PARCELS.a6d), R(PARCELS.a6v), R(PARCELS.a6mp), R(PARCELS.a3a), R(PARCELS.a3b), R(PARCELS.a1), R(PARCELS.a2)],
      surface: 0.88, camera: { view: 'follow', zoom: 1.35, tweenMs: 900 }, durationSec: 34,
    },
    { needsAtlas: true },
  ),

  step(
    'Corona radiata into the internal capsule',
    'Turn to an anterior view and thin the cortex almost away. The fan of the corona radiata converges into the posterior limb of the internal capsule, and the corticopontine families are ghosted alongside it — frontopontine anteriorly, parietopontine and occipitopontine behind. This is the crowded part of the operation: at the capsule a small lateral drift changes which system you are in, and the descending families are no longer separable by eye. The cited tractography work describes a somatotopic ordering within the capsule, and was published against earlier claims. Treat it as a group finding from one method, not as a map you can transfer to the patient in front of you.',
    'Ask where the ghosted frontopontine fibres sit relative to the corticospinal candidate within the capsule, and whether the answer survives down at the crus. Common misconception: that corticospinal fibres own the posterior limb. They share it with corticopontine, corticobulbar and thalamic traffic, and an average-space rendering cannot resolve the packing in one person’s capsule.',
    'reconstruction', ['M5', 'M7', 'D2'], ['coronaCapsule', 'corticopontine', 'cstCandidate'],
    {
      bundles: ['CST'], ghost: ['CPT_F', 'CPT_P', 'CPT_O'],
      regions: [R(PARCELS.a4), R(PARCELS.a6d)],
      surface: 0.12, camera: { view: 'anterior', zoom: 1.5, tweenMs: 1200 }, durationSec: 36,
    },
  ),

  step(
    'The deep grey the capsule runs between',
    'Bring in the subcortical meshes. Putamen and globus pallidus sit lateral to the capsule, caudate medial and anterior, thalamus posteromedial — the capsule is the space between them, which is why a lenticulostriate territory infarct and a capsular tumour approach produce such different deficits. The corticostriatal families are ghosted here so you can see projections leaving the same corona and terminating in striatum rather than descending. These meshes come from a group functional-connectivity subcortical parcellation at its coarsest scale. There are no thalamic nuclei, no subthalamic nucleus, no red nucleus and no brainstem nuclei installed; anterior and posterior thalamus are one merged object.',
    'Ask the room to name the structure lateral to the posterior limb before you label it, then ask which thalamic nucleus they would want for a motor question — and show that this atlas cannot answer, because the thalamus is a single merged mesh. Common misconception: that shipping a subcortical atlas implies nuclear resolution. Coarse scale is a property of the data, not a rendering choice.',
    'atlas', ['D3', 'M5'], ['striatopallidal', 'coronaCapsule'],
    {
      bundles: ['CST'], ghost: ['CS_A', 'CS_P', 'CS_S'],
      regions: [R(PARCELS.a4)],
      deep: true, deepRegions: ['PUT', 'GP', 'THA', 'CAU'],
      surface: 0.08, camera: { view: 'anterior', zoom: 1.6, tweenMs: 1000 }, durationSec: 34,
    },
  ),

  step(
    'Sensory return runs the other way',
    'Ghost the descending candidate and reveal what ascends. Medial lemniscus reaches thalamus; the thalamic radiations fan out to cortex, here toward 3a, 3b, 1, 2 and the parietal parcels 5L and 7PC. The reconstruction cannot tell you direction of conduction — nothing on this screen can — so “ascending” and “descending” are anatomy you bring to the picture, not a property the render measured. It matters because a purely motor account of this corridor undersells the deficit: proprioceptive loss from the same corona or capsular injury leaves a limb the patient will not use even when formal strength testing looks close to normal.',
    'Ask how anyone would distinguish an ascending from a descending streamline in this render. Nobody can: diffusion is undirected and the colour encodes local orientation only. Common misconception: that a motor-only picture predicts the postoperative deficit. Ask who has looked after a patient with preserved power and a functionally useless hand after a parietal or thalamocortical injury.',
    'reconstruction', ['M6', 'D2', 'M5'], ['sensoryReturn', 'cstCandidate'],
    {
      bundles: ['ML', 'TR_A', 'TR_S', 'TR_P'], ghost: ['CST'],
      regions: [R(PARCELS.a3b), R(PARCELS.a3a), R(PARCELS.a1), R(PARCELS.a2), R(PARCELS.a5L), R(PARCELS.a7PC)],
      deep: true, deepRegions: ['THA'],
      surface: 0.3, camera: { view: 'follow', zoom: 1.3, tweenMs: 1000 }, durationSec: 36,
    },
    { needsAtlas: true },
  ),

  step(
    'The cerebellar loop is why a bundle-only picture misleads',
    'Three families now: dentato-rubro-thalamic with the superior and middle cerebellar peduncles, and the corticospinal and frontopontine candidates ghosted behind them. The cortico-ponto-cerebello-thalamo-cortical loop is the reason motor morbidity does not track one descending bundle. The cited posterior fossa cohort reports proximal dentatothalamocortical involvement in postoperative cerebellar mutism, and the cited medial frontal series reports the supplementary motor area syndrome after resections that spared area 4 — akinesia and speech arrest arising from territory lit here as 6mp, SCEF and 24dd, which that series describes as typically transient. Neither finding is measured in this scene; both are clinical evidence you are importing.',
    'Ask what deficit they expect from a medial frontal resection that never touches the precentral gyrus, and how they would counsel the family about recovery before the operation. Common misconception: that SMA syndrome and cerebellar mutism are curiosities rather than predictable consequences of interrupting a loop. Keep every recovery claim attached to its cited series and its follow-up, never to this picture.',
    'reconstruction', ['M8', 'M9', 'M7', 'D2'], ['cerebellarLoop', 'corticopontine'],
    {
      bundles: ['DRTT', 'SCP', 'MCP'], ghost: ['CST', 'CPT_F'],
      regions: [R(PARCELS.a6mp), R(PARCELS.SCEF), R(PARCELS.a24dd), R(PARCELS.a6ma)],
      deep: true, deepRegions: ['THA'],
      surface: 0.25, camera: { view: 'posterior', zoom: 1.3, tweenMs: 1200 }, durationSec: 38,
    },
    { needsAtlas: true },
  ),

  step(
    'What the display and the trace encode — and what they cannot',
    'The trace is running. It sweeps the bundle’s shape at a fixed authored rate; it is not conduction, not direction, not velocity. Colour encodes local fibre orientation, opacity is a display setting, and the number of lines drawn is a sampling cap rather than a fibre census. The cited work is blunt about the failure modes: tensor models break where fibres cross, tractography pipelines generate anatomically plausible pathways that are false positives, and DTI-based tracts loaded into neuronavigation shift once the brain moves. Everything on screen is a candidate reconstruction from a group atlas, rendered for teaching, with no case registration and no per-streamline verification.',
    'Pause the trace mid-sweep and ask what physiological quantity just stopped. Nothing did. Common misconception: that an animated tract shows a signal travelling along it. Then ask which of the three cited failure modes — crossing fibres, false positives, intraoperative shift — each person in the room has personally been caught by, and what they changed afterwards.',
    'model_metric', ['M10', 'M11', 'M12', 'D2'], ['displaySupport', 'cstCandidate'],
    {
      bundles: ['CST'], ghost: ['CPT_F', 'ML'],
      regions: [R(PARCELS.a4)],
      surface: 0.4, camera: { view: 'follow', zoom: 1.25, tweenMs: 900 },
      trace: true, durationSec: 30,
    },
  ),

  step(
    'What does this scene support?',
    'A corticospinal candidate runs from beneath the lit area 4 parcel, past putamen and thalamus, toward the crus, with the cerebellar and lemniscal families ghosted around it. A resident looks at this and says: “so we stay behind the central sulcus and the motor tract is safe.” Before you answer, name the modality, the population and the registration status of everything on screen. Then decide what this picture licenses you to claim in a consent conversation, and what it does not. Take a prediction from the room before revealing the answer.',
    'Make them commit to an answer out loud first. Ask each person to name the modality, the population and the unresolved question. Common misconception: that spatial compatibility between a bundle and a parcel is evidence of a connection in this patient. Close by asking what evidence would actually settle it, and who is responsible for obtaining that before the incision.',
    'schematic', ['M1', 'M2', 'M4', 'D2', 'M11'], ['cstCandidate', 'atlasProvenance'],
    {
      bundles: ['CST'], ghost: ['DRTT', 'ML'],
      regions: [R(PARCELS.a4), R(PARCELS.a6mp), R(PARCELS.a3b)],
      deep: true, deepRegions: ['THA', 'PUT'],
      surface: 0.35, camera: { view: 'follow', zoom: 1.2, tweenMs: 1000 }, durationSec: 30,
    },
    {
      needsAtlas: true,
      question: 'A corticospinal candidate runs beneath the lit area 4 parcel, past putamen and thalamus, toward the crus. A resident says: “so we stay behind the central sulcus and the motor tract is safe.” What does this scene actually license you to say?',
      answer: 'Very little of it. The scene supports one claim: a population-averaged reconstruction is spatially compatible with a population parcellation, in a group brain, with no registration to any patient. It does not show this patient’s fibres, does not order the bundle by body part, and does not include the premotor and postcentral origins that already break the “behind the sulcus” argument — nor the corticopontine, corticostriatal, lemniscal, thalamocortical and dentato-rubro-thalamic systems sharing the same corridor. Function in an individual needs individual evidence: mapping, stimulation, or that patient’s own examination.',
    },
  ),
]);

/* -------------------------------------------------------------------------- */
/* The lesson                                                                  */
/* -------------------------------------------------------------------------- */
export const MOTOR_CST_V2 = Object.freeze({
  id: 'motor-cst',
  title: 'Motor maps & the corticospinal candidate',
  minutes: 16,
  summary: 'Ten steps from a population cortical map to a population bundle atlas and back to the operating room, saying at every stage which claim the picture can carry and which it cannot.',
  goals: Object.freeze([
    'Name the atlas, the population and the registration status before naming any structure.',
    'Separate a candidate reconstruction from a functional measurement and from a clinical association.',
    'Account for the premotor and postcentral origins of the descending system, not area 4 alone.',
    'Trace the shared corridor — corticopontine, corticostriatal, lemniscal, thalamocortical, dentato-rubro-thalamic — that a bundle-only picture hides.',
    'State plainly what the render, the colour and the trace encode, and what is simply not installed.',
  ]),
  steps: STEPS,
  version: '2',
  reviewStatus: 'draft',
  audience: 'Residents and fellows',
  referenceOnly: true,
  sceneGrammar: SCENE_GRAMMAR_VERSION,
});

/* -------------------------------------------------------------------------- */
/* Self-check — cheap structural guard, mirrors validateLessons()'s discipline  */
/* -------------------------------------------------------------------------- */
const EVIDENCE_CLASSES = Object.freeze(['atlas', 'reconstruction', 'association', 'schematic',
  'recovery_overlay', 'functional_measurement', 'model_metric', 'experimental_anatomy', 'conceptual_model']);
const words = s => s.trim().split(/\s+/).length;

export function validateMotorCstV2(lesson = MOTOR_CST_V2) {
  if (lesson.id !== 'motor-cst' || lesson.version !== '2' || lesson.reviewStatus !== 'draft')
    throw new Error('motor-cst v2: identity or review state');
  if (lesson.steps.length !== 10) throw new Error('motor-cst v2: expected 10 steps');
  for (const [i, s] of lesson.steps.entries()) {
    const at = `motor-cst v2 step ${i + 1}`;
    if (!s.title || !s.text || !s.notes) throw new Error(`${at}: missing prose`);
    if (!EVIDENCE_CLASSES.includes(s.evidenceClass)) throw new Error(`${at}: evidence class`);
    if (words(s.text) < 70 || words(s.text) > 120) throw new Error(`${at}: text is ${words(s.text)} words`);
    if (words(s.notes) < 40 || words(s.notes) > 80) throw new Error(`${at}: notes are ${words(s.notes)} words`);
    if (!s.sources.length || s.sources.some(id => !MOTOR_CST_V2_SOURCES[id]))
      throw new Error(`${at}: unresolved source`);
    if (s.regions.some(id => !MOTOR_CST_V2_REGIONS[id])) throw new Error(`${at}: unresolved region card`);
    const sc = s.scene;
    if (!SCENE_SIDES.includes(sc.side)) throw new Error(`${at}: side`);
    for (const b of [...sc.bundles, ...sc.ghost])
      if (!LESSON_BUNDLE_FAMILIES.includes(b)) throw new Error(`${at}: bundle family ${b}`);
    if (sc.bundles.some(b => sc.ghost.includes(b))) throw new Error(`${at}: family both lit and ghosted`);
    for (const r of sc.regions)
      if (!Number.isInteger(r.id) || r.id < 1 || r.id > 180 || !SCENE_HEMIS.includes(r.hemi))
        throw new Error(`${at}: region ${JSON.stringify(r)}`);
    if (typeof sc.deep !== 'boolean') throw new Error(`${at}: deep`);
    if (!sc.deep && sc.deepRegions.length) throw new Error(`${at}: deepRegions without deep`);
    for (const d of sc.deepRegions)
      if (!LESSON_DEEP_STEMS.includes(d)) throw new Error(`${at}: deep stem ${d}`);
    if (typeof sc.surface !== 'number' || sc.surface < 0.08 || sc.surface > 0.95)
      throw new Error(`${at}: surface ${sc.surface}`);
    if (!sc.camera || !SCENE_VIEWS.includes(sc.camera.view)) throw new Error(`${at}: camera view`);
    if (typeof sc.trace !== 'boolean') throw new Error(`${at}: trace`);
    if (!Number.isFinite(sc.durationSec) || sc.durationSec < 25 || sc.durationSec > 40)
      throw new Error(`${at}: durationSec ${sc.durationSec}`);
  }
  const last = lesson.steps[lesson.steps.length - 1];
  if (!last.question || !last.answer) throw new Error('motor-cst v2: final step must be a quiz');
  return true;
}
validateMotorCstV2();

export default MOTOR_CST_V2;
