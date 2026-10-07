# motor-cst v2 — sources, scene grammar and review notes

Companion to `viewer/lessons/motor-cst.v2.js`.

**Status: DRAFT. Awaiting the surgeon-owner's anatomical review. Nothing here is validated.**
Language: English, matching the existing lesson drafts. Audience: neurosurgery residents
and fellows. No patient data, no case, no PHI.

---

## 1. What the lesson is allowed to point at

| Layer | File | What is installed | What is **not** installed |
|---|---|---|---|
| Cortex | `viewer/atlas/surface.json` → `cortex-L.glb` / `cortex-R.glb` | HCP S1200 group-average midthickness surface, MSMAll registered, 32 492 vertices per hemisphere, MNI | Any individual's surface; any registration to a case |
| Parcellation | `viewer/atlas/surface.json` → `sets.glasser` | HCP-MMP1, 360 regions, one label table **per hemisphere** | Functional borders; individual boundaries; any functional map |
| Bundles | `viewer/atlas/tracts.json` → `tracts.bin` | HCP1065 population-averaged bundles, 87 entries, CC BY-SA 4.0, at most 220 streamlines per bundle resampled to 28 points | Patient streamlines; direction of conduction; axon counts |
| Subcortex | `viewer/atlas/subcortex.json` → `subcortex/*.glb` | Melbourne Subcortex Atlas scale 1, 14 gross structures | Thalamic nuclei (anterior + posterior thalamus are **merged**); STN; red nucleus; **no brainstem nuclei at all** |

Every step of the lesson names the layer it is standing on before it names a structure,
and steps 2, 6 and 9 say explicitly what is missing.

---

## 2. HCP-MMP1 parcel ids used

Read from `viewer/atlas/surface.json`, set `glasser`. The **L and R tables number these
identically**, so every region in the lesson uses `hemi:'follow'` safely.

| Label | id | L key | R key |
|---|---|---|---|
| 4 | **8** | `L_4_ROI` | `R_4_ROI` |
| 3b | **9** | `L_3b_ROI` | `R_3b_ROI` |
| 55b | **12** | `L_55b_ROI` | `R_55b_ROI` |
| 5L | **39** | `L_5L_ROI` | `R_5L_ROI` |
| 24dd | **40** | `L_24dd_ROI` | `R_24dd_ROI` |
| SCEF | **43** | `L_SCEF_ROI` | `R_SCEF_ROI` |
| 6ma | **44** | `L_6ma_ROI` | `R_6ma_ROI` |
| 7PC | **47** | `L_7PC_ROI` | `R_7PC_ROI` |
| 1 | **51** | `L_1_ROI` | `R_1_ROI` |
| 2 | **52** | `L_2_ROI` | `R_2_ROI` |
| 3a | **53** | `L_3a_ROI` | `R_3a_ROI` |
| 6d | **54** | `L_6d_ROI` | `R_6d_ROI` |
| 6mp | **55** | `L_6mp_ROI` | `R_6mp_ROI` |
| 6v | **56** | `L_6v_ROI` | `R_6v_ROI` |

All 14 requested parcels exist. Nothing was missing, so no STOP was raised.

## 3. Subcortical structures used

`scene.deepRegions` carries **stems without a hemisphere suffix**, mirroring the way
`scene.bundles` carries families without a side suffix. The resolver appends `-lh` / `-rh`
from the resolved side, exactly as `bundle:'CST'` became `CST_L` in grammar v1.

| Stem in the lesson | Atlas ids in `subcortex.json` | Name |
|---|---|---|
| `THA` | `THA-lh`, `THA-rh` | Thalamus (anterior + posterior **merged** at scale 1) |
| `PUT` | `PUT-lh`, `PUT-rh` | Putamen |
| `GP` | `GP-lh`, `GP-rh` | Globus pallidus |
| `CAU` | `CAU-lh`, `CAU-rh` | Caudate nucleus |

## 4. Bundle families used

All verified present in `viewer/atlas/tracts.json`.

`CST`, `CPT_F`, `CPT_P`, `CPT_O`, `CS_A`, `CS_P`, `CS_S`, `ML`, `TR_A`, `TR_S`, `TR_P`,
`DRTT`, `SCP`, `MCP`. (`CC` is declared in the permitted vocabulary but not used by any step.)

Sided families are stored as `<family>_L` / `<family>_R`; `MCP` and `SCP` are stored
sideless and must be used verbatim.

---

## 5. Step map

| # | Title | Evidence class | Scene |
|---|---|---|---|
| 1 | Name the surface before you name a tract | `atlas` | parcels 4·6d·6v·6mp·6ma·SCEF·55b, surface 0.92, camera left ×1.4 |
| 2 | Somatotopy is a measurement, not a colour on this brain | `functional_measurement` | parcels 6mp·4·SCEF·24dd·6ma, surface 0.92, camera medial ×1.3 |
| 3 | Reveal the descending reconstruction | `reconstruction` | bundle CST, surface 0.35, camera left ×1.2 |
| 4 | The candidate has more than one cortical origin | `reconstruction` | ghost CST, parcels 4·6d·6v·6mp·3a·3b·1·2, surface 0.88, camera left ×1.35 |
| 5 | Corona radiata into the internal capsule | `reconstruction` | CST + ghost CPT_F/CPT_P/CPT_O, surface 0.12, camera anterior ×1.5 |
| 6 | The deep grey the capsule runs between | `atlas` | CST + ghost CS_A/CS_P/CS_S, deep PUT·GP·THA·CAU, surface 0.08, camera anterior ×1.6 |
| 7 | Sensory return runs the other way | `reconstruction` | ML + TR_A/TR_S/TR_P, ghost CST, deep THA, surface 0.30, camera left ×1.3 |
| 8 | The cerebellar loop is why a bundle-only picture misleads | `reconstruction` | DRTT + SCP + MCP, ghost CST/CPT_F, deep THA, surface 0.25, camera posterior ×1.3 |
| 9 | What the display and the trace encode — and what they cannot | `model_metric` | CST + ghost CPT_F/ML, surface 0.40, camera left ×1.25, **trace on** |
| 10 | What does this scene support? (quiz) | `schematic` | CST + ghost DRTT/ML, deep THA·PUT, surface 0.35, camera left ×1.2 |

Durations run 30–38 s; every step is inside the authored 25–40 s band. Prose is 91–113
words per step, presenter notes 54–66 words. A self-check in the module (`validateMotorCstV2`)
enforces those bands, the scene vocabulary, the surface range and the duration range at
import time.

---

## 6. References (Vancouver)

### Primary teaching references — all PMIDs verified against the NCBI esummary record on 2026-09-07

Verification method: `esummary.fcgi?db=pubmed&id=<PMID>&retmode=json`, matching title,
first author and year against what is cited below.

1. Glasser MF, Coalson TS, Robinson EC, Hacker CD, Harwell J, Yacoub E, et al. A multi-modal
   parcellation of human cerebral cortex. Nature. 2016;536(7615):171–8. PMID: 27437579.
   *esummary matched:* "A multi-modal parcellation of human cerebral cortex." · Glasser MF · Nature · 2016 Aug 11.

2. Gordon EM, Chauvin RJ, Van AN, Rajesh A, Nielsen A, Newbold DJ, et al. A somato-cognitive
   action network alternates with effector regions in motor cortex. Nature. 2023;617(7960):351–9.
   PMID: 37076628.
   *esummary matched:* "A somato-cognitive action network alternates with effector regions in motor cortex." · Gordon EM · Nature · 2023 May.

3. Penfield W, Rasmussen T. The Cerebral Cortex of Man: A Clinical Study of Localization of
   Function. New York: Macmillan; 1950. **No PMID — this is a 1950 monograph, not a PubMed-indexed
   article. Stated openly rather than omitted.**

4. Usuda N, Sugawara SK, Fukuyama H, Nakazawa K, Amemiya K, Nishimura Y. Quantitative comparison
   of corticospinal tracts arising from different cortical areas in humans. Neurosci Res.
   2022;183:30–49. PMID: 35787428.
   *esummary matched:* "Quantitative comparison of corticospinal tracts arising from different cortical areas in humans." · Usuda N · Neurosci Res · 2022 Oct.

5. Holodny AI, Gor DM, Watts R, Gutin PH, Ulug AM. Diffusion-tensor MR tractography of somatotopic
   organization of corticospinal tracts in the internal capsule: initial anatomic results in
   contradistinction to prior reports. Radiology. 2005;234(3):649–53. PMID: 15665224.
   *esummary matched:* title as above · Holodny AI · Radiology · 2005 Mar.

6. Behrens TE, Johansen-Berg H, Woolrich MW, Smith SM, Wheeler-Kingshott CA, Boulby PA, et al.
   Non-invasive mapping of connections between human thalamus and cortex using diffusion imaging.
   Nat Neurosci. 2003;6(7):750–7. PMID: 12808459.
   *esummary matched:* title as above · Behrens TE · Nat Neurosci · 2003 Jul.

7. Schmahmann JD, Pandya DN. Anatomic organization of the basilar pontine projections from
   prefrontal cortices in rhesus monkey. J Neurosci. 1997;17(1):438–58. PMID: 8987769.
   *esummary matched:* title as above · Schmahmann JD · J Neurosci · 1997 Jan 1.

8. Morris EB, Phillips NS, Laningham FH, Patay Z, Gajjar A, Wallace D, et al. Proximal
   dentatothalamocortical tract involvement in posterior fossa syndrome. Brain. 2009;132(Pt 11):3087–95.
   PMID: 19805491.
   *esummary matched:* title as above · Morris EB · Brain · 2009 Nov.

9. Krainik A, Lehéricy S, Duffau H, Vlaicu M, Poupon F, Capelle L, et al. Role of the supplementary
   motor area in motor deficit following medial frontal lobe surgery. Neurology. 2001;57(5):871–8.
   PMID: 11552019.
   *esummary matched:* title as above · Krainik A · Neurology · 2001 Sep 11.

10. Farquharson S, Tournier JD, Calamante F, Fabinyi G, Schneider-Kolsky M, Jackson GD, et al.
    White matter fiber tractography: why we need to move beyond DTI. J Neurosurg. 2013;118(6):1367–77.
    PMID: 23540269.
    *esummary matched:* title as above · Farquharson S · J Neurosurg · 2013 Jun.

11. Maier-Hein KH, Neher PF, Houde JC, Côté MA, Garyfallidis E, Zhong J, et al. The challenge of
    mapping the human connectome based on diffusion tractography. Nat Commun. 2017;8(1):1349.
    PMID: 29116093.
    *esummary matched:* title as above · Maier-Hein KH · Nat Commun · 2017 Nov 7.

12. Nimsky C, Ganslandt O, Hastreiter P, Wang R, Benner T, Sorensen AG, et al. Preoperative and
    intraoperative diffusion tensor imaging-based fiber tracking in glioma surgery. Neurosurgery.
    2005;56(1):130–7; discussion 138. PMID: 15617595.
    *esummary matched:* title as above · Nimsky C · Neurosurgery · 2005.

### Dataset provenance — what is actually rendered (PMIDs also verified)

13. Van Essen DC, Smith SM, Barch DM, Behrens TE, Yacoub E, Ugurbil K. The WU-Minn Human Connectome
    Project: an overview. Neuroimage. 2013;80:62–79. PMID: 23684880. → the S1200 group-average surface.
    *esummary matched:* title as above · Van Essen DC · Neuroimage · 2013 Oct 15.

14. Yeh FC. Population-based tract-to-region connectome of the human brain and its hierarchical
    topology. Nat Commun. 2022;13(1):4933. PMID: 35995773. → the HCP1065 bundle atlas.
    *esummary matched:* title as above · Yeh FC · Nat Commun · 2022 Aug 22.

15. Tian Y, Margulies DS, Breakspear M, Zalesky A. Topographic organization of the human subcortex
    unveiled with functional connectivity gradients. Nat Neurosci. 2020;23(11):1421–32. PMID: 32989295.
    → the Melbourne Subcortex Atlas scale-1 meshes.
    *esummary matched:* title as above · Tian Y · Nat Neurosci · 2020 Nov.

### Unverified, excluded

Nothing was cited without verification. Two candidate leads were considered and dropped
before they entered the lesson:

- **Catani et al., 2013 · frontal aslant / verbal fluency** (PMID 23820597, *Brain*) — verified,
  but it belongs to the FAT language lesson, not to a motor lesson; excluded on scope, not on
  provenance.
- **A dedicated "cerebellar mutism" reference beyond Morris 2009** — PubMed searches returned
  case reports and unrelated imaging papers rather than a primary cohort suitable for citation
  here. None was verified as a fit, so none was included. Morris 2009 carries the
  dentatothalamocortical / posterior fossa syndrome claim on its own.

NCBI was reachable on the first attempt; no STOP was raised.

---

## 7. Hard rules honoured in the draft

- **No fibre counts, percentages or timings stated as fact.** Step 4 deliberately refuses to
  quote a proportion for area 4 versus premotor/postcentral origins, and the presenter note
  tells the presenter to refuse as well. Step 3 says "calibre" is not obtainable.
- **No claim that the atlas locates function in an individual.** Steps 1, 2 and 10 say it
  explicitly; step 2 states that the effector map is *not installed*.
- **Never "the" corticospinal tract.** Every mention is "a corticospinal candidate" or "the
  corticospinal bundle/reconstruction". The two places where the definite phrase appears are
  presenter notes quoting the phrase the presenter is told to correct.
- **Every functional-territory claim is a measurement from a cited study.** Effector /
  inter-effector → Gordon 2023. Homunculus → Penfield & Rasmussen 1950. Capsular somatotopy →
  Holodny 2005, flagged as method-bound. SMA syndrome → Krainik 2001. Cerebellar mutism →
  Morris 2009. None of these is drawn on screen.
- **No PHI, no patient case.** Every use of "patient" is generic and hypothetical.

## 8. Open questions for the owner's review

1. Step 5 calls the corticopontine ordering "frontopontine anteriorly, parietopontine and
   occipitopontine behind" in the capsule/crus — confirm the wording you want for teaching, and
   whether the crus should get its own step.
2. Step 6 asserts caudate "medial and anterior", putamen/GP "lateral", thalamus "posteromedial"
   relative to the capsule. Confirm this is the framing you want given the anterior camera.
3. Step 8 attributes SMA-syndrome recovery to Krainik 2001 as "typically transient". Confirm you
   are content with that hedge, or supply a preferred series.
4. `TR_A`/`TR_S`/`TR_P` are lit together in step 7. If you want the posterior radiation split out
   for a visual-field aside, that is a candidate eleventh step.
5. `CC` is declared in the permitted vocabulary but unused. Say whether a callosal/transcallosal
   motor step belongs in v2 or in a separate lesson.
