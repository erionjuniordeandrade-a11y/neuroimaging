# TractLab

A personal reference-station tractography viewer — reverse-engineering the *experience*
of Brainlab Elements Fibertracking (interactive ROI-Boolean seeding, live re-track,
DTI-FACT parity comparison) on top of superior SS3T multi-tissue CSD tractography.

**Scope:** one case, one user, one Mac. Study tool, not a medical device. Loopback-only,
no cloud, no remote. 

## Status

**Slice 1 shipped; hardening in progress.** The loopback viewer has live MPR/ROI
painting, Explore bank loading, typed server outcomes, and a single structured tract
descriptor whose laterality comes from id tokens only. `./verify.sh` runs the Python + JS
suites (`--browser` adds the Playwright gate against the real case); the app remains
research/preview only.

## Display modes

- **Clinical** is the default: presentation glow, constellation points, and
  continuous animation are off.
- **Presenter** uses subdued tubes and cortical context, with no duplicate
  additive glow or autorotation. Deep link with `?profile=presenter`.
- **Teaching** adds a bounded symmetric out-and-back trace over ghosted tubes.
  Deep link with `?profile=teaching`. The trace demonstrates displayed polyline
  continuity only; it does not establish axonal direction, neural conduction,
  function, necessity, safety, or outcome.

Colour, evidence, hull, picking, atlas, and provenance controls remain live in
all three modes. Teaching automatically pauses its trace for reduced-motion
preference.

Legacy `?presentation=1` and `?teaching=1` links still work. The guided reference
atlas and six draft lessons are frozen on the separate
`feat/atlas-lessons-20260907` branch.

## Data (PHI stays out of this repo)

Patient imaging is **read by path** from `~/tractlab-data/cases/local-case/`
(FODs `wmfod_norm.mif`, `t1c_brain_dwi`, masks, the `wb_ifod2` corpus). It is **never**
copied into this tree. A `.gitignore` + a fail-closed `pre-commit` hook (imaging
extensions, rendered rasters, >1 MB blobs) enforce that. Run
`scripts/install_hooks.sh` once per clone — the hook lives in `hooks/` and is
not active until `core.hooksPath` points there.

