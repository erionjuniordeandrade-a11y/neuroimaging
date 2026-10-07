# Case Capsule — v1 spec

One offline, double-clickable HTML file per imaging case. A local Python
pipeline does the heavy work (DICOM decode, co-registration of CT and MRI
into one space, resampling, packing); the HTML only views and annotates.

Use: surgeon review, patient consultation (pt-BR), teaching. Not a diagnostic
device. Every automatic output is shown as unreviewed until the surgeon marks
it reviewed.

## Non-negotiables

1. **Offline.** The capsule makes zero network requests. It carries a CSP meta
   tag: `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';
   img-src data: blob:; connect-src data: blob:; worker-src blob:; font-src data:`.
   No CDN, no external fonts, no analytics. All code is inlined.
2. **No identifiers.** The pipeline never copies DICOM identifying fields
   (names, IDs, birth date, accession, institution, physician, dates other than
   a coarse study year, UIDs, device serials) into the capsule. The only case
   label is the `--label` string the surgeon passes. Removing identifiers does
   not remove a face: face removal is opt-in. The capsule shows
   "Pseudonimizado" unless the face was removed from every volume. Verified by
   a canary test (see Tests).
3. **Geometry is truth.** Orientation comes from ImagePositionPatient /
   ImageOrientationPatient (via dcm2niix + SimpleITK), never from file order.
   Patient left is displayed on image right in axial and coronal views
   (radiological convention). Distances are true millimetres.
4. **Values are truth.** CT volumes keep Hounsfield units (rescale slope and
   intercept applied). MR volumes keep their raw scalar ("a.u.").
5. **Never overwrite.** Saving annotations from the viewer downloads a new file
   with an incremented version suffix; the opened file is never modified.

## Repository layout

```
capsule/            Python package (pipeline) — owned by the pipeline worker
  cli.py            `capsule scan`, `capsule build`
  ingest.py         DICOM folder -> series table -> NIfTI via dcm2niix
  deid.py           allowlist of fields that may enter the manifest
  register.py       SimpleITK rigid co-registration
  resample.py       common grid, head crop, dtype packing
  pack.py           manifest + volume blobs -> viewer template -> .capsule.html
  phantom.py        synthetic DICOM series generator (tests + demos)
viewer/
  template.html     the whole viewer (HTML+CSS+JS inline) — owned by the viewer worker
  dev_fixture.py    writes a phantom payload into the template without the pipeline
tests/              pytest (pipeline) + Playwright (viewer, real Chromium)
```

Run everything with `uv run` (pyproject declares pydicom, nibabel, numpy,
SimpleITK, pytest, playwright, and Pillow for anatomy QA montages). dcm2niix is resolved from `$DCM2NIIX`, then
`PATH`, then `~/fsl/bin/dcm2niix`.

## Pipeline

### `capsule scan <dicom_dir>`
Recursively reads headers only (pydicom, `stop_before_pixels`), groups by
SeriesInstanceUID, and prints one row per series: series number, modality,
series description, sequence hints (ScanningSequence/SequenceName/contrast
agent present yes/no), rows×cols×slices, pixel spacing, slice thickness,
transfer syntax. It prints **no** patient/study identifiers and no file paths
inside the folder (only counts). Handles DICOMDIR folders and files without a
`.dcm` extension; skips non-DICOM files silently with a count.

### `capsule build <dicom_dir> --series N[,N...] --label "Caso 01" -o out/x.capsule.html [--reference N] [--spacing 1.0] [--max-voxels 64000000] [--crop-ras X0,Y0,Z0,X1,Y1,Z1 | --crop-around tumour --margin-mm M] [--anatomy auto|none] [--deface]`
1. Convert each chosen series with dcm2niix into a temp dir (`-z n -b y -ba y
   -f series_%s`), which handles gantry tilt, compressed transfer syntaxes,
   enhanced multi-frame. For 4D outputs (e.g. DWI), keep volume 0 only in v1
   and record that in the manifest.
2. Reference volume: `--reference` if given, else the CT if one is chosen,
   else the first MR series.
3. Rigid registration (SimpleITK `Euler3DTransform`, Mattes mutual
   information, 3-level pyramid, geometry-centred initialiser, then moments if
   the metric is worse). Every non-reference volume is registered to the
   reference. Record the final metric value and the transform parameters in the
   manifest. No deformable registration in v1.
4. Common grid: one isotropic `--spacing` (default 1.0 mm, allowed 0.4–1.5 mm).
   The uncropped grid keeps the reference volume's orientation; an explicit RAS
   or mask crop uses an RAS-aligned grid so its corners stay inside the requested
   box even when the reference is oblique. By default, crop to the head foreground
   bounding box plus 10 mm (CT: largest component of voxels > −500 HU after a
   2 mm binary opening; MR: Otsu foreground). `--crop-ras x0,y0,z0,x1,y1,z1`
   selects an RAS+ mm box intersected with that foreground box. Alternatively,
   `--crop-around MASK_ID --margin-mm M` uses a known lesion mask's RAS bounding
   box plus M mm. The manifest records the effective RAS box after intersection
   with the head foreground box. `build-nifti` takes that mask from `--crop-mask PATH:ID`, with
   ID matching MASK_ID; `build` uses its selected RTSTRUCT lesion (`tumour`).
   The default 64,000,000 voxel budget is per volume (`--max-voxels`). Without
   an explicit crop, the pipeline raises spacing as needed and prints the actual
   spacing. An over-budget explicit crop fails with its voxel count and a
   remedy. All volumes share this grid (linear interpolation for images,
   nearest neighbour for masks). Final dimensions, voxel count and estimated
   float32 raw bytes are printed.
5. Packing: CT → int16 HU. MR → uint16 after linear scaling of [min, max] (never a percentile clip, which
   flattens small bright structures such as an enhancing lesion)
   into 0..65535, with `slope`/`intercept` recorded so the viewer recovers the
   original scalar. Each array is C-order bytes with x fastest
   (`array[z, y, x]`), gzip-compressed, base64-encoded.
6. Write the capsule: the template with the manifest and blobs injected (see
   Capsule format). Print the output path, size in MB and per-volume stats.

`capsule build` and `build-nifti` accept optional `--deface` (default off).
It requires an MR volume and a successfully produced brain mask; otherwise the
build fails with an explanation. On the common grid, the pipeline projects the
brain mask onto the sagittal anterior-superior plane, shifts its
anterior-inferior convex-hull edge 5 mm away from the brain, and sets voxels
anterior/inferior to that plane to air in every volume (MR minimum, CT −1000
HU). Brain voxels are protected, and the same region is removed from render
head masks and generated anatomy labels. This option is off by default to
retain nasal and orbital anatomy needed for skull-base planning.

### Automatic anatomy layers (pipeline round 3)

`capsule build` defaults to `--anatomy auto` (or `$CAPSULE_ANATOMY` when set)
and to the v2 viewer. `--anatomy none` disables the tool runs; automatic
anatomy requires `--viewer v2`. Tests set `CAPSULE_ANATOMY=none` so the fast
suite does not download weights or run segmentation.

The MR layer runs FreeSurfer `mri_synthseg` on the reference MR (else the first
MR), with `--robust --cpu --threads 1` and without `--parc`. The current local
runtime is FreeSurfer 7.4.1; the manifest records the exact detected version.
The SynthSeg output is a 1 mm label volume in its own physical geometry. The
pipeline maps it onto the capsule grid by nearest neighbour using that series'
physical geometry and registration transform. It keeps the left/right lateral
and inferior lateral ventricles, third and fourth ventricles, brainstem,
left/right cerebellum (cortex and white matter merged), thalami, caudates,
putamina, pallida, hippocampi and amygdalae. Cerebral cortex and cerebral white
matter are excluded because the brain render already shows those surfaces.
On Apple Silicon with an x86 FreeSurfer install, the same FreeSurfer Python
script runs in a temporary native Python environment with TensorFlow 2.17's
legacy `tf-keras` API; a local `numpy.float128` alias handles FreeSurfer 7.4.1
on arm64 without changing the installation.

The CT layer runs one TotalSegmentator multi-label prediction for each verified
open task present in the installed class map: `craniofacial_structures`,
`headneck_bones_vessels`, `head_glands_cavities`, `ventricle_parts`, and the
skull, globes and optic nerves from `oculomotor_muscles`. The
current installed version is 2.18.0. The class map is queried at runtime; a
missing expected class is reported for that task rather than guessed. The CT
reference is used when present, else the first CT. Outputs use nearest-neighbor
resampling onto the same capsule grid. Each task becomes one uint8 anatomy
volume. `TotalSegmentator` code and open-task weights are Apache-2.0. The
FreeSurfer code and SynthSeg model use the FreeSurfer Software License
Agreement; both method/version and licence are stored on each item.

Tasks requiring an unregistered licence are skipped and reported. In the
installed class map, `brain_structures` and `face` are licensed. The separate
`teeth` task is trained for narrow-field CBCT, so it is not applied to a
general head CT; the craniofacial task supplies the head CT tooth labels.
Parotid and submandibular glands from `head_glands_cavities` are omitted
because the prescribed group vocabulary has no gland group.
`head_muscles` and `headneck_muscles` are not emitted because the shared anatomy
group vocabulary has no muscle group. The class maps currently do not provide a
mastoid or vertebral artery label in the selected tasks.

An anatomy item is `{id, blob, for_volume, source, method, licence, reviewed,
labels}`. IDs are `anat_mr` and `anat_ct_<task>`; blobs are
`anatomy_<id>`. Each item is one multi-label uint8 volume on the common grid,
`source: "auto"`, and `reviewed: false`. Label values are unique within the
item. `volume_ml` is independently derived as voxel count × all three grid
spacings / 1000; no scalar image is changed. Tool failures are printed with
their exact task and diagnostic while the capsule continues with other
successful anatomy items.

`scripts/anatomy_review.py CAPSULE` writes a TSV with every label's voxel count
and `volume_ml`, plus a JSON control report and a PNG montage. The controls
recount manifest volumes from the decoded bytes, compare left/right paired
volumes (flagging asymmetry above 1.5), check that SynthSeg left/right lateral
ventricle centroids have negative/positive RAS x, and require the same-capsule
brain render mask to cover at least 95% of SynthSeg non-background voxels.
The montage has axial, coronal, and sagittal overlays through each anatomy
item and must be visually inspected.

### Capsule format (the only contract between pipeline and viewer)

Inside `template.html` the pipeline replaces exactly one placeholder,
`<!--CAPSULE_PAYLOAD-->`, with:

```html
<script id="capsule-manifest" type="application/json">{...}</script>
<script id="capsule-blob-<blob_id>" type="application/octet-stream" data-encoding="gzip+base64">...</script>
```

Manifest (JSON, schema `case-capsule/1`):

```json
{
  "schema": "case-capsule/1",
  "version": 1,
  "created_utc": "2026-09-26T18:00:00Z",
  "generator": "case-capsule 0.1.0",
  "case": {"label": "Caso 01", "anonymized": false, "study_year": 2025,
           "deidentification": {"identifiers_removed": true, "face_removed": false, "method": null}},
  "locale": "pt-BR",
  "grid": {
    "dims": [nx, ny, nz],
    "spacing_mm": [sx, sy, sz],
    "requested_spacing_mm": 1.0,
    "spacing_raised": false,
    "crop": {"ras_mm": [-20, -30, -40, 20, 30, 40], "source": "ras", "margin_mm": 0.0},
    "affine_ras": [[...4...],[...],[...],[0,0,0,1]]
  },
  "volumes": [{
    "id": "ct", "blob": "ct", "kind": "CT", "label": "TC crânio",
    "dtype": "int16", "slope": 1.0, "intercept": 0.0, "units": "HU",
    "source_spacing_mm": [0.45, 0.45, 0.6], "resampled": true,
    "series": {"number": 3, "description": "...", "modality": "CT", "frames_used": "all"},
    "stats": {"min": -1024, "max": 3071, "p01": -1000, "p99": 1500},
    "window_presets": [{"name": "Cérebro", "center": 40, "width": 80}],
    "registration": {"reference": true}
  }],
  "masks": [{"id": "lesion", "blob": "mask_lesion", "label": "Lesão", "color": "#E4572E",
             "blob_sha256": "<sha256>", "volume_ml": 12.3, "source": "surgeon", "reviewed": false}],
  "anatomy": [{"id": "anat_mr", "blob": "anatomy_anat_mr", "for_volume": "mr",
               "blob_sha256": "<sha256>",
               "source": "auto", "method": "synthseg 2.0 (FreeSurfer 7.4.1, --robust)",
               "licence": "FreeSurfer Software License Agreement", "reviewed": false,
               "labels": [{"value": 1, "key": "brainstem", "name": "Tronco encefálico",
                           "group": "Tronco e cerebelo", "color": "#CF896C", "volume_ml": 18.4}]},
              {"id": "anat_mr_gyri", "blob": "anatomy_anat_mr_gyri", "for_volume": "mr",
               "labels": [{"value": 1, "key": "precentral_left", "soft_volume_ml": 22.1}],
               "qc": {"method": "synthseg hard-vs-soft parcel agreement",
                      "ratio_bounds": [0.5, 2.0], "max_failing": 2, "n_failing": 0,
                      "failing": [], "verdict": "PASS",
                      "calibration": "pilot: 1 template + 2 patient scans"}}],
  "annotations": [],
  "tour": []
}
```

`affine_ras` maps voxel index (i, j, k) to RAS+ millimetres. Masks are uint8
(0/1) on the same grid. `annotations` items: `{"id","type":"distance"|"angle"|
"point"|"trajectory","points_ras":[[x,y,z],...],"label","value","units"}`.
`tour` items (patient mode steps): `{"id","title","text","view":{"layout",
"crosshair_ras","window":{"volume","center","width"},"camera":{...},
"visible_masks":[...],"visible_annotations":[...]}}`. Unknown fields must be
preserved on save.

The `anat_mr_gyri` item may include parcel `qc` with the SynthSeg hard-label to
soft-volume agreement gate. Its `verdict` is `PASS`, `FAIL`, or `UNAVAILABLE`;
when its `--vol` CSV is missing or invalid, `n_failing` is null and `failing` is
empty. Each gyri label's `soft_volume_ml` is null when that CSV is unavailable.
The viewer hides `FAIL` gyri parcels and their volumes until the surgeon enables the overlay, then keeps a QC warning visible. It marks failing labels as `falhou QC`; `UNAVAILABLE` and legacy items display a QC status note. Whole-structure `anat_mr` behavior is unchanged.

### Content hashes and review signatures

New fields are optional under `case-capsule/1`; older capsules continue to
load. New builds compute `blob_sha256` as lowercase SHA-256 over each mask,
anatomy, and primary tract blob's raw C-order bytes before gzip/base64
encoding. Tracts also carry `source_sha256`, the hash of the original `.tck`
bytes. Source names, paths, and header command strings are never copied.

Every reviewable mask, anatomy item, or tract may have
`review: {"by": string, "at_utc": ISO-8601, "blob_sha256": hex}` while keeping
the legacy `reviewed` boolean. A review counts only when `reviewed` is true,
the reviewer and timestamp are valid, and `review.blob_sha256` equals the
item's current `blob_sha256` and the decoded blob. A changed mask loses its
review. Old `reviewed: true` items without a review signature are shown in
surgeon mode as "revisado (sem assinatura)" and count as unreviewed in patient
mode. A surgeon enters `case.reviewer` in the viewer before signing; the name
and each review signature are saved with the capsule.

An optional `case.deidentification` is
`{"identifiers_removed": true, "face_removed": bool, "method": string|null}`.
New builds set `case.anonymized` to the same value as `face_removed`. The
viewer trusts `face_removed` for its badge; old capsules without this record
are treated as pseudonymized. `volume.skull_stripped_input: true` is present
when the MR has almost no signal outside the brain (`signal_outside_brain_fraction`
< 0.05 on the `head` mask: voxels beyond 3 mm of the brain above 10 % of the brain
median, per brain voxel). The MR `head` mask records
`brain_outside_head_fraction_before`, measured before the brain is unioned into
the filled head mask.

Tract entries may include `outlier_blob` and `outlier_blob_sha256` for the
sampled, step-decimated streamlines removed by the display filter, plus
`n_streamlines_outlier_blob`. The original `n_streamlines_outliers` remains the
full dropped count. The `provenance` object is either `{ "recorded": false }`
or `{ "recorded": true, ... }` with only scalar allowlisted tracking fields
(algorithm, seeding, select/count, step, angle, cutoff, length bounds, ACT,
SIFT2, prior/template, software/version, b-values, directions, and voxel size).
The secondary outlier blob has its own hash because it is a distinct payload.

**Mask roles (r2).** Every mask keeps `{id, blob, label, color, volume_ml, source,
reviewed}` (uint8 0/1 on the grid) and adds:

- `role`: `"lesion"` (default when absent) or `"render"`. A render mask is display-only:
  the viewer uses it only to hide voxels outside the mask in its 3D render copy of
  `for_volume` (set them to that volume's minimum, not 0 — 0 HU is water). It never
  appears in the mask list and never changes 2D values or readouts.
- `for_volume`: the volume id the mask belongs to.
- `source`: `"dataset"` (e.g. an RTSTRUCT), `"auto"` (pipeline algorithm) or `"surgeon"`.
- Render masks also carry `method` (the algorithm that actually ran, e.g. `synthstrip` or
  `bet` for the brain); lesions from an RTSTRUCT carry `roi`, `native_volume_ml` (voxel
  count on the referenced native series) and `planar_volume_ml` (sum of contour shoelace
  areas × slice spacing, an independent check of the raster).

Mask ids are globally unique. A kind of render mask is named once (`head`, `brain`); the
same kind for a further volume gets `<name>_<volume id>` (e.g. `head_mr_3`). Viewer lookup
for preset NAME on volume V: `role == "render" && for_volume == V && (id == NAME ||
id.startsWith(NAME + "_"))`. Presets: "TC Osso" → `head` of the CT, "RM Cérebro" →
`brain`, "RM Pele" → `head` of the MR.

**Grid fidelity (r9).** `grid.requested_spacing_mm` records the requested
isotropic spacing and `grid.spacing_raised` says whether the voxel budget
increased it. Both fields are optional for older capsules. An explicit crop
adds optional `grid.crop: {"ras_mm": [x0,y0,z0,x1,y1,z1], "source": "ras" |
"mask:<id>", "margin_mm": number}`. Each new volume may also record optional
`source_spacing_mm: [sx,sy,sz]` from the source image header and `resampled`.
`resampled` is false only when source and common-grid spacing match within
0.001 mm, orientation and registration introduce no rotation or scaling, and
the grid origin falls on a source voxel centre. Older capsules without these
fields continue to load. In surgeon mode, the viewer shows native and common
grid resolution for resampled volumes; patient mode omits it.

**CTA vessels beside bone (r9).** The paired-CT mask is the r7 bone-masked
subtraction united with a min-max subtraction. Min-max core: the contrast scan,
Gaussian-smoothed (0.5 mm) and eroded (local minimum over a one-voxel box), must
exceed the native scan, smoothed and dilated (local maximum over the same box), by
>60 HU, with contrast >120 HU inside the head mask. A bone edge shifted by up to
one voxel between phases cannot pass in either direction, so no bone exclusion
zone is needed and lumens in the carotid canal or against the inner table are
kept. The core regrows by the same box into voxels with contrast >120 HU that
still exceed the one-voxel native maximum by >30 HU (not clipped to the head
mask, which can miss a canal-wall voxel). The r7 mask is added and components
below 100 mm³ are removed. The method string records these parameters. Each CTA
mask records component count, largest-component fraction and mL within 2 mm of
native bone (>200 HU). The older bone-masked method stays available as
`cta_vessel_mask_r7`. Verification: synthetic volumes (canal, inner-table sinus,
one-voxel shift plus kernel mismatch with phases swapped) and one pilot patient
CTA scored locally against TotalSegmentator cervical ICA labels (N=1; not
validated). The box is one voxel, so the method depends on the grid: the pilot was
scored at 0.6 mm; build CTA capsules with `--spacing 0.6` (default 1 mm). The
component count uses the same 26-connectivity as the size filter.
No public paired CTA/non-contrast CT exists.

The pipeline emits render masks **only for `--viewer v2`** (v1 would draw them as lesions):

- CT `head`: HU > −500, 3 mm ball opening, largest connected component, then enclosed
  holes filled per axial slice and in 3D (excludes the table/headrest).
- MR `head` per MR volume: Otsu on log intensity of positive voxels, 2 mm opening, largest
  component, enclosed holes filled (excludes frame fiducials). When a brain
  mask exists, the pipeline records the brain-outside-head fraction before
  repair, then fills the union of head and brain. It flags a volume as
  `skull_stripped_input: true` when there is almost no signal outside the brain
  (`signal_outside_brain_fraction` < 0.05); a small head mask alone is not enough,
  since log-Otsu can miss much of a real T2 head. The viewer hides its MR skin preset.
- MR `brain`: SynthStrip (FreeSurfer model `synthstrip.1.pt`, run through `uv` with torch
  2.14.0, surfa 0.6.3, numpy 1.26.4) on the reference MR (else the first MR), falling back
  to FSL `bet -m -f 0.4 -R`; `--brain-mask {synthstrip,bet,none}` (default
  `$CAPSULE_BRAIN_MASK` or synthstrip; tests default to none). One brain array is written
  per MR volume (they share the registered grid).

Hole filling only closes regions fully enclosed by the mask, so it cannot grow the mask
outward. The 3D pass leaves open cavities (airway, ear canal) unfilled. The per-axial-slice
pass can fill air that is enclosed in that plane (e.g. the pharyngeal lumen at some levels).
Neither pass changes a voxel value, and the mask is display-only: filled air keeps its air
value and never forms a rendered surface or enters a lesion volume.

`capsule build ... --rtstruct PATH [--rtstruct-roi NAME] [--rtstruct-label TEXT]` adds a
lesion mask `tumour` (label default "Schwannoma vestibular (dataset)", color `#E4572E`,
source `dataset`, reviewed false, role `lesion`). The RTSTRUCT must reference exactly one
selected series (matched by SeriesInstanceUID in code). CLOSED_PLANAR contours are mapped
through that series' IPP/IOP geometry to continuous indices (each contour must lie on a
slice within 0.1 voxel), filled by the even-odd rule on pixel centres, XOR-combined per
slice (holes), then resampled onto the grid through that series' registration (linear,
≥ 0.5). Lesion masks are emitted for v1 and v2.

CT window presets (center/width): Cérebro 40/80, Subdural 75/215, AVC 35/40,
Partes moles 40/400, Osso 600/2800, Osso temporal 700/4000. MR presets:
"Auto" from p01..p99 of the volume, plus manual.

## Export to DICOM (r10)

### `capsule export --capsule SAVED.capsule.html --dicom SOURCE_DIR --out OUT_DIR [--series N] [--masks ID1,ID2] [--include-unreviewed]`

Writes `seg.dcm` (DICOM-SEG, BINARY, highdicom), `rtstruct.dcm` (rt-utils) and
`export.json` into `OUT_DIR`, on the **native grid of the original DICOM series**
(not the capsule's common grid). `OUT_DIR` must be absent or empty; nothing is
ever overwritten.

- **Mask selection.** Without `--masks`: every mask whose role is `lesion`,
  `structure` or `segmentation` (or a pre-role mask with source `surgeon`/`dataset`);
  render masks (`role: "render"`, ids head/brain/vessels) are never exported by default.
  Only **signed-reviewed** masks are exported: `reviewed === true`, non-empty
  `review.by`, valid `review.at_utc`, and `review.blob_sha256 == blob_sha256 ==`
  sha256 of the blob. A legacy `reviewed: true` without `review` counts as unsigned.
  A mask named in `--masks` that is unsigned is an error, not a silent skip.
- **`--include-unreviewed`** exports unsigned masks too; every segment/ROI label and
  both SeriesDescriptions then end in `NAO REVISADO`, and unsigned SEG segments are
  typed SEMIAUTOMATIC with algorithm "Case Capsule" (signed ones MANUAL).
- **Target series.** `--series N` picks the capsule volume with `series.number == N`
  (must be unique); without it, all selected masks must share one `for_volume`.
  The source folder must contain exactly one series N, and its spacing must match
  the volume's `source_spacing_mm` (±0.01 mm) or the export stops.
- **Geometry.** Each mask is resampled nearest-neighbour from `grid.affine_ras` onto
  the source series, through the volume's `registration.moving_to_reference_ras`
  (identity for the reference / registration-disabled volume), converted RAS→LPS.
  A mask that lands entirely outside the series is an error.
- **Identity** (patient, study, frame of reference, referenced SOP instances) comes
  from the source DICOM only, never from the capsule. New Series/SOP UIDs are
  generated; SeriesNumber = N + 1000.
- **Coding.** role `structure` → SCT 91723000 Anatomical Structure; everything else
  → SCT 49755003 Morphologically Altered Structure / 4147007 Mass.
- **Receipt** `export.json`: per mask `{id, label, reviewed_by, review_at_utc,
  blob_sha256, voxels_native, volume_ml_native}`, `target_series_number`,
  `n_referenced_instances`, `software_versions`. It and stdout carry no patient
  names, IDs, UIDs or paths (`reviewed_by` is the surgeon's signature).
- **Staging.** Source slices are copied into a hidden temporary folder beside
  `OUT_DIR` (never the system temp dir) and removed on exit.

Verified on public Vestibular-Schwannoma-SEG (tests/test_export.py): T1 RTSTRUCT →
capsule → export Dice vs the original contour 0.925 (RTSTRUCT) / 0.949 (SEG);
cross-series export onto T2 matches the registered expectation (SEG 1.000,
RTSTRUCT 0.938); a 5 mm shift drops SEG Dice to 0.479 (control).

## Viewer (template.html)

Vanilla JS + WebGL2, no framework, no external requests. pt-BR UI strings from
one string table (English table alongside). Dark glass UI, accent `#C9A84C`,
system font stack, inline SVG icons. Decompress blobs with the native
`DecompressionStream('gzip')`; upload each volume as a WebGL2 3D texture
(R16I/R16UI or normalised float; pick what renders correctly and document it).

**Surgeon mode (Cirurgião)**
- 2×2 layout (axial, coronal, sagittal, 3D) plus single-view maximise, all
  linked by one crosshair in RAS mm. Orientation letters on every 2D view.
- Scroll = slice; drag = crosshair; wheel+modifier = zoom; pan; reset.
- Oblique: rotate the coronal/sagittal planes about the crosshair (like the
  reference viewer) with a handle on the axial view.
- Base volume selector and an overlay (fusion) volume selector with opacity
  slider and colormap (grey, hot, cool); e.g. CT base with MR overlay.
- Windowing: presets per volume, level/width sliders + numeric inputs,
  right-drag to window, histogram strip, invert.
- Slab: thin / MIP / MinIP / mean with thickness in mm.
- 3D: GPU ray-marching of the base volume with modes Osso (CT threshold),
  Pele+osso, MIP, and mask surfaces rendered in their colours; threshold and
  opacity sliders; cut at crosshair (off/axial/coronal/sagittal); slice planes
  drawn in 3D; drag to rotate, scroll to zoom.
- Readout bar: RAS mm under the cursor, voxel index, value with units (HU for
  CT) for the base and overlay volumes.
- Tools: distance (mm), angle (°), point, trajectory (entry→target line shown
  in all views, length in mm), ROI circle with mean ± SD in units.
- Lesion mask: seeded 3D region growing on a chosen volume (click seed,
  intensity window from the seed neighbourhood, adjustable tolerance, 6-connected,
  bounded by a max radius), then brush add/erase on 2D slices; live volume in mL
  = voxel count × voxel volume. "Revisado pelo cirurgião" checkbox per mask.
- Tour editor: "Adicionar passo" captures the current view into a tour step
  with a title and pt-BR text typed by the surgeon; reorder/delete steps.
- "Salvar cápsula": serialises the current document (template + manifest with
  new annotations/masks/tour + all blobs) into a new file
  `<name>.v<version+1>.capsule.html` via Blob download. Masks are re-encoded
  gzip+base64 with `CompressionStream`.
- Screenshot (PNG) of the active view.

**Patient mode (Paciente)** — pt-BR only
- Large, calm layout: one main view (3D or one slice) and the tour text
  panel; "Anterior / Próximo" through tour steps; each step restores its view.
- Only items with a valid person/time/blob-bound review signature are shown;
  unsigned legacy `reviewed: true` items are hidden. The surgeon enters a
  reviewer name before signing an item.
- No measurement tools, no numbers except those the surgeon wrote in step text.
- If the tour is empty, patient mode shows a notice that the surgeon has not
  prepared the explanation yet.

**Status/badges:** "Pseudonimizado" unless `case.deidentification.face_removed`
is true, then "Anonimizado"; also "Offline", GPU name if available, capsule
version. Handles loading (progress while decompressing), empty (no volumes),
error (bad payload, WebGL2 unavailable) states with pt-BR messages.

## Tests (the verify gate)

`uv run pytest -q` must pass, including:

- **Phantom** (`capsule/phantom.py`): writes synthetic DICOM series of the same
  "head" — a CT (HU: air −1000, soft tissue 40, a bone shell 1000, a 20 mm
  "lesion" sphere at 60 HU) and an MR T1 of the same object with a known rigid
  offset (e.g. 7° about z + 4 mm translation) and anisotropic voxels (1×1×3 mm
  oblique acquisition allowed). A small 8 mm marker sits on the **patient
  left** only. Optional tilted-gantry CT variant.
- **Canary de-id:** phantom headers carry canary strings in every identifying
  field (generated, e.g. `CANARY-<random>`); after `build`, the capsule and the
  scan output contain none of them (fixed-string search on the file bytes and
  on the decoded manifest).
- **Geometry:** in RAS+, +x points to the patient's RIGHT, so patient left is
  −x. In the built capsule the patient-left marker's RAS centroid has x < 0, and
  the lesion centroid matches the phantom truth within 1 mm.
- **HU preserved:** mean HU inside the lesion sphere (eroded) = 60 ± 5; inside
  air = −1000 ± 10.
- **Registration:** after build, the MR lesion centroid lies within 1.0 mm of
  the CT lesion centroid, and the recovered transform differs from the known
  one by < 1° and < 1 mm; same-path control: a build with registration disabled
  (`--no-register`) shows the known offset (> 3 mm), proving the test can fail.
- **Offline:** the capsule contains no `http://`/`https://` outside the CSP
  string and comments; Playwright loads it from `file://` in Chromium with all
  network requests recorded → zero requests other than the file itself; zero
  console errors.
- **Viewer smoke (Playwright):** loads the phantom capsule, WebGL2 context ok,
  the readout at the lesion centre shows 60 ± 5 HU, the axial view shows the
  left marker on the image right half (read pixels of the axial canvas), a
  saved capsule round-trips (download → reopen → annotation still present).

## Tracts and NIfTI input (pipeline v1.1)

### `capsule build-nifti --volume PATH:KIND:LABEL [--volume ...] [--volume-registration LABEL:METHOD] --label L -o OUT [--tract PATH:LABEL[:#RRGGBB] ...] [--max-streamlines 1500] [--tract-step-mm 1.0] [--spacing 1.0] [--max-voxels 64000000] [--crop-ras X0,Y0,Z0,X1,Y1,Z1 | --crop-around ID --crop-mask PATH:ID --margin-mm M] [--viewer v1|v2] [--deface]`
For volumes already in one world space (e.g. TractLab outputs). KIND is `CT`
or `MR`; the first volume is the reference grid (same crop/resample/pack as
`build`). No registration is run: each volume records
`registration: {"reference": <bool>, "method": "none-shared-world"}` unless
`--volume-registration LABEL:METHOD` supplies a method for that label. The
option records metadata and does not run registration. Only voxels and geometry
are read; no NIfTI header text (descrip, aux_file,
intent_name, db_name, ...) enters the capsule. `series` is
`{"number": null, "description": "(nifti)", ...}`.

`build` accepts the same spacing, voxel-budget and crop controls, except
`--crop-around ID --margin-mm M` selects the lesion produced by its RTSTRUCT
input (currently ID `tumour`). Crop boxes are mutually exclusive. With NIfTI
input, `--crop-mask PATH:ID` is a 3D mask in shared world coordinates used only
to choose the crop; it is not emitted as a lesion.

`capsule build` also accepts `--tract`, `--max-streamlines` and `--viewer`.
DICOM volume labels come from the allowlisted description, else a fixed
weighting token found in it (`RM T1`, `RM T2`, `RM FLAIR`, ...; never the free
text), else `TC`/`RM`; duplicates get ` (série N)`.

**R16 backend.** `--lesion-mask PATH[:LABEL]` packs a nearest-neighbour lesion
mask on the capsule grid; `--tract-max-length-mm` drives full-source tract
trust before display cleanup, with contralateral tortuosity, maximum-length
cap, and informational lesion-rim metrics. `metrics.yield_ratio` is the tract's
full-source streamline count over its contralateral tract's count (null without
a contralateral of at least `min_streamlines`); `yield_flag` is true below 1/3. Like the rim flag,
it is informational and never changes the verdict. `--brain-volume-gate MIN,MAX|off`
records method attempts and suppresses a failed brain render mask.
`--brain-mask-file PATH[:synthstrip|bet]` gates and packs a precomputed mask in
the volumes' world space without running a tool; its attempt carries
`"source": "file"`. MR vessel
render masks are limited to post-contrast-labelled volumes and clipped to the
brain mask.

`--viewer v2` (NiiVue) is the default for `capsule build` and `build-nifti`;
v1 is the legacy WebGL viewer. Automatic anatomy is v2-only, so `--viewer v1`
requires `--anatomy none`; `build-nifti` does not run automatic anatomy. The
viewers use `viewer/template.html` (v1) or `viewer2/template.html` (v2) and fail clearly
if the selected template is absent. An existing output path is always an error.

### `capsule dwi`

`capsule dwi --dwi DWI.nii.gz --bvec DWI.bvec --bval DWI.bval --rpe REVERSE_B0.nii.gz --t1 T1.nii.gz --work OUTSIDE_REPO --label LABEL -o OUT.capsule.html [--json DWI.json] [--rpe-bvec B0.bvec --rpe-bval B0.bval] [--pe-dir j-] [--readout-time 0.05] [--flair FLAIR.nii.gz] [--lesion LESION_T1.nii.gz] [--profile full|fast] [--seeds N] [--bundles ...] [--threads N] [--act auto|off] [--no-deface] [--adopt-existing] [--check-tools] [--dry-run]`

The ordered, resumable stages are `preproc`, `t1`, `register`, `fod`,
`track`, `qc`, and `capsule`. Their outputs retain the legacy `nii/`, `pp/`,
`t1/`, and `<profile>/{nifti,tracts,qc}` layout. A successful stage writes a
JSON marker with its parameters, tool versions and output sizes; matching
markers with intact outputs are skipped. Input, tool and atlas files in the
parameters are identified by path, size and SHA-256, so a rewrite with the same
content does not rerun a stage. Older markers that recorded file times still
match while the file time is unchanged. `--adopt-existing` can mark only
legacy `preproc` and `t1` outputs as adopted. Other stages can be skipped only
when their own successful marker still matches.

`full` uses SS3T and 1,500,000 seeds per bundle. `fast` uses two-tissue
MSMT-CSD and 750,000 seeds per bundle; its run record states that shortcut.
Before any stage, the command checks the MRtrix, FSL/XTRACT, ANTs and
profile-specific dependencies. Registration records the asymmetric orientation
sentinel with mirrored positive control and the independent FLIRT-versus-ANTs
check against three 2 mm shift controls. A failed sentinel or regcheck stops
before tracking. Bundle QC records descriptive sanity checks for streamline
count, midline crossings, and wrong-hemisphere centroids, plus the seed budget.
These checks are not a tract-validity test. `--dry-run` prints the ordered argv
plan without creating files or invoking external tools.

Anatomically constrained tracking (r16, `--act auto`, the default) adds an
`act` stage between `register` and `fod`: SynthStrip on the DWI-world T1+C,
`fslmaths -mas`, `5ttgen fsl -premasked -nocrop`, then, with `--lesion`, the
lesion is moved to DWI space with the registration's FLIRT matrix, resampled
nearest-neighbour onto the 5TT grid and written as pathological tissue with
`5ttedit -path`; `5ttcheck` validates the result. Every `tckgen` then gets
`-act 5tt.mif -backtrack -crop_at_gmwmi`. `--act off` restores the earlier
plan unchanged. The run's provenance records the ACT settings and each
bundle's seed/include/exclude ROI file names. A failure in any bundle reports
every failed bundle, not only the first.

The final capsule reuses `build-nifti`, with the T1+C volume, optional FLAIR,
and generated tracts, `--brain-mask synthstrip` (when the ACT stage ran, also
`--brain-mask-file` with its SynthStrip mask, so build-nifti gates and packs that
mask instead of running SynthStrip a second time), `--deface` (unless
`--no-deface`), `--tract-max-length-mm` equal to the tracking maximum length
(so tract trust can measure the cap fraction) and, with `--lesion`, the
DWI-space lesion as `--lesion-mask` (an unreviewed pipeline mask used for the
tract rim metric). The optional `--volume-registration LABEL:METHOD` records
the alignment method for an already aligned NIfTI volume. Tracts remain
`reviewed: false` and are labelled as unreviewed anatomical aids; they are not
diagnostic findings.

### `tracts` (manifest)
```json
"tracts": [{"id": "t01", "blob": "tract_t01", "blob_sha256": "<sha256>",
            "source_sha256": "<sha256>", "provenance": {"recorded": false},
            "label": "Trato corticoespinhal E",
            "color": "#E4572E", "format": "tck", "n_streamlines": 1500,
            "n_streamlines_source": 6971, "step_mm": 1.0, "n_points": 220626,
            "n_points_before_step": 342778, "source": "tractlab", "reviewed": false,
            "n_streamlines_outliers": 20, "outlier_filter": "...",
            "outlier_blob": "tract_outlier_t01", "outlier_blob_sha256": "<sha256>",
            "n_streamlines_outlier_blob": 4,
            "fraction_points_inside_grid": 1.0}]
```
The blob is an MRtrix `.tck` (Float32LE, gzip+base64 like volumes) with a
minimal header only (`mrtrix tracks`, `datatype`, `count`, `file`, `END`);
source header lines (which can carry paths/identifiers) are never copied.
Points are world RAS+ mm, the same space as `grid.affine_ras`; the pipeline
applies no transform, crop or resample to them, so tracts must already be in
the reference volume's world space. Streamlines are a deterministic uniform
subset (seeded per tract) when the source exceeds `--max-streamlines`.
`--tract-step-mm` (default 1.0, 0 = off, max 5) then keeps, along each
streamline, the first point, the first point past each further multiple of the
step in arc length, and the last point; points are selected, never
interpolated, so kept coordinates are bit-identical to the source.
`fraction_points_inside_grid` is informational. Tracts are unreviewed
automatic outputs (`reviewed: false`) until the surgeon signs them in surgeon
mode. The list shows the full filtered share and rule; "Mostrar fibras
filtradas" adds the sampled outlier blob in the tract colour. Patient mode has
no filter toggle or percentage.

### r16b viewer trust display

The viewer reads optional `manifest.tracts[].trust`. `FAIL` tracts start hidden;
their row appends "reprovado no QC de confiabilidade" and the reasons from
`failing`: `tortuosity_ratio` shows "tortuosidade {r}× a do lado oposto" with
one decimal and a comma, `median_tortuosity` shows "trajeto excessivamente
tortuoso", and `cap_fraction` shows "fibras no comprimento máximo". A surgeon
can show a failed tract; while any failed tract is visible the viewer shows the
"Feixe reprovado no QC" chip. A `PASS` tract with `rim_flag: true` shows
"passa junto à lesão" and remains visible by default. `UNAVAILABLE` shows
"QC indisponível"; legacy tracts without `trust` show "sem QC". In every mode
and verdict (tested in surgeon mode, the default), `yield_flag: true` appends "poucas fibras: {r}× menos que o lado
oposto (não prova ausência do trato)", with r = 1 / `yield_ratio` and one
decimal, joined by " · " on its own line under the count. When either tract of a
left/right pair carries a boolean `yield_flag`, the older surgeon-mode pair warning
("Assimetria de contagem E/D") is not shown; capsules without it keep that warning. These strings
contain no millimetric distances. If top-level `brain_mask_qc.verdict` is
`FAIL`, the viewer shows "Máscara cerebral reprovada no QC — render do cérebro
indisponível" beside the 3D preset control and does not offer the brain preset.
If no other safe 3D preset is available, the 3D pane is hidden.

## Demo data (r2)

All demo inputs live under the gitignored `data/public/` (a symlink to
`~/case-capsule/data`). Capsules built from them inherit the input licence.

**Thin head CT — CQ500, case CQ500-CT-401, series 3** (SeriesDescription "0.625mm",
SOFT kernel, 512×512×256, 0.529 mm pixels, 0.625 mm slices, 8° gantry tilt; reads.csv:
normal by all three readers).
- Source: qure.ai CQ500 (login-free S3). Licence: **Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International** (as stated on the Academic
  Torrents record, academictorrents.com/details/47e9d8aab761e75fd0a81982fa62bddf3a173831).
  **Non-commercial and share-alike: CT demo capsules may not be used commercially and must
  be shared under the same licence.**
- Citation: Chilamkurthy S, Ghosh R, Tanamala S, Biviji M, Campeau NG, Venugopal VK,
  Mahajan V, Rao P, Warier P. Development and validation of deep learning algorithms for
  detection of critical findings in head CT scans. arXiv:1803.05854.
  doi:10.48550/arXiv.1803.05854.
- Download (70.2 MB zip, sha256
  `c4d810ddf2a453d6141ee28fd68fcac53d0756f8c1a147ec74a764cbac54944a`; 129 MB written):
  `uv run --with pylibjpeg --with pylibjpeg-openjpeg python scripts/fetch_cq500_thin.py --case CQ500-CT-401 --series 3 --out data/public/CQ500/head-ct-thin`.
  The script rewrites the JPEG 2000 lossless pixels as Explicit VR Little Endian (the
  bundled dcm2niix cannot decode J2K) and round-trip checks every pixel array.
- Gantry tilt: dcm2niix writes the sheared stack and a resampled `_Tilt_1` volume; ingest
  uses `_Tilt_1`. `tests/test_tilt_real.py` checks it against raw DICOM pixels at their
  IPP/IOP positions (896 samples: median |ΔHU| 0.0, p90 3.0; uncorrected-stack control p90 462.5).

Search log (≤ 1.25 mm, whole skull, login-free, open licence), 2026-09-27: NBIA anonymous
v1 API — CPTAC-AML "Head WO 5mm" (5 mm) and "COR Head WO" (2 mm coronal); CMB-MEL
"Bone 2.0 Axial" (2 mm), "AXIAL WO" (3 mm); EAY131 (3.75 mm); several "HEAD" body-part
series were chest CT; HNSCC/CPTAC-GBM not listed anonymously. IDC (idc-index v24): no CT in
cptac_gbm, hnscc, tcga_gbm/lgg/hnsc. NLM Visible Human 1 mm CT rejected (NLM terms, not an
open licence). The TCIA guest-token route was not tried. CQ500 is the only thin option
found; the CPTAC-AML 5 mm CT (CC BY 4.0) remains available as `scripts/demo_build.sh cptac`.

**Vestibular schwannoma MR + RTSTRUCT — TCIA Vestibular-Schwannoma-SEG** (one subject;
series 2 T1 contrast 3D GR, series 3 T2 3D SE). Licence: Creative Commons Attribution 4.0
International (NBIA series metadata). Data citation: Shapey J, Kujawa A, Dorent R, et al.
Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated
Dataset and Baseline Algorithm (version 2) [Data set]. The Cancer Imaging Archive; 2021.
doi:10.7937/TCIA.9YTJ-5Q73. Publication: Scientific Data 2021;8(1).
doi:10.1038/s41597-021-01064-w.
- RTSTRUCT download (subject matched in code, no identifier printed):
  `uv run python scripts/fetch_vs_rtstruct.py` → `data/public/Vestibular-Schwannoma-SEG/rtstruct/`
  `rtstruct_T1.dcm` (sha256 `95618b8912caf9687e266640a08c7876bb62fea4e015fa0ad4f401d3684dc103`)
  and `rtstruct_T2.dcm` (sha256 `930f72136e141d727e974c35a2ec90e3873eb338bd780be0eb64e26cb874669a`).
  ROIs: `AN` (tumour), `cochlea`, `*Skull`. No per-subject volume or laterality is published
  (the TCIA contours zip holds only LPS contour points), so the tumour volume is checked
  against the planar contour volume.
