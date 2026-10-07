# CR peel

CR peel reformats one unstripped post-contrast T1 onto a shared stack of inward offset surfaces. The grey channel is the single-sample reformat; a separately generated overlay marks thin, bright, in-brain candidates as enhancing vessels. This is a research view for inspecting surface-correlated enhancement, not a localization or export workflow.

> Research visualization only · enhancing vessels · not navigation, not a device

There is no DICOM, PACS, Stealth, Brainlab, registration, or risk-score export.

## Input manifest

The CLI accepts only a case root containing `manifest.json`. It refuses unless top-level `deid` is exactly `true` and this object is present:

```json
{
  "deid": true,
  "case_id": "deidentified-case-slug",
  "inputs": {
    "t1c_unstripped": {
      "root": "/absolute/input/directory",
      "path": "t1c.nii.gz",
      "brain_mask": "t1c_mask.nii.gz",
      "series": "3D post-contrast series",
      "contrast": "gadolinium",
      "provenance": {
        "scanner": "source description",
        "resampled": "conformation description",
        "mask": "mask method"
      }
    }
  }
}
```

`root` may begin with `~`; after expansion it must be an absolute existing directory. `path` and `brain_mask` must be relative paths that remain beneath `root`, including after symlink resolution. The two NIfTI files must have exactly equal shapes and affines. Both are reoriented with `nibabel.as_closest_canonical`; voxel sizes must then be isotropic within 2%. The CLI never resamples silently.

## CLI

```bash
python3 -m tractlab.cr <case_root> \
  [--depths 0:30:1] \
  [--vessel-pct 97.5] \
  [--tophat-frac 0.5] \
  [--head-thr 200] \
  [--direction-sigma 8] \
  [--dry-run]
```

In this source-layout checkout, the established repository interpreter invocation is:

```bash
PYTHONPATH=src ~/fsl/bin/python3 -m tractlab.cr cases/local-case-2 --dry-run
```

`START:STOP:STEP` is inclusive when `STOP` lies on the step sequence. Before loading either image, the CLI hashes both input files. It then requires exact T1/mask grid equality, RAS canonicalisation without resampling, `mm` spatial units, and a Euclidean isotropic canonical grid: for `A = affine[:3, :3]`, every element of `A.T @ A` must lie within 2% of `vx² I`. A refusal prints the measured Gram matrix. Output depth tags are collision-checked and symlinked `cr/` paths are refused before computation. `--dry-run` performs these checks plus brain-mask and head QC, prints the full parameters, and writes nothing.

Head QC prints one actual superior intensity column at the rounded brain-mask centroid `(x, y)`, starting at the top brain-mask voxel in that column and continuing upward for at most 40 voxels. It refuses if the rounded centroid column does not intersect the mask. It also prints the pre-selection component count, head fraction of the FOV, a suggested threshold equal to 0.25 × the volume p99, and the number of brain voxels outside `envelope(head_mask, vx, env_sigma_mm)`. It refuses any nonzero outside-envelope count and keeps the head-fraction range at 10%–95%.

## Outputs

The CLI prevalidates every output name, then stages the complete binary bundle under `<case_root>/cr/.stage-<pid>/`. Only after all files are staged and both input hashes are rechecked does it remove the older bundle and move the new files into `cr/` with `os.replace`; `summary.json` is moved last. A failed run removes its stage directory and restores the prior `cr/` contents.

- `peel.npz`: `faces`, world-space float32 `verts_world` with shape `(depth, vertex, 3)`, `grey`, `vmax`, `vessel`, `in_brain`, `valid`, and `depths_mm`.
- `d{DD}_grey.ply`: binary little-endian PLY with direct uchar RGB from the per-depth p2–p99.3 grey window over in-brain vertices. Before a depth reaches the brain mask, the finite surface samples provide the context-window fallback.
- `d{DD}_vmax.ply`: the same geometry with the identical p2–p99.3/fallback window applied to the per-vertex slab maximum.
- `d{DD}_vessel.ply`: the same grey surface, with selected enhancing-vessel vertices set to RGB `[140, 18, 30]`.
- `summary.json`: the tracked derivation receipt with schema `tractlab.cr.summary/1`.

The receipt binds the pre-load SHA-256 of both input files, original shape and affine, and canonical processing grid, and records `verified_before_publish: true` only after the second hash check passes. It stores the manifest root string verbatim as `root_literal` and only a SHA-256 of the resolved absolute root; a free-text `case_id` is omitted unless it fully matches `[a-z0-9-]+`. Before publication, a recursive self-scan refuses any string value containing `/Users/`, `/home/`, or the current username. The receipt also records relative manifest paths, series/contrast/provenance, voxel size, Git commit and dirty flag, SHA-256 of `git diff HEAD` whenever dirty, every `PeelParams` value, envelope containment, marker and threshold results, mesh counts, per-depth residual/fold counts, an ISO timestamp, the honesty wall, and the SHA-256 of every derived binary output. A receipt cannot contain its own stable digest; Git tracks `summary.json`, while its `outputs` map binds `peel.npz` and every PLY.

## Slicer loader

Run `slicer_cr.py` with the case root as the script argument. It loads the T1 and both mesh variants for every receipt depth, applies the PLY `RGB` point scalar with direct mapping, watermarks every current slice and 3-D view, and opens at the marker depth. The console exposes `show_depth(d)`, `set_tint(on)`, `screenshot(path)`, and `contact_sheet(depths)`.

## Hair trap

The default threshold of 200 is a measured operating point for the reference 1.5 T SPGR, not a portable MR intensity standard. A threshold near 40 included hair and cushion signal and shifted every peel depth by about 7 mm. On every new scanner or sequence, read the printed scalp-column profile and obtain owner review before trusting the marker. The suggested threshold is context, not automatic approval.

## How to read `folded_faces`

For each depth, `folded_faces` counts shared-topology faces whose orientation opposes the smoothed inward depth field, collapses, or whose edge stretches beyond `max_edge_stretch` relative to depth zero. `folded_faces_convexity` is the same proxy restricted to the upper-convexity view. Zero means those proxies found no fold or bridge; it does not prove the absence of every triangle–triangle self-intersection. A nonzero count identifies a depth requiring geometric review, especially when the convexity count is nonzero. Read these counts beside `depth_residual_mm`: a low residual proves arrival at the requested level set, not valid embedding.

## Known gaps (not this slice)

Slicer panel UI · web/three.js viewer · artery/vein separation · MRV/SWI fusion · pial-surface sampling · unrolled 2-D CR view · nav registration/export (closed door, permanent) · intra-op photo validation (research question).
