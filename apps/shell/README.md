# Eidos

Eidos (Greek εἶδος, "form, what is seen") is one local window for the neuroimaging apps. Research and teaching only, not for clinical use.

```bash
uv sync --all-packages
uv run eidos                      # http://127.0.0.1:8790/  (alias: neuro-workbench)
```

The window has a case list and four tabs per case:

| Tab | What it shows | Served by |
|---|---|---|
| Case · CT / MRI / CTA / tracts | One shared viewer: every volume, mask and tract of the case in one MPR + 3D scene | the workbench (NiiVue), from files decoded once into the cache |
| Capsule viewer | Capsule viewer2: CT/MR fusion, CTA, masks, tracts, tour | the capsule file, through the workbench |
| Tract evidence | TractLab workstation | a TractLab child process on a free loopback port |
| Atlas | TractLab reference atlas | the same TractLab child |

## Cases

- **Demo case** (always listed): a synthetic phantom capsule plus generated tract curves.
  They are not one subject and show no anatomy. The first run builds the phantom with
  `capsule/viewer2/dev_fixture.py` into `~/Library/Caches/neuroimaging-workbench/`
  (about 45 minutes on this Mac); later runs reuse it. `--phantom FILE` uses a prebuilt phantom instead.
- `--capsule-dir DIR` (repeatable) lists every `*.capsule.html` in that folder, imaging only.
- `--manifest FILE` (repeatable) lists one TractLab case, tract evidence only.

The workbench never scans a default patient folder. A case appears only when you name its source.

## Case tab

The Case tab loads all layers into one NiiVue view in scanner RAS millimetres:

- base volume (CT first) with window presets: brain, soft tissue, bone, vessels;
- one fusion volume with opacity and colour map;
- structure masks, flagged "unreviewed" until a reviewer signs them;
- tracts with streamline counts, a 2D slab control and a 3D clip plane;
- layouts 2×2, axial, coronal, sagittal and 3D.

A capsule is decoded once into `<cache>/scenes/capsule-<hash>/`. A TractLab manifest
is served in place, by layer key only; files outside the scene list return 404.

## Limits

- The 3D view is a volume render, not a segmented surface render.
- Capsule and TractLab sources are not linked per case yet. When a scene joins two
  sources, the viewer shows a red banner: alignment is assumed, not registered.
- An opaque fusion layer can hide tracts in 3D. Lower its opacity.

## Tests

```bash
uv run pytest apps/shell/tests -q
```
