# Eidos

Eidos (Greek εἶδος, "form, what is seen") is one local window for the neuroimaging apps. Research and teaching only, not for clinical use.

```bash
uv sync --all-packages
uv run eidos                      # http://127.0.0.1:8790/  (alias: neuro-workbench)
```

The window opens on **Patients**, the local patient archive, and has four tabs per case:

| Tab | What it shows | Served by |
|---|---|---|
| Case · CT / MRI / CTA / tracts | One shared viewer: every volume, mask and tract of the case in one MPR + 3D scene | the workbench (NiiVue), from files decoded once into the cache |
| Capsule viewer | Capsule viewer2: CT/MR fusion, CTA, masks, tracts, tour | the capsule file, through the workbench |
| Tract evidence | TractLab workstation | a TractLab child process on a free loopback port |
| Atlas | TractLab reference atlas | the same TractLab child |

## Patients (archive)

Eidos keeps a local archive of patient studies, in the way a PACS viewer does:

- **Import folder or CD** or **Import .zip** copies every DICOM image into the archive and
  skips files it already has. `uv run eidos --import PATH` does the same from the shell and prints counts.
- The Patients screen lists each patient with their studies and series. Search matches name,
  ID, birth date, study date, accession, modality and series description. A click opens the study in the Case tab.
- Series with fewer than 3 images (scouts, reports) are listed but not drawn.

The archive is **not de-identified**. It holds patient names and IDs, so it stays on the
computer that imported it:

- location `~/Library/Application Support/Eidos/archive` (`--archive-dir` or `EIDOS_ARCHIVE` to change, `--no-archive` to turn off);
- folder mode 0700, files 0600, file names are hashes with no identifiers;
- disk encryption comes from FileVault. Check it with `fdesetup status`;
- the server logs nothing and import reports carry counts only.

Never put the archive folder in a repository, a cloud folder or a shared drive.

## Cases

- **Demo case** (always listed): a synthetic phantom capsule plus generated tract curves.
  They are not one subject and show no anatomy. The first run builds the phantom with
  `capsule/viewer2/dev_fixture.py` into `~/Library/Caches/neuroimaging-workbench/`
  (about 45 minutes on this Mac); later runs reuse it. `--phantom FILE` uses a prebuilt phantom instead.
- `--capsule-dir DIR` (repeatable) lists every `*.capsule.html` in that folder, imaging only.
- `--manifest FILE` (repeatable) lists one TractLab case, tract evidence only.

Eidos never scans a patient folder by itself. A study appears only after you import it; a capsule or manifest only when you name it.

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
