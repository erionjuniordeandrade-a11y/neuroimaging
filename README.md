# Neuroimaging

Open-source neurosurgical imaging tools for research and teaching:
CT/MR fusion, CTA vessels, DWI tractography, and offline "case capsules"
that a surgeon can review and explain to a patient.

> **Not a medical device. Not for clinical use, diagnosis, treatment planning
> or surgical navigation.** Research and teaching use only. No regulatory
> clearance of any kind is claimed. Outputs have not been clinically validated.

| Folder | What it does |
|---|---|
| `packages/neuro-core/` | Shared Python core: image grids and units, affines, hashing. |
| `capsule/` | Builds one offline HTML "case capsule" per case: CT/MR fusion, CTA vessels, DWI tractography, trust and QC gates, pt-BR patient tour. |
| `tractlab/` | Loopback tractography workstation: ROI seeding, bundle banks, atlas, lesion-centred viewer, ingest and pipeline runner. |
| `capsule-mac/` | Native macOS host app that finds and opens capsule files. |
| `apps/shell/` | Neuro Workbench: one local window with a case list and one shared CT / MRI / CTA / tracts viewer. `uv run neuro-workbench`, then open http://127.0.0.1:8790/ (see `apps/shell/README.md`). |

## Install and test

The Python packages form one [uv](https://docs.astral.sh/uv/) workspace (Python 3.12).

```bash
uv sync --all-packages
uv run --package neuro-core pytest packages/neuro-core
cd capsule && uv run pytest
uv run --package neuro-workbench pytest apps/shell/tests
cd tractlab && ./verify.sh
cd capsule-mac && xcrun swift test
```

Tests that need a local case folder skip when that folder is absent.
Some pipelines call external tools (MRtrix3, FSL, dcm2niix, ANTs). They are
not bundled; install them yourself under their own licences.

## Data

No patient data is in this repository, and none may be added. Tools read
imaging by path from folders outside the tree. The `.gitignore` files and the
`tractlab/hooks/pre-commit` hook deny imaging files. For a public demo case,
use OpenNeuro ds000221 (CC0).

The reference-atlas meshes (`tractlab/viewer/atlas/*.glb`, `*.bin`) are not
shipped, because their upstream data terms (HCP Open Access, CC BY-SA 4.0,
Melbourne Subcortex, MNI) need separate acceptance. See
`tractlab/THIRD_PARTY_NOTICES.md` and regenerate them with
`tractlab/scripts/import-reference-atlas.py` from the upstream sources.

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`. Third-party components keep
their own licences (`capsule/vendor/NIIVUE_LICENSE.txt`,
`tractlab/THIRD_PARTY_NOTICES.md`).
