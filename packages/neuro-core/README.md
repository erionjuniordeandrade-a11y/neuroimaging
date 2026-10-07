# neuro-core

Shared code for capsule and tractlab. It holds one implementation each of:

- `neuro_core.grid`: the nibabel voxel/world contract and the LPS to RAS affine.
- `neuro_core.paths`: manifest path containment.
- `neuro_core.hashing`: chunked file sha256.
- `neuro_core.tck`: a minimal MRtrix `.tck` codec that never copies source headers.

The apps re-export these names from their old modules, so existing imports keep working.
Ingest, de-identification and case manifests stay app-specific for now.
