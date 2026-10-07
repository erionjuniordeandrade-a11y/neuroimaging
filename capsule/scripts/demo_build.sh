#!/usr/bin/env bash
# Build the public demo capsules into out/ (never overwrites: every file gets a timestamp suffix).
# Usage: scripts/demo_build.sh [a|b|c|cptac|all]   (DICOM demos use v2 automatic anatomy)
# Inputs are read-only public/CC0 data; nothing is written outside out/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="${CASE_CAPSULE_PUBLIC:-$HOME/case-capsule/data/public}"
LEIPZIG="${TRACTLAB_DEMO:-$HOME/tractlab/cases/demo-leipzig-sub-010005}"
VIEWER="${VIEWER:-v2}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/out"
WHICH="${1:-all}"

if [[ "$VIEWER" == "v2" && ! -f "$ROOT/viewer2/template.html" ]]; then
  echo "demo_build: viewer2/template.html is missing; build the NiiVue viewer first (automatic anatomy requires v2)" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$ROOT"

# a: thin CT, CQ500 (CC BY-NC-SA 4.0: non-commercial, share-alike; see SPEC "Demo data")
build_a() {
  uv run capsule build "$PUBLIC/CQ500/head-ct-thin" --series 3 --label "CQ500 TC crânio fina (CC BY-NC-SA)" \
    --viewer "$VIEWER" --anatomy auto -o "$OUT/cq500-head-ct-thin.$VIEWER.$STAMP.capsule.html"
}

build_cptac() {
  uv run capsule build "$PUBLIC/CPTAC-AML/head-ct" --series 2 --label "CPTAC-AML TC crânio" \
    --viewer "$VIEWER" -o "$OUT/cptac-aml-head-ct.$VIEWER.$STAMP.capsule.html"
}

build_b() {
  uv run capsule build "$PUBLIC/Vestibular-Schwannoma-SEG/brain-mri" --series 2,3 \
    --label "Schwannoma vestibular RM" --viewer "$VIEWER" --anatomy auto \
    --rtstruct "$PUBLIC/Vestibular-Schwannoma-SEG/rtstruct/rtstruct_T1.dcm" \
    -o "$OUT/vs-seg-brain-mri.$VIEWER.$STAMP.capsule.html"
}

build_c() {
  local bank="$LEIPZIG/tracts/bank"
  uv run capsule build-nifti \
    --volume "$LEIPZIG/nifti/t1_brain_dwi.nii.gz:MR:RM T1" \
    --tract "$bank/cst_l_motor_pons.tck:Trato corticoespinhal E:#E4572E" \
    --tract "$bank/cst_r_motor_pons.tck:Trato corticoespinhal D:#F2A07B" \
    --tract "$bank/ifof_l_occ_front.tck:FOFI E:#4C9BE8" \
    --tract "$bank/ifof_r_occ_front.tck:FOFI D:#9CC8F3" \
    --tract "$bank/or_l_meyer.tck:Radiação óptica (alça de Meyer) E:#59C36A" \
    --tract "$bank/or_r_meyer.tck:Radiação óptica (alça de Meyer) D:#A5E0AE" \
    --tract "$bank/uf_l_temp_orb.tck:Fascículo uncinado E:#C9A84C" \
    --tract "$bank/uf_r_temp_orb.tck:Fascículo uncinado D:#E6D39E" \
    --tract "$bank/slf2_l_mfg_ipl.tck:FLS II E:#B06AD9" \
    --tract "$bank/slf2_r_mfg_ipl.tck:FLS II D:#D6AEEC" \
    --tract "$bank/cing_l_ant_post.tck:Cíngulo E:#3CC8C8" \
    --tract "$bank/cing_r_ant_post.tck:Cíngulo D:#97E3E3" \
    --max-streamlines "${MAX_STREAMLINES:-800}" --label "Leipzig demo (CC0) — tratos" --viewer "$VIEWER" \
    -o "$OUT/leipzig-t1-tracts.$VIEWER.$STAMP.capsule.html"
}

case "$WHICH" in
  a) build_a ;;
  b) build_b ;;
  c) build_c ;;
  cptac) build_cptac ;;
  all) build_a && build_b && build_c ;;
  *) echo "usage: $0 [a|b|c|all]" >&2; exit 2 ;;
esac
