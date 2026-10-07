#!/usr/bin/env bash
# Re-fetch the D0 Leipzig subject (DWI + MP2RAGE T1w + DWI SE fmap only).
# Does not fetch func / T2 / FLAIR. Requires aws CLI + network.
set -euo pipefail
AWS="${AWS:-/opt/homebrew/bin/aws}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/cases/demo-leipzig-sub-010005/raw"
SUB=sub-010005
SES=ses-01
BUCKET=s3://openneuro.org/ds000221

mkdir -p "$ROOT/$SUB/$SES/dwi" "$ROOT/$SUB/$SES/anat" "$ROOT/$SUB/$SES/fmap"
cp_s3() { "$AWS" s3 cp "$1" "$2" --no-sign-request; }

cp_s3 "$BUCKET/dataset_description.json" "$ROOT/dataset_description.json"
cp_s3 "$BUCKET/participants.tsv" "$ROOT/participants.tsv"
for f in ${SUB}_${SES}_dwi.bval ${SUB}_${SES}_dwi.bvec ${SUB}_${SES}_dwi.json ${SUB}_${SES}_dwi.nii.gz; do
  cp_s3 "$BUCKET/$SUB/$SES/dwi/$f" "$ROOT/$SUB/$SES/dwi/$f"
done
for f in ${SUB}_${SES}_acq-mp2rage_T1w.nii.gz ${SUB}_${SES}_acq-mp2rage_defacemask.nii.gz; do
  cp_s3 "$BUCKET/$SUB/$SES/anat/$f" "$ROOT/$SUB/$SES/anat/$f"
done
for f in \
  ${SUB}_${SES}_acq-SEfmapDWI_dir-AP_epi.json \
  ${SUB}_${SES}_acq-SEfmapDWI_dir-AP_epi.nii.gz \
  ${SUB}_${SES}_acq-SEfmapDWI_dir-PA_epi.json \
  ${SUB}_${SES}_acq-SEfmapDWI_dir-PA_epi.nii.gz
do
  cp_s3 "$BUCKET/$SUB/$SES/fmap/$f" "$ROOT/$SUB/$SES/fmap/$f"
done
echo "fetched into $ROOT"
