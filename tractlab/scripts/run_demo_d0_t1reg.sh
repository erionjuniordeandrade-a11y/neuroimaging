#!/usr/bin/env bash
# House T1→DWI rigid path for D0 Leipzig sub-010005.
# Same mechanism as tractlab-data/code/run_tractography.sh (3T T1→DWI):
#   bet T1 (-f 0.4 -B) → mask b0 with dwi2mask (not bet) →
#   flirt 6-dof -cost normmi (b0_brain → t1_brain) → invert →
#   resample T1-brain onto the upsampled b0/FOD grid.
# Serve-facing name follows the 3T pattern (*_brain_dwi.nii.gz); this case
# is MP2RAGE T1w, not post-gad, so the file is t1_brain_dwi.nii.gz.
# Does not switch run.py. Does not build banks. Does not approve QC.
set -euo pipefail

NTHREADS="${NTHREADS:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/cases/demo-leipzig-sub-010005"
RAW_T1="$ROOT/raw/sub-010005/ses-01/anat/sub-010005_ses-01_acq-mp2rage_T1w.nii.gz"
WORK="$ROOT/work/t1reg"
NIFTI="$ROOT/nifti"
LOG="$WORK/t1reg.log"

export FSLDIR="${FSLDIR:-$HOME/fsl}"
export ANTSPATH="${ANTSPATH:-$HOME/ants-2.6.5/bin}"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"

mkdir -p "$WORK"
# Progress bars must not hit the log (SS3T first run wrote 2.4 GB of CR updates).
exec >>"$LOG" 2>&1
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) nthreads=$NTHREADS ====="
command -v flirt >/dev/null
command -v bet >/dev/null
command -v mrconvert >/dev/null
[[ -f "$NIFTI/b0.nii.gz" ]]
[[ -f "$NIFTI/mask_up.nii.gz" ]]
[[ -f "$RAW_T1" ]]

run_mr() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $(date -u +%H:%M:%SZ) $*  -> $out"
  "$@" -quiet
}

run_fsl() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $(date -u +%H:%M:%SZ) $*  -> $out"
  "$@"
}

# 0. import T1 (MP2RAGE UNI T1w)
run_mr "$WORK/t1.nii.gz" mrconvert "$RAW_T1" "$WORK/t1.nii.gz" -nthreads "$NTHREADS"

# 1. BET T1 (house: -f 0.4 -B). Not bet on b0.
run_fsl "$WORK/t1_brain.nii.gz" bet "$WORK/t1.nii.gz" "$WORK/t1_brain.nii.gz" -f 0.4 -B

# 2. mask b0 with the existing DWI mask (house: fslmaths -mas, not bet)
run_fsl "$WORK/b0_brain.nii.gz" fslmaths "$NIFTI/b0.nii.gz" -mas "$NIFTI/mask_up.nii.gz" "$WORK/b0_brain.nii.gz"

# 3. rigid b0_brain → t1_brain (6 DOF, normmi)
run_fsl "$WORK/b02t1.mat" flirt \
  -in "$WORK/b0_brain.nii.gz" -ref "$WORK/t1_brain.nii.gz" \
  -omat "$WORK/b02t1.mat" -dof 6 -cost normmi
if [[ ! -e "$WORK/t12b0.mat" ]]; then
  echo "RUN $(date -u +%H:%M:%SZ) convert_xfm -inverse -> $WORK/t12b0.mat"
  convert_xfm -omat "$WORK/t12b0.mat" -inverse "$WORK/b02t1.mat"
fi

# 4. apply inverse onto the b0/FOD grid (house: flirt -applyxfm -ref b0).
#    Copies the reference header so T1 shares b0 affine/strides (L/A/S).
#    mrtransform -template alone can flip i-stride (R/A/S) and fail A0.
run_fsl "$WORK/t1_brain_dwi.nii.gz" flirt \
  -in "$WORK/t1_brain.nii.gz" -ref "$NIFTI/b0.nii.gz" \
  -applyxfm -init "$WORK/t12b0.mat" \
  -out "$WORK/t1_brain_dwi.nii.gz" -interp spline

# 5. serve-facing copies (gitignored nifti/) — 3T pattern, Leipzig names
if [[ ! -e "$NIFTI/t1_brain_dwi.nii.gz" ]]; then
  cp "$WORK/t1_brain_dwi.nii.gz" "$NIFTI/t1_brain_dwi.nii.gz"
fi
if [[ ! -e "$NIFTI/b02t1.mat" ]]; then
  cp "$WORK/b02t1.mat" "$NIFTI/b02t1.mat"
fi
if [[ ! -e "$NIFTI/t12b0.mat" ]]; then
  cp "$WORK/t12b0.mat" "$NIFTI/t12b0.mat"
fi

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "-- t1_brain_dwi --"
mrinfo "$NIFTI/t1_brain_dwi.nii.gz" -size -spacing
echo "-- b0 --"
mrinfo "$NIFTI/b0.nii.gz" -size -spacing
echo "-- wmfod_norm --"
mrinfo "$NIFTI/wmfod_norm.mif" -size -spacing
