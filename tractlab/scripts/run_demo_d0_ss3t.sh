#!/usr/bin/env bash
# House SS3T FOD path for D0 Leipzig sub-010005.
# denoise → degibbs → dwifslpreproc (SE AP/PA) → N4 → 1.3 mm upsample →
# dhollander → ss3t_csd_beta1 → mtnormalise → b0/mask nifti.
# Does not switch run.py. Does not build banks.
set -euo pipefail

NTHREADS="${NTHREADS:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/cases/demo-leipzig-sub-010005"
RAW="$ROOT/raw/sub-010005/ses-01"
WORK="$ROOT/work/ss3t"
NIFTI="$ROOT/nifti"
LOG="$WORK/ss3t.log"

export FSLDIR="${FSLDIR:-$HOME/fsl}"
export ANTSPATH="${ANTSPATH:-$HOME/ants-2.6.5/bin}"
export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"

mkdir -p "$WORK" "$NIFTI" "$WORK/eddyqc"
# Progress bars must not hit the log (first run wrote 2.4 GB of CR updates).
exec >>"$LOG" 2>&1
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) nthreads=$NTHREADS ====="
command -v ss3t_csd_beta1 >/dev/null
command -v eddy_cpu >/dev/null
command -v N4BiasFieldCorrection >/dev/null

run() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $(date -u +%H:%M:%SZ) $*  -> $out"
  "$@" -quiet
}

# 0. import
run "$WORK/dwi_raw.mif" mrconvert \
  "$RAW/dwi/sub-010005_ses-01_dwi.nii.gz" "$WORK/dwi_raw.mif" \
  -fslgrad "$RAW/dwi/sub-010005_ses-01_dwi.bvec" "$RAW/dwi/sub-010005_ses-01_dwi.bval" \
  -json_import "$RAW/dwi/sub-010005_ses-01_dwi.json" \
  -nthreads "$NTHREADS"

run "$WORK/fmap_ap.mif" mrconvert \
  "$RAW/fmap/sub-010005_ses-01_acq-SEfmapDWI_dir-AP_epi.nii.gz" "$WORK/fmap_ap.mif" \
  -json_import "$RAW/fmap/sub-010005_ses-01_acq-SEfmapDWI_dir-AP_epi.json" \
  -nthreads "$NTHREADS"

run "$WORK/fmap_pa.mif" mrconvert \
  "$RAW/fmap/sub-010005_ses-01_acq-SEfmapDWI_dir-PA_epi.nii.gz" "$WORK/fmap_pa.mif" \
  -json_import "$RAW/fmap/sub-010005_ses-01_acq-SEfmapDWI_dir-PA_epi.json" \
  -nthreads "$NTHREADS"

run "$WORK/se_pair.mif" mrcat "$WORK/fmap_ap.mif" "$WORK/fmap_pa.mif" "$WORK/se_pair.mif" -axis 3

# 1. denoise (first, on rawest data)
run "$WORK/dwi_den.mif" dwidenoise "$WORK/dwi_raw.mif" "$WORK/dwi_den.mif" \
  -noise "$WORK/noise.mif" -nthreads "$NTHREADS"
if [[ ! -e "$WORK/residual.mif" ]]; then
  mrcalc "$WORK/dwi_raw.mif" "$WORK/dwi_den.mif" -subtract "$WORK/residual.mif"
fi

# 2. degibbs — PartialFourier 7/8 (less ideal; house path still runs it)
run "$WORK/dwi_den_deg.mif" mrdegibbs "$WORK/dwi_den.mif" "$WORK/dwi_den_deg.mif" \
  -nthreads "$NTHREADS"

# 3. topup + eddy. DWI and AP fmap are BIDS j-; pair is AP then PA.
#    -eddy_options leading space is required by MRtrix.
run "$WORK/dwi_preproc.mif" dwifslpreproc "$WORK/dwi_den_deg.mif" "$WORK/dwi_preproc.mif" \
  -rpe_pair -se_epi "$WORK/se_pair.mif" -pe_dir ap -readout_time 0.04914 -align_seepi \
  -eddy_options " --slm=linear --repol --data_is_shelled" \
  -eddyqc_all "$WORK/eddyqc" \
  -nthreads "$NTHREADS"

# 4. N4
run "$WORK/dwi_bias.mif" dwibiascorrect ants "$WORK/dwi_preproc.mif" "$WORK/dwi_bias.mif" \
  -nthreads "$NTHREADS"

# 5. upsample to house 1.3 mm iso
run "$WORK/dwi_up.mif" mrgrid "$WORK/dwi_bias.mif" regrid "$WORK/dwi_up.mif" \
  -voxel 1.3 -nthreads "$NTHREADS"

# 6. mask + responses + SS3T
run "$WORK/mask_up.mif" dwi2mask "$WORK/dwi_up.mif" "$WORK/mask_up.mif" -nthreads "$NTHREADS"
run "$WORK/wm.txt" dwi2response dhollander "$WORK/dwi_up.mif" \
  "$WORK/wm.txt" "$WORK/gm.txt" "$WORK/csf.txt" \
  -mask "$WORK/mask_up.mif" -voxels "$WORK/sf_voxels.mif" -nthreads "$NTHREADS"

run "$WORK/wmfod.mif" ss3t_csd_beta1 "$WORK/dwi_up.mif" \
  "$WORK/wm.txt" "$WORK/wmfod.mif" \
  "$WORK/gm.txt" "$WORK/gm.mif" \
  "$WORK/csf.txt" "$WORK/csf.mif" \
  -mask "$WORK/mask_up.mif" -nthreads "$NTHREADS"

run "$WORK/wmfod_norm.mif" mtnormalise \
  "$WORK/wmfod.mif" "$WORK/wmfod_norm.mif" \
  "$WORK/gm.mif" "$WORK/gm_norm.mif" \
  "$WORK/csf.mif" "$WORK/csf_norm.mif" \
  -mask "$WORK/mask_up.mif" -nthreads "$NTHREADS"

# 7. serve-facing copies (gitignored nifti/)
if [[ ! -e "$NIFTI/wmfod_norm.mif" ]]; then
  cp "$WORK/wmfod_norm.mif" "$NIFTI/wmfod_norm.mif"
fi
run "$NIFTI/mask_up.nii.gz" mrconvert "$WORK/mask_up.mif" "$NIFTI/mask_up.nii.gz" -nthreads "$NTHREADS"
if [[ ! -e "$NIFTI/b0.nii.gz" ]]; then
  dwiextract "$WORK/dwi_up.mif" - -bzero | mrmath - mean - -axis 3 | \
    mrconvert - "$NIFTI/b0.nii.gz" -nthreads "$NTHREADS"
fi

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
mrinfo "$NIFTI/wmfod_norm.mif" -size -spacing
mrinfo "$NIFTI/b0.nii.gz" -size -spacing
