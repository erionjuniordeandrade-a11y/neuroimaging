#!/usr/bin/env bash
# Generic, resumable pipeline runner for one ingested TractLab case.
set -Eeuo pipefail

usage() {
  echo "Usage: run_case.sh <case_root>" >&2
}

if [[ $# -ne 1 || ! -d "$1" ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$1" && pwd -P)"
ROOT_REAL="$ROOT"
# MRtrix 3.0.7 Python scripts (dwibiascorrect, dwi2response -voxels, 5ttgen) shell-quote paths
# they then pass without a shell, and FreeSurfer breaks on spaces too, so a case root with
# whitespace (the default ~/Library/Application Support root) is used through a space-free alias.
# runner_state.py resolves symlinks, so case.json and manifest.json keep the real path.
if [[ "$ROOT" == *[[:space:]]* ]]; then
  ALIAS_DIR="${TMPDIR:-/tmp}"
  if [[ "$ALIAS_DIR" == *[[:space:]]* ]]; then
    ALIAS_DIR="/tmp"
  fi
  ROOT_ALIAS="${ALIAS_DIR%/}/tractlab-root-$(printf '%s' "${ROOT_REAL##*/}" | tr -c 'A-Za-z0-9._-' '_')"
  if [[ -e "$ROOT_ALIAS" && ! -L "$ROOT_ALIAS" ]]; then
    printf 'refusing to replace non-symlink case alias: %s\n' "$ROOT_ALIAS" >&2
    exit 2
  fi
  ln -sfn "$ROOT_REAL" "$ROOT_ALIAS"
  ROOT="$ROOT_ALIAS"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON="${TRACTLAB_PYTHON:-$HOME/fsl/bin/python}"
STATE_HELPER="$REPO/scripts/pipeline/runner_state.py"
WORK="$ROOT/work"
NIFTI="$ROOT/nifti"
MANIFEST="$ROOT/manifest.json"
NTHREADS="${NTHREADS:-8}"
SE_CAP="${TRACTLAB_SE_CAP:-3}"
# FreeSurfer stream for banks: "standard" recon-all, or "clinical" recon-all-clinical (FreeSurfer >= 7.4).
FS_RECON="${TRACTLAB_FS_RECON:-standard}"
if [[ "$FS_RECON" != "standard" && "$FS_RECON" != "clinical" ]]; then
  printf 'TRACTLAB_FS_RECON must be standard or clinical, got: %s\n' "$FS_RECON" >&2
  exit 2
fi
DRYRUN="${TRACTLAB_DRYRUN:-0}"
CURRENT_STAGE=""

FSLDIR="${FSLDIR:-$HOME/fsl}"
ANTSPATH="${ANTSPATH:-$HOME/ants-2.6.5/bin}"
FREESURFER_HOME="${FREESURFER_HOME:-$HOME/freesurfer}"
MRTRIX3="$HOME/mrtrix3/bin"
FS_LUT="$FREESURFER_HOME/FreeSurferColorLUT.txt"
MNI_BRAIN="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"

export FSLDIR ANTSPATH FREESURFER_HOME FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$MRTRIX3:$FREESURFER_HOME/bin:$PATH"

mkdir -p "$WORK"
"$PYTHON" "$STATE_HELPER" status init "$ROOT"

on_error() {
  local exit_code=$?
  trap - ERR
  set +e
  if [[ -n "$CURRENT_STAGE" ]]; then
    "$PYTHON" "$STATE_HELPER" status fail "$ROOT" "$CURRENT_STAGE"
  fi
  exit "$exit_code"
}
trap on_error ERR

CONFIG_TMP="$WORK/.pipeline-config.$$"
CURRENT_STAGE="ss3t"
if ! "$PYTHON" "$STATE_HELPER" config "$ROOT" >"$CONFIG_TMP" 2>"$WORK/ss3t.log"; then
  "$PYTHON" "$STATE_HELPER" status start "$ROOT" ss3t
  "$PYTHON" "$STATE_HELPER" status fail "$ROOT" ss3t
  rm -f "$CONFIG_TMP"
  exit 2
fi

declare -a CONFIG=()
exec 9<"$CONFIG_TMP"
while IFS= read -r -d '' value <&9; do
  CONFIG[${#CONFIG[@]}]="$value"
done
exec 9<&-
rm -f "$CONFIG_TMP"
if [[ ${#CONFIG[@]} -ne 13 ]]; then
  printf 'invalid runner configuration field count: %s\n' "${#CONFIG[@]}" >"$WORK/ss3t.log"
  "$PYTHON" "$STATE_HELPER" status start "$ROOT" ss3t
  "$PYTHON" "$STATE_HELPER" status fail "$ROOT" ss3t
  exit 2
fi

CASE_ID="${CONFIG[0]}"
DWI_NIFTI="${CONFIG[1]}"
DWI_BVAL="${CONFIG[2]}"
DWI_BVEC="${CONFIG[3]}"
DWI_JSON="${CONFIG[4]}"
DWI_PE_DIR="${CONFIG[5]}"
DWI_READOUT="${CONFIG[6]}"
RPE_MODE="${CONFIG[7]}"
RPE_NIFTI="${CONFIG[8]}"
RPE_JSON="${CONFIG[9]}"
T1_NIFTI="${CONFIG[10]}"
if [[ "$ROOT" != "$ROOT_REAL" ]]; then
  DWI_NIFTI="${DWI_NIFTI/#$ROOT_REAL/$ROOT}"
  DWI_BVAL="${DWI_BVAL/#$ROOT_REAL/$ROOT}"
  DWI_BVEC="${DWI_BVEC/#$ROOT_REAL/$ROOT}"
  DWI_JSON="${DWI_JSON/#$ROOT_REAL/$ROOT}"
  RPE_NIFTI="${RPE_NIFTI/#$ROOT_REAL/$ROOT}"
  RPE_JSON="${RPE_JSON/#$ROOT_REAL/$ROOT}"
  T1_NIFTI="${T1_NIFTI/#$ROOT_REAL/$ROOT}"
fi
PAIR_HEADER="${CONFIG[11]}"
REGRID_VOXEL="${CONFIG[12]}"

# MRtrix scripts create *-tmp-* scratch dirs in the working directory; keep them inside the case.
if [[ "$DRYRUN" == "1" ]]; then
  printf 'DRY cd %s\n' "$WORK"
fi
cd "$WORK"

format_command() {
  local arg quoted command_line=""
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    command_line="${command_line:+$command_line }$quoted"
  done
  printf '%s' "$command_line"
}

is_dryrun() {
  [[ "$DRYRUN" == "1" ]]
}

log_line() {
  printf '%s\n' "$*"
}

run_mr() {
  local output="$1"
  shift
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
    return 0
  fi
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$@" -quiet)"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) $(format_command "$@") -> $output"
    "$@" -quiet
  fi
}

run_fsl() {
  local output="$1"
  shift
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
    return 0
  fi
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$@")"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) $(format_command "$@") -> $output"
    "$@"
  fi
}

run_cmd() {
  local output="$1"
  shift
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
    return 0
  fi
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$@")"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) $(format_command "$@") -> $output"
    "$@"
  fi
}

run_report() {
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$@")"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) $(format_command "$@")"
    "$@"
  fi
}

run_copy() {
  local source="$1"
  local output="$2"
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
  elif is_dryrun; then
    printf 'DRY %s\n' "$(format_command cp "$source" "$output")"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) cp $source -> $output"
    cp "$source" "$output"
  fi
}

check_command() {
  local name="$1"
  if is_dryrun; then
    printf 'DRY check command -v %s\n' "$(format_command "$name")"
  else
    log_line "CHECK command -v $name"
    if ! command -v "$name" >/dev/null; then
      log_line "MISSING command: $name" >&2
      return 127
    fi
  fi
}

check_file() {
  local flag="$1"
  local file="$2"
  if is_dryrun; then
    printf 'DRY check %s %s\n' "$flag" "$(format_command "$file")"
  else
    log_line "CHECK $flag $file"
    if [[ "$flag" == "-x" && ! -x "$file" ]] || [[ "$flag" != "-x" && ! -f "$file" ]]; then
      log_line "MISSING file: $file" >&2
      return 1
    fi
  fi
}

run_extract_pair() {
  local output="$1"
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
    return 0
  fi
  if is_dryrun; then
    printf 'DRY %s | %s | %s\n' \
      "$(format_command dwiextract "$ROOT/work/ss3t/dwi_up.mif" - -bzero)" \
      "$(format_command mrmath - mean - -axis 3)" \
      "$(format_command mrconvert - "$output" -nthreads "$NTHREADS")"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) DWI b=0 extraction pipe -> $output"
    dwiextract "$ROOT/work/ss3t/dwi_up.mif" - -bzero -quiet | \
      mrmath - mean - -axis 3 -quiet | \
      mrconvert - "$output" -nthreads "$NTHREADS" -quiet
  fi
}

volume_count() {
  local dims
  dims="$(mrinfo "$1" -size -quiet)"
  set -- $dims
  if [[ $# -ge 4 ]]; then printf '%s' "$4"; else printf '1'; fi
}

# Keep the first N volumes of a series (the topup SE-EPI pair only needs a few b=0 per polarity).
run_cap_volumes() {
  local input="$1" output="$2" count="$3"
  if [[ -e "$output" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP exists: %s\n' "$output"
    else
      log_line "SKIP exists: $output"
    fi
    return 0
  fi
  if is_dryrun || [[ "$(volume_count "$input")" -gt 1 ]]; then
    run_mr "$output" mrconvert "$input" "$output" -coord 3 "0:$((count - 1))" -nthreads "$NTHREADS"
  else
    run_mr "$output" mrconvert "$input" "$output" -nthreads "$NTHREADS"
  fi
}

check_same_grid() {
  if is_dryrun; then
    printf 'DRY check same voxel grid %s %s\n' "$1" "$2"
    return 0
  fi
  local first second
  first="$(mrinfo "$1" -spacing -quiet | awk '{print $1, $2, $3}') $(mrinfo "$1" -size -quiet | awk '{print $1, $2, $3}')"
  second="$(mrinfo "$2" -spacing -quiet | awk '{print $1, $2, $3}') $(mrinfo "$2" -size -quiet | awk '{print $1, $2, $3}')"
  log_line "CHECK grid $1: $first | $2: $second"
  if [[ "$first" != "$second" ]]; then
    log_line "DWI and reverse-PE grids differ after regrid: $first vs $second" >&2
    return 1
  fi
}

run_gradcheck() {
  local work="$1"
  local report="$work/gradcheck.txt"
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command dwigradcheck "$work/dwi_up.mif" -mask "$work/mask_up.mif" -number 10000 -nthreads "$NTHREADS")"
    printf 'DRY %s\n' "$(format_command "$PYTHON" "$STATE_HELPER" gradcheck "$ROOT" "$report")"
    return 0
  fi
  if [[ -e "$report" ]]; then
    log_line "SKIP exists: $report"
  else
    log_line "RUN $(date -u +%H:%M:%SZ) dwigradcheck -> $report"
    dwigradcheck "$work/dwi_up.mif" -mask "$work/mask_up.mif" -number 10000 -nthreads "$NTHREADS" \
      >"$report.tmp" 2>&1
    mv "$report.tmp" "$report"
  fi
  "$PYTHON" "$STATE_HELPER" gradcheck "$ROOT" "$report"
}

stage_ss3t() {
  local work="$ROOT/work/ss3t"
  mkdir -p "$work/eddyqc" "$NIFTI"
  if is_dryrun; then
    printf 'DRY update manifest case_root=%s\n' "$ROOT"
  else
    "$PYTHON" "$STATE_HELPER" manifest-root "$ROOT"
  fi

  check_command ss3t_csd_beta1
  check_command eddy_cpu
  check_command N4BiasFieldCorrection

  # Header PE metadata is used only when both sidecars are complete; otherwise any partial
  # PE fields are cleared so dwifslpreproc takes -pe_dir/-readout_time from case.json alone.
  local -a pe_clear=()
  if [[ "$PAIR_HEADER" != "1" ]]; then
    pe_clear=(-clear_property PhaseEncodingDirection -clear_property TotalReadoutTime)
  fi

  run_mr "$work/dwi_raw.mif" mrconvert \
    "$DWI_NIFTI" "$work/dwi_raw.mif" \
    -fslgrad "$DWI_BVEC" "$DWI_BVAL" \
    -json_import "$DWI_JSON" \
    ${pe_clear[@]+"${pe_clear[@]}"} \
    -nthreads "$NTHREADS"

  # Zero-filled reconstructions (ReconMatrixPE > AcquisitionMatrixPE) go back to the acquired
  # grid before MP-PCA denoising and Gibbs removal, which assume un-interpolated data.
  local dwi_in="$work/dwi_raw.mif"
  if [[ -n "$REGRID_VOXEL" ]]; then
    dwi_in="$work/dwi_acq.mif"
    run_mr "$dwi_in" mrgrid "$work/dwi_raw.mif" regrid "$dwi_in" \
      -voxel "$REGRID_VOXEL" -interp sinc -nthreads "$NTHREADS"
  fi

  local -a rpe_args
  if [[ "$RPE_MODE" == "pair" ]]; then
    run_mr "$work/rpe.mif" mrconvert "$RPE_NIFTI" "$work/rpe.mif" \
      -json_import "$RPE_JSON" ${pe_clear[@]+"${pe_clear[@]}"} -nthreads "$NTHREADS"
    local rpe_in="$work/rpe.mif"
    if [[ -n "$REGRID_VOXEL" ]]; then
      rpe_in="$work/rpe_acq.mif"
      run_mr "$rpe_in" mrgrid "$work/rpe.mif" regrid "$rpe_in" \
        -voxel "$REGRID_VOXEL" -interp sinc -nthreads "$NTHREADS"
      check_same_grid "$dwi_in" "$rpe_in"
    fi
    run_mr "$work/dwi_b0_pe.mif" dwiextract "$dwi_in" "$work/dwi_b0_pe.mif" -bzero
    local se_count="$SE_CAP"
    if ! is_dryrun && [[ ! -e "$work/se_pair.mif" ]]; then
      local n_pe n_rpe
      n_pe="$(volume_count "$work/dwi_b0_pe.mif")"
      n_rpe="$(volume_count "$rpe_in")"
      (( n_pe < se_count )) && se_count="$n_pe"
      (( n_rpe < se_count )) && se_count="$n_rpe"
    fi
    run_cap_volumes "$work/dwi_b0_pe.mif" "$work/dwi_b0_cap.mif" "$se_count"
    run_cap_volumes "$rpe_in" "$work/rpe_b0_cap.mif" "$se_count"
    run_mr "$work/se_pair.mif" mrcat "$work/dwi_b0_cap.mif" "$work/rpe_b0_cap.mif" "$work/se_pair.mif" -axis 3
    # Exactly one -rpe_* option: dwifslpreproc rejects -rpe_pair combined with -rpe_header.
    rpe_args=(-rpe_pair -se_epi "$work/se_pair.mif")
    if [[ "$PAIR_HEADER" != "1" ]]; then
      rpe_args+=(-pe_dir "$DWI_PE_DIR" -readout_time "$DWI_READOUT")
    fi
    # -align_seepi is rejected by dwifslpreproc without SE-EPI data, so pair mode only.
    rpe_args+=(-align_seepi)
  else
    rpe_args=(-rpe_none -pe_dir "$DWI_PE_DIR" -readout_time "$DWI_READOUT")
  fi

  run_mr "$work/dwi_den.mif" dwidenoise "$dwi_in" "$work/dwi_den.mif" \
    -noise "$work/noise.mif" -nthreads "$NTHREADS"
  if [[ ! -e "$work/residual.mif" ]]; then
    if is_dryrun; then
      printf 'DRY %s\n' "$(format_command mrcalc "$dwi_in" "$work/dwi_den.mif" -subtract "$work/residual.mif")"
    else
      mrcalc "$dwi_in" "$work/dwi_den.mif" -subtract "$work/residual.mif" -quiet
    fi
  elif is_dryrun; then
    printf 'DRY SKIP exists: %s\n' "$work/residual.mif"
  fi

  run_mr "$work/dwi_den_deg.mif" mrdegibbs "$work/dwi_den.mif" "$work/dwi_den_deg.mif" \
    -nthreads "$NTHREADS"
  run_mr "$work/dwi_preproc.mif" dwifslpreproc "$work/dwi_den_deg.mif" "$work/dwi_preproc.mif" \
    "${rpe_args[@]}" \
    -eddy_options " --slm=linear --repol --data_is_shelled --nthr=$NTHREADS" \
    -eddyqc_all "$work/eddyqc" \
    -nthreads "$NTHREADS"
  run_mr "$work/dwi_bias.mif" dwibiascorrect ants "$work/dwi_preproc.mif" "$work/dwi_bias.mif" \
    -nthreads "$NTHREADS"
  run_mr "$work/dwi_up.mif" mrgrid "$work/dwi_bias.mif" regrid "$work/dwi_up.mif" \
    -voxel 1.3 -nthreads "$NTHREADS"
  run_mr "$work/mask_up.mif" dwi2mask "$work/dwi_up.mif" "$work/mask_up.mif" -nthreads "$NTHREADS"
  run_gradcheck "$work"
  run_mr "$work/wm.txt" dwi2response dhollander "$work/dwi_up.mif" \
    "$work/wm.txt" "$work/gm.txt" "$work/csf.txt" \
    -mask "$work/mask_up.mif" -voxels "$work/sf_voxels.mif" -nthreads "$NTHREADS"
  run_mr "$work/wmfod.mif" ss3t_csd_beta1 "$work/dwi_up.mif" \
    "$work/wm.txt" "$work/wmfod.mif" \
    "$work/gm.txt" "$work/gm.mif" \
    "$work/csf.txt" "$work/csf.mif" \
    -mask "$work/mask_up.mif" -nthreads "$NTHREADS"
  run_mr "$work/wmfod_norm.mif" mtnormalise \
    "$work/wmfod.mif" "$work/wmfod_norm.mif" \
    "$work/gm.mif" "$work/gm_norm.mif" \
    "$work/csf.mif" "$work/csf_norm.mif" \
    -mask "$work/mask_up.mif" -nthreads "$NTHREADS"

  run_copy "$work/wmfod_norm.mif" "$NIFTI/wmfod_norm.mif"
  run_mr "$NIFTI/mask_up.nii.gz" mrconvert "$work/mask_up.mif" "$NIFTI/mask_up.nii.gz" -nthreads "$NTHREADS"
  run_extract_pair "$NIFTI/b0.nii.gz"
  run_report mrinfo "$NIFTI/wmfod_norm.mif" -size -spacing
  run_report mrinfo "$NIFTI/b0.nii.gz" -size -spacing
}

stage_t1reg() {
  local work="$ROOT/work/t1reg"
  mkdir -p "$work" "$NIFTI"
  export FSLOUTPUTTYPE=NIFTI_GZ

  check_command flirt
  check_command bet
  check_command mrconvert
  check_file -f "$NIFTI/b0.nii.gz"
  check_file -f "$NIFTI/mask_up.nii.gz"
  check_file -f "$T1_NIFTI"

  run_mr "$work/t1.nii.gz" mrconvert "$T1_NIFTI" "$work/t1.nii.gz" -nthreads "$NTHREADS"
  run_fsl "$work/t1_brain.nii.gz" bet "$work/t1.nii.gz" "$work/t1_brain.nii.gz" -f 0.4 -B
  run_fsl "$work/b0_brain.nii.gz" fslmaths "$NIFTI/b0.nii.gz" -mas "$NIFTI/mask_up.nii.gz" "$work/b0_brain.nii.gz"
  run_fsl "$work/b02t1.mat" flirt \
    -in "$work/b0_brain.nii.gz" -ref "$work/t1_brain.nii.gz" \
    -omat "$work/b02t1.mat" -dof 6 -cost normmi
  if [[ ! -e "$work/t12b0.mat" ]]; then
    if is_dryrun; then
      printf 'DRY %s\n' "$(format_command convert_xfm -omat "$work/t12b0.mat" -inverse "$work/b02t1.mat")"
    else
      log_line "RUN $(date -u +%H:%M:%SZ) convert_xfm -inverse -> $work/t12b0.mat"
      convert_xfm -omat "$work/t12b0.mat" -inverse "$work/b02t1.mat"
    fi
  elif is_dryrun; then
    printf 'DRY SKIP exists: %s\n' "$work/t12b0.mat"
  fi
  run_fsl "$work/t1_brain_dwi.nii.gz" flirt \
    -in "$work/t1_brain.nii.gz" -ref "$NIFTI/b0.nii.gz" \
    -applyxfm -init "$work/t12b0.mat" \
    -out "$work/t1_brain_dwi.nii.gz" -interp spline

  run_copy "$work/t1_brain_dwi.nii.gz" "$NIFTI/t1_brain_dwi.nii.gz"
  run_copy "$work/b02t1.mat" "$NIFTI/b02t1.mat"
  run_copy "$work/t12b0.mat" "$NIFTI/t12b0.mat"
  run_report mrinfo "$NIFTI/t1_brain_dwi.nii.gz" -size -spacing
  run_report mrinfo "$NIFTI/b0.nii.gz" -size -spacing
  run_report mrinfo "$NIFTI/wmfod_norm.mif" -size -spacing
}

stage_banks() {
  local work="$ROOT/work/banks"
  local subjects_dir="$ROOT/work/freesurfer"
  local fs_id="$CASE_ID"
  local fs_sub="$subjects_dir/$fs_id"
  local anat="$work/anat"
  local tract_dir="$ROOT/tracts/connectome/pre/freesurfer/path-to-wm-v2"
  local tracks="$tract_dir/wholebrain_act_ifod2_10000000.tck"
  local tracks_100k="$tract_dir/wholebrain_act_ifod2_100000.tck"
  local seeds="$tract_dir/accepted_seeds_10000000.txt"
  local five_tt_dwi="$anat/5tt_freesurfer_path_to_wm_v2_in_dwi.mif"
  local warp_prefix="$ROOT/normative/mni2case_"
  mkdir -p "$work" "$anat" "$tract_dir" "$subjects_dir" \
    "$ROOT/tracts/bank" "$ROOT/tracts/roi" "$ROOT/normative" "$ROOT/qc"
  export SUBJECTS_DIR="$subjects_dir"

  if is_dryrun; then
    printf 'DRY check -f %s\n' "$(format_command "$FREESURFER_HOME/SetUpFreeSurfer.sh")"
  elif [[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]]; then
    # FreeSurfer's setup scripts tolerate failing probes (e.g. grep | wc on ~/matlab/startup.m);
    # the runner's ERR trap and pipefail must be off while they run, not just -e/-u.
    trap - ERR
    set +Eeuo pipefail
    # shellcheck disable=SC1091
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    set -Eeuo pipefail
    trap on_error ERR
    export SUBJECTS_DIR="$subjects_dir"
    export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$MRTRIX3:$FREESURFER_HOME/bin:$PATH"
  fi

  if [[ "$FS_RECON" == "clinical" ]]; then
    check_command recon-all-clinical.sh
  else
    check_command recon-all
  fi
  check_command bbregister
  check_file -x "$MRTRIX3/tckgen"
  check_file -x "$MRTRIX3/5ttgen"
  check_file -f "$NIFTI/wmfod_norm.mif"
  check_file -f "$NIFTI/b0.nii.gz"
  check_file -f "$NIFTI/t1_brain_dwi.nii.gz"
  check_file -f "$T1_NIFTI"
  check_file -f "$FS_LUT"
  check_file -f "$MNI_BRAIN"

  # Re-point a link left by another FreeSurfer install so one recon never mixes versions.
  if [[ ! -e "$subjects_dir/fsaverage" || ( -L "$subjects_dir/fsaverage" \
        && "$(readlink "$subjects_dir/fsaverage")" != "$FREESURFER_HOME/subjects/fsaverage" ) ]]; then
    if is_dryrun; then
      printf 'DRY %s\n' "$(format_command ln -sfn "$FREESURFER_HOME/subjects/fsaverage" "$subjects_dir/fsaverage")"
    else
      ln -sfn "$FREESURFER_HOME/subjects/fsaverage" "$subjects_dir/fsaverage"
    fi
  elif is_dryrun; then
    printf 'DRY SKIP exists: %s\n' "$subjects_dir/fsaverage"
  fi

  local fs_t1="$fs_sub/mri/T1.mgz"
  if [[ "$FS_RECON" == "clinical" ]]; then
    # recon-all-clinical (SynthSeg/SynthSR/SynthDist) for scans where classic recon-all stalls, e.g.
    # enhancing dura on post-contrast T1. It writes no T1.mgz/orig.mgz: SynthSR stands in for T1, and
    # orig.mgz links to norm.mgz, the conformed grid its aparc+aseg and surfaces share (bbregister template).
    fs_t1="$fs_sub/mri/synthSR.mgz"
    local clinical_done="$fs_sub/scripts/tractlab-recon-all-clinical.done"
    if [[ -f "$clinical_done" ]]; then
      if is_dryrun; then
        printf 'DRY SKIP recon-all-clinical (done): %s\n' "$fs_sub"
      else
        log_line "SKIP recon-all-clinical (done): $fs_sub"
      fi
    else
      run_report recon-all-clinical.sh -i "$T1_NIFTI" -subjid "$fs_id" -threads "$NTHREADS" -sdir "$subjects_dir"
      if is_dryrun; then
        printf 'DRY %s\n' "$(format_command ln -sfn norm.mgz "$fs_sub/mri/orig.mgz")"
      else
        check_file -f "$fs_sub/mri/norm.mgz"
        check_file -f "$fs_t1"
        ln -sfn norm.mgz "$fs_sub/mri/orig.mgz"
        printf 'recon-all-clinical %s\n' "$(cat "$FREESURFER_HOME/build-stamp.txt" 2>/dev/null || echo unknown)" >"$clinical_done"
      fi
    fi
  elif [[ -f "$fs_sub/scripts/recon-all.done" ]]; then
    if is_dryrun; then
      printf 'DRY SKIP recon-all (done): %s\n' "$fs_sub"
    else
      log_line "SKIP recon-all (done): $fs_sub"
    fi
  else
    if [[ ! -d "$fs_sub/mri/orig" ]]; then
      run_report recon-all -s "$fs_id" -i "$T1_NIFTI" -all -openmp "$NTHREADS"
    else
      run_report recon-all -s "$fs_id" -autorecon2 -autorecon3 -openmp "$NTHREADS"
    fi
    check_file -f "$fs_sub/scripts/recon-all.done"
  fi
  check_file -f "$fs_sub/mri/aparc+aseg.mgz"

  run_mr "$anat/t1_fs.mif" "$MRTRIX3/mrconvert" "$fs_t1" "$anat/t1_fs.mif" -nthreads "$NTHREADS"
  run_mr "$anat/5tt_freesurfer_fs_raw.mif" "$MRTRIX3/5ttgen" freesurfer \
    "$fs_sub/mri/aparc+aseg.mgz" "$anat/5tt_freesurfer_fs_raw.mif" \
    -lut "$FS_LUT" -nocrop -sgm_amyg_hipp -nthreads "$NTHREADS"
  run_mr "$anat/5tt_auto_cat5.mif" "$MRTRIX3/mrconvert" \
    "$anat/5tt_freesurfer_fs_raw.mif" -coord 3 4 "$anat/5tt_auto_cat5.mif"
  run_cmd "$anat/5tt_freesurfer_path_to_wm_v2.mif" "$MRTRIX3/5ttedit" \
    "$anat/5tt_freesurfer_fs_raw.mif" "$anat/5tt_freesurfer_path_to_wm_v2.mif" \
    -wm "$anat/5tt_auto_cat5.mif"

  run_cmd "$anat/b0_to_fs.dat" bbregister --s "$fs_id" --mov "$NIFTI/b0.nii.gz" \
    --reg "$anat/b0_to_fs.dat" --dti --init-coreg --o "$anat/b0_in_fs.nii.gz"
  if [[ ! -e "$anat/b0_to_fs.mat" ]]; then
    if is_dryrun; then
      printf 'DRY %s\n' "$(format_command tkregister2 --mov "$NIFTI/b0.nii.gz" --targ "$fs_sub/mri/orig.mgz" --reg "$anat/b0_to_fs.dat" --fslregout "$anat/b0_to_fs.mat" --noedit)"
    else
      log_line "RUN $(date -u +%H:%M:%SZ) tkregister2 --fslregout"
      tkregister2 --mov "$NIFTI/b0.nii.gz" --targ "$fs_sub/mri/orig.mgz" \
        --reg "$anat/b0_to_fs.dat" --fslregout "$anat/b0_to_fs.mat" --noedit
    fi
  elif is_dryrun; then
    printf 'DRY SKIP exists: %s\n' "$anat/b0_to_fs.mat"
  fi
  run_cmd "$anat/b0_to_fs.mrtrix" "$MRTRIX3/transformconvert" \
    "$anat/b0_to_fs.mat" "$NIFTI/b0.nii.gz" "$fs_sub/mri/orig.mgz" \
    flirt_import "$anat/b0_to_fs.mrtrix"
  run_mr "$five_tt_dwi" "$MRTRIX3/mrtransform" \
    "$anat/5tt_freesurfer_path_to_wm_v2.mif" "$five_tt_dwi" \
    -linear "$anat/b0_to_fs.mrtrix" -inverse -nthreads "$NTHREADS"
  run_report "$MRTRIX3/5ttcheck" "$five_tt_dwi"
  run_mr "$anat/aparc_fs.mif" "$MRTRIX3/mrconvert" \
    "$fs_sub/mri/aparc+aseg.mgz" "$anat/aparc_fs.mif" -nthreads "$NTHREADS"
  run_mr "$NIFTI/aparc_dwi.nii.gz" "$MRTRIX3/mrtransform" \
    "$anat/aparc_fs.mif" "$NIFTI/aparc_dwi.nii.gz" \
    -linear "$anat/b0_to_fs.mrtrix" -inverse \
    -template "$NIFTI/b0.nii.gz" -strides "$NIFTI/b0.nii.gz" -interp nearest -datatype uint32 \
    -nthreads "$NTHREADS"

  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$PYTHON" "$STATE_HELPER" laterality "$NIFTI/aparc_dwi.nii.gz" "$NIFTI/b0.nii.gz")"
  else
    "$PYTHON" "$STATE_HELPER" laterality "$NIFTI/aparc_dwi.nii.gz" "$NIFTI/b0.nii.gz"
  fi

  run_mr "$tracks" "$MRTRIX3/tckgen" "$NIFTI/wmfod_norm.mif" "$tracks" \
    -algorithm iFOD2 -seed_dynamic "$NIFTI/wmfod_norm.mif" \
    -act "$five_tt_dwi" -backtrack -crop_at_gmwmi \
    -select 10000000 -seeds 0 -cutoff 0.06 -minlength 10 -maxlength 250 \
    -output_seeds "$seeds" -nthreads "$NTHREADS"
  run_report "$MRTRIX3/tckinfo" "$tracks" -count
  run_mr "$tracks_100k" "$MRTRIX3/tckedit" "$tracks" "$tracks_100k" \
    -number 100000 -nthreads "$NTHREADS"

  run_report env "PYTHONPATH=$REPO/src" "$PYTHON" "$REPO/scripts/rebuild_strict_banks.py" \
    --case-root "$ROOT" --manifest "$MANIFEST" --nthreads "$NTHREADS"
  if is_dryrun; then
    printf 'DRY %s\n' "$(format_command "$PYTHON" "$STATE_HELPER" manifest-banks "$ROOT" "$RPE_MODE")"
  else
    "$PYTHON" "$STATE_HELPER" manifest-banks "$ROOT" "$RPE_MODE"
  fi
  run_report env "PYTHONPATH=$REPO/src" "$PYTHON" "$REPO/scripts/prune_banks.py" \
    --case-root "$ROOT" --manifest "$MANIFEST" \
    --fod "$NIFTI/wmfod_norm.mif"

  if [[ ! -e "${warp_prefix}1Warp.nii.gz" ]]; then
    run_report "$ANTSPATH/antsRegistrationSyN.sh" -d 3 \
      -f "$NIFTI/t1_brain_dwi.nii.gz" \
      -m "$MNI_BRAIN" \
      -o "$warp_prefix" \
      -n 4
  elif is_dryrun; then
    printf 'DRY SKIP exists: %s\n' "${warp_prefix}1Warp.nii.gz"
  fi
  run_report env "PYTHONPATH=$REPO/src" "$PYTHON" -m tractlab.atlas_prep --manifest "$MANIFEST" --full-syn
  run_report env "PYTHONPATH=$REPO/src" "$PYTHON" -m tractlab.atlas_prep --manifest "$MANIFEST" --parcellation --full-syn
  run_report "$MRTRIX3/mrinfo" "$NIFTI/aparc_dwi.nii.gz" -size -spacing
  run_report "$MRTRIX3/mrinfo" "$five_tt_dwi" -size -spacing
  run_report "$MRTRIX3/tckinfo" "$tracks" -count
}

stage_scalar_maps() {
  if is_dryrun; then
    check_command "$HOME/mrtrix3/bin/dwi2tensor"
    check_command "$HOME/mrtrix3/bin/tensor2metric"
    check_command "$HOME/mrtrix3/bin/mrgrid"
  fi
  run_report env "TRACTLAB_PYTHON=$PYTHON" "PYTHONPATH=$REPO/src" \
    "$REPO/scripts/make_scalar_maps.sh" --case-root "$ROOT"
}

run_stage() {
  local stage="$1"
  local function_name="$2"
  local log="$ROOT/work/$stage.log"
  CURRENT_STAGE="$stage"
  "$PYTHON" "$STATE_HELPER" status start "$ROOT" "$stage"
  if is_dryrun; then
    printf 'DRY STAGE %s\n' "$stage"
    if [[ "${TRACTLAB_DRYRUN_FAIL_AT:-}" == "$stage" ]]; then
      printf 'Injected dry-run failure at stage %s\n' "$stage" >>"$log"
      return 86
    fi
    "$function_name"
  else
    {
      log_line "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) stage=$stage nthreads=$NTHREADS ====="
      "$function_name"
    } >>"$log" 2>&1
  fi
  "$PYTHON" "$STATE_HELPER" status complete "$ROOT" "$stage"
}

for stage in ss3t t1reg banks scalar_maps; do
  case "$stage" in
    ss3t) run_stage "$stage" stage_ss3t ;;
    t1reg) run_stage "$stage" stage_t1reg ;;
    banks) run_stage "$stage" stage_banks ;;
    scalar_maps) run_stage "$stage" stage_scalar_maps ;;
  esac
done

CURRENT_STAGE="scalar_maps"
"$PYTHON" "$STATE_HELPER" status finish "$ROOT"
