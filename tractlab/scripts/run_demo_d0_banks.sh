#!/usr/bin/env bash
# House bank + atlas_prep path for D0 Leipzig sub-010005.
# Reuses 3T connectome commands (recon-all → 5ttgen freesurfer → path-to-wm-v2
# → bbregister → 10M ACT iFOD2) then tractlab rebuild_strict_banks.py,
# prune_banks.py, and python -m tractlab.atlas_prep.
# Does not switch run.py. Does not approve atlas/parcellation QC.
# Does not edit track.py / serve.py.
# Must be launched with nohup — recon-all is foreground and dies on SIGHUP if the parent shell exits.
set -euo pipefail

NTHREADS="${NTHREADS:-8}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$REPO/cases/demo-leipzig-sub-010005"
MANIFEST="$ROOT/manifest.json"
RAW_T1="$ROOT/raw/sub-010005/ses-01/anat/sub-010005_ses-01_acq-mp2rage_T1w.nii.gz"
NIFTI="$ROOT/nifti"
WORK="$ROOT/work/banks"
FS_ID="sub-010005"
SUBJECTS_DIR="$ROOT/work/freesurfer"
FS_SUB="$SUBJECTS_DIR/$FS_ID"
ANAT="$WORK/anat"
TRACT_DIR="$ROOT/tracts/connectome/pre/freesurfer/path-to-wm-v2"
TRACKS="$TRACT_DIR/wholebrain_act_ifod2_10000000.tck"
TRACKS_100K="$TRACT_DIR/wholebrain_act_ifod2_100000.tck"
SEEDS="$TRACT_DIR/accepted_seeds_10000000.txt"
FIVE_TT_DWI="$ANAT/5tt_freesurfer_path_to_wm_v2_in_dwi.mif"
LOG="$WORK/banks.log"

export FSLDIR="${FSLDIR:-$HOME/fsl}"
export ANTSPATH="${ANTSPATH:-$HOME/ants-2.6.5/bin}"
export FREESURFER_HOME="${FREESURFER_HOME:-$HOME/freesurfer}"
export FSLOUTPUTTYPE=NIFTI_GZ
export SUBJECTS_DIR
# User-required PATH, then house mrtrix3 (tckgen/tckedit/5ttgen) + FreeSurfer.
export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$HOME/mrtrix3/bin:$FREESURFER_HOME/bin:$PATH"

MRTRIX3="$HOME/mrtrix3/bin"
FS_LUT="$FREESURFER_HOME/FreeSurferColorLUT.txt"
MNI_BRAIN="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"

mkdir -p "$WORK" "$ANAT" "$TRACT_DIR" "$SUBJECTS_DIR" "$ROOT/tracts/bank" "$ROOT/tracts/roi" "$ROOT/normative" "$ROOT/qc"
# Progress bars must not hit the log.
exec >>"$LOG" 2>&1
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) nthreads=$NTHREADS pid=$$ ====="
echo "$$" > "$WORK/banks.pid"

if [[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]]; then
  set +eu
  # shellcheck disable=SC1091
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
  set -eu
  export SUBJECTS_DIR
  export PATH="$HOME/mrtrix3tissue/pyshim:$HOME/mrtrix3tissue/bin:$ANTSPATH:$FSLDIR/share/fsl/bin:$FSLDIR/bin:$HOME/mrtrix3/bin:$FREESURFER_HOME/bin:$PATH"
fi
echo "FS sourced recon-all=$(command -v recon-all)"

command -v recon-all >/dev/null
command -v bbregister >/dev/null
[[ -x "$MRTRIX3/tckgen" ]]
[[ -x "$MRTRIX3/5ttgen" ]]
[[ -f "$NIFTI/wmfod_norm.mif" ]]
[[ -f "$NIFTI/b0.nii.gz" ]]
[[ -f "$NIFTI/t1_brain_dwi.nii.gz" ]]
[[ -f "$RAW_T1" ]]
[[ -f "$FS_LUT" ]]

if [[ ! -e "$SUBJECTS_DIR/fsaverage" ]]; then
  ln -sfn "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/fsaverage"
fi

run_mr() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $(date -u +%H:%M:%SZ) $*  -> $out"
  "$@" -quiet
}

run_cmd() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $(date -u +%H:%M:%SZ) $*  -> $out"
  "$@"
}

# ── 1. FreeSurfer recon-all (native MP2RAGE T1w) ─────────────────────────────
# FS 7.4 dropped `recon-all -s $id` / `-make` resume. Always pass an explicit
# stage. A leftover IsRunning.* from a killed job will hang the WAIT branch
# forever — drop it when the recorded PROCESSID is dead.
_fs_running_files() {
  ls "$FS_SUB/scripts"/IsRunning.lh "$FS_SUB/scripts"/IsRunning.rh \
     "$FS_SUB/scripts"/IsRunning.lh+rh 2>/dev/null || true
}
_clear_stale_fs_lock() {
  local lock pid
  for lock in $(_fs_running_files); do
    pid="$(awk '/^PROCESSID/{print $2}' "$lock" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "WAIT recon-all pid=$pid still live ($lock)"
      return 1
    fi
    echo "STALE $lock (pid=${pid:-none} dead) — removing"
    rm -f "$lock"
  done
  return 0
}

if [[ -f "$FS_SUB/scripts/recon-all.done" ]]; then
  echo "SKIP recon-all (done): $FS_SUB"
else
  if ! _clear_stale_fs_lock; then
    echo "WAIT recon-all already running in $FS_SUB"
    while ! _clear_stale_fs_lock; do
      sleep 60
    done
    [[ -f "$FS_SUB/scripts/recon-all.done" ]] || { echo "FAIL recon-all did not finish"; exit 1; }
  else
    if [[ ! -d "$FS_SUB/mri/orig" ]]; then
      echo "RUN $(date -u +%H:%M:%SZ) recon-all -s $FS_ID -i T1 -all -openmp $NTHREADS"
      recon-all -s "$FS_ID" -i "$RAW_T1" -all -openmp "$NTHREADS"
    else
      echo "RUN $(date -u +%H:%M:%SZ) recon-all -s $FS_ID -autorecon2 -autorecon3 -openmp $NTHREADS"
      recon-all -s "$FS_ID" -autorecon2 -autorecon3 -openmp "$NTHREADS"
    fi
    [[ -f "$FS_SUB/scripts/recon-all.done" ]] || { echo "FAIL recon-all.done missing"; exit 1; }
  fi
fi
[[ -f "$FS_SUB/mri/aparc+aseg.mgz" ]]

# ── 2. 5TT (freesurfer + path-to-wm-v2) ──────────────────────────────────────
run_mr "$ANAT/t1_fs.mif" "$MRTRIX3/mrconvert" "$FS_SUB/mri/T1.mgz" "$ANAT/t1_fs.mif" -nthreads "$NTHREADS"
run_mr "$ANAT/5tt_freesurfer_fs_raw.mif" "$MRTRIX3/5ttgen" freesurfer \
  "$FS_SUB/mri/aparc+aseg.mgz" "$ANAT/5tt_freesurfer_fs_raw.mif" \
  -lut "$FS_LUT" -nocrop -sgm_amyg_hipp -nthreads "$NTHREADS"
run_mr "$ANAT/5tt_auto_cat5.mif" "$MRTRIX3/mrconvert" \
  "$ANAT/5tt_freesurfer_fs_raw.mif" -coord 3 4 "$ANAT/5tt_auto_cat5.mif"
run_cmd "$ANAT/5tt_freesurfer_path_to_wm_v2.mif" "$MRTRIX3/5ttedit" \
  "$ANAT/5tt_freesurfer_fs_raw.mif" "$ANAT/5tt_freesurfer_path_to_wm_v2.mif" \
  -wm "$ANAT/5tt_auto_cat5.mif"

# ── 3. bbregister b0 → FS, then warp 5TT + native aparc onto DWI ─────────────
run_cmd "$ANAT/b0_to_fs.dat" bbregister --s "$FS_ID" --mov "$NIFTI/b0.nii.gz" \
  --reg "$ANAT/b0_to_fs.dat" --dti --init-fsl --o "$ANAT/b0_in_fs.nii.gz"
if [[ ! -e "$ANAT/b0_to_fs.mat" ]]; then
  echo "RUN $(date -u +%H:%M:%SZ) tkregister2 --fslregout"
  tkregister2 --mov "$NIFTI/b0.nii.gz" --targ "$FS_SUB/mri/orig.mgz" \
    --reg "$ANAT/b0_to_fs.dat" --fslregout "$ANAT/b0_to_fs.mat" --noedit
fi
run_cmd "$ANAT/b0_to_fs.mrtrix" "$MRTRIX3/transformconvert" \
  "$ANAT/b0_to_fs.mat" "$NIFTI/b0.nii.gz" "$FS_SUB/mri/orig.mgz" \
  flirt_import "$ANAT/b0_to_fs.mrtrix"

# House 5TT: header-only inverse (keep 1 mm FS grid, update scanner space).
run_mr "$FIVE_TT_DWI" "$MRTRIX3/mrtransform" \
  "$ANAT/5tt_freesurfer_path_to_wm_v2.mif" "$FIVE_TT_DWI" \
  -linear "$ANAT/b0_to_fs.mrtrix" -inverse -nthreads "$NTHREADS"
echo "RUN $(date -u +%H:%M:%SZ) 5ttcheck"
"$MRTRIX3/5ttcheck" "$FIVE_TT_DWI"

# aparc_dwi: native FreeSurfer labels on the FOD/b0 grid (viewer + bank ROIs).
run_mr "$ANAT/aparc_fs.mif" "$MRTRIX3/mrconvert" \
  "$FS_SUB/mri/aparc+aseg.mgz" "$ANAT/aparc_fs.mif" -nthreads "$NTHREADS"
run_mr "$NIFTI/aparc_dwi.nii.gz" "$MRTRIX3/mrtransform" \
  "$ANAT/aparc_fs.mif" "$NIFTI/aparc_dwi.nii.gz" \
  -linear "$ANAT/b0_to_fs.mrtrix" -inverse \
  -template "$NIFTI/b0.nii.gz" -interp nearest -datatype uint32 \
  -nthreads "$NTHREADS"

# mrtransform -template keeps MRtrix size but can permute nibabel strides
# (same trap as t1reg: template alone is not a nibabel grid match).
echo "RUN $(date -u +%H:%M:%SZ) conform aparc_dwi strides to b0"
B0_STRIDES="$("$MRTRIX3/mrinfo" "$NIFTI/b0.nii.gz" -strides | tr -s '[:space:]' ',' | sed 's/,$//')"
"$MRTRIX3/mrconvert" "$NIFTI/aparc_dwi.nii.gz" "$ANAT/aparc_dwi_strided.nii.gz" \
  -strides "$B0_STRIDES" -force -quiet
mv -f "$ANAT/aparc_dwi_strided.nii.gz" "$NIFTI/aparc_dwi.nii.gz"

echo "RUN $(date -u +%H:%M:%SZ) laterality check aparc_dwi"
PYTHONPATH="$REPO/src" python3 - <<'PY'
import sys
import nibabel as nib
import numpy as np
from pathlib import Path
root = Path(str(Path.home()) + "/tractlab/cases/demo-leipzig-sub-010005")
aparc = nib.load(str(root / "nifti/aparc_dwi.nii.gz"))
b0 = nib.load(str(root / "nifti/b0.nii.gz"))
lab = np.asanyarray(aparc.dataobj)
if aparc.shape != b0.shape:
    sys.exit(f"aparc_dwi shape {aparc.shape} != b0 {b0.shape}")
if not np.allclose(aparc.affine, b0.affine, atol=1e-3):
    sys.exit("aparc_dwi affine != b0 affine (nibabel authority)")
aff = aparc.affine
def centroid(label):
    vox = np.argwhere(lab == label)
    if vox.size == 0:
        sys.exit(f"empty label {label}")
    world = nib.affines.apply_affine(aff, vox.mean(axis=0))
    return world
lx, rx = centroid(1024)[0], centroid(2024)[0]  # precentral L/R, RAS x
print(f"precentral L x={lx:.2f} R x={rx:.2f} (RAS: R>L expected)")
if not (lx + 10 < rx):
    sys.exit(f"laterality fail: L x={lx:.2f} not left of R x={rx:.2f}")
print("laterality OK")
PY

# ── 4. 10M ACT iFOD2 (house flags) + 100k Explore filter ─────────────────────
run_mr "$TRACKS" "$MRTRIX3/tckgen" "$NIFTI/wmfod_norm.mif" "$TRACKS" \
  -algorithm iFOD2 -seed_dynamic "$NIFTI/wmfod_norm.mif" \
  -act "$FIVE_TT_DWI" -backtrack -crop_at_gmwmi \
  -select 10000000 -seeds 0 -cutoff 0.06 -minlength 10 -maxlength 250 \
  -output_seeds "$SEEDS" -nthreads "$NTHREADS"
echo "RUN $(date -u +%H:%M:%SZ) tckinfo 10M"
"$MRTRIX3/tckinfo" "$TRACKS" -count
run_mr "$TRACKS_100K" "$MRTRIX3/tckedit" "$TRACKS" "$TRACKS_100K" \
  -number 100000 -nthreads "$NTHREADS"

# ── 5. Multi-ROI banks + prune ───────────────────────────────────────────────
echo "RUN $(date -u +%H:%M:%SZ) rebuild_strict_banks.py"
PYTHONPATH="$REPO/src" python3 "$REPO/scripts/rebuild_strict_banks.py" \
  --case-root "$ROOT" --manifest "$MANIFEST" --nthreads "$NTHREADS"

echo "RUN $(date -u +%H:%M:%SZ) honesty note + filter_bank"
PYTHONPATH="$REPO/src" python3 - <<'PY'
import json
from pathlib import Path
man_p = Path(str(Path.home()) + "/tractlab/cases/demo-leipzig-sub-010005/manifest.json")
man = json.loads(man_p.read_text())
honest = (
    "Research/preview only — not navigation. Leipzig had AP/PA SE fmap + "
    "topup/eddy; eddy QUAD warning only. Residual geometry uncertainty remains."
)
inputs = man.setdefault("inputs", {})
for k, meta in list(inputs.items()):
    if k.startswith("bank_") and isinstance(meta, dict):
        note = str(meta.get("note") or "")
        meta["note"] = (note.replace("no reverse-PE, not navigation", honest)
                        .replace("Research only — no reverse-PE, not navigation.", honest))
fb = Path("tracts/connectome/pre/freesurfer/path-to-wm-v2/wholebrain_act_ifod2_100000.tck")
tck = man_p.parent / fb
inputs["filter_bank"] = {
    "path": str(fb),
    "n_streamlines": 100000,
    "engine": "ACT iFOD2 whole-brain 100k (subset of 10M)",
    "note": "Live Explore filter corpus. FILTERED PREVIEW. " + honest,
}
man["status"] = "banks-built-unsigned-priors"
man["phase0"] = (
    "FODs + owner-signed T1 QC 2026-08-16; ACT 10M banks built; "
    "atlas/parcellation QC unsigned. run.py still on 3T clinical case."
)
man_p.write_text(json.dumps(man, indent=2) + "\n")
print("manifest honesty + filter_bank updated")
PY

echo "RUN $(date -u +%H:%M:%SZ) prune_banks.py"
PYTHONPATH="$REPO/src" python3 "$REPO/scripts/prune_banks.py" \
  --case-root "$ROOT" --manifest "$MANIFEST" \
  --fod "$NIFTI/wmfod_norm.mif"

# ── 6. Atlas priors + Schaefer parcellation (unsigned) ───────────────────────
# Pre-run full SyN in-shell (no Python 2h timeout), then atlas_prep reuses warp.
WARP_PREFIX="$ROOT/normative/mni2case_"
if [[ ! -e "${WARP_PREFIX}1Warp.nii.gz" ]]; then
  echo "RUN $(date -u +%H:%M:%SZ) antsRegistrationSyN.sh"
  "$ANTSPATH/antsRegistrationSyN.sh" -d 3 \
    -f "$NIFTI/t1_brain_dwi.nii.gz" \
    -m "$MNI_BRAIN" \
    -o "$WARP_PREFIX" \
    -n 4
fi
echo "RUN $(date -u +%H:%M:%SZ) atlas_prep tracts"
PYTHONPATH="$REPO/src" python3 -m tractlab.atlas_prep --manifest "$MANIFEST" --full-syn
echo "RUN $(date -u +%H:%M:%SZ) atlas_prep --parcellation"
PYTHONPATH="$REPO/src" python3 -m tractlab.atlas_prep --manifest "$MANIFEST" --parcellation --full-syn

echo "===== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "-- aparc_dwi --"
"$MRTRIX3/mrinfo" "$NIFTI/aparc_dwi.nii.gz" -size -spacing
echo "-- 5TT --"
"$MRTRIX3/mrinfo" "$FIVE_TT_DWI" -size -spacing
echo "-- 10M --"
"$MRTRIX3/tckinfo" "$TRACKS" -count
echo "-- banks --"
ls -l "$ROOT/tracts/bank"/*.tck | awk '{print $5, $9}'
echo "-- normative --"
ls "$ROOT/normative"/norm_*.nii.gz | wc -l
