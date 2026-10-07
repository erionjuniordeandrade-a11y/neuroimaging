#!/usr/bin/env bash
# W3 pilot: brainchop DKatlas fast-anatomy lane, end-to-end on demo-leipzig-sub-010005.
# Mirrors the house 5TT recipe in scripts/run_demo_d0_banks.sh (5ttgen freesurfer
# -lut -nocrop -sgm_amyg_hipp, then 5ttedit -wm on the extracted "path" tissue
# channel) but starting from brainchop's DKatlas segmentation instead of
# recon-all. Writes only under cases/demo-leipzig-sub-010005/work/brainchop/.
# Nothing here enters the manifest — this is a pilot measurement, not a lane
# switch (docs/ARCH-annex-D-verification-roadmap-slice.md, CONTEXT.md).
set -euo pipefail

NTHREADS="${NTHREADS:-8}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$REPO/cases/demo-leipzig-sub-010005"
WORK="$ROOT/work/brainchop"
T1="$ROOT/work/t1reg/t1.nii.gz"
FS_ANAT="$ROOT/work/banks/anat"
FS_AGG_RAW="$FS_ANAT/5tt_freesurfer_fs_raw.mif"
FS_APARC="$FS_ANAT/aparc_fs.mif"
NIFTI="$ROOT/nifti"
T12B0="$NIFTI/t12b0.mat"
B0="$NIFTI/b0.nii.gz"
ROI_DIR="$ROOT/tracts/roi"
LOG="$WORK/brainchop_anatomy.log"

export FSLDIR="${FSLDIR:-$HOME/fsl}"
export FREESURFER_HOME="${FREESURFER_HOME:-$HOME/freesurfer}"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$HOME/mrtrix3/bin:$FREESURFER_HOME/bin:$PATH"

MRTRIX3="$HOME/mrtrix3/bin"
FS_LUT="$FREESURFER_HOME/FreeSurferColorLUT.txt"
PY="$HOME/fsl/bin/python3"

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) brainchop_anatomy.sh pid=$$ ====="

[[ -f "$T1" ]] || { echo "FAIL missing $T1"; exit 1; }
[[ -f "$FS_AGG_RAW" ]] || { echo "FAIL missing $FS_AGG_RAW (run run_demo_d0_banks.sh first)"; exit 1; }
[[ -f "$FS_APARC" ]] || { echo "FAIL missing $FS_APARC"; exit 1; }
[[ -x "$MRTRIX3/5ttgen" ]] || { echo "FAIL 5ttgen not found"; exit 1; }

t0=$(date +%s)

# ── 1. DKatlas segmentation + FreeSurfer-id remap (src/tractlab/brainchop_lane.py) ──
echo "RUN $(date -u +%H:%M:%SZ) run_dkatlas($T1)"
PYTHONPATH="$REPO/src" "$PY" - "$T1" "$WORK" <<'PY'
import sys, json, time
from tractlab.brainchop_lane import run_dkatlas, Outcome

t1_path, out_dir = sys.argv[1], sys.argv[2]
t0 = time.time()
result = run_dkatlas(t1_path, out_dir, timeout_s=300.0)
wall = time.time() - t0
print(f"run_dkatlas outcome={result.outcome.value} wall_s={wall:.2f}")
if result.outcome is not Outcome.OK:
    print(f"FAIL run_dkatlas: {result.warning}")
    sys.exit(1)
print(f"aparc_path={result.aparc_path}")
print(json.dumps(result.sidecar, indent=2))
PY

APARC_BRAINCHOP="$WORK/aparc_brainchop.nii.gz"
[[ -f "$APARC_BRAINCHOP" ]] || { echo "FAIL $APARC_BRAINCHOP missing after run_dkatlas"; exit 1; }

# ── 2. 5TT from the brainchop aparc, mirroring run_demo_d0_banks.sh exactly ──
FIVE_TT_RAW="$WORK/5tt_brainchop_fs_raw.mif"
FIVE_TT_CAT5="$WORK/5tt_brainchop_auto_cat5.mif"
FIVE_TT="$WORK/5tt_brainchop.mif"

echo "RUN $(date -u +%H:%M:%SZ) 5ttgen freesurfer (brainchop aparc)"
t_5ttgen0=$(date +%s)
"$MRTRIX3/5ttgen" freesurfer "$APARC_BRAINCHOP" "$FIVE_TT_RAW" \
  -lut "$FS_LUT" -nocrop -sgm_amyg_hipp -nthreads "$NTHREADS" -force -quiet
t_5ttgen1=$(date +%s)
echo "5ttgen freesurfer wall_s=$((t_5ttgen1 - t_5ttgen0))"

echo "RUN $(date -u +%H:%M:%SZ) mrconvert -coord 3 4 (path tissue channel)"
"$MRTRIX3/mrconvert" "$FIVE_TT_RAW" -coord 3 4 "$FIVE_TT_CAT5" -force -quiet

echo "RUN $(date -u +%H:%M:%SZ) 5ttedit -wm (path-to-wm, house recipe)"
"$MRTRIX3/5ttedit" "$FIVE_TT_RAW" "$FIVE_TT" -wm "$FIVE_TT_CAT5" -force

echo "RUN $(date -u +%H:%M:%SZ) 5ttcheck"
"$MRTRIX3/5ttcheck" "$FIVE_TT"

# Receipt: inputs + outputs + sha256 + mrtrix version + exact argv for this step.
FIVE_TT_RECEIPT="$WORK/5tt_receipt.json"
"$PY" - "$FIVE_TT_RECEIPT" "$APARC_BRAINCHOP" "$FIVE_TT_RAW" "$FIVE_TT_CAT5" "$FIVE_TT" \
  "$MRTRIX3/5ttgen" "$MRTRIX3/mrconvert" "$MRTRIX3/5ttedit" "$MRTRIX3/5ttcheck" \
  "$FS_LUT" "$NTHREADS" <<'PY'
import hashlib
import json
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


(out_path, aparc_brainchop, five_tt_raw, five_tt_cat5, five_tt,
 ttgen_bin, mrconvert_bin, ttedit_bin, ttcheck_bin, fs_lut, nthreads) = sys.argv[1:12]

version = subprocess.run([ttgen_bin, "-version"], capture_output=True, text=True).stdout.splitlines()[0]

receipt = {
    "step": "5tt_from_brainchop_aparc",
    "mrtrix_version": version,
    "commands": [
        [ttgen_bin, "freesurfer", aparc_brainchop, five_tt_raw,
         "-lut", fs_lut, "-nocrop", "-sgm_amyg_hipp", "-nthreads", nthreads, "-force", "-quiet"],
        [mrconvert_bin, five_tt_raw, "-coord", "3", "4", five_tt_cat5, "-force", "-quiet"],
        [ttedit_bin, five_tt_raw, five_tt, "-wm", five_tt_cat5, "-force"],
        [ttcheck_bin, five_tt],
    ],
    "inputs": {"aparc_brainchop": {"path": aparc_brainchop, "sha256": sha256(aparc_brainchop)}},
    "outputs": {
        "5tt_brainchop_fs_raw": {"path": five_tt_raw, "sha256": sha256(five_tt_raw)},
        "5tt_brainchop_auto_cat5": {"path": five_tt_cat5, "sha256": sha256(five_tt_cat5)},
        "5tt_brainchop": {"path": five_tt, "sha256": sha256(five_tt)},
    },
}
with open(out_path, "w") as f:
    json.dump(receipt, f, indent=2)
print(f"wrote {out_path}")
PY

t1_end=$(date +%s)
echo "TOTAL brainchop_lane wall_s=$((t1_end - t0))"

# ── 3. Comparison vs the FreeSurfer lane (numbers only from this run) ──
echo "RUN $(date -u +%H:%M:%SZ) comparison (5TT Dice, aparc-region Dice, DWI-grid ROI Dice)"

# 3a. Regrid brainchop 5TT onto the FS 5TT grid (nearest, explicit strides).
FIVE_TT_ON_FS="$WORK/5tt_brainchop_on_fs_grid.mif"
"$MRTRIX3/mrgrid" "$FIVE_TT" regrid -template "$FS_AGG_RAW" -interp nearest "$FIVE_TT_ON_FS" -force -quiet
FIVE_TT_ON_FS_NII="$WORK/5tt_brainchop_on_fs_grid.nii.gz"
FS_RAW_NII="$WORK/5tt_freesurfer_fs_raw_std.nii.gz"
"$MRTRIX3/mrconvert" "$FIVE_TT_ON_FS" -strides 1,2,3,4 "$FIVE_TT_ON_FS_NII" -force -quiet
"$MRTRIX3/mrconvert" "$FS_AGG_RAW" -strides 1,2,3,4 "$FS_RAW_NII" -force -quiet

# 3b. Regrid brainchop aparc onto the FS aparc grid (nearest) for region Dice.
APARC_BRAINCHOP_ON_FS="$WORK/aparc_brainchop_on_fs_grid.mif"
"$MRTRIX3/mrgrid" "$APARC_BRAINCHOP" regrid -template "$FS_APARC" -interp nearest "$APARC_BRAINCHOP_ON_FS" -force -quiet
APARC_BRAINCHOP_ON_FS_NII="$WORK/aparc_brainchop_on_fs_grid.nii.gz"
FS_APARC_NII="$WORK/aparc_fs_std.nii.gz"
"$MRTRIX3/mrconvert" "$APARC_BRAINCHOP_ON_FS" -strides 1,2,3 "$APARC_BRAINCHOP_ON_FS_NII" -force -quiet
"$MRTRIX3/mrconvert" "$FS_APARC" -strides 1,2,3 "$FS_APARC_NII" -force -quiet

# 3c. brainchop aparc -> DWI grid (t12b0.mat, nearest) for ROI Dice.
APARC_BRAINCHOP_DWI="$WORK/aparc_brainchop_dwi.nii.gz"
"$FSLDIR/bin/flirt" -in "$APARC_BRAINCHOP" -ref "$B0" -applyxfm -init "$T12B0" \
  -interp nearestneighbour -out "$APARC_BRAINCHOP_DWI"

REPORT_JSON="$WORK/comparison_report.json"
PYTHONPATH="$REPO/src" "$PY" "$REPO/scripts/brainchop_compare.py" \
  --five-tt-brainchop "$FIVE_TT_ON_FS_NII" \
  --five-tt-freesurfer "$FS_RAW_NII" \
  --aparc-brainchop-fs-grid "$APARC_BRAINCHOP_ON_FS_NII" \
  --aparc-freesurfer "$FS_APARC_NII" \
  --aparc-brainchop-dwi "$APARC_BRAINCHOP_DWI" \
  --roi-dir "$ROI_DIR" \
  --out "$REPORT_JSON"

echo "wrote $REPORT_JSON"

# Receipt: inputs + sha256 + mrtrix version + exact argv for this step
# (the regrids/flirt/compare commands, not the Dice numbers themselves —
# those live in $REPORT_JSON).
COMPARISON_RECEIPT="$WORK/comparison_receipt.json"
"$PY" - "$COMPARISON_RECEIPT" \
  "$FIVE_TT" "$FS_AGG_RAW" "$FIVE_TT_ON_FS" "$FIVE_TT_ON_FS_NII" "$FS_RAW_NII" \
  "$APARC_BRAINCHOP" "$FS_APARC" "$APARC_BRAINCHOP_ON_FS" "$APARC_BRAINCHOP_ON_FS_NII" "$FS_APARC_NII" \
  "$T12B0" "$B0" "$APARC_BRAINCHOP_DWI" "$ROI_DIR" \
  "$MRTRIX3/mrgrid" "$MRTRIX3/mrconvert" "$FSLDIR/bin/flirt" "$REPO/scripts/brainchop_compare.py" \
  "$REPORT_JSON" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


(out_path, five_tt, fs_agg_raw, five_tt_on_fs, five_tt_on_fs_nii, fs_raw_nii,
 aparc_bc, fs_aparc, aparc_bc_on_fs, aparc_bc_on_fs_nii, fs_aparc_nii,
 t12b0, b0, aparc_bc_dwi, roi_dir,
 mrgrid_bin, mrconvert_bin, flirt_bin, compare_py, report_json) = sys.argv[1:21]

mrtrix_version = subprocess.run([mrgrid_bin, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
roi_files = sorted(p.name for p in Path(roi_dir).glob("*.nii.gz"))

inputs = {
    "5tt_brainchop": {"path": five_tt, "sha256": sha256(five_tt)},
    "5tt_freesurfer_fs_raw": {"path": fs_agg_raw, "sha256": sha256(fs_agg_raw)},
    "aparc_brainchop": {"path": aparc_bc, "sha256": sha256(aparc_bc)},
    "aparc_freesurfer": {"path": fs_aparc, "sha256": sha256(fs_aparc)},
    "t12b0_transform": {"path": t12b0, "sha256": sha256(t12b0)},
    "b0_reference": {"path": b0, "sha256": sha256(b0)},
}
outputs = {
    "5tt_brainchop_on_fs_grid_nii": {"path": five_tt_on_fs_nii, "sha256": sha256(five_tt_on_fs_nii)},
    "5tt_freesurfer_fs_raw_std_nii": {"path": fs_raw_nii, "sha256": sha256(fs_raw_nii)},
    "aparc_brainchop_on_fs_grid_nii": {"path": aparc_bc_on_fs_nii, "sha256": sha256(aparc_bc_on_fs_nii)},
    "aparc_fs_std_nii": {"path": fs_aparc_nii, "sha256": sha256(fs_aparc_nii)},
    "aparc_brainchop_dwi": {"path": aparc_bc_dwi, "sha256": sha256(aparc_bc_dwi)},
    "comparison_report": {"path": report_json, "sha256": sha256(report_json)},
}

receipt = {
    "step": "comparison_regrid_and_dice",
    "mrtrix_version": mrtrix_version,
    "roi_files_used": roi_files,
    "commands": [
        [mrgrid_bin, five_tt, "regrid", "-template", fs_agg_raw, "-interp", "nearest", five_tt_on_fs, "-force", "-quiet"],
        [mrconvert_bin, five_tt_on_fs, "-strides", "1,2,3,4", five_tt_on_fs_nii, "-force", "-quiet"],
        [mrconvert_bin, fs_agg_raw, "-strides", "1,2,3,4", fs_raw_nii, "-force", "-quiet"],
        [mrgrid_bin, aparc_bc, "regrid", "-template", fs_aparc, "-interp", "nearest", aparc_bc_on_fs, "-force", "-quiet"],
        [mrconvert_bin, aparc_bc_on_fs, "-strides", "1,2,3", aparc_bc_on_fs_nii, "-force", "-quiet"],
        [mrconvert_bin, fs_aparc, "-strides", "1,2,3", fs_aparc_nii, "-force", "-quiet"],
        [flirt_bin, "-in", aparc_bc, "-ref", b0, "-applyxfm", "-init", t12b0,
         "-interp", "nearestneighbour", "-out", aparc_bc_dwi],
        ["python3", compare_py,
         "--five-tt-brainchop", five_tt_on_fs_nii, "--five-tt-freesurfer", fs_raw_nii,
         "--aparc-brainchop-fs-grid", aparc_bc_on_fs_nii, "--aparc-freesurfer", fs_aparc_nii,
         "--aparc-brainchop-dwi", aparc_bc_dwi, "--roi-dir", roi_dir, "--out", report_json],
    ],
    "inputs": inputs,
    "outputs": outputs,
}
with open(out_path, "w") as f:
    json.dump(receipt, f, indent=2)
print(f"wrote {out_path}")
PY

t2_end=$(date +%s)
echo "TOTAL comparison wall_s=$((t2_end - t1_end))"
echo "TOTAL script wall_s=$((t2_end - t0))"
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) done ====="
