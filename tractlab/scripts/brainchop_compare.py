#!/usr/bin/env python3
"""W3 pilot comparison: brainchop DKatlas lane vs the FreeSurfer lane.

Pure numbers-in, numbers-out. Every volume passed in is already regridded to
a common grid by the caller (scripts/brainchop_anatomy.sh, mrgrid -interp
nearest); this script only loads arrays and computes Dice. No imaging tool
calls happen here.

Fails OPEN, on purpose, whenever the inputs don't actually share a grid or a
requested ROI can't be found/compared: refusing (non-zero exit, no partial
report on disk) is the correct behavior for a mismatched pair — a Dice number
computed on a false-shared grid is worse than no number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

# mrtrix 5TT channel order: 0 cortical GM, 1 subcortical GM, 2 WM, 3 CSF, 4 pathological.
FIVE_TT_CHANNELS = ["cortical_gm", "subcortical_gm", "wm", "csf", "pathological"]

# FreeSurfer aparc+aseg ids for the three region comparisons the pilot asked for.
REGION_IDS = {
    "precentral": {"lh": 1024, "rh": 2024},
    "superiorfrontal": {"lh": 1028, "rh": 2028},
    "parsopercularis_plus_parstriangularis": {"lh": [1018, 1020], "rh": [2018, 2020]},
}

# ROI file (tracts/roi/) <-> FreeSurfer aparc label this pilot compares it against.
# dil1 used where available (minimal dilation, closest to the raw gyral outline);
# CST has no dil1 variant, so dil2 is used and reported as such.
ROI_TO_LABEL = {
    "fat_r_sfg_dil1.nii.gz": ("ctx-rh-superiorfrontal", 2028, "dil1"),
    "fat_l_sfg_dil1.nii.gz": ("ctx-lh-superiorfrontal", 1028, "dil1"),
    "fat_r_ifg_dil1.nii.gz": ("ctx-rh-parsopercularis", 2018, "dil1"),
    "fat_l_ifg_dil1.nii.gz": ("ctx-lh-parsopercularis", 1018, "dil1"),
    "cst_r_motor_dil2.nii.gz": ("ctx-rh-precentral", 2024, "dil2 (no dil1 variant exists)"),
    "cst_l_motor_dil2.nii.gz": ("ctx-lh-precentral", 1024, "dil2 (no dil1 variant exists)"),
}

AFFINE_ATOL = 1e-3


class ComparisonError(RuntimeError):
    """A comparison was refused: mismatched/missing input, never a partial result."""


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return None  # both empty: undefined, not a perfect/zero score
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _load_checked(path_a: str, path_b: str, label: str):
    """Load two images and refuse unless they share a real common grid.

    Equal shape is not enough — two images can have identical voxel counts
    while covering different real-world extents (different voxel size,
    origin, or orientation). That is a FALSE-shared grid: voxel (i,j,k) in
    one does not correspond to voxel (i,j,k) in the other, and a voxel-wise
    Dice on that pair is meaningless. Refuse rather than silently compute it.
    """
    img_a = nib.load(path_a)
    img_b = nib.load(path_b)
    a = np.asanyarray(img_a.dataobj)
    b = np.asanyarray(img_b.dataobj)
    if a.shape != b.shape:
        raise ComparisonError(f"{label}: shape mismatch {a.shape} vs {b.shape} ({path_a} vs {path_b})")
    if not np.allclose(img_a.affine, img_b.affine, atol=AFFINE_ATOL):
        raise ComparisonError(
            f"{label}: shapes match {a.shape} but affines differ beyond {AFFINE_ATOL} tolerance "
            f"({path_a} vs {path_b}) — refusing a voxel-wise comparison on a false-shared grid"
        )
    return a, b


def five_tt_dice(brainchop_path: str, freesurfer_path: str) -> dict:
    bc, fs = _load_checked(brainchop_path, freesurfer_path, "5TT")
    out = {}
    for i, name in enumerate(FIVE_TT_CHANNELS):
        bc_bin = bc[..., i] > 0.5
        fs_bin = fs[..., i] > 0.5
        out[name] = {
            "dice": dice(bc_bin, fs_bin),
            "brainchop_voxels": int(bc_bin.sum()),
            "freesurfer_voxels": int(fs_bin.sum()),
        }
    return out


def region_dice(brainchop_aparc_path: str, fs_aparc_path: str) -> dict:
    bc, fs = _load_checked(brainchop_aparc_path, fs_aparc_path, "aparc region")
    out = {}
    for region, hemis in REGION_IDS.items():
        out[region] = {}
        for hemi, ids in hemis.items():
            id_list = ids if isinstance(ids, list) else [ids]
            bc_bin = np.isin(bc, id_list)
            fs_bin = np.isin(fs, id_list)
            out[region][hemi] = {
                "dice": dice(bc_bin, fs_bin),
                "freesurfer_ids": id_list,
                "brainchop_voxels": int(bc_bin.sum()),
                "freesurfer_voxels": int(fs_bin.sum()),
            }
        # bilateral combined, as literally asked for in the pilot spec
        all_ids = [i for ids in hemis.values() for i in (ids if isinstance(ids, list) else [ids])]
        bc_bin = np.isin(bc, all_ids)
        fs_bin = np.isin(fs, all_ids)
        out[region]["bilateral"] = {
            "dice": dice(bc_bin, fs_bin),
            "freesurfer_ids": all_ids,
            "brainchop_voxels": int(bc_bin.sum()),
            "freesurfer_voxels": int(fs_bin.sum()),
        }
    return out


def roi_dice(brainchop_aparc_dwi_path: str, roi_dir: str) -> dict:
    """Refuses (raises ComparisonError) on the FIRST missing or mismatched ROI
    rather than recording a per-item error and returning a partial report —
    an incomplete ROI comparison is not a usable pilot result."""
    bc_img = nib.load(brainchop_aparc_dwi_path)
    bc = np.asanyarray(bc_img.dataobj)
    out = {}
    for roi_file, (label_name, fs_id, dilation) in ROI_TO_LABEL.items():
        roi_path = os.path.join(roi_dir, roi_file)
        if not os.path.exists(roi_path):
            raise ComparisonError(f"ROI file missing: {roi_path} (needed for {label_name})")
        roi_img = nib.load(roi_path)
        roi = np.asanyarray(roi_img.dataobj)
        if roi.shape != bc.shape:
            raise ComparisonError(
                f"ROI {roi_file}: shape {roi.shape} != brainchop-dwi shape {bc.shape}"
            )
        if not np.allclose(roi_img.affine, bc_img.affine, atol=AFFINE_ATOL):
            raise ComparisonError(
                f"ROI {roi_file}: shapes match {roi.shape} but affine differs from "
                f"brainchop-dwi beyond {AFFINE_ATOL} tolerance"
            )
        roi_bin = roi > 0
        bc_bin = bc == fs_id
        out[roi_file] = {
            "compared_to_label": label_name,
            "freesurfer_id": fs_id,
            "dilation": dilation,
            "dice": dice(bc_bin, roi_bin),
            "roi_voxels": int(roi_bin.sum()),
            "brainchop_label_voxels": int(bc_bin.sum()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--five-tt-brainchop", required=True)
    ap.add_argument("--five-tt-freesurfer", required=True)
    ap.add_argument("--aparc-brainchop-fs-grid", required=True)
    ap.add_argument("--aparc-freesurfer", required=True)
    ap.add_argument("--aparc-brainchop-dwi", required=True)
    ap.add_argument("--roi-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        report = {
            "five_tt_dice": five_tt_dice(args.five_tt_brainchop, args.five_tt_freesurfer),
            "region_dice_fs_grid": region_dice(args.aparc_brainchop_fs_grid, args.aparc_freesurfer),
            "roi_dice_dwi_grid": roi_dice(args.aparc_brainchop_dwi, args.roi_dir),
        }
    except ComparisonError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
