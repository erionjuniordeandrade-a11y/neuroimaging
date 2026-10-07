#!/usr/bin/env python3
"""Rebuild primary banks with slightly stricter multi-ROI from aparc + 10M ACT.

Stricter than the dil2/dil3 liberality used previously:
  - association / OR ROIs: binary dilation **1** (was 2)
  - CST motor: dilation **2** (was 3); pons kept at dil3
  - unilateral banks: exclude contralateral cortex (unchanged)
  - association banks: also exclude brainstem (16) + cerebellar GM/WM
  - slightly higher minlength per family
  - optional lesion exclude for display purity (off by default — lesion can abut true CST)

Does not re-track. Uses tckedit on the whole-brain ACT 10M corpus.
Writes bank/*.tck, then refreshes bank/raw/ for prune_banks.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tractlab import derivation as dv  # noqa: E402 — floor label is the one source of truth

TCKEDIT = os.path.expanduser("~/mrtrix3/bin/tckedit")
TCKINFO = os.path.expanduser("~/mrtrix3/bin/tckinfo")

# FreeSurfer aparc+aseg labels (Desikan)
L = {
    "thal_l": 10,
    "thal_r": 49,
    "bs": 16,
    "cereb_l": 7,
    "cereb_l_wm": 8,
    "cereb_r": 46,
    "cereb_r_wm": 47,
    "precentral_l": 1024,
    "precentral_r": 2024,
    "sfg_l": 1028,
    "sfg_r": 2028,
    "pop_l": 1018,
    "pop_r": 2018,
    "ptr_l": 1020,
    "ptr_r": 2020,
    "cmf_l": 1003,
    "cmf_r": 2003,
    "smg_l": 1031,
    "smg_r": 2031,
    # SLF-I / SLF-II Desikan waypoints (Thiebaut de Schotten-style cortical ends)
    "suppar_l": 1029,  # superiorparietal
    "suppar_r": 2029,
    "precun_l": 1025,  # precuneus
    "precun_r": 2025,
    "infpar_l": 1008,  # inferiorparietal (angular / IPL)
    "infpar_r": 2008,
    "rmf_l": 1027,  # rostralmiddlefrontal
    "rmf_r": 2027,
    "latorb_l": 1012,
    "latorb_r": 2012,
    "medorb_l": 1014,
    "medorb_r": 2014,
    "porb_l": 1019,
    "porb_r": 2019,
    "tempole_l": 1033,
    "tempole_r": 2033,
    "cuneus_l": 1005,
    "cuneus_r": 2005,
    "latocc_l": 1011,
    "latocc_r": 2011,
    "lingual_l": 1013,
    "lingual_r": 2013,
    "pericalc_l": 1021,
    "pericalc_r": 2021,
    "midtemp_l": 1015,
    "midtemp_r": 2015,
    "inftemp_l": 1009,
    "inftemp_r": 2009,
    "fus_l": 1007,
    "fus_r": 2007,
    "cacc_l": 1002,
    "cacc_r": 2002,
    "isthmus_l": 1010,
    "isthmus_r": 2010,
    "pcc_l": 1023,
    "pcc_r": 2023,
    "racc_l": 1026,
    "racc_r": 2026,
    "cc": [251, 252, 253, 254, 255],
}


def dilate(mask: np.ndarray, it: int) -> np.ndarray:
    if it <= 0:
        return mask.astype(bool)
    return ndimage.binary_dilation(mask, iterations=it)


def save_mask(path: Path, mask: np.ndarray, aff) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(path))


def build_rois(case: Path) -> dict[str, Path]:
    """Write stricter dil1/dil2 ROIs; return name→path."""
    aparc = case / "nifti/aparc_dwi.nii.gz"
    img = nib.load(str(aparc))
    lab = np.asanyarray(img.dataobj).astype(np.int32)
    aff = img.affine
    roi = case / "tracts/roi"
    out: dict[str, Path] = {}

    def labels(*ids):
        flat = []
        for x in ids:
            if isinstance(x, (list, tuple)):
                flat.extend(x)
            else:
                flat.append(x)
        return np.isin(lab, flat)

    # --- brainstem / cerebellum excludes (shared) ---
    bs_cereb = labels(L["bs"], L["cereb_l"], L["cereb_l_wm"], L["cereb_r"], L["cereb_r_wm"])
    p = roi / "exclude_bs_cereb.nii.gz"
    save_mask(p, bs_cereb, aff)
    out["exclude_bs_cereb"] = p

    cc = labels(*L["cc"])
    p = roi / "exclude_cc.nii.gz"
    save_mask(p, dilate(cc, 1), aff)
    out["exclude_cc"] = p

    # Hemisphere cortex = aparc 1001–1035 / 2001–2035, undilated (matches the
    # reference case's masks voxel for voxel). Kept when present; derived for a
    # freshly ingested case.
    for side, lo in (("l", 1001), ("r", 2001)):
        p = roi / f"hemi_{side}_cortex.nii.gz"
        if not p.is_file():
            save_mask(p, (lab >= lo) & (lab <= lo + 34), aff)
        out[f"hemi_{side}"] = p

    # CST motor dil2 (stricter than dil3)
    for side, pre, name in [
        ("l", L["precentral_l"], "cst_l_motor_dil2.nii.gz"),
        ("r", L["precentral_r"], "cst_r_motor_dil2.nii.gz"),
    ]:
        p = roi / name
        save_mask(p, dilate(lab == pre, 2), aff)
        out[f"cst_{side}_motor"] = p
    # reuse existing pons dil3 if present
    pons_r = roi / "cst_r_pons_dil3.nii.gz"
    if not pons_r.is_file():
        # fallback: brainstem anterior-ish — use BS dilated small
        save_mask(pons_r, dilate(lab == L["bs"], 2), aff)
    out["cst_r_pons"] = pons_r
    # Midline pons is shared — do not dilate whole brainstem as a left proxy
    out["cst_l_pons"] = pons_r

    # FAT: SFG + POP/PTR dil1 (strict) + dil2 (soft / peri-lesional tolerant)
    for side, sfg, pop, ptr in [
        ("l", L["sfg_l"], L["pop_l"], L["ptr_l"]),
        ("r", L["sfg_r"], L["pop_r"], L["ptr_r"]),
    ]:
        p1 = roi / f"fat_{side}_sfg_dil1.nii.gz"
        p2 = roi / f"fat_{side}_ifg_dil1.nii.gz"
        save_mask(p1, dilate(lab == sfg, 1), aff)
        save_mask(p2, dilate(labels(pop, ptr), 1), aff)
        out[f"fat_{side}_sfg"] = p1
        out[f"fat_{side}_ifg"] = p2
        p1s = roi / f"fat_{side}_sfg_dil2.nii.gz"
        p2s = roi / f"fat_{side}_ifg_dil2.nii.gz"
        save_mask(p1s, dilate(lab == sfg, 2), aff)
        save_mask(p2s, dilate(labels(pop, ptr), 2), aff)
        out[f"fat_{side}_sfg_soft"] = p1s
        out[f"fat_{side}_ifg_soft"] = p2s

    # SLF-I (dorsal): SFG ↔ superior parietal + precuneus dil1 / dil2 soft
    for side, sfg, suppar, precun in [
        ("l", L["sfg_l"], L["suppar_l"], L["precun_l"]),
        ("r", L["sfg_r"], L["suppar_r"], L["precun_r"]),
    ]:
        p1 = roi / f"slf1_{side}_front_dil1.nii.gz"
        p2 = roi / f"slf1_{side}_pari_dil1.nii.gz"
        save_mask(p1, dilate(lab == sfg, 1), aff)
        save_mask(p2, dilate(labels(suppar, precun), 1), aff)
        out[f"slf1_{side}_front"] = p1
        out[f"slf1_{side}_pari"] = p2
        p1s = roi / f"slf1_{side}_front_dil2.nii.gz"
        p2s = roi / f"slf1_{side}_pari_dil2.nii.gz"
        save_mask(p1s, dilate(lab == sfg, 2), aff)
        save_mask(p2s, dilate(labels(suppar, precun), 2), aff)
        out[f"slf1_{side}_front_soft"] = p1s
        out[f"slf1_{side}_pari_soft"] = p2s

    # SLF-II (middle): RMF+CMF ↔ inferior parietal dil1 / dil2 soft
    for side, rmf, cmf, infpar in [
        ("l", L["rmf_l"], L["cmf_l"], L["infpar_l"]),
        ("r", L["rmf_r"], L["cmf_r"], L["infpar_r"]),
    ]:
        p1 = roi / f"slf2_{side}_front_dil1.nii.gz"
        p2 = roi / f"slf2_{side}_pari_dil1.nii.gz"
        save_mask(p1, dilate(labels(rmf, cmf), 1), aff)
        save_mask(p2, dilate(lab == infpar, 1), aff)
        out[f"slf2_{side}_front"] = p1
        out[f"slf2_{side}_pari"] = p2
        p1s = roi / f"slf2_{side}_front_dil2.nii.gz"
        p2s = roi / f"slf2_{side}_pari_dil2.nii.gz"
        save_mask(p1s, dilate(labels(rmf, cmf), 2), aff)
        save_mask(p2s, dilate(lab == infpar, 2), aff)
        out[f"slf2_{side}_front_soft"] = p1s
        out[f"slf2_{side}_pari_soft"] = p2s

    # SLF-III (ventral): SMG + (CMF+POP+PTR) dil1 / dil2 soft
    for side, smg, cmf, pop, ptr in [
        ("l", L["smg_l"], L["cmf_l"], L["pop_l"], L["ptr_l"]),
        ("r", L["smg_r"], L["cmf_r"], L["pop_r"], L["ptr_r"]),
    ]:
        p1 = roi / f"slf3_{side}_smg_dil1.nii.gz"
        p2 = roi / f"slf3_{side}_frontal_dil1.nii.gz"
        save_mask(p1, dilate(lab == smg, 1), aff)
        save_mask(p2, dilate(labels(cmf, pop, ptr), 1), aff)
        out[f"slf3_{side}_smg"] = p1
        out[f"slf3_{side}_front"] = p2
        p1s = roi / f"slf3_{side}_smg_dil2.nii.gz"
        p2s = roi / f"slf3_{side}_frontal_dil2.nii.gz"
        save_mask(p1s, dilate(lab == smg, 2), aff)
        save_mask(p2s, dilate(labels(cmf, pop, ptr), 2), aff)
        out[f"slf3_{side}_smg_soft"] = p1s
        out[f"slf3_{side}_front_soft"] = p2s

    # IFOF: occipital visual set + orbital/ventral frontal dil1
    for side, vis, front in [
        (
            "l",
            [L["cuneus_l"], L["latocc_l"], L["lingual_l"], L["pericalc_l"]],
            [L["latorb_l"], L["medorb_l"], L["porb_l"]],
        ),
        (
            "r",
            [L["cuneus_r"], L["latocc_r"], L["lingual_r"], L["pericalc_r"]],
            [L["latorb_r"], L["medorb_r"], L["porb_r"]],
        ),
    ]:
        p1 = roi / f"ifof_{side}_occ_dil1.nii.gz"
        p2 = roi / f"ifof_{side}_front_dil1.nii.gz"
        save_mask(p1, dilate(labels(*vis), 1), aff)
        save_mask(p2, dilate(labels(*front), 1), aff)
        out[f"ifof_{side}_occ"] = p1
        out[f"ifof_{side}_front"] = p2

    # UF: temporal pole + orbital dil1
    for side, temp, orb in [
        ("l", L["tempole_l"], [L["latorb_l"], L["medorb_l"], L["porb_l"]]),
        ("r", L["tempole_r"], [L["latorb_r"], L["medorb_r"], L["porb_r"]]),
    ]:
        p1 = roi / f"uf_{side}_temp_dil1.nii.gz"
        p2 = roi / f"uf_{side}_orb_dil1.nii.gz"
        save_mask(p1, dilate(lab == temp, 1), aff)
        save_mask(p2, dilate(labels(*orb), 1), aff)
        out[f"uf_{side}_temp"] = p1
        out[f"uf_{side}_orb"] = p2

    # Cingulum: ant (rACC+cACC) + post (PCC+isthmus) dil1
    for side, ant, post in [
        ("l", [L["racc_l"], L["cacc_l"]], [L["pcc_l"], L["isthmus_l"]]),
        ("r", [L["racc_r"], L["cacc_r"]], [L["pcc_r"], L["isthmus_r"]]),
    ]:
        p1 = roi / f"cing_{side}_ant_dil1.nii.gz"
        p2 = roi / f"cing_{side}_post_dil1.nii.gz"
        save_mask(p1, dilate(labels(*ant), 1), aff)
        save_mask(p2, dilate(labels(*post), 1), aff)
        out[f"cing_{side}_ant"] = p1
        out[f"cing_{side}_post"] = p2

    # OR: thalamus dil1 + visual dil1; temporal dil1 for Meyer
    for side, thal, vis, temp in [
        (
            "l",
            L["thal_l"],
            [L["cuneus_l"], L["latocc_l"], L["lingual_l"], L["pericalc_l"]],
            [L["midtemp_l"], L["inftemp_l"], L["fus_l"]],
        ),
        (
            "r",
            L["thal_r"],
            [L["cuneus_r"], L["latocc_r"], L["lingual_r"], L["pericalc_r"]],
            [L["midtemp_r"], L["inftemp_r"], L["fus_r"]],
        ),
    ]:
        p1 = roi / f"or_{side}_thal_dil1.nii.gz"
        p2 = roi / f"or_{side}_vis_dil1.nii.gz"
        p3 = roi / f"or_{side}_temp_dil1.nii.gz"
        save_mask(p1, dilate(lab == thal, 1), aff)
        save_mask(p2, dilate(labels(*vis), 1), aff)
        save_mask(p3, dilate(labels(*temp), 1), aff)
        out[f"or_{side}_thal"] = p1
        out[f"or_{side}_vis"] = p2
        out[f"or_{side}_temp"] = p3

    # Forceps: keep dil1 frontal/occipital bilat + CC body dil0
    for side, front, occ in [
        (
            "l",
            [L["sfg_l"], L["cmf_l"], L["racc_l"]],
            [L["cuneus_l"], L["latocc_l"], L["lingual_l"], L["pericalc_l"]],
        ),
        (
            "r",
            [L["sfg_r"], L["cmf_r"], L["racc_r"]],
            [L["cuneus_r"], L["latocc_r"], L["lingual_r"], L["pericalc_r"]],
        ),
    ]:
        pf = roi / f"cc_forceps_minor_front_{side}_dil1.nii.gz"
        po = roi / f"cc_forceps_major_occ_{side}_dil1.nii.gz"
        save_mask(pf, dilate(labels(*front), 1), aff)
        save_mask(po, dilate(labels(*occ), 1), aff)
        out[f"fmin_front_{side}"] = pf
        out[f"fmaj_occ_{side}"] = po

    p = roi / "cc_body_dil0.nii.gz"
    save_mask(p, labels(*L["cc"]), aff)
    out["cc_body"] = p

    # Forceps minor: SFG bilat + genu (CC_Anterior + Mid_Anterior)
    for side, sfg in [("l", L["sfg_l"]), ("r", L["sfg_r"])]:
        p = roi / f"cc_fmin_sfg_{side}_dil1.nii.gz"
        save_mask(p, dilate(lab == sfg, 1), aff)
        out[f"fmin_sfg_{side}"] = p
    p = roi / "cc_genu_dil1.nii.gz"
    save_mask(p, dilate(labels(254, 255), 1), aff)
    out["cc_genu"] = p

    print("ROIs written:")
    for k, v in sorted(out.items()):
        if v.is_file():
            n = int((np.asanyarray(nib.load(str(v)).dataobj) > 0).sum())
            print(f"  {k:22s} {v.name:40s} vx={n}")
    return out


def tck_count(path: Path) -> int:
    out = subprocess.run(
        [TCKINFO, str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    for line in out.stdout.splitlines():
        if line.strip().startswith("count:"):
            return int(line.split(":", 1)[1])
    return 0


def extract(
    corpus: Path,
    out_tck: Path,
    *,
    includes: list[Path],
    excludes: list[Path],
    minlength: float,
    maxlength: float,
    nthreads: int = 8,
) -> int:
    argv = [
        TCKEDIT,
        str(corpus),
        str(out_tck),
        "-minlength",
        str(minlength),
        "-maxlength",
        str(maxlength),
        "-force",
        "-nthreads",
        str(nthreads),
        "-quiet",
    ]
    for p in includes:
        argv.extend(["-include", str(p)])
    for p in excludes:
        argv.extend(["-exclude", str(p)])
    print(" ", " ".join(argv[0:3]), f"+{len(includes)}inc -{len(excludes)}exc L=[{minlength},{maxlength}]")
    subprocess.run(argv, check=True, timeout=900)
    return tck_count(out_tck)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recipes(roi: dict[str, Path]) -> list[dict]:
    """Bank rebuild recipes → manifest bank_* keys."""
    H = {
        "l": roi["hemi_r"],  # exclude contralateral
        "r": roi["hemi_l"],
    }
    bs = roi["exclude_bs_cereb"]
    cc = roi["exclude_cc"]

    def uni(side: str, *more_excl: Path) -> list[Path]:
        return [H[side], bs, *more_excl]

    return [
        # CST
        {
            "key": "bank_cst_r",
            "file": "cst_r_motor_pons.tck",
            "includes": [roi["cst_r_motor"], roi["cst_r_pons"]],
            "excludes": [H["r"]],
            "minlength": 80,
            "maxlength": 250,
            "label": "CST-R candidate (3T ACT 10M, strict ROI)",
            "role": "true_cst",
            "default": True,
            "engine": "ACT iFOD2 10M -> precentral dil2 + pons dil3, exclude L",
        },
        {
            "key": "bank_cst_l",
            "file": "cst_l_motor_pons.tck",
            "includes": [roi["cst_l_motor"], roi["cst_l_pons"]],
            "excludes": [H["l"]],
            "minlength": 80,
            "maxlength": 250,
            "label": "CST-L candidate (3T ACT 10M, strict ROI)",
            "role": "true_cst",
            "engine": "ACT iFOD2 10M -> precentral dil2 + pons dil3, exclude R",
        },
        # FAT
        {
            "key": "bank_fat_r",
            "file": "fat_r_sfg_ifg.tck",
            "includes": [roi["fat_r_sfg"], roi["fat_r_ifg"]],
            "excludes": uni("r", cc),
            "minlength": 45,
            "maxlength": 180,
            "label": "FAT-R (SFG↔IFG, strict)",
            "role": "true_fat",
            "engine": "ACT iFOD2 10M -> SFG+POP/PTR dil1, exclude L+BS+CC",
        },
        {
            "key": "bank_fat_l",
            "file": "fat_l_sfg_ifg.tck",
            "includes": [roi["fat_l_sfg"], roi["fat_l_ifg"]],
            "excludes": uni("l", cc),
            "minlength": 45,
            "maxlength": 180,
            "label": "FAT-L (SFG↔IFG, strict)",
            "role": "true_fat",
            "engine": "ACT iFOD2 10M -> SFG+POP/PTR dil1, exclude R+BS+CC",
        },
        # FAT soft (dil2) — more peri-lesional tolerant; not true_* identity
        {
            "key": "bank_fat_r_soft",
            "file": "fat_r_sfg_ifg_soft.tck",
            "includes": [roi["fat_r_sfg_soft"], roi["fat_r_ifg_soft"]],
            "excludes": uni("r", cc),
            "minlength": 35,
            "maxlength": 180,
            "label": "FAT-R soft (dil2 · exploratory)",
            "role": "soft_fat",
            "engine": "ACT iFOD2 10M -> SFG+POP/PTR dil2 soft, exclude L+BS+CC",
        },
        {
            "key": "bank_fat_l_soft",
            "file": "fat_l_sfg_ifg_soft.tck",
            "includes": [roi["fat_l_sfg_soft"], roi["fat_l_ifg_soft"]],
            "excludes": uni("l", cc),
            "minlength": 35,
            "maxlength": 180,
            "label": "FAT-L soft (dil2 · exploratory)",
            "role": "soft_fat",
            "engine": "ACT iFOD2 10M -> SFG+POP/PTR dil2 soft, exclude R+BS+CC",
        },
        # SLF-I (dorsal)
        {
            "key": "bank_slf1_r",
            "file": "slf1_r_sfg_pari.tck",
            "includes": [roi["slf1_r_front"], roi["slf1_r_pari"]],
            "excludes": uni("r", cc),
            "minlength": 45,
            "maxlength": 200,
            "label": "SLF-I-R (strict SFG↔SPL/precuneus)",
            "role": "true_slf1",
            "engine": "ACT iFOD2 10M -> SFG+SPL/precuneus dil1, exclude L+BS+CC",
        },
        {
            "key": "bank_slf1_l",
            "file": "slf1_l_sfg_pari.tck",
            "includes": [roi["slf1_l_front"], roi["slf1_l_pari"]],
            "excludes": uni("l", cc),
            "minlength": 45,
            "maxlength": 200,
            "label": "SLF-I-L (strict SFG↔SPL/precuneus)",
            "role": "true_slf1",
            "engine": "ACT iFOD2 10M -> SFG+SPL/precuneus dil1, exclude R+BS+CC",
        },
        {
            "key": "bank_slf1_r_soft",
            "file": "slf1_r_sfg_pari_soft.tck",
            "includes": [roi["slf1_r_front_soft"], roi["slf1_r_pari_soft"]],
            "excludes": uni("r", cc),
            "minlength": 35,
            "maxlength": 200,
            "label": "SLF-I-R soft (dil2 · exploratory)",
            "role": "soft_slf1",
            "engine": "ACT iFOD2 10M -> SFG+SPL/precuneus dil2 soft",
        },
        {
            "key": "bank_slf1_l_soft",
            "file": "slf1_l_sfg_pari_soft.tck",
            "includes": [roi["slf1_l_front_soft"], roi["slf1_l_pari_soft"]],
            "excludes": uni("l", cc),
            "minlength": 35,
            "maxlength": 200,
            "label": "SLF-I-L soft (dil2 · exploratory)",
            "role": "soft_slf1",
            "engine": "ACT iFOD2 10M -> SFG+SPL/precuneus dil2 soft",
        },
        # SLF-II (middle)
        {
            "key": "bank_slf2_r",
            "file": "slf2_r_mfg_ipl.tck",
            "includes": [roi["slf2_r_front"], roi["slf2_r_pari"]],
            "excludes": uni("r", cc),
            "minlength": 40,
            "maxlength": 190,
            "label": "SLF-II-R (strict MFG↔IPL)",
            "role": "true_slf2",
            "engine": "ACT iFOD2 10M -> RMF/CMF+IPL dil1, exclude L+BS+CC",
        },
        {
            "key": "bank_slf2_l",
            "file": "slf2_l_mfg_ipl.tck",
            "includes": [roi["slf2_l_front"], roi["slf2_l_pari"]],
            "excludes": uni("l", cc),
            "minlength": 40,
            "maxlength": 190,
            "label": "SLF-II-L (strict MFG↔IPL)",
            "role": "true_slf2",
            "engine": "ACT iFOD2 10M -> RMF/CMF+IPL dil1, exclude R+BS+CC",
        },
        {
            "key": "bank_slf2_r_soft",
            "file": "slf2_r_mfg_ipl_soft.tck",
            "includes": [roi["slf2_r_front_soft"], roi["slf2_r_pari_soft"]],
            "excludes": uni("r", cc),
            "minlength": 30,
            "maxlength": 190,
            "label": "SLF-II-R soft (dil2 · exploratory)",
            "role": "soft_slf2",
            "engine": "ACT iFOD2 10M -> RMF/CMF+IPL dil2 soft",
        },
        {
            "key": "bank_slf2_l_soft",
            "file": "slf2_l_mfg_ipl_soft.tck",
            "includes": [roi["slf2_l_front_soft"], roi["slf2_l_pari_soft"]],
            "excludes": uni("l", cc),
            "minlength": 30,
            "maxlength": 190,
            "label": "SLF-II-L soft (dil2 · exploratory)",
            "role": "soft_slf2",
            "engine": "ACT iFOD2 10M -> RMF/CMF+IPL dil2 soft",
        },
        # SLF-III (ventral)
        {
            "key": "bank_slf3_r",
            "file": "slf3_r_smg_frontal.tck",
            "includes": [roi["slf3_r_smg"], roi["slf3_r_front"]],
            "excludes": uni("r", cc),
            "minlength": 40,
            "maxlength": 180,
            "label": "SLF-III-R (strict)",
            "role": "true_slf3",
            "engine": "ACT iFOD2 10M -> SMG+CMF/POP/PTR dil1, exclude L+BS+CC",
        },
        {
            "key": "bank_slf3_l",
            "file": "slf3_l_smg_frontal.tck",
            "includes": [roi["slf3_l_smg"], roi["slf3_l_front"]],
            "excludes": uni("l", cc),
            "minlength": 40,
            "maxlength": 180,
            "label": "SLF-III-L (strict)",
            "role": "true_slf3",
            "engine": "ACT iFOD2 10M -> SMG+CMF/POP/PTR dil1, exclude R+BS+CC",
        },
        {
            "key": "bank_slf3_r_soft",
            "file": "slf3_r_smg_frontal_soft.tck",
            "includes": [roi["slf3_r_smg_soft"], roi["slf3_r_front_soft"]],
            "excludes": uni("r", cc),
            "minlength": 30,
            "maxlength": 180,
            "label": "SLF-III-R soft (dil2 · exploratory)",
            "role": "soft_slf3",
            "engine": "ACT iFOD2 10M -> SMG+CMF/POP/PTR dil2 soft",
        },
        {
            "key": "bank_slf3_l_soft",
            "file": "slf3_l_smg_frontal_soft.tck",
            "includes": [roi["slf3_l_smg_soft"], roi["slf3_l_front_soft"]],
            "excludes": uni("l", cc),
            "minlength": 30,
            "maxlength": 180,
            "label": "SLF-III-L soft (dil2 · exploratory)",
            "role": "soft_slf3",
            "engine": "ACT iFOD2 10M -> SMG+CMF/POP/PTR dil2 soft",
        },
        # IFOF
        {
            "key": "bank_ifof_r",
            "file": "ifof_r_occ_front.tck",
            "includes": [roi["ifof_r_occ"], roi["ifof_r_front"]],
            "excludes": uni("r"),
            "minlength": 80,
            "maxlength": 220,
            "label": "IFOF-R (strict)",
            "role": "true_ifof",
            "engine": "ACT iFOD2 10M -> visual+orbital dil1, exclude L+BS",
        },
        {
            "key": "bank_ifof_l",
            "file": "ifof_l_occ_front.tck",
            "includes": [roi["ifof_l_occ"], roi["ifof_l_front"]],
            "excludes": uni("l"),
            "minlength": 80,
            "maxlength": 220,
            "label": "IFOF-L (strict)",
            "role": "true_ifof",
            "engine": "ACT iFOD2 10M -> visual+orbital dil1, exclude R+BS",
        },
        # UF
        {
            "key": "bank_uf_r",
            "file": "uf_r_temp_orb.tck",
            "includes": [roi["uf_r_temp"], roi["uf_r_orb"]],
            "excludes": uni("r", cc),
            "minlength": 35,
            "maxlength": 150,
            "label": "UF-R (strict)",
            "role": "true_uf",
            "engine": "ACT iFOD2 10M -> temporal pole+orbital dil1, exclude L+BS+CC",
        },
        {
            "key": "bank_uf_l",
            "file": "uf_l_temp_orb.tck",
            "includes": [roi["uf_l_temp"], roi["uf_l_orb"]],
            "excludes": uni("l", cc),
            "minlength": 35,
            "maxlength": 150,
            "label": "UF-L (strict)",
            "role": "true_uf",
            "engine": "ACT iFOD2 10M -> temporal pole+orbital dil1, exclude R+BS+CC",
        },
        # Cingulum
        {
            "key": "bank_cing_r",
            "file": "cing_r_ant_post.tck",
            "includes": [roi["cing_r_ant"], roi["cing_r_post"]],
            "excludes": uni("r"),
            "minlength": 40,
            "maxlength": 180,
            "label": "Cingulum-R (strict)",
            "role": "true_cing",
            "engine": "ACT iFOD2 10M -> ant+post cing dil1, exclude L+BS",
        },
        {
            "key": "bank_cing_l",
            "file": "cing_l_ant_post.tck",
            "includes": [roi["cing_l_ant"], roi["cing_l_post"]],
            "excludes": uni("l"),
            "minlength": 40,
            "maxlength": 180,
            "label": "Cingulum-L (strict)",
            "role": "true_cing",
            "engine": "ACT iFOD2 10M -> ant+post cing dil1, exclude R+BS",
        },
        # OR
        {
            "key": "bank_or_r",
            "file": "or_r_thal_vis.tck",
            "includes": [roi["or_r_thal"], roi["or_r_vis"]],
            "excludes": uni("r"),
            "minlength": 45,
            "maxlength": 200,
            "label": "Optic radiation R (strict)",
            "role": "true_or",
            "engine": "ACT iFOD2 10M -> thal+visual dil1, exclude L+BS",
        },
        {
            "key": "bank_or_l",
            "file": "or_l_thal_vis.tck",
            "includes": [roi["or_l_thal"], roi["or_l_vis"]],
            "excludes": uni("l"),
            "minlength": 45,
            "maxlength": 200,
            "label": "Optic radiation L (strict)",
            "role": "true_or",
            "engine": "ACT iFOD2 10M -> thal+visual dil1, exclude R+BS",
        },
        {
            "key": "bank_or_r_meyer",
            "file": "or_r_meyer.tck",
            "includes": [roi["or_r_thal"], roi["or_r_temp"], roi["or_r_vis"]],
            "excludes": uni("r"),
            "minlength": 45,
            "maxlength": 200,
            "label": "OR-R Meyer-ish (strict)",
            "role": "true_or",
            "engine": "ACT iFOD2 10M -> thal+temp+vis dil1, exclude L+BS",
        },
        {
            "key": "bank_or_l_meyer",
            "file": "or_l_meyer.tck",
            "includes": [roi["or_l_thal"], roi["or_l_temp"], roi["or_l_vis"]],
            "excludes": uni("l"),
            "minlength": 45,
            "maxlength": 200,
            "label": "OR-L Meyer-ish (strict)",
            "role": "true_or",
            "engine": "ACT iFOD2 10M -> thal+temp+vis dil1, exclude R+BS",
        },
        # Callosal — forceps minor uses SFG bilat + genu (not full frontal+whole CC)
        {
            "key": "bank_cc_forceps_minor",
            "file": "cc_forceps_minor.tck",
            "includes": [
                roi["fmin_sfg_l"],
                roi["fmin_sfg_r"],
                roi["cc_genu"],
            ],
            "excludes": [bs],
            "minlength": 60,
            "maxlength": 180,
            "label": "Forceps minor (strict SFG+genu)",
            "role": "true_cc",
            "engine": "ACT iFOD2 10M -> bilat SFG dil1 + genu dil1, exclude BS",
        },
        {
            "key": "bank_cc_forceps_major",
            "file": "cc_forceps_major.tck",
            "includes": [roi["fmaj_occ_l"], roi["fmaj_occ_r"], roi["cc_body"]],
            "excludes": [bs],
            "minlength": 60,
            "maxlength": 220,
            "label": "Forceps major (strict)",
            "role": "true_cc",
            "engine": "ACT iFOD2 10M -> bilat occipital dil1 + CC, exclude BS",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case-root",
        default=os.path.expanduser(
            "~/tractlab-data/cases/local-case"
        ),
    )
    ap.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "cases/local-case/manifest.json"
        ),
    )
    ap.add_argument("--rois-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of bank keys")
    ap.add_argument("--nthreads", type=int, default=8)
    args = ap.parse_args()

    case = Path(os.path.expanduser(args.case_root)).resolve()
    corpus = case / (
        "tracts/connectome/pre/freesurfer/path-to-wm-v2/"
        "wholebrain_act_ifod2_10000000.tck"
    )
    if not corpus.is_file():
        print(f"corpus missing: {corpus}", file=sys.stderr)
        return 1
    if not Path(TCKEDIT).is_file():
        print(f"tckedit missing: {TCKEDIT}", file=sys.stderr)
        return 1

    roi = build_rois(case)
    if args.rois_only:
        return 0

    bank = case / "tracts/bank"
    raw = bank / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    man_path = Path(args.manifest).resolve()
    with open(man_path) as f:
        man = json.load(f)
    inputs = man.setdefault("inputs", {})

    recs = recipes(roi)
    if args.only:
        want = set(args.only)
        recs = [r for r in recs if r["key"] in want]

    summary = []
    for r in recs:
        out = bank / r["file"]
        print(f"\n=== {r['key']} → {out.name} ===")
        for p in r["includes"] + r["excludes"]:
            if not Path(p).is_file():
                print(f"  MISSING ROI {p}", file=sys.stderr)
                return 1
        n = extract(
            corpus,
            out,
            includes=list(r["includes"]),
            excludes=list(r["excludes"]),
            minlength=r["minlength"],
            maxlength=r["maxlength"],
            nthreads=args.nthreads,
        )
        print(f"  count={n}")
        if n == 0:
            print("  WARN empty bank", file=sys.stderr)
        # refresh raw baseline for prune
        shutil.copy2(out, raw / out.name)
        # drop stale sift2
        sift = out.with_suffix("").as_posix() + ".sift2.txt"
        if Path(sift).is_file():
            Path(sift).unlink()

        meta = inputs.setdefault(r["key"], {})
        meta.update(
            {
                "path": f"tracts/bank/{r['file']}",
                "bytes": int(out.stat().st_size) if out.is_file() else 0,
                "sha256": sha256(out) if out.is_file() else "",
                "label": r["label"],
                "n_streamlines": n,
                "engine": r["engine"] + " | strict multi-ROI",
                "role": r["role"],
                "note": (
                    f"Stricter multi-ROI (dil1 association / CST motor dil2). "
                    f"n={n}. Research only — {dv.floor_label(man)}, not navigation."
                ),
            }
        )
        if r.get("default"):
            meta["default"] = True
        summary.append((r["key"], n))

    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    print(f"\nManifest updated: {man_path}")
    print("Summary:")
    for k, n in summary:
        print(f"  {k:28s} n={n}")
    print("\nNext: PYTHONPATH=src python scripts/prune_banks.py ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
