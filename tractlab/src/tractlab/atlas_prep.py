"""Prepare XTRACT population priors in case space (Slice A1) [white-box].

- XTRACT.xml is the sole name authority (refuse unknown names).
- Grid authority = nibabel affine (mrinfo banned).
- Auto QC may only **block**; human ``--approve`` is the only path that approves
  (ADR-0002). Regenerating any norm_* voids atlas_prior_qc.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import nibabel as nib
from defusedxml import ElementTree as ET
from neuro_core.hashing import sha256_file

from . import derivation
from .grid import load_grid, grid_id
from .priors import atlas_prior_qc_ok

ANTS_BIN = os.path.expanduser("~/ants-2.6.5/bin")
FSL_DIR = os.path.expanduser("~/fsl")
XTRACT_XML = os.path.join(FSL_DIR, "data/atlases/XTRACT.xml")
XTRACT_4D = os.path.join(
    FSL_DIR, "data/atlases/XTRACT/xtract-tract-atlases-prob-1mm.nii.gz"
)
MNI_BRAIN = os.path.join(FSL_DIR, "data/standard/MNI152_T1_1mm_brain.nii.gz")

# Bank-paired default set (plan: both hemispheres + forceps). Official names
# must match XTRACT.xml text exactly.
DEFAULT_PRIORS: dict[str, str] = {
    "cst_l": "Corticospinal Tract L",
    "cst_r": "Corticospinal Tract R",
    "fa_l": "Frontal Aslant Tract L",
    "fa_r": "Frontal Aslant Tract R",
    "slf1_l": "Superior Longitudinal Fasciculus 1 L",
    "slf1_r": "Superior Longitudinal Fasciculus 1 R",
    "slf2_l": "Superior Longitudinal Fasciculus 2 L",
    "slf2_r": "Superior Longitudinal Fasciculus 2 R",
    "slf3_l": "Superior Longitudinal Fasciculus 3 L",
    "slf3_r": "Superior Longitudinal Fasciculus 3 R",
    "af_l": "Arcuate Fasciculus L",
    "af_r": "Arcuate Fasciculus R",
    "ifo_l": "Inferior Fronto-Occipital Fasciculus L",
    "ifo_r": "Inferior Fronto-Occipital Fasciculus R",
    "uf_l": "Uncinate Fasciculus L",
    "uf_r": "Uncinate Fasciculus R",
    "cbd_l": "Cingulum subsection: Dorsal L",
    "cbd_r": "Cingulum subsection: Dorsal R",
    "or_l": "Optic Radiation L",
    "or_r": "Optic Radiation R",
    "fmi": "Forceps Minor",
    "fma": "Forceps Major",
}


def load_registry(xml_path: str = XTRACT_XML) -> dict[int, str]:
    root = ET.parse(xml_path).getroot()
    return {int(l.get("index")): l.text.strip() for l in root.iter("label")}


def resolve_wanted(
    wanted: dict[str, str],
    registry: dict[int, str],
) -> dict[str, tuple[int, str]]:
    """Map short key → (atlas_index, official_name). Refuse unknown names."""
    name_to_idx = {v: k for k, v in registry.items()}
    out: dict[str, tuple[int, str]] = {}
    for key, official in wanted.items():
        if official not in name_to_idx:
            raise ValueError(
                f"NOT IN XTRACT REGISTRY: {official!r} (key={key}) — refusing to guess"
            )
        out[key] = (name_to_idx[official], official)
    return out


@dataclass(frozen=True)
class AutoQC:
    ok: bool
    cst_symmetry_ok: bool
    hull_containment_ok: bool
    cst_overlap: float | None
    hull_frac: float | None
    notes: tuple[str, ...]


def auto_qc_priors(
    *,
    norm_dir: Path,
    mask_path: str,
    keys: list[str],
    cst_l: str = "cst_l",
    cst_r: str = "cst_r",
    thr: float = 0.25,
    symmetry_min: float = 0.15,
    hull_min: float = 0.85,
) -> AutoQC:
    """Block-only auto QC: CST L/R spatial symmetry + brain-mask containment."""
    notes: list[str] = []
    mask = np.asanyarray(nib.load(mask_path).dataobj) > 0

    def load_prob(key: str) -> np.ndarray | None:
        p = norm_dir / f"norm_{key}.nii.gz"
        if not p.is_file():
            return None
        v = np.asanyarray(nib.load(str(p)).dataobj, dtype=np.float64)
        if float(np.nanmax(v)) > 1.5:
            v = v / 100.0
        return np.clip(v, 0, 1)

    # CST symmetry: Dice of thresholded L vs R after x-flip of L onto R grid
    cst_ok = True
    overlap = None
    vl = load_prob(cst_l)
    vr = load_prob(cst_r)
    if vl is None or vr is None:
        cst_ok = False
        notes.append("missing cst_l or cst_r for symmetry check")
    else:
        bl = vl >= thr
        br = vr >= thr
        bl_flip = np.flip(bl, axis=0)  # approximate L↔R in L/P/S grids
        inter = float(np.logical_and(bl_flip, br).sum())
        union = float(np.logical_or(bl_flip, br).sum())
        overlap = inter / union if union > 0 else 0.0
        if overlap < symmetry_min:
            cst_ok = False
            notes.append(f"CST L/R symmetry dice {overlap:.3f} < {symmetry_min}")

    # Hull containment: fraction of suprathreshold prior mass inside brain mask
    hull_ok = True
    hull_fracs = []
    for key in keys:
        v = load_prob(key)
        if v is None:
            hull_ok = False
            notes.append(f"missing prior {key}")
            continue
        if v.shape != mask.shape:
            hull_ok = False
            notes.append(f"shape mismatch {key}")
            continue
        sel = v >= thr
        n = int(sel.sum())
        if n == 0:
            notes.append(f"empty prior at thr {key}")
            continue
        frac = float(np.logical_and(sel, mask).sum() / n)
        hull_fracs.append(frac)
        if frac < hull_min:
            hull_ok = False
            notes.append(f"{key} hull containment {frac:.3f} < {hull_min}")

    mean_hull = float(np.mean(hull_fracs)) if hull_fracs else None
    ok = cst_ok and hull_ok
    return AutoQC(
        ok=ok,
        cst_symmetry_ok=cst_ok,
        hull_containment_ok=hull_ok,
        cst_overlap=overlap,
        hull_frac=mean_hull,
        notes=tuple(notes),
    )


def _run(argv: list[str], *, timeout_s: int) -> None:
    subprocess.run(argv, check=True, timeout=timeout_s, capture_output=True, text=True)


def ensure_mni_to_case_warp(
    *,
    t1_path: str,
    out_prefix: Path,
    full_syn: bool = True,
    timeout_s: int = 7200,
) -> tuple[Path, Path]:
    """Return (warp, affine). Run ANTs if missing."""
    warp = Path(str(out_prefix) + "1Warp.nii.gz")
    affine = Path(str(out_prefix) + "0GenericAffine.mat")
    if warp.is_file() and affine.is_file():
        return warp, affine
    ants = Path(ANTS_BIN)
    script = ants / ("antsRegistrationSyN.sh" if full_syn else "antsRegistrationSyNQuick.sh")
    if not script.is_file():
        raise FileNotFoundError(script)
    if not Path(MNI_BRAIN).is_file():
        raise FileNotFoundError(MNI_BRAIN)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # fixed=t1 (case), moving=MNI
    _run(
        [
            str(script), "-d", "3",
            "-f", t1_path,
            "-m", MNI_BRAIN,
            "-o", str(out_prefix),
            "-n", "4",
        ],
        timeout_s=timeout_s,
    )
    if not warp.is_file() or not affine.is_file():
        raise RuntimeError(f"ANTs did not produce warp/affine under {out_prefix}")
    return warp, affine


def apply_atlas_warp(
    *,
    atlas_4d: str,
    t1_path: str,
    warp: Path,
    affine: Path,
    out_4d: Path,
    timeout_s: int = 1800,
) -> None:
    apply = Path(ANTS_BIN) / "antsApplyTransforms"
    if not apply.is_file():
        raise FileNotFoundError(apply)
    out_4d.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(apply), "-d", "3", "-e", "3",
            "-i", atlas_4d, "-r", t1_path,
            "-t", str(warp), "-t", str(affine),
            "-n", "Linear", "-o", str(out_4d),
        ],
        timeout_s=timeout_s,
    )


def extract_norm_volumes(
    *,
    warped_4d: Path,
    t1_path: str,
    out_dir: Path,
    resolved: dict[str, tuple[int, str]],
    case_grid_path: str,
    case_root: Path | None = None,
    artifact_manifest: dict | None = None,
) -> dict[str, dict]:
    """Split 4D atlas into norm_*.nii.gz on the **tracking/b0 grid**.

    Warped atlas lands on T1 grid; if T1==b0 grid (this case), save directly.
    Else resample with nearest via affine equality check fail → error (no silent).
    """
    g_t1 = load_grid(t1_path)
    g_case = load_grid(case_grid_path)
    if g_t1.shape != g_case.shape or not np.allclose(g_t1.affine, g_case.affine, atol=1e-3):
        raise ValueError(
            "T1 grid ≠ tracking grid — refuse silent resample "
            f"(t1 {g_t1.shape} case {g_case.shape})"
        )
    img = nib.load(str(warped_4d))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 4:
        raise ValueError(f"expected 4-D atlas, got {data.shape}")
    if data.shape[:3] != g_t1.shape:
        raise ValueError(f"warped spatial {data.shape[:3]} vs t1 {g_t1.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for key, (idx, official) in resolved.items():
        if idx < 0 or idx >= data.shape[3]:
            raise ValueError(f"atlas index {idx} out of range for {key}")
        vol = data[..., idx].astype(np.float32)
        out = (
            derivation.artifact_path(
                artifact_manifest, "normative", f"norm_{key}.nii.gz"
            )
            if artifact_manifest is not None
            else out_dir / f"norm_{key}.nii.gz"
        )
        # Write with case affine (nibabel authority)
        nib.save(nib.Nifti1Image(vol, g_case.affine), str(out))
        nz = vol[vol > 0]
        rel_path = (
            out.resolve().relative_to(Path(case_root).resolve()).as_posix()
            if case_root is not None else f"normative/{out.name}"
        )
        index[key] = {
            "official_name": official,
            "atlas_index": idx,
            "file": out.name,
            "path": rel_path,
            "bytes": int(out.stat().st_size),
            "sha256": sha256_file(out),
            "voxels_nonzero": int((vol > 0).sum()),
            "prob_max": float(nz.max()) if nz.size else 0.0,
        }
    return index


def void_atlas_qc(man: dict) -> None:
    """Any regeneration voids human approval and stamps the current derivation."""
    if derivation.qc_block_for(man, "atlas_prior_qc") is not None:
        derivation.set_qc_block(man, "atlas_prior_qc", {
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "voided": True,
            "note": "voided by atlas_prep regeneration",
        })


def write_contour_qc_sheet(
    *,
    b0_path: str,
    mask_path: str,
    norm_dir: Path,
    keys: list[str],
    out_png: Path,
    thr: float = 0.25,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b0 = np.asanyarray(nib.load(b0_path).dataobj, dtype=np.float64)
    mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
    k = b0.shape[2] // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#0d0f12")
    sl = np.rot90(b0[:, :, k])
    sl = sl / (np.percentile(sl[sl > 0], 98) + 1e-9) if (sl > 0).any() else sl
    ax.imshow(np.clip(sl, 0, 1), cmap="gray", origin="lower")
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(keys), 1)))
    for c, key in zip(colors, keys):
        p = norm_dir / f"norm_{key}.nii.gz"
        if not p.is_file():
            continue
        v = np.asanyarray(nib.load(str(p)).dataobj, dtype=np.float64)
        if float(np.nanmax(v)) > 1.5:
            v = v / 100.0
        cont = np.rot90((v[:, :, k] >= thr).astype(float))
        ax.contour(cont, levels=[0.5], colors=[c], linewidths=0.8)
    ax.set_title(
        "Atlas prior contours on b0 (p≥25%) — POPULATION · not patient",
        color="#e8c56a",
        fontsize=10,
    )
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return sha256_file(out_png)


def update_case_manifest_priors(
    man_path: Path,
    *,
    tract_index: dict[str, dict],
    auto: AutoQC,
    sheet_sha: str,
    reg_note: str,
) -> None:
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    void_atlas_qc(man)
    root = Path(man["case_root"]).resolve()
    inputs = derivation.active_inputs(man)
    # drop old norm_* then re-add
    for k in list(inputs.keys()):
        if k.startswith("norm_"):
            del inputs[k]
    for key, meta in tract_index.items():
        record = {
            "path": meta["path"],
            "bytes": meta["bytes"],
            "sha256": meta["sha256"],
            "label": meta["official_name"],
            "official_name": meta["official_name"],
            "atlas_index": meta["atlas_index"],
            "note": "XTRACT population prior · not patient anatomy · not navigation",
            "registration": reg_note,
        }
        derivs = man.get("derivations")
        active = derivation.active_derivation(man)
        if isinstance(derivs, dict) and len(derivs) > 1 and active != next(iter(derivs)):
            record["sha256_by_field"] = {"path": meta["sha256"]}
            record["bytes_by_field"] = {"path": meta["bytes"]}
        inputs[f"norm_{key}"] = record
    derivation.set_qc_block(man, "atlas_prior_qc", {
        "auto": {
            "ok": auto.ok,
            "cst_symmetry_ok": auto.cst_symmetry_ok,
            "hull_containment_ok": auto.hull_containment_ok,
            "cst_overlap": auto.cst_overlap,
            "hull_frac": auto.hull_frac,
            "notes": list(auto.notes),
        },
        "approved_by": None,
        "date": None,
        "sheet_sha": sheet_sha,
        "sheet_path": derivation.artifact_path(
            {**man, "case_root": str(root)}, "qc", "atlas_prior_contours.png"
        ).relative_to(root).as_posix(),
    })
    derivation.write_manifest_atomic(man_path, man)


def approve_atlas_qc(
    manifest_path: str,
    *,
    approved_by: str,
    sheet_sha: str | None = None,
) -> dict:
    man_path = Path(manifest_path)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    layout = {**man, "case_root": str(root)}
    inputs = derivation.active_inputs(man)
    sheet = derivation.artifact_path(layout, "qc", "atlas_prior_contours.png")
    if not sheet.is_file():
        raise FileNotFoundError(f"QC sheet missing: {sheet}")
    live = sha256_file(sheet)
    if sheet_sha and sheet_sha != live:
        raise ValueError("sheet_sha mismatch")
    qc = derivation.qc_block_for(layout, "atlas_prior_qc") or {}
    auto = qc.get("auto") or {}
    if not auto.get("ok", False):
        raise ValueError("auto QC blocked — will not record human approval")
    # require all norm_* present
    norms = [k for k in inputs if k.startswith("norm_")]
    if not norms:
        raise ValueError("no norm_* inputs declared")
    for k in norms:
        rel = inputs[k]["path"]
        if not (root / rel).is_file():
            raise ValueError(f"missing prior file for {k}")
    result = derivation.set_qc_block(man, "atlas_prior_qc", {
        "auto": auto,
        "approved_by": approved_by,
        "date": date.today().isoformat(),
        "sheet_sha": live,
        "sheet_path": sheet.relative_to(root).as_posix(),
    })
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    return result


def prepare(
    *,
    manifest_path: str,
    full_syn: bool = False,
    reuse_warp: bool = True,
    wanted: dict[str, str] | None = None,
) -> dict:
    man_path = Path(manifest_path)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    layout = {**man, "case_root": str(root)}
    inputs = derivation.active_inputs(man)
    t1 = root / inputs["t1"]["path"]
    b0 = root / inputs["b0"]["path"]
    mask = root / inputs["mask"]["path"]
    norm = derivation.artifact_dir(layout, "normative")
    # Resolve all producer outputs before clearing the previous signature.
    # This keeps invalid/symlinked layouts from mutating the manifest.
    prefix = derivation.artifact_path(layout, "normative", "mni2case_")
    warp = derivation.artifact_path(layout, "normative", "mni2case_1Warp.nii.gz")
    affine = derivation.artifact_path(layout, "normative", "mni2case_0GenericAffine.mat")
    warped_4d = derivation.artifact_path(layout, "normative", "xtract_prob_case.nii.gz")
    sheet = derivation.artifact_path(layout, "qc", "atlas_prior_contours.png")
    if derivation.qc_block_for(man, "atlas_prior_qc") is not None:
        void_atlas_qc(man)
        derivation.write_manifest_atomic(man_path, man)
    norm.mkdir(parents=True, exist_ok=True)

    reg = load_registry()
    wanted = wanted or DEFAULT_PRIORS
    resolved = resolve_wanted(wanted, reg)

    if reuse_warp and warp.is_file() and affine.is_file():
        reg_tool = "existing warp (reuse)"
    else:
        warp, affine = ensure_mni_to_case_warp(
            t1_path=str(t1), out_prefix=prefix, full_syn=full_syn,
        )
        reg_tool = "antsRegistrationSyN.sh" if full_syn else "antsRegistrationSyNQuick.sh"

    if not warped_4d.is_file() or not reuse_warp:
        apply_atlas_warp(
            atlas_4d=XTRACT_4D,
            t1_path=str(t1),
            warp=warp,
            affine=affine,
            out_4d=warped_4d,
        )

    index = extract_norm_volumes(
        warped_4d=warped_4d,
        t1_path=str(t1),
        out_dir=norm,
        resolved=resolved,
        case_grid_path=str(b0),
        case_root=root,
        artifact_manifest=layout,
    )
    auto = auto_qc_priors(
        norm_dir=norm,
        mask_path=str(mask),
        keys=list(resolved.keys()),
    )
    sheet_sha = write_contour_qc_sheet(
        b0_path=str(b0),
        mask_path=str(mask),
        norm_dir=norm,
        keys=list(resolved.keys()),
        out_png=sheet,
    )
    reg_note = (
        f"{reg_tool}; fixed=t1c_brain_dwi; moving=MNI152_1mm_brain; "
        "XTRACT pop atlas; not navigation"
    )
    update_case_manifest_priors(
        man_path,
        tract_index=index,
        auto=auto,
        sheet_sha=sheet_sha,
        reg_note=reg_note,
    )
    # also write repo-side case manifest if this is the data case
    return {
        "n_priors": len(index),
        "auto_ok": auto.ok,
        "auto_notes": list(auto.notes),
        "sheet": str(sheet),
        "sheet_sha": sheet_sha,
        "grid_id": grid_id(load_grid(str(b0))),
    }


def apply_label_warp_nn(
    *,
    labels_mni: Path,
    t1_path: str,
    warp: Path,
    affine: Path,
    out_labels: Path,
    timeout_s: int = 1800,
) -> None:
    """Warp integer labels with NearestNeighbor — never linear (label tripwire)."""
    apply = Path(ANTS_BIN) / "antsApplyTransforms"
    if not apply.is_file():
        raise FileNotFoundError(apply)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(apply), "-d", "3",
            "-i", str(labels_mni), "-r", t1_path,
            "-t", str(warp), "-t", str(affine),
            "-n", "NearestNeighbor",
            "-o", str(out_labels),
        ],
        timeout_s=timeout_s,
    )


@dataclass(frozen=True)
class ParcelAutoQC:
    ok: bool
    label_complete: bool
    lut_ok: bool
    hull_ok: bool
    symmetry_ok: bool
    n_labels: int
    hull_frac: float | None
    lr_ratio: float | None
    notes: tuple[str, ...]


def auto_qc_parcellation(
    *,
    labels_path: Path,
    mask_path: str,
    lut: dict[int, dict],
    expected_n: int = 200,
    hull_min: float = 0.85,
    lr_ratio_lo: float = 0.55,
    lr_ratio_hi: float = 1.85,
) -> ParcelAutoQC:
    notes: list[str] = []
    vol = np.asanyarray(nib.load(str(labels_path)).dataobj).astype(np.int32)
    mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
    if vol.shape != mask.shape:
        return ParcelAutoQC(
            False, False, False, False, False, 0, None, None,
            ("shape mismatch labels vs mask",),
        )
    present = set(int(x) for x in np.unique(vol) if int(x) > 0)
    expected = set(lut.keys())
    label_complete = present == expected
    if not label_complete:
        missing = sorted(expected - present)[:8]
        extra = sorted(present - expected)[:8]
        notes.append(f"label set mismatch missing={missing} extra={extra}")

    lut_ok = len(lut) == expected_n and all(
        1 <= int(m.get("network_id") or 0) <= 7 for m in lut.values()
    )
    if not lut_ok:
        notes.append("LUT registry incomplete or network_id out of 1..7")

    in_mask = vol[mask]
    cort = in_mask > 0
    if cort.any():
        # hull: fraction of labeled voxels that fall in brain mask vs all labeled
        all_lab = (vol > 0).sum()
        in_lab = cort.sum()
        hull_frac = float(in_lab / all_lab) if all_lab else 0.0
    else:
        hull_frac = 0.0
    hull_ok = hull_frac >= hull_min
    if not hull_ok:
        notes.append(f"hull containment {hull_frac:.3f} < {hull_min}")

    # L/R by hemi token in LUT (network volume symmetry)
    from .atlas_fetch import label_to_network_map

    mapping = label_to_network_map(lut)
    l_vol = r_vol = 0
    for lab, meta in lut.items():
        n = int((vol == lab).sum())
        if meta.get("hemi") == "L":
            l_vol += n
        elif meta.get("hemi") == "R":
            r_vol += n
    lr_ratio = (l_vol / r_vol) if r_vol > 0 else None
    symmetry_ok = (
        lr_ratio is not None and lr_ratio_lo <= lr_ratio <= lr_ratio_hi
    )
    if not symmetry_ok:
        notes.append(f"L/R volume ratio {lr_ratio} outside [{lr_ratio_lo},{lr_ratio_hi}]")

    ok = label_complete and lut_ok and hull_ok and symmetry_ok
    return ParcelAutoQC(
        ok=ok,
        label_complete=label_complete,
        lut_ok=lut_ok,
        hull_ok=hull_ok,
        symmetry_ok=symmetry_ok,
        n_labels=len(present),
        hull_frac=hull_frac,
        lr_ratio=lr_ratio,
        notes=tuple(notes),
    )


# Brighter palette for QC sheets only (3D ghost stays desaturated in YEO7_COLORS)
_QC_YEO7_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.85, 0.25, 0.90),  # Visual — magenta
    2: (0.30, 0.75, 0.95),  # Somatomotor — cyan
    3: (0.20, 0.85, 0.35),  # Dorsal Attn — green
    4: (0.95, 0.55, 0.15),  # Ventral Attn — orange
    5: (0.95, 0.90, 0.25),  # Limbic — yellow
    6: (0.35, 0.45, 0.95),  # Control — blue
    7: (0.95, 0.30, 0.35),  # Default — red
}
_QC_WASH_ALPHA = 0.62
_QC_ALL_ALPHA = 0.48


def write_parcellation_qc_sheet(
    *,
    b0_path: str,
    labels_path: Path,
    lut: dict[int, dict],
    out_png: Path,
) -> str:
    """Human-readable QC sheet: bright network washes on b0 (not the 3D ghost palette).

    Picks the axial slice with max labeled cortex so panels are not empty mid-brain.
    Panel 0 composites all networks; panels 1–7 are single-network territory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .atlas_fetch import YEO7_NETWORKS

    b0 = np.asanyarray(nib.load(b0_path).dataobj, dtype=np.float64)
    labels = np.asanyarray(nib.load(str(labels_path)).dataobj).astype(np.int32)
    # prefer the axial with most cortical labels (mid-slab often looks empty)
    lab_per_z = (labels > 0).sum(axis=(0, 1))
    k = int(np.argmax(lab_per_z)) if lab_per_z.max() > 0 else b0.shape[2] // 2
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), facecolor="#0d0f12")
    axes = axes.ravel()
    sl = np.rot90(b0[:, :, k])
    sl = sl / (np.percentile(sl[sl > 0], 98) + 1e-9) if (sl > 0).any() else sl
    sl_clip = np.clip(sl, 0, 1)

    def _network_mask_slice(nid: int) -> np.ndarray:
        keep = {lab for lab, m in lut.items() if int(m.get("network_id") or 0) == nid}
        return np.rot90(np.isin(labels[:, :, k], list(keep)).astype(float))

    def _wash(ax, cont: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
        rgba = np.zeros(cont.shape + (4,))
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = np.where(cont > 0, alpha, 0.0)
        ax.imshow(rgba, origin="lower")
        if cont.any():
            ax.contour(cont, levels=[0.5], colors=[color], linewidths=0.9, origin="lower")

    # panel 0: b0 + all networks wash
    axes[0].imshow(sl_clip, cmap="gray", origin="lower")
    for nid in range(1, 8):
        cont = _network_mask_slice(nid)
        c = _QC_YEO7_COLORS.get(nid, (0.7, 0.7, 0.7))
        _wash(axes[0], cont, c, _QC_ALL_ALPHA)
    axes[0].set_title(f"all networks (z={k})", color="#e8c56a", fontsize=9)
    axes[0].axis("off")

    for nid in range(1, 8):
        ax = axes[nid]
        ax.imshow(sl_clip, cmap="gray", origin="lower")
        cont = _network_mask_slice(nid)
        c = _QC_YEO7_COLORS.get(nid, (0.7, 0.7, 0.7))
        _wash(ax, cont, c, _QC_WASH_ALPHA)
        n_vox = int(cont.sum())
        ax.set_title(
            f"{YEO7_NETWORKS.get(nid, str(nid))} ({n_vox})",
            color="#ccc",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(
        "Parcellation prior (Schaefer-200/Yeo-7) on b0 — POPULATION · not patient",
        color="#e8c56a",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return sha256_file(out_png)


def prepare_parcellation(
    *,
    manifest_path: str,
    reuse_warp: bool = True,
    full_syn: bool = False,
) -> dict:
    from .atlas_fetch import (
        fetch_schaefer200_yeo7,
        load_schaefer_lut,
        sha256_file as fetch_sha,
    )

    man_path = Path(manifest_path)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    layout = {**man, "case_root": str(root)}
    inputs = derivation.active_inputs(man)
    t1 = root / inputs["t1"]["path"]
    b0 = root / inputs["b0"]["path"]
    mask = root / inputs["mask"]["path"]
    norm = derivation.artifact_dir(layout, "normative")
    pin_path = derivation.artifact_path(layout, "normative", "schaefer200_pin.json")
    lut_case = derivation.artifact_path(
        layout, "normative", "Schaefer2018_200Parcels_7Networks_order.txt"
    )
    prefix = derivation.artifact_path(layout, "normative", "mni2case_")
    warp = derivation.artifact_path(layout, "normative", "mni2case_1Warp.nii.gz")
    affine = derivation.artifact_path(layout, "normative", "mni2case_0GenericAffine.mat")
    out_labels = derivation.artifact_path(
        layout, "normative", "parc_schaefer200_yeo7.nii.gz"
    )
    sheet = derivation.artifact_path(layout, "qc", "parcellation_prior_qc.png")
    if derivation.qc_block_for(man, "parcellation_qc") is not None:
        derivation.set_qc_block(man, "parcellation_qc", {
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "voided": True,
            "note": "voided by parcellation prep regeneration",
        })
        derivation.write_manifest_atomic(man_path, man)
    norm.mkdir(parents=True, exist_ok=True)

    nii_mni, lut_src = fetch_schaefer200_yeo7()
    # pin observed hashes into a small case-local pin file for audit
    pin = {
        "nii_sha256": fetch_sha(nii_mni),
        "lut_sha256": fetch_sha(lut_src),
        "source": "CBIG Schaefer2018 200/7 FSLMNI152 1mm",
    }
    pin_path.write_text(
        json.dumps(pin, indent=2) + "\n"
    )
    # re-fetch with pin to enforce
    nii_mni, lut_src = fetch_schaefer200_yeo7(
        expected_nii_sha=pin["nii_sha256"],
        expected_lut_sha=pin["lut_sha256"],
    )
    lut = load_schaefer_lut(lut_src)
    # copy LUT into case for redistributable serve
    shutil.copy2(lut_src, lut_case)

    if not (reuse_warp and warp.is_file() and affine.is_file()):
        warp, affine = ensure_mni_to_case_warp(
            t1_path=str(t1), out_prefix=prefix, full_syn=full_syn,
        )
        reg_tool = "antsRegistrationSyN.sh" if full_syn else "antsRegistrationSyNQuick.sh"
    else:
        reg_tool = "existing warp (reuse)"

    apply_label_warp_nn(
        labels_mni=nii_mni,
        t1_path=str(t1),
        warp=warp,
        affine=affine,
        out_labels=out_labels,
    )
    # Grid check: labels must match tracking grid (same as T1 on this case)
    g_t1 = load_grid(str(t1))
    g_b0 = load_grid(str(b0))
    if g_t1.shape != g_b0.shape or not np.allclose(g_t1.affine, g_b0.affine, atol=1e-3):
        raise ValueError("T1 grid ≠ tracking grid — refuse silent label resample")
    g_lab = load_grid(str(out_labels))
    if g_lab.shape != g_b0.shape or not np.allclose(g_lab.affine, g_b0.affine, atol=1e-3):
        raise ValueError("warped labels grid mismatch vs b0")

    # NN tripwire: unique positive labels must be subset of LUT (no fractional junk)
    lab_vol = np.asanyarray(nib.load(str(out_labels)).dataobj).astype(np.int32)
    uniq = set(int(x) for x in np.unique(lab_vol) if int(x) > 0)
    if not uniq.issubset(set(lut.keys())):
        raise ValueError(
            f"NN tripwire FAIL: labels outside LUT (e.g. {sorted(uniq - set(lut.keys()))[:5]})"
        )

    auto = auto_qc_parcellation(
        labels_path=out_labels, mask_path=str(mask), lut=lut,
    )
    sheet_sha = write_parcellation_qc_sheet(
        b0_path=str(b0), labels_path=out_labels, lut=lut, out_png=sheet,
    )

    # The previous signature was cleared atomically before any output write.
    # Keep the active block explicitly unsigned while recording this result.
    derivs = man.get("derivations")
    active = derivation.active_derivation(man)
    nonfirst = isinstance(derivs, dict) and len(derivs) > 1 and active != next(iter(derivs))
    lut_bytes = int(lut_case.stat().st_size)
    lut_sha = sha256_file(lut_case)
    inputs["parc_schaefer200_yeo7"] = {
        "path": out_labels.resolve().relative_to(root.resolve()).as_posix(),
        "lut_path": lut_case.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": int(out_labels.stat().st_size),
        "sha256": sha256_file(out_labels),
        "label": "Schaefer-200 / Yeo-7 parcellation prior",
        "n_parcels": 200,
        "n_networks": 7,
        "interpolation": "NearestNeighbor",
        "registration": (
            f"{reg_tool}; fixed=t1c; moving=MNI; Schaefer200/Yeo7; not navigation"
        ),
        "note": "POPULATION territory prior · not patient · not navigation",
        "atlas_pin": pin,
    }
    if nonfirst:
        inputs["parc_schaefer200_yeo7"]["sha256_by_field"] = {
            "path": inputs["parc_schaefer200_yeo7"]["sha256"],
            "lut_path": lut_sha,
        }
        inputs["parc_schaefer200_yeo7"]["bytes_by_field"] = {
            "path": inputs["parc_schaefer200_yeo7"]["bytes"],
            "lut_path": lut_bytes,
        }
    derivation.set_qc_block(man, "parcellation_qc", {
        "auto": {
            "ok": auto.ok,
            "label_complete": auto.label_complete,
            "lut_ok": auto.lut_ok,
            "hull_ok": auto.hull_ok,
            "symmetry_ok": auto.symmetry_ok,
            "n_labels": auto.n_labels,
            "hull_frac": auto.hull_frac,
            "lr_ratio": auto.lr_ratio,
            "notes": list(auto.notes),
        },
        "approved_by": None,
        "date": None,
        "sheet_sha": sheet_sha,
        "sheet_path": sheet.relative_to(root).as_posix(),
    })
    derivation.write_manifest_atomic(man_path, man)
    return {
        "auto_ok": auto.ok,
        "auto_notes": list(auto.notes),
        "sheet": str(sheet),
        "sheet_sha": sheet_sha,
        "n_labels": auto.n_labels,
        "interpolation": "NearestNeighbor",
        "registration": reg_tool,
    }


def approve_parcellation_qc(
    manifest_path: str,
    *,
    approved_by: str,
    sheet_sha: str | None = None,
) -> dict:
    man_path = Path(manifest_path)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    layout = {**man, "case_root": str(root)}
    sheet = derivation.artifact_path(layout, "qc", "parcellation_prior_qc.png")
    if not sheet.is_file():
        raise FileNotFoundError(sheet)
    live = sha256_file(sheet)
    if sheet_sha and sheet_sha != live:
        raise ValueError("sheet_sha mismatch")
    qc = derivation.qc_block_for(layout, "parcellation_qc") or {}
    auto = qc.get("auto") or {}
    if not auto.get("ok", False):
        raise ValueError("auto QC blocked — will not record human approval")
    inputs = derivation.active_inputs(man)
    meta = inputs.get("parc_schaefer200_yeo7")
    if not meta or not (root / meta["path"]).is_file():
        raise ValueError("missing parc_schaefer200_yeo7")
    result = derivation.set_qc_block(man, "parcellation_qc", {
        "auto": auto,
        "approved_by": approved_by,
        "date": date.today().isoformat(),
        "sheet_sha": live,
        "sheet_path": sheet.relative_to(root).as_posix(),
    })
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    # default tracts path (backward compatible): no subcommand
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--approve", metavar="WHO")
    ap.add_argument("--sheet-sha", default=None)
    ap.add_argument("--full-syn", action="store_true")
    ap.add_argument("--no-reuse-warp", action="store_true")
    ap.add_argument(
        "--parcellation",
        action="store_true",
        help="prepare Schaefer-200/Yeo-7 parcellation prior (A2)",
    )
    ap.add_argument(
        "--approve-parcellation",
        metavar="WHO",
        help="human sign-off for parcellation_qc",
    )

    args = ap.parse_args(argv)
    if not args.manifest:
        print("--manifest required", file=sys.stderr)
        return 2

    if args.approve_parcellation:
        try:
            rec = approve_parcellation_qc(
                args.manifest,
                approved_by=args.approve_parcellation,
                sheet_sha=args.sheet_sha,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"APPROVE FAILED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(rec, indent=2))
        return 0

    if args.parcellation:
        try:
            summary = prepare_parcellation(
                manifest_path=args.manifest,
                reuse_warp=not args.no_reuse_warp,
                full_syn=args.full_syn,
            )
        except Exception as e:  # noqa: BLE001
            print(f"PARCELLATION PREP FAILED: {e}", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("auto_ok") else 1

    if args.approve:
        try:
            rec = approve_atlas_qc(
                args.manifest, approved_by=args.approve, sheet_sha=args.sheet_sha,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"APPROVE FAILED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(rec, indent=2))
        return 0

    try:
        summary = prepare(
            manifest_path=args.manifest,
            full_syn=args.full_syn,
            reuse_warp=not args.no_reuse_warp,
        )
    except Exception as e:  # noqa: BLE001
        print(f"PREP FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("auto_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
