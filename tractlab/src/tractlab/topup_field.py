"""Standalone topup field (Hz) planner — does not re-run eddy / dwifslpreproc.

The D0 house mask lives on the 1.3 mm upsampled grid. The SE-EPI pair and
native DWI share a coarser grid. ``compute_delta`` multiplies |Hz| by the
*field* header zoom, so the field must stay on the native SE grid. Resampling
it onto ``mask_up`` would silently shrink every millimetre.

Default is dry-run. ``--apply`` is the only path that writes or calls FSL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from .preproc import NoReversePE

_PE = {
    "i": (1, 0, 0),
    "i-": (-1, 0, 0),
    "j": (0, 1, 0),
    "j-": (0, -1, 0),
    "k": (0, 0, 1),
    "k-": (0, 0, -1),
}

FSLMERGE = "fslmerge"
FSLMATHS = "fslmaths"
BET = "bet"
TOPUP = "topup"


@dataclass(frozen=True)
class TopupFieldPlan:
    case_root: Path
    out_dir: Path
    field_path: Path
    mask_path: Path
    acqparams_path: Path
    acqparams_text: str
    commands: list[list[str]]
    source: dict


@dataclass
class TopupFieldResult:
    ok: bool
    ran: bool
    failed_cmd: list[str] | None
    receipt_path: Path | None


def bids_pe_to_fsl(pe_dir: str) -> tuple[int, int, int]:
    key = str(pe_dir).strip()
    if key not in _PE:
        raise ValueError(f"unknown PhaseEncodingDirection {pe_dir!r}")
    return _PE[key]


def acqparams_text(*, ap_pe: str, pa_pe: str, readout_s: float, n_ap: int, n_pa: int) -> str:
    ap = bids_pe_to_fsl(ap_pe)
    pa = bids_pe_to_fsl(pa_pe)
    trt = f"{float(readout_s):.5f}".rstrip("0").rstrip(".")
    if trt == "-0":
        trt = "0"
    lines = [f"{ap[0]} {ap[1]} {ap[2]} {trt}" for _ in range(n_ap)]
    lines += [f"{pa[0]} {pa[1]} {pa[2]} {trt}" for _ in range(n_pa)]
    return "\n".join(lines) + "\n"


def _n_vols(path: Path) -> int:
    shape = nib.load(str(path)).header.get_data_shape()
    return int(shape[3]) if len(shape) > 3 else 1


def _sidecar(nii: Path) -> dict:
    name = nii.name
    base = nii.with_name(name[: -len(".nii.gz")]) if name.endswith(".nii.gz") else nii.with_suffix("")
    return json.loads(base.with_suffix(".json").read_text())


def _b02b0() -> Path:
    fsldir = os.environ.get("FSLDIR", "")
    return Path(fsldir) / "etc" / "flirtsch" / "b02b0.cnf"


def plan_topup_field(case_root: Path) -> TopupFieldPlan | NoReversePE:
    case_root = Path(case_root)
    manifest = json.loads((case_root / "manifest.json").read_text())
    raw = manifest.get("raw") or {}
    fmap_ap = raw.get("fmap_dwi_ap")
    fmap_pa = raw.get("fmap_dwi_pa")
    if not fmap_ap or not fmap_pa:
        return NoReversePE(
            reason="reverse-PE fieldmaps missing (raw.fmap_dwi_ap / "
            "raw.fmap_dwi_pa required for topup); refusing to plan"
        )
    ap_path = case_root / fmap_ap["path"]
    pa_path = case_root / fmap_pa["path"]
    ap_js = _sidecar(ap_path)
    pa_js = _sidecar(pa_path)
    ap_pe = ap_js["PhaseEncodingDirection"]
    pa_pe = pa_js["PhaseEncodingDirection"]
    if bids_pe_to_fsl(ap_pe) != tuple(-x for x in bids_pe_to_fsl(pa_pe)):
        raise ValueError(
            f"AP/PA PhaseEncodingDirection are not opposite: {ap_pe!r} vs {pa_pe!r}"
        )
    ap_trt = float(ap_js["TotalReadoutTime"])
    pa_trt = float(pa_js["TotalReadoutTime"])
    if ap_trt != pa_trt:
        raise ValueError(
            f"AP/PA TotalReadoutTime disagree: {ap_trt} vs {pa_trt}"
        )
    n_ap = _n_vols(ap_path)
    n_pa = _n_vols(pa_path)
    text = acqparams_text(
        ap_pe=ap_pe, pa_pe=pa_pe, readout_s=ap_trt, n_ap=n_ap, n_pa=n_pa,
    )
    out_dir = case_root / "work" / "topup_field" / "d1"
    imain = out_dir / "imain.nii.gz"
    mean_b0 = out_dir / "mean_b0.nii.gz"
    brain = out_dir / "brain"
    field = out_dir / "field_hz.nii.gz"
    mask = out_dir / "mask.nii.gz"
    acq = out_dir / "acqparams.txt"
    topup_out = out_dir / "topup_out"
    config = _b02b0()
    commands = [
        [FSLMERGE, "-t", str(imain), str(ap_path), str(pa_path)],
        [FSLMATHS, str(imain), "-Tmean", str(mean_b0)],
        [BET, str(mean_b0), str(brain), "-f", "0.3", "-n", "-m"],
        [FSLMATHS, str(out_dir / "brain_mask.nii.gz"), "-bin", str(mask)],
        [
            TOPUP,
            f"--imain={imain}",
            f"--datain={acq}",
            f"--config={config}",
            f"--out={topup_out}",
            f"--fout={field}",
        ],
    ]
    return TopupFieldPlan(
        case_root=case_root,
        out_dir=out_dir,
        field_path=field,
        mask_path=mask,
        acqparams_path=acq,
        acqparams_text=text,
        commands=commands,
        source={
            "ap": str(ap_path),
            "pa": str(pa_path),
            "ap_pe": ap_pe,
            "pa_pe": pa_pe,
            "readout_s": ap_trt,
            "n_ap": n_ap,
            "n_pa": n_pa,
        },
    )


def validate_field_hz(
    field_path: Path,
    *,
    expected_shape: tuple[int, ...],
    receipt: dict,
) -> None:
    if receipt.get("units") != "Hz":
        raise ValueError("field receipt units must be Hz (FSL topup --fout)")
    img = nib.load(str(field_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {data.shape}")
    if tuple(int(x) for x in data.shape) != tuple(int(x) for x in expected_shape):
        raise ValueError(f"field shape {data.shape} != expected {expected_shape}")
    if not np.isfinite(data).any():
        raise ValueError("field has no finite voxels")
    if np.nanmax(np.abs(data)) == 0:
        raise ValueError("field is all-zero — topup --fout did not write a field")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_topup_field(
    plan: TopupFieldPlan,
    *,
    apply: bool,
    runner: Callable = subprocess.run,
) -> TopupFieldResult:
    receipt_path = plan.out_dir / "receipt.json"
    if not apply:
        return TopupFieldResult(ok=True, ran=False, failed_cmd=None, receipt_path=None)
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    plan.acqparams_path.write_text(plan.acqparams_text)
    for cmd in plan.commands:
        result = runner(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return TopupFieldResult(
                ok=False, ran=True, failed_cmd=cmd, receipt_path=None,
            )
    receipt = {
        "units": "Hz",
        "tool": "topup",
        "--fout": True,
        "source": plan.source,
        "ap_sha256": _sha256(Path(plan.source["ap"])),
        "pa_sha256": _sha256(Path(plan.source["pa"])),
        "field": str(plan.field_path),
        "mask": str(plan.mask_path),
        "commands": plan.commands,
    }
    ap_img = nib.load(plan.source["ap"])
    expected = tuple(int(x) for x in ap_img.header.get_data_shape()[:3])
    validate_field_hz(plan.field_path, expected_shape=expected, receipt=receipt)
    receipt["field_sha256"] = _sha256(plan.field_path)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return TopupFieldResult(ok=True, ran=True, failed_cmd=None, receipt_path=receipt_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tractlab.topup_field",
        description="Plan (default) or --apply a native-grid topup field in Hz",
    )
    ap.add_argument("case_root")
    ap.add_argument("--apply", action="store_true",
                    help="run FSL topup; default is dry-run (print plan only)")
    args = ap.parse_args(argv)
    planned = plan_topup_field(Path(args.case_root))
    if isinstance(planned, NoReversePE):
        print(f"Error: {planned.reason}", file=sys.stderr)
        return 1
    payload = {
        "dry_run": not args.apply,
        "field": str(planned.field_path),
        "mask": str(planned.mask_path),
        "acqparams": planned.acqparams_text,
        "source": planned.source,
        "commands": planned.commands,
        "note": (
            "field stays on the SE-EPI/native DWI grid; do not resample onto "
            "nifti/mask_up.nii.gz. delta_qc must be called with --mask pointing "
            "at this native mask."
        ),
    }
    print(json.dumps(payload, indent=2))
    if not args.apply:
        return 0
    res = run_topup_field(planned, apply=True)
    if not res.ok:
        print(f"FAILED: {res.failed_cmd}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "receipt": str(res.receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
