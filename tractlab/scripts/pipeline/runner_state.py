#!/usr/bin/env python3
"""Case config and atomic state helpers for run_case.sh."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("ss3t", "t1reg", "banks", "scalar_maps")
CASE_ID_RE = re.compile(r"^case-[0-9a-f]{8}$")
PE_DIRS = {"i", "i-", "j", "j-", "k", "k-"}
FILTER_BANK_PATH = (
    "tracts/connectome/pre/freesurfer/path-to-wm-v2/"
    "wholebrain_act_ifod2_100000.tck"
)


class RunnerError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _case_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RunnerError(f"case root is not a directory: {root}")
    return root


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _contained_input(root: Path, rel: object, label: str) -> Path:
    if not isinstance(rel, str) or not rel or "\n" in rel or "\r" in rel:
        raise RunnerError(f"case.json inputs.{label} must be a single-line relative path")
    candidate = Path(rel)
    if candidate.is_absolute():
        raise RunnerError(f"case.json inputs.{label} must be relative to the case root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RunnerError(f"case.json inputs.{label} escapes the case root") from exc
    if not resolved.is_file():
        raise RunnerError(f"case.json input is missing: {label}")
    return resolved


def _read_config(root_arg: str | Path) -> dict[str, str]:
    root = _case_root(root_arg)
    case = _load_json(root / "case.json", "case.json")
    if case.get("schema") != "tractlab.case/1":
        raise RunnerError("case.json schema must be tractlab.case/1")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
        raise RunnerError("case.json case_id must be case- followed by eight lowercase hex digits")
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise RunnerError("case.json inputs must be an object")
    dwi = inputs.get("dwi")
    t1 = inputs.get("t1")
    rpe = inputs.get("rpe")
    if not isinstance(dwi, dict) or not isinstance(t1, dict) or not isinstance(rpe, dict):
        raise RunnerError("case.json must include dwi, rpe, and t1 input objects")

    dwi_nifti = _contained_input(root, dwi.get("nifti"), "dwi.nifti")
    dwi_bval = _contained_input(root, dwi.get("bval"), "dwi.bval")
    dwi_bvec = _contained_input(root, dwi.get("bvec"), "dwi.bvec")
    dwi_json_path = _contained_input(root, dwi.get("json"), "dwi.json")
    t1_nifti = _contained_input(root, t1.get("nifti"), "t1.nifti")
    rpe_mode = rpe.get("mode")
    if not isinstance(rpe_mode, str) or rpe_mode not in {"pair", "none"}:
        raise RunnerError("case.json inputs.rpe.mode must be pair or none")

    pe_dir = dwi.get("pe_dir")
    readout = dwi.get("total_readout_time")
    if not isinstance(pe_dir, str) or pe_dir not in PE_DIRS:
        raise RunnerError("case.json inputs.dwi.pe_dir must be a BIDS phase-encoding direction")
    if isinstance(readout, bool) or not isinstance(readout, (float, int)) or not math.isfinite(readout) or readout <= 0:
        raise RunnerError("case.json inputs.dwi.total_readout_time must be a positive number")

    regrid = dwi.get("regrid_voxel_mm")
    regrid_value = ""
    if regrid is not None:
        if (
            not isinstance(regrid, list)
            or len(regrid) != 3
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                for v in regrid
            )
        ):
            raise RunnerError("case.json inputs.dwi.regrid_voxel_mm must be null or three positive numbers")
        regrid_value = ",".join(f"{float(v):g}" for v in regrid)

    rpe_nifti = ""
    rpe_json_path = ""
    pair_header = False
    if rpe_mode == "pair":
        rpe_nifti = str(_contained_input(root, rpe.get("nifti"), "rpe.nifti"))
        rpe_json_path_obj = _contained_input(root, rpe.get("json"), "rpe.json")
        rpe_json_path = str(rpe_json_path_obj)
        dwi_sidecar = _load_json(dwi_json_path, "DWI sidecar")
        rpe_sidecar = _load_json(rpe_json_path_obj, "reverse-PE sidecar")
        pair_header = all(
            isinstance(sidecar.get("TotalReadoutTime"), (float, int))
            and not isinstance(sidecar.get("TotalReadoutTime"), bool)
            and math.isfinite(sidecar["TotalReadoutTime"])
            and sidecar["TotalReadoutTime"] > 0
            and sidecar.get("PhaseEncodingDirection") in PE_DIRS
            for sidecar in (dwi_sidecar, rpe_sidecar)
        )

    manifest = _load_json(root / "manifest.json", "manifest.json")
    if manifest.get("case_id") != case_id:
        raise RunnerError("manifest.json case_id does not match case.json")
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise RunnerError("manifest.json inputs must be an object")
    expected_outputs = {
        "b0": "nifti/b0.nii.gz",
        "fod": "nifti/wmfod_norm.mif",
        "mask": "nifti/mask_up.nii.gz",
        "t1": "nifti/t1_brain_dwi.nii.gz",
    }
    for key, rel in expected_outputs.items():
        meta = manifest_inputs.get(key)
        if not isinstance(meta, dict) or meta.get("path") != rel:
            raise RunnerError(f"manifest.json inputs.{key}.path must be {rel}")

    return {
        "case_id": case_id,
        "dwi_nifti": str(dwi_nifti),
        "dwi_bval": str(dwi_bval),
        "dwi_bvec": str(dwi_bvec),
        "dwi_json": str(dwi_json_path),
        "pe_dir": str(pe_dir),
        "readout_time": str(readout),
        "rpe_mode": str(rpe_mode),
        "rpe_nifti": rpe_nifti,
        "rpe_json": rpe_json_path,
        "t1_nifti": str(t1_nifti),
        "pair_header": "1" if pair_header else "0",
        "regrid_voxel": regrid_value,
    }


_GRADCHECK_ROW = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+(none|[012])\s+\((\d),\s*(\d),\s*(\d)\)\s+(scanner|image)\s*$"
)


def _gradcheck(root: Path, report_path: Path) -> None:
    """Parse dwigradcheck output; fail closed unless the unaltered table wins."""
    try:
        text = report_path.read_text(errors="replace")
    except OSError as exc:
        raise RunnerError(f"cannot read dwigradcheck report: {exc}") from exc
    rows = []
    for line in text.splitlines():
        match = _GRADCHECK_ROW.match(line)
        if match:
            rows.append(
                {
                    "mean_length": float(match.group(1)),
                    "flip": match.group(2),
                    "permutation": [int(match.group(i)) for i in (3, 4, 5)],
                    "basis": match.group(6),
                }
            )
    if not rows:
        raise RunnerError("dwigradcheck report has no result table")
    top = rows[0]
    identity = top["flip"] == "none" and top["permutation"] == [0, 1, 2]
    allow = os.environ.get("TRACTLAB_ALLOW_GRAD_FLIP") == "1"
    verdict = "pass" if identity else ("override" if allow else "fail")
    _atomic_json(
        root / "work" / "ss3t" / "gradcheck.json",
        {"schema": "tractlab.gradcheck/1", "verdict": verdict, "top": rows[:3]},
    )
    if identity:
        print(f"gradcheck OK: unaltered gradient table wins (mean length {top['mean_length']:g})")
        return
    described = (
        f"flip {top['flip']}, permutation ({', '.join(map(str, top['permutation']))}), "
        f"{top['basis']} basis, mean length {top['mean_length']:g}"
    )
    if not allow:
        raise RunnerError(
            f"dwigradcheck prefers an altered gradient table ({described}); check the scanner export. "
            "To proceed anyway, rerun with TRACTLAB_ALLOW_GRAD_FLIP=1 (recorded as a case warning)."
        )
    case_path = root / "case.json"
    case = _load_json(case_path, "case.json")
    warnings = case.setdefault("warnings", [])
    warnings.append(f"dwigradcheck preferred an altered gradient table ({described}); overridden by the operator.")
    _atomic_json(case_path, case)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_status(root: Path, action: str, stage: str | None = None) -> None:
    path = root / "status.json"
    if action == "init":
        payload = {
            "schema": "tractlab.status/1",
            "state": "running",
            "stage": STAGES[0],
            "stages": [
                {"name": name, "state": "pending", "started_utc": None, "ended_utc": None}
                for name in STAGES
            ],
            "error": None,
        }
        _atomic_json(path, payload)
        return

    payload = _load_json(path, "status.json")
    if action == "finish":
        payload.update({"state": "done", "stage": STAGES[-1], "error": None})
        _atomic_json(path, payload)
        return

    if stage not in STAGES:
        raise RunnerError(f"invalid pipeline stage: {stage}")
    payload["stage"] = stage
    if action == "start":
        payload["state"] = "running"
        payload["error"] = None
        for record in payload["stages"]:
            if record["name"] == stage:
                record.update({"state": "running", "started_utc": _now(), "ended_utc": None})
                break
    elif action in {"complete", "fail"}:
        failed = action == "fail"
        for record in payload["stages"]:
            if record["name"] == stage:
                record["state"] = "failed" if failed else "done"
                if record.get("started_utc") is None:
                    record["started_utc"] = _now()
                record["ended_utc"] = _now()
                break
        if failed:
            payload["state"] = "failed"
            log_path = root / "work" / f"{stage}.log"
            try:
                lines = log_path.read_text(errors="replace").splitlines()[-20:]
            except OSError:
                lines = []
            payload["error"] = "\n".join(lines) or f"Stage {stage} failed without log output"
        else:
            payload["state"] = "running"
            payload["error"] = None
    else:
        raise RunnerError(f"invalid status action: {action}")
    _atomic_json(path, payload)


def _manifest_root(root: Path) -> None:
    path = root / "manifest.json"
    manifest = _load_json(path, "manifest.json")
    case = _load_json(root / "case.json", "case.json")
    if manifest.get("case_id") != case.get("case_id"):
        raise RunnerError("manifest.json case_id does not match case.json")
    manifest["case_root"] = str(root)
    _atomic_json(path, manifest)


def _manifest_banks(root: Path, rpe_mode: str) -> None:
    if rpe_mode not in {"pair", "none"}:
        raise RunnerError("reverse-PE mode must be pair or none")
    path = root / "manifest.json"
    manifest = _load_json(path, "manifest.json")
    inputs = manifest.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise RunnerError("manifest.json inputs must be an object")

    if rpe_mode == "none":
        honesty = "Research/preview only — not navigation. no reverse-PE: residual EPI distortion uncorrected."
    else:
        honesty = "Research/preview only — not navigation. Reverse-PE pair used; residual EPI distortion may remain."
    for key, meta in list(inputs.items()):
        if key.startswith("bank_") and isinstance(meta, dict):
            note = str(meta.get("note") or "")
            base = note.split(" Research only", 1)[0].split(" Research/preview only", 1)[0].rstrip()
            meta["note"] = (base + " " + honesty).strip()
    inputs["filter_bank"] = {
        "path": FILTER_BANK_PATH,
        "n_streamlines": 100000,
        "engine": "ACT iFOD2 whole-brain 100k (subset of 10M)",
        "note": "Live Explore filter corpus. FILTERED PREVIEW. " + honesty,
    }
    manifest["status"] = "banks-built-unsigned-priors"
    manifest["phase0"] = "SS3T FODs and ACT 10M banks built; atlas/parcellation QC unsigned."
    _atomic_json(path, manifest)


def _laterality(aparc_path: str, b0_path: str) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RunnerError("laterality check requires nibabel and numpy") from exc
    aparc = nib.load(aparc_path)
    b0 = nib.load(b0_path)
    labels = np.asanyarray(aparc.dataobj)
    if aparc.shape != b0.shape:
        raise RunnerError(f"aparc_dwi shape {aparc.shape} != b0 {b0.shape}")

    def centroid_x(label: int) -> float:
        voxels = np.argwhere(labels == label)
        if voxels.size == 0:
            raise RunnerError(f"empty aparc label {label}")
        return float(nib.affines.apply_affine(aparc.affine, voxels.mean(axis=0))[0])

    left_x = centroid_x(1024)
    right_x = centroid_x(2024)
    print(f"precentral L x={left_x:.2f} R x={right_x:.2f} (RAS: R>L expected)")
    if not left_x + 10 < right_x:
        raise RunnerError(f"laterality fail: L x={left_x:.2f} not left of R x={right_x:.2f}")
    print("laterality OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config")
    config.add_argument("case_root")

    status = sub.add_parser("status")
    status.add_argument("action", choices=("init", "start", "complete", "fail", "finish"))
    status.add_argument("case_root")
    status.add_argument("stage", nargs="?")

    manifest_root = sub.add_parser("manifest-root")
    manifest_root.add_argument("case_root")

    manifest_banks = sub.add_parser("manifest-banks")
    manifest_banks.add_argument("case_root")
    manifest_banks.add_argument("rpe_mode", choices=("pair", "none"))

    gradcheck = sub.add_parser("gradcheck")
    gradcheck.add_argument("case_root")
    gradcheck.add_argument("report")

    laterality = sub.add_parser("laterality")
    laterality.add_argument("aparc")
    laterality.add_argument("b0")

    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            values = _read_config(args.case_root)
            for key in (
                "case_id", "dwi_nifti", "dwi_bval", "dwi_bvec", "dwi_json",
                "pe_dir", "readout_time", "rpe_mode", "rpe_nifti", "rpe_json",
                "t1_nifti", "pair_header", "regrid_voxel",
            ):
                sys.stdout.buffer.write(values[key].encode() + b"\0")
        elif args.command == "status":
            root = _case_root(args.case_root)
            _write_status(root, args.action, args.stage)
        elif args.command == "manifest-root":
            _manifest_root(_case_root(args.case_root))
        elif args.command == "manifest-banks":
            _manifest_banks(_case_root(args.case_root), args.rpe_mode)
        elif args.command == "gradcheck":
            _gradcheck(_case_root(args.case_root), Path(args.report))
        elif args.command == "laterality":
            _laterality(args.aparc, args.b0)
    except (RunnerError, OSError, ValueError) as exc:
        print(f"pipeline runner: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
