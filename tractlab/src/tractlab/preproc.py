"""Generic reverse-PE preprocessing planner (ADR-0004 topup slice).

Mirrors ``scripts/run_demo_d0_ss3t.sh``'s denoise -> degibbs -> dwifslpreproc
chain, but generic to any case manifest instead of the hardcoded D0 demo
paths, and stops short of the SS3T/CSD stages (out of scope here).

Security + honesty rules (same idiom as ``track.py``):

  * argv list only — never a shell string. Tool names, flags, and paths are
    built from the case manifest and its DWI JSON sidecar; nothing is
    interpolated into a shell.
  * Readout time and phase-encode direction are read from the DWI JSON
    sidecar VERBATIM and passed straight through to ``dwifslpreproc``.
    They are never hardcoded, guessed, or normalized.
  * Fail-closed: a case missing either reverse-PE fieldmap (``fmap_dwi_ap``,
    ``fmap_dwi_pa``) never gets a silently-uncorrected plan. It gets a typed
    ``NoReversePE`` outcome naming "fmap" in the reason, so callers cannot
    mistake "we skipped topup" for "topup ran".
  * ``run_preproc`` stops at the first nonzero return code and names the
    failed command — it never runs step N+1 after step N failed.

Fix round 1 (task-5-report.md): the plan is FOUR commands, not three —
``dwidenoise``, ``mrdegibbs``, ``mrcat`` (builds the AP+PA reverse-PE pair),
``dwifslpreproc`` (consumes that pair via ``-se_epi``). The original 3-command
scope left ``-rpe_pair`` un-fed and was ruled a plan defect.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DWIDENOISE = "dwidenoise"
MRDEGIBBS = "mrdegibbs"
MRCAT = "mrcat"
DWIFSLPREPROC = "dwifslpreproc"

# MRtrix requires this exact leading space inside the -eddy_options value;
# it must stay ONE argv token (never split on the space).
EDDY_OPTIONS = " --slm=linear --repol --data_is_shelled"


@dataclass(frozen=True)
class PreprocPlan:
    case_root: Path
    out_dir: Path
    commands: list[list[str]]


@dataclass(frozen=True)
class NoReversePE:
    reason: str


@dataclass
class PreprocResult:
    ok: bool
    failed_cmd: list[str] | None
    log_path: Path


def _strip_nii_gz(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[: -len(".nii.gz")])
    return path.with_suffix("")


def plan_preproc(case_root: Path) -> PreprocPlan | NoReversePE:
    """Read the manifest + DWI JSON sidecar and build the argv plan.

    Fail-closed: missing reverse-PE fieldmaps never produce a plan.
    """
    manifest = json.loads((case_root / "manifest.json").read_text())
    raw = manifest.get("raw", {})
    dwi = raw.get("dwi")
    if not dwi or "path" not in dwi:
        raise ValueError("manifest raw.dwi.path is required")

    fmap_ap = raw.get("fmap_dwi_ap")
    fmap_pa = raw.get("fmap_dwi_pa")
    if not fmap_ap or not fmap_pa:
        return NoReversePE(
            reason="reverse-PE fieldmaps missing (raw.fmap_dwi_ap / "
            "raw.fmap_dwi_pa required for topup); refusing to plan an "
            "uncorrected preprocessing run"
        )
    fmap_ap_path = case_root / fmap_ap["path"]
    fmap_pa_path = case_root / fmap_pa["path"]

    dwi_path = case_root / dwi["path"]
    base = _strip_nii_gz(dwi_path)
    bvec = base.with_suffix(".bvec")
    bval = base.with_suffix(".bval")
    sidecar = json.loads(base.with_suffix(".json").read_text())
    pe_dir = sidecar["PhaseEncodingDirection"]
    readout_time = sidecar["TotalReadoutTime"]

    out_dir = case_root / "work" / "preproc" / "d1"
    dwi_den = out_dir / "dwi_den.mif"
    dwi_deg = out_dir / "dwi_den_deg.mif"
    dwi_preproc = out_dir / "dwi_preproc.mif"
    noise = out_dir / "noise.mif"
    se_pair = out_dir / "se_pair.mif"
    eddyqc = out_dir / "eddyqc"

    commands = [
        [DWIDENOISE, str(dwi_path), str(dwi_den), "-noise", str(noise)],
        [MRDEGIBBS, str(dwi_den), str(dwi_deg)],
        [MRCAT, str(fmap_ap_path), str(fmap_pa_path), str(se_pair), "-axis", "3"],
        [
            DWIFSLPREPROC, str(dwi_deg), str(dwi_preproc),
            "-rpe_pair",
            "-se_epi", str(se_pair),
            "-pe_dir", str(pe_dir),
            "-readout_time", str(readout_time),
            "-align_seepi",
            "-eddy_options", EDDY_OPTIONS,
            "-eddyqc_all", str(eddyqc),
            "-fslgrad", str(bvec), str(bval),
        ],
    ]
    return PreprocPlan(case_root=case_root, out_dir=out_dir, commands=commands)


def run_preproc(
    plan: PreprocPlan, runner: Callable = subprocess.run
) -> PreprocResult:
    """Execute the plan's commands in order; stop at the first failure.

    ``runner`` is injected exactly like ``track.py``'s subprocess seam so
    tests never spawn real MRtrix processes.
    """
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = plan.out_dir / "preproc.log"
    with open(log_path, "w") as log:
        for cmd in plan.commands:
            log.write("+ " + " ".join(cmd) + "\n")
            result = runner(cmd, capture_output=True, text=True)
            stdout = getattr(result, "stdout", None)
            stderr = getattr(result, "stderr", None)
            if stdout:
                log.write(stdout)
            if stderr:
                log.write(stderr)
            if result.returncode != 0:
                log.write(f"FAILED rc={result.returncode}: {cmd[0]}\n")
                return PreprocResult(ok=False, failed_cmd=cmd, log_path=log_path)
    return PreprocResult(ok=True, failed_cmd=None, log_path=log_path)
