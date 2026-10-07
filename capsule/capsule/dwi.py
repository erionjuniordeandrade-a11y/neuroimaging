from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Sequence

import nibabel as nib
import numpy as np
from neuro_core.hashing import sha256_file

from . import dwi_sentinel, masks as maskops


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLES = tuple(
    f"{name}_{side}"
    for name in ("cst", "af", "slf1", "slf2", "slf3", "ifo", "ilf", "uf", "fa", "or")
    for side in ("l", "r")
)
FAST_SHORTCUT = "2-tissue CSD instead of SS3T; 750k seeds instead of 1.5M"
TRACKING = {"algorithm": "iFOD2", "cutoff": 0.06, "min_length_mm": 20, "max_length_mm": 250}
QC_CROSSING_MM = 5.0
QC_WARNING_FRACTION = 0.01
QC_WARNING_MIN_STREAMLINES = 500

MRTRIX_TOOLS = (
    "mrconvert", "mrgrid", "dwidenoise", "mrdegibbs", "dwiextract", "mrcat",
    "dwifslpreproc", "dwi2mask", "mrmath", "dwibiascorrect", "dwi2response",
    "dwi2fod", "mtnormalise", "transformconvert", "mrtransform", "tckgen", "tckinfo",
    "mrcalc", "5ttgen", "5ttedit", "5ttcheck",
)
ACT_MRTRIX_TOOLS = {"5ttgen", "5ttedit", "5ttcheck"}
FSL_TOOLS = (
    "fslreorient2std", "flirt", "bet", "fnirt", "invwarp", "applywarp", "fslmaths",
    "fslstats", "fslcc", "convert_xfm", "topup", "applytopup", "eddy_quad",
    "run_first_all", "fast",
)
ACT_FSL_TOOLS = {"run_first_all", "fast"}
FSL_TOOL_ALTERNATIVES = (("eddy_openmp", "eddy_cuda", "eddy"),)
ANTS_TOOLS = ("antsRegistration", "N4BiasFieldCorrection")
INPUT_LAYOUT = {
    "dwi": "s7.nii.gz",
    "bvec": "s7.bvec",
    "bval": "s7.bval",
    "json": "s7_rt.json",
    "rpe": "s8.nii.gz",
    "rpe_bvec": "s8.bvec",
    "rpe_bval": "s8.bval",
    "t1": "s10.nii.gz",
    "flair": "s9.nii.gz",
}


@dataclass(frozen=True)
class Toolchain:
    mrtrix_bin: Path
    fsldir: Path
    ants_bin: Path
    ss3t: Path
    py311: Path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Toolchain":
        env = os.environ if env is None else env

        def configured(key: str, default: str) -> Path:
            return Path(env.get(key, default)).expanduser().resolve()

        return cls(
            mrtrix_bin=configured("MRTRIX_BIN", "~/mrtrix3/bin"),
            fsldir=configured("FSLDIR", "~/fsl"),
            ants_bin=configured("ANTS_BIN", "~/ants-2.6.5/bin"),
            ss3t=configured("SS3T", "~/MRtrix3Tissue/bin/ss3t_csd_beta1"),
            py311=configured("PY311", "/opt/homebrew/bin/python3.11"),
        )

    def mr(self, name: str) -> str:
        return str(self.mrtrix_bin / name)

    def fsl(self, name: str) -> str:
        return str(self.fsldir / "bin" / name)

    def ants(self, name: str) -> str:
        return str(self.ants_bin / name)

    @property
    def xtract(self) -> Path:
        return self.fsldir / "data" / "xtract_data" / "Human"

    @property
    def mni_template(self) -> Path:
        return self.fsldir / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"


@dataclass(frozen=True)
class StageSpec:
    name: str
    directory: Path
    params: dict
    outputs: tuple[Path, ...]
    key_outputs: tuple[Path, ...]
    commands: tuple[tuple[str, ...], ...] = ()
    register_plan: _RegisterPlan | None = None
    track_plan: _TrackPlan | None = None


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    marker: Path


@dataclass(frozen=True)
class _RegisterCommand:
    argv: tuple[str, ...]
    capture: str | None = None


@dataclass(frozen=True)
class _RegisterPlan:
    prepare: tuple[_RegisterCommand, ...]
    sentinel: tuple[_RegisterCommand, ...]
    header: tuple[_RegisterCommand, ...]
    regcheck: tuple[_RegisterCommand, ...]

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        return tuple(command.argv for phase in (
            self.prepare, self.sentinel, self.header, self.regcheck
        ) for command in phase)


@dataclass(frozen=True)
class _ROICommand:
    argv: tuple[str, ...]
    validate_nonempty: Path | None = None


@dataclass(frozen=True)
class _TrackBundlePlan:
    bundle: str
    tckgen: tuple[str, ...]
    tckinfo: tuple[str, ...]
    temporary: Path
    final: Path
    roi_files: _BundleRoiFiles | None = None


@dataclass(frozen=True)
class _BundleRoiFiles:
    seed: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]


@dataclass(frozen=True)
class _TrackPlan:
    roi_commands: tuple[_ROICommand, ...]
    bundles: tuple[_TrackBundlePlan, ...]
    concurrent_jobs: int

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        commands = [command.argv for command in self.roi_commands]
        for start in range(0, len(self.bundles), self.concurrent_jobs):
            batch = self.bundles[start:start + self.concurrent_jobs]
            commands.extend(bundle.tckgen for bundle in batch)
            commands.extend(bundle.tckinfo for bundle in batch)
        return tuple(commands)


class StageCommandError(RuntimeError):
    def __init__(self, stage: str, argv: Sequence[str], returncode: int, detail: str = ""):
        super().__init__(f"stage {stage}: command exited {returncode}: {list(argv)!r}"
                         + (f"\n{detail.strip()}" if detail.strip() else ""))
        self.stage = stage
        self.returncode = returncode


class ZeroStreamlineError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def required_tool_paths(
    profile: str,
    *,
    bundles: Sequence[str] = DEFAULT_BUNDLES,
    act: str = "auto",
    env: dict[str, str] | None = None,
) -> list[Path]:
    """Return every required executable and data file; directory checks are added separately."""
    if profile not in {"full", "fast"}:
        raise ValueError("profile must be 'full' or 'fast'")
    if act not in {"auto", "off"}:
        raise ValueError("act must be 'auto' or 'off'")
    tools = Toolchain.from_env(env)
    paths = [tools.mrtrix_bin / name for name in MRTRIX_TOOLS
             if act == "auto" or name not in ACT_MRTRIX_TOOLS]
    paths.extend(tools.fsldir / "bin" / name for name in FSL_TOOLS
                 if act == "auto" or name not in ACT_FSL_TOOLS)
    paths.extend(tools.ants_bin / name for name in ANTS_TOOLS)
    paths.extend((tools.mni_template, tools.fsldir / "etc" / "flirtsch" / "T1_2_MNI152_2mm.cnf"))
    paths.extend(tools.xtract / bundle / "seed.nii.gz" for bundle in bundles)
    if act == "auto":
        env = os.environ if env is None else env
        freesurfer = Path(env.get("FREESURFER_HOME", Path.home() / "freesurfer")).expanduser()
        paths.extend((freesurfer / "python" / "scripts" / "mri_synthstrip",
                      freesurfer / "models" / "synthstrip.1.pt"))
    if profile == "full":
        paths.extend((tools.ss3t, tools.py311))
    return list(dict.fromkeys(paths))


def check_tools(
    profile: str,
    *,
    bundles: Sequence[str] = DEFAULT_BUNDLES,
    act: str = "auto",
    env: dict[str, str] | None = None,
) -> list[str]:
    """Return all absent dependencies without running any external program."""
    tools = Toolchain.from_env(env)
    missing = []
    for directory in (tools.mrtrix_bin, tools.fsldir / "bin", tools.ants_bin):
        if not directory.is_dir():
            missing.append(str(directory))
    human = tools.xtract
    if not human.is_dir():
        missing.append(str(human))
    for path in required_tool_paths(profile, bundles=bundles, act=act, env=env):
        executable = path.parent in {tools.mrtrix_bin, tools.fsldir / "bin", tools.ants_bin} \
            or path == tools.py311
        if not path.is_file() or (executable and not os.access(path, os.X_OK)):
            missing.append(str(path))
    for alternatives in FSL_TOOL_ALTERNATIVES:
        candidates = [tools.fsldir / "bin" / name for name in alternatives]
        if not any(path.is_file() and os.access(path, os.X_OK) for path in candidates):
            missing.extend(str(path) for path in candidates)
    if act == "auto":
        search_path = (env or os.environ).get("PATH")
        if shutil.which("uv", path=search_path) is None:
            missing.append("uv (required for SynthStrip)")
    return list(dict.fromkeys(missing))


def validate_work_path(path: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    work = _resolved(path)
    root = _resolved(repo_root)
    try:
        work.relative_to(root)
    except ValueError:
        return work
    raise ValueError(f"work directory is inside the Git repository: {work}")


def validate_output_path(path: str | Path, *, allow_existing: bool = False) -> Path:
    output = _resolved(path)
    if output.exists() and not allow_existing:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if output.exists() and output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output}")
    return output


def _file_signature(path: str | Path) -> dict:
    resolved = _resolved(path)
    stat = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"input is not a file: {resolved}")
    return {"path": str(resolved), "bytes": stat.st_size, "sha256": _sha256(resolved)}


def _params_match(recorded, current) -> bool:
    """Compare stage params; file signatures match by content, legacy mtime ones by time."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        if {"path", "bytes", "sha256"} <= current.keys() and "mtime_ns" in recorded \
                and "sha256" not in recorded:
            try:
                stat = Path(current["path"]).stat()
            except OSError:
                return False
            return recorded.get("path") == current["path"] and recorded.get("bytes") == current["bytes"] \
                and recorded["mtime_ns"] == stat.st_mtime_ns
        return recorded.keys() == current.keys() and all(
            _params_match(recorded[key], current[key]) for key in recorded)
    if isinstance(recorded, list) and isinstance(current, list):
        return len(recorded) == len(current) and all(map(_params_match, recorded, current))
    return recorded == current


def _act_enabled(args) -> bool:
    return getattr(args, "act", "auto") == "auto"


def _act_provenance(args) -> dict:
    enabled = _act_enabled(args)
    lesion = getattr(args, "lesion", None)
    return {
        "enabled": enabled,
        "method": "5ttgen fsl" if enabled else None,
        "pathological_tissue": _resolved(lesion).name if lesion else None,
        "backtrack": enabled,
        "crop_at_gmwmi": enabled,
    }


def _input_paths(args) -> dict[str, Path]:
    names = ("dwi", "bvec", "bval", "json", "rpe", "rpe_bvec", "rpe_bval", "t1", "flair", "lesion")
    return {name: _resolved(value) for name in names
            if (value := getattr(args, name, None)) is not None}


def _validate_args(args) -> tuple[Path, Path, Toolchain]:
    required = ("dwi", "bvec", "bval", "rpe", "t1", "label", "work", "output")
    absent = [f"--{name.replace('_', '-')}" for name in required
              if getattr(args, name, None) is None or getattr(args, name, None) == ""]
    if absent:
        raise ValueError("required for a DWI run: " + ", ".join(absent))
    if (args.rpe_bvec is None) != (args.rpe_bval is None):
        raise ValueError("--rpe-bvec and --rpe-bval must be supplied together")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.seeds is not None and args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if args.readout_time <= 0:
        raise ValueError("--readout-time must be greater than 0")
    if not args.bundles or len(args.bundles) != len(set(args.bundles)):
        raise ValueError("--bundles must contain unique bundle names")
    if any(not re.fullmatch(r"[a-z0-9]+_[lr]", bundle) for bundle in args.bundles):
        raise ValueError("bundle names must use XTRACT form, such as cst_l")

    inputs = _input_paths(args)
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} input not found: {path}")
    work = validate_work_path(args.work)
    output = _resolved(args.output)
    return work, output, Toolchain.from_env()


def _relative_or_absolute(path: Path, work: Path) -> str:
    try:
        return path.resolve().relative_to(work.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _marker_path(spec: StageSpec) -> Path:
    return spec.directory / f".stage-{spec.name}.json"


def _stage_is_current(spec: StageSpec, profile: str, work_dir: Path) -> bool:
    marker = _marker_path(spec)
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    if record.get("stage") != spec.name or record.get("profile") != profile \
            or not _params_match(record.get("params"), _jsonable(spec.params)):
        return False
    if "adopted" in record and not isinstance(record["adopted"], bool):
        return False
    adopted = record.get("adopted") is True
    if adopted and spec.name not in {"preproc", "t1"}:
        return False
    expected_outputs = spec.key_outputs if adopted else spec.outputs
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_outputs):
        return False
    for item, expected_path in zip(outputs, expected_outputs):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) \
                or type(item.get("bytes")) is not int:
            return False
        try:
            path = Path(item["path"])
            actual = path if path.is_absolute() else work_dir / path
            if actual.resolve() != expected_path.resolve() or not actual.is_file():
                return False
            if actual.stat().st_size != item.get("bytes"):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def _write_stage_marker(
    spec: StageSpec,
    *,
    profile: str,
    work_dir: Path,
    tool_versions: dict,
    started_utc: str,
    adopted: bool = False,
) -> None:
    spec.directory.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": spec.name,
        "profile": profile,
        "params": _jsonable(spec.params),
        "tool_versions": _jsonable(tool_versions),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "outputs": [{"path": _relative_or_absolute(path, work_dir), "bytes": path.stat().st_size}
                    for path in (spec.key_outputs if adopted else spec.outputs)],
    }
    if adopted:
        record["adopted"] = True
    marker = _marker_path(spec)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=spec.directory,
                                     prefix=f".{marker.name}.", suffix=".tmp", delete=False) as temp:
        json.dump(record, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temp_path = Path(temp.name)
    os.replace(temp_path, marker)


def run_stage(
    spec: StageSpec,
    *,
    profile: str,
    work_dir: str | Path,
    tool_versions: dict,
    adopt_existing: bool,
    action: Callable[[], None],
) -> StageResult:
    """Run a stage unless its parameter and output marker is current; adopt only preproc/t1."""
    work = _resolved(work_dir)
    marker = _marker_path(spec)
    if _stage_is_current(spec, profile, work):
        return StageResult(spec.name, "skipped", marker)
    if adopt_existing and spec.name in {"preproc", "t1"} and not marker.exists() \
            and spec.key_outputs and all(path.is_file() for path in spec.key_outputs):
        started = _utc_now()
        _write_stage_marker(spec, profile=profile, work_dir=work, tool_versions=tool_versions,
                            started_utc=started, adopted=True)
        return StageResult(spec.name, "adopted", marker)
    started = _utc_now()
    spec.directory.mkdir(parents=True, exist_ok=True)
    action()
    missing = [str(path) for path in spec.outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"stage {spec.name}: expected outputs are missing: {missing}")
    _write_stage_marker(spec, profile=profile, work_dir=work, tool_versions=tool_versions,
                        started_utc=started)
    return StageResult(spec.name, "ran", marker)


def _stage_specs(args, work: Path, output: Path, tools: Toolchain) -> list[StageSpec]:
    profile = args.profile
    run_dir = work / profile
    nifti = run_dir / "nifti"
    tracts = run_dir / "tracts"
    roi = tracts / "roi"
    qc = run_dir / "qc"
    pp = work / "pp"
    t1 = work / "t1"
    seeds = args.seeds if args.seeds is not None else (750_000 if profile == "fast" else 1_500_000)
    signatures = {name: _file_signature(path) for name, path in _input_paths(args).items()}
    tool_paths = {"mrtrix_bin": str(tools.mrtrix_bin), "FSLDIR": str(tools.fsldir),
                  "ants_bin": str(tools.ants_bin), "SS3T": str(tools.ss3t) if profile == "full" else None,
                  "PY311": str(tools.py311) if profile == "full" else None}

    preproc_files = [
        "dwi_raw.mif", "rev_raw.mif", "dwi_2mm.mif", "rev_2mm.mif", "dwi_den.mif", "noise.mif",
        "dwi_dg.mif", "rev_dg.mif", "b0_fwd.mif", "b0_rev.mif", "seepi.mif", "dwi_pp.mif",
        "mask.mif", "b0_pp.nii.gz", "b0_raw.nii.gz",
    ]
    preproc_commands = _preproc_commands(args, work, tools)
    preproc = StageSpec(
        "preproc", pp,
        {"tool_paths": tool_paths,
         "inputs": {key: signatures[key] for key in signatures if key in {"dwi", "bvec", "bval", "json", "rpe", "rpe_bvec", "rpe_bval"}},
         "pe_dir": args.pe_dir, "readout_time": args.readout_time, "threads": args.threads,
         "regrid_vox_mm": 2.0, "denoise": "dwidenoise", "degibbs": "mrdegibbs",
         "eddy": {"slm": "linear", "repol": True, "rpe": "pair"}},
        tuple(pp / name for name in preproc_files),
        (pp / "dwi_pp.mif", pp / "mask.mif"),
        tuple(tuple(command) for command in preproc_commands),
    )

    t1_outputs = [
        t1 / "t1c.nii.gz", t1 / "t1c_1mm.nii.gz", t1 / "t1c_brain.nii.gz",
        t1 / "t1c_brain_mask.nii.gz", t1 / "t12mni.mat", t1 / "t12mni_warp.nii.gz",
        t1 / "t1_mni_nl.nii.gz", t1 / "mni2t1_warp.nii.gz",
    ]
    if args.flair:
        t1_outputs.extend((t1 / "flair.nii.gz", t1 / "flair2t1.mat", t1 / "flair_t1.nii.gz"))
    t1_spec = StageSpec(
        "t1", t1,
        {"tool_paths": tool_paths,
         "inputs": {key: signatures[key] for key in signatures if key in {"t1", "flair"}},
         "reorient": "fslreorient2std", "isotropic_mm": 1.0, "bet_f": 0.4,
         "flirt_to_mni_dof": 12, "fnirt_config": "T1_2_MNI152_2mm",
         "flair_registration": "flirt dof6 normmi" if args.flair else None},
        tuple(t1_outputs),
        tuple(path for path in (t1 / "t1c_1mm.nii.gz", t1 / "t1c_brain.nii.gz",
                                t1 / "mni2t1_warp.nii.gz", t1 / "t12mni_warp.nii.gz",
                                t1 / "flair.nii.gz" if args.flair else None,
                                t1 / "flair_t1.nii.gz" if args.flair else None) if path is not None),
        tuple(tuple(command) for command in _t1_commands(args, work, tools)),
    )

    register_outputs = [
        nifti / name for name in (
            "dwi_bc.mif", "dwi.mif", "mask.nii.gz", "b0.nii.gz", "b0_brain.nii.gz", "b02t1.mat",
            "t12b0.mat", "b0_in_t1.nii.gz", "b0_vols.mif", "brainmask_t1_ero.nii.gz",
            "t1c.nii.gz", "t1c_brain.nii.gz",
            "mni2t1_warp.nii.gz", "t12mni_warp.nii.gz", "b02t1_mrtrix.txt", "t1c_dwiworld.nii.gz",
            "b0_on_t1world.nii.gz", "shift2_x.mat", "shift2_y.mat", "shift2_z.mat",
            "b02t1_sx.mat", "b02t1_sy.mat", "b02t1_sz.mat", "b0_in_t1_sx.nii.gz",
            "b0_in_t1_sy.nii.gz", "b0_in_t1_sz.nii.gz", "b0_in_t1_ants.nii.gz",
            "ants_b02t1_0GenericAffine.mat",
        )
    ]
    register_outputs.extend((qc / "orientation_sentinel.json", qc / "regcheck.json",
                             qc / "header_direction.json"))
    if args.flair:
        register_outputs.extend((nifti / "flair.nii.gz", nifti / "flair_t1.nii.gz",
                                 nifti / "flair_dwiworld.nii.gz"))
    register_outputs.extend((nifti / "_sentinel" / f"sent_{tag}_{space}.nii.gz"
                             for tag in ("true", "mirrored_control")
                             for space in ("mni", "dwi")))
    register_plan = _register_plan(args, work, tools)
    register = StageSpec(
        "register", nifti,
        {"tool_paths": tool_paths,
         "inputs": {key: signatures[key] for key in signatures if key in {"t1", "flair"}},
         "preproc_marker_params": preproc.params, "t1_marker_params": t1_spec.params,
         "bias_correction": "ants", "regrid_vox_mm": 1.25,
         "b0_to_t1": {"method": "FLIRT", "dof": 6, "cost": "normmi"},
         "flair_registration": "FLIRT dof6 normmi, then header-only rigid to DWI" if args.flair else None,
         "sentinel": "asymmetric six-marker with mirrored positive control",
         "regcheck": "FLIRT vs ANTs rigid MI, three 2 mm shift controls"},
        tuple(register_outputs),
        (nifti / "b0.nii.gz", nifti / "b02t1.mat", nifti / "t12b0.mat",
         qc / "orientation_sentinel.json", qc / "regcheck.json"),
        register_plan.commands,
        register_plan,
    )

    act_spec = None
    act_record = _act_provenance(args)
    if act_record["enabled"]:
        t1_dwiworld = nifti / "t1c_dwiworld.nii.gz"
        synthstrip_mask = nifti / "t1c_dwiworld_synthstrip_mask.nii.gz"
        brainmasked_t1 = nifti / "t1c_dwiworld_brainmasked.nii.gz"
        raw_5tt = nifti / "5tt_raw.mif"
        final_5tt = nifti / "5tt.mif"
        synthstrip = maskops._synthstrip_command(t1_dwiworld, synthstrip_mask)
        if synthstrip is None:
            raise FileNotFoundError("ACT requires SynthStrip, its FreeSurfer model, and uv")
        freesurfer = maskops._freesurfer_home()
        synthstrip_signature = {
            "command": tuple(synthstrip),
            "uv": _file_signature(synthstrip[0]),
            "script": _file_signature(freesurfer / "python" / "scripts" / "mri_synthstrip"),
            "model": _file_signature(freesurfer / "models" / "synthstrip.1.pt"),
        }
        act_commands = [
            tuple(synthstrip),
            (tools.fsl("fslmaths"), str(t1_dwiworld), "-mas", str(synthstrip_mask),
             str(brainmasked_t1)),
            (tools.mr("5ttgen"), "fsl", str(brainmasked_t1), str(raw_5tt), "-premasked", "-nocrop"),
        ]
        act_outputs = [synthstrip_mask, brainmasked_t1, raw_5tt]
        act_params = {
            "tool_paths": {**tool_paths, "FREESURFER_HOME": str(freesurfer)},
            "register_params": register.params,
            "inputs": {"lesion": signatures.get("lesion")},
            "enabled": True,
            "method": "5ttgen fsl",
            "synthstrip": synthstrip_signature,
            "t1_source": str(t1_dwiworld),
            "brain_mask": str(synthstrip_mask),
            "premasked": True,
            "nocrop": True,
        }
        if args.lesion:
            lesion_dwi = roi / "lesion1_dwi.nii.gz"
            lesion_on_5tt = nifti / "lesion_on5tt.mif"
            act_commands.extend((
                (tools.fsl("flirt"), "-in", str(_resolved(args.lesion)), "-ref",
                 str(nifti / "b0.nii.gz"), "-applyxfm", "-init", str(nifti / "t12b0.mat"),
                 "-interp", "nearestneighbour", "-out", str(lesion_dwi)),
                (tools.mr("mrtransform"), str(lesion_dwi), "-template", str(raw_5tt),
                 "-interp", "nearest", str(lesion_on_5tt), "-force"),
                (tools.mr("5ttedit"), str(raw_5tt), "-path", str(lesion_on_5tt),
                 str(final_5tt), "-force"),
            ))
            act_outputs.extend((lesion_dwi, lesion_on_5tt, final_5tt))
            act_params["lesion_transform"] = "FLIRT nearestneighbour to b0, then MRtrix nearest to 5TT"
        else:
            act_commands.append((tools.mr("mrconvert"), str(raw_5tt), str(final_5tt), "-force"))
            act_outputs.append(final_5tt)
        act_commands.append((tools.mr("5ttcheck"), str(final_5tt)))
        act_spec = StageSpec(
            "act", nifti, act_params, tuple(act_outputs), (final_5tt,),
            tuple(act_commands),
        )

    fod_names = ["wm.txt", "gm.txt", "csf.txt", "voxels.mif", "wmfod.mif", "csf.mif", "wmfod_norm.mif", "csf_norm.mif"]
    if profile == "full":
        fod_names.extend(("gm.mif", "gm_norm.mif"))
    fod = StageSpec(
        "fod", nifti,
        {"tool_paths": tool_paths, "register_params": register.params,
         "profile": profile, "algorithm": "SS3T" if profile == "full" else "msmt_csd",
         "tissues": ["WM", "GM", "CSF"] if profile == "full" else ["WM", "CSF"],
         "response": "dhollander", "normalization": "mtnormalise"},
        tuple(nifti / name for name in fod_names),
        (nifti / "wmfod_norm.mif",),
        tuple(tuple(command) for command in _fod_commands(args, work, tools)),
    )

    track_plan, roi_outputs, tck_outputs = _tracking_plan(args, work, tools)
    track_outputs = tuple([*roi_outputs, *tck_outputs])
    bundle_rois = _bundle_roi_provenance(track_plan)
    xtract_signatures = {
        bundle: {source_name: _file_signature(tools.xtract / bundle / source_name)
                 for source_name, _ in _roi_files(tools.xtract / bundle, bundle)}
        for bundle in args.bundles
    }
    track = StageSpec(
        "track", tracts,
        {"tool_paths": tool_paths, "register_params": register.params, "fod_params": fod.params,
         "xtract_inputs": xtract_signatures,
         "bundles": list(args.bundles), "seeds_per_bundle": seeds,
         "threads_total": args.threads, "threads_per_job": _tracking_resources(args.threads)[0],
         "concurrent_jobs": _tracking_resources(args.threads)[1], "tracking": TRACKING,
         "roi_threshold": 0.1, "roi_interpolation": "trilinear",
         "act": act_spec.params if act_spec else {"enabled": False},
         "bundle_rois": bundle_rois},
        track_outputs, tuple(tck_outputs),
        track_plan.commands, track_plan=track_plan,
    )

    lesion_output = roi / "lesion1_dwi.nii.gz"
    qc_outputs = [qc / "bundles.json"]
    qc_commands = []
    if args.lesion and not _act_enabled(args):
        qc_outputs.append(lesion_output)
        qc_commands.append((tools.fsl("flirt"), "-in", str(_resolved(args.lesion)), "-ref",
                            str(nifti / "b0.nii.gz"), "-applyxfm", "-init", str(nifti / "t12b0.mat"),
                            "-interp", "nearestneighbour", "-out", str(lesion_output)))
    qc_spec = StageSpec(
        "qc", qc,
        {"tool_paths": tool_paths, "track_params": track.params,
         "bundles": list(args.bundles), "seeds_per_bundle": seeds,
         "midline_crossing_mm": QC_CROSSING_MM, "warning_fraction": QC_WARNING_FRACTION,
         "warning_min_streamlines": QC_WARNING_MIN_STREAMLINES,
         "lesion": signatures.get("lesion") if not _act_enabled(args) else None},
        tuple(qc_outputs), (qc / "bundles.json",), tuple(tuple(command) for command in qc_commands),
    )

    label = f"{args.label} (tracts unreviewed)"
    provenance = tracts / "capsule-dwi-provenance.json"
    run_record = run_dir / "dwi_run.json"
    capsule_argv = _capsule_argv(args, output, nifti, tracts, label)
    provenance_record = {
        "pipeline": "capsule dwi", "profile": profile, "seeds": seeds,
        "algorithm": "iFOD2", "cutoff": TRACKING["cutoff"],
        "act": act_record, "bundle_rois": bundle_rois,
    }
    capsule = StageSpec(
        "capsule", run_dir,
        {"tool_paths": tool_paths, "qc_params": qc_spec.params, "track_params": track.params,
         "label": label, "output": str(output), "profile": profile,
         "seeds_per_bundle": seeds, "capsule_builder": "capsule.cli build-nifti",
         "spacing_mm": 1.0, "brain_mask": "synthstrip", "tract_labels": list(args.bundles),
         "provenance": provenance_record},
        (output, provenance, run_record), (output, run_record),
        (tuple(["capsule", *capsule_argv]),),
    )
    stages = [preproc, t1_spec, register]
    if act_spec:
        stages.append(act_spec)
    stages.extend((fod, track, qc_spec, capsule))
    return stages


def _preproc_commands(args, work: Path, tools: Toolchain) -> list[list[str]]:
    nii, pp = work / "nii", work / "pp"
    dwi_source = nii / INPUT_LAYOUT["dwi"]
    rpe_source = nii / INPUT_LAYOUT["rpe"]
    dwi_raw, rev_raw = pp / "dwi_raw.mif", pp / "rev_raw.mif"
    commands = []
    forward = [tools.mr("mrconvert"), str(dwi_source), "-fslgrad", str(nii / INPUT_LAYOUT["bvec"]),
               str(nii / INPUT_LAYOUT["bval"])]
    if args.json:
        forward.extend(("-json_import", str(nii / INPUT_LAYOUT["json"])))
    forward.extend((str(dwi_raw), "-force"))
    commands.append(forward)
    reverse = [tools.mr("mrconvert"), str(rpe_source)]
    if args.rpe_bvec:
        reverse.extend(("-fslgrad", str(nii / INPUT_LAYOUT["rpe_bvec"]), str(nii / INPUT_LAYOUT["rpe_bval"])))
    reverse.extend((str(rev_raw), "-force"))
    commands.append(reverse)
    commands.extend([
        [tools.mr("mrgrid"), str(dwi_raw), "regrid", "-vox", "2", "-interp", "sinc", str(pp / "dwi_2mm.mif"), "-force"],
        [tools.mr("mrgrid"), str(rev_raw), "regrid", "-vox", "2", "-interp", "sinc", str(pp / "rev_2mm.mif"), "-force"],
        [tools.mr("dwidenoise"), str(pp / "dwi_2mm.mif"), str(pp / "dwi_den.mif"), "-noise", str(pp / "noise.mif"),
         "-nthreads", str(args.threads), "-force"],
        [tools.mr("mrdegibbs"), str(pp / "dwi_den.mif"), str(pp / "dwi_dg.mif"), "-nthreads", str(args.threads), "-force"],
        [tools.mr("dwiextract"), str(pp / "dwi_dg.mif"), "-bzero", str(pp / "b0_fwd_all.mif"), "-force"],
        [tools.mr("mrconvert"), str(pp / "b0_fwd_all.mif"), "-coord", "3", "0:2", "-set_property",
         "TotalReadoutTime", str(args.readout_time), str(pp / "b0_fwd.mif"), "-force"],
        [tools.mr("mrdegibbs"), str(pp / "rev_2mm.mif"), str(pp / "rev_dg.mif"), "-force"],
        [tools.mr("dwiextract"), str(pp / "rev_dg.mif"), "-bzero", str(pp / "b0_rev_all.mif"), "-force"],
        [tools.mr("mrconvert"), str(pp / "b0_rev_all.mif"), "-coord", "3", "0:2", str(pp / "b0_rev.mif"), "-force"],
        [tools.mr("mrcat"), str(pp / "b0_fwd.mif"), str(pp / "b0_rev.mif"), "-axis", "3", str(pp / "seepi.mif"), "-force"],
        [tools.mr("dwifslpreproc"), str(pp / "dwi_dg.mif"), str(pp / "dwi_pp.mif"), "-rpe_pair",
         "-se_epi", str(pp / "seepi.mif"), "-pe_dir", args.pe_dir, "-readout_time", str(args.readout_time),
         "-align_seepi", "-eddy_options", " --slm=linear --repol", "-eddyqc_all", str(pp / "eddyqc"),
         "-scratch", str(pp / "scratch"), "-nocleanup", "-nthreads", str(args.threads), "-force"],
        [tools.mr("dwi2mask"), str(pp / "dwi_pp.mif"), str(pp / "mask.mif"), "-force"],
        [tools.mr("dwiextract"), str(pp / "dwi_pp.mif"), "-bzero", str(pp / "b0_pp_vols.mif"), "-force"],
        [tools.mr("mrmath"), str(pp / "b0_pp_vols.mif"), "mean", str(pp / "b0_pp.nii.gz"), "-axis", "3", "-force"],
        [tools.mr("dwiextract"), str(pp / "dwi_dg.mif"), "-bzero", str(pp / "b0_raw_vols.mif"), "-force"],
        [tools.mr("mrmath"), str(pp / "b0_raw_vols.mif"), "mean", str(pp / "b0_raw.nii.gz"), "-axis", "3", "-force"],
    ])
    return commands


def _t1_commands(args, work: Path, tools: Toolchain) -> list[list[str]]:
    nii, t1 = work / "nii", work / "t1"
    commands = [
        [tools.fsl("fslreorient2std"), str(nii / INPUT_LAYOUT["t1"]), str(t1 / "t1c.nii.gz")],
        [tools.fsl("flirt"), "-in", str(t1 / "t1c.nii.gz"), "-ref", str(t1 / "t1c.nii.gz"),
         "-applyisoxfm", "1.0", "-out", str(t1 / "t1c_1mm.nii.gz"), "-interp", "sinc"],
        [tools.fsl("bet"), str(t1 / "t1c_1mm.nii.gz"), str(t1 / "t1c_brain.nii.gz"), "-f", "0.4", "-B", "-m"],
        [tools.fsl("flirt"), "-in", str(t1 / "t1c_brain.nii.gz"), "-ref", str(tools.mni_template),
         "-omat", str(t1 / "t12mni.mat"), "-dof", "12"],
        [tools.fsl("fnirt"), f"--in={t1 / 't1c_1mm.nii.gz'}", f"--aff={t1 / 't12mni.mat'}",
         f"--cout={t1 / 't12mni_warp.nii.gz'}", "--config=T1_2_MNI152_2mm",
         f"--iout={t1 / 't1_mni_nl.nii.gz'}"],
        [tools.fsl("invwarp"), "-w", str(t1 / "t12mni_warp.nii.gz"), "-o", str(t1 / "mni2t1_warp.nii.gz"),
         "-r", str(t1 / "t1c_1mm.nii.gz")],
    ]
    if args.flair:
        commands.extend([
            [tools.fsl("fslreorient2std"), str(nii / INPUT_LAYOUT["flair"]), str(t1 / "flair.nii.gz")],
            [tools.fsl("flirt"), "-in", str(t1 / "flair.nii.gz"), "-ref", str(t1 / "t1c_1mm.nii.gz"),
             "-omat", str(t1 / "flair2t1.mat"), "-dof", "6", "-cost", "normmi", "-out",
             str(t1 / "flair_t1.nii.gz")],
        ])
    return commands


def _register_plan(args, work: Path, tools: Toolchain) -> _RegisterPlan:
    nifti = work / args.profile / "nifti"
    pp = work / "pp"
    prepare = [
        [tools.mr("dwibiascorrect"), "ants", str(pp / "dwi_pp.mif"), str(nifti / "dwi_bc.mif"),
         "-mask", str(pp / "mask.mif"), "-force"],
        [tools.mr("mrgrid"), str(nifti / "dwi_bc.mif"), "regrid", "-vox", "1.25", str(nifti / "dwi.mif"), "-force"],
        [tools.mr("dwi2mask"), str(nifti / "dwi.mif"), str(nifti / "mask.nii.gz"), "-force"],
        [tools.mr("mrconvert"), str(nifti / "mask.nii.gz"), "-datatype", "uint8",
         str(nifti / "mask.nii.gz"), "-force"],
        [tools.mr("dwiextract"), str(nifti / "dwi.mif"), "-bzero", str(nifti / "b0_vols.mif"), "-force"],
        [tools.mr("mrmath"), str(nifti / "b0_vols.mif"), "mean", str(nifti / "b0.nii.gz"), "-axis", "3", "-force"],
        [tools.fsl("fslmaths"), str(nifti / "b0.nii.gz"), "-mas", str(nifti / "mask.nii.gz"),
         str(nifti / "b0_brain.nii.gz")],
        [tools.fsl("flirt"), "-in", str(nifti / "b0_brain.nii.gz"), "-ref", str(nifti / "t1c_brain.nii.gz"),
         "-omat", str(nifti / "b02t1.mat"), "-dof", "6", "-cost", "normmi", "-out", str(nifti / "b0_in_t1.nii.gz")],
        [tools.fsl("convert_xfm"), "-omat", str(nifti / "t12b0.mat"), "-inverse", str(nifti / "b02t1.mat")],
    ]
    sentinel = tuple(_RegisterCommand(tuple(command)) for command in
                     dwi_sentinel.command_plan(nifti, fsldir=tools.fsldir))
    header = [
        [tools.mr("transformconvert"), str(nifti / "b02t1.mat"), str(nifti / "b0_brain.nii.gz"),
         str(nifti / "t1c_brain.nii.gz"), "flirt_import", str(nifti / "b02t1_mrtrix.txt"), "-force"],
        [tools.mr("mrtransform"), str(nifti / "t1c.nii.gz"), "-linear", str(nifti / "b02t1_mrtrix.txt"),
         "-inverse", str(nifti / "t1c_dwiworld.nii.gz"), "-force"],
    ]
    if args.flair:
        header.append([tools.mr("mrtransform"), str(nifti / "flair_t1.nii.gz"), "-linear",
                       str(nifti / "b02t1_mrtrix.txt"), "-inverse",
                       str(nifti / "flair_dwiworld.nii.gz"), "-force"])
    header.extend([
        [tools.mr("mrtransform"), str(nifti / "b0_brain.nii.gz"), "-template", str(nifti / "t1c_dwiworld.nii.gz"),
         "-interp", "linear", str(nifti / "b0_on_t1world.nii.gz"), "-force"],
        _RegisterCommand((tools.fsl("fslcc"), str(nifti / "b0_on_t1world.nii.gz"),
                          str(nifti / "b0_in_t1.nii.gz")), "header_direction"),
    ])
    regcheck = [
        [tools.ants("antsRegistration"), "-d", "3", "-o",
         f"[{nifti / 'ants_b02t1_'},{nifti / 'b0_in_t1_ants.nii.gz'}]",
         "-r", f"[{nifti / 't1c_brain.nii.gz'},{nifti / 'b0_brain.nii.gz'},1]",
         "-m", f"MI[{nifti / 't1c_brain.nii.gz'},{nifti / 'b0_brain.nii.gz'},1,32,Regular,0.25]",
         "-t", "Rigid[0.1]", "-c", "[1000x500x250,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox", "-n", "Linear"],
        [tools.fsl("fslmaths"), str(nifti / "t1c_brain.nii.gz"), "-bin", "-ero", "-ero", str(nifti / "brainmask_t1_ero.nii.gz")],
    ]
    for axis, output in zip("xyz", ("sx", "sy", "sz")):
        regcheck.extend([
            [tools.fsl("convert_xfm"), "-omat", str(nifti / f"b02t1_{output}.mat"), "-concat",
             str(nifti / f"shift2_{axis}.mat"), str(nifti / "b02t1.mat")],
            [tools.fsl("flirt"), "-in", str(nifti / "b0_brain.nii.gz"), "-ref", str(nifti / "t1c_brain.nii.gz"),
             "-applyxfm", "-init", str(nifti / f"b02t1_{output}.mat"), "-out",
             str(nifti / f"b0_in_t1_{output}.nii.gz")],
            _RegisterCommand((tools.fsl("fslcc"), "-m", str(nifti / "brainmask_t1_ero.nii.gz"),
                              str(nifti / "b0_in_t1.nii.gz"),
                              str(nifti / f"b0_in_t1_{output}.nii.gz")), f"control_{axis}"),
        ])
    regcheck.append(_RegisterCommand(
        (tools.fsl("fslcc"), "-m", str(nifti / "brainmask_t1_ero.nii.gz"),
         str(nifti / "b0_in_t1.nii.gz"), str(nifti / "b0_in_t1_ants.nii.gz")),
        "flirt_vs_ants"))
    return _RegisterPlan(
        prepare=tuple(_RegisterCommand(tuple(command)) for command in prepare),
        sentinel=sentinel,
        header=tuple(command if isinstance(command, _RegisterCommand)
                     else _RegisterCommand(tuple(command)) for command in header),
        regcheck=tuple(command if isinstance(command, _RegisterCommand)
                       else _RegisterCommand(tuple(command)) for command in regcheck),
    )


def _fod_commands(args, work: Path, tools: Toolchain) -> list[list[str]]:
    nifti = work / args.profile / "nifti"
    mask = nifti / "mask.nii.gz"
    commands = [[tools.mr("dwi2response"), "dhollander", str(nifti / "dwi.mif"),
                 str(nifti / "wm.txt"), str(nifti / "gm.txt"), str(nifti / "csf.txt"),
                 "-mask", str(mask), "-voxels", str(nifti / "voxels.mif"), "-force"]]
    if args.profile == "full":
        commands.append([str(tools.py311), str(tools.ss3t), str(nifti / "dwi.mif"),
                         str(nifti / "wm.txt"), str(nifti / "wmfod.mif"),
                         str(nifti / "gm.txt"), str(nifti / "gm.mif"),
                         str(nifti / "csf.txt"), str(nifti / "csf.mif"),
                         "-mask", str(mask), "-nthreads", str(args.threads), "-force"])
        commands.append([tools.mr("mtnormalise"), str(nifti / "wmfod.mif"), str(nifti / "wmfod_norm.mif"),
                         str(nifti / "gm.mif"), str(nifti / "gm_norm.mif"),
                         str(nifti / "csf.mif"), str(nifti / "csf_norm.mif"),
                         "-mask", str(mask), "-force"])
    else:
        commands.append([tools.mr("dwi2fod"), "msmt_csd", str(nifti / "dwi.mif"),
                         str(nifti / "wm.txt"), str(nifti / "wmfod.mif"),
                         str(nifti / "csf.txt"), str(nifti / "csf.mif"),
                         "-mask", str(mask), "-nthreads", str(args.threads), "-force"])
        commands.append([tools.mr("mtnormalise"), str(nifti / "wmfod.mif"), str(nifti / "wmfod_norm.mif"),
                         str(nifti / "csf.mif"), str(nifti / "csf_norm.mif"),
                         "-mask", str(mask), "-force"])
    return commands


def _tracking_plan(args, work: Path, tools: Toolchain) -> tuple[_TrackPlan, list[Path], list[Path]]:
    nifti = work / args.profile / "nifti"
    tracts = work / args.profile / "tracts"
    roi = tracts / "roi"
    roi_commands: list[_ROICommand] = []
    roi_outputs: list[Path] = []
    tck_outputs: list[Path] = []
    bundles: list[_TrackBundlePlan] = []
    for bundle in args.bundles:
        source_dir = tools.xtract / bundle
        roi_items = [("seed.nii.gz", f"{bundle}_seed.nii.gz", "seed_image")]
        for filename in ("target.nii.gz", "target1.nii.gz", "target2.nii.gz"):
            if (source_dir / filename).is_file():
                roi_items.append((filename, f"{bundle}_{filename}", "include"))
        if (source_dir / "exclude.nii.gz").is_file():
            roi_items.append(("exclude.nii.gz", f"{bundle}_exclude.nii.gz", "exclude"))
        args_by_kind = []
        roi_names = {"seed": None, "include": [], "exclude": []}
        for source_name, output_name, kind in roi_items:
            transformed = roi / output_name.replace(".nii.gz", "_warp.nii.gz")
            thresholded = roi / output_name
            roi_commands.extend((
                _ROICommand((tools.fsl("applywarp"), "-i", str(source_dir / source_name),
                             "-r", str(nifti / "b0.nii.gz"), "-w", str(nifti / "mni2t1_warp.nii.gz"),
                             f"--postmat={nifti / 't12b0.mat'}", "-o", str(transformed),
                             "--interp=trilinear")),
                _ROICommand((tools.fsl("fslmaths"), str(transformed), "-thr", "0.1", "-bin",
                             str(thresholded))),
                _ROICommand((tools.fsl("fslstats"), str(thresholded), "-V"), thresholded),
            ))
            roi_outputs.extend((transformed, thresholded))
            if kind == "seed_image":
                args_by_kind.extend(("-seed_image", str(thresholded)))
                roi_names["seed"] = f"roi/{output_name}"
            elif kind == "include":
                args_by_kind.extend(("-include", str(thresholded)))
                roi_names["include"].append(f"roi/{output_name}")
            else:
                args_by_kind.extend(("-exclude", str(thresholded)))
                roi_names["exclude"].append(f"roi/{output_name}")
        out = tracts / f"fx_{bundle}.tck"
        tmp = tracts / f"fx_{bundle}.tmp.tck"
        act_args = (("-act", str(nifti / "5tt.mif"), "-backtrack", "-crop_at_gmwmi")
                    if _act_enabled(args) else ())
        tckgen = (tools.mr("tckgen"), str(nifti / "wmfod_norm.mif"), str(tmp),
                  "-algorithm", TRACKING["algorithm"], *act_args, *args_by_kind,
                  "-mask", str(nifti / "mask.nii.gz"),
                  "-seeds", str(args.seeds if args.seeds is not None else
                                 (750_000 if args.profile == "fast" else 1_500_000)),
                  "-minlength", str(TRACKING["min_length_mm"]), "-maxlength",
                  str(TRACKING["max_length_mm"]), "-cutoff", str(TRACKING["cutoff"]),
                  "-nthreads", str(_tracking_resources(args.threads)[0]), "-force", "-quiet")
        bundle_rois = _BundleRoiFiles(roi_names["seed"], tuple(roi_names["include"]),
                                      tuple(roi_names["exclude"]))
        bundles.append(_TrackBundlePlan(bundle, tckgen, (tools.mr("tckinfo"), str(tmp)), tmp, out,
                                        bundle_rois))
        tck_outputs.append(out)
    _, concurrent_jobs = _tracking_resources(args.threads)
    return _TrackPlan(tuple(roi_commands), tuple(bundles), concurrent_jobs), roi_outputs, tck_outputs


def _bundle_roi_provenance(plan: _TrackPlan) -> dict[str, dict[str, str | list[str] | None]]:
    return {
        job.bundle: {
            "seed": job.roi_files.seed if job.roi_files else None,
            "include": list(job.roi_files.include) if job.roi_files else [],
            "exclude": list(job.roi_files.exclude) if job.roi_files else [],
        }
        for job in plan.bundles
    }


def _tracking_resources(threads: int) -> tuple[int, int]:
    threads_per_job = min(4, threads)
    concurrent_jobs = max(1, min(4, threads // threads_per_job))
    return threads_per_job, concurrent_jobs


def _capsule_argv(args, output: Path, nifti: Path, tracts: Path, label: str) -> list[str]:
    argv = ["build-nifti", "--volume", f"{nifti / 't1c_dwiworld.nii.gz'}:MR:T1+C"]
    if args.flair:
        argv.extend(("--volume", f"{nifti / 'flair_dwiworld.nii.gz'}:MR:FLAIR"))
        argv.extend(("--volume-registration",
                     "FLAIR:FLIRT dof6 normmi, then header-only rigid to DWI"))
    argv.extend(("--label", label, "-o", str(output), "--spacing", "1",
                 "--brain-mask", "synthstrip", "--tract-max-length-mm",
                 str(TRACKING["max_length_mm"])))
    if _act_enabled(args):
        # The ACT stage already ran SynthStrip on this T1; a second run peaks near 29 GB and can be OOM-killed.
        argv.extend(("--brain-mask-file", f"{nifti / 't1c_dwiworld_synthstrip_mask.nii.gz'}:synthstrip"))
    if not getattr(args, "no_deface", False):
        argv.append("--deface")
    if getattr(args, "lesion", None):
        argv.extend(("--lesion-mask",
                     f"{tracts / 'roi' / 'lesion1_dwi.nii.gz'}:Lesão (ROI de rastreamento)"))
    for bundle in args.bundles:
        argv.extend(("--tract", f"{tracts / f'fx_{bundle}.tck'}:{bundle}"))
    return argv


def _tool_environment(tools: Toolchain, threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["FSLDIR"] = str(tools.fsldir)
    env.setdefault("FREESURFER_HOME", str(Path.home() / "freesurfer"))
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    prefix = [tools.mrtrix_bin, tools.fsldir / "bin", tools.ants_bin]
    if str(tools.ss3t) not in {"", "."}:
        prefix.append(tools.ss3t.parent)
    prefix.append(tools.py311.parent)
    path = [str(path) for path in prefix]
    if env.get("PATH"):
        path.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path)
    return env


class CommandRunner:
    def __init__(self, log_path: Path, tools: Toolchain, work_dir: Path, threads: int):
        self.log_path = log_path
        self.tools = tools
        self.work_dir = work_dir
        self.env = _tool_environment(tools, threads)
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(text.rstrip() + "\n")
                log.flush()

    def run(self, argv: Sequence[str], *, stage: str, cwd: Path | None = None,
            allow_failure: bool = False) -> subprocess.CompletedProcess:
        argv = [str(item) for item in argv]
        self.log(f"[{_utc_now()}] stage={stage} argv={json.dumps(argv, ensure_ascii=False)}")
        result = subprocess.run(argv, cwd=cwd or self.work_dir, env=self.env,
                                capture_output=True, text=True, check=False)
        self.log(f"[{_utc_now()}] stage={stage} exit_code={result.returncode}")
        if result.stdout:
            self.log(f"stdout: {result.stdout[-4000:]}")
        if result.stderr:
            self.log(f"stderr: {result.stderr[-4000:]}")
        if result.returncode and not allow_failure:
            raise StageCommandError(stage, argv, result.returncode, result.stderr or result.stdout)
        return result


def _tool_versions(runner: CommandRunner, tools: Toolchain, profile: str) -> dict:
    probes = [
        ("MRtrix3", [tools.mr("mrconvert"), "-version"]),
        ("FSL", [tools.fsl("flirt"), "-version"]),
        ("ANTs", [tools.ants("antsRegistration"), "--version"]),
    ]
    if profile == "full":
        probes.append(("Python", [str(tools.py311), "--version"]))
    versions = {}
    for name, argv in probes:
        result = runner.run(argv, stage="tool-version", allow_failure=True)
        text = (result.stdout or result.stderr).strip()
        versions[name] = {"version": text[:500] or "unreported", "exit_code": result.returncode}
    if profile == "full":
        stat = tools.ss3t.stat()
        versions["MRtrix3Tissue"] = {"version": "CLI does not report a version; executable file identity",
                                     "path": str(tools.ss3t), "bytes": stat.st_size,
                                     "mtime_ns": stat.st_mtime_ns}
    return versions


def _materialize_inputs(args, work: Path, stage_name: str) -> None:
    selected = {"preproc": {"dwi", "bvec", "bval", "json", "rpe", "rpe_bvec", "rpe_bval"},
                "t1": {"t1", "flair"}}[stage_name]
    nii = work / "nii"
    nii.mkdir(parents=True, exist_ok=True)
    for name, source in _input_paths(args).items():
        if name not in selected:
            continue
        destination = nii / INPUT_LAYOUT[name]
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"stage {stage_name}: legacy input path is not a file: {destination}")
            if destination.resolve() == source.resolve():
                continue
            if destination.stat().st_size != source.stat().st_size \
                    or _sha256(destination) != _sha256(source):
                raise ValueError(f"stage {stage_name}: existing input file differs from supplied {name}: {destination}")
            continue
        shutil.copy2(source, destination)


_sha256 = sha256_file


def _run_fixed_commands(spec: StageSpec, runner: CommandRunner, cwd: Path) -> None:
    for argv in spec.commands:
        runner.run(argv, stage=spec.name, cwd=cwd)


def _run_preproc(spec: StageSpec, args, work: Path, runner: CommandRunner) -> None:
    _materialize_inputs(args, work, "preproc")
    _run_fixed_commands(spec, runner, work)


def _run_t1(spec: StageSpec, args, work: Path, runner: CommandRunner) -> None:
    _materialize_inputs(args, work, "t1")
    _run_fixed_commands(spec, runner, work)


def _write_shift_matrices(nifti: Path) -> None:
    matrices = {
        "x": "1 0 0 2\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        "y": "1 0 0 0\n0 1 0 2\n0 0 1 0\n0 0 0 1\n",
        "z": "1 0 0 0\n0 1 0 0\n0 0 1 2\n0 0 0 1\n",
    }
    for axis, contents in matrices.items():
        (nifti / f"shift2_{axis}.mat").write_text(contents, encoding="ascii")


def _copy_t1_artifacts(args, work: Path) -> None:
    source_dir = work / "t1"
    destination_dir = work / args.profile / "nifti"
    pairs = [("t1c_1mm.nii.gz", "t1c.nii.gz"), ("t1c_brain.nii.gz", "t1c_brain.nii.gz"),
             ("mni2t1_warp.nii.gz", "mni2t1_warp.nii.gz"), ("t12mni_warp.nii.gz", "t12mni_warp.nii.gz")]
    if args.flair:
        pairs.extend((("flair.nii.gz", "flair.nii.gz"), ("flair_t1.nii.gz", "flair_t1.nii.gz")))
    for source_name, destination_name in pairs:
        source = source_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(f"register: required T1-stage output is missing: {source}")
        shutil.copy2(source, destination_dir / destination_name)


def _correlation(result: subprocess.CompletedProcess, description: str) -> float:
    values = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", result.stdout)
    if not values:
        raise RuntimeError(f"register: could not parse fslcc result for {description}: {result.stdout!r}")
    value = float(values[-1])
    if not np.isfinite(value):
        raise RuntimeError(f"register: non-finite fslcc result for {description}")
    return value


def _run_register(spec: StageSpec, args, work: Path, qc: Path, tools: Toolchain,
                  runner: CommandRunner) -> dict:
    plan = spec.register_plan
    if plan is None:
        raise RuntimeError("register: command plan is missing")
    _materialize_inputs(args, work, "t1")
    nifti = work / args.profile / "nifti"
    qc.mkdir(parents=True, exist_ok=True)
    _copy_t1_artifacts(args, work)
    _write_shift_matrices(nifti)
    _run_register_phase(plan.prepare, runner, nifti)
    planted = dwi_sentinel.prepare(nifti, fsldir=tools.fsldir)
    _run_register_phase(plan.sentinel, runner, nifti)
    sentinel = dwi_sentinel.evaluate(nifti, qc / "orientation_sentinel.json", planted)
    if sentinel["verdict"] != "PASS":
        raise RuntimeError("register: orientation sentinel FAIL; tracking was stopped")
    header = _run_register_phase(plan.header, runner, nifti)
    direction_corr = header["header_direction"]
    measurements = _run_register_phase(plan.regcheck, runner, nifti)
    controls = {axis: measurements[f"control_{axis}"] for axis in "xyz"}
    flirt_vs_ants = measurements["flirt_vs_ants"]
    regcheck = {
        "flirt_vs_ants": flirt_vs_ants,
        "controls": controls,
        "verdict": "PASS" if flirt_vs_ants > max(controls.values()) else "FAIL",
    }
    (qc / "regcheck.json").write_text(json.dumps(regcheck, indent=2) + "\n", encoding="utf-8")
    direction_record = {"correlation": direction_corr, "verdict": "NOT_GATED",
                        "reason": "the source workflow defines no numeric threshold"}
    (qc / "header_direction.json").write_text(json.dumps(direction_record, indent=2) + "\n",
                                               encoding="utf-8")
    if regcheck["verdict"] != "PASS":
        raise RuntimeError("register: regcheck FAIL; tracking was stopped")
    return {"orientation_sentinel": sentinel["verdict"], "regcheck": regcheck["verdict"],
            "header_direction_corr": direction_corr}


def _run_register_phase(commands: tuple[_RegisterCommand, ...], runner: CommandRunner,
                        nifti: Path) -> dict[str, float]:
    captured = {}
    for command in commands:
        result = runner.run(command.argv, stage="register", cwd=nifti)
        if command.capture is not None:
            captured[command.capture] = _correlation(result, command.capture)
    return captured


def _run_fod(spec: StageSpec, runner: CommandRunner, nifti: Path) -> None:
    _run_fixed_commands(spec, runner, nifti)


def _run_act(spec: StageSpec, args, work: Path, runner: CommandRunner) -> None:
    (work / args.profile / "tracts" / "roi").mkdir(parents=True, exist_ok=True)
    _run_fixed_commands(spec, runner, work / args.profile / "nifti")


def _roi_volume_count(result: subprocess.CompletedProcess, path: Path) -> int:
    values = re.findall(r"\d+", result.stdout)
    if not values:
        raise RuntimeError(f"track: could not read ROI voxel count for {path}: {result.stdout!r}")
    count = int(values[0])
    if count <= 0:
        raise RuntimeError(f"track: warped XTRACT ROI is empty: {path}")
    return count


def _run_track(spec: StageSpec, args, work: Path, runner: CommandRunner) -> dict[str, int]:
    nifti = work / args.profile / "nifti"
    (work / args.profile / "tracts" / "roi").mkdir(parents=True, exist_ok=True)
    plan = spec.track_plan
    if plan is None:
        raise RuntimeError("track: command plan is missing")
    for command in plan.roi_commands:
        result = runner.run(command.argv, stage="track", cwd=nifti)
        if command.validate_nonempty is not None:
            _roi_volume_count(result, command.validate_nonempty)

    counts = {}
    for start in range(0, len(plan.bundles), plan.concurrent_jobs):
        batch = plan.bundles[start:start + plan.concurrent_jobs]
        failures = []
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(runner.run, job.tckgen, stage="track", cwd=nifti): job
                       for job in batch}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append((futures[future].bundle, exc))
        if failures:
            failures.sort(key=lambda item: item[0])
            detail = "; ".join(f"{bundle}: {error}" for bundle, error in failures)
            raise RuntimeError(f"track: {len(failures)} bundle(s) failed: {detail}") from failures[0][1]
        for job in batch:
            counts[job.bundle] = _track_one(job, runner, nifti)
    return counts


def _track_one(job: _TrackBundlePlan, runner: CommandRunner, cwd: Path) -> int:
    info = runner.run(job.tckinfo, stage="track", cwd=cwd)
    match = re.search(r"^\s*count\s*:\s*(\d+)", info.stdout, flags=re.I | re.M)
    if match is None:
        match = re.search(r"^\s*(\d+)\s*$", info.stdout, flags=re.M)
    if match is None:
        raise RuntimeError(f"track: tckinfo did not report a count for {job.temporary}: {info.stdout!r}")
    count = int(match.group(1))
    if count == 0:
        failed = job.final.with_name(job.final.stem + ".FAILED-0.tck")
        os.replace(job.temporary, failed)
        raise ZeroStreamlineError(f"track: {job.bundle} produced zero streamlines; retained {failed.name}")
    os.replace(job.temporary, job.final)
    return count


def _roi_files(source_dir: Path, bundle: str) -> list[tuple[str, str]]:
    files = [("seed.nii.gz", f"{bundle}_seed.nii.gz")]
    for filename in ("target.nii.gz", "target1.nii.gz", "target2.nii.gz"):
        if (source_dir / filename).is_file():
            files.append((filename, f"{bundle}_{filename}"))
    if (source_dir / "exclude.nii.gz").is_file():
        files.append(("exclude.nii.gz", f"{bundle}_exclude.nii.gz"))
    return files


def _run_qc(spec: StageSpec, args, work: Path, runner: CommandRunner) -> dict:
    nifti = work / args.profile / "nifti"
    tracts = work / args.profile / "tracts"
    qc = work / args.profile / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    if args.lesion and spec.commands:
        roi = tracts / "roi"
        roi.mkdir(parents=True, exist_ok=True)
        runner.run(spec.commands[0], stage="qc", cwd=nifti)
    report = bundle_qc(nifti / "mask.nii.gz", tracts, seeds_per_bundle=(
        args.seeds if args.seeds is not None else (750_000 if args.profile == "fast" else 1_500_000)),
        bundles=args.bundles)
    (qc / "bundles.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def bundle_qc(mask_path: str | Path, tracts_dir: str | Path, *, seeds_per_bundle: int,
              bundles: Sequence[str] | None = None) -> dict:
    """Measure crossings and wrong-hemisphere centroids from synthetic or real .tck inputs."""
    mask_image = nib.load(str(mask_path))
    indices = np.argwhere(np.asarray(mask_image.dataobj) > 0)
    if not len(indices):
        raise ValueError("bundle QC mask is empty")
    midline = float(nib.affines.apply_affine(mask_image.affine, indices.mean(axis=0))[0])
    root = Path(tracts_dir)
    paths = [root / f"fx_{bundle}.tck" for bundle in bundles] if bundles is not None \
        else sorted(root.glob("fx_*.tck"))
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"bundle QC tract not found: {path}")
        bundle = path.name[3:-4]
        streamlines = nib.streamlines.load(str(path)).streamlines
        count = len(streamlines)
        crossing = 0
        wrong_hemi = 0
        if count:
            means = []
            for streamline in streamlines:
                points = np.asarray(streamline, dtype=float)
                crossing += int((points[:, 0] < midline - QC_CROSSING_MM).any()
                                and (points[:, 0] > midline + QC_CROSSING_MM).any())
                means.append(float(points[:, 0].mean() - midline))
            side = bundle.rsplit("_", 1)[-1]
            if side == "l":
                wrong_hemi = sum(value > 0 for value in means)
            elif side == "r":
                wrong_hemi = sum(value < 0 for value in means)
        crossing_fraction = crossing / count if count else None
        verdict = "FAIL" if count == 0 else (
            "WARN" if crossing_fraction > QC_WARNING_FRACTION or count < QC_WARNING_MIN_STREAMLINES
            or wrong_hemi / count > QC_WARNING_FRACTION else "PASS")
        rows.append({"bundle": bundle, "streamlines": count, "midline_cross": crossing,
                     "cross_frac": round(crossing_fraction, 4) if count else None,
                     "centroid_wrong_hemi": wrong_hemi, "verdict": verdict})
    overall = "FAIL" if any(row["verdict"] == "FAIL" for row in rows) else (
        "WARN" if any(row["verdict"] == "WARN" for row in rows) else "PASS")
    return {"scope": "descriptive", "tract_validity": "not assessed",
            "midline_x": round(midline, 2), "seeds_per_bundle": seeds_per_bundle,
            "verdict": overall, "bundles": rows}


def _write_capsule_provenance(path: Path, profile: str, seeds: int, *, act: dict | None = None,
                              bundle_rois: dict | None = None) -> None:
    record = {"pipeline": "capsule dwi", "profile": profile, "seeds": seeds,
              "algorithm": TRACKING["algorithm"], "cutoff": TRACKING["cutoff"],
              "act": act, "bundle_rois": bundle_rois}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _run_capsule(spec: StageSpec, args, work: Path, output: Path, runner: CommandRunner,
                 check_verdicts: dict) -> None:
    run_dir = work / args.profile
    tracts = run_dir / "tracts"
    seeds = args.seeds if args.seeds is not None else (750_000 if args.profile == "fast" else 1_500_000)
    provenance = tracts / "capsule-dwi-provenance.json"
    track_params = spec.params.get("track_params", {})
    _write_capsule_provenance(provenance, args.profile, seeds, act=track_params.get("act"),
                              bundle_rois=track_params.get("bundle_rois"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    from . import cli

    argv = list(spec.commands[0][1:])
    runner.log(f"[{_utc_now()}] stage=capsule argv={json.dumps(['capsule', *argv], ensure_ascii=False)}")
    environment_keys = ("PATH", "FSLDIR", "FSLOUTPUTTYPE")
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    os.environ.update({key: runner.env[key] for key in environment_keys})
    try:
        try:
            status = cli.main(argv, command_runner=lambda command: runner.run(
                command, stage="capsule", cwd=work))
        except SystemExit as exc:
            status = int(exc.code or 0)
        except Exception:
            runner.log(f"[{_utc_now()}] stage=capsule exit_code=1")
            raise
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    runner.log(f"[{_utc_now()}] stage=capsule exit_code={status}")
    if status != 0:
        raise RuntimeError(f"stage capsule: existing build-nifti returned {status}")
    inputs = {name: {"path": str(path), "bytes": path.stat().st_size}
              for name, path in _input_paths(args).items()}
    run_record = {
        "inputs": inputs,
        "profile": args.profile,
        "seeds": seeds,
        "shortcut": FAST_SHORTCUT if args.profile == "fast" else None,
        "bundles": list(args.bundles),
        "per_bundle_counts": _read_counts(run_dir / "qc" / "bundles.json"),
        "check_verdicts": check_verdicts,
        "capsule_path": str(output),
        "tracts_reviewed": False,
    }
    (run_dir / "dwi_run.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    runner.log(f"capsule output={output} bytes={output.stat().st_size}")


def _read_counts(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["bundle"]: row["streamlines"] for row in data["bundles"]}


def _check_verdicts(work: Path, profile: str) -> dict:
    qc = work / profile / "qc"
    sentinel = json.loads((qc / "orientation_sentinel.json").read_text(encoding="utf-8"))
    regcheck = json.loads((qc / "regcheck.json").read_text(encoding="utf-8"))
    direction = json.loads((qc / "header_direction.json").read_text(encoding="utf-8"))
    bundles = json.loads((qc / "bundles.json").read_text(encoding="utf-8"))
    return {"orientation_sentinel": sentinel["verdict"], "regcheck": regcheck["verdict"],
            "header_direction_corr": direction,
            "bundle_qc": bundles["verdict"]}


def _plan_report(args, work: Path, output: Path, specs: list[StageSpec]) -> dict:
    seeds = args.seeds if args.seeds is not None else (750_000 if args.profile == "fast" else 1_500_000)
    stages = []
    for spec in specs:
        marker = _marker_path(spec)
        if _stage_is_current(spec, args.profile, work):
            status = "skipped"
            commands = []
        elif args.adopt_existing and spec.name in {"preproc", "t1"} and not marker.exists() \
                and spec.key_outputs and all(path.is_file() for path in spec.key_outputs):
            status = "adopted"
            commands = []
        else:
            status = "planned"
            commands = [list(command) for command in spec.commands]
        stages.append({"stage": spec.name, "status": status, "params": _jsonable(spec.params),
                       "commands": commands})
    return {
        "profile": args.profile,
        "settings": {"fod_algorithm": "SS3T" if args.profile == "full" else "msmt_csd", "seeds": seeds},
        "shortcut": FAST_SHORTCUT if args.profile == "fast" else None,
        "readout_time_note": "GE gives no readout time; topup/eddy are invariant to uniform scale",
        "work": str(work), "output": str(output), "stages": stages,
    }


def configure_parser(parser) -> None:
    parser.add_argument("--dwi")
    parser.add_argument("--bvec")
    parser.add_argument("--bval")
    parser.add_argument("--json", dest="json")
    parser.add_argument("--rpe")
    parser.add_argument("--rpe-bvec")
    parser.add_argument("--rpe-bval")
    parser.add_argument("--pe-dir", default="j-")
    parser.add_argument("--readout-time", type=float, default=0.05)
    parser.add_argument("--t1")
    parser.add_argument("--flair")
    parser.add_argument("--lesion")
    parser.add_argument("--label")
    parser.add_argument("-o", "--output")
    parser.add_argument("--work")
    parser.add_argument("--profile", choices=("full", "fast"), default="full")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--bundles", nargs="+", default=list(DEFAULT_BUNDLES))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--adopt-existing", action="store_true")
    parser.add_argument("--check-tools", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--act", choices=("auto", "off"), default="auto",
                        help="anatomically constrained tracking from a SynthStrip-masked T1 5TT (default: auto)")
    parser.add_argument("--no-deface", action="store_true",
                        help="build the capsule without face removal")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capsule dwi")
    configure_parser(parser)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["dwi"]:
        arguments = arguments[1:]
    return run(parser.parse_args(arguments))


def run(args) -> int:
    """Run a pipeline request, or print its fully ordered plan without writes or execution."""
    missing = check_tools(args.profile, bundles=args.bundles, act=getattr(args, "act", "auto"))
    if missing:
        print("Missing DWI dependencies:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    if args.check_tools:
        print(f"DWI tool check passed for profile={args.profile}")
        return 0
    try:
        work, output, tools = _validate_args(args)
        specs = _stage_specs(args, work, output, tools)
        capsule_spec = specs[-1]
        if output.exists() and _stage_is_current(capsule_spec, args.profile, work):
            pass
        elif output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        if args.dry_run:
            print(json.dumps(_plan_report(args, work, output, specs), indent=2))
            return 0

        run_dir = work / args.profile
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "dwi.log"
        shortcut_line = f"shortcut={FAST_SHORTCUT}" if args.profile == "fast" else "shortcut=none"
        seeds = args.seeds if args.seeds is not None else (750_000 if args.profile == "fast" else 1_500_000)
        readout_line = "GE gives no readout time; topup/eddy are invariant to uniform scale"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"profile={args.profile}\n{shortcut_line}\n"
                      f"seeds_per_bundle={seeds}\nreadout_time={args.readout_time}; {readout_line}\n")
        runner = CommandRunner(log_path, tools, work, args.threads)
        versions = _tool_versions(runner, tools, args.profile)
        check_verdicts = {}
        for spec in specs:
            if spec.name == "preproc":
                action = lambda spec=spec: _run_preproc(spec, args, work, runner)
            elif spec.name == "t1":
                action = lambda spec=spec: _run_t1(spec, args, work, runner)
            elif spec.name == "register":
                action = lambda spec=spec: check_verdicts.update(_run_register(
                    spec, args, work, work / args.profile / "qc", tools, runner))
            elif spec.name == "act":
                action = lambda spec=spec: _run_act(spec, args, work, runner)
            elif spec.name == "fod":
                action = lambda spec=spec: _run_fod(spec, runner, work / args.profile / "nifti")
            elif spec.name == "track":
                action = lambda spec=spec: check_verdicts.update({"per_bundle_counts": _run_track(
                    spec, args, work, runner)})
            elif spec.name == "qc":
                action = lambda spec=spec: check_verdicts.update({"bundle_qc": _run_qc(
                    spec, args, work, runner)["verdict"]})
            else:
                action = lambda spec=spec: _run_capsule(
                    spec, args, work, output, runner, _check_verdicts(work, args.profile))
            result = run_stage(spec, profile=args.profile, work_dir=work, tool_versions=versions,
                               adopt_existing=args.adopt_existing, action=action)
            runner.log(f"stage={spec.name} status={result.status}")
        print(f"DWI capsule written: {output}")
        print(f"profile={args.profile}; seeds_per_bundle={args.seeds or (750_000 if args.profile == 'fast' else 1_500_000)}")
        print("tracts_reviewed=false")
        return 0
    except ZeroStreamlineError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, FileExistsError, IsADirectoryError,
            RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"capsule dwi: {exc}", file=sys.stderr)
        return 1
