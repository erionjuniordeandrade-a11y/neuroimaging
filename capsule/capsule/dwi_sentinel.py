"""Check orientation through the MNI, T1, and DWI transform chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Callable

import nibabel as nib
import numpy as np


MARKERS = {
    "RIGHT": dict(value=3000.0, axis=0, mm=40.0, expect="+"),
    "LEFT": dict(value=2800.0, axis=0, mm=-60.0, expect="-"),
    "ANTERIOR": dict(value=2400.0, axis=1, mm=50.0, expect="+"),
    "POSTERIOR": dict(value=2200.0, axis=1, mm=-70.0, expect="-"),
    "SUPERIOR": dict(value=2000.0, axis=2, mm=55.0, expect="+"),
    "INFERIOR": dict(value=1800.0, axis=2, mm=-45.0, expect="-"),
}
RADIUS_MM = 8.0


def _fsldir(fsldir: str | Path | None = None) -> Path:
    return Path(fsldir or os.environ.get("FSLDIR", Path.home() / "fsl")).expanduser()


def plant(dst: str | Path, *, mirror: bool = False, mni_path: str | Path) -> dict:
    """Plant distinct, asymmetric markers using the MNI image affine."""
    img = nib.load(str(mni_path))
    affine = img.affine
    shape = img.shape[:3]
    volume = np.zeros(shape, np.float32)
    axis_of = {}
    for world_axis in range(3):
        array_axis = int(np.argmax(np.abs(affine[world_axis, :3])))
        axis_of[world_axis] = (array_axis, float(np.sign(affine[world_axis, array_axis])),
                               float(abs(affine[world_axis, array_axis])))

    origin_voxel = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), [0.0, 0.0, 0.0])).astype(int)
    placed = {}
    for name, spec in MARKERS.items():
        array_axis, sign, step = axis_of[spec["axis"]]
        voxel = origin_voxel.copy()
        voxel[array_axis] = int(round(voxel[array_axis] + sign * spec["mm"] / step))
        radius = int(round(RADIUS_MM / step))
        slices = tuple(slice(max(0, value - radius), min(size, value + radius + 1))
                       for value, size in zip(voxel, shape))
        volume[slices] = spec["value"]
        placed[name] = {
            "value": spec["value"],
            "planted_world": [round(float(value), 2)
                              for value in nib.affines.apply_affine(affine, [voxel])[0]],
            "intended_mm": spec["mm"],
            "intended_world_axis": spec["axis"],
        }

    if mirror:
        volume = np.flip(volume, axis=axis_of[0][0]).copy()
    nib.save(nib.Nifti1Image(volume, affine, img.header.copy()), str(dst))
    return placed


def command_plan(nd: str | Path, *, fsldir: str | Path | None = None) -> list[list[str]]:
    """Return the two applywarp argv lists used by this sentinel."""
    nd = Path(nd)
    fsl = _fsldir(fsldir)
    temporary = nd / "_sentinel"
    warp = nd / "mni2t1_warp.nii.gz"
    post = nd / "t12b0.mat"
    reference = nd / "b0.nii.gz"
    commands = []
    for tag in ("true", "mirrored_control"):
        source = temporary / f"sent_{tag}_mni.nii.gz"
        target = temporary / f"sent_{tag}_dwi.nii.gz"
        commands.append([
            str(fsl / "bin" / "applywarp"), "-i", str(source), "-r", str(reference),
            "-w", str(warp), f"--postmat={post}", "-o", str(target), "--interp=nn",
        ])
    return commands


def prepare(nd: str | Path, *, fsldir: str | Path | None = None) -> dict[str, dict]:
    """Plant the real and mirrored marker volumes before the shared warp commands run."""
    nd = Path(nd)
    temporary = nd / "_sentinel"
    temporary.mkdir(parents=True, exist_ok=True)
    mni_path = _fsldir(fsldir) / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
    for required in (nd / "mni2t1_warp.nii.gz", nd / "t12b0.mat", nd / "b0.nii.gz", mni_path):
        if not required.exists():
            raise FileNotFoundError(f"orientation sentinel input is missing: {required}")
    return {
        "true": plant(temporary / "sent_true_mni.nii.gz", mni_path=mni_path),
        "mirrored_control": plant(temporary / "sent_mirrored_control_mni.nii.gz",
                                   mirror=True, mni_path=mni_path),
    }


def measure(path: str | Path) -> dict:
    """Measure each marker's destination world coordinates."""
    img = nib.load(str(path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    result = {}
    for name, spec in MARKERS.items():
        voxels = np.argwhere(np.isclose(volume, spec["value"], rtol=0, atol=1.0))
        if not len(voxels):
            result[name] = None
            continue
        world = nib.affines.apply_affine(img.affine, voxels).mean(0)
        result[name] = {"n_voxels": int(len(voxels)),
                        "world": [round(float(value), 2) for value in world]}
    return result


def judge(found: dict) -> tuple[bool, list[tuple[str, float | None, str, bool]]]:
    """Require every named marker to land on its expected world-axis sign."""
    rows = []
    passed = True
    for name, spec in MARKERS.items():
        item = found.get(name)
        if item is None:
            rows.append((name, None, "LOST", False))
            passed = False
            continue
        coordinate = item["world"][spec["axis"]]
        correct = coordinate > 0 if spec["expect"] == "+" else coordinate < 0
        rows.append((name, coordinate, "ok" if correct else "WRONG SIDE", correct))
        passed = passed and correct
    return passed, rows


def evaluate(nd: str | Path, out_json: str | Path, planted: dict[str, dict]) -> dict:
    """Measure both controls, save their verdicts at the required explicit path, and return them."""
    nd = Path(nd)
    temporary = nd / "_sentinel"
    results = {}
    for tag in ("true", "mirrored_control"):
        found = measure(temporary / f"sent_{tag}_dwi.nii.gz")
        ok, rows = judge(found)
        results[tag] = {"planted": planted[tag], "measured": found,
                        "pass_": ok, "rows": rows}

    real = results["true"]
    control = results["mirrored_control"]
    blind = control["pass_"]
    passed = not blind and real["pass_"]
    record = {
        "chain": "MNI -> mni2t1_warp -> t12b0.mat -> DWI",
        "real": real,
        "mirrored_control": control,
        "control_correctly_failed": not blind,
        "verdict": "PASS" if passed else "FAIL",
    }
    destination = Path(out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(
    nd: str | Path,
    out_json: str | Path,
    *,
    fsldir: str | Path | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> int:
    """Run both orientation controls and write the result to the explicit JSON path."""
    nd = Path(nd)
    planted = prepare(nd, fsldir=fsldir)
    runner = command_runner or (lambda argv, **kwargs: _default_command_runner(
        argv, fsldir=_fsldir(fsldir), **kwargs))
    for argv in command_plan(nd, fsldir=fsldir):
        result = runner(argv, cwd=nd, stage="register")
        if result.returncode:
            raise RuntimeError(f"orientation sentinel applywarp failed ({result.returncode}): "
                               f"{result.stderr or result.stdout}")
    record = evaluate(nd, out_json, planted)
    return 0 if record["verdict"] == "PASS" else 1


def _default_command_runner(argv: list[str], *, cwd: Path, stage: str, fsldir: Path):
    env = {**os.environ, "FSLDIR": str(fsldir), "FSLOUTPUTTYPE": "NIFTI_GZ"}
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)
