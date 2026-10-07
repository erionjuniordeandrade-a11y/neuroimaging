"""Manifest-driven CR peel command line runner.

Research visualization only · enhancing vessels · not navigation, not a device
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import BinaryIO

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from tractlab.cr.peel import PeelParams, PeelResult, envelope, head_mask, run_peel
from tractlab.serve import _resolve_case_path


HONESTY_WALL = "Research visualization only · enhancing vessels · not navigation, not a device"
SUMMARY_SCHEMA = "tractlab.cr.summary/1"


class CliError(ValueError):
    """A user-facing validation refusal."""


@dataclass(frozen=True)
class ManifestInput:
    root: Path
    root_literal: str
    t1_path: Path
    mask_path: Path
    payload: dict


@dataclass(frozen=True)
class LoadedInput:
    t1_image: nib.spatialimages.SpatialImage
    mask_image: nib.spatialimages.SpatialImage
    canonical_affine: np.ndarray
    canonical_shape: tuple[int, int, int]
    volume: np.ndarray
    brain: np.ndarray
    vx: float


@dataclass(frozen=True)
class OutputPlan:
    output_dir: Path
    depths_mm: tuple[float, ...]
    depth_tags: tuple[str, ...]
    binary_names: tuple[str, ...]


def _parse_depths(value: str) -> tuple[float, ...]:
    try:
        start, stop, step = (float(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("depths must use START:STOP:STEP") from exc
    if not all(math.isfinite(item) for item in (start, stop, step)):
        raise argparse.ArgumentTypeError("depths must be finite")
    if start < 0.0 or stop < start or step <= 0.0:
        raise argparse.ArgumentTypeError("depths require 0 <= START <= STOP and STEP > 0")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    if count > 10_000:
        raise argparse.ArgumentTypeError("depths expand to more than 10000 columns")
    return tuple(float(start + index * step) for index in range(count))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m tractlab.cr")
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--depths", type=_parse_depths, default=_parse_depths("0:30:1"))
    parser.add_argument("--vessel-pct", type=float, default=97.5)
    parser.add_argument("--tophat-frac", type=float, default=0.5)
    parser.add_argument("--head-thr", type=float, default=200.0)
    parser.add_argument("--direction-sigma", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _params_from_args(args: argparse.Namespace) -> PeelParams:
    numeric = {
        "vessel-pct": args.vessel_pct,
        "tophat-frac": args.tophat_frac,
        "head-thr": args.head_thr,
        "direction-sigma": args.direction_sigma,
    }
    for name, value in numeric.items():
        if not math.isfinite(value):
            raise CliError(f"--{name} must be finite")
    if not 0.0 <= args.vessel_pct <= 100.0:
        raise CliError("--vessel-pct must be between 0 and 100")
    if args.tophat_frac < 0.0:
        raise CliError("--tophat-frac must be non-negative")
    if args.direction_sigma <= 0.0:
        raise CliError("--direction-sigma must be positive")
    return PeelParams(
        depths_mm=args.depths,
        vessel_pct=float(args.vessel_pct),
        tophat_frac=float(args.tophat_frac),
        head_thr=float(args.head_thr),
        direction_sigma_mm=float(args.direction_sigma),
    )


def _load_manifest(case_root: Path) -> dict:
    manifest_path = case_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CliError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"invalid manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CliError("manifest.json must contain a JSON object")
    if manifest.get("deid") is not True:
        raise CliError("manifest field 'deid' must be true")
    return manifest


def _require_object(parent: dict, field: str) -> dict:
    value = parent.get(field)
    if not isinstance(value, dict):
        raise CliError(f"manifest field '{field}' must be an object")
    return value


def _require_text(parent: dict, field: str) -> str:
    value = parent.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CliError(f"manifest field '{field}' must be a non-empty string")
    return value


def _validate_input(manifest: dict) -> ManifestInput:
    inputs = _require_object(manifest, "inputs")
    payload = _require_object(inputs, "t1c_unstripped")
    root_text = _require_text(payload, "root")
    for field in ("path", "brain_mask", "series", "contrast"):
        _require_text(payload, field)
    _require_object(payload, "provenance")

    root = Path(root_text).expanduser()
    if not root.is_absolute():
        raise CliError("manifest field 'inputs.t1c_unstripped.root' must be an absolute directory")
    root = root.resolve()
    if not root.is_dir():
        raise CliError(f"input root is not a directory: {root}")
    try:
        t1_path = Path(
            _resolve_case_path(str(root), payload["path"], what="inputs.t1c_unstripped.path")
        )
        mask_path = Path(
            _resolve_case_path(
                str(root), payload["brain_mask"], what="inputs.t1c_unstripped.brain_mask"
            )
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    for label, path in (("T1", t1_path), ("brain mask", mask_path)):
        if not path.is_file():
            raise CliError(f"{label} file not found: {path}")
    return ManifestInput(
        root=root,
        root_literal=root_text,
        t1_path=t1_path,
        mask_path=mask_path,
        payload=payload,
    )


def _load_matching_images(
    spec: ManifestInput,
) -> tuple[nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage]:
    try:
        t1_image = nib.load(str(spec.t1_path))
        mask_image = nib.load(str(spec.mask_path))
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise CliError(f"could not load NIfTI input: {exc}") from exc
    if t1_image.shape != mask_image.shape or not np.array_equal(t1_image.affine, mask_image.affine):
        raise CliError(
            "T1/brain-mask grid mismatch; exact shape and affine equality are required\n"
            f"T1 shape: {t1_image.shape}\n"
            f"mask shape: {mask_image.shape}\n"
            f"T1 affine:\n{np.array2string(t1_image.affine)}\n"
            f"mask affine:\n{np.array2string(mask_image.affine)}"
        )
    return t1_image, mask_image


def _canonicalize(spec: ManifestInput) -> LoadedInput:
    t1_image, mask_image = _load_matching_images(spec)
    if len(t1_image.shape) != 3:
        raise CliError(f"T1 and brain mask must be 3-D, got shape {t1_image.shape}")
    t1_canonical = nib.as_closest_canonical(t1_image)
    mask_canonical = nib.as_closest_canonical(mask_image)
    if t1_canonical.shape != mask_canonical.shape or not np.array_equal(
        t1_canonical.affine, mask_canonical.affine
    ):
        raise CliError("canonical T1/brain-mask grids differ after orientation")

    spatial_units = {
        "T1": t1_canonical.header.get_xyzt_units()[0],
        "brain mask": mask_canonical.header.get_xyzt_units()[0],
    }
    invalid_units = [label for label, unit in spatial_units.items() if unit != "mm"]
    if invalid_units:
        detail = ", ".join(f"{label}={spatial_units[label]!r}" for label in invalid_units)
        raise CliError(f"canonical spatial units must be mm; got {detail}")

    zooms = np.asarray(t1_canonical.header.get_zooms()[:3], dtype=np.float64)
    if zooms.shape != (3,) or not np.all(np.isfinite(zooms)) or np.any(zooms <= 0.0):
        raise CliError(f"invalid canonical voxel sizes: {zooms.tolist()}")
    vx = float(zooms[0])
    linear = np.asarray(t1_canonical.affine[:3, :3], dtype=np.float64)
    gram = linear.T @ linear
    target = np.eye(3, dtype=np.float64) * vx**2
    if not np.all(np.isfinite(gram)) or np.any(np.abs(gram - target) > 0.02 * vx**2):
        raise CliError(
            f"canonical affine is not Euclidean isotropic within 2% (voxel sizes "
            f"{zooms.tolist()}); conform the T1 and brain mask first (no silent resampling)\n"
            f"A.T @ A:\n{np.array2string(gram, precision=8)}"
        )

    volume = np.asarray(t1_canonical.dataobj, dtype=np.float32)
    brain = np.asarray(mask_canonical.dataobj) > 0
    if not np.all(np.isfinite(volume)):
        raise CliError("T1 contains non-finite intensities")
    if not brain.any():
        raise CliError("brain mask is empty")
    return LoadedInput(
        t1_image=t1_image,
        mask_image=mask_image,
        canonical_affine=np.asarray(t1_canonical.affine, dtype=np.float64),
        canonical_shape=tuple(int(item) for item in t1_canonical.shape),
        volume=volume,
        brain=brain,
        vx=vx,
    )


def _head_qc(volume: np.ndarray, brain: np.ndarray, vx: float, params: PeelParams) -> None:
    threshold = params.head_thr
    blur_vox = params.head_blur_mm / vx
    candidate = ndi.gaussian_filter(np.asarray(volume, np.float32), blur_vox) > threshold
    candidate = ndi.binary_opening(candidate, iterations=2)
    candidate = ndi.binary_fill_holes(candidate)
    _, component_count = ndi.label(candidate)
    finite = volume[np.isfinite(volume)]
    volume_p99 = float(np.percentile(finite, 99))
    suggested = 0.25 * volume_p99
    brain_voxels = np.argwhere(brain)
    centroid = np.rint(brain_voxels.mean(axis=0)).astype(int)
    column_x = int(np.clip(centroid[0], 0, volume.shape[0] - 1))
    column_y = int(np.clip(centroid[1], 0, volume.shape[1] - 1))
    column_brain = np.flatnonzero(brain[column_x, column_y])
    if column_brain.size == 0:
        raise CliError("rounded brain-centroid superior column does not intersect the brain mask")
    brain_top = int(column_brain.max())
    profile_stop = min(brain_top + 40, volume.shape[2])
    profile = volume[column_x, column_y, brain_top:profile_stop]
    profile_text = ", ".join(f"{float(value):.3g}" for value in profile)
    print(
        f"head QC: superior column x={column_x}, y={column_y}, brain_top_z={brain_top}; "
        f"values z=[{brain_top},{profile_stop}) = [{profile_text}]"
    )
    try:
        head = head_mask(volume, threshold, blur_vox)
    except ValueError as exc:
        print(f"head QC: head-mask components = {component_count}; fraction_fov = 0.0000 (0.0%)")
        print(
            f"head QC: suggested head threshold = {suggested:.3g} "
            f"(0.25 x volume p99 {volume_p99:.3g})"
        )
        raise CliError(str(exc)) from exc

    fraction = float(head.mean())
    print(
        f"head QC: head-mask components = {component_count}; "
        f"fraction_fov = {fraction:.4f} ({100.0 * fraction:.1f}%)"
    )
    print(
        f"head QC: suggested head threshold = {suggested:.3g} "
        f"(0.25 x volume p99 {volume_p99:.3g})"
    )
    head_envelope = envelope(head, vx, params.env_sigma_mm)
    brain_outside = int(np.count_nonzero(brain & ~head_envelope))
    print(f"head QC: brain-mask voxels outside head envelope = {brain_outside}")
    if brain_outside:
        raise CliError(
            f"{brain_outside} brain-mask voxels lie outside the head envelope: "
            "head_thr ate the scalp there (check the superior column profile)"
        )
    # a 0.7 mm head in a 256 mm FOV is ~27 % of the volume; <10 % means the threshold
    # ate the scalp, >95 % means it swallowed the table/cushion
    if fraction < 0.10 or fraction > 0.95:
        raise CliError(
            f"head_mask occupies {100.0 * fraction:.1f}% of the FOV; required range is 10%-95%"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hashes(spec: ManifestInput) -> dict[str, str]:
    return {
        "t1c_unstripped": _sha256(spec.t1_path),
        "brain_mask": _sha256(spec.mask_path),
    }


def _verify_input_hashes(spec: ManifestInput, expected: dict[str, str]) -> None:
    current = _input_hashes(spec)
    changed = [label for label, digest in current.items() if digest != expected[label]]
    if changed:
        raise CliError(
            "input changed after pre-load hashing and before publish: " + ", ".join(changed)
        )


def _atomic_binary(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_binary(path, lambda stream: stream.write(encoded))


def _depth_tag(depth: float) -> str:
    if math.isclose(depth, round(depth), abs_tol=1e-8):
        return f"d{int(round(depth)):02d}"
    text = f"{depth:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        return f"d{int(text):02d}"
    whole, fraction = text.split(".")
    return f"d{int(whole):02d}p{fraction}"


def _assert_safe_output_components(plan: OutputPlan) -> None:
    output_dir = plan.output_dir
    if output_dir.is_symlink():
        raise CliError("refusing symlinked CR output directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise CliError("CR output path exists but is not a directory")
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_symlink():
                raise CliError(f"refusing symlinked CR output component: {child.name}")
            if child.is_dir():
                raise CliError(f"refusing unexpected CR output directory: {child.name}")
    for name in (*plan.binary_names, "summary.json"):
        if (output_dir / name).is_symlink():
            raise CliError(f"refusing symlinked CR output component: {name}")


def _output_plan(case_root: Path, depths_mm: Sequence[float]) -> OutputPlan:
    depths = tuple(float(depth) for depth in depths_mm)
    tags: list[str] = []
    seen: set[str] = set()
    for depth in depths:
        tag = _depth_tag(depth)
        if tag in seen:
            raise CliError(f"depth filename collision at {depth} mm ({tag})")
        seen.add(tag)
        tags.append(tag)
    binary_names = ["peel.npz"]
    for tag in tags:
        binary_names.extend(
            (f"{tag}_grey.ply", f"{tag}_vmax.ply", f"{tag}_vessel.ply")
        )
    if len(binary_names) != len(set(binary_names)):
        raise CliError("derived output filename collision")
    plan = OutputPlan(
        output_dir=case_root / "cr",
        depths_mm=depths,
        depth_tags=tuple(tags),
        binary_names=tuple(binary_names),
    )
    _assert_safe_output_components(plan)
    return plan


def _grey_rgb(grey: np.ndarray, in_brain: np.ndarray) -> np.ndarray:
    finite = np.isfinite(grey)
    reference = grey[in_brain & finite]
    # Depths before the cortex can have no in-brain vertices. The prototype's safe
    # fallback keeps those context meshes visible while all populated depths use the
    # contract's per-depth in-brain p2-p99.3 window.
    if reference.size == 0:
        reference = grey[finite]
    if reference.size == 0:
        raise CliError("cannot window a depth with no finite grey samples")
    low, high = (float(item) for item in np.percentile(reference, (2.0, 99.3)))
    scaled = np.clip((grey - low) / max(high - low, 1e-6), 0.0, 1.0)
    channel = np.rint(scaled * 255.0).astype(np.uint8)
    return np.repeat(channel[:, None], 3, axis=1)


def _write_ply(
    stream: BinaryIO, verts_world: np.ndarray, faces: np.ndarray, rgb: np.ndarray
) -> None:
    verts = np.asarray(verts_world, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    colours = np.asarray(rgb, dtype=np.uint8)
    if verts.ndim != 2 or verts.shape[1] != 3 or colours.shape != verts.shape:
        raise CliError("invalid PLY vertex or RGB array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise CliError("invalid PLY face array")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    stream.write(header.encode("ascii"))
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    vertex_data = np.empty(len(verts), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = verts.T
    vertex_data["r"], vertex_data["g"], vertex_data["b"] = colours.T
    stream.write(vertex_data.tobytes())
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_data = np.empty(len(triangles), dtype=face_dtype)
    face_data["count"] = 3
    face_data["indices"] = triangles
    stream.write(face_data.tobytes())


def _git_state() -> tuple[str, bool, str | None]:
    repository = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "-C", str(repository), "diff", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CliError(f"could not read Git code revision: {exc}") from exc
    if len(revision) != 40:
        raise CliError(f"unexpected Git revision: {revision!r}")
    dirty = bool(status.strip())
    diff_sha256 = hashlib.sha256(tracked_diff).hexdigest() if dirty else None
    return revision, dirty, diff_sha256


def _input_receipt(
    spec: ManifestInput, loaded: LoadedInput, input_hashes: dict[str, str]
) -> dict[str, dict]:
    canonical_grid = {
        "canonical_shape": list(loaded.canonical_shape),
        "canonical_affine": loaded.canonical_affine.tolist(),
    }
    return {
        "t1c_unstripped": {
            "root_literal": spec.root_literal,
            "root_sha256": hashlib.sha256(str(spec.root).encode("utf-8")).hexdigest(),
            "path": spec.payload["path"],
            "series": spec.payload["series"],
            "contrast": spec.payload["contrast"],
            "provenance": spec.payload["provenance"],
            "sha256": input_hashes["t1c_unstripped"],
            "shape": list(loaded.t1_image.shape),
            "affine": np.asarray(loaded.t1_image.affine, dtype=np.float64).tolist(),
            **canonical_grid,
        },
        "brain_mask": {
            "path": spec.payload["brain_mask"],
            "sha256": input_hashes["brain_mask"],
            "shape": list(loaded.mask_image.shape),
            "affine": np.asarray(loaded.mask_image.affine, dtype=np.float64).tolist(),
            **canonical_grid,
        },
    }


def _string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def _assert_phi_safe_receipt(receipt: dict) -> None:
    username = getpass.getuser()
    for value in _string_values(receipt):
        if "/Users/" in value or "/home/" in value or (username and username in value):
            raise CliError("receipt PHI self-scan found a home path or current username")


def _publish_staged_bundle(plan: OutputPlan, stage_dir: Path) -> None:
    output_dir = plan.output_dir
    previous_dir = stage_dir / ".previous"
    failed_dir = stage_dir / ".failed-new"
    previous: list[str] = []
    published: list[str] = []
    try:
        current_entries = []
        for child in output_dir.iterdir():
            if child == stage_dir:
                continue
            if child.is_symlink():
                raise CliError(f"refusing symlinked CR output component: {child.name}")
            if child.is_dir():
                raise CliError(f"refusing unexpected CR output directory: {child.name}")
            current_entries.append(child)

        previous_dir.mkdir()
        for child in sorted(current_entries, key=lambda item: item.name):
            os.replace(child, previous_dir / child.name)
            previous.append(child.name)

        for name in (*plan.binary_names, "summary.json"):
            os.replace(stage_dir / name, output_dir / name)
            published.append(name)
    except Exception as publish_error:
        rollback_errors: list[OSError] = []
        failed_dir.mkdir(exist_ok=True)
        for name in reversed(published):
            try:
                current = output_dir / name
                if current.exists() or current.is_symlink():
                    os.replace(current, failed_dir / name)
            except OSError as exc:
                rollback_errors.append(exc)
        for name in reversed(previous):
            try:
                prior = previous_dir / name
                if prior.exists() or prior.is_symlink():
                    os.replace(prior, output_dir / name)
            except OSError as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            raise CliError(
                f"CR publish failed and rollback had {len(rollback_errors)} error(s)"
            ) from publish_error
        raise
    shutil.rmtree(stage_dir)


def _write_outputs(
    case_root: Path,
    plan: OutputPlan,
    manifest: dict,
    spec: ManifestInput,
    loaded: LoadedInput,
    input_hashes: dict[str, str],
    params: PeelParams,
    result: PeelResult,
    revision: str,
    dirty: bool,
    code_diff_sha256: str | None,
) -> Path:
    if tuple(float(depth) for depth in result.depths_mm) != plan.depths_mm:
        raise CliError("peel result depths differ from the prevalidated output plan")
    _assert_safe_output_components(plan)
    output_dir = plan.output_dir
    output_dir_preexisting = output_dir.exists()
    stage_dir = output_dir / f".stage-{os.getpid()}"
    verts_world = nib.affines.apply_affine(
        loaded.canonical_affine, result.verts_vox.reshape(-1, 3)
    ).reshape(result.verts_vox.shape).astype(np.float32)

    def write_npz(stream: BinaryIO) -> None:
        np.savez_compressed(
            stream,
            faces=np.asarray(result.faces, dtype=np.int32),
            verts_world=verts_world,
            grey=np.asarray(result.grey, dtype=np.float32),
            vmax=np.asarray(result.vmax, dtype=np.float32),
            vessel=np.asarray(result.vessel, dtype=bool),
            in_brain=np.asarray(result.in_brain, dtype=bool),
            valid=np.asarray(result.valid, dtype=bool),
            depths_mm=np.asarray(result.depths_mm, dtype=np.float32),
        )

    try:
        if not output_dir_preexisting:
            output_dir.mkdir()
        if output_dir.is_symlink():
            raise CliError("refusing symlinked CR output directory")
        stage_dir.mkdir()
        output_paths = [stage_dir / "peel.npz"]
        _atomic_binary(output_paths[0], write_npz)

        for index, tag in enumerate(plan.depth_tags):
            grey_rgb = _grey_rgb(result.grey[index], result.in_brain[index])
            vmax_rgb = _grey_rgb(result.vmax[index], result.in_brain[index])
            vessel_rgb = grey_rgb.copy()
            vessel_rgb[result.vessel[index]] = np.array([140, 18, 30], dtype=np.uint8)
            for variant, colours in (
                ("grey", grey_rgb),
                ("vmax", vmax_rgb),
                ("vessel", vessel_rgb),
            ):
                path = stage_dir / f"{tag}_{variant}.ply"
                _atomic_binary(
                    path,
                    lambda stream, vertices=verts_world[index], rgb=colours: _write_ply(
                        stream, vertices, result.faces, rgb
                    ),
                )
                output_paths.append(path)

        if tuple(path.name for path in output_paths) != plan.binary_names:
            raise CliError("staged outputs differ from the prevalidated output plan")
        outputs = {
            (Path("cr") / path.name).as_posix(): _sha256(path) for path in output_paths
        }

        _verify_input_hashes(spec, input_hashes)
        receipt = {
            "schema": SUMMARY_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "honesty_wall": HONESTY_WALL,
            "inputs": _input_receipt(spec, loaded, input_hashes),
            "verified_before_publish": True,
            "vx": loaded.vx,
            "code_revision": revision,
            "code_dirty": dirty,
            "params": asdict(params),
            "marker_depth_mm": float(result.marker_depth_mm),
            "marker_p10_p90_mm": [float(item) for item in result.marker_p10_p90_mm],
            "band_mm": [float(item) for item in result.band_mm],
        "vessel_thr": float(result.vessel_thr),
            "tophat_thr": float(result.tophat_thr),
            "n_vertices": int(result.verts_vox.shape[1]),
            "n_faces": int(len(result.faces)),
            "brain_outside_envelope": int(result.brain_outside_envelope),
            "per_depth": [
                {
                    "depth_mm": float(depth),
                    "depth_residual_mm": float(result.depth_residual_mm[index]),
                    "folded_faces": int(result.folded_faces[index]),
                    "folded_faces_convexity": int(result.folded_faces_convexity[index]),
                    "invalid_columns": int((~result.valid[index]).sum()),
                }
                for index, depth in enumerate(result.depths_mm)
            ],
            # The receipt cannot cryptographically contain its own digest. Git tracks the
            # receipt; this map binds every derived binary artifact it describes.
            "outputs": outputs,
        }
        case_id = manifest.get("case_id")
        if isinstance(case_id, str) and re.fullmatch(r"[a-z0-9-]+", case_id):
            receipt["case_id"] = case_id
        if dirty:
            if code_diff_sha256 is None:
                raise CliError("dirty Git state is missing its tracked-diff digest")
            receipt["code_diff_sha256"] = code_diff_sha256
        _assert_phi_safe_receipt(receipt)
        _atomic_json(stage_dir / "summary.json", receipt)
        _publish_staged_bundle(plan, stage_dir)
        return output_dir / "summary.json"
    except Exception:
        if stage_dir.exists() and stage_dir.is_dir():
            shutil.rmtree(stage_dir)
        if not output_dir_preexisting and output_dir.exists() and output_dir.is_dir():
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        params = _params_from_args(args)
        case_root = args.case_root.expanduser().resolve()
        output_plan = _output_plan(case_root, params.depths_mm)
        manifest = _load_manifest(case_root)
        spec = _validate_input(manifest)
        input_hashes = _input_hashes(spec)
        loaded = _canonicalize(spec)
        _head_qc(loaded.volume, loaded.brain, loaded.vx, params)
        if args.dry_run:
            print(
                f"dry run: would run {len(params.depths_mm)} depths "
                f"({params.depths_mm[0]:g}-{params.depths_mm[-1]:g} mm) at vx={loaded.vx:g} mm"
            )
            print(f"dry run: would write CR outputs under {case_root / 'cr'}")
            print(f"dry run: PeelParams = {json.dumps(asdict(params), sort_keys=True)}")
            return 0

        revision, dirty, code_diff_sha256 = _git_state()
        print(
            f"running CR peel: shape={loaded.canonical_shape} vx={loaded.vx:g} mm "
            f"depths={len(params.depths_mm)}"
        )
        result = run_peel(loaded.volume, loaded.brain, loaded.vx, params)
        summary_path = _write_outputs(
            case_root,
            output_plan,
            manifest,
            spec,
            loaded,
            input_hashes,
            params,
            result,
            revision,
            dirty,
            code_diff_sha256,
        )
        print(
            f"marker={result.marker_depth_mm:.3g} mm; vessel_thr={result.vessel_thr:.6g}; "
            f"tophat_thr={result.tophat_thr:.6g}"
        )
        print(f"wrote {summary_path}")
        return 0
    except (CliError, OSError, ValueError) as exc:
        print(f"tractlab.cr: refused: {exc}", file=sys.stderr)
        return 2
