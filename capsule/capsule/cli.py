"""Command line entry points for scanning and building an offline capsule."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import SimpleITK as sitk

from .ingest import Series, convert_series, scan_series
from .pack import write_capsule
from .tracts import LesionGrid, compute_tract_trust, decimate, decode_tck, encode_tck, outlier_mask, subsample
from . import anatomy as anatomyops
from . import deface as defaceops
from . import masks as maskops
from .resample import (DEFAULT_MAX_VOXELS, GridPlan, affine_ras, common_grid, first_volume,
                       mask_ras_bounds, pack_scalar, resample_to_grid, scalar_stats)
from .segment import DEFAULT_NNINTERACTIVE_PYTHON, segment_capsule
from . import dwi as dwiops


_CT_WINDOWS = [("Cérebro", 40, 80), ("Subdural", 75, 215), ("AVC", 35, 40),
               ("Partes moles", 40, 400), ("Osso", 600, 2800), ("Osso temporal", 700, 4000)]


def _selected_series(all_series: list[Series], numbers: str) -> list[Series]:
    try:
        requested = [int(part.strip()) for part in numbers.split(",")]
    except ValueError as exc:
        raise ValueError("--series must be comma-separated series numbers") from exc
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--series must contain unique series numbers")
    result = []
    for number in requested:
        matches = [series for series in all_series if series.number == number]
        if len(matches) != 1:
            raise ValueError(f"series {number} has {len(matches)} matches; select a unique series number")
        if matches[0].modality not in {"CT", "MR"}:
            raise ValueError(f"series {number} is not CT or MR")
        result.append(matches[0])
    return result


def _reference_index(selected: list[Series], number: int | None) -> int:
    if number is not None:
        for index, series in enumerate(selected):
            if series.number == number:
                return index
        raise ValueError("--reference must be one of the selected series")
    return next((index for index, series in enumerate(selected) if series.modality == "CT"), 0)


def _volume_id(kind: str, number: int | None, used: set[str]) -> str:
    base = "ct" if kind == "CT" else "mr"
    if base not in used:
        return base
    suffix = number if number is not None else len(used) + 1
    candidate = f"{base}_{suffix}"
    while candidate in used:
        suffix = f"{suffix}x"
        candidate = f"{base}_{suffix}"
    return candidate


REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_TEMPLATES = {"v1": Path("viewer") / "template.html", "v2": Path("viewer2") / "template.html"}
# One base hue per bundle family (cst, af, ...); the right hemisphere gets a lighter tint of the same hue, so
# no two tracts in a set share a colour and a left/right pair still reads as one bundle.
_TRACT_COLORS = ["#E4572E", "#4C9BE8", "#59C36A", "#C9A84C", "#B06AD9", "#3CC8C8",
                 "#E8438F", "#F2E14E", "#6A6FF0", "#A8D63A"]
_LEFT, _RIGHT = {"l", "left", "lh"}, {"r", "right", "rh"}


def _tract_family(label: str) -> tuple[str, str | None]:
    tokens = [t for t in re.split(r"[_\-\s.]+", label.lower()) if t]
    side = next(("l" if t in _LEFT else "r" for t in tokens if t in _LEFT | _RIGHT), None)
    return "_".join(t for t in tokens if t not in _LEFT | _RIGHT) or label.lower(), side


def _family_base(index: int) -> str:
    if index < len(_TRACT_COLORS):
        return _TRACT_COLORS[index]
    hue = (index * 137.508) % 360  # golden angle: later families never land on an earlier hue
    r, g, b = (round(255 * c) for c in colorsys.hls_to_rgb(hue / 360, 0.6, 0.7))
    return f"#{r:02X}{g:02X}{b:02X}"


def _tint(color: str, amount: float = 0.5) -> str:
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02X}" for c in rgb)


def default_tract_colors(labels: list[str]) -> list[str]:
    """Distinct default colours: family hue for left/unsided, lighter tint for right; duplicates are spread."""
    families: dict[str, int] = {}
    colors: list[str] = []
    for label in labels:
        family, side = _tract_family(label)
        base = _family_base(families.setdefault(family, len(families)))
        color = _tint(base) if side == "r" else base
        while color in colors:  # e.g. two tracts with the same label
            families[f"{family}#{len(families)}"] = len(families)
            color = _family_base(len(families) - 1)
        colors.append(color)
    return colors
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
_POSTCONTRAST_MR_LABEL = re.compile(r"(\+C\b|C\+|T1C|GAD|POS|CONTR|POST)", re.IGNORECASE)
_DEFAULT_LESION_LABEL = "Lesão (ROI de rastreamento)"


def _template(viewer: str) -> Path:
    template = REPO_ROOT / VIEWER_TEMPLATES[viewer]
    if not template.is_file():
        raise FileNotFoundError(f"--viewer {viewer}: template {VIEWER_TEMPLATES[viewer]} not found in the repository; "
                                "build that viewer first or use --viewer v1")
    return template


def _check_output(output: str) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists (never overwritten): {destination}")
    return destination


def _parse_tract(spec: str) -> tuple[Path, str, str | None]:
    color = None
    head, sep, tail = spec.rpartition(":")
    if sep and _HEX.fullmatch(tail):
        spec, color = head, tail.upper()
    path, sep, label = spec.rpartition(":")
    if not sep or not path or not label.strip():
        raise ValueError("--tract must be PATH:LABEL[:#RRGGBB]")
    return Path(path), label.strip(), color


def _parse_lesion_mask(spec: str) -> tuple[Path, str]:
    path, separator, label = spec.rpartition(":")
    if separator:
        if not path or not label.strip():
            raise ValueError("--lesion-mask must be PATH[:LABEL]")
        return Path(path), label.strip()
    if not spec:
        raise ValueError("--lesion-mask must be PATH[:LABEL]")
    return Path(spec), _DEFAULT_LESION_LABEL


def _parse_brain_mask_file(spec: str) -> tuple[Path, str]:
    path, separator, method = spec.rpartition(":")
    if separator and method in {"synthstrip", "bet"} and path:
        return Path(path), method
    if separator and method.strip() and "/" not in method:
        raise ValueError("--brain-mask-file must be PATH[:synthstrip|bet]")
    if not spec:
        raise ValueError("--brain-mask-file must be PATH[:synthstrip|bet]")
    return Path(spec), "synthstrip"


def _parse_brain_volume_gate(value: str) -> tuple[float, float] | None:
    if value.strip().lower() == "off":
        return None
    try:
        lower, upper = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--brain-volume-gate must be MIN,MAX or off") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or not (0 < lower <= upper):
        raise argparse.ArgumentTypeError("--brain-volume-gate must be a positive MIN,MAX range or off")
    return lower, upper


def _parse_volume(spec: str) -> tuple[Path, str, str]:
    parts = spec.rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[2].strip():
        raise ValueError("--volume must be PATH:KIND:LABEL")
    kind = parts[1].strip().upper()
    if kind not in {"CT", "MR"}:
        raise ValueError("--volume KIND must be CT or MR")
    return Path(parts[0]), kind, parts[2].strip()


def _parse_crop_ras(value: str) -> list[float]:
    try:
        coords = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--crop-ras must be x0,y0,z0,x1,y1,z1 in RAS+ mm") from exc
    if len(coords) != 6 or not all(math.isfinite(coord) for coord in coords):
        raise ValueError("--crop-ras must contain six finite coordinates in RAS+ mm")
    if any(coords[index] > coords[index + 3] for index in range(3)):
        raise ValueError("--crop-ras lower coordinates must not exceed upper coordinates")
    return coords


def _crop_mask_spec(value: str) -> tuple[Path, str]:
    path, separator, mask_id = value.rpartition(":")
    if not separator or not path or not mask_id.strip():
        raise ValueError("--crop-mask must be PATH:ID")
    return Path(path), mask_id.strip()


def _crop_request(args: argparse.Namespace) -> tuple[list[float] | None, str | None, float]:
    crop_around = getattr(args, "crop_around", None)
    raw_ras = getattr(args, "crop_ras", None)
    margin = getattr(args, "margin_mm", None)
    crop_mask = getattr(args, "crop_mask", None)
    if crop_around and raw_ras:
        raise ValueError("--crop-ras and --crop-around are mutually exclusive")
    if margin is not None and not crop_around:
        raise ValueError("--margin-mm requires --crop-around")
    if crop_mask and not crop_around:
        raise ValueError("--crop-mask requires --crop-around")
    if raw_ras:
        return _parse_crop_ras(raw_ras), "ras", 0.0
    if crop_around:
        if margin is None:
            raise ValueError("--crop-around requires --margin-mm")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("--margin-mm must be a finite non-negative value")
        return None, f"mask:{crop_around}", float(margin)
    return None, None, 0.0


def _registration_for(reference: sitk.Image, image: sitk.Image, index: int, reference_index: int,
                      no_register: bool):
    if index == reference_index:
        return sitk.Transform(3, sitk.sitkIdentity), {"reference": True}
    if no_register:
        return sitk.Transform(3, sitk.sitkIdentity), {"reference": False, "disabled": True}
    from .register import moving_to_fixed_ras, register
    transform, metric, metadata = register(reference, image)
    return transform, {"reference": False, "metric_value": float(metric),
                       "transform_parameters": metadata,
                       "moving_to_reference_ras": moving_to_fixed_ras(transform)}


def _resampled_flag(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> bool:
    """True when source voxels must be interpolated onto a different grid lattice."""
    source_spacing = np.asarray(image.GetSpacing(), dtype=float)
    grid_spacing = np.asarray(grid.GetSpacing(), dtype=float)
    if not np.allclose(source_spacing, grid_spacing, atol=1e-3, rtol=0):
        return True
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=float)
    transform_matrix = np.column_stack([
        np.asarray(transform.TransformPoint(tuple(np.eye(3)[axis])), dtype=float) - origin
        for axis in range(3)
    ])
    source_direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    grid_direction = np.asarray(grid.GetDirection(), dtype=float).reshape(3, 3)
    index_mapping = (np.diag(1.0 / source_spacing) @ source_direction.T @ transform_matrix
                     @ grid_direction @ np.diag(grid_spacing))
    if not np.allclose(index_mapping, np.eye(3), atol=1e-3, rtol=0):
        return True
    source_origin = transform.TransformPoint(grid.GetOrigin())
    source_index = image.TransformPhysicalPointToContinuousIndex(source_origin)
    return not np.allclose(source_index, np.rint(source_index), atol=1e-3, rtol=0)


_PROVENANCE_ALIASES = {
    "algorithm": "algorithm", "seeding": "seeding", "seed": "seeding", "select": "select",
    "step": "step", "stepmm": "step", "angle": "angle", "maxangle": "angle",
    "cutoff": "cutoff", "minlength": "min_length", "maxlength": "max_length",
    "act": "act", "sift2": "sift2", "prior": "prior", "template": "template",
    "software": "software", "softwareversion": "version", "version": "version",
    "mrtrixversion": "version", "bvalues": "b_values", "bvalue": "b_values",
    "directions": "directions", "ndirections": "directions", "voxelsize": "voxel_size",
    "voxelspacing": "voxel_size", "pipeline": "pipeline", "profile": "profile", "seeds": "seeds",
}
_TRACKING_FLAGS = {
    "algorithm": "algorithm", "seed_dynamic": "seeding", "seed_gmwmi": "seeding",
    "seed_image": "seeding", "seed_rejection": "seeding", "select": "select",
    "step": "step", "angle": "angle", "cutoff": "cutoff", "minlength": "min_length",
    "maxlength": "max_length", "act": "act", "sift2": "sift2", "prior": "prior",
    "template": "template", "bvalue": "b_values", "directions": "directions", "vox": "voxel_size",
}


def _safe_provenance_scalar(key: str, value):
    """Keep allowlisted scalar values while excluding dates, paths, and file-like strings."""
    canonical = re.sub(r"[^a-z0-9]", "", str(key).lower())
    field = _PROVENANCE_ALIASES.get(canonical)
    if not field or not isinstance(value, (str, int, float, bool)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or len(value) > 80 or "/" in value or "\\" in value:
            return None
        if re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", value):
            return None
        if re.search(r"\.(?:nii(?:\.gz)?|mif|tck|txt|json|bval|bvec)$", value, re.I):
            return None
        if not re.fullmatch(r"[A-Za-z0-9_.+ -]+", value):
            return None
    return field, value


def _parameter_provenance(source_bytes: bytes, tract_path: Path) -> dict:
    fields = {}

    def add(key, value):
        item = _safe_provenance_scalar(key, value)
        if item:
            fields.setdefault(*item)

    try:
        header = source_bytes.split(b"END\n", 1)[0].decode("latin-1", "replace")
    except UnicodeDecodeError:
        header = ""
    for line in header.splitlines()[1:]:
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized == "commandhistory":
            try:
                tokens = shlex.split(value)
            except ValueError:
                continue
            if tokens and re.fullmatch(r"tck[a-z0-9_]+", tokens[0], re.I):
                fields.setdefault("software", "MRtrix")
            for index, token in enumerate(tokens):
                flag = token.lstrip("-").lower()
                target = _TRACKING_FLAGS.get(flag)
                if not target:
                    continue
                if flag in {"act", "sift2"}:
                    fields.setdefault(target, True)
                    continue
                if flag in {"seed_dynamic", "seed_gmwmi", "seed_image", "seed_rejection"}:
                    fields.setdefault(target, flag.removeprefix("seed_"))
                    continue
                if index + 1 < len(tokens):
                    add(target, tokens[index + 1])
        else:
            add(key, value)

    # TractLab may store a shared scalar parameter record next to its bank.
    # SIFT2 text files are per-streamline weights, not tracking parameters.
    for sibling in tract_path.parent.iterdir():
        if not sibling.is_file() or sibling == tract_path or ".sift2." in sibling.name.lower():
            continue
        if sibling.suffix.lower() not in {".json", ".log", ".txt", ".cfg", ".conf", ".ini"}:
            continue
        try:
            text = sibling.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            data = None

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    add(key, item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        if data is not None:
            visit(data)
        else:
            for line in text.splitlines():
                key, sep, value = line.partition(":")
                if not sep:
                    key, sep, value = line.partition("=")
                if sep:
                    add(key.strip(), value.strip())
    return {"recorded": bool(fields), **fields} if fields else {"recorded": False}


def _load_tracts(specs: list[str], max_streamlines: int, step_mm: float = 0.0, *,
                 tract_max_length_mm: float = 250.0, lesion: LesionGrid | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Read, QC, subsample and re-encode tracts; world coordinates are never transformed."""
    entries, blobs = [], {}
    parsed = [_parse_tract(spec) for spec in specs]
    labels = [label for _, label, _ in parsed]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError("tract labels must be unique for trust QC: " + ", ".join(duplicates))
    defaults = default_tract_colors([label for _, label, _ in parsed])
    sources = []
    for index, (path, _, _) in enumerate(parsed, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"tract file not found: {path}")
        source_bytes = path.read_bytes()
        source = decode_tck(source_bytes)
        if not source:
            raise ValueError(f"tract {index} contains no streamlines")
        sources.append((source_bytes, source))
    trust = compute_tract_trust({label: source for (_, label, _), (_, source) in zip(parsed, sources)},
                                tract_max_length_mm=tract_max_length_mm, lesion=lesion)
    for index, ((path, label, color), (source_bytes, source)) in enumerate(zip(parsed, sources), start=1):
        # Short fragments and strays far from the bundle's mean course are dropped before subsampling.
        keep = outlier_mask(source)
        cleaned = [streamline for streamline, ok in zip(source, keep) if ok]
        outliers = [streamline for streamline, ok in zip(source, keep) if not ok]
        subset = subsample(cleaned, max_streamlines, seed=index)
        kept = [decimate(streamline, step_mm) for streamline in subset]
        tract_id = f"t{index:02d}"
        blob_id = f"tract_{tract_id}"
        kept_bytes = encode_tck(kept)
        blobs[blob_id] = np.frombuffer(kept_bytes, dtype=np.uint8)
        entry = {"id": tract_id, "blob": blob_id, "blob_sha256": hashlib.sha256(kept_bytes).hexdigest(),
                 "label": label,
                 "color": color or defaults[index - 1],
                 "format": "tck", "n_streamlines": len(kept), "n_streamlines_source": len(source),
                 "n_streamlines_outliers": len(outliers),
                 "outlier_filter": "length>=0.5*median & meancurve-distance<=median+3*MAD",
                 "source": "tractlab", "reviewed": False, "step_mm": step_mm,
                 "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                 "provenance": _parameter_provenance(source_bytes, path),
                 "trust": trust[label],
                 "n_points": int(sum(len(s) for s in kept)),
                 "n_points_before_step": int(sum(len(s) for s in subset))}
        if outliers:
            ratio = len(subset) / len(cleaned) if cleaned else 0.0
            outlier_count = min(len(outliers), max(1, int(round(len(outliers) * ratio))))
            if outlier_count:
                outlier_subset = subsample(outliers, outlier_count, seed=index)
                dropped_bytes = encode_tck([decimate(streamline, step_mm) for streamline in outlier_subset])
                dropped_blob = f"tract_outlier_{tract_id}"
                blobs[dropped_blob] = np.frombuffer(dropped_bytes, dtype=np.uint8)
                entry.update({"outlier_blob": dropped_blob,
                              "outlier_blob_sha256": hashlib.sha256(dropped_bytes).hexdigest(),
                              "n_streamlines_outlier_blob": outlier_count})
        entries.append(entry)
    return entries, blobs


def _points_inside_grid(manifest: dict, blob: np.ndarray) -> float:
    points = np.concatenate(decode_tck(blob.tobytes()))
    inverse = np.linalg.inv(np.asarray(manifest["grid"]["affine_ras"], dtype=float))
    ijk = points.astype(float) @ inverse[:3, :3].T + inverse[:3, 3]
    dims = np.asarray(manifest["grid"]["dims"], dtype=float)
    return float(np.mean(np.all((ijk >= -0.5) & (ijk <= dims - 0.5), axis=1)))


_RENDER_STYLE = {"head": ("Cabeça (render 3D)", "#E8DCC4"), "brain": ("Cérebro (render 3D)", "#D9A6A0"),
                "vessels": ("Vasos (render 3D)", "#3A60D6"), "cta": ("Vasos da angio-TC (render 3D)", "#D8433A")}


VESSEL_MIN_VOXELS = 2000  # below this the volume shows no enhancing vessel tree worth a preset
VESSEL_MAX_BRAIN_FRACTION = 0.05


def vessels_plausible(n_vessels: int, n_brain: int) -> bool:
    """On a T2 the bright sulcal CSF is tubular too: a 'vessel' mask above a few % of the brain is CSF, not vessels."""
    return VESSEL_MIN_VOXELS <= n_vessels <= VESSEL_MAX_BRAIN_FRACTION * n_brain


CTA_MIN_ML = 2.0  # a head CTA subtraction tree is ~25 mL; bone-edge residue of a failed pairing is < 1 mL
CTA_MAX_HEAD_FRACTION = 0.03  # unmasked bone-edge residue covers ~8 % of the head


def cta_plausible(volume_ml: float, n_vessels: int, n_head: int) -> bool:
    return volume_ml >= CTA_MIN_ML and n_vessels <= CTA_MAX_HEAD_FRACTION * n_head


def _cta_masks(manifest, arrays, used, grid, spacing, cts: list[tuple[str, np.ndarray, np.ndarray]]):
    """Each CT is tried as the angiogram against every other selected CT as its non-contrast pair; the pair
    with the largest plausible tree wins. A venous phase with little residual contrast yields none."""
    if len(cts) < 2:
        return
    for volume_id, values, head in cts:
        best = None
        for other_id, other, _ in cts:
            if other_id == volume_id:
                continue
            vessels, record = maskops.cta_vessel_mask(values, other, head, grid)
            ml = float(vessels.sum()) * spacing ** 3 / 1000.0
            if cta_plausible(ml, int(vessels.sum()), int(head.sum())) and (best is None or ml > best[0]):
                best = (ml, vessels, {**record, "subtracted": other_id})
        if best is None:
            print(f"angio-CT vessels on {volume_id}: none (no selected CT gives a plausible subtraction tree)")
            continue
        ml, vessels, record = best
        label, color = _RENDER_STYLE["cta"]
        _add_mask(manifest, arrays, used, spacing, "vessels", volume_id, vessels, label=label, color=color,
                  source="auto", reviewed=False, role="render", computed_on=volume_id, **record)
        print(f"angio-CT vessels on {volume_id} minus {record['subtracted']}: {ml:.1f} mL, "
              f"{record['components']} components, largest {record['largest_component_fraction']:.0%}, "
              f"{record['ml_within_2mm_of_bone']:.2f} mL within 2 mm of bone")


def _mask_id(name: str, used: set[str], volume_id: str) -> str:
    if name not in used:
        return name
    base = f"{name}_{volume_id}"
    candidate, suffix = base, 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _add_mask(manifest: dict, arrays: dict, used: set[str], grid_spacing: float, name: str, volume_id: str,
              mask: np.ndarray, **fields) -> dict:
    mask_id = _mask_id(name, used, volume_id)
    used.add(mask_id)
    blob = f"mask_{mask_id}"
    arrays[blob] = mask.astype(np.uint8)
    entry = {"id": mask_id, "blob": blob, "dtype": "uint8", "for_volume": volume_id,
             "volume_ml": round(float(mask.sum()) * grid_spacing ** 3 / 1000.0, 2),
             "blob_sha256": hashlib.sha256(arrays[blob].tobytes(order="C")).hexdigest(), **fields}
    manifest["masks"].append(entry)
    return entry


def _pack_external_lesion(args, manifest: dict, arrays: dict, used: set[str], grid: sitk.Image,
                          spacing: float, resampled: list[tuple[str, str, np.ndarray, bool]]) -> LesionGrid | None:
    parsed = getattr(args, "_lesion_mask", None)
    if parsed is None and getattr(args, "lesion_mask", None):
        parsed = _parse_lesion_mask(args.lesion_mask)
    if parsed is None:
        return None
    path, label = parsed
    if not path.is_file():
        raise FileNotFoundError(f"lesion mask not found: {path}")
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError("--lesion-mask must be a 3D image")
    binary = sitk.Cast(image > 0, sitk.sitkUInt8)
    on_grid = sitk.Resample(binary, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask = sitk.GetArrayFromImage(on_grid) > 0
    reference_id = next((volume_id for volume_id, _, _, reference in resampled if reference), resampled[0][0])
    _add_mask(manifest, arrays, used, spacing, "lesion", reference_id, mask, label=label, color="#E4572E",
              source="pipeline", reviewed=False, role="lesion")
    return LesionGrid(mask_kji=mask, affine_ras=np.asarray(affine_ras(grid), dtype=float))


def _select_brain(args, resampled, grid, manifest: dict | None = None):
    mr = [item for item in resampled if item[1] == "MR"]
    if not mr or args.brain_mask == "none":
        return None
    source_id, _, values, _ = next((item for item in mr if item[3]), mr[0])
    command_runner = getattr(args, "_command_runner", None)
    gate = getattr(args, "brain_volume_gate", maskops.DEFAULT_BRAIN_VOLUME_GATE)
    kwargs = {"command_runner": command_runner} if command_runner is not None else {}
    mask_file = getattr(args, "_brain_mask_file", None)
    try:
        if mask_file is not None:
            brain, record = maskops.brain_mask_from_file(mask_file[0], grid, mask_file[1], volume_gate=gate)
        else:
            try:
                brain, record = maskops.brain_mask(values, grid, args.brain_mask, volume_gate=gate, **kwargs)
            except TypeError as error:
                if "volume_gate" not in str(error):
                    raise
                brain, record = maskops.brain_mask(values, grid, args.brain_mask, **kwargs)
    except maskops.BrainMaskQCError as error:
        if manifest is not None:
            manifest["brain_mask_qc"] = error.qc
        print(f"brain mask: QC failed on {source_id}")
        return None
    qc = record.get("qc")
    if qc is not None and qc["verdict"] == "FAIL":
        if manifest is not None:
            manifest["brain_mask_qc"] = qc
        print(f"brain mask: QC failed on {source_id}")
        return None
    print(f"brain mask: {record['method']} on {source_id} in {record.get('seconds', 0)} s")
    brain_record = {"method": record["method"], "computed_on": source_id}
    if qc is not None:
        brain_record["qc"] = qc
    return brain, brain_record


def _render_masks(args, manifest, arrays, used, grid, spacing,
                  resampled: list[tuple[str, str, np.ndarray, bool]], brain_result=None, face_region=None):
    """Display-only masks for the v2 viewer: CT head; MR brain (reference MR, else first MR) + head per MR."""
    mr = [item for item in resampled if item[1] == "MR"]
    if brain_result is None:
        brain_result = _select_brain(args, resampled, grid, manifest)
    brain, brain_record = brain_result if brain_result is not None else (None, {})
    volume_meta = {item["id"]: item for item in manifest["volumes"]}
    cts = []
    for volume_id, kind, values, _ in resampled:
        head = maskops.ct_head(values, grid) if kind == "CT" else maskops.mr_head(values, grid)
        head_fields = {}
        if brain is not None:
            brain_count = int(brain.sum())
            outside_fraction = float(np.count_nonzero(brain & ~head) / brain_count) if brain_count else 0.0
            head_fields["brain_outside_head_fraction_before"] = outside_fraction
            # Skull-stripped = no signal outside the brain. A small head mask alone is not evidence: log-Otsu
            # can miss most of a real T2 head (VS demo: 63 % of the brain outside it) while the scalp is there.
            if kind == "MR":
                outside_signal = maskops.signal_outside_brain_fraction(values, brain, grid)
                head_fields["signal_outside_brain_fraction"] = round(outside_signal, 4)
                if outside_signal < 0.05:
                    volume_meta[volume_id]["skull_stripped_input"] = True
            # Preserve the brain in the render mask even when log-Otsu contracts on a
            # brain-extracted MR. Filling only closes enclosed holes in this union.
            head = maskops.fill_enclosed(head | brain, grid)
        if face_region is not None:
            head &= ~face_region
        if kind == "CT":
            cts.append((volume_id, values, head))
        label, color = _RENDER_STYLE["head"]
        _add_mask(manifest, arrays, used, spacing, "head", volume_id, head, label=label, color=color,
                  source="auto", reviewed=False, role="render",
                  method="ct>-500HU+open3mm+lcc+fill" if kind == "CT" else "log-otsu+open2mm+lcc+fill",
                  **head_fields)
        if kind == "MR" and brain is not None:
            label, color = _RENDER_STYLE["brain"]
            _add_mask(manifest, arrays, used, spacing, "brain", volume_id, brain, label=label, color=color,
                      source="auto", reviewed=False, role="render", **brain_record)
            if _POSTCONTRAST_MR_LABEL.search(volume_meta[volume_id].get("label", "")):
                vessels, record = maskops.vessel_mask(values, brain, head, grid)
                vessels &= brain
                kept = vessels_plausible(int(vessels.sum()), int(brain.sum()))
                if kept:
                    label, color = _RENDER_STYLE["vessels"]
                    _add_mask(manifest, arrays, used, spacing, "vessels", volume_id, vessels, label=label, color=color,
                              source="auto", reviewed=False, role="render", computed_on=volume_id, **record)
                print(f"vessel mask on {volume_id}: {int(vessels.sum())} voxels, {record['components']} components"
                      f"{'' if kept else ' (not kept: outside the plausible vessel range)'}")
    _cta_masks(manifest, arrays, used, grid, spacing, cts)


def _write(args: argparse.Namespace, destination: Path, grid: sitk.Image, spacing: float,
           volumes: list[dict], study_year: int | None, *, grid_plan: GridPlan | None = None) -> dict:
    """volumes: dicts with image, transform, kind, label, series, registration."""
    if grid_plan is None:
        grid_plan = GridPlan(image=grid, requested_spacing_mm=spacing, spacing_mm=spacing,
                             spacing_raised=False, crop=None)
    grid, spacing = grid_plan.image, grid_plan.spacing_mm
    face_removed = bool(getattr(args, "deface", False))
    if face_removed and (getattr(args, "brain_mask", "none") == "none"
                         or not any(volume["kind"] == "MR" for volume in volumes)):
        raise ValueError("--deface requires an available brain mask from an MR volume; select --brain-mask synthstrip or bet")
    template = _template(args.viewer)
    manifest = {
        "schema": "case-capsule/1", "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generator": "case-capsule 0.1.0",
        "case": {"label": args.label, "anonymized": face_removed, "study_year": study_year,
                 "deidentification": {"identifiers_removed": True, "face_removed": face_removed,
                                      "method": "quickshear-hull-buffer-5mm" if face_removed else None}},
        "locale": "pt-BR",
        "grid": {"dims": list(grid.GetSize()), "spacing_mm": [spacing] * 3,
                 "affine_ras": affine_ras(grid),
                 "requested_spacing_mm": grid_plan.requested_spacing_mm,
                 "spacing_raised": grid_plan.spacing_raised},
        "volumes": [], "masks": [], "anatomy": [], "tracts": [], "annotations": [], "tour": [],
    }
    if grid_plan.crop is not None:
        manifest["grid"]["crop"] = grid_plan.crop
    arrays: dict[str, np.ndarray] = {}
    used_ids: set[str] = set()
    used_masks: set[str] = set()
    resampled = []
    for volume in volumes:
        kind = volume["kind"]
        raw = resample_to_grid(volume["image"], grid, volume["transform"], kind)
        volume_id = _volume_id(kind, volume["series"].get("number"), used_ids)
        used_ids.add(volume_id)
        volume["id"] = volume_id
        resampled.append((volume_id, kind, raw, volume["registration"].get("reference", False)))

    brain_result, face_region = None, None
    if face_removed:
        brain_result = _select_brain(args, resampled, grid, manifest)
        if brain_result is None:
            raise ValueError("--deface requires an available brain mask; the selected MR produced none")
        face_region = defaceops.face_removal_mask(brain_result[0], grid)
        resampled = [(volume_id, kind, defaceops.apply_deface(raw, face_region, kind), is_reference)
                     for volume_id, kind, raw, is_reference in resampled]

    for volume, (volume_id, kind, raw, _) in zip(volumes, resampled):
        for lesion in volume.get("lesions", []):
            on_grid = maskops.to_grid(lesion["native"], volume["image"], grid, volume["transform"])
            _add_mask(manifest, arrays, used_masks, spacing, lesion["id"], volume_id, on_grid,
                      label=lesion["label"], color=lesion["color"], source="dataset", reviewed=False,
                      role="lesion", roi=lesion["roi"], planar_volume_ml=lesion["planar_volume_ml"],
                      native_volume_ml=lesion["native_volume_ml"])
        packed, slope, intercept = pack_scalar(raw, kind)
        arrays[volume_id] = packed
        stats = scalar_stats(raw)
        if kind == "CT":
            presets = [{"name": name, "center": center, "width": width} for name, center, width in _CT_WINDOWS]
        else:
            p01, p99 = stats["p01"], stats["p99"]
            presets = [{"name": "Auto", "center": (p01 + p99) / 2, "width": max(p99 - p01, 1)}]
        manifest["volumes"].append({
            "id": volume_id, "blob": volume_id, "kind": kind, "label": volume["label"],
            "dtype": str(packed.dtype), "slope": slope, "intercept": intercept,
            "source_spacing_mm": [float(value) for value in volume["image"].GetSpacing()],
            "resampled": _resampled_flag(volume["image"], grid, volume["transform"]),
            "units": "HU" if kind == "CT" else "a.u.",
            "series": volume["series"], "stats": stats, "window_presets": presets,
            "registration": volume["registration"],
        })
    lesion = _pack_external_lesion(args, manifest, arrays, used_masks, grid, spacing, resampled)
    tracts, tract_blobs = _load_tracts(
        args.tract or [], args.max_streamlines, args.tract_step_mm,
        tract_max_length_mm=getattr(args, "tract_max_length_mm", 250.0), lesion=lesion)
    manifest["tracts"] = tracts
    if args.viewer == "v2":
        # Render masks are a v2 contract; the v1 viewer would draw them as lesions.
        _render_masks(args, manifest, arrays, used_masks, grid, spacing, resampled, brain_result, face_region)
    anatomy_mode = getattr(args, "anatomy", "none")
    if anatomy_mode == "auto":
        if args.viewer != "v2":
            raise ValueError("--anatomy auto requires --viewer v2")
        items, anatomy_arrays, blocked, records = anatomyops.produce_anatomy(volumes, grid)
        manifest["anatomy"].extend(items)
        if face_region is not None:
            for item in items:
                anatomy_arrays[item["blob"]][face_region] = 0
        for item in items:
            item["blob_sha256"] = hashlib.sha256(anatomy_arrays[item["blob"]].tobytes(order="C")).hexdigest()
        arrays.update(anatomy_arrays)
        for line in blocked:
            print(line)
        for item in items:
            record = records[item["id"]]
            print(f"anatomy {item['id']}: {record} ({len(item['labels'])} labels)")
    for tract in tracts:
        tract["fraction_points_inside_grid"] = round(_points_inside_grid(manifest, tract_blobs[tract["blob"]]), 4)
    arrays.update(tract_blobs)
    write_capsule(template, destination, manifest, arrays)
    return manifest


def _dicom_labels(selected: list[Series]) -> list[str]:
    """Surgeon-facing pt-BR labels: allowlisted description or a fixed weighting token; always unique."""
    labels = []
    for series in selected:
        prefix = "TC" if series.modality == "CT" else "RM"
        if series.description != "(omitted)":
            label = series.description
        elif series.modality == "MR" and series.weighting:
            label = f"RM {series.weighting}"
        else:
            label = prefix
        labels.append(label)
    counts = {label: labels.count(label) for label in labels}
    return [f"{label} (série {series.number})" if counts[label] > 1 else label
            for label, series in zip(labels, selected)]


def build(args: argparse.Namespace) -> tuple[Path, dict]:
    destination = _check_output(args.output)
    _template(args.viewer)
    if args.anatomy == "auto" and args.viewer != "v2":
        raise ValueError("--anatomy auto requires --viewer v2")
    all_series, _ = scan_series(args.dicom_dir)
    selected = _selected_series(all_series, args.series)
    reference_index = _reference_index(selected, args.reference)
    labels = _dicom_labels(selected)
    crop_ras, crop_source, crop_margin = _crop_request(args)
    with tempfile.TemporaryDirectory(prefix="case-capsule-") as temporary:
        work = Path(temporary)
        images = []
        frames_used = []
        for series in selected:
            image, frames = first_volume(sitk.ReadImage(str(convert_series(series, work))))
            images.append(image)
            frames_used.append(frames)

        lesions_by_index: dict[int, list[dict]] = {}
        if args.rtstruct:
            structure = maskops.read_rtstruct(args.rtstruct, args.rtstruct_roi)
            matches = [i for i, series in enumerate(selected) if series.uid in structure.referenced_series]
            if len(matches) != 1:
                raise ValueError("--rtstruct does not reference exactly one of the selected series")
            native_image = images[matches[0]]
            native = maskops.rasterize(structure, native_image)
            if not native.any():
                raise ValueError("--rtstruct contour rasterized to an empty mask")
            voxel_ml = float(np.prod(native_image.GetSpacing())) / 1000.0
            lesions_by_index[matches[0]] = [{
                "id": "tumour", "native": native, "roi": structure.name, "label": args.rtstruct_label,
                "color": "#E4572E", "native_volume_ml": round(float(native.sum()) * voxel_ml, 3),
                "planar_volume_ml": round(maskops.planar_volume_ml(structure, native_image), 3)}]

        reference = images[reference_index]
        precomputed = {}
        if getattr(args, "crop_around", None):
            mask_id = args.crop_around
            crop_index = next((index for index, lesions in lesions_by_index.items()
                               if any(lesion["id"] == mask_id for lesion in lesions)), None)
            if crop_index is None:
                raise ValueError(f"--crop-around {mask_id!r} requires that lesion in the selected RTSTRUCT")
            transform, registration = _registration_for(
                reference, images[crop_index], crop_index, reference_index, args.no_register)
            precomputed[crop_index] = (transform, registration)
            lesion = next(lesion for lesion in lesions_by_index[crop_index] if lesion["id"] == mask_id)
            mask_on_reference = maskops.to_grid(lesion["native"], images[crop_index], reference, transform)
            crop_ras = mask_ras_bounds(mask_on_reference, reference, crop_margin)
        grid_plan = common_grid(reference, selected[reference_index].modality, args.spacing,
                                max_voxels=getattr(args, "max_voxels", DEFAULT_MAX_VOXELS),
                                crop_ras=crop_ras, crop_source=crop_source,
                                crop_margin_mm=crop_margin)
        years = {series.year for series in selected if series.year is not None}
        volumes = []
        for index, (series, image) in enumerate(zip(selected, images)):
            transform, registration = precomputed.get(index) or _registration_for(
                reference, image, index, reference_index, args.no_register)
            volumes.append({"image": image, "transform": transform, "kind": series.modality,
                            "label": labels[index], "registration": registration,
                            "lesions": lesions_by_index.get(index, []),
                            "series": {"number": series.number, "description": series.description,
                                       "modality": series.modality, "frames_used": frames_used[index]}})
        manifest = _write(args, destination, grid_plan.image, grid_plan.spacing_mm, volumes,
                          next(iter(years)) if len(years) == 1 else None, grid_plan=grid_plan)
    return destination, manifest


def build_nifti(args: argparse.Namespace) -> tuple[Path, dict]:
    """NIfTI volumes assumed to share one world space: no registration, no header text copied."""
    destination = _check_output(args.output)
    _template(args.viewer)
    specs = [_parse_volume(spec) for spec in args.volume]
    registration_methods = {}
    for spec in getattr(args, "volume_registration", ()):
        try:
            label, method = spec.split(":", 1)
        except ValueError as exc:
            raise ValueError("--volume-registration must be LABEL:METHOD") from exc
        if not label or not method or label in registration_methods:
            raise ValueError("--volume-registration requires unique labels and non-empty methods")
        registration_methods[label] = method
    unknown_labels = registration_methods.keys() - {label for _, _, label in specs}
    if unknown_labels:
        raise ValueError("--volume-registration label not found in --volume: "
                         + ", ".join(sorted(unknown_labels)))
    crop_ras, crop_source, crop_margin = _crop_request(args)
    volumes = []
    for index, (path, kind, label) in enumerate(specs):
        if not path.is_file():
            raise FileNotFoundError(f"volume file not found: {path}")
        image, frames = first_volume(sitk.ReadImage(str(path)))
        # Keep only voxels + geometry; drop every header metadata key (descrip, aux_file, intent_name, ...).
        clean = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
        clean.CopyInformation(image)
        volumes.append({"image": clean, "transform": sitk.Transform(3, sitk.sitkIdentity), "kind": kind,
                        "label": label,
                        "registration": {"reference": index == 0,
                                         "method": registration_methods.get(label, "none-shared-world")},
                        "series": {"number": None, "description": "(nifti)", "modality": kind,
                                   "frames_used": frames}})
    reference = volumes[0]["image"]
    if getattr(args, "crop_around", None):
        crop_mask_spec = getattr(args, "crop_mask", None)
        if not crop_mask_spec:
            raise ValueError("build-nifti --crop-around requires --crop-mask PATH:ID")
        mask_path, mask_id = _crop_mask_spec(crop_mask_spec)
        if mask_id != args.crop_around:
            raise ValueError("--crop-mask ID must match --crop-around ID")
        if not mask_path.is_file():
            raise FileNotFoundError(f"crop mask not found: {mask_path}")
        mask_image = sitk.ReadImage(str(mask_path))
        if mask_image.GetDimension() != 3:
            raise ValueError("--crop-mask must be a 3D image")
        mask_image = sitk.Cast(mask_image > 0, sitk.sitkUInt8)
        mask_on_reference = sitk.Resample(mask_image, reference, sitk.Transform(3, sitk.sitkIdentity),
                                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        crop_mask = sitk.GetArrayFromImage(mask_on_reference) > 0
        crop_ras = mask_ras_bounds(crop_mask, reference, crop_margin)
    grid_plan = common_grid(reference, volumes[0]["kind"], args.spacing,
                            max_voxels=getattr(args, "max_voxels", DEFAULT_MAX_VOXELS),
                            crop_ras=crop_ras, crop_source=crop_source,
                            crop_margin_mm=crop_margin)
    manifest = _write(args, destination, grid_plan.image, grid_plan.spacing_mm, volumes,
                      None, grid_plan=grid_plan)
    return destination, manifest


def main(argv: list[str] | None = None, *, command_runner=None) -> int:
    parser = argparse.ArgumentParser(prog="capsule")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="print de-identified DICOM series table")
    scan.add_argument("dicom_dir")
    make = sub.add_parser("build", help="create one offline HTML capsule")
    make.add_argument("dicom_dir")
    make.add_argument("--series", required=True)
    make.add_argument("--label", required=True)
    make.add_argument("-o", "--output", required=True)
    make.add_argument("--reference", type=int)
    make.add_argument("--spacing", type=float, default=1.0)
    make.add_argument("--max-voxels", type=int, default=DEFAULT_MAX_VOXELS,
                      help="maximum grid voxels per volume (default: 64 million)")
    make.add_argument("--crop-ras", metavar="X0,Y0,Z0,X1,Y1,Z1",
                      help="RAS+ mm crop box, intersected with the head foreground box")
    make.add_argument("--crop-around", metavar="MASK_ID",
                      help="crop around a selected RTSTRUCT lesion mask")
    make.add_argument("--margin-mm", type=float,
                      help="margin around --crop-around mask (required with --crop-around)")
    make.add_argument("--no-register", action="store_true")
    make.add_argument("--rtstruct", metavar="PATH", help="DICOM RTSTRUCT whose ROI becomes the lesion mask "
                      "(must reference one of the selected series)")
    make.add_argument("--rtstruct-roi", metavar="NAME", help="ROI name (default: the single non-helper ROI or AN/TV)")
    make.add_argument("--rtstruct-label", default="Schwannoma vestibular (dataset)")
    make.add_argument("--anatomy", choices=["auto", "none"],
                      default=os.environ.get("CAPSULE_ANATOMY", "auto"),
                      help="v2 automatic anatomy layers (default: $CAPSULE_ANATOMY or auto)")
    nifti = sub.add_parser("build-nifti", help="create a capsule from NIfTI volumes that share one world space")
    nifti.add_argument("--volume", action="append", required=True, metavar="PATH:KIND:LABEL",
                       help="KIND is CT or MR; the first volume is the reference grid")
    nifti.add_argument("--volume-registration", action="append", default=[], metavar="LABEL:METHOD",
                       help="record the registration method for a volume label")
    nifti.add_argument("--label", required=True)
    nifti.add_argument("-o", "--output", required=True)
    nifti.add_argument("--spacing", type=float, default=1.0)
    nifti.add_argument("--max-voxels", type=int, default=DEFAULT_MAX_VOXELS,
                       help="maximum grid voxels per volume (default: 64 million)")
    nifti.add_argument("--crop-ras", metavar="X0,Y0,Z0,X1,Y1,Z1",
                       help="RAS+ mm crop box, intersected with the head foreground box")
    nifti.add_argument("--crop-around", metavar="MASK_ID",
                       help="crop around a NIfTI mask identified by --crop-mask")
    nifti.add_argument("--crop-mask", metavar="PATH:ID",
                       help="3D NIfTI mask used by --crop-around; ID must match MASK_ID")
    nifti.add_argument("--margin-mm", type=float,
                       help="margin around --crop-around mask (required with --crop-around)")
    for command in (make, nifti):
        command.add_argument("--tract", action="append", metavar="PATH:LABEL[:#RRGGBB]",
                             help="MRtrix .tck in the same world RAS mm space as the volumes")
        command.add_argument("--lesion-mask", metavar="PATH[:LABEL]",
                             help="3D NIfTI lesion mask in the same world RAS space as the volumes")
        command.add_argument("--max-streamlines", type=int, default=1500)
        command.add_argument("--tract-step-mm", type=float, default=1.0,
                             help="keep source points about this far apart along each streamline "
                                  "(endpoints always kept; 0 keeps every point)")
        command.add_argument("--tract-max-length-mm", type=float, default=250.0,
                             help="tckgen maximum length used for tract trust QC (default: 250)")
        command.add_argument("--brain-volume-gate", type=_parse_brain_volume_gate,
                             default=maskops.DEFAULT_BRAIN_VOLUME_GATE, metavar="MIN,MAX|off",
                             help="acceptable brain-mask volume in mL (default: 900,1800; off disables the gate)")
        command.add_argument("--viewer", choices=sorted(VIEWER_TEMPLATES), default=None,
                             help="viewer (default v2; v1 is the legacy WebGL viewer)")
        command.add_argument("--brain-mask", choices=["synthstrip", "bet", "none"],
                             default=os.environ.get("CAPSULE_BRAIN_MASK", "synthstrip"),
                             help="v2 MR brain render mask (SynthStrip falls back to FSL BET)")
        command.add_argument("--brain-mask-file", metavar="PATH[:synthstrip|bet]",
                             help="precomputed MR brain mask in the volumes' world space; gated and packed "
                                  "instead of running --brain-mask (e.g. the capsule dwi ACT SynthStrip mask)")
        command.add_argument("--deface", action="store_true",
                             help="remove the face on the common grid (requires an MR brain mask; default off)")
    dwi = sub.add_parser("dwi", help="run the staged DWI tractography pipeline")
    dwiops.configure_parser(dwi)
    segment = sub.add_parser("segment", help="run nnInteractive point prompts in a new capsule")
    segment.add_argument("input", metavar="IN.capsule.html")
    segment.add_argument("-o", "--output", required=True, metavar="OUT.capsule.html")
    segment.add_argument("--device", choices=["mps", "cpu"], default="mps")
    segment.add_argument("--python", dest="python_path", default=None, metavar="PATH",
                         help=f"nnInteractive Python (default: $CASE_CAPSULE_NNINTERACTIVE_PYTHON or {DEFAULT_NNINTERACTIVE_PYTHON})")
    export = sub.add_parser("export", help="export reviewed masks as DICOM-SEG and RTSTRUCT")
    export.add_argument("--capsule", required=True, metavar="SAVED.capsule.html")
    export.add_argument("--dicom", required=True, metavar="SOURCE_DICOM_DIR")
    export.add_argument("--out", required=True, metavar="OUT_DIR",
                        help="new or empty output directory; never defaults to the home directory")
    export.add_argument("--series", type=int, metavar="N", help="target DICOM SeriesNumber")
    export.add_argument("--masks", metavar="ID1,ID2", help="comma-separated capsule mask IDs")
    export.add_argument("--include-unreviewed", action="store_true",
                        help="include unsigned masks and mark every exported label as NAO REVISADO")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    for index in range(len(raw_argv) - 1):
        if raw_argv[index] == "--crop-ras" and raw_argv[index + 1].startswith("-"):
            raw_argv[index] = f"--crop-ras={raw_argv[index + 1]}"
            del raw_argv[index + 1]
            break
    args = parser.parse_args(raw_argv)
    if command_runner is not None:
        args._command_runner = command_runner
    if args.command in ("build", "build-nifti") and args.viewer is None:
        args.viewer = "v2"
    try:
        if args.command == "scan":
            series, skipped = scan_series(args.dicom_dir)
            print("number | modality | description | sequence (ScanningSequence/SequenceName/contrast) | rows×cols×slices | pixel spacing mm | slice thickness mm | transfer syntax")
            for item in series:
                hints = f"{item.scanning_sequence}/{item.sequence_name}/{'yes' if item.contrast else 'no'}"
                print(f"{item.number} | {item.modality} | {item.description} | {hints} | "
                      f"{item.rows}×{item.columns}×{len(item.files)} | "
                      f"{item.pixel_spacing[0]:g}×{item.pixel_spacing[1]:g} | "
                      f"{item.slice_thickness if item.slice_thickness is not None else '?'} | {item.transfer_syntax}")
            print(f"{len(series)} series; {skipped} non-DICOM or unsupported files skipped")
        elif args.command == "dwi":
            return dwiops.run(args)
        elif args.command == "segment":
            destination, result = segment_capsule(args.input, args.output, python_path=args.python_path, device=args.device)
            print(f"{destination} ({destination.stat().st_size / 1_000_000:.2f} MB)")
            for mask_id in result["masks_added"]:
                print(f"mask {mask_id}: structure segmentation added")
            for prompt in result["manifest"].get("seg_prompts", []):
                if prompt.get("status") in {"done", "empty"}:
                    print(f"prompt {prompt.get('name', prompt.get('id'))}: {prompt['status']}")
        elif args.command == "export":
            from .export import export_capsule

            receipt = export_capsule(args.capsule, args.dicom, args.out, args.series,
                                     args.masks, args.include_unreviewed)
            for mask in receipt["masks"]:
                print(f"{mask['id']}: {mask['volume_ml_native']:.3f} mL")
        else:
            if args.max_streamlines < 1:
                raise ValueError("--max-streamlines must be at least 1")
            if not 0 <= args.tract_step_mm <= 5:
                raise ValueError("--tract-step-mm must be between 0 and 5 mm")
            if not math.isfinite(args.tract_max_length_mm) or args.tract_max_length_mm <= 0:
                raise ValueError("--tract-max-length-mm must be finite and positive")
            if getattr(args, "lesion_mask", None):
                lesion_path, lesion_label = _parse_lesion_mask(args.lesion_mask)
                if not lesion_path.is_file():
                    raise FileNotFoundError(f"lesion mask not found: {lesion_path}")
                args._lesion_mask = (lesion_path, lesion_label)
            if getattr(args, "brain_mask_file", None):
                if args.brain_mask == "none":
                    raise ValueError("--brain-mask-file cannot be combined with --brain-mask none")
                brain_path, brain_method = _parse_brain_mask_file(args.brain_mask_file)
                if not brain_path.is_file():
                    raise FileNotFoundError(f"brain mask file not found: {brain_path}")
                args._brain_mask_file = (brain_path, brain_method)
            destination, manifest = (build if args.command == "build" else build_nifti)(args)
            grid = manifest.get("grid")
            if grid is not None:
                if grid["spacing_raised"]:
                    print(f"grid spacing raised from {grid['requested_spacing_mm']:g} mm to "
                          f"{grid['spacing_mm'][0]:g} mm to fit --max-voxels "
                          f"({args.max_voxels:,} voxels per volume)")
                voxel_count = math.prod(grid["dims"])
                raw_bytes_per_volume = voxel_count * 4
                raw_bytes_total = raw_bytes_per_volume * len(manifest["volumes"])
                print(f"grid: dims {'×'.join(map(str, grid['dims']))}; voxel count {voxel_count:,}; "
                      f"estimated raw bytes {raw_bytes_per_volume:,} per volume "
                      f"({raw_bytes_total:,} total as float32)")
            size_mb = destination.stat().st_size / 1_000_000
            print(f"{destination} ({size_mb:.2f} MB)")
            for volume in manifest["volumes"]:
                print(f"{volume['id']}: {volume['kind']} {volume['stats']}")
            for mask in manifest["masks"]:
                print(f"mask {mask['id']}: {mask['role']} for {mask['for_volume']} {mask['volume_ml']} mL "
                      f"({mask['source']}{', ' + mask['method'] if 'method' in mask else ''})")
            for tract in manifest["tracts"]:
                print(f"{tract['id']}: tract {tract['label']!r} {tract['n_streamlines']}/{tract['n_streamlines_source']} "
                      f"streamlines, {tract['n_points']}/{tract['n_points_before_step']} points (step {tract['step_mm']} mm); "
                      f"{tract['fraction_points_inside_grid']:.1%} of points inside the grid; "
                      f"trust {tract['trust']['verdict']}")
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        parser.exit(1, f"capsule: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
