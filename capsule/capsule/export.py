"""Export reviewed capsule masks to DICOM-SEG and RTSTRUCT on a source series."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import highdicom as hd
from highdicom.color import CIELabColor
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg import SegmentAlgorithmTypeValues, SegmentDescription, Segmentation
from highdicom.sr import CodedConcept
import numpy as np
import pydicom
from pydicom.uid import generate_uid
import SimpleITK as sitk
from rt_utils import RTStructBuilder

from . import masks as maskops
from .ingest import Series, scan_series
from .pack import parse_capsule


_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
_DEFAULT_MASK_ROLES = {"lesion", "structure", "segmentation"}
_RENDER_IDS = {"head", "brain", "vessels"}
_REVIEW_MARKER = "NAO REVISADO"


def _version(name: str, fallback: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback


def _output_directory(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("output path must be a directory")
        try:
            nonempty = next(destination.iterdir(), None) is not None
        except OSError:
            raise ValueError("output directory cannot be inspected") from None
        if nonempty:
            raise ValueError("output directory is non-empty; choose a new --out")
    return destination


def _manifest_dimensions(manifest: dict) -> tuple[int, int, int]:
    if manifest.get("schema") != "case-capsule/1":
        raise ValueError("capsule schema must be case-capsule/1")
    try:
        dims = tuple(int(value) for value in manifest["grid"]["dims"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("capsule grid dimensions are invalid") from None
    if len(dims) != 3 or any(value <= 0 for value in dims):
        raise ValueError("capsule grid dimensions are invalid")
    return dims


def _valid_utc(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None \
        and timestamp.utcoffset().total_seconds() == 0


def _decoded_masks(manifest: dict, blobs: dict[str, bytes], dims_xyz: tuple[int, int, int]):
    voxel_count = math.prod(dims_xyz)
    result = []
    for item in manifest.get("masks", []):
        if not isinstance(item, dict):
            continue
        blob_id = item.get("blob")
        raw = blobs.get(blob_id) if isinstance(blob_id, str) else None
        if raw is None:
            raise ValueError("capsule mask blob is missing")
        try:
            dtype = np.dtype(item.get("dtype", "uint8"))
        except TypeError:
            raise ValueError("capsule mask dtype is invalid") from None
        if dtype.hasobject or len(raw) != voxel_count * dtype.itemsize:
            raise ValueError("capsule mask dimensions do not match the grid")
        array = np.frombuffer(raw, dtype=dtype).reshape((dims_xyz[2], dims_xyz[1], dims_xyz[0])) > 0
        digest = hashlib.sha256(raw).hexdigest()
        descriptor_hash = item.get("blob_sha256")
        review = item.get("review") if isinstance(item.get("review"), dict) else {}
        reviewer = review.get("by")
        timestamp = review.get("at_utc")
        reviewed = (
            item.get("reviewed") is True
            and isinstance(reviewer, str)
            and bool(reviewer.strip())
            and _valid_utc(timestamp)
            and isinstance(descriptor_hash, str)
            and descriptor_hash == digest
            and review.get("blob_sha256") == digest
        )
        result.append({
            "item": item,
            "array": array,
            "blob_sha256": digest,
            "reviewed": reviewed,
            "reviewed_by": reviewer.strip() if reviewed else None,
            "review_at_utc": timestamp if reviewed else None,
        })
    return result


def _is_default_mask(item: dict) -> bool:
    role = str(item.get("role", "")).strip().lower()
    mask_id = str(item.get("id", "")).strip().lower()
    if role == "render" or mask_id in _RENDER_IDS:
        return False
    if role in _DEFAULT_MASK_ROLES:
        return True
    # Older surgeon/dataset masks predate the role field; keep those selectable by default.
    return not role and item.get("source") in {"surgeon", "dataset"}


def _select_masks(decoded: list[dict], mask_ids: list[str] | None, include_unreviewed: bool) -> list[dict]:
    by_id = {entry["item"].get("id"): entry for entry in decoded}
    if mask_ids is None:
        candidates = [entry for entry in decoded if _is_default_mask(entry["item"])]
    else:
        if len(mask_ids) != len(set(mask_ids)):
            raise ValueError("--masks must contain unique mask IDs")
        missing = [mask_id for mask_id in mask_ids if mask_id not in by_id]
        if missing:
            raise ValueError("one or more requested masks are missing from the capsule")
        candidates = [by_id[mask_id] for mask_id in mask_ids]
    if not candidates:
        raise ValueError("no lesion or segmentation masks were selected")
    if not include_unreviewed:
        unsigned = [entry for entry in candidates if not entry["reviewed"]]
        if mask_ids is not None and unsigned:
            raise ValueError("a requested mask has no valid review signature; pass --include-unreviewed")
        candidates = [entry for entry in candidates if entry["reviewed"]]
    if not candidates:
        raise ValueError("no masks have a valid review signature; pass --include-unreviewed to export unsigned masks")
    for entry in candidates:
        if not entry["array"].any():
            raise ValueError("an exported mask is empty")
    return candidates


def _target_volume(manifest: dict, selected_masks: list[dict], series_number: int | None) -> tuple[dict, int]:
    volumes = manifest.get("volumes", [])
    if series_number is None:
        volume_ids = {entry["item"].get("for_volume") for entry in selected_masks}
        if None in volume_ids or len(volume_ids) != 1:
            raise ValueError("selected masks use different volumes; specify --series")
        target_id = next(iter(volume_ids))
        matches = [volume for volume in volumes if volume.get("id") == target_id]
        if len(matches) != 1:
            raise ValueError("the mask source volume is missing or ambiguous")
        try:
            target_number = int(matches[0]["series"]["number"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("the mask source volume has no DICOM series number") from None
        return matches[0], target_number
    matches = []
    for volume in volumes:
        try:
            if int(volume["series"]["number"]) == series_number:
                matches.append(volume)
        except (KeyError, TypeError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError("target series is not uniquely represented in the capsule")
    return matches[0], series_number


def _find_source_series(dicom_dir: str | Path, number: int) -> Series:
    try:
        series, _ = scan_series(dicom_dir)
    except (OSError, ValueError):
        raise ValueError("source DICOM directory could not be scanned") from None
    matches = [item for item in series if item.number == number]
    if len(matches) != 1:
        raise ValueError(f"source series number {number} is missing or ambiguous")
    return matches[0]


def _read_target_series(source: Series, work: Path, expected_spacing: list[float]) \
        -> tuple[sitk.Image, list[pydicom.Dataset], Path]:
    source_dir = work / "source-series"
    source_dir.mkdir()
    for index, path in enumerate(source.files):
        try:
            shutil.copyfile(path, source_dir / f"{index:06d}.dcm")
        except OSError:
            raise ValueError("source DICOM series could not be copied for export") from None
    reader = sitk.ImageSeriesReader()
    try:
        filenames = reader.GetGDCMSeriesFileNames(str(source_dir))
        if not filenames:
            raise ValueError("source DICOM series contains no readable images")
        reader.SetFileNames(filenames)
        image = reader.Execute()
        datasets = [pydicom.dcmread(filename) for filename in filenames]
    except Exception:
        raise ValueError("source DICOM series could not be decoded") from None
    if image.GetDimension() != 3:
        raise ValueError("source DICOM series must be three-dimensional")
    actual_spacing = np.asarray(image.GetSpacing(), dtype=float)
    expected = np.asarray(expected_spacing, dtype=float)
    if expected.shape != (3,) or not np.all(np.isfinite(expected)) \
            or not np.allclose(actual_spacing, expected, atol=0.01, rtol=0.0):
        raise ValueError("source series spacing does not match the capsule volume")
    if any(str(dataset.SeriesInstanceUID) != source.uid for dataset in datasets):
        raise ValueError("source DICOM staging contains more than one series")
    return image, datasets, source_dir


def _target_to_grid_transform(volume: dict) -> sitk.Transform:
    registration = volume.get("registration")
    if not isinstance(registration, dict):
        raise ValueError("capsule volume registration metadata is missing")
    if registration.get("reference") is True or registration.get("disabled") is True:
        return sitk.Transform(3, sitk.sitkIdentity)
    try:
        moving_to_reference_ras = np.asarray(registration["moving_to_reference_ras"], dtype=float)
        if moving_to_reference_ras.shape != (4, 4) or not np.all(np.isfinite(moving_to_reference_ras)):
            raise ValueError
        # Resampling a reference-grid mask onto a moving target grid samples each target
        # location in reference space. This is the inverse of moving the mask points
        # from the reference grid into the target series.
        target_to_grid_lps = _RAS_TO_LPS @ moving_to_reference_ras @ _RAS_TO_LPS
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(target_to_grid_lps[:3, :3].ravel().tolist())
        transform.SetTranslation(target_to_grid_lps[:3, 3].tolist())
        return transform
    except (KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError("capsule registration metadata is invalid or non-invertible") from None


def _map_masks_to_native(selected_masks: list[dict], manifest: dict, volume: dict,
                         native_image: sitk.Image) -> list[dict]:
    grid = maskops.grid_from_manifest(manifest)
    transform = _target_to_grid_transform(volume)
    mapped = []
    for entry in selected_masks:
        source = sitk.GetImageFromArray(entry["array"].astype(np.uint8))
        source.CopyInformation(grid)
        native = sitk.Resample(
            source, native_image, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        native_array = sitk.GetArrayFromImage(native).astype(bool)
        if not native_array.any():
            raise ValueError("a selected mask falls outside the target DICOM series")
        mapped.append({**entry, "native_array": native_array})
    return mapped


def _label(value: object, include_unreviewed: bool) -> str:
    label = str(value or "Segmentation")
    marker = f" - {_REVIEW_MARKER}" if include_unreviewed else ""
    return label[:max(1, 64 - len(marker))] + marker


def _rgb(value: object) -> tuple[int, int, int]:
    if isinstance(value, str) and len(value) == 7 and value.startswith("#"):
        try:
            return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            pass
    return (255, 255, 255)


def _segment_codes(item: dict) -> tuple[CodedConcept, CodedConcept]:
    # SPEC: an absent role means lesion. Anatomy (role "structure") must not reach PACS coded as a mass.
    role = str(item.get("role") or "lesion").strip().lower()
    if role == "structure":
        anatomy = CodedConcept("91723000", "SCT", "Anatomical Structure")
        return anatomy, anatomy
    return (CodedConcept("49755003", "SCT", "Morphologically Altered Structure"),
            CodedConcept("4147007", "SCT", "Mass"))


def _new_segmentation(source_images: list[pydicom.Dataset], masks_native: list[dict], series_number: int,
                      include_unreviewed: bool) -> Segmentation:
    segments = []
    pixel_array = np.stack([entry["native_array"] for entry in masks_native], axis=-1).astype(np.uint8)
    for number, entry in enumerate(masks_native, start=1):
        signed = entry["reviewed"]
        algorithm = SegmentAlgorithmTypeValues.MANUAL if signed else SegmentAlgorithmTypeValues.SEMIAUTOMATIC
        algorithm_info = None if signed else AlgorithmIdentificationSequence(
            name="Case Capsule",
            family=CodedConcept("CC01", "99CASE", "Case Capsule"),
            version="0.1.0",
        )
        color = CIELabColor.from_rgb(*_rgb(entry["item"].get("color")))
        category, property_type = _segment_codes(entry["item"])
        segments.append(SegmentDescription(
            segment_number=number,
            segment_label=_label(entry["item"].get("label"), include_unreviewed),
            segmented_property_category=category,
            segmented_property_type=property_type,
            algorithm_type=algorithm,
            algorithm_identification=algorithm_info,
            display_color=color,
        ))
    segmentation = Segmentation(
        source_images=source_images,
        pixel_array=pixel_array,
        segmentation_type="BINARY",
        segment_descriptions=segments,
        series_instance_uid=generate_uid(),
        series_number=max(1, series_number + 1000),
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="Case Capsule",
        manufacturer_model_name="Capsule Export",
        software_versions=_version("case-capsule", "0.1.0"),
        device_serial_number="1",
        content_description="Case Capsule" + (f" - {_REVIEW_MARKER}" if include_unreviewed else ""),
        omit_empty_frames=True,
    )
    segmentation.SeriesDescription = "Case Capsule" + (f" - {_REVIEW_MARKER}" if include_unreviewed else "")
    return segmentation


def _new_rtstruct(source_dir: Path, source_images: list[pydicom.Dataset], masks_native: list[dict],
                  series_number: int, include_unreviewed: bool, path: Path) -> None:
    try:
        structure = RTStructBuilder.create_new(dicom_series_path=str(source_dir))
        for entry in masks_native:
            structure.add_roi(
                mask=entry["native_array"].transpose(1, 2, 0).astype(bool),
                color=list(_rgb(entry["item"].get("color"))),
                name=_label(entry["item"].get("label"), include_unreviewed),
            )
        dataset = structure.ds
        dataset.FrameOfReferenceUID = source_images[0].FrameOfReferenceUID
        dataset.SeriesDescription = "Case Capsule" + (f" - {_REVIEW_MARKER}" if include_unreviewed else "")
        dataset.SeriesNumber = max(1, series_number + 1000)
        dataset.SeriesInstanceUID = generate_uid()
        dataset.SOPInstanceUID = generate_uid()
        if getattr(dataset, "file_meta", None) is not None:
            dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        with redirect_stdout(io.StringIO()):
            structure.save(str(path))
    except Exception:
        raise ValueError("RTSTRUCT output could not be created from the selected masks") from None


def _receipt(masks_native: list[dict], spacing: tuple[float, float, float], series_number: int,
             n_referenced_instances: int) -> dict:
    voxel_ml = math.prod(spacing) / 1000.0
    return {
        "masks": [
            {
                "id": entry["item"].get("id"),
                "label": str(entry["item"].get("label", "")),
                "reviewed_by": entry["reviewed_by"],
                "review_at_utc": entry["review_at_utc"],
                "blob_sha256": entry["blob_sha256"],
                "voxels_native": int(entry["native_array"].sum()),
                "volume_ml_native": float(entry["native_array"].sum()) * voxel_ml,
            }
            for entry in masks_native
        ],
        "target_series_number": series_number,
        "n_referenced_instances": n_referenced_instances,
        "software_versions": {
            "case_capsule": _version("case-capsule", "0.1.0"),
            "highdicom": hd.__version__,
            "rt_utils": _version("rt-utils", "unknown"),
            "pydicom": pydicom.__version__,
            "SimpleITK": sitk.Version_VersionString(),
        },
    }


def export_capsule(capsule_path: str | Path, dicom_dir: str | Path, out_dir: str | Path,
                   series_number: int | None = None, mask_ids: str | None = None,
                   include_unreviewed: bool = False) -> dict:
    """Write one SEG, one RTSTRUCT and a PHI-minimized receipt into an empty directory."""
    destination = _output_directory(out_dir)
    capsule = Path(capsule_path)
    if not capsule.is_file():
        raise ValueError("capsule input must be a file")
    try:
        manifest, blobs = parse_capsule(capsule)
    except OSError:
        raise ValueError("capsule input could not be read") from None
    dims_xyz = _manifest_dimensions(manifest)
    decoded = _decoded_masks(manifest, blobs, dims_xyz)
    parsed_mask_ids = None
    if mask_ids is not None:
        parsed_mask_ids = [value.strip() for value in mask_ids.split(",") if value.strip()]
        if not parsed_mask_ids:
            raise ValueError("--masks must name one or more mask IDs")
    selected = _select_masks(decoded, parsed_mask_ids, include_unreviewed)
    target_volume, target_number = _target_volume(manifest, selected, series_number)
    source = _find_source_series(dicom_dir, target_number)
    try:
        expected_spacing = target_volume["source_spacing_mm"]
    except (KeyError, TypeError):
        raise ValueError("capsule target volume has no source spacing") from None

    # Stage beside --out, never in the system temp dir: source slices are patient data and must
    # stay under the directory the operator chose for outputs.
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError("export outputs could not be written") from None
    with tempfile.TemporaryDirectory(prefix=".case-capsule-export-", dir=destination.parent) as temporary:
        work = Path(temporary)
        target_image, source_images, source_dir = _read_target_series(source, work, expected_spacing)
        masks_native = _map_masks_to_native(selected, manifest, target_volume, target_image)
        staged = work / "outputs"
        staged.mkdir()
        series_description_has_unreviewed = include_unreviewed
        segmentation = _new_segmentation(source_images, masks_native, target_number,
                                         series_description_has_unreviewed)
        try:
            segmentation.save_as(str(staged / "seg.dcm"))
        except Exception:
            raise ValueError("DICOM-SEG output could not be encoded") from None
        _new_rtstruct(source_dir, source_images, masks_native, target_number,
                      series_description_has_unreviewed, staged / "rtstruct.dcm")
        receipt = _receipt(masks_native, tuple(float(value) for value in target_image.GetSpacing()),
                           target_number, len(source_images))
        (staged / "export.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")

        try:
            destination.mkdir(parents=True, exist_ok=True)
            if any(destination.iterdir()):
                raise ValueError("output directory is non-empty; choose a new --out")
            for filename in ("seg.dcm", "rtstruct.dcm", "export.json"):
                with (staged / filename).open("rb") as source_file:
                    with (destination / filename).open("xb") as output_file:
                        shutil.copyfileobj(source_file, output_file)
        except FileExistsError:
            raise ValueError("output directory is non-empty; choose a new --out") from None
        except OSError:
            raise ValueError("export outputs could not be written") from None
    return receipt
