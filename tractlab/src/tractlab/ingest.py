"""Anonymized DICOM ingestion for the personal TractLab app."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np


# Patient identity surviving conversion means anonymisation failed: the case is removed.
_PATIENT_KEYS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "AccessionNumber",
    "AcquisitionDateTime",
    "SeriesInstanceUID",
    "StudyInstanceUID",
}
# Site and staff identifiers that dcm2niix -ba y keeps (GE portal exports): scrubbed after conversion.
_SITE_KEYS = {
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "DeviceSerialNumber",
}
_PHI_KEYS = _PATIENT_KEYS | _SITE_KEYS
_NOMINAL_READOUT_TIME = 0.05
_REGRID_MIN_RATIO = 1.2
_POST_CONTRAST_RE = re.compile(
    r"\+\s?C\b|\bC\+|\bGDO?\b|\bGAD|GADOLIN|\bPOST[ -]?CONTRAST|\bP[OÓ]S[ -]?CONTRAST"
)
_T1_TERMS = ("MPRAGE", "MP2RAGE UNI", "SPGR", "BRAVO", "TFE", "T1")
_B0_MAX = 50.0
_DEFAULT_CASES_ROOT = Path.home() / "Library/Application Support/TractLab/cases"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_cases_root(cases_root: Path) -> str | None:
    resolved = cases_root.expanduser().resolve()
    repo = _repo_root().resolve()
    try:
        resolved.relative_to(repo)
    except ValueError:
        pass
    else:
        return "Cases root must be outside the TractLab repository."
    if any(part in {"Dropbox", "Desktop", "Downloads"} for part in resolved.parts):
        return "Cases root must not be inside Dropbox, Desktop, or Downloads."
    return None


def _is_derived(metadata: dict[str, Any]) -> bool:
    image_type = metadata.get("ImageType", [])
    if isinstance(image_type, str):
        values = image_type.replace("\\", " ").replace(",", " ").split()
    else:
        values = image_type or []
    tokens = {
        token.strip().upper()
        for value in values
        for token in str(value).replace("\\", " ").replace(",", " ").split()
    }
    return bool(tokens.intersection({"DERIVED", "ADC", "TRACE", "FA"}))


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _bvec_columns(series: dict[str, Any], nvolumes: int) -> list[list[float]]:
    raw = series.get("bvecs")
    if raw is None:
        return []
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return []
    if values.ndim != 2:
        return []
    if values.shape == (3, nvolumes):
        values = values.T
    elif values.shape != (nvolumes, 3):
        return []
    return values.tolist()


def _unique_diffusion_directions(series: dict[str, Any]) -> int:
    bvals = _as_float_list(series.get("bvals"))
    if not bvals:
        return 0
    vectors = _bvec_columns(series, len(bvals))
    if len(vectors) != len(bvals):
        return 0
    unique: set[tuple[float, float, float]] = set()
    for bvalue, vector in zip(bvals, vectors):
        if bvalue <= _B0_MAX:
            continue
        unit = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(unit))
        if norm < 1e-8:
            continue
        unit /= norm
        first_nonzero = next((value for value in unit if abs(value) > 1e-6), 1.0)
        if first_nonzero < 0:
            unit *= -1
        unique.add(tuple(float(round(value, 4)) for value in unit))
    return len(unique)


def _volume_count(series: dict[str, Any]) -> int:
    try:
        return int(series.get("nvolumes") or len(series.get("bvals") or []) or 1)
    except (TypeError, ValueError):
        return 1


def _image_dimensions(metadata: dict[str, Any]) -> tuple[int, ...] | None:
    value = metadata.get("ImageSize") or metadata.get("MatrixSize")
    if value is None:
        return None
    try:
        result = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None
    return result if result else None


def _voxel_sizes(metadata: dict[str, Any]) -> tuple[float, ...] | None:
    direct = _as_float_list(metadata.get("VoxelSize") or metadata.get("VoxelSizes"))
    if direct:
        return tuple(direct[:3]) if len(direct) >= 3 else None
    pixel_spacing = _as_float_list(metadata.get("PixelSpacing"))
    slice_spacing = metadata.get("SpacingBetweenSlices", metadata.get("SliceThickness"))
    try:
        if pixel_spacing and len(pixel_spacing) >= 2 and slice_spacing is not None:
            return (pixel_spacing[1], pixel_spacing[0], float(slice_spacing))
    except (TypeError, ValueError):
        return None
    return None


def _same_geometry(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a = first.get("metadata", {})
    b = second.get("metadata", {})
    first_dims, second_dims = _image_dimensions(a), _image_dimensions(b)
    first_vox, second_vox = _voxel_sizes(a), _voxel_sizes(b)
    if first_dims is None or first_dims != second_dims:
        return False
    if first_vox is None or second_vox is None or len(first_vox) != len(second_vox):
        return False
    return all(
        math.isclose(x, y, rel_tol=0.0, abs_tol=1e-4)
        for x, y in zip(first_vox, second_vox)
    )


def _phase_direction(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("PhaseEncodingDirection")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in {"i", "i-", "j", "j-", "k", "k-"} else None


def _opposite_phase(value: str) -> str:
    return value[:-1] if value.endswith("-") else f"{value}-"


def _is_epi(metadata: dict[str, Any]) -> bool:
    scanning = str(metadata.get("ScanningSequence", "")).upper()
    tokens = {token.strip() for token in scanning.replace("/", "\\").split("\\")}
    text = " ".join(
        str(metadata.get(key, "")) for key in ("SequenceName", "SeriesDescription")
    ).upper()
    return "EP" in tokens or any(term in text for term in ("EPI", "SEFMAP"))


def _is_b0_series(series: dict[str, Any]) -> bool:
    bvals = _as_float_list(series.get("bvals"))
    return bvals is None or not bvals or all(abs(value) <= _B0_MAX for value in bvals)


def _has_b0(series: dict[str, Any]) -> bool:
    """True for a pure b=0 series or a diffusion series that contains b=0 volumes."""
    bvals = _as_float_list(series.get("bvals"))
    return bvals is None or not bvals or any(abs(value) <= _B0_MAX for value in bvals)


def _b0_indices(series: dict[str, Any]) -> list[int]:
    bvals = _as_float_list(series.get("bvals")) or []
    return [index for index, value in enumerate(bvals) if abs(value) <= _B0_MAX]


def _is_orig(series: dict[str, Any]) -> bool:
    """GE exports the unfiltered reconstruction as a twin series prefixed "ORIG:"."""
    return str(series.get("metadata", {}).get("SeriesDescription", "")).strip().upper().startswith("ORIG:")


def _readout_time(metadata: dict[str, Any]) -> tuple[float, str, str | None]:
    """Return (seconds, source, warning) for the DWI total readout time."""
    value = metadata.get("TotalReadoutTime")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value), "sidecar", None
    spacing = metadata.get("EffectiveEchoSpacing")
    recon = metadata.get("ReconMatrixPE")
    try:
        derived = float(spacing) * (int(recon) - 1)
    except (TypeError, ValueError):
        derived = math.nan
    if math.isfinite(derived) and derived > 0:
        return derived, "derived", None
    return (
        _NOMINAL_READOUT_TIME,
        "nominal",
        "DWI sidecar has no readout time (typical of GE portal exports); a nominal "
        f"{_NOMINAL_READOUT_TIME} s is used. Topup/eddy displacement fields are invariant to a "
        "readout time shared by both polarities, so the value only rescales the estimated field.",
    )


def _regrid_voxel(metadata: dict[str, Any]) -> list[float] | None:
    """Acquired voxel size when the scanner zero-filled k-space (ReconMatrixPE >> AcquisitionMatrixPE)."""
    try:
        ratio = float(metadata["ReconMatrixPE"]) / float(metadata["AcquisitionMatrixPE"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    voxels = _voxel_sizes(metadata)
    if voxels is None or len(voxels) != 3 or not math.isfinite(ratio) or ratio < _REGRID_MIN_RATIO:
        return None
    axis = {"i": 0, "j": 1}.get((_phase_direction(metadata) or "j")[0])
    if axis is None:
        return None
    inplane = round(voxels[axis] * ratio, 3)
    return [inplane, inplane, round(float(voxels[2]), 3)]


def _t1_like(metadata: dict[str, Any]) -> bool:
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("ScanningSequence", "SequenceName", "SeriesDescription")
    ).upper()
    return any(term in text for term in _T1_TERMS)


def _post_contrast(metadata: dict[str, Any]) -> bool:
    for key in ("ContrastBolusAgent", "ContrastBolusVolume", "PostContrast"):
        value = metadata.get(key)
        if value not in (None, "", 0, 0.0, False):
            return True
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("SeriesDescription", "ProtocolName", "SequenceName")
    ).upper()
    return _POST_CONTRAST_RE.search(text) is not None


def classify(series: list[dict]) -> dict:
    """Classify parsed sidecars and diffusion tables without touching the filesystem."""
    warnings: list[str] = []
    refusals: list[str] = []
    dwi_candidates = []
    for item in series:
        metadata = item.get("metadata", {})
        bvals = _as_float_list(item.get("bvals"))
        if (
            not _is_derived(metadata)
            and bvals
            and any(value > _B0_MAX for value in bvals)
            and _unique_diffusion_directions(item) >= 6
        ):
            dwi_candidates.append(item)
    dwi = max(dwi_candidates, key=lambda item: (_volume_count(item), _is_orig(item))) if dwi_candidates else None
    if dwi is not None and _is_orig(dwi) and any(
        item is not dwi and _volume_count(item) == _volume_count(dwi) for item in dwi_candidates
    ):
        warnings.append("Selected the unfiltered ORIG: DWI series over its filtered twin.")
    if dwi is None:
        refusals.append("No usable DWI series with at least six diffusion directions was found.")

    t1_candidates = []
    for item in series:
        metadata = item.get("metadata", {})
        voxel_sizes = _voxel_sizes(metadata)
        if (
            not _is_derived(metadata)
            and str(metadata.get("MRAcquisitionType", "")).upper() == "3D"
            and voxel_sizes is not None
            and len(voxel_sizes) == 3
            and all(value <= 1.3 for value in voxel_sizes)
            and _t1_like(metadata)
        ):
            t1_candidates.append(item)
    t1 = None
    if t1_candidates:
        t1 = max(
            t1_candidates,
            key=lambda item: (
                not _post_contrast(item.get("metadata", {})),
                -max(_voxel_sizes(item.get("metadata", {})) or (math.inf,)),
                _volume_count(item),
            ),
        )
    else:
        refusals.append("No suitable non-derived 3D T1 series at 1.3 mm or finer was found.")

    rpe = None
    dwi_phase = _phase_direction(dwi.get("metadata", {})) if dwi else None
    rpe_candidates = [
        item
        for item in series
        if item is not dwi
        and not _is_derived(item.get("metadata", {}))
        and _is_epi(item.get("metadata", {}))
        and _has_b0(item)
        and dwi is not None
        and _same_geometry(dwi, item)
    ]
    # Prefer a pure b=0 reverse-PE series, then the unfiltered ORIG: reconstruction.
    rpe_candidates.sort(key=lambda item: (not _is_b0_series(item), not _is_orig(item)))
    unknown_rpe_polarity = any(
        _phase_direction(item.get("metadata", {})) is None for item in rpe_candidates
    )
    if dwi is not None and dwi_phase is not None:
        opposite = _opposite_phase(dwi_phase)
        rpe = next(
            (
                item
                for item in rpe_candidates
                if _phase_direction(item.get("metadata", {})) == opposite
            ),
            None,
        )
    if rpe is None:
        if dwi is not None and dwi_phase is None:
            warnings.append("Phase-encoding polarity is unknown; reverse-PE pairing was skipped.")
        elif unknown_rpe_polarity:
            warnings.append("Reverse-PE polarity is unknown; reverse-PE pairing was skipped.")
        else:
            warnings.append("No matching reverse-PE EPI b=0 series was found.")
    elif dwi_phase is None or _phase_direction(rpe.get("metadata", {})) is None:
        rpe = None
        warnings.append("Phase-encoding polarity is unknown; reverse-PE pairing was skipped.")
    elif not _is_b0_series(rpe):
        warnings.append(
            "Reverse-PE series is a diffusion acquisition "
            f"({rpe.get('metadata', {}).get('SeriesDescription', 'unnamed')}); "
            f"only its {len(_b0_indices(rpe))} b=0 volume(s) are used."
        )

    post_contrast = bool(t1 and _post_contrast(t1.get("metadata", {})))
    if post_contrast:
        warnings.append("Selected T1 series appears post-contrast.")
    return {
        "dwi": dwi,
        "rpe": rpe,
        "rpe_mode": "pair" if rpe else "none",
        "t1": t1,
        "t1_post_contrast": post_contrast,
        "warnings": warnings,
        "refusals": refusals,
    }


def run_dcm2niix(dicom_dir: Path, output_dir: Path, binary: str) -> None:
    """Convert real DICOM input while keeping command output and source path private."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                binary,
                "-b",
                "y",
                "-ba",
                "y",
                "-z",
                "y",
                "-f",
                "%s_%p",
                "-o",
                str(output_dir),
                str(dicom_dir),
            ],
            capture_output=True,
            check=False,
        )  # bytes: series names can carry non-UTF-8 text (e.g. Latin-1 "mm\xb2")
    except OSError as exc:
        raise RuntimeError("dcm2niix could not be started.") from exc
    if completed.returncode != 0:
        raise RuntimeError("dcm2niix conversion failed.")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _drop_keys(value: Any, keys: set[str]) -> int:
    removed = 0
    if isinstance(value, dict):
        for key in [key for key in value if key in keys]:
            del value[key]
            removed += 1
        for nested in value.values():
            removed += _drop_keys(nested, keys)
    elif isinstance(value, list):
        for nested in value:
            removed += _drop_keys(nested, keys)
    return removed


def scrub_site_keys(output_dir: Path) -> int:
    """Remove site/staff identifiers from converted sidecars; return the number of fields removed."""
    removed = 0
    for path in sorted(output_dir.glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Converted sidecar validation failed.") from exc
        count = _drop_keys(metadata, _SITE_KEYS)
        if count:
            path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            removed += count
    return removed


def extract_rpe_b0(case_root: Path, rpe: dict[str, Any]) -> dict[str, Any]:
    """Write only the b=0 volumes of a reverse-PE diffusion series; return the updated series record."""
    if _is_b0_series(rpe):
        return rpe
    indices = _b0_indices(rpe)
    source = case_root / rpe["nifti"]
    sidecar = case_root / rpe["json"]
    stem = sidecar.name[: -len(".json")]
    out_nifti = sidecar.with_name(f"{stem}_rpe_b0.nii.gz")
    out_json = sidecar.with_name(f"{stem}_rpe_b0.json")
    image = nib.load(str(source))
    data = np.asanyarray(image.dataobj)[..., indices]
    header = image.header.copy()
    nib.save(type(image)(data, image.affine, header), str(out_nifti))
    shutil.copyfile(sidecar, out_json)
    relative = lambda path: path.relative_to(case_root).as_posix()
    return {
        **rpe,
        "nifti": relative(out_nifti),
        "json": relative(out_json),
        "bval": None,
        "bvec": None,
        "bvals": [0.0] * len(indices),
        "bvecs": None,
        "nvolumes": len(indices),
    }


def phi_check(output_dir: Path) -> list[str]:
    """Return identifying sidecar keys; callers must remove the case if nonempty."""
    found: set[str] = set()
    for path in sorted(output_dir.glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Converted sidecar validation failed.") from exc
        if isinstance(metadata, dict):
            found.update(_PHI_KEYS.intersection(_walk_keys(metadata)))
    return sorted(found)


def _read_converted_series(case_root: Path, output_dir: Path) -> list[dict]:
    result = []
    for sidecar_path in sorted(output_dir.glob("*.json")):
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Converted sidecar parsing failed.") from exc
        if not isinstance(metadata, dict):
            continue
        nifti_path = sidecar_path.with_suffix(".nii.gz")
        if not nifti_path.exists():
            nifti_path = sidecar_path.with_suffix(".nii")
        if not nifti_path.exists():
            continue
        try:
            image = nib.load(str(nifti_path))
            metadata = dict(metadata)
            metadata["ImageSize"] = list(image.shape[:3])
            metadata["VoxelSize"] = [float(value) for value in image.header.get_zooms()[:3]]
        except Exception as exc:
            raise RuntimeError("Converted NIfTI validation failed.") from exc

        bval_path = sidecar_path.with_suffix(".bval")
        bvec_path = sidecar_path.with_suffix(".bvec")
        bvals = None
        bvecs = None
        if bval_path.exists():
            try:
                bvals = np.asarray(np.loadtxt(bval_path), dtype=float).reshape(-1).tolist()
            except (OSError, ValueError) as exc:
                raise RuntimeError("Converted b-value validation failed.") from exc
        if bvec_path.exists():
            try:
                values = np.asarray(np.loadtxt(bvec_path), dtype=float)
                if values.ndim == 1:
                    values = values.reshape(3, -1)
                bvecs = values.T.tolist() if values.shape[0] == 3 else values.tolist()
            except (OSError, ValueError) as exc:
                raise RuntimeError("Converted b-vector validation failed.") from exc

        relative = lambda path: path.relative_to(case_root).as_posix()
        result.append(
            {
                "metadata": metadata,
                "nifti": relative(nifti_path),
                "json": relative(sidecar_path),
                "bval": relative(bval_path) if bval_path.exists() else None,
                "bvec": relative(bvec_path) if bvec_path.exists() else None,
                "bvals": bvals,
                "bvecs": bvecs,
                "nvolumes": int(image.shape[3]) if len(image.shape) >= 4 else 1,
            }
        )
    return result


def _blank_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _blank_shape(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return []
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    if isinstance(value, str):
        return ""
    return None


def _unsigned_qc_block(template: dict[str, Any]) -> dict[str, Any]:
    block = _blank_shape(template)
    for key in ("approved_by", "date", "sheet_sha", "sheet_path"):
        if key in block:
            block[key] = None
    return block


def write_manifest_skeleton(case_root: Path, case_id: str) -> None:
    """Write the minimal viewer manifest from the packaged disclaimer and unsigned QC shapes."""
    source = Path(__file__).with_name("templates") / "manifest_skeleton.json"
    try:
        demo = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("The manifest skeleton template could not be read.") from exc

    manifest = {
        "case_id": case_id,
        "case_root": ".",
        "disclaimer": demo["disclaimer"],
        "status": "ingested",
        "inputs": {
            "b0": {"path": "nifti/b0.nii.gz"},
            "fod": {"path": "nifti/wmfod_norm.mif"},
            "mask": {"path": "nifti/mask_up.nii.gz"},
            "t1": {"path": "nifti/t1_brain_dwi.nii.gz"},
        },
    }
    for key in ("t1_qc", "atlas_prior_qc", "parcellation_qc"):
        manifest[key] = _unsigned_qc_block(demo[key])
    (case_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_case(
    case_root: Path,
    *,
    case_id: str,
    label: str,
    n_files: int,
    classification: dict[str, Any],
) -> None:
    dwi = classification["dwi"]
    t1 = classification["t1"]
    rpe = classification["rpe"]
    warnings = list(classification["warnings"])
    readout, readout_source, readout_warning = _readout_time(dwi["metadata"])
    if readout_warning:
        warnings.append(readout_warning)
    regrid = _regrid_voxel(dwi["metadata"])
    if regrid:
        warnings.append(
            "DWI was reconstructed on a zero-filled grid (ReconMatrixPE "
            f"{dwi['metadata'].get('ReconMatrixPE')} vs acquired {dwi['metadata'].get('AcquisitionMatrixPE')}); "
            f"the pipeline regrids to the acquired {regrid[0]:g} x {regrid[1]:g} x {regrid[2]:g} mm before denoising."
        )
    case = {
        "schema": "tractlab.case/1",
        "case_id": case_id,
        "label": label,
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {"n_files": n_files},
        "inputs": {
            "dwi": {
                "nifti": dwi["nifti"],
                "bval": dwi["bval"],
                "bvec": dwi["bvec"],
                "json": dwi["json"],
                "pe_dir": dwi["metadata"].get("PhaseEncodingDirection"),
                "total_readout_time": readout,
                "total_readout_time_source": readout_source,
                "regrid_voxel_mm": regrid,
            },
            "rpe": (
                {"mode": "pair", "nifti": rpe["nifti"], "json": rpe["json"]}
                if rpe
                else {"mode": "none"}
            ),
            "t1": {
                "nifti": t1["nifti"],
                "json": t1["json"],
                "post_contrast": classification["t1_post_contrast"],
            },
        },
        "warnings": warnings,
    }
    classification["warnings"] = warnings
    (case_root / "case.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def run_ingest(
    dicom_dir: str | Path,
    *,
    cases_root: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    root = Path(cases_root).expanduser() if cases_root is not None else _DEFAULT_CASES_ROOT
    refusal = _validate_cases_root(root)
    if refusal:
        return {
            "exit_code": 2,
            "ok": False,
            "case_id": None,
            "case_root": None,
            "refusals": [refusal],
            "warnings": [],
        }
    source = Path(dicom_dir).expanduser()
    if not source.is_dir():
        return {
            "exit_code": 2,
            "ok": False,
            "case_id": None,
            "case_root": None,
            "refusals": ["The selected DICOM folder is unavailable."],
            "warnings": [],
        }

    try:
        n_files = _count_files(source)
        if n_files == 0:
            return {
                "exit_code": 2,
                "ok": False,
                "case_id": None,
                "case_root": None,
                "refusals": ["The selected DICOM folder contains no files."],
                "warnings": [],
            }
    except OSError:
        return {
            "exit_code": 2,
            "ok": False,
            "case_id": None,
            "case_root": None,
            "refusals": ["The selected DICOM folder could not be read."],
            "warnings": [],
        }

    case_id = f"case-{secrets.token_hex(4)}"
    case_root = root.resolve() / case_id
    output_dir = case_root / "raw/dcm2niix"
    try:
        root.mkdir(parents=True, exist_ok=True)
        case_root.mkdir()
    except OSError:
        return {
            "exit_code": 1,
            "ok": False,
            "case_id": None,
            "case_root": None,
            "refusals": ["The cases folder could not be created."],
            "warnings": [],
        }

    binary = os.environ.get("DCM2NIIX") or str(Path.home() / "fsl/bin/dcm2niix")
    try:
        run_dcm2niix(source, output_dir, binary)
        scrubbed = scrub_site_keys(output_dir)
        leaked_keys = phi_check(output_dir)
        if leaked_keys:
            shutil.rmtree(case_root, ignore_errors=True)
            return {
                "exit_code": 3,
                "ok": False,
                "case_id": case_id,
                "case_root": None,
                "refusals": ["Identifying DICOM fields remained in a converted sidecar; case removed."],
                "warnings": [],
            }
        series = _read_converted_series(case_root, output_dir)
        classification = classify(series)
        if classification["refusals"]:
            shutil.rmtree(case_root, ignore_errors=True)
            return {
                "exit_code": 2,
                "ok": False,
                "case_id": None,
                "case_root": None,
                "refusals": classification["refusals"],
                "warnings": classification["warnings"],
            }
        if scrubbed:
            classification["warnings"].append(
                f"Removed {scrubbed} site/staff identifier field(s) from converted sidecars."
            )
        if classification["rpe"] is not None:
            classification["rpe"] = extract_rpe_b0(case_root, classification["rpe"])
        write_case(
            case_root,
            case_id=case_id,
            label=label or "Imported case",
            n_files=n_files,
            classification=classification,
        )
        write_manifest_skeleton(case_root, case_id)
    except Exception:
        shutil.rmtree(case_root, ignore_errors=True)
        return {
            "exit_code": 1,
            "ok": False,
            "case_id": None,
            "case_root": None,
            "refusals": ["DICOM ingestion failed validation."],
            "warnings": [],
        }

    return {
        "exit_code": 0,
        "ok": True,
        "case_id": case_id,
        "case_root": str(case_root),
        "refusals": [],
        "warnings": classification["warnings"],
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in ("ok", "case_id", "case_root", "refusals", "warnings")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tractlab.ingest")
    parser.add_argument("dicom_dir")
    parser.add_argument("--cases-root", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_ingest(args.dicom_dir, cases_root=args.cases_root, label=args.label)
    if args.json:
        print(json.dumps(_public_result(result), ensure_ascii=False))
    elif result["ok"]:
        print(f"Created {result['case_id']}.")
        for warning in result["warnings"]:
            print(f"Warning: {warning}")
    else:
        for refusal in result["refusals"]:
            print(refusal, file=sys.stderr)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
